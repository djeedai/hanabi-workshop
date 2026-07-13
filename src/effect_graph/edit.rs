//! Pure mutations of an [`EffectGraph`].
//!
//! Every operation here takes `&mut EffectGraph` (plus, where needed, the type
//! registry) and returns whatever the caller must keep to invert the change.
//! They are the building blocks the edit channel ([`crate::edits`]) drives: the
//! channel mutates the graph through these, re-bakes the result to the preview
//! [`EffectAsset`], and records the returned inverse on
//! the undo stack. Nothing here touches the ECS, assets, or rendering, so each
//! op is unit-testable in isolation.
//!
//! [`EffectAsset`]: bevy_hanabi::EffectAsset

use std::any::TypeId;

use bevy::{
    math::{UVec2, Vec2, Vec3, Vec4},
    reflect::{PartialReflect, ReflectRef, TypeRegistry},
};
use bevy_hanabi::{
    Attribute, CpuValue, Expr, ExprHandle, Gradient, Module, ReflectModifier, SimulationCondition,
    SimulationSpace, SpawnerSettings, Value,
};

use super::{
    model::{
        EditValue, EffectGraph, ExprNode, GradientVec3, GradientVec4, GraphLink, GraphNode,
        ImageBinding, InputSlot, MAX_SELECT_IMAGE_INPUTS, ModifierNodeData, NodeId, NodePayload,
        PortRef, PropertyDef, PropertyId, SharedStr, SlotId, TextureSlotDef, is_select_image_input,
    },
    schema::{ConfigKind, FieldRole, modifier_schema},
};
use crate::{document::ModifierGroup, proxy};

// ---------------------------------------------------------------------------
// Header settings.
// ---------------------------------------------------------------------------

pub fn set_effect_name(graph: &mut EffectGraph, new: SharedStr) -> SharedStr {
    std::mem::replace(&mut graph.header.name, new)
}

pub fn set_simulation_space(graph: &mut EffectGraph, new: SimulationSpace) -> SimulationSpace {
    std::mem::replace(&mut graph.header.simulation_space, new)
}

pub fn set_simulation_condition(
    graph: &mut EffectGraph,
    new: SimulationCondition,
) -> SimulationCondition {
    std::mem::replace(&mut graph.header.simulation_condition, new)
}

pub fn set_spawner(graph: &mut EffectGraph, new: SpawnerSettings) -> SpawnerSettings {
    std::mem::replace(&mut graph.header.spawner, new)
}

pub fn set_capacity(graph: &mut EffectGraph, new: u32) -> u32 {
    std::mem::replace(&mut graph.header.capacity, new)
}

pub fn set_z_layer_2d(graph: &mut EffectGraph, new: f32) -> f32 {
    std::mem::replace(&mut graph.header.z_layer_2d, new)
}

// ---------------------------------------------------------------------------
// Properties (addressed by stable PropertyId).
// ---------------------------------------------------------------------------

/// Add a new property, returning its freshly-allocated id.
///
/// Edit-only properties may share a display name; exposed-name uniqueness is
/// enforced at bake time.
pub fn add_property(
    graph: &mut EffectGraph,
    name: SharedStr,
    default: Value,
    exposed: bool,
) -> PropertyId {
    let id = graph.alloc_property_id();
    graph.properties.push(PropertyDef {
        id,
        name,
        default,
        exposed,
    });
    id
}

/// A property removed from the graph, captured so the removal can be undone.
///
/// Holds the removed definition, every `Property` reference node that was
/// deleted (each with its incident links and stack membership), and, for every
/// consumer input port the property fed, the inline default the property's
/// value replaced — enough to restore the exact prior state.
#[derive(Debug, Clone)]
pub struct RemovedProperty {
    pub def: PropertyDef,
    pub nodes: Vec<RemovedNode>,
    pub inlined: Vec<InlinedPort>,
}

/// One consumer input port whose inline default was overwritten when a property
/// it referenced was removed.
#[derive(Debug, Clone)]
pub struct InlinedPort {
    pub port: PortRef,
    /// The literal the port carried before, or `None` if it had no inline slot
    /// (one was created to hold the inlined value and must be dropped to undo).
    pub previous: Option<Value>,
}

/// Re-insert a previously-removed property and undo the inlining of its value.
///
/// Restores each consumer port's prior inline default and re-inserts the
/// deleted `Property` reference nodes with their incident links. The inverse of
/// [`remove_property`].
pub fn restore_property(graph: &mut EffectGraph, removed: RemovedProperty) {
    let RemovedProperty {
        def,
        nodes,
        inlined,
    } = removed;
    graph.properties.push(def);
    for InlinedPort { port, previous } in inlined {
        match previous {
            Some(v) => {
                set_input_default(graph, port.node, &port.port, v);
            }
            None => remove_input_slot(graph, port.node, &port.port),
        }
    }
    for node in nodes {
        insert_node(graph, node);
    }
}

/// Remove the property `id`.
///
/// Every `ExprNode::Property(id)` reference node is deleted, and the property's
/// default value is inlined into each input port the node fed (so the consumer
/// keeps its value as a plain literal). Returns the captured state for the
/// inverse, or `None` if no such property exists.
pub fn remove_property(graph: &mut EffectGraph, id: PropertyId) -> Option<RemovedProperty> {
    let pos = graph.properties.iter().position(|p| p.id == id)?;
    let def = graph.properties.remove(pos);

    let ref_nodes: Vec<NodeId> = graph
        .nodes
        .iter()
        .filter(|n| matches!(&n.payload, NodePayload::Expr(ExprNode::Property(pid)) if *pid == id))
        .map(|n| n.id)
        .collect();

    let mut nodes = Vec::new();
    let mut inlined = Vec::new();
    for node_id in ref_nodes {
        let removed = remove_node(graph, node_id)?;
        // Inline the default into every port this reference node fed.
        for link in &removed.links {
            if link.from.node == node_id {
                let previous = set_input_default(graph, link.to.node, &link.to.port, def.default);
                inlined.push(InlinedPort {
                    port: link.to.clone(),
                    previous,
                });
            }
        }
        nodes.push(removed);
    }
    Some(RemovedProperty {
        def,
        nodes,
        inlined,
    })
}

/// Remove an inline input slot by port name, if present.
fn remove_input_slot(graph: &mut EffectGraph, node: NodeId, port: &str) {
    if let Some(node) = graph.node_mut(node) {
        node.inputs.retain(|s| &*s.name != port);
    }
}

/// Rename property `id`, returning its previous name, or `None` if absent.
pub fn rename_property(
    graph: &mut EffectGraph,
    id: PropertyId,
    new: SharedStr,
) -> Option<SharedStr> {
    let prop = graph.properties.iter_mut().find(|p| p.id == id)?;
    Some(std::mem::replace(&mut prop.name, new))
}

/// Replace property `id`'s default value.
///
/// Returns the previous value, or `None`.
pub fn set_property_default(graph: &mut EffectGraph, id: PropertyId, new: Value) -> Option<Value> {
    let prop = graph.properties.iter_mut().find(|p| p.id == id)?;
    Some(std::mem::replace(&mut prop.default, new))
}

/// Toggle property `id`'s exposed flag.
///
/// Returns the previous value, or `None`.
pub fn set_property_exposed(
    graph: &mut EffectGraph,
    id: PropertyId,
    exposed: bool,
) -> Option<bool> {
    let prop = graph.properties.iter_mut().find(|p| p.id == id)?;
    Some(std::mem::replace(&mut prop.exposed, exposed))
}

// ---------------------------------------------------------------------------
// Modifier stacks.
// ---------------------------------------------------------------------------

/// Reorder the member at `from` to `to` within `group`'s stack.
///
/// `to` is the target index *after* the source is removed (matching the move
/// semantics of the edit channel). Returns `true` on success.
pub fn move_stack_member(
    graph: &mut EffectGraph,
    group: ModifierGroup,
    from: usize,
    to: usize,
) -> bool {
    let Some(stack) = graph.stacks.iter_mut().find(|s| s.group == group) else {
        return false;
    };
    if from >= stack.members.len() || to >= stack.members.len() {
        return false;
    }
    // Rotate only the affected sub-slice by one, so elements outside
    // `[from, to]` are never touched (unlike remove + insert, which shifts
    // every following element twice).
    match from.cmp(&to) {
        std::cmp::Ordering::Less => stack.members[from..=to].rotate_left(1),
        std::cmp::Ordering::Greater => stack.members[to..=from].rotate_right(1),
        std::cmp::Ordering::Equal => {}
    }
    true
}

/// A modifier removed from a stack, captured so the removal can be undone.
///
/// The node itself, the links that targeted it, and the index it occupied.
#[derive(Debug, Clone)]
pub struct RemovedModifier {
    pub group: ModifierGroup,
    pub at: usize,
    pub node: GraphNode,
    pub links: Vec<GraphLink>,
}

/// Remove the modifier at `idx` in `group`.
///
/// Drops it from the stack, removes the node, and removes every link that fed
/// it. Orphaned operand expression nodes are left in place (harmless; they bake
/// to nothing if unreferenced). Returns the captured state for the inverse, or
/// `None` if the index is out of range.
pub fn remove_modifier(
    graph: &mut EffectGraph,
    group: ModifierGroup,
    idx: usize,
) -> Option<RemovedModifier> {
    let node_id = {
        let stack = graph.stacks.iter_mut().find(|s| s.group == group)?;
        if idx >= stack.members.len() {
            return None;
        }
        stack.members.remove(idx)
    };
    let node_pos = graph.nodes.iter().position(|n| n.id == node_id)?;
    let node = graph.nodes.remove(node_pos);
    let mut links = Vec::new();
    graph.links.retain(|l| {
        if l.to.node == node_id || l.from.node == node_id {
            links.push(l.clone());
            false
        } else {
            true
        }
    });
    Some(RemovedModifier {
        group,
        at: idx,
        node,
        links,
    })
}

/// Re-insert a removed modifier node and its links at `at` in `group`.
///
/// The inverse of [`remove_modifier`]. Returns `false` if `group`'s stack is
/// missing.
pub fn insert_modifier(graph: &mut EffectGraph, removed: RemovedModifier) -> bool {
    let RemovedModifier {
        group,
        at,
        node,
        links,
    } = removed;
    let Some(stack) = graph.stacks.iter().position(|s| s.group == group) else {
        return false;
    };
    let node_id = node.id;
    graph.nodes.push(node);
    graph.links.extend(links);
    let members = &mut graph.stacks[stack].members;
    let at = at.min(members.len());
    members.insert(at, node_id);
    true
}

/// Build a default modifier node and insert it at `at` in `group`'s stack.
///
/// The node's configuration and required expression-input defaults are read
/// from the registry factory's freshly-built instance, so the node bakes back
/// to that same modifier. Returns the new node id, or `None` if the type is not
/// a registered modifier.
pub fn add_modifier_from_template(
    graph: &mut EffectGraph,
    registry: &TypeRegistry,
    group: ModifierGroup,
    type_id: TypeId,
    at: usize,
) -> Option<NodeId> {
    let (payload, inputs) = default_modifier_payload(registry, type_id)?;
    let id = graph.alloc_node_id();
    graph.nodes.push(GraphNode {
        id,
        payload,
        inputs,
    });
    let stack = graph.stacks.iter_mut().find(|s| s.group == group)?;
    let at = at.min(stack.members.len());
    stack.members.insert(at, id);
    Some(id)
}

/// Build the payload and inline-default input slots of a modifier node.
///
/// Projects the registry factory's instance for `type_id` through the modifier
/// schema (a narrow, factory-only "raise").
fn default_modifier_payload(
    registry: &TypeRegistry,
    type_id: TypeId,
) -> Option<(NodePayload, Vec<InputSlot>)> {
    let registration = registry.get(type_id)?;
    let reflect_modifier = registration.data::<ReflectModifier>()?;
    let schema = modifier_schema(registration.type_info())?;
    let type_path: SharedStr = SharedStr::from(registration.type_info().type_path());

    // The factory allocates sensible default literals into a scratch module; we
    // read those back as the ports' inline defaults and discard the module.
    let mut scratch = Module::default();
    let boxed = (reflect_modifier.factory)(&mut scratch);
    let modifier = boxed.as_reflect();

    let ReflectRef::Struct(s) = modifier.reflect_ref() else {
        return None;
    };

    let mut inputs = Vec::new();
    for field in schema.ports() {
        // A texture port carries an image binding, not a value literal; seed it
        // unbound so the consumer is usable without a separate Image node.
        if matches!(field.role, FieldRole::Texture) {
            inputs.push(InputSlot {
                name: field.name.clone(),
                default: ImageBinding::Unbound.into(),
            });
            continue;
        }
        let optional = matches!(field.role, FieldRole::ExprPort { optional: true });
        let Some(value) = s.field(&field.name) else {
            continue;
        };
        let handle = if optional {
            value
                .try_downcast_ref::<Option<ExprHandle>>()
                .and_then(|o| *o)
        } else {
            value.try_downcast_ref::<ExprHandle>().copied()
        };
        // Non-optional ports must carry a default so the graph bakes; optional
        // ports left unset get no slot (the factory default applies on bake).
        if let Some(handle) = handle
            && let Some(Expr::Literal(lit)) = scratch.get(handle)
            && let Some(v) = proxy::literal_value(lit)
        {
            inputs.push(InputSlot {
                name: field.name.clone(),
                default: v.into(),
            });
        }
    }

    let mut config = std::collections::BTreeMap::new();
    for field in schema.config() {
        let FieldRole::Config(kind) = field.role else {
            continue;
        };
        let Some(value) = s.field(&field.name) else {
            continue;
        };
        if let Some(edit) = read_config_value(value, kind) {
            config.insert(field.name.clone(), edit);
        }
    }

    Some((
        NodePayload::Modifier(ModifierNodeData::Known { type_path, config }),
        inputs,
    ))
}

/// Read a modifier configuration field's current value into an [`EditValue`].
///
/// Driven by its schema-classified [`ConfigKind`]. Best-effort: a field that
/// can't be read returns `None` and is simply omitted (baking then falls back
/// to the factory default).
fn read_config_value(field: &dyn PartialReflect, kind: ConfigKind) -> Option<EditValue> {
    match kind {
        ConfigKind::Bool => field
            .try_downcast_ref::<bool>()
            .map(|v| EditValue::Bool(*v)),
        ConfigKind::U32 => field.try_downcast_ref::<u32>().map(|v| EditValue::U32(*v)),
        ConfigKind::UVec2 => field
            .try_downcast_ref::<UVec2>()
            .map(|v| EditValue::UVec2(*v)),
        ConfigKind::Attribute => field
            .try_downcast_ref::<Attribute>()
            .map(|v| EditValue::Attribute(*v)),
        ConfigKind::CpuVec3 => field
            .try_downcast_ref::<CpuValue<Vec3>>()
            .map(|v| EditValue::CpuVec3(*v)),
        ConfigKind::CpuVec4 => field
            .try_downcast_ref::<CpuValue<Vec4>>()
            .map(|v| EditValue::CpuVec4(*v)),
        ConfigKind::Gradient3 => field
            .try_downcast_ref::<Gradient<Vec3>>()
            .map(|v| EditValue::Gradient3(GradientVec3::Analytical(v.clone()))),
        ConfigKind::Gradient4 => field
            .try_downcast_ref::<Gradient<Vec4>>()
            .map(|v| EditValue::Gradient4(GradientVec4::Analytical(v.clone()))),
        ConfigKind::Scalar => read_scalar_value(field).map(EditValue::Scalar),
        ConfigKind::Enum => match field.reflect_ref() {
            ReflectRef::Enum(e) => Some(EditValue::Enum {
                type_path: SharedStr::from(field.reflect_type_path()),
                variant: SharedStr::from(e.variant_name()),
            }),
            _ => None,
        },
        ConfigKind::Flags => read_flags_bits(field).map(|bits| EditValue::Flags {
            type_path: SharedStr::from(field.reflect_type_path()),
            bits,
        }),
        ConfigKind::Raw => None,
    }
}

/// Read a scalar/vector reflected field into a `bevy_hanabi` [`Value`].
fn read_scalar_value(field: &dyn PartialReflect) -> Option<Value> {
    if let Some(v) = field.try_downcast_ref::<f32>() {
        Some(Value::from(*v))
    } else if let Some(v) = field.try_downcast_ref::<Vec2>() {
        Some(Value::from(*v))
    } else if let Some(v) = field.try_downcast_ref::<Vec3>() {
        Some(Value::from(*v))
    } else if let Some(v) = field.try_downcast_ref::<Vec4>() {
        Some(Value::from(*v))
    } else if let Some(v) = field.try_downcast_ref::<i32>() {
        Some(Value::from(*v))
    } else {
        field.try_downcast_ref::<u32>().map(|v| Value::from(*v))
    }
}

/// Read the inner integer of a bitflags newtype (tuple struct over one
/// integer).
fn read_flags_bits(field: &dyn PartialReflect) -> Option<u64> {
    let ReflectRef::TupleStruct(ts) = field.reflect_ref() else {
        return None;
    };
    let inner = ts.field(0)?;
    if let Some(v) = inner.try_downcast_ref::<u8>() {
        Some(*v as u64)
    } else if let Some(v) = inner.try_downcast_ref::<u16>() {
        Some(*v as u64)
    } else if let Some(v) = inner.try_downcast_ref::<u32>() {
        Some(*v as u64)
    } else {
        inner.try_downcast_ref::<u64>().copied()
    }
}

// ---------------------------------------------------------------------------
// Expression input defaults.
// ---------------------------------------------------------------------------

/// Set the inline default of `node`'s input `port` to `new`.
///
/// Returns the previous value. If the port had no slot yet (was relying on a
/// bake-time default), one is created and `None` is returned. `None` is also
/// returned if `node` does not exist.
pub fn set_input_default(
    graph: &mut EffectGraph,
    node: NodeId,
    port: &str,
    new: Value,
) -> Option<Value> {
    let node = graph.node_mut(node)?;
    if let Some(slot) = node.inputs.iter_mut().find(|s| &*s.name == port) {
        std::mem::replace(&mut slot.default, new.into()).as_value()
    } else {
        node.inputs.push(InputSlot {
            name: SharedStr::from(port),
            default: new.into(),
        });
        None
    }
}

/// Set the inline image binding of an input port.
///
/// Returns the previous binding (unbound if the port had none), or `None` if
/// `node` does not exist. A missing slot is created.
pub fn set_input_image_binding(
    graph: &mut EffectGraph,
    node: NodeId,
    port: &str,
    new: ImageBinding,
) -> Option<ImageBinding> {
    let node = graph.node_mut(node)?;
    if let Some(slot) = node.inputs.iter_mut().find(|s| &*s.name == port) {
        let prev = slot.default.as_image().cloned().unwrap_or_default();
        slot.default = new.into();
        Some(prev)
    } else {
        node.inputs.push(InputSlot {
            name: SharedStr::from(port),
            default: new.into(),
        });
        Some(ImageBinding::Unbound)
    }
}

/// Set a standalone `ExprNode::Literal` node's value.
///
/// Returns the previous value, or `None` if `node` is not a literal expression
/// node.
pub fn set_literal_node(graph: &mut EffectGraph, node: NodeId, new: Value) -> Option<Value> {
    let node = graph.node_mut(node)?;
    if let NodePayload::Expr(ExprNode::Literal(v)) = &mut node.payload {
        Some(std::mem::replace(v, new))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// SetAttributeModifier attribute retargeting.
// ---------------------------------------------------------------------------

/// Retarget the `attribute` of a `SetAttributeModifier` node at `idx` in
/// `group`.
///
/// When the new attribute's value type differs from the node's inline `value`
/// literal, the literal is reset (to `reset_value` on the undo path, or the new
/// attribute's default otherwise) so the baked modifier stays type-correct.
/// Returns `(old_attribute, rewritten_old_literal)` for the inverse, or an
/// error message if the node is not a retargetable `SetAttributeModifier`.
pub fn set_modifier_attribute(
    graph: &mut EffectGraph,
    group: ModifierGroup,
    idx: usize,
    new: Attribute,
    reset_value: Option<Value>,
) -> Result<(Attribute, Option<Value>), String> {
    let node_id = graph
        .stack(group)
        .and_then(|s| s.members.get(idx).copied())
        .ok_or_else(|| format!("no modifier at {group:?}#{idx}"))?;
    let node = graph
        .node_mut(node_id)
        .ok_or("modifier node missing from graph")?;

    // Read the current attribute (scoped borrow of the payload).
    let old_attr = match &node.payload {
        NodePayload::Modifier(ModifierNodeData::Known { config, .. }) => {
            match config.get("attribute") {
                Some(EditValue::Attribute(a)) => *a,
                _ => return Err("modifier has no `attribute` field".to_string()),
            }
        }
        _ => return Err("selected node is not an editable modifier".to_string()),
    };

    // Decide whether the inline `value` literal must be rewritten.
    let rewrote_old = {
        let slot = node.inputs.iter_mut().find(|s| &*s.name == "value");
        match (reset_value, slot) {
            // Undo path: force the literal back to the captured value.
            (Some(v), Some(slot)) => std::mem::replace(&mut slot.default, v.into()).as_value(),
            (Some(_), None) => None,
            // Forward path: reset only when the value type changes.
            (None, Some(slot))
                if slot.default.as_value().map(|v| v.value_type()) != Some(new.value_type()) =>
            {
                std::mem::replace(&mut slot.default, new.default_value().into()).as_value()
            }
            (None, _) => None,
        }
    };

    // Write the new attribute (re-borrow now that the slot borrow has ended).
    if let NodePayload::Modifier(ModifierNodeData::Known { config, .. }) = &mut node.payload {
        config.insert("attribute".into(), EditValue::Attribute(new));
    }
    Ok((old_attr, rewrote_old))
}

/// Set a modifier node's non-expression config `field` to `new`.
///
/// Returns the previous [`EditValue`] for the inverse, or `None` if `node` is
/// not a known modifier carrying that field.
pub fn set_modifier_config(
    graph: &mut EffectGraph,
    node: NodeId,
    field: &str,
    new: EditValue,
) -> Option<EditValue> {
    let node = graph.node_mut(node)?;
    let NodePayload::Modifier(ModifierNodeData::Known { config, .. }) = &mut node.payload else {
        return None;
    };
    let old = config.get(field)?.clone();
    config.insert(SharedStr::from(field), new);
    Some(old)
}

// ---------------------------------------------------------------------------
// Standalone nodes (expression nodes on the canvas).
// ---------------------------------------------------------------------------

/// Add a standalone expression node, returning its freshly-allocated id.
///
/// Carries the given operand input defaults. The node is free (not a stack
/// member); modifier nodes are added through [`add_modifier_from_template`]
/// instead.
pub fn add_expr_node(graph: &mut EffectGraph, expr: ExprNode, inputs: Vec<InputSlot>) -> NodeId {
    let id = graph.alloc_node_id();
    graph.nodes.push(GraphNode {
        id,
        payload: NodePayload::Expr(expr),
        inputs,
    });
    id
}

/// A node removed from the graph, captured so the removal can be undone.
///
/// The node itself, the links incident to it (as source or target), and its
/// stack membership if it happened to be a stack member.
#[derive(Debug, Clone)]
pub struct RemovedNode {
    pub node: GraphNode,
    pub links: Vec<GraphLink>,
    /// `Some((group, index))` if the node was a member of a modifier stack;
    /// `None` for a free node (the common case for expression nodes).
    pub member_of: Option<(ModifierGroup, usize)>,
}

/// Remove the node `id`.
///
/// Drops it from any stack it belonged to, removes the node, and removes every
/// link incident to it (as source or target). Returns the captured state for
/// the inverse, or `None` if no such node exists.
pub fn remove_node(graph: &mut EffectGraph, id: NodeId) -> Option<RemovedNode> {
    let node_pos = graph.nodes.iter().position(|n| n.id == id)?;
    let member_of = graph.stacks.iter().find_map(|s| {
        s.members
            .iter()
            .position(|m| *m == id)
            .map(|idx| (s.group, idx))
    });
    if let Some((group, idx)) = member_of
        && let Some(stack) = graph.stacks.iter_mut().find(|s| s.group == group)
    {
        stack.members.remove(idx);
    }
    let node = graph.nodes.remove(node_pos);
    let mut links = Vec::new();
    graph.links.retain(|l| {
        if l.to.node == id || l.from.node == id {
            links.push(l.clone());
            false
        } else {
            true
        }
    });
    Some(RemovedNode {
        node,
        links,
        member_of,
    })
}

/// Re-insert a removed node with its incident links and stack membership.
///
/// The inverse of [`remove_node`].
pub fn insert_node(graph: &mut EffectGraph, removed: RemovedNode) {
    let RemovedNode {
        node,
        links,
        member_of,
    } = removed;
    let node_id = node.id;
    graph.nodes.push(node);
    graph.links.extend(links);
    if let Some((group, idx)) = member_of
        && let Some(stack) = graph.stacks.iter_mut().find(|s| s.group == group)
    {
        let idx = idx.min(stack.members.len());
        stack.members.insert(idx, node_id);
    }
}

// ---------------------------------------------------------------------------
// Links.
// ---------------------------------------------------------------------------

/// Connect an output port to an input port.
///
/// Returns any link that was displaced because the target input already had one
/// (an input takes at most one link). The inverse of an add that displaced
/// `old` is `add_link(old)` (which displaces the new link and restores `old`);
/// an add that displaced nothing inverts via [`remove_link_to`] on `link.to`.
///
/// Validity (no cycles, type compatibility, forward-only stage flow) is
/// enforced by the graph view before the edit is emitted, so this op only
/// maintains the at-most-one-link-per-input invariant.
pub fn add_link(graph: &mut EffectGraph, link: GraphLink) -> Option<GraphLink> {
    let displaced = remove_link_to(graph, &link.to);
    graph.links.push(link);
    displaced
}

/// Remove the single link targeting input port `to`.
///
/// Returns it (for the inverse), or `None` if no link targeted it.
pub fn remove_link_to(graph: &mut EffectGraph, to: &PortRef) -> Option<GraphLink> {
    let pos = graph.links.iter().position(|l| &l.to == to)?;
    Some(graph.links.remove(pos))
}

// ---------------------------------------------------------------------------
// Image source nodes and texture slots.
// ---------------------------------------------------------------------------

/// Add an image source node with its initial binding.
///
/// The node id is allocated so a caller predicting the next id (for layout
/// seeding) stays correct. Returns the new node id.
pub fn add_image_node(graph: &mut EffectGraph, binding: ImageBinding) -> NodeId {
    let node_id = graph.alloc_node_id();
    graph.nodes.push(GraphNode {
        id: node_id,
        payload: NodePayload::Expr(ExprNode::Image(binding)),
        inputs: Vec::new(),
    });
    node_id
}

/// Set the binding of image node `node`.
///
/// Returns the previous binding for the inverse, or `None` if `node` is not an
/// image node.
pub fn set_image_node_binding(
    graph: &mut EffectGraph,
    node: NodeId,
    binding: ImageBinding,
) -> Option<ImageBinding> {
    let NodePayload::Expr(ExprNode::Image(cur)) = &mut graph.node_mut(node)?.payload else {
        return None;
    };
    Some(std::mem::replace(cur, binding))
}

/// Re-derive a [`ExprNode::SelectImage`] node's image-input count from its
/// links.
///
/// The count is a pure function of the links targeting `node`: it becomes the
/// highest linked image index plus two, so exactly one empty trailing image
/// port is offered, or one when nothing is linked. Clamped to the maximum the
/// schema enumerates. Because it is derived, undo needs no separate record —
/// restoring the links and re-running this reproduces the count.
///
/// [`ExprNode::SelectImage`]: crate::effect_graph::model::ExprNode::SelectImage
pub fn normalize_select_image(graph: &mut EffectGraph, node: NodeId) {
    if !matches!(
        graph.node(node).map(|n| &n.payload),
        Some(NodePayload::Expr(ExprNode::SelectImage { .. }))
    ) {
        return;
    }
    let highest = graph
        .links
        .iter()
        .filter(|l| l.to.node == node && is_select_image_input(&l.to.port))
        .filter_map(|l| select_image_input_index(&l.to.port))
        .max();
    let count = match highest {
        Some(i) => (i + 2).min(MAX_SELECT_IMAGE_INPUTS as u32),
        None => 1,
    };
    if let Some(NodePayload::Expr(ExprNode::SelectImage { count: c })) =
        graph.node_mut(node).map(|n| &mut n.payload)
    {
        *c = count;
    }
}

/// Parse a `SelectImage` image-input port name (`image{N}`) to its index.
fn select_image_input_index(port: &str) -> Option<u32> {
    port.strip_prefix("image").and_then(|n| n.parse().ok())
}

/// Add a texture slot.
///
/// Auto-named and appended at the end (the highest sampling index). Returns the
/// new slot id.
pub fn add_texture_slot(graph: &mut EffectGraph) -> SlotId {
    let slot_id = graph.alloc_slot_id();
    let name = SharedStr::from(format!("texture {}", graph.texture_slots.len() + 1));
    graph
        .texture_slots
        .push(TextureSlotDef { id: slot_id, name });
    slot_id
}

/// A texture slot removed from the list, captured for the inverse.
#[derive(Debug, Clone)]
pub struct RemovedTextureSlot {
    pub slot: TextureSlotDef,
    pub at: usize,
}

/// Remove the texture slot `id`.
///
/// Returns the captured slot and its index for the inverse, or `None` if no
/// such slot exists. Image bindings referencing the slot are left dangling (the
/// Material panel only offers removal of unreferenced slots).
pub fn remove_texture_slot(graph: &mut EffectGraph, id: SlotId) -> Option<RemovedTextureSlot> {
    let at = graph.texture_slots.iter().position(|s| s.id == id)?;
    let slot = graph.texture_slots.remove(at);
    Some(RemovedTextureSlot { slot, at })
}

/// Re-insert a removed texture slot at its original index.
///
/// The inverse of [`remove_texture_slot`].
pub fn insert_texture_slot(graph: &mut EffectGraph, removed: RemovedTextureSlot) {
    let at = removed.at.min(graph.texture_slots.len());
    graph.texture_slots.insert(at, removed.slot);
}

/// Rename the texture slot `id`.
///
/// Returns the previous name, or `None` if no such slot exists.
pub fn rename_texture_slot(
    graph: &mut EffectGraph,
    id: SlotId,
    new: SharedStr,
) -> Option<SharedStr> {
    let slot = graph.texture_slots.iter_mut().find(|s| s.id == id)?;
    Some(std::mem::replace(&mut slot.name, new))
}

/// Move the texture slot `id` to index `to`, shifting the others.
///
/// Slot order is the sampling index, so this reassigns indices. Returns the
/// slot's previous index for the inverse, or `None` if no such slot exists.
pub fn reorder_texture_slot(graph: &mut EffectGraph, id: SlotId, to: usize) -> Option<usize> {
    let from = graph.texture_slots.iter().position(|s| s.id == id)?;
    let to = to.min(graph.texture_slots.len().saturating_sub(1));
    if from != to {
        let slot = graph.texture_slots.remove(from);
        graph.texture_slots.insert(to, slot);
    }
    Some(from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effect_graph::{demo::demo_graph, schema::OUTPUT_PORT};

    #[test]
    fn select_image_count_tracks_links() {
        let mut g = demo_graph();
        let sel = add_expr_node(
            &mut g,
            ExprNode::SelectImage { count: 1 },
            vec![InputSlot {
                name: SharedStr::from("index"),
                default: Value::from(0u32).into(),
            }],
        );
        let img_a = add_image_node(&mut g, ImageBinding::Unbound);
        let img_b = add_image_node(&mut g, ImageBinding::Unbound);

        let count = |g: &EffectGraph| match &g.node(sel).unwrap().payload {
            NodePayload::Expr(ExprNode::SelectImage { count }) => *count,
            _ => panic!("not a SelectImage"),
        };
        let link_to = |port: &str, src: NodeId| GraphLink {
            from: PortRef {
                node: src,
                port: OUTPUT_PORT.into(),
            },
            to: PortRef {
                node: sel,
                port: SharedStr::from(port),
            },
        };

        assert_eq!(count(&g), 1, "starts with one empty image port");

        add_link(&mut g, link_to("image0", img_a));
        normalize_select_image(&mut g, sel);
        assert_eq!(count(&g), 2, "connecting the trailing port grows the node");

        add_link(&mut g, link_to("image1", img_b));
        normalize_select_image(&mut g, sel);
        assert_eq!(count(&g), 3);

        remove_link_to(&mut g, &link_to("image1", img_b).to);
        normalize_select_image(&mut g, sel);
        assert_eq!(count(&g), 2, "clearing the highest port shrinks the node");

        remove_link_to(&mut g, &link_to("image0", img_a).to);
        normalize_select_image(&mut g, sel);
        assert_eq!(count(&g), 1, "back to a single empty port");
    }

    #[test]
    fn image_node_keeps_initial_binding() {
        let mut graph = demo_graph();
        let binding = ImageBinding::Asset("textures/patterns/smoke.png".into());
        let node = add_image_node(&mut graph, binding.clone());

        assert_eq!(
            graph.node(node).map(|node| &node.payload),
            Some(&NodePayload::Expr(ExprNode::Image(binding)))
        );
    }

    #[test]
    fn add_and_remove_expr_node() {
        let mut g = demo_graph();
        let before = g.nodes.len();
        let id = add_expr_node(&mut g, ExprNode::Literal(Value::from(2.0f32)), Vec::new());
        assert_eq!(g.nodes.len(), before + 1);
        assert!(matches!(
            g.node(id).unwrap().payload,
            NodePayload::Expr(ExprNode::Literal(_))
        ));

        let removed = remove_node(&mut g, id).expect("removed");
        assert_eq!(g.nodes.len(), before);
        assert!(g.node(id).is_none(), "node gone");
        assert!(removed.member_of.is_none(), "free node has no membership");

        insert_node(&mut g, removed);
        assert_eq!(g.nodes.len(), before + 1);
        assert!(g.node(id).is_some(), "node restored");
    }

    #[test]
    fn remove_node_drops_incident_links() {
        let mut g = demo_graph();
        // The demo wires several Property/operator expr nodes into modifiers.
        // Pick the source node of the first link and remove it; the link to its
        // target must be captured and restored by the inverse.
        let link = g.links.first().cloned().expect("demo has links");
        let source = link.from.node;
        let before_links = g.links.len();

        let removed = remove_node(&mut g, source).expect("removed");
        assert!(removed.links.contains(&link), "incident link captured");
        assert!(
            !g.links
                .iter()
                .any(|l| l.from.node == source || l.to.node == source),
            "incident links dropped"
        );

        insert_node(&mut g, removed);
        assert_eq!(g.links.len(), before_links, "links restored");
        assert!(g.links.contains(&link), "exact link restored");
    }

    #[test]
    fn remove_node_restores_stack_membership() {
        let mut g = demo_graph();
        let group = ModifierGroup::Init;
        let before = g.stack(group).unwrap().members.clone();
        let member = before[1];

        let removed = remove_node(&mut g, member).expect("removed");
        assert_eq!(removed.member_of, Some((group, 1)));
        assert_eq!(g.stack(group).unwrap().members.len(), before.len() - 1);

        insert_node(&mut g, removed);
        assert_eq!(
            g.stack(group).unwrap().members,
            before,
            "membership restored"
        );
    }

    #[test]
    fn header_setters_round_trip() {
        let mut g = demo_graph();
        let old = set_z_layer_2d(&mut g, 3.0);
        assert_eq!(g.header.z_layer_2d, 3.0);
        set_z_layer_2d(&mut g, old);
        assert_eq!(g.header.z_layer_2d, 0.0);
    }

    #[test]
    fn add_and_remove_property() {
        let mut g = demo_graph();
        let before = g.properties.len();
        let id = add_property(&mut g, "extra".into(), Value::from(1.0f32), false);
        assert_eq!(g.properties.len(), before + 1);
        let removed = remove_property(&mut g, id).expect("removed");
        assert_eq!(g.properties.len(), before);
        assert_eq!(&*removed.def.name, "extra");
        assert!(removed.nodes.is_empty(), "fresh property has no references");
        assert!(removed.inlined.is_empty());
    }

    #[test]
    fn remove_property_inlines_references() {
        let mut g = demo_graph();
        // The demo exposes `gravity`, referenced by a Property node linked into
        // a consumer's input port.
        let gravity = g
            .properties
            .iter()
            .find(|p| &*p.name == "gravity")
            .expect("gravity property")
            .id;
        let default = g
            .properties
            .iter()
            .find(|p| p.id == gravity)
            .unwrap()
            .default;

        let removed = remove_property(&mut g, gravity).expect("removed");
        assert!(!removed.nodes.is_empty(), "gravity is referenced");
        assert!(!removed.inlined.is_empty(), "value inlined into a consumer");

        // The reference nodes are gone, not demoted to literals.
        for rn in &removed.nodes {
            assert!(g.node(rn.node.id).is_none());
        }
        // Each consumer port now carries the property's default inline.
        for InlinedPort { port, .. } in &removed.inlined {
            let slot = g
                .node(port.node)
                .unwrap()
                .inputs
                .iter()
                .find(|s| *s.name == *port.port)
                .expect("inlined slot");
            assert_eq!(slot.default.as_value(), Some(default));
        }

        // Restoring re-inserts the property, its nodes, and their links.
        restore_property(&mut g, removed);
        assert!(g.properties.iter().any(|p| p.id == gravity));
        assert!(g.nodes.iter().any(|n| matches!(
            n.payload,
            NodePayload::Expr(ExprNode::Property(p)) if p == gravity
        )));
    }

    #[test]
    fn move_stack_member_reorders() {
        let mut g = demo_graph();
        let group = ModifierGroup::Render;
        let before = g.stack(group).unwrap().members.clone();
        assert!(before.len() >= 2);
        assert!(move_stack_member(&mut g, group, 0, 1));
        let after = g.stack(group).unwrap().members.clone();
        assert_eq!(after[0], before[1]);
        assert_eq!(after[1], before[0]);
    }

    #[test]
    fn remove_and_reinsert_modifier() {
        let mut g = demo_graph();
        let group = ModifierGroup::Init;
        let before = g.stack(group).unwrap().members.clone();
        let removed = remove_modifier(&mut g, group, 0).expect("removed");
        assert_eq!(g.stack(group).unwrap().members.len(), before.len() - 1);
        assert!(g.node(before[0]).is_none(), "node gone");
        insert_modifier(&mut g, removed);
        assert_eq!(g.stack(group).unwrap().members, before);
        assert!(g.node(before[0]).is_some(), "node restored");
    }

    #[test]
    fn set_input_default_updates_slot() {
        let mut g = demo_graph();
        // Find a modifier node that has at least one input slot.
        let node_id = g
            .nodes
            .iter()
            .find(|n| matches!(n.payload, NodePayload::Modifier(_)) && !n.inputs.is_empty())
            .map(|n| n.id)
            .expect("a modifier with an input slot");
        let port = g.node(node_id).unwrap().inputs[0].name.clone();
        let old = set_input_default(&mut g, node_id, &port, Value::from(42.0f32));
        assert!(old.is_some());
        let slot = g
            .node(node_id)
            .unwrap()
            .inputs
            .iter()
            .find(|s| s.name == port)
            .unwrap();
        assert_eq!(slot.default.as_value(), Some(Value::from(42.0f32)));
    }

    #[test]
    fn add_link_displaces_and_removes() {
        let mut g = demo_graph();
        // The demo links `spawn_speed` into SetVelocitySphere.speed. Grab that
        // existing link's target input port.
        let existing = g.links.first().cloned().expect("demo has links");
        let to = existing.to.clone();
        let before = g.links.len();

        // Add a new link into the same input port from a different source node.
        let other_source = g
            .nodes
            .iter()
            .map(|n| n.id)
            .find(|&id| id != existing.from.node && id != to.node)
            .expect("a third node");
        let new_link = GraphLink {
            from: PortRef {
                node: other_source,
                port: OUTPUT_PORT.into(),
            },
            to: to.clone(),
        };
        let displaced = add_link(&mut g, new_link.clone()).expect("displaced existing link");
        assert_eq!(displaced, existing, "returns the link it replaced");
        assert_eq!(
            g.links.len(),
            before,
            "an input still holds exactly one link"
        );
        assert!(g.links.contains(&new_link), "new link present");
        assert!(!g.links.contains(&existing), "old link gone");

        // Removing it returns it and clears the port.
        let removed = remove_link_to(&mut g, &to).expect("removed");
        assert_eq!(removed, new_link);
        assert_eq!(g.links.len(), before - 1);
        assert!(remove_link_to(&mut g, &to).is_none(), "port now empty");
    }

    #[test]
    fn add_link_to_empty_input_displaces_nothing() {
        let mut g = demo_graph();
        // Find a modifier input port with no incoming link.
        let (node_id, port) = g
            .nodes
            .iter()
            .filter(|n| matches!(n.payload, NodePayload::Modifier(_)))
            .flat_map(|n| n.inputs.iter().map(move |s| (n.id, s.name.clone())))
            .find(|(node, port)| {
                !g.links
                    .iter()
                    .any(|l| l.to.node == *node && l.to.port == *port)
            })
            .expect("an unlinked modifier input");
        let source = g
            .nodes
            .iter()
            .map(|n| n.id)
            .find(|&id| id != node_id)
            .unwrap();
        let link = GraphLink {
            from: PortRef {
                node: source,
                port: OUTPUT_PORT.into(),
            },
            to: PortRef {
                node: node_id,
                port,
            },
        };
        assert!(
            add_link(&mut g, link.clone()).is_none(),
            "nothing displaced on an empty input"
        );
        assert!(g.links.contains(&link));
    }

    /// A modifier added from a template bakes back into that same modifier.
    ///
    /// Config + required input defaults stay intact.
    #[test]
    fn add_modifier_from_template_bakes() {
        use bevy::prelude::*;

        use crate::modifier_registry::ModifierRegistryPlugin;

        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()));
        app.add_plugins(ModifierRegistryPlugin);
        let registry = app.world().resource::<AppTypeRegistry>().read();

        // AccelModifier is an Update modifier the demo also uses.
        let accel_type_id = std::any::TypeId::of::<bevy_hanabi::AccelModifier>();
        let mut g = demo_graph();
        let before = g.stack(ModifierGroup::Update).unwrap().members.len();
        let id = add_modifier_from_template(
            &mut g,
            &registry,
            ModifierGroup::Update,
            accel_type_id,
            before,
        )
        .expect("template added");
        assert_eq!(
            g.stack(ModifierGroup::Update).unwrap().members.len(),
            before + 1
        );
        assert!(matches!(
            g.node(id).unwrap().payload,
            NodePayload::Modifier(ModifierNodeData::Known { .. })
        ));
        super::super::bake::bake(&g, &registry).expect("graph with added modifier bakes");
    }
}
