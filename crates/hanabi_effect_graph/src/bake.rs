//! Baking an [`EffectGraph`] into a runtime [`bevy_hanabi::EffectAsset`].
//!
//! The edit model expresses wiring with explicit [`GraphLink`]s and per-input
//! inline defaults; the runtime model wires expressions through `ExprHandle`
//! arena indices inside a [`Module`]. Baking rebuilds that arena: it walks the
//! expression nodes in dependency order, materializing each as a `Module`
//! expression and recording the resulting handle, then resolves every operand
//! to either a linked source node's handle or its inline-default literal.
//!
//! Properties bake per [`PropertyDef::exposed`]: an exposed property becomes a
//! real `Module` property (settable per instance at runtime), while an
//! edit-only property is inlined to a literal constant at each reference, so it
//! has no runtime cost.
//!
//! This module covers expression and property baking. Modifier instantiation
//! and final [`EffectAsset`] assembly build on the `NodeId → ExprHandle` map it
//! produces.
//!
//! [`GraphLink`]: crate::model::GraphLink

use std::collections::HashMap;

use bevy::math::{UVec2, Vec2, Vec3, Vec4};
use bevy::reflect::{
    DynamicEnum, DynamicVariant, PartialReflect, Reflect, ReflectMut, TypeRegistry,
};
use bevy_hanabi::graph::expr::PropertyHandle;
use bevy_hanabi::{BoxedModifier, EffectAsset, ExprHandle, ModifierContext, Module, Value};
use bevy_hanabi::ReflectModifier;

use super::model::{
    EditValue, EffectGraph, ExprNode, GradientVec3, GradientVec4, ModifierNodeData, NodeId,
    NodePayload, PortRef, PropertyDef, PropertyId, SharedStr,
};
use super::schema::{FieldRole, expr_input_ports, modifier_schema};
use crate::ModifierGroup;

/// What a [`BakeError`] is attributed to, so the UI can surface it in context
/// (e.g. highlight the offending node or property, or show a graph-level banner).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BakeSubject {
    /// A specific graph node (e.g. an expression with a missing operand).
    Node(NodeId),
    /// A specific user property (e.g. an exposed-name conflict).
    Property(PropertyId),
    /// The graph as a whole, with no single element to blame.
    Graph,
}

/// A problem encountered while baking, attributed to the element to blame.
/// Baking collects every error it can rather than stopping at the first.
#[derive(Debug, Clone, PartialEq)]
pub struct BakeError {
    pub subject: BakeSubject,
    pub message: String,
}

impl BakeError {
    fn node(node: NodeId, message: impl Into<String>) -> Self {
        Self {
            subject: BakeSubject::Node(node),
            message: message.into(),
        }
    }

    fn property(id: PropertyId, message: impl Into<String>) -> Self {
        Self {
            subject: BakeSubject::Property(id),
            message: message.into(),
        }
    }

    #[allow(dead_code)]
    fn graph(message: impl Into<String>) -> Self {
        Self {
            subject: BakeSubject::Graph,
            message: message.into(),
        }
    }
}

/// Resolved property bindings produced by [`bake_properties`]: the runtime
/// handle of each exposed property, plus every property's definition indexed by
/// stable id (used to resolve [`ExprNode::Property`] references).
struct PropertyBindings<'a> {
    handles: HashMap<PropertyId, PropertyHandle>,
    defs: HashMap<PropertyId, &'a PropertyDef>,
}

/// Register exposed properties into `module` and index every property by its
/// stable id for later reference resolution.
///
/// Properties are referenced by id, not name, so display names are free to
/// collide. The one name constraint is on **exposed** properties: each becomes a
/// runtime `Module` property keyed by name, so a name shared by two exposed
/// properties is an inconsistency that blocks baking. It is reported as a
/// [`BakeError`] (never a panic — `Module::add_property` would panic on a
/// duplicate name, so the second add is skipped) so the author can fix it.
fn bake_properties<'a>(
    graph: &'a EffectGraph,
    module: &mut Module,
    errors: &mut Vec<BakeError>,
) -> PropertyBindings<'a> {
    let mut handles = HashMap::new();
    let mut defs = HashMap::with_capacity(graph.properties.len());
    let mut exposed_names: HashMap<&str, PropertyId> = HashMap::new();
    for prop in &graph.properties {
        if defs.insert(prop.id, prop).is_some() {
            // Two properties sharing an id is a structural inconsistency (ids are
            // unique by construction); references to it would be ambiguous.
            errors.push(BakeError::property(
                prop.id,
                format!("duplicate property id {}", prop.id.get()),
            ));
            continue;
        }
        if prop.exposed {
            let name: &str = &prop.name;
            if exposed_names.contains_key(name) {
                errors.push(BakeError::property(
                    prop.id,
                    format!("two exposed properties share the name '{name}'; rename one to bake"),
                ));
                continue;
            }
            exposed_names.insert(name, prop.id);
            let handle = module.add_property(name, prop.default);
            handles.insert(prop.id, handle);
        }
    }
    PropertyBindings { handles, defs }
}

/// Expression-node baking context: the graph, the property bindings, the
/// `Module` under construction, and the running `NodeId → ExprHandle` cache.
/// The graph origin of a baked `Expr::Literal`, used to map a value tweak to the
/// promotable module expression it produced. Lets the live-tweak fast path
/// upload a new value through the proxy property bound to that expression
/// instead of re-baking the whole graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LiteralSite {
    /// A literal expression node, identified by its node id.
    Node(NodeId),
    /// An inline default on an input port (modifier or operator), identified by
    /// the owning node and the port name.
    Input { node: NodeId, port: SharedStr },
}

/// Provenance from a bake: every baked literal mapped to the graph site that
/// produced it. Keyed into the baked module's expression arena.
pub type LiteralSites = HashMap<LiteralSite, ExprHandle>;

struct ExprBaker<'a, 'm> {
    graph: &'a EffectGraph,
    props: &'a PropertyBindings<'a>,
    module: &'m mut Module,
    handles: HashMap<NodeId, ExprHandle>,
    /// Maps each baked literal back to the graph site it came from, so a value
    /// tweak can target the corresponding module expression without re-baking.
    literal_sites: HashMap<LiteralSite, ExprHandle>,
    /// Nodes on the current DFS stack, for cycle detection.
    visiting: Vec<NodeId>,
}

impl ExprBaker<'_, '_> {
    /// Resolve a node to its `ExprHandle`, baking it (and its operands) on
    /// first visit and caching the result. Returns `None` once an error has
    /// been recorded for this subtree.
    fn resolve(&mut self, node_id: NodeId, errors: &mut Vec<BakeError>) -> Option<ExprHandle> {
        if let Some(h) = self.handles.get(&node_id) {
            return Some(*h);
        }
        if self.visiting.contains(&node_id) {
            errors.push(BakeError::node(node_id, "expression cycle"));
            return None;
        }

        let node = self.graph.node(node_id).or_else(|| {
            errors.push(BakeError::node(
                node_id,
                format!("link references missing node {}", node_id.get()),
            ));
            None
        })?;
        let NodePayload::Expr(expr) = &node.payload else {
            errors.push(BakeError::node(
                node_id,
                "expected an expression node as a link source",
            ));
            return None;
        };

        self.visiting.push(node_id);
        let handle = self.bake_expr(node_id, expr, errors);
        self.visiting.pop();

        if let Some(h) = handle {
            self.handles.insert(node_id, h);
        }
        handle
    }

    /// Bake one expression node, resolving its operand ports first.
    fn bake_expr(
        &mut self,
        node_id: NodeId,
        expr: &ExprNode,
        errors: &mut Vec<BakeError>,
    ) -> Option<ExprHandle> {
        let handle = match expr {
            ExprNode::Literal(v) => {
                let h = self.module.lit(*v);
                self.literal_sites.insert(LiteralSite::Node(node_id), h);
                h
            }
            ExprNode::Property(id) => self.bake_property_ref(node_id, *id, errors)?,
            ExprNode::Attribute(a) => self.module.attr(*a),
            ExprNode::ParentAttribute(a) => self.module.parent_attr(*a),
            ExprNode::BuiltIn(op) => self.module.builtin(*op),
            ExprNode::Unary(op) => {
                let inner = self.operand(node_id, "in", errors)?;
                self.module.unary(*op, inner)
            }
            ExprNode::Binary(op) => {
                let lhs = self.operand(node_id, "lhs", errors)?;
                let rhs = self.operand(node_id, "rhs", errors)?;
                self.module.binary(*op, lhs, rhs)
            }
            ExprNode::Ternary(op) => {
                let a = self.operand(node_id, "a", errors)?;
                let b = self.operand(node_id, "b", errors)?;
                let c = self.operand(node_id, "c", errors)?;
                self.module.ternary(*op, a, b, c)
            }
            ExprNode::Cast(ty) => {
                let inner = self.operand(node_id, "in", errors)?;
                self.module.cast(inner, *ty)
            }
        };
        Some(handle)
    }

    /// Bake a property reference (by stable id): the property's runtime handle
    /// if exposed, otherwise its default value inlined as a literal. A reference
    /// to a missing or duplicate-named exposed property is reported, not fatal.
    fn bake_property_ref(
        &mut self,
        node_id: NodeId,
        id: PropertyId,
        errors: &mut Vec<BakeError>,
    ) -> Option<ExprHandle> {
        let Some(def) = self.props.defs.get(&id) else {
            errors.push(BakeError::node(
                node_id,
                format!("reference to unknown property id {}", id.get()),
            ));
            return None;
        };
        if def.exposed {
            // An exposed property with no handle was dropped as a duplicate-name
            // conflict during registration; that error is already recorded.
            let Some(handle) = self.props.handles.get(&id) else {
                return None;
            };
            Some(self.module.prop(*handle))
        } else {
            let h = self.module.lit(def.default);
            // Record provenance so a live `SetPropertyDefault` on this unexposed
            // property can push its new value through the proxy property this
            // inlined literal is promoted to, instead of re-baking.
            self.literal_sites.insert(LiteralSite::Node(node_id), h);
            Some(h)
        }
    }

    /// Resolve the value feeding input port `port` of `node_id`: the source of a
    /// link into that port if one exists, else the port's inline default
    /// literal. Errors if neither is available.
    fn operand(
        &mut self,
        node_id: NodeId,
        port: &str,
        errors: &mut Vec<BakeError>,
    ) -> Option<ExprHandle> {
        if let Some(source) = self.linked_source(node_id, port) {
            return self.resolve(source, errors);
        }
        if let Some(default) = self.inline_default(node_id, port) {
            return Some(self.record_inline_literal(node_id, port, default));
        }
        errors.push(BakeError::node(
            node_id,
            format!("input port '{port}' is neither linked nor given a default"),
        ));
        None
    }

    /// Like [`operand`] but for an optional input port: a missing
    /// link *and* missing inline default is not an error — the port is simply
    /// left unconnected (the field stays at its factory default / `None`).
    ///
    /// [`operand`]: Self::operand
    fn operand_optional(
        &mut self,
        node_id: NodeId,
        port: &str,
        errors: &mut Vec<BakeError>,
    ) -> Option<ExprHandle> {
        if let Some(source) = self.linked_source(node_id, port) {
            return self.resolve(source, errors);
        }
        let default = self.inline_default(node_id, port)?;
        Some(self.record_inline_literal(node_id, port, default))
    }

    /// Bake an inline-default `value` into a module literal and record its
    /// graph site so a later value tweak can find it.
    fn record_inline_literal(
        &mut self,
        node_id: NodeId,
        port: &str,
        value: bevy_hanabi::Value,
    ) -> ExprHandle {
        let h = self.module.lit(value);
        self.literal_sites.insert(
            LiteralSite::Input {
                node: node_id,
                port: port.into(),
            },
            h,
        );
        h
    }

    /// The source node of the (single) link targeting `node_id`'s `port`.
    fn linked_source(&self, node_id: NodeId, port: &str) -> Option<NodeId> {
        let target = PortRef {
            node: node_id,
            port: port.into(),
        };
        self.graph
            .links
            .iter()
            .find(|l| l.to == target)
            .map(|l| l.from.node)
    }

    /// The inline default literal for `node_id`'s input `port`, if declared.
    fn inline_default(&self, node_id: NodeId, port: &str) -> Option<bevy_hanabi::Value> {
        let node = self.graph.node(node_id)?;
        node.inputs
            .iter()
            .find(|s| &*s.name == port)
            .map(|s| s.default)
    }

    /// Bake one modifier node into a runtime [`BoxedModifier`].
    ///
    /// The modifier instance is created by the registered
    /// [`ReflectModifier::factory`] (which allocates sensible default literals
    /// into the module), then its fields are overwritten by reflection: each
    /// expression-port field is set to the [`operand`] feeding
    /// it, and each configuration field to the matching [`EditValue`] from the
    /// node's config bag. Every failure (unregistered type, type mismatch,
    /// unbakeable value) is collected as a [`BakeError`] rather than panicking.
    ///
    /// [`operand`]: Self::operand
    fn bake_modifier(
        &mut self,
        node_id: NodeId,
        registry: &TypeRegistry,
        errors: &mut Vec<BakeError>,
    ) -> Option<BoxedModifier> {
        let node = self.graph.node(node_id).or_else(|| {
            errors.push(BakeError::node(
                node_id,
                format!("stack references missing node {}", node_id.get()),
            ));
            None
        })?;
        let NodePayload::Modifier(data) = &node.payload else {
            errors.push(BakeError::node(node_id, "expected a modifier node in a stack"));
            return None;
        };
        let (type_path, config) = match data {
            ModifierNodeData::Known { type_path, config } => (type_path, config),
            ModifierNodeData::Unknown { type_path, .. } => {
                errors.push(BakeError::node(
                    node_id,
                    format!("modifier type '{type_path}' is not registered; cannot bake"),
                ));
                return None;
            }
        };

        let Some(registration) = registry.get_with_type_path(type_path) else {
            errors.push(BakeError::node(
                node_id,
                format!("modifier type '{type_path}' is not in the type registry"),
            ));
            return None;
        };
        let Some(reflect_modifier) = registration.data::<ReflectModifier>() else {
            errors.push(BakeError::node(
                node_id,
                format!("type '{type_path}' is registered but is not a modifier"),
            ));
            return None;
        };
        let Some(schema) = modifier_schema(registration.type_info()) else {
            errors.push(BakeError::node(
                node_id,
                format!("modifier type '{type_path}' does not reflect as a struct"),
            ));
            return None;
        };

        let mut boxed = (reflect_modifier.factory)(self.module);

        // Expression ports: overwrite each field with the resolved operand
        // (linked source handle or inline-default literal). Optional ports left
        // unconnected keep the factory default.
        for field in schema.ports() {
            let optional = matches!(field.role, FieldRole::ExprPort { optional: true });
            let handle = if optional {
                self.operand_optional(node_id, &field.name, errors)
            } else {
                self.operand(node_id, &field.name, errors)
            };
            if let Some(handle) = handle
                && !set_expr_field(boxed.as_reflect_mut(), &field.name, handle, optional)
            {
                errors.push(BakeError::node(
                    node_id,
                    format!("could not set expression field '{}'", field.name),
                ));
            }
        }

        // Configuration fields (including textures): apply each value present in
        // the config bag; absent fields keep their factory default.
        for field in schema.config() {
            let Some(value) = config.get(field.name.as_ref()) else {
                continue;
            };
            if let Err(message) =
                apply_config_field(boxed.as_reflect_mut(), &field.name, value)
            {
                errors.push(BakeError::node(node_id, message));
            }
        }

        Some(boxed)
    }
}

/// Set an `ExprHandle` (or `Option<ExprHandle>`) field by name. Returns `false`
/// if the field is absent or not of the expected handle type.
fn set_expr_field(
    reflect: &mut dyn Reflect,
    name: &str,
    handle: ExprHandle,
    optional: bool,
) -> bool {
    let ReflectMut::Struct(s) = reflect.reflect_mut() else {
        return false;
    };
    let Some(field) = s.field_mut(name) else {
        return false;
    };
    if optional && let Some(slot) = field.try_downcast_mut::<Option<ExprHandle>>() {
        *slot = Some(handle);
        return true;
    }
    if let Some(slot) = field.try_downcast_mut::<ExprHandle>() {
        *slot = handle;
        return true;
    }
    false
}

/// Apply an [`EditValue`] to the named configuration field of a modifier.
fn apply_config_field(
    reflect: &mut dyn Reflect,
    name: &str,
    value: &EditValue,
) -> Result<(), String> {
    let ReflectMut::Struct(s) = reflect.reflect_mut() else {
        return Err("modifier does not reflect as a struct".to_string());
    };
    let field = s
        .field_mut(name)
        .ok_or_else(|| format!("modifier has no field '{name}'"))?;
    apply_edit_value(field, value, name)
}

/// Write one [`EditValue`] into a reflected field. Most variants wrap the field's
/// exact runtime type and are assigned directly; enums and bitflags are built
/// from their stored identity. Values that have no faithful `bevy_hanabi` 0.18
/// representation (texture-LUT gradients, pinned texture assets) report an error.
fn apply_edit_value(
    field: &mut dyn PartialReflect,
    value: &EditValue,
    name: &str,
) -> Result<(), String> {
    match value {
        EditValue::Bool(b) => assign(field, *b, name),
        EditValue::U32(u) => assign(field, *u, name),
        EditValue::UVec2(v) => assign(field, *v, name),
        EditValue::Color(c) => assign(field, *c, name),
        EditValue::Attribute(a) => assign(field, *a, name),
        EditValue::CpuVec3(v) => assign(field, v.clone(), name),
        EditValue::CpuVec4(v) => assign(field, v.clone(), name),
        EditValue::Scalar(v) => assign_scalar(field, v, name),
        EditValue::Gradient3(g) => match g {
            GradientVec3::Analytical(grad) => assign(field, grad.clone(), name),
            GradientVec3::Lut(_) => Err(format!(
                "field '{name}': texture-LUT gradient has no bevy_hanabi 0.18 representation"
            )),
        },
        EditValue::Gradient4(g) => match g {
            GradientVec4::Analytical(grad) => assign(field, grad.clone(), name),
            GradientVec4::Lut(_) => Err(format!(
                "field '{name}': texture-LUT gradient has no bevy_hanabi 0.18 representation"
            )),
        },
        EditValue::Enum { variant, .. } => assign_enum(field, variant, name),
        EditValue::Flags { bits, .. } => assign_flags(field, *bits, name),
        EditValue::Texture(_) => Err(format!(
            "field '{name}': texture baking is not yet supported"
        )),
        EditValue::Raw(_) => Err(format!(
            "field '{name}': raw config values cannot be baked"
        )),
    }
}

/// Assign a concrete value to a field, failing if the field is of another type.
fn assign<T: Reflect>(field: &mut dyn PartialReflect, value: T, name: &str) -> Result<(), String> {
    match field.try_downcast_mut::<T>() {
        Some(slot) => {
            *slot = value;
            Ok(())
        }
        None => Err(format!(
            "field '{name}': expected {}, found {}",
            std::any::type_name::<T>(),
            field.reflect_type_path()
        )),
    }
}

/// Assign a `bevy_hanabi` [`Value`] to a concrete scalar/vector field.
fn assign_scalar(field: &mut dyn PartialReflect, value: &Value, name: &str) -> Result<(), String> {
    match value {
        Value::Scalar(s) => {
            if let Some(slot) = field.try_downcast_mut::<f32>() {
                *slot = s.as_f32();
            } else if let Some(slot) = field.try_downcast_mut::<i32>() {
                *slot = s.as_i32();
            } else if let Some(slot) = field.try_downcast_mut::<u32>() {
                *slot = s.as_u32();
            } else if let Some(slot) = field.try_downcast_mut::<bool>() {
                *slot = s.as_bool();
            } else {
                return Err(scalar_mismatch(name, field));
            }
        }
        Value::Vector(v) => {
            if let Some(slot) = field.try_downcast_mut::<Vec2>() {
                *slot = v.as_vec2();
            } else if let Some(slot) = field.try_downcast_mut::<Vec3>() {
                *slot = v.as_vec3();
            } else if let Some(slot) = field.try_downcast_mut::<Vec4>() {
                *slot = v.as_vec4();
            } else if let Some(slot) = field.try_downcast_mut::<UVec2>() {
                *slot = v.as_uvec2();
            } else {
                return Err(scalar_mismatch(name, field));
            }
        }
        Value::Matrix(_) => return Err(scalar_mismatch(name, field)),
        _ => return Err(scalar_mismatch(name, field)),
    }
    Ok(())
}

fn scalar_mismatch(name: &str, field: &dyn PartialReflect) -> String {
    format!(
        "field '{name}': scalar value does not match field type {}",
        field.reflect_type_path()
    )
}

/// Set a data-less enum field to the variant of the given name (by reflect
/// apply, which matches the active variant by name).
fn assign_enum(field: &mut dyn PartialReflect, variant: &str, name: &str) -> Result<(), String> {
    let dynamic = DynamicEnum::new(variant.to_string(), DynamicVariant::Unit);
    field
        .try_apply(&dynamic)
        .map_err(|e| format!("field '{name}': cannot select enum variant '{variant}': {e:?}"))
}

/// Set a bitflags newtype field (a tuple struct wrapping one integer) to `bits`,
/// narrowed to the field's actual integer width.
fn assign_flags(field: &mut dyn PartialReflect, bits: u64, name: &str) -> Result<(), String> {
    let ReflectMut::TupleStruct(ts) = field.reflect_mut() else {
        return Err(format!("field '{name}': flags field is not a tuple struct"));
    };
    let inner = ts
        .field_mut(0)
        .ok_or_else(|| format!("field '{name}': flags newtype has no inner value"))?;
    if let Some(slot) = inner.try_downcast_mut::<u8>() {
        *slot = bits as u8;
    } else if let Some(slot) = inner.try_downcast_mut::<u16>() {
        *slot = bits as u16;
    } else if let Some(slot) = inner.try_downcast_mut::<u32>() {
        *slot = bits as u32;
    } else if let Some(slot) = inner.try_downcast_mut::<u64>() {
        *slot = bits;
    } else {
        return Err(format!(
            "field '{name}': unsupported flags integer type {}",
            inner.reflect_type_path()
        ));
    }
    Ok(())
}

/// Build a [`Module`] from `graph`'s expression nodes and properties, returning
/// the module and the `NodeId → ExprHandle` map for every expression node that
/// is reachable from a modifier or another expression.
///
/// Only expression nodes reachable as operands or modifier inputs are
/// materialized; a dangling expression node with no consumer contributes
/// nothing to the arena. Errors (cycles, unknown properties, missing inputs,
/// wrong node kinds) are collected rather than fatal, so the caller can surface
/// all of them at once.
pub fn bake_module(
    graph: &EffectGraph,
) -> Result<(Module, HashMap<NodeId, ExprHandle>), Vec<BakeError>> {
    let mut module = Module::default();
    let mut errors = Vec::new();

    let props = bake_properties(graph, &mut module, &mut errors);

    let mut baker = ExprBaker {
        graph,
        props: &props,
        module: &mut module,
        handles: HashMap::new(),
        literal_sites: HashMap::new(),
        visiting: Vec::new(),
    };

    // Resolve every expression node that participates in the graph. A node is a
    // participant if it is the source or target of a link, or an operand-bearing
    // expression; resolving each pulls in its operand subtree transitively.
    let participants = expr_participants(graph);
    for node_id in participants {
        baker.resolve(node_id, &mut errors);
    }

    let handles = std::mem::take(&mut baker.handles);
    drop(baker);

    if errors.is_empty() {
        Ok((module, handles))
    } else {
        Err(errors)
    }
}

/// Bake a whole [`EffectGraph`] into a runtime [`EffectAsset`].
///
/// Builds the expression [`Module`] (properties + every expression reachable
/// from a stack modifier), instantiates each stack's modifiers in execution
/// order (`Init → Update → Render`), and assembles them with the header into an
/// `EffectAsset`. Every problem is collected as a [`BakeError`] (attributed to
/// the node, property, or graph at fault) rather than panicking; the asset is
/// returned only when the graph bakes cleanly.
pub fn bake(graph: &EffectGraph, registry: &TypeRegistry) -> Result<EffectAsset, Vec<BakeError>> {
    bake_with_provenance(graph, registry).map(|(asset, _sites)| asset)
}

/// Like [`bake`], but also returns the [`LiteralSites`] provenance mapping every
/// baked literal to its graph origin. Used by the live-tweak path to bind value
/// edits to the proxy properties promoted from those literals.
pub fn bake_with_provenance(
    graph: &EffectGraph,
    registry: &TypeRegistry,
) -> Result<(EffectAsset, LiteralSites), Vec<BakeError>> {
    let mut module = Module::default();
    let mut errors = Vec::new();
    let props = bake_properties(graph, &mut module, &mut errors);

    let mut baker = ExprBaker {
        graph,
        props: &props,
        module: &mut module,
        handles: HashMap::new(),
        literal_sites: HashMap::new(),
        visiting: Vec::new(),
    };

    // Bake each stack's members, routing every modifier to its execution stage.
    // A modifier whose kind contradicts its stack (e.g. a render modifier in an
    // Init stack) is reported rather than silently misplaced.
    let mut init: Vec<bevy_hanabi::BoxedModifier> = Vec::new();
    let mut update: Vec<bevy_hanabi::BoxedModifier> = Vec::new();
    let mut render: Vec<Box<dyn bevy_hanabi::RenderModifier>> = Vec::new();
    for stack in &graph.stacks {
        for &member in &stack.members {
            let Some(boxed) = baker.bake_modifier(member, registry, &mut errors) else {
                continue;
            };
            match (stack.group, boxed.as_render().is_some()) {
                (ModifierGroup::Init, false) => init.push(boxed),
                (ModifierGroup::Update, false) => update.push(boxed),
                (ModifierGroup::Render, true) => {
                    render.push(boxed.as_render().unwrap().boxed_render_clone())
                }
                (ModifierGroup::Render, false) => errors.push(BakeError::node(
                    member,
                    "non-render modifier placed in a Render stack",
                )),
                (group, true) => errors.push(BakeError::node(
                    member,
                    format!("render modifier placed in a {group:?} stack"),
                )),
            }
        }
    }

    let literal_sites = std::mem::take(&mut baker.literal_sites);
    drop(baker);

    if !errors.is_empty() {
        return Err(errors);
    }

    let header = &graph.header;
    let mut asset = EffectAsset::new(header.capacity, header.spawner, module);
    asset.name = header.name.to_string();
    asset.simulation_space = header.simulation_space;
    asset.simulation_condition = header.simulation_condition;
    asset.z_layer_2d = header.z_layer_2d;
    for m in init {
        asset = asset.add_modifier(ModifierContext::Init, m);
    }
    for m in update {
        asset = asset.add_modifier(ModifierContext::Update, m);
    }
    for m in render {
        asset = asset.add_render_modifier(m);
    }
    Ok((asset, literal_sites))
}

/// Bake a graph for live preview, tagging the asset name so its compiled
/// shaders get a document-unique `hanabi/{name}_…` path. `preview_tag` is the
/// owning document's preview tag.
///
/// The tag lives only on the throwaway preview asset (and the proxy cloned from
/// it); the saved graph keeps its plain `header.name`.
pub fn bake_preview(graph: &EffectGraph, registry: &TypeRegistry, preview_tag: u64) -> EffectAsset {
    bake_preview_with_provenance(graph, registry, preview_tag).0
}

/// Like [`bake_preview`], but also returns the [`LiteralSites`] provenance for
/// the baked asset, so the live-tweak path can bind value edits to proxy
/// properties. On bake failure the provenance is empty.
pub fn bake_preview_with_provenance(
    graph: &EffectGraph,
    registry: &TypeRegistry,
    preview_tag: u64,
) -> (EffectAsset, LiteralSites) {
    let (mut asset, sites) = bake_or_empty_with_provenance(graph, registry);
    asset.name = preview_asset_name(&graph.header.name, preview_tag);
    (asset, sites)
}

/// Document-unique preview asset name: `{base}~{tag}`. The `~` separator avoids
/// the `_` that hanabi uses to delimit `{name}_{phase}_{hash}` shader paths.
pub fn preview_asset_name(base: &str, preview_tag: u64) -> String {
    format!("{base}~{preview_tag}")
}

/// Bake a graph, falling back to an empty (inert but renderable) asset when it
/// fails. Used by the seeding/reconcile path where the viewport must always
/// have *some* asset to instantiate; the bake errors are logged for the UI to
/// surface separately rather than aborting document creation.
pub fn bake_or_empty(graph: &EffectGraph, registry: &TypeRegistry) -> EffectAsset {
    bake_or_empty_with_provenance(graph, registry).0
}

/// Like [`bake_or_empty`], but also returns the [`LiteralSites`] provenance
/// (empty when the bake fails and the inert fallback is used).
pub fn bake_or_empty_with_provenance(
    graph: &EffectGraph,
    registry: &TypeRegistry,
) -> (EffectAsset, LiteralSites) {
    bake_with_provenance(graph, registry).unwrap_or_else(|errors| {
        bevy::log::error!(
            "effect graph failed to bake ({} error(s)): {errors:?}",
            errors.len()
        );
        let mut asset = EffectAsset::new(graph.header.capacity, graph.header.spawner, Module::default());
        asset.name = graph.header.name.to_string();
        (asset, LiteralSites::default())
    })
}
/// inline-defaulted operator with no incoming links is still built).
fn expr_participants(graph: &EffectGraph) -> Vec<NodeId> {
    let mut seen = Vec::new();
    let push = |id: NodeId, seen: &mut Vec<NodeId>| {
        if !seen.contains(&id) {
            seen.push(id);
        }
    };
    for link in &graph.links {
        push(link.from.node, &mut seen);
        push(link.to.node, &mut seen);
    }
    for node in &graph.nodes {
        if let NodePayload::Expr(expr) = &node.payload
            && !expr_input_ports(expr).is_empty()
        {
            push(node.id, &mut seen);
        }
    }
    seen
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        EffectHeader, GraphLink, GraphNode, InputSlot, PortRef,
    };
    use bevy_hanabi::graph::expr::BinaryOperator;
    use bevy_hanabi::{
        Attribute, Expr, SimulationCondition, SimulationSpace, SpawnerSettings, Value,
    };

    fn header() -> EffectHeader {
        EffectHeader {
            name: "t".into(),
            capacity: 32,
            spawner: SpawnerSettings::rate(1.0.into()),
            simulation_space: SimulationSpace::Global,
            simulation_condition: SimulationCondition::Always,
            z_layer_2d: 0.0,
        }
    }

    fn graph_with(nodes: Vec<GraphNode>, links: Vec<GraphLink>, props: Vec<PropertyDef>) -> EffectGraph {
        let max = nodes.iter().map(|n| n.id.get()).max().unwrap_or(0);
        EffectGraph {
            header: header(),
            properties: props,
            nodes,
            stacks: vec![],
            links,
            next_id: max + 1,
        }
    }

    fn expr_node(id: u32, expr: ExprNode, inputs: Vec<InputSlot>) -> GraphNode {
        GraphNode {
            id: NodeId::new(id).unwrap(),
            payload: NodePayload::Expr(expr),
            inputs,
        }
    }

    fn pid(n: u32) -> PropertyId {
        PropertyId::new(n).unwrap()
    }

    fn prop_def(id: u32, name: &str, default: Value, exposed: bool) -> PropertyDef {
        PropertyDef {
            id: pid(id),
            name: name.into(),
            default,
            exposed,
        }
    }

    #[test]
    fn bakes_binary_with_link_and_inline_default() {
        // n1 = attr(position); n2 = n1 + lit(2.0 via inline default on rhs)
        let n1 = expr_node(1, ExprNode::Attribute(Attribute::POSITION), vec![]);
        let n2 = expr_node(
            2,
            ExprNode::Binary(BinaryOperator::Add),
            vec![InputSlot {
                name: "rhs".into(),
                default: Value::from(2.0f32),
            }],
        );
        let link = GraphLink {
            from: PortRef {
                node: NodeId::new(1).unwrap(),
                port: "out".into(),
            },
            to: PortRef {
                node: NodeId::new(2).unwrap(),
                port: "lhs".into(),
            },
        };
        let graph = graph_with(vec![n1, n2], vec![link], vec![]);

        let (module, handles) = bake_module(&graph).expect("bake");
        assert_eq!(handles.len(), 2);
        let top = handles[&NodeId::new(2).unwrap()];
        assert!(matches!(module.get(top), Some(Expr::Binary { .. })));
    }

    #[test]
    fn exposed_property_becomes_module_property() {
        // A property reference consumed by a unary so it participates in baking.
        let prop = expr_node(1, ExprNode::Property(pid(10)), vec![]);
        let unary = expr_node(
            2,
            ExprNode::Unary(bevy_hanabi::graph::expr::UnaryOperator::Abs),
            vec![],
        );
        let link = GraphLink {
            from: PortRef {
                node: NodeId::new(1).unwrap(),
                port: "out".into(),
            },
            to: PortRef {
                node: NodeId::new(2).unwrap(),
                port: "in".into(),
            },
        };
        let graph = graph_with(
            vec![prop, unary],
            vec![link],
            vec![prop_def(10, "speed", Value::from(4.0f32), true)],
        );

        let (module, _) = bake_module(&graph).expect("bake");
        assert_eq!(module.properties().len(), 1);
        assert_eq!(module.properties()[0].name(), "speed");
    }

    #[test]
    fn edit_only_property_is_inlined() {
        let n1 = expr_node(1, ExprNode::Property(pid(10)), vec![]);
        let unary = expr_node(2, ExprNode::Unary(bevy_hanabi::graph::expr::UnaryOperator::Abs), vec![]);
        let link = GraphLink {
            from: PortRef { node: NodeId::new(1).unwrap(), port: "out".into() },
            to: PortRef { node: NodeId::new(2).unwrap(), port: "in".into() },
        };
        let graph = graph_with(
            vec![n1, unary],
            vec![link],
            vec![prop_def(10, "tweak", Value::from(7.0f32), false)],
        );

        let (module, handles) = bake_module(&graph).expect("bake");
        assert!(module.properties().is_empty());
        let lit = handles[&NodeId::new(1).unwrap()];
        assert!(matches!(module.get(lit), Some(Expr::Literal(_))));
    }

    #[test]
    fn unexposed_property_ref_records_node_site() {
        let n1 = expr_node(1, ExprNode::Property(pid(10)), vec![]);
        let graph = graph_with(
            vec![n1],
            vec![],
            vec![prop_def(10, "tweak", Value::from(7.0f32), false)],
        );

        let mut module = Module::default();
        let mut errors = Vec::new();
        let props = bake_properties(&graph, &mut module, &mut errors);
        let mut baker = ExprBaker {
            graph: &graph,
            props: &props,
            module: &mut module,
            handles: HashMap::new(),
            literal_sites: HashMap::new(),
            visiting: Vec::new(),
        };
        let node = NodeId::new(1).unwrap();
        let h = baker.resolve(node, &mut errors).expect("resolve");
        assert!(errors.is_empty());
        assert_eq!(
            baker.literal_sites.get(&LiteralSite::Node(node)).copied(),
            Some(h),
            "an unexposed property reference records a Node site for its inlined literal"
        );
        assert!(matches!(baker.module.get(h), Some(Expr::Literal(_))));
    }

    #[test]
    fn detects_cycle() {
        // n1(unary) -> n2(unary) -> n1 : a cycle.
        let n1 = expr_node(1, ExprNode::Unary(bevy_hanabi::graph::expr::UnaryOperator::Abs), vec![]);
        let n2 = expr_node(2, ExprNode::Unary(bevy_hanabi::graph::expr::UnaryOperator::Abs), vec![]);
        let links = vec![
            GraphLink {
                from: PortRef { node: NodeId::new(1).unwrap(), port: "out".into() },
                to: PortRef { node: NodeId::new(2).unwrap(), port: "in".into() },
            },
            GraphLink {
                from: PortRef { node: NodeId::new(2).unwrap(), port: "out".into() },
                to: PortRef { node: NodeId::new(1).unwrap(), port: "in".into() },
            },
        ];
        let graph = graph_with(vec![n1, n2], links, vec![]);

        let errors = bake_module(&graph).unwrap_err();
        assert!(errors.iter().any(|e| e.message.contains("cycle")));
    }

    #[test]
    fn unknown_property_errors() {
        let n1 = expr_node(1, ExprNode::Property(pid(99)), vec![]);
        let unary = expr_node(2, ExprNode::Unary(bevy_hanabi::graph::expr::UnaryOperator::Abs), vec![]);
        let link = GraphLink {
            from: PortRef { node: NodeId::new(1).unwrap(), port: "out".into() },
            to: PortRef { node: NodeId::new(2).unwrap(), port: "in".into() },
        };
        let graph = graph_with(vec![n1, unary], vec![link], vec![]);

        let errors = bake_module(&graph).unwrap_err();
        assert!(errors.iter().any(|e| {
            e.subject == BakeSubject::Node(NodeId::new(1).unwrap())
                && e.message.contains("unknown property")
        }));
    }

    #[test]
    fn duplicate_exposed_property_name_errors() {
        // Exposed properties become runtime Module properties keyed by name; a
        // name collision is an inconsistency that blocks baking (but never panics).
        let graph = graph_with(
            vec![],
            vec![],
            vec![
                prop_def(10, "dup", Value::from(1.0f32), true),
                prop_def(11, "dup", Value::from(2.0f32), true),
            ],
        );

        let errors = bake_module(&graph).unwrap_err();
        // The error is attributed to the conflicting (second) property so the UI
        // can link straight to it.
        assert!(errors.iter().any(|e| {
            e.subject == BakeSubject::Property(pid(11)) && e.message.contains("share the name 'dup'")
        }));
    }

    #[test]
    fn duplicate_edit_only_property_name_is_tolerated() {
        // Non-exposed properties are baked to literals and referenced by id, so a
        // shared display name is harmless and must not fail the bake.
        let graph = graph_with(
            vec![],
            vec![],
            vec![
                prop_def(10, "tweak", Value::from(1.0f32), false),
                prop_def(11, "tweak", Value::from(2.0f32), false),
            ],
        );

        let (module, _) = bake_module(&graph).expect("edit-only duplicates are harmless");
        assert!(module.properties().is_empty());
    }

    #[test]
    fn distinct_ids_resolve_independently() {
        // Two edit-only properties share a name but have distinct ids; each
        // reference resolves to its own value via id, not the shared name.
        let r1 = expr_node(1, ExprNode::Property(pid(10)), vec![]);
        let r2 = expr_node(2, ExprNode::Property(pid(11)), vec![]);
        let add = expr_node(3, ExprNode::Binary(BinaryOperator::Add), vec![]);
        let links = vec![
            GraphLink {
                from: PortRef { node: NodeId::new(1).unwrap(), port: "out".into() },
                to: PortRef { node: NodeId::new(3).unwrap(), port: "lhs".into() },
            },
            GraphLink {
                from: PortRef { node: NodeId::new(2).unwrap(), port: "out".into() },
                to: PortRef { node: NodeId::new(3).unwrap(), port: "rhs".into() },
            },
        ];
        let graph = graph_with(
            vec![r1, r2, add],
            links,
            vec![
                prop_def(10, "same", Value::from(1.0f32), false),
                prop_def(11, "same", Value::from(2.0f32), false),
            ],
        );

        let (module, handles) = bake_module(&graph).expect("bake");
        // Both references baked to distinct literal expressions.
        assert!(matches!(
            module.get(handles[&NodeId::new(1).unwrap()]),
            Some(Expr::Literal(_))
        ));
        assert!(matches!(
            module.get(handles[&NodeId::new(2).unwrap()]),
            Some(Expr::Literal(_))
        ));
    }

    // --- Modifier baking (B2) ---

    use std::collections::BTreeMap;

    use bevy::ecs::reflect::AppTypeRegistry;
    use bevy::reflect::TypePath;
    use bevy_hanabi::{
        ColorBlendMask, ColorBlendMode, CpuValue, SetColorModifier, SetPositionSphereModifier,
    };

    use crate::model::ModifierNodeData;

    /// A type registry populated with all built-in modifiers (and their
    /// [`ReflectModifier`] factories) via `bevy_hanabi`'s own registration.
    fn test_registry() -> AppTypeRegistry {
        let registry = AppTypeRegistry::default();
        bevy_hanabi::register_modifiers(&registry);
        registry
    }

    fn modifier_node(
        id: u32,
        type_path: &str,
        config: BTreeMap<crate::model::SharedStr, EditValue>,
        inputs: Vec<InputSlot>,
    ) -> GraphNode {
        GraphNode {
            id: NodeId::new(id).unwrap(),
            payload: NodePayload::Modifier(ModifierNodeData::Known {
                type_path: type_path.into(),
                config,
            }),
            inputs,
        }
    }

    /// Drive [`ExprBaker::bake_modifier`] for a single node, resolving operands
    /// on demand against `graph`.
    fn bake_one(
        graph: &EffectGraph,
        registry: &TypeRegistry,
        node_id: NodeId,
    ) -> (Option<BoxedModifier>, Vec<BakeError>) {
        let mut module = Module::default();
        let mut errors = Vec::new();
        let props = bake_properties(graph, &mut module, &mut errors);
        let mut baker = ExprBaker {
            graph,
            props: &props,
            module: &mut module,
            handles: HashMap::new(),
            literal_sites: HashMap::new(),
            visiting: Vec::new(),
        };
        let baked = baker.bake_modifier(node_id, registry, &mut errors);
        (baked, errors)
    }

    #[test]
    fn bakes_modifier_enum_and_flags_config() {
        let mut config = BTreeMap::new();
        config.insert(
            "blend".into(),
            EditValue::Enum {
                type_path: ColorBlendMode::type_path().into(),
                variant: "Add".into(),
            },
        );
        config.insert(
            "mask".into(),
            EditValue::Flags {
                type_path: ColorBlendMask::type_path().into(),
                bits: ColorBlendMask::RGB.bits() as u64,
            },
        );
        config.insert(
            "color".into(),
            EditValue::CpuVec4(CpuValue::Single(Vec4::new(0.2, 0.4, 0.6, 1.0))),
        );
        let node = modifier_node(1, SetColorModifier::type_path(), config, vec![]);
        let graph = graph_with(vec![node], vec![], vec![]);

        let (baked, errors) = bake_one(&graph, &test_registry().read(), NodeId::new(1).unwrap());
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        let m = baked.expect("baked");
        assert!(m.as_render().is_some(), "expected a render modifier");
        let scm = m
            .as_reflect()
            .downcast_ref::<SetColorModifier>()
            .expect("SetColorModifier");
        assert_eq!(scm.blend, ColorBlendMode::Add);
        assert_eq!(scm.mask, ColorBlendMask::RGB);
        assert_eq!(scm.color, CpuValue::Single(Vec4::new(0.2, 0.4, 0.6, 1.0)));
    }

    #[test]
    fn bakes_modifier_ports_from_inline_defaults() {
        // No links: each required port is fed by its inline-default literal.
        let node = modifier_node(
            1,
            SetPositionSphereModifier::type_path(),
            BTreeMap::new(),
            vec![
                InputSlot {
                    name: "center".into(),
                    default: Value::from(Vec3::new(1.0, 2.0, 3.0)),
                },
                InputSlot {
                    name: "radius".into(),
                    default: Value::from(5.0_f32),
                },
            ],
        );
        let graph = graph_with(vec![node], vec![], vec![]);

        let mut module = Module::default();
        let mut errors = Vec::new();
        let props = bake_properties(&graph, &mut module, &mut errors);
        let mut baker = ExprBaker {
            graph: &graph,
            props: &props,
            module: &mut module,
            handles: HashMap::new(),
            literal_sites: HashMap::new(),
            visiting: Vec::new(),
        };
        let baked = baker
            .bake_modifier(NodeId::new(1).unwrap(), &test_registry().read(), &mut errors)
            .expect("baked");
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");

        let m = baked;
        assert!(m.as_render().is_none(), "expected a plain modifier");
        let spm = m
            .as_reflect()
            .downcast_ref::<SetPositionSphereModifier>()
            .expect("SetPositionSphereModifier");
        // The port fields point at the inline-default literals in the module.
        assert_eq!(
            module.get(spm.radius),
            Some(&Expr::Literal(bevy_hanabi::graph::expr::LiteralExpr::new(
                5.0_f32
            )))
        );
        assert_eq!(
            module.get(spm.center),
            Some(&Expr::Literal(bevy_hanabi::graph::expr::LiteralExpr::new(
                Vec3::new(1.0, 2.0, 3.0)
            )))
        );
    }

    #[test]
    fn unregistered_modifier_type_errors() {
        let node = modifier_node(
            1,
            "not::a::real::Modifier",
            BTreeMap::new(),
            vec![],
        );
        let graph = graph_with(vec![node], vec![], vec![]);

        let (baked, errors) = bake_one(&graph, &test_registry().read(), NodeId::new(1).unwrap());
        assert!(baked.is_none());
        assert!(errors.iter().any(|e| {
            e.subject == BakeSubject::Node(NodeId::new(1).unwrap())
                && e.message.contains("not in the type registry")
        }));
    }

    // --- Whole-graph assembly (B3) ---

    use crate::model::{GraphStack, StackId};

    fn graph_with_stacks(nodes: Vec<GraphNode>, stacks: Vec<GraphStack>) -> EffectGraph {
        let max = nodes.iter().map(|n| n.id.get()).max().unwrap_or(0);
        EffectGraph {
            header: header(),
            properties: vec![],
            nodes,
            stacks,
            links: vec![],
            next_id: max + 1,
        }
    }

    fn stack(id: u32, group: ModifierGroup, members: Vec<u32>) -> GraphStack {
        GraphStack {
            id: StackId::new(id).unwrap(),
            group,
            members: members.into_iter().map(|m| NodeId::new(m).unwrap()).collect(),
        }
    }

    #[test]
    fn bakes_whole_graph_into_effect_asset() {
        // Init: position-sphere (ports from inline defaults). Render: set-color.
        let pos = modifier_node(
            1,
            SetPositionSphereModifier::type_path(),
            BTreeMap::new(),
            vec![
                InputSlot {
                    name: "center".into(),
                    default: Value::from(Vec3::ZERO),
                },
                InputSlot {
                    name: "radius".into(),
                    default: Value::from(2.0_f32),
                },
            ],
        );
        let color = modifier_node(2, SetColorModifier::type_path(), BTreeMap::new(), vec![]);
        let graph = graph_with_stacks(
            vec![pos, color],
            vec![
                stack(1, ModifierGroup::Init, vec![1]),
                stack(2, ModifierGroup::Render, vec![2]),
            ],
        );

        let asset = bake(&graph, &test_registry().read()).expect("bake");
        assert_eq!(asset.name, "t");
        assert_eq!(asset.capacity(), 32);
        assert_eq!(asset.init_modifiers().count(), 1);
        assert_eq!(asset.update_modifiers().count(), 0);
        assert_eq!(asset.render_modifiers().count(), 1);
        assert!(
            asset
                .init_modifiers()
                .next()
                .unwrap()
                .as_reflect()
                .downcast_ref::<SetPositionSphereModifier>()
                .is_some()
        );
        assert!(
            asset
                .render_modifiers()
                .next()
                .unwrap()
                .as_modifier()
                .as_reflect()
                .downcast_ref::<SetColorModifier>()
                .is_some()
        );
    }

    #[test]
    fn provenance_records_inline_default_sites() {
        let pos = modifier_node(
            1,
            SetPositionSphereModifier::type_path(),
            BTreeMap::new(),
            vec![
                InputSlot {
                    name: "center".into(),
                    default: Value::from(Vec3::ZERO),
                },
                InputSlot {
                    name: "radius".into(),
                    default: Value::from(2.0_f32),
                },
            ],
        );
        let graph =
            graph_with_stacks(vec![pos], vec![stack(1, ModifierGroup::Init, vec![1])]);

        let (asset, sites) =
            bake_with_provenance(&graph, &test_registry().read()).expect("bake");

        let node = NodeId::new(1).unwrap();
        let radius = sites
            .get(&LiteralSite::Input {
                node,
                port: "radius".into(),
            })
            .copied()
            .expect("radius inline-default site recorded");
        assert!(matches!(asset.module().get(radius), Some(Expr::Literal(_))));
        assert!(sites.contains_key(&LiteralSite::Input {
            node,
            port: "center".into(),
        }));
    }

    #[test]
    fn render_modifier_in_init_stack_errors() {
        let color = modifier_node(1, SetColorModifier::type_path(), BTreeMap::new(), vec![]);
        let graph = graph_with_stacks(vec![color], vec![stack(1, ModifierGroup::Init, vec![1])]);

        let errors = match bake(&graph, &test_registry().read()) {
            Err(errors) => errors,
            Ok(_) => panic!("expected a bake error"),
        };
        assert!(errors.iter().any(|e| {
            e.subject == BakeSubject::Node(NodeId::new(1).unwrap())
                && e.message.contains("render modifier placed in a Init stack")
        }));
    }
}
