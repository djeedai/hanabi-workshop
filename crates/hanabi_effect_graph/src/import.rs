//! Best-effort import of a baked [`EffectAsset`] into an [`EffectGraph`].
//!
//! Baking is lossy in one direction: an expression sub-graph collapses into a
//! flat `Module` arena, and edit-only metadata (node layout, edit-only
//! properties) is dropped. Import therefore reconstructs what is *cleanly*
//! reversible and reports the rest as [`ImportWarning`]s rather than failing:
//!
//! - **Emitter settings** — name, capacity, simulation space/condition, and 2D
//!   z-layer map back one-to-one; `asset.spawner` becomes a CPU spawn-source
//!   node linked exclusively to the imported emitter (see [`import_effect`])
//!   rather than part of the emitter settings.
//! - **Properties** — every runtime `Module` property becomes an exposed
//!   [`PropertyDef`].
//! - **Modifiers** — each modifier's non-expression fields are read back into
//!   the edit config bag via reflection (the inverse of the bake's
//!   `apply_config_field`). Expression input ports are recovered when they feed
//!   from a literal (an inline default) or a property reference (a dedicated
//!   reference node); any other expression (an operator sub-graph, an attribute
//!   read, a built-in) cannot be faithfully un-flattened and is reset to a zero
//!   default with a warning. `EmitSpawnEventModifier` and
//!   `InheritAttributeModifier` participate in inter-emitter topology that a
//!   single, flat `EffectAsset` cannot express at all — [`import_effect`] still
//!   imports them as ordinary modifier nodes (their non-topology fields
//!   round-trip normally) but reports a dedicated warning, since the resulting
//!   emitter is always imported standalone with no parent or GPU child wired
//!   up.
//!
//! [`FORMAT_VERSION`]: crate::model::FORMAT_VERSION

use std::collections::{BTreeMap, HashMap};

use bevy::{
    math::{UVec2, Vec2, Vec3, Vec4},
    reflect::{PartialReflect, Reflect, ReflectRef, TypePath, structs::Struct},
};
use bevy_hanabi::{
    Attribute, CpuValue, EffectAsset, EmitSpawnEventModifier, ExprHandle, Gradient,
    InheritAttributeModifier, Module, Value,
    graph::expr::{Expr, LiteralExpr, PropertyExpr, PropertyHandle},
};

use crate::{
    ModifierGroup,
    model::{
        EditValue, EffectGraph, EmitterGraph, EmitterId, ExprNode, GradientVec3, GradientVec4,
        GraphLink, GraphNode, GraphStack, ImageBinding, InputSlot, ModifierNodeData, NodeId,
        NodePayload, PortRef, PropertyDef, PropertyId, SharedStr, SlotId, SourceContext,
        SourceKind, SourceLink, StackId, TextureSlotDef,
    },
    schema::{ConfigKind, FieldRole, OUTPUT_PORT, modifier_schema},
};

/// A reversibility gap encountered while importing.
///
/// Surfaced to the user so the silent loss is visible. Importing never fails:
/// the graph is always returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportWarning {
    pub message: String,
}

impl ImportWarning {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ImportWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// Import a baked [`EffectAsset`] into an isolated [`EmitterGraph`].
///
/// Temporary emitter-local compatibility helper, kept for callers that need a
/// bare, unconnected [`EmitterGraph`] with no spawn source (e.g. existing
/// tests exercising import in isolation): the returned graph is not itself
/// loadable or bakeable because it has no linked spawn-source node. Prefer
/// [`import_effect`], which returns a complete, connected, one-emitter
/// [`EffectGraph`].
///
/// The returned warnings list every part of the asset that could not be
/// faithfully reversed (see the module docs); it is empty for an asset whose
/// modifiers take only literal and property inputs.
///
/// [`FORMAT_VERSION`]: crate::model::FORMAT_VERSION
pub fn import_emitter(asset: &EffectAsset) -> (EmitterGraph, Vec<ImportWarning>) {
    let id = EmitterId::new(1).expect("1 is a valid NonZeroU32");
    let mut next_id = id.get() + 1;
    build_imported_emitter(asset, id, &mut next_id)
}

/// Import a baked [`EffectAsset`] as a connected one-emitter effect graph.
///
/// Topology is entirely absent from a flat `EffectAsset`, so the imported
/// graph is always a single emitter driven by one freshly created
/// [`SourceKind::CpuSpawner`] source carrying `asset.spawner` — the graph-level
/// analogue of the version-1 nested header's `spawner` field.
/// Additionally warns (on top of [`import_emitter`]'s own per-field warnings)
/// when a recovered modifier is an `EmitSpawnEventModifier` or
/// `InheritAttributeModifier`: both encode parent/child emitter topology that
/// cannot be reconstructed from one asset, so the imported emitter never has
/// any parent or GPU-driven child wired up even though the modifier node
/// itself round-trips.
///
/// [`FORMAT_VERSION`]: crate::model::FORMAT_VERSION
pub fn import_effect(asset: &EffectAsset) -> (EffectGraph, Vec<ImportWarning>) {
    let mut effect_graph = EffectGraph::empty();
    let emitter_id = effect_graph.alloc_emitter_id();
    let (emitter, mut warnings) =
        build_imported_emitter(asset, emitter_id, &mut effect_graph.next_id);

    for node in &emitter.nodes {
        let NodePayload::Modifier(ModifierNodeData::Known { type_path, .. }) = &node.payload else {
            continue;
        };
        let unrecoverable = type_path.as_ref() == EmitSpawnEventModifier::type_path()
            || type_path.as_ref() == InheritAttributeModifier::type_path();
        if unrecoverable {
            warnings.push(ImportWarning::new(format!(
                "modifier '{type_path}' encodes inter-emitter parent/child topology that a \
                 single EffectAsset cannot express; imported as an isolated emitter with no \
                 parent or GPU child wired up"
            )));
        }
    }

    let source_id = effect_graph.alloc_source_id();
    effect_graph.sources.push(SourceContext {
        id: source_id,
        kind: SourceKind::CpuSpawner {
            settings: asset.spawner,
        },
    });
    effect_graph.source_links.push(SourceLink {
        source: source_id,
        emitter: emitter_id,
    });
    effect_graph.emitters.push(emitter);

    (effect_graph, warnings)
}

/// Build one emitter while sharing the caller's effect-wide id allocator.
///
/// Builds an [`EmitterGraph`] with the given `id`, minting every
/// node/property/slot/stack id from `next_id` (the caller's own counter for
/// [`import_emitter`], or an [`EffectGraph`]'s shared allocator for
/// [`import_effect`]).
fn build_imported_emitter(
    asset: &EffectAsset,
    id: EmitterId,
    next_id: &mut u32,
) -> (EmitterGraph, Vec<ImportWarning>) {
    let module = asset.module();

    let mut graph = EmitterGraph {
        name: asset.name.clone().into(),
        capacity: asset.capacity(),
        simulation_space: asset.simulation_space,
        simulation_condition: asset.simulation_condition,
        z_layer_2d: asset.z_layer_2d,
        ..EmitterGraph::empty(id)
    };

    // Runtime properties become exposed edit properties. Index by name so a
    // modifier's property-reference expression can resolve back to a stable id.
    let mut props_by_name: HashMap<String, PropertyId> = HashMap::new();
    for prop in module.properties() {
        let id = PropertyId::new(*next_id).expect("property id allocator overflow");
        *next_id += 1;
        graph.properties.push(PropertyDef {
            id,
            name: prop.name().into(),
            default: *prop.default_value(),
            exposed: true,
        });
        props_by_name.insert(prop.name().to_string(), id);
    }

    // Texture slots: one editor slot per baked texture-layout entry. Asset
    // bindings live in the per-instance `EffectMaterial`, not the asset file,
    // so every recovered slot is host-supplied (named); texture ports rebind
    // to it by stable id.
    let mut slot_ids: Vec<SlotId> = Vec::new();
    for slot in module.texture_layout().layout {
        let id = SlotId::new(*next_id).expect("slot id allocator overflow");
        *next_id += 1;
        graph.texture_slots.push(TextureSlotDef {
            id,
            name: slot.name.into(),
            dimension: slot.dimension,
        });
        slot_ids.push(id);
    }

    let mut importer = Importer {
        graph: &mut graph,
        module,
        props_by_name: &props_by_name,
        prop_ref_nodes: HashMap::new(),
        slot_ids,
        warnings: Vec::new(),
        next_id,
    };

    let init: Vec<NodeId> = asset
        .init_modifiers()
        .filter_map(|m| importer.import_modifier(m.as_reflect()))
        .collect();
    let update: Vec<NodeId> = asset
        .update_modifiers()
        .filter_map(|m| importer.import_modifier(m.as_reflect()))
        .collect();
    let render: Vec<NodeId> = asset
        .render_modifiers()
        .filter_map(|m| importer.import_modifier(m.as_modifier().as_reflect()))
        .collect();

    let warnings = std::mem::take(&mut importer.warnings);

    // One stack per non-empty phase, matching the bake's `Init → Update →
    // Render` execution order.
    for (group, members) in [
        (ModifierGroup::Init, init),
        (ModifierGroup::Update, update),
        (ModifierGroup::Render, render),
    ] {
        if members.is_empty() {
            continue;
        }
        let id = StackId::new(*next_id).expect("stack id allocator overflow");
        *next_id += 1;
        graph.stacks.push(GraphStack { id, group, members });
    }

    (graph, warnings)
}

/// Mutable state threaded through a single import pass.
struct Importer<'a> {
    graph: &'a mut EmitterGraph,
    module: &'a Module,
    props_by_name: &'a HashMap<String, PropertyId>,
    /// Property reference nodes created on demand, reused across ports so each
    /// property has a single `ExprNode::Property` source node.
    prop_ref_nodes: HashMap<PropertyId, NodeId>,
    /// Recovered texture slots, indexed by their baked sampling index, used to
    /// map a port's slot-index literal back to a stable [`SlotId`].
    slot_ids: Vec<SlotId>,
    warnings: Vec<ImportWarning>,
    /// Shared id counter (see [`build_imported_emitter`]), minting every node
    /// id this pass creates.
    next_id: &'a mut u32,
}

impl Importer<'_> {
    /// Mint a fresh, never-before-used [`NodeId`] from the shared counter.
    fn alloc_node_id(&mut self) -> NodeId {
        let id = NodeId::new(*self.next_id).expect("node id allocator overflow");
        *self.next_id += 1;
        id
    }

    /// Import one modifier instance into a modifier node, returning its id.
    ///
    /// The id is to be placed in a stack. Returns `None` only if the type does
    /// not reflect as a struct, in which case a warning is recorded.
    fn import_modifier(&mut self, reflect: &dyn Reflect) -> Option<NodeId> {
        let type_path = reflect.reflect_type_path();
        let Some(info) = reflect.get_represented_type_info() else {
            self.warnings.push(ImportWarning::new(format!(
                "modifier '{type_path}' has no type info; skipped"
            )));
            return None;
        };
        let Some(schema) = modifier_schema(info) else {
            self.warnings.push(ImportWarning::new(format!(
                "modifier '{type_path}' does not reflect as a struct; skipped"
            )));
            return None;
        };

        let node_id = self.alloc_node_id();

        // Configuration fields: read each back into the edit config bag.
        let mut config: BTreeMap<SharedStr, EditValue> = BTreeMap::new();
        for field in schema.config() {
            match read_config_field(reflect, &field.name, &field.role, field.type_path) {
                Ok(value) => {
                    config.insert(field.name.clone(), value);
                }
                Err(message) => self.warnings.push(ImportWarning::new(format!(
                    "modifier '{type_path}' field '{}': {message}",
                    field.name
                ))),
            }
        }

        // Expression ports: recover literal and property inputs; reset the rest.
        let mut inputs: Vec<InputSlot> = Vec::new();
        let mut links: Vec<GraphLink> = Vec::new();
        for field in schema.ports() {
            // A texture port recovers its slot-index literal back to a binding
            // on the recovered slot.
            if matches!(field.role, FieldRole::Texture) {
                let binding = self.recover_image_binding(reflect, &field.name);
                inputs.push(InputSlot {
                    name: field.name.clone(),
                    default: binding.into(),
                });
                continue;
            }
            let optional = matches!(field.role, FieldRole::ExprPort { optional: true });
            let Some(handle) = read_expr_handle(reflect, &field.name, optional) else {
                // Unconnected optional port: leave it with no inline default.
                continue;
            };
            match self.recover_port(node_id, &field.name, handle) {
                PortInput::Inline(value) => inputs.push(InputSlot {
                    name: field.name.clone(),
                    default: value.into(),
                }),
                PortInput::Link(link) => {
                    links.push(link);
                    // A linked port still carries an inline default (unused while
                    // linked) so the model stays valid if the link is later cut.
                    if let Some(default) = handle_value_type_default(self.module, handle) {
                        inputs.push(InputSlot {
                            name: field.name.clone(),
                            default: default.into(),
                        });
                    }
                }
            }
        }

        self.graph.nodes.push(GraphNode {
            id: node_id,
            payload: NodePayload::Modifier(ModifierNodeData::Known {
                type_path: type_path.into(),
                config,
            }),
            inputs,
        });
        self.graph.links.extend(links);

        Some(node_id)
    }

    /// Resolve a modifier's expression input back to an editable form.
    fn recover_port(&mut self, node: NodeId, port: &SharedStr, handle: ExprHandle) -> PortInput {
        match self.module.get(handle) {
            Some(Expr::Literal(lit)) => match literal_value(lit) {
                Some(value) => PortInput::Inline(value),
                None => {
                    self.warnings.push(ImportWarning::new(format!(
                        "port '{port}': could not read literal value; reset to 0"
                    )));
                    PortInput::Inline(Value::from(0.0_f32))
                }
            },
            Some(Expr::Property(pe)) => match self.property_ref(pe) {
                Some(ref_node) => PortInput::Link(GraphLink {
                    from: PortRef {
                        node: ref_node,
                        port: OUTPUT_PORT.into(),
                    },
                    to: PortRef {
                        node,
                        port: port.clone(),
                    },
                }),
                None => {
                    self.warnings.push(ImportWarning::new(format!(
                        "port '{port}': references an unknown property; reset to 0"
                    )));
                    PortInput::Inline(Value::from(0.0_f32))
                }
            },
            other => {
                let kind = other.map(expr_kind).unwrap_or("missing");
                self.warnings.push(ImportWarning::new(format!(
                    "port '{port}': {kind} expression input cannot be reversed; reset to default"
                )));
                let value = handle_value_type_default(self.module, handle)
                    .unwrap_or_else(|| Value::from(0.0_f32));
                PortInput::Inline(value)
            }
        }
    }

    /// Recover an image-port binding from its baked slot-index literal.
    ///
    /// A texture port bakes to a static index naming a slot; map that index
    /// back to the recovered slot's stable id. Falls back to
    /// [`ImageBinding::Unbound`] when the field is missing or not a constant
    /// index (e.g. a runtime-selected slot, unrepresentable in this revision).
    ///
    /// [`ImageBinding::Unbound`]: crate::model::ImageBinding::Unbound
    fn recover_image_binding(&mut self, reflect: &dyn Reflect, field: &SharedStr) -> ImageBinding {
        let index = read_texture_slot_index(reflect, field);
        match index.and_then(|i| self.slot_ids.get(i as usize).copied()) {
            Some(id) => ImageBinding::Slot(id),
            None => {
                self.warnings.push(ImportWarning::new(format!(
                    "texture port '{field}': slot index could not be recovered; left unbound"
                )));
                ImageBinding::Unbound
            }
        }
    }

    /// The reference node for a property expression, creating it on first use.
    fn property_ref(&mut self, pe: &PropertyExpr) -> Option<NodeId> {
        let handle = property_handle(pe)?;
        let name = self.module.get_property(handle)?.name();
        let prop_id = *self.props_by_name.get(name)?;
        if let Some(&existing) = self.prop_ref_nodes.get(&prop_id) {
            return Some(existing);
        }
        let id = self.alloc_node_id();
        self.graph.nodes.push(GraphNode {
            id,
            payload: NodePayload::Expr(ExprNode::Property(prop_id)),
            inputs: Vec::new(),
        });
        self.prop_ref_nodes.insert(prop_id, id);
        Some(id)
    }
}

/// How a recovered modifier input is represented in the edit model.
enum PortInput {
    /// An inline default literal on the modifier node's input slot.
    Inline(Value),
    /// A link from a source node (e.g. a property reference) to the port.
    Link(GraphLink),
}

/// Read one configuration field back into an [`EditValue`].
///
/// The inverse of the bake's `apply_edit_value`. Returns `Err` with a reason
/// for fields that have no faithful edit representation (textures, unmodeled
/// `Raw` types).
fn read_config_field(
    reflect: &dyn Reflect,
    name: &str,
    role: &FieldRole,
    type_path: &str,
) -> Result<EditValue, String> {
    let ReflectRef::Struct(s) = reflect.reflect_ref() else {
        return Err("modifier does not reflect as a struct".to_string());
    };
    let field = s
        .field(name)
        .ok_or_else(|| format!("no such field '{name}'"))?;

    let kind = match role {
        FieldRole::Config(kind) => *kind,
        FieldRole::Texture => {
            return Err("texture bindings cannot be read back from a baked asset".to_string());
        }
        FieldRole::ExprPort { .. } => return Err("expression port is not config".to_string()),
        FieldRole::Hidden => {
            return Err("topology-owned field is not editable config".to_string());
        }
    };

    match kind {
        ConfigKind::Bool => downcast::<bool>(field).map(EditValue::Bool),
        ConfigKind::U32 => downcast::<u32>(field).map(EditValue::U32),
        ConfigKind::UVec2 => downcast::<UVec2>(field).map(EditValue::UVec2),
        ConfigKind::Attribute => downcast::<Attribute>(field).map(EditValue::Attribute),
        ConfigKind::CpuVec3 => downcast::<CpuValue<Vec3>>(field).map(EditValue::CpuVec3),
        ConfigKind::CpuVec4 => downcast::<CpuValue<Vec4>>(field).map(EditValue::CpuVec4),
        ConfigKind::Gradient3 => downcast::<Gradient<Vec3>>(field)
            .map(|g| EditValue::Gradient3(GradientVec3::Analytical(g))),
        ConfigKind::Gradient4 => downcast::<Gradient<Vec4>>(field)
            .map(|g| EditValue::Gradient4(GradientVec4::Analytical(g))),
        ConfigKind::Scalar => read_scalar(field),
        ConfigKind::Enum => read_enum(field, type_path),
        ConfigKind::Flags => read_flags(field, type_path),
        ConfigKind::Raw => Err("unmodeled field type cannot be read back".to_string()),
    }
}

/// Clone a concrete value out of a reflected field, or describe the mismatch.
fn downcast<T: Reflect + Clone>(field: &dyn PartialReflect) -> Result<T, String> {
    field.try_downcast_ref::<T>().cloned().ok_or_else(|| {
        format!(
            "expected {}, found {}",
            std::any::type_name::<T>(),
            field.reflect_type_path()
        )
    })
}

/// Read a scalar/vector field as a `Value`.
///
/// Mirrors the bake's `assign_scalar` supported set.
fn read_scalar(field: &dyn PartialReflect) -> Result<EditValue, String> {
    if let Some(v) = field.try_downcast_ref::<f32>() {
        Ok(EditValue::Scalar(Value::from(*v)))
    } else if let Some(v) = field.try_downcast_ref::<i32>() {
        Ok(EditValue::Scalar(Value::from(*v)))
    } else if let Some(v) = field.try_downcast_ref::<u32>() {
        Ok(EditValue::Scalar(Value::from(*v)))
    } else if let Some(v) = field.try_downcast_ref::<Vec2>() {
        Ok(EditValue::Scalar(Value::from(*v)))
    } else if let Some(v) = field.try_downcast_ref::<Vec3>() {
        Ok(EditValue::Scalar(Value::from(*v)))
    } else if let Some(v) = field.try_downcast_ref::<Vec4>() {
        Ok(EditValue::Scalar(Value::from(*v)))
    } else {
        Err(format!(
            "unsupported scalar field type {}",
            field.reflect_type_path()
        ))
    }
}

/// Read a data-less enum field as its active variant name.
fn read_enum(field: &dyn PartialReflect, type_path: &str) -> Result<EditValue, String> {
    let ReflectRef::Enum(e) = field.reflect_ref() else {
        return Err("expected an enum field".to_string());
    };
    Ok(EditValue::Enum {
        type_path: type_path.into(),
        variant: e.variant_name().into(),
    })
}

/// Read a bitflags newtype (a tuple struct over one integer) as its bits.
fn read_flags(field: &dyn PartialReflect, type_path: &str) -> Result<EditValue, String> {
    let ReflectRef::TupleStruct(ts) = field.reflect_ref() else {
        return Err("flags field is not a tuple struct".to_string());
    };
    let inner = ts.field(0).ok_or("flags newtype has no inner value")?;
    let bits = if let Some(b) = inner.try_downcast_ref::<u8>() {
        *b as u64
    } else if let Some(b) = inner.try_downcast_ref::<u16>() {
        *b as u64
    } else if let Some(b) = inner.try_downcast_ref::<u32>() {
        *b as u64
    } else if let Some(b) = inner.try_downcast_ref::<u64>() {
        *b
    } else {
        return Err(format!(
            "unsupported flags integer type {}",
            inner.reflect_type_path()
        ));
    };
    Ok(EditValue::Flags {
        type_path: type_path.into(),
        bits,
    })
}

/// Read an `ExprHandle` (or `Option<ExprHandle>`) field by name.
fn read_expr_handle(reflect: &dyn Reflect, name: &str, optional: bool) -> Option<ExprHandle> {
    let ReflectRef::Struct(s) = reflect.reflect_ref() else {
        return None;
    };
    let field = s.field(name)?;
    if optional {
        return field
            .try_downcast_ref::<Option<ExprHandle>>()
            .copied()
            .flatten();
    }
    field.try_downcast_ref::<ExprHandle>().copied()
}

/// Read a static texture-slot index field by name.
fn read_texture_slot_index(reflect: &dyn Reflect, name: &str) -> Option<u32> {
    let ReflectRef::Struct(s) = reflect.reflect_ref() else {
        return None;
    };
    s.field(name)?.try_downcast_ref::<u32>().copied()
}

/// The `Value` inside a literal expression, read by reflection.
///
/// The field is private but the type is `Reflect`.
fn literal_value(lit: &LiteralExpr) -> Option<Value> {
    lit.field("value")?.try_downcast_ref::<Value>().copied()
}

/// The `PropertyHandle` inside a property expression, read by reflection.
fn property_handle(pe: &PropertyExpr) -> Option<PropertyHandle> {
    pe.field("property")?
        .try_downcast_ref::<PropertyHandle>()
        .copied()
}

/// A zero-valued [`Value`] matching the type of the expression behind `handle`.
///
/// Used as the inline default for a port whose real input could not be
/// reversed. `None` when the expression's type is not statically known.
fn handle_value_type_default(module: &Module, handle: ExprHandle) -> Option<Value> {
    use bevy_hanabi::{ScalarType, ValueType};
    Some(match module.get(handle)?.value_type()? {
        ValueType::Scalar(ScalarType::Float) => Value::from(0.0_f32),
        ValueType::Scalar(ScalarType::Int) => Value::from(0_i32),
        ValueType::Scalar(ScalarType::Uint) => Value::from(0_u32),
        ValueType::Scalar(ScalarType::Bool) => Value::from(false),
        ValueType::Vector(v) => match (v.elem_type(), v.count()) {
            (ScalarType::Float, 2) => Value::from(Vec2::ZERO),
            (ScalarType::Float, 3) => Value::from(Vec3::ZERO),
            (ScalarType::Float, 4) => Value::from(Vec4::ZERO),
            (ScalarType::Uint, 2) => Value::from(UVec2::ZERO),
            _ => return None,
        },
        _ => return None,
    })
}

fn expr_kind(expr: &Expr) -> &'static str {
    match expr {
        Expr::BuiltIn(_) => "built-in",
        Expr::Literal(_) => "literal",
        Expr::Property(_) => "property",
        Expr::Attribute(_) => "attribute",
        Expr::ParentAttribute(_) => "parent-attribute",
        Expr::Unary { .. } => "unary-operator",
        Expr::Binary { .. } => "binary-operator",
        Expr::Ternary { .. } => "ternary-operator",
        Expr::Cast(_) => "cast",
        Expr::TextureSample(_) => "texture-sample",
        Expr::TextureLoad(_) => "texture-load",
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use bevy::prelude::*;
    use bevy_hanabi::{
        ParticleTextureModifier, SlotDimension, SpawnerSettings, graph::expr::TextureLoadExpr,
    };

    use super::*;
    use crate::{
        bake::{bake, bake_emitter},
        demo::demo_emitter,
        modifier_registry::ModifierRegistryPlugin,
    };

    /// Bake-then-import the demo graph recovers its cleanly reversible parts.
    ///
    /// Recovers the emitter settings, exposed properties, per-stage modifier
    /// shape, and the two property-reference wirings.
    #[test]
    fn import_round_trips_demo_bake() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()));
        app.add_plugins(ModifierRegistryPlugin);

        let registry = app.world().resource::<AppTypeRegistry>().read();
        let asset = bake_emitter(
            &demo_emitter(),
            &registry,
            SpawnerSettings::rate(120.0.into()),
            &HashMap::new(),
        )
        .expect("demo bakes");
        drop(registry);

        let (effect_graph, _warnings) = import_effect(&asset);
        assert_eq!(effect_graph.emitters.len(), 1, "one imported emitter");
        assert_eq!(effect_graph.sources.len(), 1, "one CPU spawn source");
        let graph = &effect_graph.emitters[0];

        assert_eq!(&*graph.name, "demo");
        assert_eq!(graph.capacity, 8192);

        // Both exposed properties come back, exposed.
        let names: Vec<&str> = graph.properties.iter().map(|p| &*p.name).collect();
        assert!(names.contains(&"gravity"), "gravity property imported");
        assert!(
            names.contains(&"spawn_speed"),
            "spawn_speed property imported"
        );
        assert!(graph.properties.iter().all(|p| p.exposed));

        // One stack per phase, with the same modifier counts as the bake.
        let count = |g: ModifierGroup| {
            graph
                .stacks
                .iter()
                .find(|s| s.group == g)
                .map(|s| s.members.len())
                .unwrap_or(0)
        };
        assert_eq!(count(ModifierGroup::Init), 3);
        assert_eq!(count(ModifierGroup::Update), 1);
        assert_eq!(count(ModifierGroup::Render), 6);

        // The two property references (accel←gravity, velocity speed←spawn_speed)
        // become reference nodes wired by links.
        let prop_ref_nodes = graph
            .nodes
            .iter()
            .filter(|n| matches!(&n.payload, NodePayload::Expr(ExprNode::Property(_))))
            .count();
        assert_eq!(prop_ref_nodes, 2, "two property reference nodes");
        assert_eq!(graph.links.len(), 2, "two property links");
    }

    /// The imported demo graph must itself bake cleanly through the strict
    /// single-emitter [`bake`] convenience.
    ///
    /// No dangling references or invalid stacks introduced by import.
    #[test]
    fn imported_graph_rebakes() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()));
        app.add_plugins(ModifierRegistryPlugin);

        let registry = app.world().resource::<AppTypeRegistry>().read();
        let asset = bake_emitter(
            &demo_emitter(),
            &registry,
            SpawnerSettings::rate(120.0.into()),
            &HashMap::new(),
        )
        .expect("demo bakes");
        let (effect_graph, _) = import_effect(&asset);
        let rebaked = bake(&effect_graph, &registry).expect("imported graph rebakes");

        assert_eq!(rebaked.init_modifiers().count(), 3);
        assert_eq!(rebaked.update_modifiers().count(), 1);
        assert_eq!(rebaked.render_modifiers().count(), 6);
    }

    #[test]
    fn identifies_unreversible_texture_load_expressions() {
        let mut module = Module::default();
        module.add_texture_slot("pixels", SlotDimension::D2);
        let coordinates = module.lit(UVec2::ZERO);
        let mip_level = module.lit(0u32);
        let load = TextureLoadExpr::new(0, SlotDimension::D2, coordinates, None, Some(mip_level))
            .expect("valid 2D texture load");

        assert_eq!(expr_kind(&Expr::TextureLoad(load)), "texture-load");
    }

    #[test]
    fn imports_static_particle_texture_slot() {
        let mut module = Module::default();
        module.add_texture_slot("albedo", SlotDimension::D2);
        let asset = EffectAsset::new(32, SpawnerSettings::default(), module)
            .render(ParticleTextureModifier::new(0));

        let (graph, warnings) = import_emitter(&asset);
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
        assert_eq!(graph.texture_slots.len(), 1);
        assert_eq!(graph.texture_slots[0].name.as_ref(), "albedo");

        let modifier = graph
            .nodes
            .iter()
            .find(|node| {
                matches!(
                    &node.payload,
                    NodePayload::Modifier(ModifierNodeData::Known { type_path, .. })
                        if type_path.as_ref() == ParticleTextureModifier::type_path()
                )
            })
            .expect("particle texture modifier");
        assert_eq!(
            modifier.inputs,
            vec![InputSlot {
                name: "texture_slot".into(),
                default: ImageBinding::Slot(graph.texture_slots[0].id).into(),
            }]
        );

        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()));
        app.add_plugins(ModifierRegistryPlugin);
        let registry = app.world().resource::<AppTypeRegistry>().read();
        let rebaked = bake_emitter(
            &graph,
            &registry,
            SpawnerSettings::default(),
            &HashMap::new(),
        )
        .expect("imported texture modifier rebakes");
        assert_eq!(
            rebaked.module().texture_layout().layout[0].dimension,
            SlotDimension::D2
        );
        assert_eq!(
            rebaked
                .render_modifiers()
                .next()
                .expect("particle texture modifier")
                .as_modifier()
                .as_reflect()
                .downcast_ref::<ParticleTextureModifier>()
                .expect("particle texture modifier")
                .texture_slot,
            0
        );
    }

    #[test]
    fn imports_texture_slot_dimensions() {
        let mut module = Module::default();
        module.add_texture_slot("curve", SlotDimension::D1);
        module.add_texture_slot("volume", SlotDimension::D3);
        let asset = EffectAsset::new(32, SpawnerSettings::default(), module);

        let (graph, warnings) = import_emitter(&asset);
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
        assert_eq!(
            graph
                .texture_slots
                .iter()
                .map(|slot| slot.dimension)
                .collect::<Vec<_>>(),
            vec![SlotDimension::D1, SlotDimension::D3]
        );
    }
}
