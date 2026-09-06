//! A native [`EffectGraph`] demo with parent and child emitters.
//!
//! This is the graph-native seed for new/startup documents: it is authored
//! directly as a graph (the canonical edit model) and [`bake`]s into a
//! renderable [`EffectAsset`]. It exercises the breadth of the model —
//! exposed scalar and vector properties, an operator sub-graph, and modifiers
//! carrying enum / integral / `CpuValue` / gradient config.
//!
//! [`demo_emitter`] builds one [`EmitterGraph`] in isolation for focused tests.
//! [`demo_effect`] builds the complete authored effect: a CPU-rooted parent
//! emitter and a GPU-driven child fed by two `EmitSpawnEventModifier` nodes.
//!
//! [`bake`]: crate::bake::bake_emitter
//! [`EffectAsset`]: bevy_hanabi::EffectAsset

use std::collections::BTreeMap;

use bevy::{
    math::{UVec2, Vec3, Vec4},
    reflect::TypePath,
};
use bevy_hanabi::{
    AccelModifier, Attribute, ColorBlendMask, ColorBlendMode, ColorOverLifetimeModifier, CpuValue,
    EmitSpawnEventModifier, EventEmitCondition, FlipbookModifier, Gradient,
    InheritAttributeModifier, OrientMode, OrientModifier, SetAttributeModifier, SetColorModifier,
    SetPositionSphereModifier, SetSizeModifier, SetVelocitySphereModifier, ShapeDimension,
    SizeOverLifetimeModifier, SpawnerSettings, Value, graph::expr::BinaryOperator,
};

use super::{
    model::{
        EditValue, EffectGraph, EmitterGraph, EmitterId, ExprNode, GradientVec3, GradientVec4,
        GraphLink, GraphNode, GraphStack, InputSlot, ModifierNodeData, NodeId, NodePayload,
        PortRef, PropertyDef, PropertyId, SourceKind,
    },
    schema::OUTPUT_PORT,
};
use crate::ModifierGroup;

/// Build the demo emitter as a standalone [`EmitterGraph`].
///
/// This helper is intentionally unconnected to a spawn source. Prefer
/// [`demo_effect`] for anything that needs a complete, loadable effect graph;
/// an isolated `EmitterGraph` must be given [`SpawnerSettings`] explicitly
/// when passed to [`crate::bake::bake_emitter`].
///
/// [`FORMAT_VERSION`]: crate::model::FORMAT_VERSION
pub fn demo_emitter() -> EmitterGraph {
    let id = EmitterId::new(1).expect("1 is a valid NonZeroU32");
    let mut next_id = id.get() + 1;
    build_demo_emitter(id, &mut next_id)
}

/// Build the complete demo effect graph.
///
/// The parent is [`demo_emitter`]'s emitter, extended with two
/// `EmitSpawnEventModifier` nodes in its Update stack that both feed the same
/// GPU source context (exercising multi-emitter fan-in into one channel,
/// still within the currently supported single-child-per-parent topology —
/// see `validation::check_single_child_restriction`). The child is a small
/// GPU-driven emitter whose Init stack uses `InheritAttributeModifier` to copy
/// the parent's `POSITION` and `COLOR` attributes, exercising parent-particle
/// reads.
pub fn demo_effect() -> EffectGraph {
    let mut effect_graph = EffectGraph::empty();

    let parent_id = effect_graph.alloc_emitter_id();
    let mut parent = build_demo_emitter(parent_id, &mut effect_graph.next_id);

    // Two spawn-event nodes feed the same GPU source: both fire every frame,
    // each spawning its own share of the child's particles.
    let emit_a = add_node(
        &mut parent,
        &mut effect_graph.next_id,
        modifier::<EmitSpawnEventModifier>(config([(
            "condition",
            enum_value::<EventEmitCondition>("Always"),
        )])),
        vec![slot("count", 1u32.into())],
    );
    let emit_b = add_node(
        &mut parent,
        &mut effect_graph.next_id,
        modifier::<EmitSpawnEventModifier>(config([(
            "condition",
            enum_value::<EventEmitCondition>("Always"),
        )])),
        vec![slot("count", 2u32.into())],
    );
    let update_stack = parent
        .stacks
        .iter_mut()
        .find(|s| s.group == ModifierGroup::Update)
        .expect("build_demo_emitter always has an Update stack");
    update_stack.members.push(emit_a);
    update_stack.members.push(emit_b);

    let child_id = EmitterId::new(effect_graph.next_id).expect("id allocator overflow");
    effect_graph.next_id += 1;
    let mut child = EmitterGraph {
        name: "demo_child".into(),
        ..EmitterGraph::empty(child_id)
    };
    let inherit_position = add_node(
        &mut child,
        &mut effect_graph.next_id,
        modifier::<InheritAttributeModifier>(config([(
            "attribute",
            EditValue::Attribute(Attribute::POSITION),
        )])),
        vec![],
    );
    let inherit_color = add_node(
        &mut child,
        &mut effect_graph.next_id,
        modifier::<InheritAttributeModifier>(config([(
            "attribute",
            EditValue::Attribute(Attribute::COLOR),
        )])),
        vec![],
    );
    child.stacks.push(GraphStack {
        id: {
            let id =
                super::model::StackId::new(effect_graph.next_id).expect("id allocator overflow");
            effect_graph.next_id += 1;
            id
        },
        group: ModifierGroup::Init,
        members: vec![inherit_position, inherit_color],
    });

    let cpu_source_id = effect_graph.alloc_source_id();
    let gpu_source_id = effect_graph.alloc_source_id();

    effect_graph.sources.push(super::model::SourceContext {
        id: cpu_source_id,
        kind: SourceKind::CpuSpawner {
            settings: SpawnerSettings::rate(120.0.into()),
        },
    });
    effect_graph.sources.push(super::model::SourceContext {
        id: gpu_source_id,
        kind: SourceKind::GpuEvent,
    });

    effect_graph.source_links.push(super::model::SourceLink {
        source: cpu_source_id,
        emitter: parent_id,
    });
    effect_graph.source_links.push(super::model::SourceLink {
        source: gpu_source_id,
        emitter: child_id,
    });
    effect_graph.event_links.push(super::model::EventLink {
        node: emit_a,
        target: gpu_source_id,
    });
    effect_graph.event_links.push(super::model::EventLink {
        node: emit_b,
        target: gpu_source_id,
    });

    effect_graph.emitters.push(parent);
    effect_graph.emitters.push(child);

    effect_graph
}

/// Build the emitter shared by [`demo_emitter`] and [`demo_effect`].
///
/// `next_id` is threaded in rather than derived from `id` so every id minted
/// while authoring this emitter advances the *caller's* counter — critical for
/// `demo_effect`, which must keep allocating from the same shared
/// [`EffectGraph::next_id`] afterwards (for the emitters, child emitter, and
/// sources) without colliding with ids already used inside this emitter.
pub(crate) fn build_demo_emitter(id: EmitterId, next_id: &mut u32) -> EmitterGraph {
    let mut g = EmitterGraph {
        name: "demo".into(),
        capacity: 8192,
        ..EmitterGraph::empty(id)
    };

    // Exposed properties: a Vec3 gravity and a scalar spawn speed, wired into
    // the modifiers that consume them via dedicated property reference nodes.
    let gravity = add_property(
        &mut g,
        next_id,
        "gravity",
        Vec3::new(0.0, -2.0, 0.0).into(),
        true,
    );
    let spawn_speed = add_property(&mut g, next_id, "spawn_speed", 2.0_f32.into(), true);

    // Operator sub-graph: radius = 0.4 * 1.25, a Binary node feeding a modifier
    // input. Its operands are inline defaults (no incoming links).
    let radius = add_node(
        &mut g,
        next_id,
        NodePayload::Expr(ExprNode::Binary(BinaryOperator::Mul)),
        vec![slot("lhs", 0.4_f32.into()), slot("rhs", 1.25_f32.into())],
    );
    let gravity_ref = add_node(
        &mut g,
        next_id,
        NodePayload::Expr(ExprNode::Property(gravity)),
        vec![],
    );
    let speed_ref = add_node(
        &mut g,
        next_id,
        NodePayload::Expr(ExprNode::Property(spawn_speed)),
        vec![],
    );

    // Init stack.
    let pos = add_node(
        &mut g,
        next_id,
        modifier::<SetPositionSphereModifier>(config([(
            "dimension",
            enum_value::<ShapeDimension>("Surface"),
        )])),
        vec![
            slot("center", Vec3::ZERO.into()),
            slot("radius", 1.0_f32.into()),
        ],
    );
    let vel = add_node(
        &mut g,
        next_id,
        modifier::<SetVelocitySphereModifier>(BTreeMap::new()),
        vec![
            slot("center", Vec3::ZERO.into()),
            slot("speed", 1.0_f32.into()),
        ],
    );
    let lifetime = add_node(
        &mut g,
        next_id,
        modifier::<SetAttributeModifier>(config([(
            "attribute",
            EditValue::Attribute(Attribute::LIFETIME),
        )])),
        vec![slot("value", 2.5_f32.into())],
    );

    // Update stack.
    let accel = add_node(
        &mut g,
        next_id,
        modifier::<AccelModifier>(BTreeMap::new()),
        vec![slot("accel", Vec3::ZERO.into())],
    );

    // Render stack.
    let orient = add_node(
        &mut g,
        next_id,
        modifier::<OrientModifier>(config([(
            "mode",
            enum_value::<OrientMode>("FaceCameraPosition"),
        )])),
        vec![],
    );
    let flipbook = add_node(
        &mut g,
        next_id,
        modifier::<FlipbookModifier>(config([(
            "sprite_grid_size",
            EditValue::UVec2(UVec2::new(4, 4)),
        )])),
        vec![],
    );
    let size = add_node(
        &mut g,
        next_id,
        modifier::<SetSizeModifier>(config([(
            "size",
            EditValue::CpuVec3(CpuValue::Single(Vec3::splat(0.1))),
        )])),
        vec![],
    );
    let color = add_node(
        &mut g,
        next_id,
        modifier::<SetColorModifier>(config([
            (
                "color",
                EditValue::CpuVec4(CpuValue::Uniform((
                    Vec4::new(1.0, 0.5, 0.1, 1.0),
                    Vec4::new(1.0, 0.9, 0.3, 1.0),
                ))),
            ),
            ("blend", enum_value::<ColorBlendMode>("Overwrite")),
            (
                "mask",
                flags_value::<ColorBlendMask>(ColorBlendMask::RGBA.bits() as u64),
            ),
        ])),
        vec![],
    );
    let color_over_lifetime = add_node(
        &mut g,
        next_id,
        modifier::<ColorOverLifetimeModifier>(config([
            (
                "gradient",
                EditValue::Gradient4(GradientVec4::Analytical(color_gradient())),
            ),
            ("blend", enum_value::<ColorBlendMode>("Overwrite")),
            (
                "mask",
                flags_value::<ColorBlendMask>(ColorBlendMask::RGBA.bits() as u64),
            ),
        ])),
        vec![],
    );

    let size_over_lifetime = add_node(
        &mut g,
        next_id,
        modifier::<SizeOverLifetimeModifier>(config([
            (
                "gradient",
                EditValue::Gradient3(GradientVec3::Analytical(size_gradient())),
            ),
            ("screen_space_size", EditValue::Bool(false)),
        ])),
        vec![],
    );

    g.links = vec![
        link(radius, pos, "radius"),
        link(speed_ref, vel, "speed"),
        link(gravity_ref, accel, "accel"),
    ];

    g.stacks = vec![
        stack(next_id, ModifierGroup::Init, vec![pos, vel, lifetime]),
        stack(next_id, ModifierGroup::Update, vec![accel]),
        stack(
            next_id,
            ModifierGroup::Render,
            vec![
                orient,
                flipbook,
                size,
                color,
                size_over_lifetime,
                color_over_lifetime,
            ],
        ),
    ];

    g
}

fn size_gradient() -> Gradient<Vec3> {
    let mut gradient = Gradient::new();
    gradient.add_key(0.0, Vec3::splat(0.05));
    gradient.add_key(0.5, Vec3::splat(0.2));
    gradient.add_key(1.0, Vec3::splat(0.0));
    gradient
}

fn color_gradient() -> Gradient<Vec4> {
    let mut gradient = Gradient::new();
    gradient.add_key(0.0, Vec4::new(1.0, 0.5, 0.1, 1.0));
    gradient.add_key(0.5, Vec4::new(1.0, 0.2, 0.05, 0.7));
    gradient.add_key(1.0, Vec4::ZERO);
    gradient
}

/// Mint the next id from a caller-supplied counter, mirroring [`EffectGraph`]'s
/// allocator without requiring a whole document just to author one
/// [`EmitterGraph`].
///
/// [`demo_emitter`] seeds a local counter from its single [`EmitterId`];
/// `demo_effect` threads its actual [`EffectGraph::next_id`] counter through
/// instead — via `build_demo_emitter`'s own `next_id` parameter — so every id
/// minted while authoring an emitter for the document stays unique across the
/// whole canvas.
fn add_property(
    g: &mut EmitterGraph,
    next_id: &mut u32,
    name: &str,
    default: Value,
    exposed: bool,
) -> PropertyId {
    let id = PropertyId::new(*next_id).expect("id allocator overflow");
    *next_id += 1;
    g.properties.push(PropertyDef {
        id,
        name: name.into(),
        default,
        exposed,
    });
    id
}

fn add_node(
    g: &mut EmitterGraph,
    next_id: &mut u32,
    payload: NodePayload,
    inputs: Vec<InputSlot>,
) -> NodeId {
    let id = NodeId::new(*next_id).expect("id allocator overflow");
    *next_id += 1;
    g.nodes.push(GraphNode {
        id,
        payload,
        inputs,
    });
    id
}

fn stack(next_id: &mut u32, group: ModifierGroup, members: Vec<NodeId>) -> GraphStack {
    let id = super::model::StackId::new(*next_id).expect("id allocator overflow");
    *next_id += 1;
    GraphStack { id, group, members }
}

fn slot(name: &str, default: Value) -> InputSlot {
    InputSlot {
        name: name.into(),
        default: default.into(),
    }
}

fn link(from: NodeId, to: NodeId, port: &str) -> GraphLink {
    GraphLink {
        from: PortRef {
            node: from,
            port: OUTPUT_PORT.into(),
        },
        to: PortRef {
            node: to,
            port: port.into(),
        },
    }
}

fn modifier<T: TypePath>(config: BTreeMap<super::model::SharedStr, EditValue>) -> NodePayload {
    NodePayload::Modifier(ModifierNodeData::Known {
        type_path: T::type_path().into(),
        config,
    })
}

fn config<const N: usize>(
    entries: [(&str, EditValue); N],
) -> BTreeMap<super::model::SharedStr, EditValue> {
    entries.into_iter().map(|(k, v)| (k.into(), v)).collect()
}

fn enum_value<T: TypePath>(variant: &str) -> EditValue {
    EditValue::Enum {
        type_path: T::type_path().into(),
        variant: variant.into(),
    }
}

fn flags_value<T: TypePath>(bits: u64) -> EditValue {
    EditValue::Flags {
        type_path: T::type_path().into(),
        bits,
    }
}

#[cfg(test)]
mod tests {
    use bevy::prelude::*;

    use super::*;
    use crate::modifier_registry::ModifierRegistryPlugin;

    /// The demo graph must bake cleanly into an `EffectAsset`.
    ///
    /// With the same per-stage modifier shape as the legacy
    /// `build_demo_emitter()`.
    #[test]
    fn demo_emitter_bakes() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()));
        app.add_plugins(ModifierRegistryPlugin);

        let registry = app.world().resource::<AppTypeRegistry>().read();
        let asset = match crate::bake::bake_emitter(
            &demo_emitter(),
            &registry,
            SpawnerSettings::rate(120.0.into()),
            &std::collections::HashMap::new(),
        ) {
            Ok(asset) => asset,
            Err(errors) => panic!("demo graph failed to bake: {errors:?}"),
        };

        assert_eq!(asset.name, "demo");
        assert_eq!(asset.capacity(), 8192);
        assert_eq!(asset.init_modifiers().count(), 3);
        assert_eq!(asset.update_modifiers().count(), 1);
        assert_eq!(asset.render_modifiers().count(), 6);
    }

    /// The `.hnb` save/load path the editor uses round-trips and bakes.
    ///
    /// Serialize the demo document as a [`EffectGraphAsset`], round-trip it
    /// through the format helpers, and bake the reloaded document to the same
    /// pair of emitters.
    #[test]
    fn demo_effect_round_trips_through_hnb_and_bakes() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()));
        app.add_plugins(ModifierRegistryPlugin);

        let saved = crate::model::EffectGraphAsset {
            version: crate::model::FORMAT_VERSION,
            graph: demo_effect(),
            layout: None,
        };
        let text = crate::to_ron_string(&saved).expect("serialize .hnb");
        let loaded = crate::from_ron_bytes(text.as_bytes()).expect("deserialize .hnb");
        assert_eq!(loaded.graph, saved.graph);

        let registry = app.world().resource::<AppTypeRegistry>().read();
        let baked = match crate::bake::bake_effect(&loaded.graph, &registry) {
            Ok(baked) => baked,
            Err(errors) => panic!("reloaded document failed to bake: {errors:?}"),
        };
        assert_eq!(baked.emitters.len(), 2);
        let parent = &baked.emitters[0];
        assert!(parent.parent.is_none());
        assert_eq!(parent.asset.init_modifiers().count(), 3);
        // Base `accel` plus the two `EmitSpawnEventModifier` emitters.
        assert_eq!(parent.asset.update_modifiers().count(), 3);
        assert_eq!(parent.asset.render_modifiers().count(), 6);
        let child = &baked.emitters[1];
        assert_eq!(child.parent, Some(parent.emitter));
        assert_eq!(child.asset.init_modifiers().count(), 2);
    }

    /// `demo_effect`'s effect topology must itself be valid.
    #[test]
    fn demo_effect_topology_is_valid() {
        let errors = crate::validation::validate_topology(&demo_effect());
        assert!(errors.is_empty(), "unexpected topology errors: {errors:?}");
    }

    /// Every id minted while authoring `demo_effect` must come from the
    /// document's actual shared allocator: unique across every id kind and
    /// every emitter, with `next_id` left strictly above all of them.
    ///
    /// Regression test: `build_demo_emitter` used to mint its ids from a
    /// counter local to itself, disjoint from `effect_graph.next_id`, so
    /// the emitters/child emitter `demo_effect` allocated afterwards
    /// collided with ids the parent emitter had already used, and `next_id`
    /// was left far below the highest id actually in use.
    #[test]
    fn demo_effect_ids_are_unique_and_next_id_advances_past_them() {
        let effect_graph = demo_effect();

        let mut ids: Vec<u32> = Vec::new();
        for emitter in &effect_graph.emitters {
            ids.push(emitter.id.get());
            ids.extend(emitter.properties.iter().map(|p| p.id.get()));
            ids.extend(emitter.texture_slots.iter().map(|s| s.id.get()));
            ids.extend(emitter.nodes.iter().map(|n| n.id.get()));
            ids.extend(emitter.stacks.iter().map(|s| s.id.get()));
        }
        ids.extend(effect_graph.sources.iter().map(|s| s.id.get()));

        let unique: std::collections::HashSet<u32> = ids.iter().copied().collect();
        assert_eq!(
            unique.len(),
            ids.len(),
            "duplicate id found across demo_effect's emitters/sources: {ids:?}"
        );

        let max_id = ids.into_iter().max().expect("demo_effect mints some ids");
        assert!(
            effect_graph.next_id > max_id,
            "next_id ({}) must be strictly greater than every id in use (max {max_id})",
            effect_graph.next_id
        );
    }
}
