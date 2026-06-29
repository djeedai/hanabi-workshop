//! A native [`EffectGraph`] equivalent of the programmatic demo effect.
//!
//! This is the graph-native seed for new/startup documents: it is authored
//! directly as an [`EffectGraph`] (the canonical edit model) and [`bake`]s into
//! a renderable [`EffectAsset`]. It exercises the breadth of the model —
//! exposed scalar and vector properties, an operator sub-graph, and modifiers
//! carrying enum / integral / `CpuValue` / gradient config.
//!
//! [`bake`]: crate::bake::bake
//! [`EffectAsset`]: bevy_hanabi::EffectAsset

use std::collections::BTreeMap;

use bevy::{
    math::{UVec2, Vec3, Vec4},
    reflect::TypePath,
};
use bevy_hanabi::{
    AccelModifier, Attribute, ColorBlendMask, ColorBlendMode, ColorOverLifetimeModifier, CpuValue,
    FlipbookModifier, Gradient, OrientMode, OrientModifier, SetAttributeModifier, SetColorModifier,
    SetPositionSphereModifier, SetSizeModifier, SetVelocitySphereModifier, ShapeDimension,
    SimulationCondition, SimulationSpace, SizeOverLifetimeModifier, SpawnerSettings, Value,
    graph::expr::BinaryOperator,
};

use super::{
    model::{
        EditValue, EffectGraph, EffectHeader, ExprNode, GradientVec3, GradientVec4, GraphLink,
        GraphNode, GraphStack, InputSlot, ModifierNodeData, NodeId, NodePayload, PortRef,
        PropertyDef, PropertyId,
    },
    schema::OUTPUT_PORT,
};
use crate::ModifierGroup;

/// Build the demo effect as a native [`EffectGraph`].
pub fn demo_graph() -> EffectGraph {
    let mut g = EffectGraph {
        header: EffectHeader {
            name: "demo".into(),
            capacity: 8192,
            spawner: SpawnerSettings::rate(120.0.into()),
            simulation_space: SimulationSpace::default(),
            simulation_condition: SimulationCondition::default(),
            z_layer_2d: 0.0,
        },
        ..EffectGraph::empty()
    };

    // Exposed properties: a Vec3 gravity and a scalar spawn speed, wired into
    // the modifiers that consume them via dedicated property reference nodes.
    let gravity = add_property(&mut g, "gravity", Vec3::new(0.0, -2.0, 0.0).into(), true);
    let spawn_speed = add_property(&mut g, "spawn_speed", 2.0_f32.into(), true);

    // Operator sub-graph: radius = 0.4 * 1.25, a Binary node feeding a modifier
    // input. Its operands are inline defaults (no incoming links).
    let radius = add_node(
        &mut g,
        NodePayload::Expr(ExprNode::Binary(BinaryOperator::Mul)),
        vec![slot("lhs", 0.4_f32.into()), slot("rhs", 1.25_f32.into())],
    );
    let gravity_ref = add_node(
        &mut g,
        NodePayload::Expr(ExprNode::Property(gravity)),
        vec![],
    );
    let speed_ref = add_node(
        &mut g,
        NodePayload::Expr(ExprNode::Property(spawn_speed)),
        vec![],
    );

    // Init stack.
    let pos = add_node(
        &mut g,
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
        modifier::<SetVelocitySphereModifier>(BTreeMap::new()),
        vec![
            slot("center", Vec3::ZERO.into()),
            slot("speed", 1.0_f32.into()),
        ],
    );
    let lifetime = add_node(
        &mut g,
        modifier::<SetAttributeModifier>(config([(
            "attribute",
            EditValue::Attribute(Attribute::LIFETIME),
        )])),
        vec![slot("value", 2.5_f32.into())],
    );

    // Update stack.
    let accel = add_node(
        &mut g,
        modifier::<AccelModifier>(BTreeMap::new()),
        vec![slot("accel", Vec3::ZERO.into())],
    );

    // Render stack.
    let orient = add_node(
        &mut g,
        modifier::<OrientModifier>(config([(
            "mode",
            enum_value::<OrientMode>("FaceCameraPosition"),
        )])),
        vec![],
    );
    let flipbook = add_node(
        &mut g,
        modifier::<FlipbookModifier>(config([(
            "sprite_grid_size",
            EditValue::UVec2(UVec2::new(4, 4)),
        )])),
        vec![],
    );
    let size = add_node(
        &mut g,
        modifier::<SetSizeModifier>(config([(
            "size",
            EditValue::CpuVec3(CpuValue::Single(Vec3::splat(0.1))),
        )])),
        vec![],
    );
    let color = add_node(
        &mut g,
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
        stack(&mut g, ModifierGroup::Init, vec![pos, vel, lifetime]),
        stack(&mut g, ModifierGroup::Update, vec![accel]),
        stack(
            &mut g,
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

fn add_property(g: &mut EffectGraph, name: &str, default: Value, exposed: bool) -> PropertyId {
    let id = g.alloc_property_id();
    g.properties.push(PropertyDef {
        id,
        name: name.into(),
        default,
        exposed,
    });
    id
}

fn add_node(g: &mut EffectGraph, payload: NodePayload, inputs: Vec<InputSlot>) -> NodeId {
    let id = g.alloc_node_id();
    g.nodes.push(GraphNode {
        id,
        payload,
        inputs,
    });
    id
}

fn stack(g: &mut EffectGraph, group: ModifierGroup, members: Vec<NodeId>) -> GraphStack {
    GraphStack {
        id: g.alloc_stack_id(),
        group,
        members,
    }
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
    /// With the same per-stage modifier shape as the legacy `demo_effect()`.
    #[test]
    fn demo_graph_bakes() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()));
        app.add_plugins(ModifierRegistryPlugin);

        let registry = app.world().resource::<AppTypeRegistry>().read();
        let asset = match crate::bake::bake(&demo_graph(), &registry) {
            Ok(asset) => asset,
            Err(errors) => panic!("demo graph failed to bake: {errors:?}"),
        };

        assert_eq!(asset.name, "demo");
        assert_eq!(asset.capacity(), 8192);
        assert_eq!(asset.init_modifiers().count(), 3);
        assert_eq!(asset.update_modifiers().count(), 1);
        assert_eq!(asset.render_modifiers().count(), 5);
    }

    /// The `.hnb` save/load path the editor uses round-trips and bakes.
    ///
    /// Serialize the demo graph as an `EffectGraphAsset`, round-trip it through
    /// the format helpers, and bake the reloaded graph to the same effect.
    #[test]
    fn demo_graph_round_trips_through_hnb_and_bakes() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()));
        app.add_plugins(ModifierRegistryPlugin);

        let saved = crate::model::EffectGraphAsset {
            version: crate::model::FORMAT_VERSION,
            graph: demo_graph(),
            layout: None,
        };
        let text = crate::to_ron_string(&saved).expect("serialize .hnb");
        let loaded = crate::from_ron_bytes(text.as_bytes()).expect("deserialize .hnb");
        assert_eq!(loaded.graph, saved.graph);

        let registry = app.world().resource::<AppTypeRegistry>().read();
        let asset = match crate::bake::bake(&loaded.graph, &registry) {
            Ok(asset) => asset,
            Err(errors) => panic!("reloaded graph failed to bake: {errors:?}"),
        };
        assert_eq!(asset.init_modifiers().count(), 3);
        assert_eq!(asset.update_modifiers().count(), 1);
        assert_eq!(asset.render_modifiers().count(), 5);
    }
}
