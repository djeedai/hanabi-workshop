//! Generates and reconciles retained viewport gizmos for modifier nodes.
//!
//! Modifier types opt in through editor-owned registry metadata, while the
//! generated geometry is kept isolated on each document's render layer.

use std::{
    any::TypeId,
    collections::{HashMap, HashSet},
};

use bevy::{
    camera::visibility::RenderLayers,
    gizmos::{GizmoAsset, config::GizmoLineConfig, retained::Gizmo},
    prelude::*,
    reflect::{TypePath, TypeRegistry},
};
use bevy_egui::EguiPrimaryContextPass;
use bevy_hanabi::{
    AccelModifier, ConformToSphereModifier, KillAabbModifier, KillSphereModifier, Modifier,
    RadialAccelModifier, ScalarValue, SetPositionCircleModifier, SetPositionCone3dModifier,
    SetPositionSphereModifier, SetVelocityCircleModifier, SetVelocitySphereModifier,
    SetVelocityTangentModifier, TangentAccelModifier, Value, ValueType, VectorType,
};
#[cfg(test)]
use hanabi_effect_graph::model::EmitterId;
use hanabi_effect_graph::{
    bake::LiteralSite,
    model::{EditValue, EmitterGraph, ExprNode, GraphNode, ModifierNodeData, NodePayload},
};

use crate::{
    document::{DocumentContent, DocumentUi},
    ui::draw_editor_ui,
};

const LINE_WIDTH: f32 = 2.0;
const DEPTH_BIAS: f32 = -0.002;
const SEGMENTS: usize = 32;
const EPSILON: f32 = 1.0e-5;

/// Callback used to generate a modifier's retained viewport geometry.
///
/// The callback returns whether it appended a complete, usable visualization.
pub type ModifierGizmoFn = for<'a> fn(&ModifierGizmoContext<'a>, &mut GizmoAsset) -> bool;

/// Editor type data that generates a modifier's viewport gizmo.
///
/// Third-party modifier plugins can attach this data to their reflected type
/// registration after registering the modifier with Hanabi.
#[derive(Clone, Copy)]
pub struct ModifierGizmoProvider {
    /// Function that appends this modifier's visualization.
    pub draw: ModifierGizmoFn,
}

/// Read-only author-time inputs for one modifier gizmo provider.
///
/// Resolution intentionally stops at inline values, direct literals, and direct
/// property references. Runtime or computed expressions return `None`.
pub struct ModifierGizmoContext<'a> {
    graph: &'a EmitterGraph,
    node: &'a GraphNode,
    live_values: Option<&'a HashMap<LiteralSite, Value>>,
}

impl<'a> ModifierGizmoContext<'a> {
    #[cfg(test)]
    pub(crate) fn new(graph: &'a EmitterGraph, node: &'a GraphNode) -> Self {
        Self {
            graph,
            node,
            live_values: None,
        }
    }

    fn with_live_values(
        graph: &'a EmitterGraph,
        node: &'a GraphNode,
        live_values: &'a HashMap<LiteralSite, Value>,
    ) -> Self {
        Self {
            graph,
            node,
            live_values: Some(live_values),
        }
    }

    /// The canonical emitter graph containing the modifier.
    ///
    /// Providers may inspect related authoring data without mutating it.
    pub fn graph(&self) -> &'a EmitterGraph {
        self.graph
    }

    /// The canonical modifier node being visualized.
    ///
    /// The node is guaranteed to be the one used to construct this context.
    pub fn node(&self) -> &'a GraphNode {
        self.node
    }

    /// Resolve one input through the supported author-time paths.
    ///
    /// An incoming direct literal or property reference overrides the inline
    /// default. Dynamic and computed sources remain unresolved.
    pub fn value(&self, port: &str) -> Option<Value> {
        let node = self.node();
        let graph = self.graph();
        let input = node
            .inputs
            .iter()
            .find(|input| input.name.as_ref() == port)?;
        let mut incoming = graph
            .links
            .iter()
            .filter(|link| link.to.node == node.id && link.to.port.as_ref() == port);

        let Some(link) = incoming.next() else {
            let site = LiteralSite::Input {
                node: node.id,
                port: input.name.clone(),
            };
            return self
                .live_values
                .and_then(|values| values.get(&site).copied())
                .or_else(|| input.default.as_value());
        };
        if incoming.next().is_some() {
            return None;
        }

        let source = graph.node(link.from.node)?;
        match &source.payload {
            NodePayload::Expr(ExprNode::Literal(value)) => Some(*value),
            NodePayload::Expr(ExprNode::Property(id)) => {
                graph.property(*id).map(|property| property.default)
            }
            _ => None,
        }
    }

    /// Resolve an input as a finite, exactly typed `f32`.
    ///
    /// Numeric coercions are intentionally not performed.
    pub fn f32(&self, port: &str) -> Option<f32> {
        let value = self.value(port)?;
        let Value::Scalar(ScalarValue::Float(value)) = value else {
            return None;
        };
        value.is_finite().then_some(value)
    }

    /// Resolve an input as a finite, exactly typed `Vec3`.
    ///
    /// Numeric coercions are intentionally not performed.
    pub fn vec3(&self, port: &str) -> Option<Vec3> {
        let value = self.value(port)?;
        if value.value_type() != ValueType::Vector(VectorType::VEC3F) {
            return None;
        }
        let value = value.as_vector().as_vec3();
        value.is_finite().then_some(value)
    }

    /// Read a boolean config field or use its modifier default.
    ///
    /// A present config value of another type is treated as invalid.
    pub fn bool_config_or(&self, field: &str, default: bool) -> Option<bool> {
        match self.config_value(field) {
            None => Some(default),
            Some(EditValue::Bool(value)) => Some(*value),
            Some(_) => None,
        }
    }

    /// Read an enum config field or use its modifier default.
    ///
    /// A present config value of another type is treated as invalid.
    pub fn enum_config_or<'b>(&'b self, field: &str, default: &'b str) -> Option<&'b str> {
        match self.config_value(field) {
            None => Some(default),
            Some(EditValue::Enum { variant, .. }) => Some(variant.as_ref()),
            Some(_) => None,
        }
    }

    /// Read a modifier configuration field without coercion.
    ///
    /// Unknown modifiers and absent fields return `None`.
    pub fn config_value(&self, field: &str) -> Option<&EditValue> {
        let NodePayload::Modifier(ModifierNodeData::Known { config, .. }) = &self.node().payload
        else {
            return None;
        };
        config.get(field)
    }
}

/// Attach a gizmo provider to an already registered modifier type.
///
/// Returns `false` when the modifier has not yet been registered with the app's
/// type registry.
pub fn register_modifier_gizmo<T>(registry: &mut TypeRegistry, draw: ModifierGizmoFn) -> bool
where
    T: Modifier + Reflect + TypePath,
{
    let Some(registration) = registry.get_mut(TypeId::of::<T>()) else {
        return false;
    };
    registration.insert(ModifierGizmoProvider { draw });
    true
}

/// Adds retained modifier-gizmo lifecycle management.
///
/// Registers built-in providers and reconciles one layer-isolated preview per
/// document after each editor UI pass.
pub struct ModifierGizmoPlugin;

impl Plugin for ModifierGizmoPlugin {
    fn build(&self, app: &mut App) {
        register_builtin_modifier_gizmos(app);
        app.init_asset::<GizmoAsset>()
            .init_resource::<ModifierGizmoPreviews>()
            .add_systems(
                EguiPrimaryContextPass,
                reconcile_modifier_gizmos
                    .after(draw_editor_ui)
                    .after(crate::proxy::apply_live_value_edits),
            );
    }
}

#[derive(Component)]
struct ModifierGizmoPreview;

struct PreviewEntry {
    entity: Entity,
    asset: Handle<GizmoAsset>,
}

#[derive(Resource, Default)]
struct ModifierGizmoPreviews(HashMap<Entity, PreviewEntry>);

// Attach providers for Hanabi's built-in spatial modifiers.
fn register_builtin_modifier_gizmos(app: &mut App) {
    let app_registry = app.world().resource::<AppTypeRegistry>();
    let mut registry = app_registry.write();

    register_modifier_gizmo::<SetPositionCircleModifier>(&mut registry, position_circle);
    register_modifier_gizmo::<SetPositionSphereModifier>(&mut registry, position_sphere);
    register_modifier_gizmo::<SetPositionCone3dModifier>(&mut registry, position_cone);
    register_modifier_gizmo::<SetVelocityCircleModifier>(&mut registry, velocity_circle);
    register_modifier_gizmo::<SetVelocitySphereModifier>(&mut registry, velocity_sphere);
    register_modifier_gizmo::<SetVelocityTangentModifier>(&mut registry, velocity_tangent);
    register_modifier_gizmo::<AccelModifier>(&mut registry, accel);
    register_modifier_gizmo::<RadialAccelModifier>(&mut registry, radial_accel);
    register_modifier_gizmo::<TangentAccelModifier>(&mut registry, tangent_accel);
    register_modifier_gizmo::<ConformToSphereModifier>(&mut registry, conform_sphere);
    register_modifier_gizmo::<KillSphereModifier>(&mut registry, kill_sphere);
    register_modifier_gizmo::<KillAabbModifier>(&mut registry, kill_aabb);
}

fn reconcile_modifier_gizmos(
    mut commands: Commands,
    documents: Query<(Entity, &DocumentContent, &DocumentUi)>,
    preview_entities: Query<(), With<ModifierGizmoPreview>>,
    registry: Res<AppTypeRegistry>,
    frame_count: Res<bevy::diagnostic::FrameCount>,
    live_values: Res<crate::proxy::LiveValuePreviews>,
    mut assets: ResMut<Assets<GizmoAsset>>,
    mut previews: ResMut<ModifierGizmoPreviews>,
) {
    let registry = registry.read();
    let mut seen = HashSet::new();

    for (document, content, ui) in &documents {
        seen.insert(document);
        let asset = (ui.modifier_gizmo_frame == frame_count.0)
            .then_some(ui.modifier_gizmo_node)
            .flatten()
            .and_then(|node_id| {
                let emitter = content.effect_graph().emitter_owning_node(node_id)?;
                let graph = content.effect_graph().emitter(emitter)?;
                let node = graph.node(node_id)?;
                let document_live_values: HashMap<_, _> = live_values
                    .for_document(document, emitter)
                    .map(|(site, value)| (site.clone(), *value))
                    .collect();
                build_gizmo(graph, node, &document_live_values, &registry)
            });

        let Some(asset) = asset else {
            remove_preview(document, &mut commands, &mut assets, &mut previews);
            continue;
        };

        let layer = RenderLayers::layer(content.render_layer());
        if let Some(entry) = previews.0.get(&document)
            && preview_entities.contains(entry.entity)
            && let Some(mut current) = assets.get_mut(&entry.asset)
        {
            *current = asset;
            commands.entity(entry.entity).insert(layer);
            continue;
        }

        remove_preview(document, &mut commands, &mut assets, &mut previews);
        let handle = assets.add(asset);
        let entity = commands
            .spawn((
                Name::new("ModifierGizmoPreview"),
                ModifierGizmoPreview,
                Gizmo {
                    handle: handle.clone(),
                    line_config: GizmoLineConfig {
                        width: LINE_WIDTH,
                        perspective: false,
                        ..default()
                    },
                    depth_bias: DEPTH_BIAS,
                },
                layer,
            ))
            .id();
        commands.entity(document).add_child(entity);
        previews.0.insert(
            document,
            PreviewEntry {
                entity,
                asset: handle,
            },
        );
    }

    let stale: Vec<_> = previews
        .0
        .keys()
        .copied()
        .filter(|document| !seen.contains(document))
        .collect();
    for document in stale {
        remove_preview(document, &mut commands, &mut assets, &mut previews);
    }
}

fn remove_preview(
    document: Entity,
    commands: &mut Commands,
    assets: &mut Assets<GizmoAsset>,
    previews: &mut ModifierGizmoPreviews,
) {
    let Some(entry) = previews.0.remove(&document) else {
        return;
    };
    if let Ok(mut entity) = commands.get_entity(entry.entity) {
        entity.despawn();
    }
    assets.remove(entry.asset.id());
}

fn build_gizmo(
    graph: &EmitterGraph,
    node: &hanabi_effect_graph::model::GraphNode,
    live_values: &HashMap<LiteralSite, Value>,
    registry: &bevy::reflect::TypeRegistry,
) -> Option<GizmoAsset> {
    let NodePayload::Modifier(ModifierNodeData::Known { type_path, .. }) = &node.payload else {
        return None;
    };
    let provider = registry
        .get_with_type_path(type_path)?
        .data::<ModifierGizmoProvider>()?;
    let mut asset = GizmoAsset::new();
    if !(provider.draw)(
        &ModifierGizmoContext::with_live_values(graph, node, live_values),
        &mut asset,
    ) {
        return None;
    }
    let buffer = asset.buffer().buffer();
    let has_geometry = !buffer.list_positions.is_empty() || !buffer.strip_positions.is_empty();
    let finite = buffer
        .list_positions
        .iter()
        .chain(buffer.strip_positions.iter())
        .all(|position| position.is_finite());
    (has_geometry && finite).then_some(asset)
}

fn position_color() -> Color {
    Color::srgb(0.15, 0.8, 1.0)
}

fn velocity_color() -> Color {
    Color::srgb(0.35, 1.0, 0.45)
}

fn force_color() -> Color {
    Color::srgb(1.0, 0.62, 0.15)
}

fn kill_color(kill_inside: bool) -> Color {
    if kill_inside {
        Color::srgb(1.0, 0.2, 0.2)
    } else {
        Color::srgb(0.8, 0.25, 1.0)
    }
}

fn valid_radius(radius: f32) -> bool {
    radius.is_finite() && radius > EPSILON
}

fn valid_shape_dimension(context: &ModifierGizmoContext<'_>) -> bool {
    context
        .enum_config_or("dimension", "Surface")
        .is_some_and(|dimension| matches!(dimension, "Surface" | "Volume"))
}

fn normalized(axis: Vec3) -> Option<Vec3> {
    let length = axis.length();
    (axis.is_finite() && length.is_finite() && length > EPSILON).then(|| axis / length)
}

fn circle_basis(axis: Vec3) -> Option<(Vec3, Vec3, Vec3)> {
    let axis = normalized(axis)?;
    let reference = if axis.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let tangent = normalized(axis.cross(reference))?;
    Some((tangent, axis.cross(tangent), axis))
}

fn add_circle(asset: &mut GizmoAsset, center: Vec3, axis: Vec3, radius: f32, color: Color) -> bool {
    let Some((tangent, bitangent, _)) = circle_basis(axis) else {
        return false;
    };
    if !center.is_finite() || !valid_radius(radius) {
        return false;
    }
    for segment in 0..SEGMENTS {
        let a = std::f32::consts::TAU * segment as f32 / SEGMENTS as f32;
        let b = std::f32::consts::TAU * (segment + 1) as f32 / SEGMENTS as f32;
        asset.line(
            center + radius * (tangent * a.cos() + bitangent * a.sin()),
            center + radius * (tangent * b.cos() + bitangent * b.sin()),
            color,
        );
    }
    true
}

fn add_circle_axis(
    asset: &mut GizmoAsset,
    center: Vec3,
    axis: Vec3,
    axis_half_length: f32,
    circle_radius: f32,
    color: Color,
) -> bool {
    let Some((tangent, bitangent, axis)) = circle_basis(axis) else {
        return false;
    };
    if !center.is_finite()
        || !axis_half_length.is_finite()
        || axis_half_length <= EPSILON
        || !valid_radius(circle_radius)
    {
        return false;
    }

    asset.line(
        center - axis * axis_half_length,
        center + axis * axis_half_length,
        color,
    );
    let cross_half_size = (circle_radius * 0.12).clamp(0.04, 0.15);
    asset.line(
        center - tangent * cross_half_size,
        center + tangent * cross_half_size,
        color,
    );
    asset.line(
        center - bitangent * cross_half_size,
        center + bitangent * cross_half_size,
        color,
    );
    true
}

fn add_sphere(asset: &mut GizmoAsset, center: Vec3, radius: f32, color: Color) -> bool {
    if !center.is_finite() || !valid_radius(radius) {
        return false;
    }
    add_circle(asset, center, Vec3::X, radius, color)
        && add_circle(asset, center, Vec3::Y, radius, color)
        && add_circle(asset, center, Vec3::Z, radius, color)
}

fn add_arrow(asset: &mut GizmoAsset, start: Vec3, direction: Vec3, color: Color) -> bool {
    let length = direction.length();
    if !start.is_finite() || !direction.is_finite() || !length.is_finite() || length <= EPSILON {
        return false;
    }
    asset
        .arrow(start, start + direction, color)
        .with_tip_length((length * 0.18).min(0.25));
    true
}

fn guide_length(magnitude: f32) -> Option<f32> {
    let magnitude = magnitude.abs();
    (magnitude.is_finite() && magnitude > EPSILON).then(|| magnitude.clamp(0.35, 2.5))
}

fn add_tangent_field(
    asset: &mut GizmoAsset,
    origin: Vec3,
    axis: Vec3,
    magnitude: f32,
    color: Color,
) -> bool {
    let Some((tangent, bitangent, axis)) = circle_basis(axis) else {
        return false;
    };
    let Some(length) = guide_length(magnitude) else {
        return false;
    };
    if !origin.is_finite() {
        return false;
    }
    add_circle(asset, origin, axis, 1.0, color);
    add_circle_axis(asset, origin, axis, 1.0, 1.0, color);
    let sign = magnitude.signum();
    for radial in [tangent, bitangent, -tangent, -bitangent] {
        let start = origin + radial;
        add_arrow(asset, start, axis.cross(radial) * sign * length, color);
    }
    true
}

fn add_box(asset: &mut GizmoAsset, center: Vec3, half_size: Vec3, color: Color) -> bool {
    if !center.is_finite()
        || !half_size.is_finite()
        || half_size.min_element() < 0.0
        || half_size.max_element() <= EPSILON
    {
        return false;
    }
    let corners = [
        center + half_size * Vec3::new(-1.0, -1.0, -1.0),
        center + half_size * Vec3::new(1.0, -1.0, -1.0),
        center + half_size * Vec3::new(-1.0, 1.0, -1.0),
        center + half_size * Vec3::new(1.0, 1.0, -1.0),
        center + half_size * Vec3::new(-1.0, -1.0, 1.0),
        center + half_size * Vec3::new(1.0, -1.0, 1.0),
        center + half_size * Vec3::new(-1.0, 1.0, 1.0),
        center + half_size * Vec3::new(1.0, 1.0, 1.0),
    ];
    for (a, b) in [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ] {
        asset.line(corners[a], corners[b], color);
    }
    true
}

fn position_circle(context: &ModifierGizmoContext<'_>, asset: &mut GizmoAsset) -> bool {
    if !valid_shape_dimension(context) {
        return false;
    }
    let (Some(center), Some(axis), Some(radius)) = (
        context.vec3("center"),
        context.vec3("axis"),
        context.f32("radius"),
    ) else {
        return false;
    };
    let color = position_color();
    if !add_circle(asset, center, axis, radius, color) {
        return false;
    }
    add_circle_axis(asset, center, axis, radius.min(1.0), radius, color)
}

fn position_sphere(context: &ModifierGizmoContext<'_>, asset: &mut GizmoAsset) -> bool {
    if !valid_shape_dimension(context) {
        return false;
    }
    let (Some(center), Some(radius)) = (context.vec3("center"), context.f32("radius")) else {
        return false;
    };
    add_sphere(asset, center, radius, position_color())
}

fn position_cone(context: &ModifierGizmoContext<'_>, asset: &mut GizmoAsset) -> bool {
    if !valid_shape_dimension(context) {
        return false;
    }
    let (Some(height), Some(base_radius), Some(top_radius)) = (
        context.f32("height"),
        context.f32("base_radius"),
        context.f32("top_radius"),
    ) else {
        return false;
    };
    if height.abs() <= EPSILON
        || base_radius < 0.0
        || top_radius < 0.0
        || base_radius.max(top_radius) <= EPSILON
    {
        return false;
    }
    let color = position_color();
    if base_radius > EPSILON {
        add_circle(asset, Vec3::ZERO, Vec3::Y, base_radius, color);
    }
    if top_radius > EPSILON {
        add_circle(asset, Vec3::Y * height, Vec3::Y, top_radius, color);
    }
    for segment in 0..8 {
        let angle = std::f32::consts::TAU * segment as f32 / 8.0;
        let radial = Vec3::new(angle.cos(), 0.0, angle.sin());
        asset.line(
            radial * base_radius,
            Vec3::Y * height + radial * top_radius,
            color,
        );
    }
    true
}

fn velocity_circle(context: &ModifierGizmoContext<'_>, asset: &mut GizmoAsset) -> bool {
    let (Some(center), Some(axis), Some(speed)) = (
        context.vec3("center"),
        context.vec3("axis"),
        context.f32("speed"),
    ) else {
        return false;
    };
    let Some((tangent, bitangent, axis)) = circle_basis(axis) else {
        return false;
    };
    let Some(length) = guide_length(speed) else {
        return false;
    };
    let color = velocity_color();
    add_circle(asset, center, axis, 1.0, color);
    add_circle_axis(asset, center, axis, 1.0, 1.0, color);
    for radial in [tangent, bitangent, -tangent, -bitangent] {
        add_arrow(
            asset,
            center + radial,
            radial * speed.signum() * length,
            color,
        );
    }
    true
}

fn velocity_sphere(context: &ModifierGizmoContext<'_>, asset: &mut GizmoAsset) -> bool {
    let (Some(center), Some(speed)) = (context.vec3("center"), context.f32("speed")) else {
        return false;
    };
    let Some(length) = guide_length(speed) else {
        return false;
    };
    let color = velocity_color();
    for radial in [
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::NEG_X,
        Vec3::NEG_Y,
        Vec3::NEG_Z,
    ] {
        add_arrow(
            asset,
            center + radial * 0.5,
            radial * speed.signum() * length,
            color,
        );
    }
    true
}

fn velocity_tangent(context: &ModifierGizmoContext<'_>, asset: &mut GizmoAsset) -> bool {
    let (Some(origin), Some(axis), Some(speed)) = (
        context.vec3("origin"),
        context.vec3("axis"),
        context.f32("speed"),
    ) else {
        return false;
    };
    add_tangent_field(asset, origin, axis, speed, velocity_color())
}

fn accel(context: &ModifierGizmoContext<'_>, asset: &mut GizmoAsset) -> bool {
    let Some(acceleration) = context.vec3("accel") else {
        return false;
    };
    let magnitude = acceleration.length();
    let Some(length) = guide_length(magnitude) else {
        return false;
    };
    add_arrow(
        asset,
        Vec3::ZERO,
        acceleration / magnitude * length,
        force_color(),
    )
}

fn radial_accel(context: &ModifierGizmoContext<'_>, asset: &mut GizmoAsset) -> bool {
    let (Some(origin), Some(acceleration)) = (context.vec3("origin"), context.f32("accel")) else {
        return false;
    };
    let Some(length) = guide_length(acceleration) else {
        return false;
    };
    let color = force_color();
    for radial in [
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::NEG_X,
        Vec3::NEG_Y,
        Vec3::NEG_Z,
    ] {
        add_arrow(
            asset,
            origin + radial,
            radial * acceleration.signum() * length,
            color,
        );
    }
    true
}

fn tangent_accel(context: &ModifierGizmoContext<'_>, asset: &mut GizmoAsset) -> bool {
    let (Some(origin), Some(axis), Some(acceleration)) = (
        context.vec3("origin"),
        context.vec3("axis"),
        context.f32("accel"),
    ) else {
        return false;
    };
    add_tangent_field(asset, origin, axis, acceleration, force_color())
}

fn conform_sphere(context: &ModifierGizmoContext<'_>, asset: &mut GizmoAsset) -> bool {
    let (Some(origin), Some(radius), Some(influence)) = (
        context.vec3("origin"),
        context.f32("radius"),
        context.f32("influence_dist"),
    ) else {
        return false;
    };
    if !valid_radius(radius) || influence < 0.0 {
        return false;
    }
    let color = force_color();
    add_sphere(asset, origin, radius, color);
    if influence > EPSILON && !add_sphere(asset, origin, radius + influence, color) {
        return false;
    }
    if let Some(shell) = context.f32("shell_half_thickness") {
        if shell < 0.0 {
            return false;
        }
        if radius - shell > EPSILON && !add_sphere(asset, origin, radius - shell, color) {
            return false;
        }
        if shell > EPSILON && !add_sphere(asset, origin, radius + shell, color) {
            return false;
        }
    }
    true
}

fn kill_sphere(context: &ModifierGizmoContext<'_>, asset: &mut GizmoAsset) -> bool {
    let (Some(center), Some(squared_radius)) = (context.vec3("center"), context.f32("sqr_radius"))
    else {
        return false;
    };
    if squared_radius <= EPSILON {
        return false;
    }
    let Some(kill_inside) = context.bool_config_or("kill_inside", false) else {
        return false;
    };
    add_sphere(
        asset,
        center,
        squared_radius.sqrt(),
        kill_color(kill_inside),
    )
}

fn kill_aabb(context: &ModifierGizmoContext<'_>, asset: &mut GizmoAsset) -> bool {
    let (Some(center), Some(half_size)) = (context.vec3("center"), context.vec3("half_size"))
    else {
        return false;
    };
    let Some(kill_inside) = context.bool_config_or("kill_inside", false) else {
        return false;
    };
    add_box(asset, center, half_size, kill_color(kill_inside))
}

#[cfg(test)]
mod tests {
    use bevy::reflect::TypePath;
    use hanabi_effect_graph::model::{
        GraphLink, GraphNode, InputSlot, NodeId, PortRef, PropertyDef, PropertyId,
    };

    use super::*;

    fn id(value: u32) -> NodeId {
        NodeId::new(value).unwrap()
    }

    fn modifier_node(default: Value) -> GraphNode {
        GraphNode {
            id: id(1),
            payload: NodePayload::Modifier(ModifierNodeData::Known {
                type_path: "TestModifier".into(),
                config: Default::default(),
            }),
            inputs: vec![InputSlot {
                name: "value".into(),
                default: default.into(),
            }],
        }
    }

    fn link(from: u32) -> GraphLink {
        GraphLink {
            from: PortRef {
                node: id(from),
                port: "out".into(),
            },
            to: PortRef {
                node: id(1),
                port: "value".into(),
            },
        }
    }

    #[test]
    fn resolves_inline_literal_and_property_values() {
        let mut graph = EmitterGraph::empty(EmitterId::new(1).unwrap());
        graph.nodes.push(modifier_node(1.0_f32.into()));
        assert_eq!(
            ModifierGizmoContext::new(&graph, &graph.nodes[0]).f32("value"),
            Some(1.0)
        );

        graph.nodes.push(GraphNode {
            id: id(2),
            payload: NodePayload::Expr(ExprNode::Literal(Value::from(2.0_f32))),
            inputs: vec![],
        });
        graph.links.push(link(2));
        assert_eq!(
            ModifierGizmoContext::new(&graph, &graph.nodes[0]).f32("value"),
            Some(2.0)
        );

        let property_id = PropertyId::new(4).unwrap();
        graph.properties.push(PropertyDef {
            id: property_id,
            name: "speed".into(),
            default: Value::from(3.0_f32),
            exposed: false,
        });
        graph.nodes[1].payload = NodePayload::Expr(ExprNode::Property(property_id));
        assert_eq!(
            ModifierGizmoContext::new(&graph, &graph.nodes[0]).f32("value"),
            Some(3.0)
        );

        graph.nodes[1].payload = NodePayload::Expr(ExprNode::Property(PropertyId::new(9).unwrap()));
        assert_eq!(
            ModifierGizmoContext::new(&graph, &graph.nodes[0]).value("value"),
            None
        );
    }

    #[test]
    fn live_value_overrides_inline_default() {
        let mut graph = EmitterGraph::empty(EmitterId::new(1).unwrap());
        graph.nodes.push(modifier_node(1.0_f32.into()));
        let mut live_values = HashMap::new();
        live_values.insert(
            LiteralSite::Input {
                node: id(1),
                port: "value".into(),
            },
            Value::from(2.0_f32),
        );

        assert_eq!(
            ModifierGizmoContext::with_live_values(&graph, &graph.nodes[0], &live_values)
                .f32("value"),
            Some(2.0)
        );

        graph.nodes.push(GraphNode {
            id: id(2),
            payload: NodePayload::Expr(ExprNode::Literal(Value::from(3.0_f32))),
            inputs: vec![],
        });
        graph.links.push(link(2));
        assert_eq!(
            ModifierGizmoContext::with_live_values(&graph, &graph.nodes[0], &live_values)
                .f32("value"),
            Some(3.0)
        );
    }

    #[test]
    fn rejects_computed_wrong_type_and_non_finite_values() {
        let mut graph = EmitterGraph::empty(EmitterId::new(1).unwrap());
        graph.nodes.push(modifier_node(Vec3::ONE.into()));
        assert_eq!(
            ModifierGizmoContext::new(&graph, &graph.nodes[0]).f32("value"),
            None
        );

        graph.nodes[0].inputs[0].default = Value::from(f32::NAN).into();
        assert_eq!(
            ModifierGizmoContext::new(&graph, &graph.nodes[0]).f32("value"),
            None
        );

        graph.nodes.push(GraphNode {
            id: id(2),
            payload: NodePayload::Expr(ExprNode::Attribute(bevy_hanabi::Attribute::POSITION)),
            inputs: vec![],
        });
        graph.links.push(link(2));
        assert_eq!(
            ModifierGizmoContext::new(&graph, &graph.nodes[0]).value("value"),
            None
        );
    }

    #[test]
    fn registers_every_builtin_spatial_provider() {
        let mut app = App::new();
        app.add_plugins(crate::modifier_registry::ModifierRegistryPlugin);
        register_builtin_modifier_gizmos(&mut app);
        let registry = app.world().resource::<AppTypeRegistry>().read();
        for type_path in [
            SetPositionCircleModifier::type_path(),
            SetPositionSphereModifier::type_path(),
            SetPositionCone3dModifier::type_path(),
            SetVelocityCircleModifier::type_path(),
            SetVelocitySphereModifier::type_path(),
            SetVelocityTangentModifier::type_path(),
            AccelModifier::type_path(),
            RadialAccelModifier::type_path(),
            TangentAccelModifier::type_path(),
            ConformToSphereModifier::type_path(),
            KillSphereModifier::type_path(),
            KillAabbModifier::type_path(),
        ] {
            assert!(
                registry
                    .get_with_type_path(type_path)
                    .and_then(|registration| registration.data::<ModifierGizmoProvider>())
                    .is_some(),
                "missing gizmo provider for {type_path}"
            );
        }
    }

    #[test]
    fn circle_axis_marks_its_plane_intersection() {
        let mut asset = GizmoAsset::new();
        assert!(add_circle_axis(
            &mut asset,
            Vec3::ZERO,
            Vec3::Z,
            1.0,
            1.0,
            Color::WHITE,
        ));

        let positions = &asset.buffer().buffer().list_positions;
        assert_eq!(positions.len(), 6);
        for segment in positions.chunks_exact(2).skip(1) {
            assert!((segment[0] + segment[1]).length() <= EPSILON);
            assert!((segment[1] - segment[0]).dot(Vec3::Z).abs() <= EPSILON);
        }
    }

    #[test]
    fn sphere_provider_generates_finite_geometry() {
        let mut graph = EmitterGraph::empty(EmitterId::new(1).unwrap());
        graph.nodes.push(GraphNode {
            id: id(1),
            payload: NodePayload::Modifier(ModifierNodeData::Known {
                type_path: SetPositionSphereModifier::type_path().into(),
                config: Default::default(),
            }),
            inputs: vec![
                InputSlot {
                    name: "center".into(),
                    default: bevy_hanabi::Value::from(Vec3::new(1.0, 2.0, 3.0)).into(),
                },
                InputSlot {
                    name: "radius".into(),
                    default: bevy_hanabi::Value::from(2.0_f32).into(),
                },
            ],
        });
        let mut asset = GizmoAsset::new();
        assert!(position_sphere(
            &ModifierGizmoContext::new(&graph, &graph.nodes[0]),
            &mut asset
        ));
        let buffer = asset.buffer().buffer();
        assert!(!buffer.list_positions.is_empty());
        assert!(buffer.list_positions.iter().all(|point| point.is_finite()));
    }
}
