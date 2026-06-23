//! Data types of the [`EffectGraph`] model.
//!
//! These are plain serde data. Modifier nodes store an editable config bag
//! ([`ModifierNodeData`]) keyed by reflected field name rather than a runtime
//! modifier object, so the whole graph serializes directly to RON without a
//! reflection pass. The type registry is consulted only when *baking* the
//! graph to an [`bevy_hanabi::EffectAsset`] or *raising* one back, never for
//! (de)serialization.
//!
//! [`EffectGraph`]: crate::model::EffectGraph

use std::{collections::BTreeMap, num::NonZeroU32, sync::Arc};

use bevy::{
    asset::{Asset, AssetPath},
    math::{UVec2, Vec3, Vec4},
    reflect::TypePath,
};
use bevy_hanabi::{
    Attribute, BuiltInOperator, CpuValue, Gradient, SimulationCondition, SimulationSpace,
    SpawnerSettings, Value, ValueType,
    graph::expr::{BinaryOperator, TernaryOperator, UnaryOperator},
};
use serde::{Deserialize, Serialize};

use crate::ModifierGroup;

/// A cheaply-clonable, immutable string for names and identifiers.
///
/// Constructed once (typically from reflection) and never mutated afterwards —
/// port names, field keys, reflect type paths, enum variants, property names.
///
/// Cloning only bumps an atomic refcount (no allocation or copy), it is two
/// words instead of `String`'s three, and it is `Send + Sync` for storage in
/// ECS components. Editing such a string replaces the whole value, which is no
/// more costly than building a `String`.
pub type SharedStr = Arc<str>;

/// On-disk format version.
///
/// Bumped on any breaking change to the schema.
pub const FORMAT_VERSION: u32 = 1;

/// Identifier of a graph node, one-based and never reused within a graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NodeId(pub NonZeroU32);

impl NodeId {
    pub fn new(one_based: u32) -> Option<Self> {
        NonZeroU32::new(one_based).map(Self)
    }

    pub fn get(&self) -> u32 {
        self.0.get()
    }
}

/// Identifier of a modifier stack, one-based and never reused within a graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct StackId(pub NonZeroU32);

impl StackId {
    pub fn new(one_based: u32) -> Option<Self> {
        NonZeroU32::new(one_based).map(Self)
    }

    pub fn get(&self) -> u32 {
        self.0.get()
    }
}

/// Identifier of a user property, one-based and never reused within a graph.
///
/// Expression nodes reference a property by this stable id, not by name, so a
/// property can be freely renamed (or share a display name with another)
/// without breaking its references. Drawn from the same allocator as node and
/// stack ids so the three id spaces never collide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PropertyId(pub NonZeroU32);

impl PropertyId {
    pub fn new(one_based: u32) -> Option<Self> {
        NonZeroU32::new(one_based).map(Self)
    }

    pub fn get(&self) -> u32 {
        self.0.get()
    }
}

/// Identifier of a texture slot, one-based and never reused within a graph.
///
/// [`ImageBinding::Slot`] references a slot by this stable id, so a slot may
/// be reordered or renamed without invalidating the binding. Drawn from the
/// same counter as node, stack, and property ids so the four id spaces never
/// collide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SlotId(pub NonZeroU32);

impl SlotId {
    pub fn new(one_based: u32) -> Option<Self> {
        NonZeroU32::new(one_based).map(Self)
    }

    pub fn get(&self) -> u32 {
        self.0.get()
    }
}

/// An expression node's payload: which kind of `Expr` it produces.
///
/// Operand expressions are *not* stored here — they are links into this node's
/// derived input ports. This is a closed set (Hanabi's
/// [`bevy_hanabi::graph::Expr`] is not user-extensible), so it serializes
/// directly, unlike modifier payloads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExprNode {
    /// A shader constant. Doubles as an input port's inline default elsewhere.
    Literal(Value),
    /// Reference to a user-defined effect property, by stable id (not name, so
    /// the property may be renamed without invalidating the reference).
    Property(PropertyId),
    /// A particle attribute read (e.g. position, velocity).
    Attribute(Attribute),
    /// The same, but reading the parent particle's attribute (GPU events).
    ParentAttribute(Attribute),
    /// A built-in quantity such as the simulation time.
    BuiltIn(BuiltInOperator),
    /// Unary operation over one operand input.
    Unary(UnaryOperator),
    /// Binary operation over two operand inputs.
    Binary(BinaryOperator),
    /// Ternary operation over three operand inputs.
    Ternary(TernaryOperator),
    /// Cast of one operand input to the given value type.
    Cast(ValueType),
    /// An image source: a pinned asset, a texture slot, or unbound.
    ///
    /// A source node (no inputs) whose output is the editor's image
    /// pseudo-type. The binding it carries (see [`ImageBinding`]) is authored
    /// on the node and lowers to a `u32` slot index at bake time.
    Image(ImageBinding),
    /// Sample a texture at given coordinates.
    ///
    /// Inputs are `image` — an image source — and `coordinates` (a `vec2`); the
    /// output is the sampled `vec4`.
    TextureSample,
    /// Pick one of several image sources by a runtime index.
    ///
    /// Inputs are `index` (a `u32`) followed by `image0`, `image1`, ... image
    /// sources; the output is the image at the selected index. The image-input
    /// count is link-derived: one empty trailing port is always offered, so
    /// connecting it grows the node and clearing the highest one shrinks it.
    SelectImage { count: u32 },
}

/// A texture binding for a modifier field.
///
/// Hanabi 0.18 can only reference textures through a runtime-bound slot index
/// (an [`bevy_hanabi::EffectMaterial`] supplies the images per instance). The
/// edit model additionally lets an artist pin a specific image asset at
/// authoring time, which is the more common workflow.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextureValue {
    /// A specific image asset chosen in the editor. Bound to every instance of
    /// the effect; travels with the document. Stored as an [`AssetPath`] so it
    /// can name an asset source and/or sub-asset label, serialized as a plain
    /// path string.
    Asset(AssetPath<'static>),
    /// A named slot left unbound, for runtime code to fill per `ParticleEffect`
    /// (Hanabi's native `EffectMaterial` mechanism).
    Slot { name: SharedStr },
}

impl Default for TextureValue {
    fn default() -> Self {
        TextureValue::Slot {
            name: SharedStr::from("texture"),
        }
    }
}

/// What an image source resolves to: an asset, a texture slot, or nothing.
///
/// Carried by an [`ExprNode::Image`] node. An asset is pinned at authoring time
/// and travels with the graph; a texture slot is supplied per-`ParticleEffect`
/// by the host game through [`bevy_hanabi::EffectMaterial`], referenced by
/// stable [`SlotId`] so the slot may be renamed or reordered without
/// invalidating the binding.
///
/// [`ExprNode::Image`]: ExprNode::Image
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ImageBinding {
    /// No image bound yet; a placeholder until one is chosen.
    #[default]
    Unbound,
    /// A specific image asset chosen in the editor, bound to every instance.
    Asset(AssetPath<'static>),
    /// A host-supplied texture slot, referenced by stable id (see
    /// [`TextureSlotDef`]).
    Slot(SlotId),
}

/// A host-supplied texture slot: a stable id and a display name.
///
/// The slot's *sampling index* is its position in
/// [`EffectGraph::texture_slots`], so reordering the list reassigns indices —
/// the binding ABI that the host game targets through
/// [`bevy_hanabi::EffectMaterial`]. An [`ImageBinding::Slot`] references a slot
/// by [`id`] (not index) so it survives reordering. Asset-bound images are
/// pinned on their source node and never appear here.
///
/// [`id`]: TextureSlotDef::id
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextureSlotDef {
    /// Stable reference identity, distinct from the display name and index.
    pub id: SlotId,
    pub name: SharedStr,
}

/// A `Vec3`-valued gradient (e.g. size over lifetime).
///
/// Anticipates richer forms than Hanabi 0.18's analytical keyframe gradient.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GradientVec3 {
    /// Piecewise-linear keyframe gradient (the only form Hanabi 0.18 bakes).
    Analytical(Gradient<Vec3>),
    /// A texture-backed lookup table sampled along its length.
    Lut(TextureValue),
}

/// A `Vec4`-valued gradient (e.g. color over lifetime).
///
/// See [`GradientVec3`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GradientVec4 {
    Analytical(Gradient<Vec4>),
    Lut(TextureValue),
}

/// A directly-editable configuration value for a non-expression modifier field.
///
/// Expression inputs become ports instead. Each variant maps to a concrete
/// `bevy_hanabi` field type and, wherever the upstream type already derives
/// serde, reuses it verbatim so the on-disk form never drifts from the runtime
/// type. [`EditValue::Raw`] is the forward-compatible escape hatch for field
/// types a future Hanabi version may introduce.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EditValue {
    Bool(bool),
    U32(u32),
    /// A scalar or vector constant (Hanabi `Value` covers f32/Vec2/Vec3/Vec4
    /// and their integer counterparts).
    Scalar(Value),
    UVec2(UVec2),
    /// An RGBA color. Distinguished from a plain `Vec4` so the UI can offer a
    /// color picker rather than four sliders.
    Color(Vec4),
    Attribute(Attribute),
    CpuVec3(CpuValue<Vec3>),
    CpuVec4(CpuValue<Vec4>),
    Gradient3(GradientVec3),
    Gradient4(GradientVec4),
    Texture(TextureValue),
    /// A data-less enum, identified by reflect type path plus active variant
    /// (e.g. `ShapeDimension`, `OrientMode`, `ColorBlendMode`).
    Enum {
        type_path: SharedStr,
        variant: SharedStr,
    },
    /// A bitflags newtype (e.g. `ColorBlendMask`). Stored as `u64` to
    /// accommodate any flag width; baking narrows to the field's actual
    /// repr.
    Flags {
        type_path: SharedStr,
        bits: u64,
    },
    /// Fallback for a field type not yet modeled first-class: its value
    /// serialized as a RON fragment, preserved verbatim for round-tripping.
    Raw(String),
}

/// The payload of a modifier node.
///
/// A [`ModifierNodeData::Known`] modifier has a registered reflect type and an
/// editable config bag; expression-typed fields are not stored here — they are
/// the node's derived input ports. A [`ModifierNodeData::Unknown`] modifier
/// (type not registered locally) keeps its serialized reflect data verbatim so
/// it round-trips, but cannot be edited or baked until its type becomes
/// available.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModifierNodeData {
    Known {
        /// Reflect type path; the node-kind identity.
        type_path: SharedStr,
        /// Non-expression fields keyed by reflected field name. Iteration order
        /// for display comes from the cached schema, not this map.
        config: BTreeMap<SharedStr, EditValue>,
    },
    Unknown {
        type_path: SharedStr,
        /// The modifier's reflect data as a RON fragment.
        raw: String,
    },
}

/// A node's payload — what the node *is*.
///
/// Expression nodes carry a closed [`ExprNode`]; modifier nodes carry an
/// editable [`ModifierNodeData`] whose concrete runtime type is materialized
/// only when baking to an [`bevy_hanabi::EffectAsset`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodePayload {
    Expr(ExprNode),
    Modifier(ModifierNodeData),
}

/// Inline default for one of a node's derived input ports.
///
/// Used whenever no [`GraphLink`] targets that port. A value port carries a
/// literal [`Value`]; an image port carries an [`ImageBinding`]. Ports are
/// addressed by name (matching the modifier's reflected field name or the
/// expression operand name), which is stable across registry evolution in a way
/// indices are not.
///
/// [`Value`]: bevy_hanabi::Value
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InputDefault {
    /// A literal value feeding a value-typed port.
    Value(Value),
    /// An image binding feeding the image pseudo-typed port of a texture
    /// consumer (a sampler's `image` or a modifier's `texture_slot`).
    Image(ImageBinding),
}

impl InputDefault {
    /// The literal value if this is a value default, else `None`.
    pub fn as_value(&self) -> Option<Value> {
        match self {
            InputDefault::Value(v) => Some(*v),
            InputDefault::Image(_) => None,
        }
    }

    /// The image binding if this is an image default, else `None`.
    pub fn as_image(&self) -> Option<&ImageBinding> {
        match self {
            InputDefault::Image(b) => Some(b),
            InputDefault::Value(_) => None,
        }
    }
}

impl From<Value> for InputDefault {
    fn from(v: Value) -> Self {
        InputDefault::Value(v)
    }
}

impl From<ImageBinding> for InputDefault {
    fn from(b: ImageBinding) -> Self {
        InputDefault::Image(b)
    }
}

/// A named input port carrying its inline default.
///
/// Pairs a port name with the [`InputDefault`] used whenever no [`GraphLink`]
/// targets that port. Ports are addressed by name (matching the modifier's
/// reflected field name or the expression operand name), which is stable across
/// registry evolution in a way indices are not.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputSlot {
    pub name: SharedStr,
    pub default: InputDefault,
}

/// A node in the graph: a stable id, a payload, and its inline port defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: NodeId,
    pub payload: NodePayload,
    pub inputs: Vec<InputSlot>,
}

/// A fully-qualified port address: a node plus one of its ports by name.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PortRef {
    pub node: NodeId,
    pub port: SharedStr,
}

/// A directed link from an output port to an input port.
///
/// One output may fan out to many inputs; an input takes at most one link.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GraphLink {
    pub from: PortRef,
    pub to: PortRef,
}

/// An ordered container of modifier member nodes for one simulation phase.
///
/// The pipeline executes its stacks in `Init → Update → Render` order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphStack {
    pub id: StackId,
    pub group: ModifierGroup,
    pub members: Vec<NodeId>,
}

/// A named, editable effect parameter with a default value.
///
/// The default also fixes its value type. Expression nodes reference it by
/// [`id`] via [`ExprNode::Property`].
///
/// By default a property is *edit-only*: it exists purely as an authoring
/// convenience and every reference is inlined to a literal constant when the
/// graph is baked, so it has no runtime representation or cost. Setting
/// [`exposed`] promotes it to a real runtime property, exported to the effect's
/// `Module` and overridable per instance via `EffectProperties`.
///
/// The [`name`] is display-only and need not be unique among edit-only
/// properties. Exposed properties, however, become runtime `Module` properties
/// keyed by name, so two exposed properties sharing a name is an inconsistency
/// that blocks baking (surfaced as a bake error, never a crash) until the
/// author renames one.
///
/// [`id`]: PropertyDef::id
/// [`exposed`]: PropertyDef::exposed
/// [`name`]: PropertyDef::name
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropertyDef {
    /// Stable reference identity, distinct from the display name.
    pub id: PropertyId,
    pub name: SharedStr,
    pub default: Value,
    /// Whether this property survives baking as a runtime-settable property
    /// (`true`) or is baked into literals at each reference (`false`, default).
    #[serde(default)]
    pub exposed: bool,
}

/// Effect-level settings that are not part of the expression graph.
///
/// Mirrors the scalar configuration of an [`bevy_hanabi::EffectAsset`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectHeader {
    pub name: SharedStr,
    pub capacity: u32,
    pub spawner: SpawnerSettings,
    pub simulation_space: SimulationSpace,
    pub simulation_condition: SimulationCondition,
    pub z_layer_2d: f32,
}

/// The semantic graph: header, properties, nodes, stacks, and links.
///
/// Plus the monotonic id allocator. Diff-friendly and layout-free.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectGraph {
    pub header: EffectHeader,
    pub properties: Vec<PropertyDef>,
    /// Host-supplied texture slots, ordered by sampling index.
    ///
    /// Referenced by [`ImageBinding::Slot`] via stable [`SlotId`] and filled
    /// per-instance by the host game through [`bevy_hanabi::EffectMaterial`].
    /// Asset-bound images are pinned on their source node and do not appear
    /// here. Empty for an effect with no host-supplied textures.
    #[serde(default)]
    pub texture_slots: Vec<TextureSlotDef>,
    pub nodes: Vec<GraphNode>,
    pub stacks: Vec<GraphStack>,
    pub links: Vec<GraphLink>,
    /// Next id to hand out; only ever increases. Ids are never recycled so that
    /// links and persisted layout stay valid across edits and reloads.
    pub next_id: u32,
}

impl EffectGraph {
    /// An empty graph with a default header.
    ///
    /// A placeholder for documents whose graph is not yet populated (e.g. a
    /// legacy `EffectAsset` opened before the import path exists). Carries no
    /// nodes, stacks, links, or properties.
    pub fn empty() -> Self {
        Self {
            header: EffectHeader {
                name: "untitled".into(),
                capacity: 4096,
                spawner: SpawnerSettings::default(),
                simulation_space: SimulationSpace::default(),
                simulation_condition: SimulationCondition::default(),
                z_layer_2d: 0.0,
            },
            properties: Vec::new(),
            texture_slots: Vec::new(),
            nodes: Vec::new(),
            stacks: Vec::new(),
            links: Vec::new(),
            next_id: 1,
        }
    }

    /// Mint a fresh, never-before-used [`NodeId`].
    pub fn alloc_node_id(&mut self) -> NodeId {
        let id = NodeId::new(self.next_id).expect("node id allocator overflow");
        self.next_id += 1;
        id
    }

    /// Mint a fresh, never-before-used [`StackId`].
    ///
    /// Drawn from the same counter as node ids so the two id spaces never
    /// collide.
    pub fn alloc_stack_id(&mut self) -> StackId {
        let id = StackId::new(self.next_id).expect("stack id allocator overflow");
        self.next_id += 1;
        id
    }

    /// Mint a fresh, never-before-used [`PropertyId`].
    ///
    /// Drawn from the same counter as node and stack ids so the three id spaces
    /// never collide.
    pub fn alloc_property_id(&mut self) -> PropertyId {
        let id = PropertyId::new(self.next_id).expect("property id allocator overflow");
        self.next_id += 1;
        id
    }

    /// Mint a fresh, never-before-used [`SlotId`].
    ///
    /// Drawn from the same counter as node, stack, and property ids so the four
    /// id spaces never collide.
    pub fn alloc_slot_id(&mut self) -> SlotId {
        let id = SlotId::new(self.next_id).expect("slot id allocator overflow");
        self.next_id += 1;
        id
    }

    pub fn node(&self, id: NodeId) -> Option<&GraphNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut GraphNode> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    pub fn stack(&self, group: ModifierGroup) -> Option<&GraphStack> {
        self.stacks.iter().find(|s| s.group == group)
    }

    pub fn property(&self, id: PropertyId) -> Option<&PropertyDef> {
        self.properties.iter().find(|p| p.id == id)
    }

    pub fn texture_slot(&self, id: SlotId) -> Option<&TextureSlotDef> {
        self.texture_slots.iter().find(|s| s.id == id)
    }

    /// Sampling index of a texture slot (its position in [`texture_slots`]), by
    /// id.
    ///
    /// [`texture_slots`]: EffectGraph::texture_slots
    pub fn texture_slot_index(&self, id: SlotId) -> Option<usize> {
        self.texture_slots.iter().position(|s| s.id == id)
    }
}

/// UI layout for a graph: viewport transform plus node/stack positions.
///
/// Optional; regenerated by auto-layout when absent. Positions are stored as
/// plain `(x, y)` pairs to keep the schema independent of the math crate.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct GraphLayout {
    pub pan: (f64, f64),
    pub zoom: f64,
    pub node_pos: Vec<(NodeId, (f64, f64))>,
    pub stack_pos: Vec<(StackId, (f64, f64))>,
}

/// The loadable effect-graph asset: version, [`EffectGraph`], and layout.
///
/// Holds a schema version, the semantic [`EffectGraph`], and an optional
/// [`GraphLayout`]. This is the canonical edited and persisted unit (an
/// [`EffectAsset`] is a derived bake output of it). As a Bevy [`Asset`] it can
/// be loaded from any asset source — a `.hnb` file is just one of them — and
/// held by handle.
///
/// The schema [`version`] is validated, and migrated if older, by the asset
/// loader; the writer always stamps [`FORMAT_VERSION`].
///
/// [`EffectAsset`]: bevy_hanabi::EffectAsset
/// [`version`]: EffectGraphAsset::version
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Asset, TypePath)]
pub struct EffectGraphAsset {
    pub version: u32,
    pub graph: EffectGraph,
    pub layout: Option<GraphLayout>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize to RON and back, asserting the value is preserved exactly.
    fn round_trip<T>(value: &T)
    where
        T: Serialize + for<'de> Deserialize<'de> + PartialEq + std::fmt::Debug,
    {
        let ron = ron::ser::to_string(value).expect("serialize");
        let back: T = ron::de::from_str(&ron).unwrap_or_else(|e| panic!("deserialize {ron}: {e}"));
        assert_eq!(value, &back, "round-trip mismatch via {ron}");
    }

    #[test]
    fn edit_value_variants_round_trip() {
        round_trip(&EditValue::Bool(true));
        round_trip(&EditValue::U32(7));
        round_trip(&EditValue::Scalar(Value::from(1.5f32)));
        round_trip(&EditValue::UVec2(UVec2::new(2, 3)));
        round_trip(&EditValue::Color(Vec4::new(1.0, 0.5, 0.25, 1.0)));
        round_trip(&EditValue::Attribute(Attribute::LIFETIME));
        round_trip(&EditValue::CpuVec3(CpuValue::Single(Vec3::ONE)));
        round_trip(&EditValue::CpuVec4(CpuValue::Uniform((
            Vec4::ZERO,
            Vec4::ONE,
        ))));
        round_trip(&EditValue::Gradient3(GradientVec3::Analytical(
            Gradient::linear(Vec3::ZERO, Vec3::ONE),
        )));
        round_trip(&EditValue::Gradient4(GradientVec4::Lut(
            TextureValue::Asset("ramps/fire.png".into()),
        )));
        round_trip(&EditValue::Texture(TextureValue::Slot {
            name: "color".into(),
        }));
        round_trip(&EditValue::Enum {
            type_path: "bevy_hanabi::modifier::ShapeDimension".into(),
            variant: "Surface".into(),
        });
        round_trip(&EditValue::Flags {
            type_path: "bevy_hanabi::modifier::output::ColorBlendMask".into(),
            bits: 0b101,
        });
        round_trip(&EditValue::Raw("(some: \"future field\")".to_string()));
    }

    #[test]
    fn modifier_node_data_round_trips() {
        let mut config = BTreeMap::new();
        config.insert(
            "color".into(),
            EditValue::CpuVec4(CpuValue::Single(Vec4::ONE)),
        );
        config.insert(
            "blend".into(),
            EditValue::Enum {
                type_path: "bevy_hanabi::modifier::output::ColorBlendMode".into(),
                variant: "Overwrite".into(),
            },
        );
        round_trip(&ModifierNodeData::Known {
            type_path: "bevy_hanabi::modifier::output::SetColorModifier".into(),
            config,
        });
        round_trip(&ModifierNodeData::Unknown {
            type_path: "my_crate::CustomModifier".into(),
            raw: "(strength: 2.0)".to_string(),
        });
    }

    #[test]
    fn effect_graph_asset_round_trips() {
        let n1 = NodeId::new(1).unwrap();
        let n2 = NodeId::new(2).unwrap();
        let stack = StackId::new(3).unwrap();
        let speed = PropertyId::new(5).unwrap();
        let tint = PropertyId::new(6).unwrap();
        let slot = SlotId::new(7).unwrap();
        let n_image = NodeId::new(8).unwrap();
        let n_sample = NodeId::new(9).unwrap();

        let graph = EffectGraph {
            header: EffectHeader {
                name: "demo".into(),
                capacity: 4096,
                spawner: SpawnerSettings::rate(64.0.into()),
                simulation_space: SimulationSpace::Local,
                simulation_condition: SimulationCondition::WhenVisible,
                z_layer_2d: 0.0,
            },
            properties: vec![
                PropertyDef {
                    id: speed,
                    name: "speed".into(),
                    default: Value::from(3.0f32),
                    exposed: true,
                },
                PropertyDef {
                    id: tint,
                    name: "tint".into(),
                    default: Value::from(Vec4::ONE),
                    exposed: false,
                },
            ],
            texture_slots: vec![TextureSlotDef {
                id: slot,
                name: "noise".into(),
            }],
            nodes: vec![
                GraphNode {
                    id: n1,
                    payload: NodePayload::Expr(ExprNode::Property(speed)),
                    inputs: vec![],
                },
                GraphNode {
                    id: n2,
                    payload: NodePayload::Modifier(ModifierNodeData::Known {
                        type_path: "bevy_hanabi::modifier::velocity::SetVelocitySphereModifier"
                            .into(),
                        config: BTreeMap::new(),
                    }),
                    inputs: vec![InputSlot {
                        name: "speed".into(),
                        default: Value::from(1.0f32).into(),
                    }],
                },
                GraphNode {
                    id: n_image,
                    payload: NodePayload::Expr(ExprNode::Image(ImageBinding::Slot(slot))),
                    inputs: vec![],
                },
                GraphNode {
                    id: n_sample,
                    payload: NodePayload::Expr(ExprNode::TextureSample),
                    inputs: vec![InputSlot {
                        name: "coordinates".into(),
                        default: Value::from(bevy::math::Vec2::ZERO).into(),
                    }],
                },
            ],
            stacks: vec![GraphStack {
                id: stack,
                group: ModifierGroup::Init,
                members: vec![n2],
            }],
            links: vec![
                GraphLink {
                    from: PortRef {
                        node: n1,
                        port: "out".into(),
                    },
                    to: PortRef {
                        node: n2,
                        port: "speed".into(),
                    },
                },
                GraphLink {
                    from: PortRef {
                        node: n_image,
                        port: "out".into(),
                    },
                    to: PortRef {
                        node: n_sample,
                        port: "image".into(),
                    },
                },
            ],
            next_id: 10,
        };

        let asset = EffectGraphAsset {
            version: FORMAT_VERSION,
            graph,
            layout: Some(GraphLayout {
                pan: (10.0, -5.0),
                zoom: 1.25,
                node_pos: vec![(n1, (0.0, 0.0)), (n2, (200.0, 40.0))],
                stack_pos: vec![(stack, (100.0, 300.0))],
            }),
        };

        round_trip(&asset);
    }
}
