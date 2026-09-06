//! Data types of the [`EffectGraph`] authoring model.
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
    math::{UVec2, UVec3, Vec2, Vec3, Vec4},
    reflect::TypePath,
};
use bevy_hanabi::{
    Attribute, BuiltInOperator, CpuValue, Gradient, ScalarType, SimulationCondition,
    SimulationSpace, SlotDimension, SpawnerSettings, Value, ValueType, VectorType,
    graph::expr::{BinaryOperator, TernaryOperator, UnaryOperator},
};
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};

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

/// On-disk schema version stamped into every [`EffectGraphAsset`].
///
/// Bumped only on a breaking change to the schema; additive changes rely on
/// serde field defaults instead of a bump. See [`from_ron_bytes`] for the read
/// path, version gate, and migration ladder.
///
/// Version 2 introduced the multi-emitter [`EffectGraph`]: a v1 file held one
/// bare emitter whose nested header carried `SpawnerSettings` directly; a v2
/// file holds a forest of emitters driven by effect-level [`SourceContext`]s,
/// with `SpawnerSettings` living on a
/// [`SourceKind::CpuSpawner`] instead. See `loader::migrate_v1_to_v2` for the
/// upgrade.
///
/// [`from_ron_bytes`]: crate::from_ron_bytes
pub const FORMAT_VERSION: u32 = 2;

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

/// Identifier of an emitter pipeline within an [`EffectGraph`].
///
/// Drawn from the same global counter as node, stack, property, slot, and
/// source ids so every identity on the canvas is unique regardless of which
/// emitter or context it belongs to. See [`EffectGraph::alloc_emitter_id`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EmitterId(pub NonZeroU32);

impl EmitterId {
    pub fn new(one_based: u32) -> Option<Self> {
        NonZeroU32::new(one_based).map(Self)
    }

    pub fn get(&self) -> u32 {
        self.0.get()
    }
}

/// Identifier of an effect-level spawn source within an [`EffectGraph`].
///
/// Identifies a [`SourceContext`] (a `CpuSpawner` or `GpuEvent`). Drawn from
/// the same global counter as every other id kind so it never collides with a
/// node, stack, property, slot, or emitter id. See
/// [`EffectGraph::alloc_source_id`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SourceId(pub NonZeroU32);

impl SourceId {
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
    /// Reference to a user-defined emitter property, by stable id (not name, so
    /// the property may be renamed without invalidating the reference).
    Property(PropertyId),
    /// A particle attribute read (e.g. position, velocity).
    Attribute(Attribute),
    /// A particle age read with optional lifetime normalization and clamping.
    ///
    /// This is an authoring convenience for the common `age / lifetime`
    /// expression. A disabled `normalized` flag lowers to a plain `age` read,
    /// preserving the behavior of a legacy [`ExprNode::Attribute`] AGE node.
    Age {
        #[serde(default)]
        normalized: bool,
        #[serde(default)]
        clamped: bool,
    },
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
    /// Sample a one-dimensional texture at a floating-point coordinate.
    TextureSample1d,
    /// Sample a texture at given coordinates.
    ///
    /// Inputs are `image` — an image source — and `coordinates` (a `vec2`); the
    /// output is the sampled `vec4`.
    TextureSample2d,
    /// Sample a three-dimensional texture at floating-point coordinates.
    TextureSample3d,
    /// Load one unfiltered texel from a one-dimensional texture.
    TextureLoad1d,
    /// Load one unfiltered texel from a two-dimensional texture.
    TextureLoad2d,
    /// Load one unfiltered texel from a three-dimensional texture.
    TextureLoad3d,
    /// Pick one of several image sources by a runtime index.
    ///
    /// Inputs are `index` (a `u32`) followed by `image0`, `image1`, ... image
    /// sources; the output is the image at the selected index. The image-input
    /// count is link-derived: one empty trailing port is always offered, so
    /// connecting it grows the node and clearing the highest one shrinks it.
    SelectImage { count: u32 },
}

/// Greatest number of image inputs a [`ExprNode::SelectImage`] node exposes.
pub const MAX_SELECT_IMAGE_INPUTS: usize = 16;

/// Port names of a [`ExprNode::SelectImage`] node: the `index` selector first,
/// then up to [`MAX_SELECT_IMAGE_INPUTS`] image inputs. Sliced by image count.
const SELECT_IMAGE_PORTS: [&str; MAX_SELECT_IMAGE_INPUTS + 1] = [
    "index", "image0", "image1", "image2", "image3", "image4", "image5", "image6", "image7",
    "image8", "image9", "image10", "image11", "image12", "image13", "image14", "image15",
];

/// Whether `port` on a [`ExprNode::SelectImage`] node is one of its image
/// inputs (every port but the `index` selector).
pub fn is_select_image_input(port: &str) -> bool {
    port != "index"
}

/// Operator type-schema derivations for an expression node.
///
/// The editor infers a port's type from the value feeding it and flows operator
/// output types up from operands, so these methods are the authoritative source
/// of an operator's ports, image inputs, operand defaults, operand-type
/// constraints, and result type.
impl ExprNode {
    /// Texture dimension read by this expression, if it reads a texture.
    pub fn texture_dimension(&self) -> Option<SlotDimension> {
        match self {
            ExprNode::TextureSample2d | ExprNode::TextureLoad2d => Some(SlotDimension::D2),
            ExprNode::TextureSample1d | ExprNode::TextureLoad1d => Some(SlotDimension::D1),
            ExprNode::TextureSample3d | ExprNode::TextureLoad3d => Some(SlotDimension::D3),
            _ => None,
        }
    }

    /// Operand input ports of this node, in evaluation order.
    ///
    /// Empty for source nodes (literal, property, attribute, built-in), which
    /// take no inputs.
    ///
    /// Names match the editor's established convention so the two derivations
    /// of a node's ports agree with the schema used when baking.
    pub fn input_ports(&self) -> &'static [&'static str] {
        match self {
            ExprNode::Unary(_) | ExprNode::Cast(_) => &["in"],
            ExprNode::Binary(_) => &["lhs", "rhs"],
            ExprNode::Ternary(_) => &["a", "b", "c"],
            ExprNode::TextureSample1d | ExprNode::TextureSample2d | ExprNode::TextureSample3d => {
                &["image", "coordinates"]
            }
            ExprNode::TextureLoad1d | ExprNode::TextureLoad2d | ExprNode::TextureLoad3d => {
                &["image", "coordinates", "mip_level"]
            }
            ExprNode::SelectImage { count } => {
                let n = (*count as usize).min(MAX_SELECT_IMAGE_INPUTS);
                &SELECT_IMAGE_PORTS[..=n]
            }
            ExprNode::Literal(_)
            | ExprNode::Property(_)
            | ExprNode::Attribute(_)
            | ExprNode::Age { .. }
            | ExprNode::ParentAttribute(_)
            | ExprNode::BuiltIn(_)
            | ExprNode::Image(_) => &[],
        }
    }

    /// Whether this node has at least one image-typed input port.
    ///
    /// True for the texture sampler and the image selector; these are the only
    /// expression nodes that consume the editor's image pseudo-type.
    pub fn has_image_input(&self) -> bool {
        matches!(
            self,
            ExprNode::TextureSample1d
                | ExprNode::TextureSample2d
                | ExprNode::TextureSample3d
                | ExprNode::TextureLoad1d
                | ExprNode::TextureLoad2d
                | ExprNode::TextureLoad3d
                | ExprNode::SelectImage { .. }
        )
    }

    /// Whether `port` on this node is an image-typed input port.
    ///
    /// The sampler's `image` operand and every `SelectImage` image input are
    /// image ports; the selector's `index` and all other operands are value
    /// ports.
    pub fn port_is_image(&self, port: &str) -> bool {
        match self {
            expr if expr.texture_dimension().is_some() => port == "image",
            ExprNode::SelectImage { .. } => is_select_image_input(port),
            _ => false,
        }
    }

    /// Neutral inline default seeded for a new operand `port`.
    ///
    /// The editor infers a port's type from the value that feeds it, so a
    /// freshly created node's inline default doubles as the operand's *type*
    /// declaration and must match what the operator's WGSL requires. Most
    /// operators are polymorphic over scalar and vector floats and take a
    /// neutral `f32`; the operators handled explicitly below constrain an
    /// operand to a specific type, so seeding an `f32` there would bake to
    /// invalid code (e.g. `cross(f32, f32)`). An image-typed port carries an
    /// unbound [`ImageBinding`] rather than a literal.
    ///
    /// The constrained cases follow the WGSL builtin, swizzle, and constructor
    /// rules, mirroring naga's math-function overloads
    /// (`MathFunction::overloads`).
    pub fn operand_default(&self, port: &str) -> InputDefault {
        use BinaryOperator as B;
        use UnaryOperator as U;

        if self.port_is_image(port) {
            return ImageBinding::Unbound.into();
        }

        let value = match (self, port) {
            (ExprNode::TextureSample1d, "coordinates") => Value::from(0.0f32),
            (ExprNode::TextureSample2d, "coordinates") => Value::from(Vec2::ZERO),
            (ExprNode::TextureSample3d, "coordinates") => Value::from(Vec3::ZERO),
            (ExprNode::TextureLoad1d, "coordinates" | "mip_level") => Value::from(0u32),
            (ExprNode::TextureLoad2d, "coordinates") => Value::from(UVec2::ZERO),
            (ExprNode::TextureLoad2d, "mip_level") => Value::from(0u32),
            (ExprNode::TextureLoad3d, "coordinates") => Value::from(UVec3::ZERO),
            (ExprNode::TextureLoad3d, "mip_level") => Value::from(0u32),
            (ExprNode::SelectImage { .. }, "index") => Value::from(0u32),
            // Operands constrained to a `vec3`: cross/dot products, the `vec4`
            // constructor's `xyz` operand, `normalize`, and the `.z` swizzle.
            (ExprNode::Binary(B::Cross | B::Dot), _)
            | (ExprNode::Binary(B::Vec4XyzW), "lhs")
            | (ExprNode::Unary(U::Normalize | U::Z), _) => Value::from(Vec3::ZERO),
            // Operands needing at least a `vec2`: the `.x`/`.y` swizzles.
            (ExprNode::Unary(U::X | U::Y), _) => Value::from(Vec2::ZERO),
            // Operands constrained to a `vec4`: the `.w` swizzle and byte packing.
            (ExprNode::Unary(U::W | U::Pack4x8snorm | U::Pack4x8unorm), _) => {
                Value::from(Vec4::ZERO)
            }
            // Operands that must be a `u32` bit pattern: byte unpacking.
            (ExprNode::Unary(U::Unpack4x8snorm | U::Unpack4x8unorm), _) => Value::from(0u32),
            // Operands that must be boolean: the `all`/`any` reductions.
            (ExprNode::Unary(U::All | U::Any), _) => Value::from(false),
            // Everything else is polymorphic over scalar and vector floats.
            _ => Value::from(0.0f32),
        };
        value.into()
    }

    /// Whether every value operand of this node must share a single type.
    ///
    /// True for the operators whose WGSL form requires identically typed
    /// operands, so linking one operand should retype the sibling inline
    /// defaults to match. The multiply and divide operators are excluded
    /// because WGSL broadcasts a scalar against a vector, making a scalar
    /// operand a valid, common choice; the vector constructors and
    /// fixed-type builtins are excluded because their operands carry their
    /// own required types.
    pub fn operands_share_type(&self) -> bool {
        use BinaryOperator as B;
        use TernaryOperator as T;

        matches!(
            self,
            ExprNode::Binary(
                B::Add
                    | B::Sub
                    | B::Min
                    | B::Max
                    | B::Remainder
                    | B::Step
                    | B::Atan2
                    | B::Distance
                    | B::UniformRand
                    | B::NormalRand
                    | B::GreaterThan
                    | B::GreaterThanOrEqual
                    | B::LessThan
                    | B::LessThanOrEqual,
            ) | ExprNode::Ternary(T::Mix | T::Clamp | T::SmoothStep)
        )
    }

    /// Output [`ValueType`] of this operator node, given a resolver for the
    /// type feeding each operand port.
    ///
    /// Covers the applied operators ([`Unary`], [`Binary`], [`Ternary`]);
    /// source nodes carry their own type. Numeric operations are
    /// component-wise and yield their first operand's type, but several
    /// results the naive first-operand rule gets wrong are handled
    /// explicitly: reductions (`dot`, `distance`, `length`, `all`, `any`)
    /// collapse to a scalar, comparisons yield a boolean of the operand's
    /// rank, swizzles extract a component scalar, constructors build a
    /// fixed-width vector, and byte pack/unpack convert between `u32` and
    /// `vec4<f32>`.
    ///
    /// `operand` returns `None` for an image-typed or unresolved port; the
    /// result is `None` when the output type cannot be determined.
    ///
    /// [`ValueType`]: bevy_hanabi::ValueType
    /// [`Unary`]: ExprNode::Unary
    /// [`Binary`]: ExprNode::Binary
    /// [`Ternary`]: ExprNode::Ternary
    pub fn output_value_type(
        &self,
        mut operand: impl FnMut(&str) -> Option<ValueType>,
    ) -> Option<ValueType> {
        use BinaryOperator as B;
        use UnaryOperator as U;

        let f32t = ValueType::Scalar(ScalarType::Float);
        let boolt = ValueType::Scalar(ScalarType::Bool);
        let first = self.input_ports().first().copied().and_then(&mut operand);

        Some(match self {
            // Reductions to a scalar float.
            ExprNode::Binary(B::Dot | B::Distance) | ExprNode::Unary(U::Length) => f32t,
            // Boolean reductions of a vector.
            ExprNode::Unary(U::All | U::Any) => boolt,
            // Byte-pattern conversions.
            ExprNode::Unary(U::Pack4x8snorm | U::Pack4x8unorm) => {
                ValueType::Scalar(ScalarType::Uint)
            }
            ExprNode::Unary(U::Unpack4x8snorm | U::Unpack4x8unorm) => {
                ValueType::Vector(VectorType::VEC4F)
            }
            // Fixed-width vector constructors and the cross product.
            ExprNode::Binary(B::Vec2) => ValueType::Vector(VectorType::VEC2F),
            ExprNode::Ternary(TernaryOperator::Vec3) => ValueType::Vector(VectorType::VEC3F),
            ExprNode::Binary(B::Vec4XyzW) => ValueType::Vector(VectorType::VEC4F),
            ExprNode::Binary(B::Cross) => ValueType::Vector(VectorType::VEC3F),
            // Swizzle: the input vector's component scalar.
            ExprNode::Unary(U::X | U::Y | U::Z | U::W) => match first? {
                ValueType::Vector(v) => ValueType::Scalar(v.elem_type()),
                scalar => scalar,
            },
            // Comparisons: a boolean of the operands' rank.
            ExprNode::Binary(
                B::GreaterThan | B::GreaterThanOrEqual | B::LessThan | B::LessThanOrEqual,
            ) => match first? {
                ValueType::Vector(v) => {
                    ValueType::Vector(VectorType::new(ScalarType::Bool, v.count() as u8))
                }
                ValueType::Scalar(_) => boolt,
                other => other,
            },
            // Everything else is component-wise: the first operand's type.
            _ => return first,
        })
    }
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
    /// the emitter; travels with the document. Stored as an [`AssetPath`] so it
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
    /// A specific image asset with its inspected texture view dimension.
    ///
    /// New selections use this variant so texture reads can reject an
    /// incompatible binding before generating shader code. [`Asset`] remains
    /// for backward compatibility with documents written before dimensions
    /// were recorded and is treated as 2D.
    TypedAsset {
        path: AssetPath<'static>,
        dimension: SlotDimension,
    },
    /// A host-supplied texture slot, referenced by stable id (see
    /// [`TextureSlotDef`]).
    Slot(SlotId),
}

/// A host-supplied texture slot: a stable id and a display name.
///
/// The slot's *sampling index* is its position in
/// [`EmitterGraph::texture_slots`], so reordering the list reassigns indices —
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
    /// Texture view dimension expected at this slot.
    ///
    /// Omitted from older documents, which therefore retain their original
    /// two-dimensional behavior.
    #[serde(default)]
    pub dimension: SlotDimension,
}

/// A `Vec3`-valued gradient (e.g. size over lifetime).
///
/// Texture LUTs lower through expression nodes rather than this config field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GradientVec3 {
    /// Piecewise-linear keyframe gradient stored directly in a modifier.
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

/// A named, editable emitter parameter with a default value.
///
/// The default also fixes its value type. Expression nodes reference it by
/// [`id`] via [`ExprNode::Property`].
///
/// By default a property is *edit-only*: it exists purely as an authoring
/// convenience and every reference is inlined to a literal constant when the
/// graph is baked, so it has no runtime representation or cost. Setting
/// [`exposed`] promotes it to a real runtime property, exported to the
/// emitter's `Module` and overridable per instance via `EffectProperties`.
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

/// One self-contained emitter pipeline and its settings.
///
/// Owns its full emitter state — name, capacity, simulation settings,
/// properties, texture slots, and expression/modifier graph — but none of the
/// inter-emitter topology that drives it. Allocation of its [`id`] and of
/// every node/stack/property/slot id nested inside it is the containing
/// [`EffectGraph`]'s job, and how it is spawned is expressed by a
/// [`SourceLink`] rather than stored here. Diff-friendly and layout-free.
///
/// [`id`]: EmitterGraph::id
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EmitterGraph {
    /// Stable reference identity, unique across the whole [`EffectGraph`].
    pub id: EmitterId,
    pub name: SharedStr,
    pub capacity: u32,
    pub simulation_space: SimulationSpace,
    pub simulation_condition: SimulationCondition,
    pub z_layer_2d: f32,
    pub properties: Vec<PropertyDef>,
    /// Host-supplied texture slots, ordered by sampling index.
    ///
    /// Referenced by [`ImageBinding::Slot`] via stable [`SlotId`] and filled
    /// per-instance by the host game through [`bevy_hanabi::EffectMaterial`].
    /// Asset-bound images are pinned on their source node and do not appear
    /// here. Empty for an emitter with no host-supplied textures.
    #[serde(default)]
    pub texture_slots: Vec<TextureSlotDef>,
    pub nodes: Vec<GraphNode>,
    pub stacks: Vec<GraphStack>,
    pub links: Vec<GraphLink>,
}

impl EmitterGraph {
    /// An empty emitter with default settings and the given id.
    ///
    /// A placeholder for an emitter not yet populated (e.g. a legacy
    /// `EffectAsset` opened before the import path exists). Carries no nodes,
    /// stacks, links, or properties, and is not connected to any source —
    /// callers add a [`SourceContext`] and [`SourceLink`] separately.
    pub fn empty(id: EmitterId) -> Self {
        Self {
            id,
            name: "untitled".into(),
            capacity: 4096,
            simulation_space: SimulationSpace::default(),
            simulation_condition: SimulationCondition::default(),
            z_layer_2d: 0.0,
            properties: Vec::new(),
            texture_slots: Vec::new(),
            nodes: Vec::new(),
            stacks: Vec::new(),
            links: Vec::new(),
        }
    }
}

#[derive(Deserialize)]
struct EmitterGraphData {
    id: EmitterId,
    #[serde(default, deserialize_with = "deserialize_present")]
    header: Option<LegacyEmitterHeader>,
    #[serde(default, deserialize_with = "deserialize_present")]
    name: Option<SharedStr>,
    #[serde(default, deserialize_with = "deserialize_present")]
    capacity: Option<u32>,
    #[serde(default, deserialize_with = "deserialize_present")]
    simulation_space: Option<SimulationSpace>,
    #[serde(default, deserialize_with = "deserialize_present")]
    simulation_condition: Option<SimulationCondition>,
    #[serde(default, deserialize_with = "deserialize_present")]
    z_layer_2d: Option<f32>,
    properties: Vec<PropertyDef>,
    #[serde(default)]
    texture_slots: Vec<TextureSlotDef>,
    nodes: Vec<GraphNode>,
    stacks: Vec<GraphStack>,
    links: Vec<GraphLink>,
}

/// Deserialize a field as `Some(x)` if present and `None` otherwise.
fn deserialize_present<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

#[derive(Deserialize)]
struct LegacyEmitterHeader {
    name: SharedStr,
    capacity: u32,
    simulation_space: SimulationSpace,
    simulation_condition: SimulationCondition,
    z_layer_2d: f32,
}

impl<'de> Deserialize<'de> for EmitterGraph {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let data = EmitterGraphData::deserialize(deserializer)?;
        let header = data.header.as_ref();
        Ok(Self {
            id: data.id,
            name: data
                .name
                .or_else(|| header.map(|h| h.name.clone()))
                .ok_or_else(|| D::Error::missing_field("name"))?,
            capacity: data
                .capacity
                .or_else(|| header.map(|h| h.capacity))
                .ok_or_else(|| D::Error::missing_field("capacity"))?,
            simulation_space: data
                .simulation_space
                .or_else(|| header.map(|h| h.simulation_space))
                .ok_or_else(|| D::Error::missing_field("simulation_space"))?,
            simulation_condition: data
                .simulation_condition
                .or_else(|| header.map(|h| h.simulation_condition))
                .ok_or_else(|| D::Error::missing_field("simulation_condition"))?,
            z_layer_2d: data
                .z_layer_2d
                .or_else(|| header.map(|h| h.z_layer_2d))
                .ok_or_else(|| D::Error::missing_field("z_layer_2d"))?,
            properties: data.properties,
            texture_slots: data.texture_slots,
            nodes: data.nodes,
            stacks: data.stacks,
            links: data.links,
        })
    }
}

impl EmitterGraph {
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
    /// [`texture_slots`]: EmitterGraph::texture_slots
    pub fn texture_slot_index(&self, id: SlotId) -> Option<usize> {
        self.texture_slots.iter().position(|s| s.id == id)
    }
}

/// What kind of spawn source a [`SourceContext`] is.
///
/// The authoring representation of one emitter's spawn input. A CPU source
/// owns that emitter's [`SpawnerSettings`]; a GPU source represents particles
/// spawned by another emitter's events. The source is linked to its emitter
/// through a [`SourceLink`] so it can exist as a visible, temporarily
/// unconnected graph node while editing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SourceKind {
    /// A CPU-driven spawner with its own timing settings.
    ///
    /// The moved-out counterpart of the version-1 header's `spawner` field.
    CpuSpawner { settings: SpawnerSettings },
    /// A GPU spawn-event sink fed by one or more `EmitSpawnEventModifier`
    /// nodes via an [`EventLink`].
    ///
    /// Carries no settings of its own — timing is driven entirely by however
    /// many spawn events its linked `EmitSpawnEventModifier` nodes write.
    GpuEvent,
}

/// An effect-level spawn source: a stable id plus its [`SourceKind`].
///
/// Lives in [`EffectGraph::sources`] and drives at most one emitter through a
/// [`SourceLink`]. A valid completed graph gives every emitter exactly one
/// source; separation here supports explicit source nodes and incomplete
/// mid-edit states, not shared spawning configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceContext {
    /// Stable reference identity, unique across the whole [`EffectGraph`].
    pub id: SourceId,
    pub kind: SourceKind,
}

/// A directed topology link from a spawn source context to the emitter it
/// drives.
///
/// Each source drives at most one emitter, and each emitter accepts at most one
/// source. This one-to-one link is the per-emitter ownership relationship for
/// spawning; the invariants are enforced by edit-time validation rather than
/// this data type so a graph can be temporarily incomplete while editing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceLink {
    pub source: SourceId,
    #[serde(alias = "effect")]
    pub emitter: EmitterId,
}

/// A directed topology link from an `EmitSpawnEventModifier` node's event
/// output to a GPU Event source context's multiple-link input.
///
/// Unlike a [`GraphLink`], an event link is expected to cross emitter
/// boundaries: `node` lives inside the parent emitter's Update stack, while
/// `target` is an effect-level [`SourceContext`] with [`SourceKind::GpuEvent`]
/// that will drive a *different* (child) emitter. `node` is a bare [`NodeId`]
/// rather than an `(EmitterId, NodeId)` pair because node ids are already
/// unique across the whole [`EffectGraph`]; use
/// [`EffectGraph::emitter_owning_node`] to resolve its owning emitter. Multiple
/// event links may target the same context (fan-in); whether they all
/// resolve to one consistent parent emitter is a validation concern, not
/// something this type enforces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EventLink {
    pub node: NodeId,
    pub target: SourceId,
}

/// The complete authored effect: emitters and the topology that drives them.
///
/// Owns the single monotonic id allocator shared by every node, stack,
/// property, slot, emitter, and source id on the canvas, so all of them stay
/// unique regardless of which emitter or context they belong to. Diff-friendly
/// and layout-free — see [`GraphLayout`] for persisted positions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectGraph {
    #[serde(alias = "effects")]
    pub emitters: Vec<EmitterGraph>,
    pub sources: Vec<SourceContext>,
    pub source_links: Vec<SourceLink>,
    pub event_links: Vec<EventLink>,
    /// Next id to hand out; only ever increases. Ids are never recycled so
    /// that links and persisted layout stay valid across edits and reloads.
    pub next_id: u32,
}

impl EffectGraph {
    /// An empty effect graph: no emitters, sources, or links.
    ///
    /// A placeholder for a not-yet-populated document (mirrors
    /// [`EmitterGraph::empty`] one level up).
    pub fn empty() -> Self {
        Self {
            emitters: Vec::new(),
            sources: Vec::new(),
            source_links: Vec::new(),
            event_links: Vec::new(),
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
    /// Drawn from the same counter as every other id kind so they never
    /// collide.
    pub fn alloc_stack_id(&mut self) -> StackId {
        let id = StackId::new(self.next_id).expect("stack id allocator overflow");
        self.next_id += 1;
        id
    }

    /// Mint a fresh, never-before-used [`PropertyId`].
    ///
    /// Drawn from the same counter as every other id kind so they never
    /// collide.
    pub fn alloc_property_id(&mut self) -> PropertyId {
        let id = PropertyId::new(self.next_id).expect("property id allocator overflow");
        self.next_id += 1;
        id
    }

    /// Mint a fresh, never-before-used [`SlotId`].
    ///
    /// Drawn from the same counter as every other id kind so they never
    /// collide.
    pub fn alloc_slot_id(&mut self) -> SlotId {
        let id = SlotId::new(self.next_id).expect("slot id allocator overflow");
        self.next_id += 1;
        id
    }

    /// Mint a fresh, never-before-used [`EmitterId`].
    ///
    /// Drawn from the same counter as every other id kind so they never
    /// collide.
    pub fn alloc_emitter_id(&mut self) -> EmitterId {
        let id = EmitterId::new(self.next_id).expect("emitter id allocator overflow");
        self.next_id += 1;
        id
    }

    /// Mint a fresh, never-before-used [`SourceId`].
    ///
    /// Drawn from the same counter as every other id kind so they never
    /// collide.
    pub fn alloc_source_id(&mut self) -> SourceId {
        let id = SourceId::new(self.next_id).expect("source id allocator overflow");
        self.next_id += 1;
        id
    }

    /// The emitter pipeline with the given id, if any.
    pub fn emitter(&self, id: EmitterId) -> Option<&EmitterGraph> {
        self.emitters.iter().find(|e| e.id == id)
    }

    /// The emitter pipeline with the given id, if any (mutable).
    pub fn emitter_mut(&mut self, id: EmitterId) -> Option<&mut EmitterGraph> {
        self.emitters.iter_mut().find(|e| e.id == id)
    }

    /// The spawn source context with the given id, if any.
    pub fn source(&self, id: SourceId) -> Option<&SourceContext> {
        self.sources.iter().find(|s| s.id == id)
    }

    /// The spawn source context with the given id, if any (mutable).
    pub fn source_mut(&mut self, id: SourceId) -> Option<&mut SourceContext> {
        self.sources.iter_mut().find(|s| s.id == id)
    }

    /// The emitter that contains a node with the given id, if any.
    ///
    /// Node ids are unique across the whole effect graph, so at most one
    /// emitter can own a given [`NodeId`]; this is how an [`EventLink`]'s
    /// `node` resolves back to its parent emitter.
    pub fn emitter_owning_node(&self, node: NodeId) -> Option<EmitterId> {
        self.emitters
            .iter()
            .find(|e| e.node(node).is_some())
            .map(|e| e.id)
    }

    /// The emitter that contains a stack with the given id, if any.
    ///
    /// Stack ids are unique across the whole effect graph, so at most one
    /// emitter can own a given [`StackId`].
    pub fn emitter_owning_stack(&self, stack: StackId) -> Option<EmitterId> {
        self.emitters
            .iter()
            .find(|e| e.stacks.iter().any(|s| s.id == stack))
            .map(|e| e.id)
    }

    /// The source context driving the given emitter, if linked.
    pub fn source_for_emitter(&self, emitter: EmitterId) -> Option<SourceId> {
        self.source_links
            .iter()
            .find(|l| l.emitter == emitter)
            .map(|l| l.source)
    }

    /// The emitter driven by the given source context, if linked.
    pub fn emitter_for_source(&self, source: SourceId) -> Option<EmitterId> {
        self.source_links
            .iter()
            .find(|l| l.source == source)
            .map(|l| l.emitter)
    }

    /// Every spawn-event node linked to the given GPU Event source context.
    pub fn events_for_source(&self, source: SourceId) -> impl Iterator<Item = NodeId> + '_ {
        self.event_links
            .iter()
            .filter(move |l| l.target == source)
            .map(|l| l.node)
    }

    /// The parent emitter that spawns particles into `emitter` via a GPU source
    /// context, if any.
    ///
    /// `None` for a CPU-rooted emitter (driven directly by a
    /// [`SourceKind::CpuSpawner`]) or one with no linked source at all. When
    /// `emitter`'s source is a [`SourceKind::GpuEvent`] fed by emitters from
    /// more than one distinct emitter — an invalid, mid-edit topology (see
    /// `validation::validate_topology`) — the first event node's owning emitter
    /// in [`event_links`] order is returned.
    ///
    /// [`event_links`]: EffectGraph::event_links
    pub fn parent_emitter(&self, emitter: EmitterId) -> Option<EmitterId> {
        let source = self.source_for_emitter(emitter)?;
        match &self.source(source)?.kind {
            SourceKind::CpuSpawner { .. } => None,
            SourceKind::GpuEvent => self
                .events_for_source(source)
                .find_map(|node| self.emitter_owning_node(node)),
        }
    }
}

/// UI layout for a graph: viewport transform plus node/stack/source positions.
///
/// Optional; regenerated by auto-layout when absent. Positions are stored as
/// plain `(x, y)` pairs to keep the schema independent of the math crate.
/// Scoped to the whole [`EffectGraph`] rather than per-emitter: node and stack
/// ids are already unique across every contained emitter, so one flat layout
/// covers the repeated Init/Update/Render stack triplets of any number of
/// emitters without needing an `EmitterId` key, and [`source_pos`] extends the
/// same canvas to the effect-level source contexts.
///
/// [`source_pos`]: GraphLayout::source_pos
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct GraphLayout {
    pub pan: (f64, f64),
    pub zoom: f64,
    pub node_pos: Vec<(NodeId, (f64, f64))>,
    pub stack_pos: Vec<(StackId, (f64, f64))>,
    /// Canvas position of each spawn source context (`CpuSpawner`/`GpuEvent`),
    /// keyed by [`SourceId`].
    ///
    /// Absent (empty) for a graph migrated from a pre-2 file until the editor
    /// places the migrated CPU Spawner; defaults to empty so an older
    /// [`GraphLayout`] without this field still deserializes.
    #[serde(default)]
    pub source_pos: Vec<(SourceId, (f64, f64))>,
}

/// The loadable effect graph asset.
///
/// Holds a schema version, the semantic [`EffectGraph`], and an optional
/// [`GraphLayout`]. This is the canonical edited and persisted unit (each
/// contained emitter's [`bevy_hanabi::EffectAsset`] is a derived bake output of
/// it — see `bake_effect`). As a Bevy [`Asset`] it can be loaded from any asset
/// source — a `.hnb` file is just one of them — and held by handle.
///
/// The schema [`version`] is checked by the asset loader, which rejects
/// versions newer than [`FORMAT_VERSION`] and upgrades older ones through its
/// migration ladder; the writer always stamps [`FORMAT_VERSION`].
///
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
    fn age_expression_defaults_clamping_when_deserializing() {
        let age: ExprNode =
            ron::de::from_str("Age(normalized: true)").expect("deserialize normalized age");
        assert_eq!(
            age,
            ExprNode::Age {
                normalized: true,
                clamped: false,
            }
        );
        round_trip(&ExprNode::Attribute(Attribute::AGE));
        round_trip(&ExprNode::Age {
            normalized: true,
            clamped: true,
        });
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
        let root_emitter = EmitterId::new(10).unwrap();
        let cpu_source = SourceId::new(11).unwrap();
        let n_emit = NodeId::new(12).unwrap();
        let child_emitter = EmitterId::new(13).unwrap();
        let gpu_source = SourceId::new(14).unwrap();

        let root = EmitterGraph {
            id: root_emitter,
            name: "demo".into(),
            capacity: 4096,
            simulation_space: SimulationSpace::Local,
            simulation_condition: SimulationCondition::WhenVisible,
            z_layer_2d: 0.0,
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
                dimension: SlotDimension::D2,
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
                    payload: NodePayload::Expr(ExprNode::TextureSample2d),
                    inputs: vec![InputSlot {
                        name: "coordinates".into(),
                        default: Value::from(bevy::math::Vec2::ZERO).into(),
                    }],
                },
                GraphNode {
                    id: n_emit,
                    payload: NodePayload::Modifier(ModifierNodeData::Known {
                        type_path: "bevy_hanabi::modifier::EmitSpawnEventModifier".into(),
                        config: BTreeMap::new(),
                    }),
                    inputs: vec![],
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
        };

        let child = EmitterGraph::empty(child_emitter);

        let effect_graph = EffectGraph {
            emitters: vec![root, child],
            sources: vec![
                SourceContext {
                    id: cpu_source,
                    kind: SourceKind::CpuSpawner {
                        settings: SpawnerSettings::rate(64.0.into()),
                    },
                },
                SourceContext {
                    id: gpu_source,
                    kind: SourceKind::GpuEvent,
                },
            ],
            source_links: vec![
                SourceLink {
                    source: cpu_source,
                    emitter: root_emitter,
                },
                SourceLink {
                    source: gpu_source,
                    emitter: child_emitter,
                },
            ],
            event_links: vec![EventLink {
                node: n_emit,
                target: gpu_source,
            }],
            next_id: 15,
        };

        let asset = EffectGraphAsset {
            version: FORMAT_VERSION,
            graph: effect_graph,
            layout: Some(GraphLayout {
                pan: (10.0, -5.0),
                zoom: 1.25,
                node_pos: vec![(n1, (0.0, 0.0)), (n2, (200.0, 40.0))],
                stack_pos: vec![(stack, (100.0, 300.0))],
                source_pos: vec![(cpu_source, (-200.0, 300.0)), (gpu_source, (500.0, 0.0))],
            }),
        };

        round_trip(&asset);
    }

    #[test]
    fn renamed_fields_accept_previous_v2_names() {
        let graph: EffectGraph = ron::de::from_str(
            "(effects: [], sources: [], source_links: [], event_links: [], next_id: 1)",
        )
        .expect("deserialize previous effects field");
        assert!(graph.emitters.is_empty());

        let link: SourceLink = ron::de::from_str("(source: (1), effect: (2))")
            .expect("deserialize previous effect field");
        assert_eq!(link.emitter, EmitterId::new(2).unwrap());

        let emitter: EmitterGraph = ron::de::from_str(
            r#"(
                id: (1),
                header: (
                    name: "legacy",
                    capacity: 128,
                    simulation_space: Local,
                    simulation_condition: Always,
                    z_layer_2d: 2.5,
                ),
                properties: [],
                texture_slots: [],
                nodes: [],
                stacks: [],
                links: [],
            )"#,
        )
        .expect("deserialize previous nested emitter header");
        assert_eq!(emitter.name.as_ref(), "legacy");
        assert_eq!(emitter.capacity, 128);
        assert_eq!(emitter.simulation_space, SimulationSpace::Local);
        assert_eq!(emitter.simulation_condition, SimulationCondition::Always);
        assert_eq!(emitter.z_layer_2d, 2.5);
    }

    #[test]
    fn serialization_uses_current_emitter_field_names() {
        let emitter_id = EmitterId::new(1).unwrap();
        let source_id = SourceId::new(2).unwrap();
        let graph = EffectGraph {
            emitters: vec![EmitterGraph::empty(emitter_id)],
            sources: vec![],
            source_links: vec![SourceLink {
                source: source_id,
                emitter: emitter_id,
            }],
            event_links: vec![],
            next_id: 3,
        };

        let ron = ron::ser::to_string(&graph).expect("serialize");
        assert!(ron.contains("emitters:"));
        assert!(ron.contains("emitter:"));
        assert!(ron.contains("name:"));
        assert!(!ron.contains("effects:"));
        assert!(!ron.contains("effect:"));
        assert!(!ron.contains("header:"));
    }

    #[test]
    fn effect_graph_lookup_helpers_resolve_topology() {
        let emitter_id = EmitterId::new(1).unwrap();
        let node_id = NodeId::new(2).unwrap();
        let stack_id = StackId::new(3).unwrap();
        let source_id = SourceId::new(4).unwrap();

        let mut emitter = EmitterGraph::empty(emitter_id);
        emitter.nodes.push(GraphNode {
            id: node_id,
            payload: NodePayload::Modifier(ModifierNodeData::Known {
                type_path: "bevy_hanabi::modifier::EmitSpawnEventModifier".into(),
                config: BTreeMap::new(),
            }),
            inputs: vec![],
        });
        emitter.stacks.push(GraphStack {
            id: stack_id,
            group: ModifierGroup::Update,
            members: vec![node_id],
        });

        let effect_graph = EffectGraph {
            emitters: vec![emitter],
            sources: vec![SourceContext {
                id: source_id,
                kind: SourceKind::GpuEvent,
            }],
            source_links: vec![SourceLink {
                source: source_id,
                emitter: emitter_id,
            }],
            event_links: vec![EventLink {
                node: node_id,
                target: source_id,
            }],
            next_id: 5,
        };

        assert_eq!(effect_graph.emitter_owning_node(node_id), Some(emitter_id));
        assert_eq!(
            effect_graph.emitter_owning_stack(stack_id),
            Some(emitter_id)
        );
        assert_eq!(effect_graph.source_for_emitter(emitter_id), Some(source_id));
        assert_eq!(effect_graph.emitter_for_source(source_id), Some(emitter_id));
        assert_eq!(
            effect_graph
                .events_for_source(source_id)
                .collect::<Vec<_>>(),
            vec![node_id]
        );
        assert_eq!(
            effect_graph.emitter_owning_node(NodeId::new(99).unwrap()),
            None
        );
    }

    #[test]
    fn effect_graph_parent_emitter_resolves_gpu_chain() {
        let parent_id = EmitterId::new(1).unwrap();
        let child_id = EmitterId::new(2).unwrap();
        let event_node = NodeId::new(3).unwrap();
        let cpu_source = SourceId::new(4).unwrap();
        let gpu_source = SourceId::new(5).unwrap();

        let mut parent = EmitterGraph::empty(parent_id);
        parent.nodes.push(GraphNode {
            id: event_node,
            payload: NodePayload::Modifier(ModifierNodeData::Known {
                type_path: "bevy_hanabi::modifier::EmitSpawnEventModifier".into(),
                config: BTreeMap::new(),
            }),
            inputs: vec![],
        });

        let effect_graph = EffectGraph {
            emitters: vec![parent, EmitterGraph::empty(child_id)],
            sources: vec![
                SourceContext {
                    id: cpu_source,
                    kind: SourceKind::CpuSpawner {
                        settings: SpawnerSettings::default(),
                    },
                },
                SourceContext {
                    id: gpu_source,
                    kind: SourceKind::GpuEvent,
                },
            ],
            source_links: vec![
                SourceLink {
                    source: cpu_source,
                    emitter: parent_id,
                },
                SourceLink {
                    source: gpu_source,
                    emitter: child_id,
                },
            ],
            event_links: vec![EventLink {
                node: event_node,
                target: gpu_source,
            }],
            next_id: 6,
        };

        // The CPU-rooted parent has no parent of its own.
        assert_eq!(effect_graph.parent_emitter(parent_id), None);
        // The GPU-driven child resolves back to the emitter owning its emitter.
        assert_eq!(effect_graph.parent_emitter(child_id), Some(parent_id));
    }

    #[test]
    fn effect_graph_allocator_never_reuses_ids_across_kinds() {
        let mut effect_graph = EffectGraph::empty();
        let n = effect_graph.alloc_node_id();
        let s = effect_graph.alloc_stack_id();
        let p = effect_graph.alloc_property_id();
        let slot = effect_graph.alloc_slot_id();
        let emitter = effect_graph.alloc_emitter_id();
        let source = effect_graph.alloc_source_id();

        assert_eq!(n.get(), 1);
        assert_eq!(s.get(), 2);
        assert_eq!(p.get(), 3);
        assert_eq!(slot.get(), 4);
        assert_eq!(emitter.get(), 5);
        assert_eq!(source.get(), 6);
        assert_eq!(effect_graph.next_id, 7);
    }

    #[test]
    fn expr_node_ports() {
        assert_eq!(
            ExprNode::Literal(Value::from(1.0f32)).input_ports(),
            &[] as &[&str]
        );
        assert_eq!(ExprNode::Unary(UnaryOperator::Abs).input_ports(), &["in"]);
        assert_eq!(
            ExprNode::Binary(BinaryOperator::Add).input_ports(),
            &["lhs", "rhs"]
        );
        assert_eq!(
            ExprNode::Image(ImageBinding::Unbound).input_ports(),
            &[] as &[&str]
        );
        assert_eq!(
            ExprNode::TextureSample2d.input_ports(),
            &["image", "coordinates"]
        );
        assert_eq!(
            ExprNode::TextureLoad3d.input_ports(),
            &["image", "coordinates", "mip_level"]
        );
        assert_eq!(
            ExprNode::SelectImage { count: 1 }.input_ports(),
            &["index", "image0"]
        );
        assert_eq!(
            ExprNode::SelectImage { count: 3 }.input_ports(),
            &["index", "image0", "image1", "image2"]
        );
    }

    #[test]
    fn select_image_inputs_are_image_typed() {
        let n = ExprNode::SelectImage { count: 2 };
        assert!(n.has_image_input());
        assert!(!n.port_is_image("index"));
        assert!(n.port_is_image("image0"));
        assert!(n.port_is_image("image1"));
        assert!(ExprNode::TextureSample2d.port_is_image("image"));
        assert!(!ExprNode::TextureSample2d.port_is_image("coordinates"));
        assert_eq!(
            ExprNode::TextureSample1d.texture_dimension(),
            Some(SlotDimension::D1)
        );
        assert_eq!(
            ExprNode::TextureSample3d.texture_dimension(),
            Some(SlotDimension::D3)
        );
        assert_eq!(
            ExprNode::TextureLoad2d.texture_dimension(),
            Some(SlotDimension::D2)
        );
    }

    #[test]
    fn operand_defaults_match_operator_type() {
        use bevy::math::{Vec2, Vec3, Vec4};

        let vt = |n: &ExprNode, p: &str| n.operand_default(p).as_value().map(|v| v.value_type());
        let f32t = Some(ValueType::Scalar(ScalarType::Float));

        // Polymorphic arithmetic keeps the neutral scalar.
        assert_eq!(vt(&ExprNode::Binary(BinaryOperator::Add), "lhs"), f32t);
        // Cross and dot force both operands to a vector.
        assert_eq!(
            vt(&ExprNode::Binary(BinaryOperator::Cross), "rhs"),
            Some(Value::from(Vec3::ZERO).value_type())
        );
        assert_eq!(
            vt(&ExprNode::Binary(BinaryOperator::Dot), "lhs"),
            Some(Value::from(Vec3::ZERO).value_type())
        );
        // The vec4 constructor takes a vec3 xyz and a scalar w.
        assert_eq!(
            vt(&ExprNode::Binary(BinaryOperator::Vec4XyzW), "lhs"),
            Some(Value::from(Vec3::ZERO).value_type())
        );
        assert_eq!(vt(&ExprNode::Binary(BinaryOperator::Vec4XyzW), "rhs"), f32t);
        // Swizzles need a wide-enough vector; packing a vec4; unpacking a u32.
        assert_eq!(
            vt(&ExprNode::Unary(UnaryOperator::X), "in"),
            Some(Value::from(Vec2::ZERO).value_type())
        );
        assert_eq!(
            vt(&ExprNode::Unary(UnaryOperator::W), "in"),
            Some(Value::from(Vec4::ZERO).value_type())
        );
        assert_eq!(
            vt(&ExprNode::Unary(UnaryOperator::Pack4x8snorm), "in"),
            Some(Value::from(Vec4::ZERO).value_type())
        );
        assert_eq!(
            vt(&ExprNode::Unary(UnaryOperator::Unpack4x8snorm), "in"),
            Some(ValueType::Scalar(ScalarType::Uint))
        );
        // all/any operate on booleans.
        assert_eq!(
            vt(&ExprNode::Unary(UnaryOperator::All), "in"),
            Some(ValueType::Scalar(ScalarType::Bool))
        );
        // The sampler's image port carries a binding, not a literal.
        assert_eq!(
            ExprNode::TextureLoad1d
                .operand_default("coordinates")
                .as_value(),
            Some(Value::from(0u32))
        );
        assert!(
            ExprNode::TextureSample2d
                .operand_default("image")
                .as_image()
                .is_some()
        );
        assert_eq!(
            ExprNode::TextureLoad3d
                .operand_default("coordinates")
                .as_value(),
            Some(Value::from(UVec3::ZERO))
        );
    }

    #[test]
    fn output_types_are_operator_aware() {
        let vec3 = ValueType::Vector(VectorType::VEC3F);
        let vec3_operand = |_: &str| Some(vec3);

        // dot collapses vectors to a scalar; cross stays a vec3.
        assert_eq!(
            ExprNode::Binary(BinaryOperator::Dot).output_value_type(vec3_operand),
            Some(ValueType::Scalar(ScalarType::Float))
        );
        assert_eq!(
            ExprNode::Binary(BinaryOperator::Cross).output_value_type(vec3_operand),
            Some(vec3)
        );
        // A swizzle extracts the input vector's component scalar.
        assert_eq!(
            ExprNode::Unary(UnaryOperator::X).output_value_type(vec3_operand),
            Some(ValueType::Scalar(ScalarType::Float))
        );
        // A comparison yields a boolean of the operands' rank.
        assert_eq!(
            ExprNode::Binary(BinaryOperator::GreaterThan).output_value_type(vec3_operand),
            Some(ValueType::Vector(VectorType::new(ScalarType::Bool, 3)))
        );
        // Constructors and pack/unpack have fixed result types.
        assert_eq!(
            ExprNode::Ternary(TernaryOperator::Vec3)
                .output_value_type(|_| Some(ValueType::Scalar(ScalarType::Float))),
            Some(vec3)
        );
        assert_eq!(
            ExprNode::Unary(UnaryOperator::Unpack4x8snorm)
                .output_value_type(|_| Some(ValueType::Scalar(ScalarType::Uint))),
            Some(ValueType::Vector(VectorType::VEC4F))
        );
        // Component-wise arithmetic keeps the first operand's type.
        assert_eq!(
            ExprNode::Binary(BinaryOperator::Add).output_value_type(vec3_operand),
            Some(vec3)
        );
    }

    #[test]
    fn share_type_operators_are_element_wise_only() {
        use BinaryOperator as B;
        use TernaryOperator as T;

        // Element-wise operators requiring matching operands opt in.
        for op in [B::Add, B::Sub, B::Min, B::Max, B::Distance, B::GreaterThan] {
            assert!(ExprNode::Binary(op).operands_share_type());
        }
        for op in [T::Mix, T::Clamp, T::SmoothStep] {
            assert!(ExprNode::Ternary(op).operands_share_type());
        }
        // Multiply and divide broadcast a scalar, so they opt out.
        assert!(!ExprNode::Binary(B::Mul).operands_share_type());
        assert!(!ExprNode::Binary(B::Div).operands_share_type());
        // Constructors and fixed-type builtins carry their own operand types.
        assert!(!ExprNode::Binary(B::Cross).operands_share_type());
        assert!(!ExprNode::Binary(B::Vec2).operands_share_type());
        assert!(!ExprNode::Ternary(T::Vec3).operands_share_type());
    }
}
