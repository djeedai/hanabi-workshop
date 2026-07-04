//! Reflection-derived field schema for modifier nodes.
//!
//! A modifier struct's fields fall into three roles in the graph model, and the
//! split is derived **once** from the type's [`TypeInfo`] (not per edit):
//!
//! - [`FieldRole::ExprPort`] — an `ExprHandle` (or `Option<ExprHandle>`) field
//!   becomes a connectable input port; its value comes from a link or an inline
//!   default, never from the modifier instance.
//! - [`FieldRole::Texture`] — an `ExprHandle` field that is *semantically* a
//!   texture binding (Hanabi types it as a slot-index expression). It becomes a
//!   connectable, image-typed port that accepts an image source, like the
//!   texture-sampling expression node.
//! - [`FieldRole::Config`] — every other field is editable configuration,
//!   classified to the [`EditValue`] variant it maps to.
//!
//! Reflection alone cannot tell that an `ExprHandle` is a texture (there is no
//! distinct type), so that one case is supplied by a small hint table keyed on
//! the modifier type and field name. Everything else is purely structural.
//!
//! [`EditValue`]: super::model::EditValue

use bevy::{
    math::{Vec2, Vec3, Vec4},
    reflect::{TypeInfo, Typed},
};
use bevy_hanabi::{ScalarType, Value, ValueType};

use super::model::SharedStr;

/// Name of the single output port every node exposes.
///
/// Expression nodes produce one value; modifier nodes expose their stack
/// membership through this port for uniform addressing in [`PortRef`].
///
/// [`PortRef`]: super::model::PortRef
pub const OUTPUT_PORT: &str = "out";

/// A neutral zero literal of value type `ty`, for seeding an operand default.
///
/// Returns `None` for types the editor does not seed as inline literals (matrix
/// or non-float vector types), leaving such operands at their existing default.
pub fn value_type_zero(ty: ValueType) -> Option<Value> {
    Some(match ty {
        ValueType::Scalar(ScalarType::Float) => Value::from(0.0f32),
        ValueType::Scalar(ScalarType::Int) => Value::from(0i32),
        ValueType::Scalar(ScalarType::Uint) => Value::from(0u32),
        ValueType::Scalar(ScalarType::Bool) => Value::from(false),
        ValueType::Vector(v) => match (v.elem_type(), v.count()) {
            (ScalarType::Float, 2) => Value::from(Vec2::ZERO),
            (ScalarType::Float, 3) => Value::from(Vec3::ZERO),
            (ScalarType::Float, 4) => Value::from(Vec4::ZERO),
            _ => return None,
        },
        _ => return None,
    })
}

/// Which [`EditValue`] variant a non-expression config field maps to.
///
/// [`EditValue`]: super::model::EditValue
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigKind {
    Bool,
    U32,
    /// A scalar or vector constant carried as a Hanabi `Value`.
    Scalar,
    UVec2,
    Attribute,
    CpuVec3,
    CpuVec4,
    Gradient3,
    Gradient4,
    /// A data-less enum (active variant by name).
    Enum,
    /// A bitflags newtype over an integer.
    Flags,
    /// A type not modeled first-class; round-tripped as a RON fragment.
    Raw,
}

/// The role a modifier struct field plays in the graph model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldRole {
    /// An expression input rendered as a connectable port. `optional` is `true`
    /// for an `Option<ExprHandle>` field (the port may be left unconnected with
    /// no inline default).
    ExprPort { optional: bool },
    /// An `ExprHandle` field that is semantically a texture binding. Rendered
    /// as a connectable, image-typed port that accepts an image source; it
    /// evaluates to a slot index at bake time.
    Texture,
    /// An editable configuration value of the given kind.
    Config(ConfigKind),
}

/// One field of a modifier, with its reflected name, type path, and role.
#[derive(Debug, Clone)]
pub struct FieldSchema {
    pub name: SharedStr,
    pub type_path: &'static str,
    pub role: FieldRole,
}

/// The classified field layout of one modifier type.
///
/// Built from reflection and cached by callers; the field order matches the
/// struct's declaration order.
#[derive(Debug, Clone)]
pub struct ModifierSchema {
    pub type_path: &'static str,
    pub fields: Vec<FieldSchema>,
}

impl ModifierSchema {
    /// Fields that are expression inputs (ports), in declaration order.
    ///
    /// Includes texture-slot fields, which are image-typed ports accepting an
    /// image source.
    pub fn ports(&self) -> impl Iterator<Item = &FieldSchema> {
        self.fields
            .iter()
            .filter(|f| matches!(f.role, FieldRole::ExprPort { .. } | FieldRole::Texture))
    }

    /// Fields that are editable configuration.
    pub fn config(&self) -> impl Iterator<Item = &FieldSchema> {
        self.fields
            .iter()
            .filter(|f| matches!(f.role, FieldRole::Config(_)))
    }
}

/// Build the [`ModifierSchema`] for a modifier type `T`.
///
/// Built from `T`'s static type info. Returns `None` if `T` does not reflect as
/// a struct.
pub fn modifier_schema_of<T: Typed>() -> Option<ModifierSchema> {
    modifier_schema(T::type_info())
}

/// Build a [`ModifierSchema`] from a modifier type's [`TypeInfo`].
///
/// Returns `None` for non-struct types (no modifier in Hanabi is a tuple/enum).
pub fn modifier_schema(info: &TypeInfo) -> Option<ModifierSchema> {
    let TypeInfo::Struct(s) = info else {
        return None;
    };
    let type_path = s.type_path();
    let fields = s
        .iter()
        .map(|f| FieldSchema {
            name: SharedStr::from(f.name()),
            type_path: f.type_path(),
            role: classify_field(type_path, f.name(), f.type_path(), f.type_info()),
        })
        .collect();
    Some(ModifierSchema { type_path, fields })
}

/// Classify one field into its [`FieldRole`].
fn classify_field(
    modifier_path: &str,
    field_name: &str,
    field_path: &str,
    field_info: Option<&TypeInfo>,
) -> FieldRole {
    if is_expr_handle(field_path) && is_texture_field(modifier_path, field_name) {
        return FieldRole::Texture;
    }
    if let Some(optional) = expr_port_kind(field_path) {
        return FieldRole::ExprPort { optional };
    }
    FieldRole::Config(config_kind(field_path, field_info))
}

/// The last path segment of a type path, ignoring generic arguments.
///
/// E.g. `bevy_hanabi::CpuValue<glam::Vec4>` → `CpuValue`.
fn base_name(path: &str) -> &str {
    let head = path.split('<').next().unwrap_or(path);
    head.rsplit("::").next().unwrap_or(head)
}

fn is_expr_handle(path: &str) -> bool {
    base_name(path) == "ExprHandle"
}

/// Returns `Some(optional)` for an expression-input field.
///
/// `optional` marks an `Option<ExprHandle>`.
fn expr_port_kind(path: &str) -> Option<bool> {
    if is_expr_handle(path) {
        Some(false)
    } else if base_name(path) == "Option" && path.contains("ExprHandle") {
        Some(true)
    } else {
        None
    }
}

/// Hint: an `ExprHandle` field that should be treated as a texture binding.
///
/// Hanabi 0.18 has no distinct texture type, so this is keyed by name.
fn is_texture_field(modifier_path: &str, field_name: &str) -> bool {
    base_name(modifier_path) == "ParticleTextureModifier" && field_name == "texture_slot"
}

/// A single named bit of a bitflags config type.
#[derive(Debug, Clone)]
pub struct FlagDef {
    pub name: &'static str,
    pub bits: u64,
}

/// The atomic (single-bit) flags of a bitflags config type, by reflect path.
///
/// A bitflags newtype's named constants are associated consts, not reflected
/// variants, so they can't be enumerated through reflection the way enum
/// variants can. Known flag types are therefore listed here, sourcing their
/// bit values from the upstream constants so they can't drift. The composite
/// aliases (e.g. `RGB`, `RGBA`) are intentionally omitted — only the
/// independently-toggleable bits are returned. Empty for an unknown type.
pub fn flag_defs(type_path: &str) -> Vec<FlagDef> {
    if base_name(type_path) == "ColorBlendMask" {
        use bevy_hanabi::ColorBlendMask;
        return vec![
            FlagDef {
                name: "R",
                bits: ColorBlendMask::R.bits() as u64,
            },
            FlagDef {
                name: "G",
                bits: ColorBlendMask::G.bits() as u64,
            },
            FlagDef {
                name: "B",
                bits: ColorBlendMask::B.bits() as u64,
            },
            FlagDef {
                name: "A",
                bits: ColorBlendMask::A.bits() as u64,
            },
        ];
    }
    Vec::new()
}

fn config_kind(path: &str, info: Option<&TypeInfo>) -> ConfigKind {
    // Match known wrapper types by name first: some of them (notably
    // `CpuValue<T>`) reflect as enums, so a blanket enum check would mis-classify
    // them. Only types we don't recognize fall back to the generic enum probe.
    match base_name(path) {
        "bool" => ConfigKind::Bool,
        "u32" => ConfigKind::U32,
        "UVec2" => ConfigKind::UVec2,
        "f32" | "i32" | "i64" | "u64" | "Vec2" | "Vec3" | "Vec4" | "IVec2" | "IVec3" | "IVec4"
        | "UVec3" | "UVec4" => ConfigKind::Scalar,
        "Attribute" => ConfigKind::Attribute,
        "CpuValue" if path.contains("Vec3") => ConfigKind::CpuVec3,
        "CpuValue" if path.contains("Vec4") => ConfigKind::CpuVec4,
        "Gradient" if path.contains("Vec3") => ConfigKind::Gradient3,
        "Gradient" if path.contains("Vec4") => ConfigKind::Gradient4,
        "ColorBlendMask" => ConfigKind::Flags,
        _ if matches!(info, Some(TypeInfo::Enum(_))) => ConfigKind::Enum,
        _ => ConfigKind::Raw,
    }
}

#[cfg(test)]
mod tests {
    use bevy_hanabi::{
        ColorOverLifetimeModifier, ConformToSphereModifier, ParticleTextureModifier,
        SetColorModifier, SetPositionSphereModifier, SizeOverLifetimeModifier, Value,
    };

    use super::*;

    fn role_of<'a>(schema: &'a ModifierSchema, name: &str) -> &'a FieldRole {
        &schema
            .fields
            .iter()
            .find(|f| &*f.name == name)
            .unwrap_or_else(|| panic!("field {name} not found in {}", schema.type_path))
            .role
    }

    #[test]
    fn set_color_modifier_fields() {
        let s = modifier_schema_of::<SetColorModifier>().unwrap();
        // color: CpuValue<Vec4>, blend: ColorBlendMode (enum), mask: ColorBlendMask
        // (flags)
        assert_eq!(
            *role_of(&s, "color"),
            FieldRole::Config(ConfigKind::CpuVec4)
        );
        assert_eq!(*role_of(&s, "blend"), FieldRole::Config(ConfigKind::Enum));
        assert_eq!(*role_of(&s, "mask"), FieldRole::Config(ConfigKind::Flags));
        assert_eq!(s.ports().count(), 0);
    }

    #[test]
    fn position_sphere_ports_and_enum() {
        let s = modifier_schema_of::<SetPositionSphereModifier>().unwrap();
        assert_eq!(
            *role_of(&s, "center"),
            FieldRole::ExprPort { optional: false }
        );
        assert_eq!(
            *role_of(&s, "radius"),
            FieldRole::ExprPort { optional: false }
        );
        assert_eq!(
            *role_of(&s, "dimension"),
            FieldRole::Config(ConfigKind::Enum)
        );
        assert_eq!(s.ports().count(), 2);
    }

    #[test]
    fn texture_slot_is_a_texture_port() {
        let s = modifier_schema_of::<ParticleTextureModifier>().unwrap();
        assert_eq!(*role_of(&s, "texture_slot"), FieldRole::Texture);
        assert_eq!(
            *role_of(&s, "sample_mapping"),
            FieldRole::Config(ConfigKind::Enum)
        );
        // The texture field is a connectable port, not config.
        assert_eq!(s.ports().count(), 1);
        assert_eq!(s.ports().next().unwrap().name.as_ref(), "texture_slot");
    }

    #[test]
    fn optional_expr_handle_is_optional_port() {
        let s = modifier_schema_of::<ConformToSphereModifier>().unwrap();
        assert_eq!(
            *role_of(&s, "shell_half_thickness"),
            FieldRole::ExprPort { optional: true }
        );
        assert_eq!(
            *role_of(&s, "origin"),
            FieldRole::ExprPort { optional: false }
        );
    }

    #[test]
    fn gradients_and_bool() {
        let s = modifier_schema_of::<SizeOverLifetimeModifier>().unwrap();
        assert_eq!(
            *role_of(&s, "gradient"),
            FieldRole::Config(ConfigKind::Gradient3)
        );
        assert_eq!(
            *role_of(&s, "screen_space_size"),
            FieldRole::Config(ConfigKind::Bool)
        );

        let c = modifier_schema_of::<ColorOverLifetimeModifier>().unwrap();
        assert_eq!(
            *role_of(&c, "gradient"),
            FieldRole::Config(ConfigKind::Gradient4)
        );
    }

    #[test]
    fn value_type_zero_covers_seedable_types() {
        use bevy::math::{Vec2, Vec3, Vec4};

        let cases = [
            Value::from(0.0f32),
            Value::from(0i32),
            Value::from(0u32),
            Value::from(false),
            Value::from(Vec2::ZERO),
            Value::from(Vec3::ZERO),
            Value::from(Vec4::ZERO),
        ];
        for expected in cases {
            assert_eq!(value_type_zero(expected.value_type()), Some(expected));
        }
    }
}
