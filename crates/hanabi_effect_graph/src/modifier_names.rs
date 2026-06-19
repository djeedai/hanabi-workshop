//! User-facing display names for `bevy_hanabi` modifiers.
//!
//! Hanabi 0.18 exposes no display-name API — only `Reflect`'s type
//! path. We keep a curated table for every built-in modifier so the UI
//! reads naturally ("Set Position (Sphere)" rather than
//! `SetPositionSphereModifier`). Anything not in the table (custom
//! user modifiers, future Hanabi additions) falls back to a generic
//! CamelCase prettifier that splits the type name into words and
//! drops the trailing `Modifier` suffix.
//!
//! Callers pass the short type name (i.e. `reflect_short_type_path()`
//! on the modifier).

use std::borrow::Cow;

/// Curated display name for a modifier, by short type name.
///
/// The short type name is e.g. `"SetPositionSphereModifier"`. Falls back to a
/// CamelCase prettifier for unknown types.
pub fn display_name_for_type(short_type_name: &str) -> Cow<'static, str> {
    if let Some(name) = builtin_display_name(short_type_name) {
        Cow::Borrowed(name)
    } else {
        Cow::Owned(prettify_camel_case(short_type_name))
    }
}

/// Like [`display_name_for_type`], but for a `dyn bevy_hanabi::Modifier`.
///
/// Special-cases instance-dependent names — e.g. `SetAttributeModifier` is
/// rendered as `"Set Attribute (lifetime)"` so users don't have to click each
/// row to discover which attribute it targets.
pub fn display_name_for_modifier(m: &dyn bevy_hanabi::Modifier) -> Cow<'static, str> {
    let short = m.as_reflect().reflect_short_type_path();
    if short == "SetAttributeModifier"
        && let Some(sam) = m
            .as_reflect()
            .downcast_ref::<bevy_hanabi::SetAttributeModifier>()
    {
        return Cow::Owned(format!("Set Attribute ({})", sam.attribute.name()));
    }
    display_name_for_type(short)
}

/// Curated table for every built-in `bevy_hanabi` 0.18 modifier.
///
/// Returns `None` for unrecognized types so callers can fall back.
///
/// Note: the section dividers below group entries by *purpose*, not by
/// `ModifierContext` — most of these modifiers report more than one context
/// (e.g. `SetAttributeModifier` is valid in both Init and Update), so grouping
/// by "panel section" would be misleading.
fn builtin_display_name(short: &str) -> Option<&'static str> {
    Some(match short {
        // Position
        "SetPositionCircleModifier" => "Set Position (Circle)",
        "SetPositionSphereModifier" => "Set Position (Sphere)",
        "SetPositionCone3dModifier" => "Set Position (Cone)",
        // Velocity
        "SetVelocityCircleModifier" => "Set Velocity (Circle)",
        "SetVelocitySphereModifier" => "Set Velocity (Sphere)",
        "SetVelocityTangentModifier" => "Set Velocity (Tangent)",
        // Generic attribute writes
        "SetAttributeModifier" => "Set Attribute",
        "InheritAttributeModifier" => "Inherit Attribute",
        // Forces
        "AccelModifier" => "Acceleration",
        "RadialAccelModifier" => "Radial Acceleration",
        "TangentAccelModifier" => "Tangential Acceleration",
        "LinearDragModifier" => "Linear Drag",
        "ConformToSphereModifier" => "Conform To Sphere",
        // Lifetime / culling
        "KillSphereModifier" => "Kill (Sphere)",
        "KillAabbModifier" => "Kill (AABB)",
        // Events
        "EmitSpawnEventModifier" => "Emit Spawn Event",
        // Color
        "SetColorModifier" => "Set Color",
        "ColorOverLifetimeModifier" => "Color Over Lifetime",
        // Size
        "SetSizeModifier" => "Set Size",
        "SizeOverLifetimeModifier" => "Size Over Lifetime",
        "ScreenSpaceSizeModifier" => "Screen-Space Size",
        // Texture / sprite
        "ParticleTextureModifier" => "Particle Texture",
        "FlipbookModifier" => "Flipbook",
        // Orientation / misc render
        "OrientModifier" => "Orient",
        "RoundModifier" => "Round",
        _ => return None,
    })
}

/// Best-effort prettifier for unknown modifier types.
///
/// Drops a trailing `Modifier` suffix and splits remaining CamelCase / digit
/// boundaries into space-separated words.
fn prettify_camel_case(name: &str) -> String {
    let trimmed = name.strip_suffix("Modifier").unwrap_or(name);
    if trimmed.is_empty() {
        return name.to_string();
    }

    let mut out = String::with_capacity(trimmed.len() + 4);
    let chars: Vec<char> = trimmed.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if i > 0 {
            let prev = chars[i - 1];
            let next = chars.get(i + 1).copied();
            // Insert a space at:
            //  - lower/digit → upper boundary  (e.g. "Set|Color")
            //  - upper → upper followed by lower (end of an acronym, e.g. "AABB|Box" —
            //    though we don't have one, this keeps "HTTPRequest" → "HTTP Request"
            //    correct).
            let boundary = (!prev.is_uppercase() && c.is_uppercase())
                || (prev.is_uppercase()
                    && c.is_uppercase()
                    && matches!(next, Some(n) if n.is_lowercase()));
            if boundary {
                out.push(' ');
            }
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curated_names() {
        assert_eq!(
            display_name_for_type("SetPositionSphereModifier"),
            "Set Position (Sphere)"
        );
        assert_eq!(
            display_name_for_type("ColorOverLifetimeModifier"),
            "Color Over Lifetime"
        );
        assert_eq!(display_name_for_type("KillAabbModifier"), "Kill (AABB)");
    }

    #[test]
    fn fallback_prettifies_unknown() {
        assert_eq!(display_name_for_type("MyCustomModifier"), "My Custom");
        assert_eq!(display_name_for_type("FooBar"), "Foo Bar");
        assert_eq!(display_name_for_type("HTTPRequestModifier"), "HTTP Request");
    }
}
