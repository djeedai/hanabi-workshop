//! Modifier list mutation primitives.
//!
//! Phase 5b — adding / removing / reordering modifiers in an
//! [`EffectAsset`]. The asset's three modifier vectors are private
//! and `#[reflect(ignore)]`, so we cannot mutate them in place.
//! Instead, every mutation rebuilds a new `EffectAsset` (preserving
//! the existing `Module` arena — and therefore every `ExprHandle`)
//! and overwrites the slot in `Assets<EffectAsset>`.
//!
//! Catalog: each [`AddModifierKind`] variant maps to a small factory
//! that allocates fresh literals into the canonical `Module` and
//! returns a boxed modifier. The factory is intentionally minimal —
//! deeper modifier-creation flows (gradients, textures, attribute
//! pickers) are out of scope for the first add/remove/reorder pass.

use bevy::math::{Vec3, Vec4};
use bevy_hanabi::{
    AccelModifier, Attribute, BoxedModifier, EffectAsset, LinearDragModifier,
    ModifierContext, Module, OrientMode, OrientModifier, RenderModifier, SetAttributeModifier,
    SetColorModifier, SetPositionSphereModifier, SetSizeModifier, SetVelocitySphereModifier,
    ShapeDimension,
};

use crate::document::ModifierGroup;

/// Owned, clonable wrapper around either a plain modifier or a render
/// modifier. We need the `Render` discriminant because adding a render
/// modifier goes through the dedicated `add_render_modifier` builder
/// path; an `as_render()` downcast on a `BoxedModifier` would lose the
/// concrete type.
pub enum BoxedAnyModifier {
    Plain(BoxedModifier),
    Render(Box<dyn RenderModifier>),
}

impl Clone for BoxedAnyModifier {
    fn clone(&self) -> Self {
        match self {
            Self::Plain(m) => Self::Plain(m.boxed_clone()),
            Self::Render(m) => Self::Render(m.boxed_render_clone()),
        }
    }
}

impl std::fmt::Debug for BoxedAnyModifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (kind, path) = match self {
            Self::Plain(m) => ("Plain", m.as_reflect().reflect_short_type_path()),
            Self::Render(m) => ("Render", m.as_modifier().as_reflect().reflect_short_type_path()),
        };
        write!(f, "BoxedAnyModifier::{kind}({path})")
    }
}

impl BoxedAnyModifier {
    pub fn short_type_name(&self) -> &str {
        match self {
            Self::Plain(m) => m.as_reflect().reflect_short_type_path(),
            Self::Render(m) => m.as_modifier().as_reflect().reflect_short_type_path(),
        }
    }
}

/// Curated catalog of modifier "templates" the user can add from the
/// Outline panel. Each variant knows how to allocate its literals into
/// a `Module` and produce a [`BoxedAnyModifier`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddModifierKind {
    SetLifetime,
    SetPositionSphere,
    SetVelocitySphere,
    Accel,
    LinearDrag,
    SetColor,
    SetSize,
    OrientFaceCamera,
}

impl AddModifierKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::SetLifetime => "Set lifetime",
            Self::SetPositionSphere => "Set position (sphere)",
            Self::SetVelocitySphere => "Set velocity (sphere)",
            Self::Accel => "Acceleration",
            Self::LinearDrag => "Linear drag",
            Self::SetColor => "Set color",
            Self::SetSize => "Set size",
            Self::OrientFaceCamera => "Orient (face camera)",
        }
    }

    /// Build a fresh instance, allocating any required literals into
    /// `module`. The caller is responsible for splicing the result
    /// into the appropriate modifier list.
    pub fn make(self, module: &mut Module) -> BoxedAnyModifier {
        match self {
            Self::SetLifetime => {
                let v = module.lit(5.0_f32);
                BoxedAnyModifier::Plain(Box::new(SetAttributeModifier::new(
                    Attribute::LIFETIME,
                    v,
                )))
            }
            Self::SetPositionSphere => {
                let center = module.lit(Vec3::ZERO);
                let radius = module.lit(1.0_f32);
                BoxedAnyModifier::Plain(Box::new(SetPositionSphereModifier {
                    center,
                    radius,
                    dimension: ShapeDimension::Volume,
                }))
            }
            Self::SetVelocitySphere => {
                let center = module.lit(Vec3::ZERO);
                let speed = module.lit(1.0_f32);
                BoxedAnyModifier::Plain(Box::new(SetVelocitySphereModifier {
                    center,
                    speed,
                }))
            }
            Self::Accel => {
                BoxedAnyModifier::Plain(Box::new(AccelModifier::constant(
                    module,
                    Vec3::new(0.0, -9.81, 0.0),
                )))
            }
            Self::LinearDrag => BoxedAnyModifier::Plain(Box::new(
                LinearDragModifier::constant(module, 1.0),
            )),
            Self::SetColor => BoxedAnyModifier::Render(Box::new(SetColorModifier::new(
                Vec4::new(1.0, 1.0, 1.0, 1.0),
            ))),
            Self::SetSize => BoxedAnyModifier::Render(Box::new(SetSizeModifier {
                size: Vec3::new(0.1, 0.1, 0.1).into(),
            })),
            Self::OrientFaceCamera => BoxedAnyModifier::Render(Box::new(OrientModifier::new(
                OrientMode::ParallelCameraDepthPlane,
            ))),
        }
    }

    /// Variants offered for a given modifier group in the Add menu.
    pub fn options_for(group: ModifierGroup) -> &'static [AddModifierKind] {
        match group {
            ModifierGroup::Init => &[
                Self::SetLifetime,
                Self::SetPositionSphere,
                Self::SetVelocitySphere,
            ],
            ModifierGroup::Update => &[Self::Accel, Self::LinearDrag],
            ModifierGroup::Render => &[Self::SetColor, Self::SetSize, Self::OrientFaceCamera],
        }
    }
}

/// Map a `ModifierGroup` to the canonical bevy_hanabi context flag for
/// the non-render groups. Render is handled separately. Kept available
/// for future use; not currently called.
#[allow(dead_code)]
pub fn group_context(group: ModifierGroup) -> Option<ModifierContext> {
    match group {
        ModifierGroup::Init => Some(ModifierContext::Init),
        ModifierGroup::Update => Some(ModifierContext::Update),
        ModifierGroup::Render => None,
    }
}

/// Rebuild an `EffectAsset` with caller-supplied edits applied to its
/// three modifier lists. The closure receives mutable references to
/// snapshots of the existing lists (cloned from the source via the
/// modifiers' own `boxed_clone` / `boxed_render_clone`).
///
/// The output asset preserves all scalar fields, the mesh handle, and
/// — crucially — the existing `Module` (so every `ExprHandle` already
/// in use remains valid).
pub fn rebuild_with_modifiers<F>(asset: &EffectAsset, f: F) -> EffectAsset
where
    F: FnOnce(
        &mut Vec<BoxedModifier>,
        &mut Vec<BoxedModifier>,
        &mut Vec<Box<dyn RenderModifier>>,
    ),
{
    let mut init: Vec<BoxedModifier> =
        asset.init_modifiers().map(|m| m.boxed_clone()).collect();
    let mut update: Vec<BoxedModifier> =
        asset.update_modifiers().map(|m| m.boxed_clone()).collect();
    let mut render: Vec<Box<dyn RenderModifier>> = asset
        .render_modifiers()
        .map(|m| m.boxed_render_clone())
        .collect();

    f(&mut init, &mut update, &mut render);

    let module = asset.module().clone();
    let mut new = EffectAsset::new(asset.capacity(), asset.spawner, module);
    new.name = asset.name.clone();
    new.simulation_space = asset.simulation_space;
    new.simulation_condition = asset.simulation_condition;
    new.z_layer_2d = asset.z_layer_2d;
    new.prng_seed = asset.prng_seed;
    new.motion_integration = asset.motion_integration;
    new.alpha_mode = asset.alpha_mode;
    new.mesh = asset.mesh.clone();

    for m in init {
        new = new.add_modifier(ModifierContext::Init, m);
    }
    for m in update {
        new = new.add_modifier(ModifierContext::Update, m);
    }
    for m in render {
        new = new.add_render_modifier(m);
    }

    new
}
