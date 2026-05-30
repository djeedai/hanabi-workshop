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
    /// Short type name of the Hanabi modifier struct this template
    /// constructs. Used to look up display names from the curated
    /// table in `crate::ui::modifier_names`, so the Add menu and the
    /// resulting Outline row read identically.
    ///
    /// `SetLifetime` is a special case: it constructs a generic
    /// `SetAttributeModifier`, but in the Add menu we want the more
    /// specific "Set Lifetime" wording.
    fn hanabi_short_type_name(self) -> &'static str {
        match self {
            Self::SetLifetime => "SetAttributeModifier",
            Self::SetPositionSphere => "SetPositionSphereModifier",
            Self::SetVelocitySphere => "SetVelocitySphereModifier",
            Self::Accel => "AccelModifier",
            Self::LinearDrag => "LinearDragModifier",
            Self::SetColor => "SetColorModifier",
            Self::SetSize => "SetSizeModifier",
            Self::OrientFaceCamera => "OrientModifier",
        }
    }

    pub fn label(self) -> std::borrow::Cow<'static, str> {
        match self {
            // Special-cased: the underlying modifier is the generic
            // SetAttributeModifier, but the template fixes the
            // attribute to LIFETIME, so we surface that.
            Self::SetLifetime => std::borrow::Cow::Borrowed("Set Lifetime"),
            // OrientFaceCamera is one of several Orient configurations,
            // so override with the specific wording.
            Self::OrientFaceCamera => std::borrow::Cow::Borrowed("Orient (Face Camera)"),
            other => {
                crate::ui::modifier_names::display_name_for_type(other.hanabi_short_type_name())
            }
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

    /// Modifier contexts this template is valid in. Mirrors
    /// `Modifier::context()` on the underlying Hanabi type so the
    /// Add-menu filtering matches what Hanabi will actually accept.
    /// Many Set* modifiers run in both Init and Update.
    pub fn context(self) -> ModifierContext {
        match self {
            // SetAttributeModifier is Init | Update, but a "Set
            // lifetime" template only makes semantic sense at Init
            // (overwriting lifetime mid-flight is almost never what
            // you want), so we intentionally narrow it.
            Self::SetLifetime => ModifierContext::Init,
            // Position / velocity setters: Init | Update per Hanabi.
            Self::SetPositionSphere | Self::SetVelocitySphere => {
                ModifierContext::Init | ModifierContext::Update
            }
            Self::Accel | Self::LinearDrag => ModifierContext::Update,
            // All render modifiers report Render via the
            // `impl_mod_render!` macro in Hanabi.
            Self::SetColor | Self::SetSize | Self::OrientFaceCamera => ModifierContext::Render,
        }
    }

    /// Every variant in this enum, in stable display order.
    pub const ALL: &'static [AddModifierKind] = &[
        Self::SetLifetime,
        Self::SetPositionSphere,
        Self::SetVelocitySphere,
        Self::Accel,
        Self::LinearDrag,
        Self::SetColor,
        Self::SetSize,
        Self::OrientFaceCamera,
    ];

    /// Variants offered for a given modifier group in the Add menu.
    /// Filters [`Self::ALL`] by whether the template's context
    /// contains the group's flag, so a single template can appear in
    /// multiple groups (e.g. Set Position is valid in both Init and
    /// Update).
    pub fn options_for(group: ModifierGroup) -> impl Iterator<Item = AddModifierKind> {
        let ctx_flag = group_context(group);
        Self::ALL
            .iter()
            .copied()
            .filter(move |k| k.context().contains(ctx_flag))
    }
}

/// Map a [`ModifierGroup`] to its [`ModifierContext`] flag.
pub fn group_context(group: ModifierGroup) -> ModifierContext {
    match group {
        ModifierGroup::Init => ModifierContext::Init,
        ModifierGroup::Update => ModifierContext::Update,
        ModifierGroup::Render => ModifierContext::Render,
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
