//! Modifier list mutation primitives.
//!
//! Phase 5b — adding / removing / reordering modifiers in an
//! [`EffectAsset`]. The asset's three modifier vectors are private
//! and `#[reflect(ignore)]`, so we cannot mutate them in place.
//! Instead, every mutation rebuilds a new `EffectAsset` (preserving
//! the existing `Module` arena — and therefore every `ExprHandle`)
//! and overwrites the slot in `Assets<EffectAsset>`.
//!
//! Factory closures for individual modifier types live in
//! [`crate::modifier_registry`] (an ECS resource). This module owns
//! the [`BoxedAnyModifier`] wrapper, the rebuild helper, and the
//! `ModifierGroup ↔ ModifierContext` mapping.

use bevy_hanabi::{BoxedModifier, EffectAsset, ModifierContext, RenderModifier};

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
            Self::Render(m) => (
                "Render",
                m.as_modifier().as_reflect().reflect_short_type_path(),
            ),
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

/// Conversion from our discrete "which list" enum to Hanabi's
/// bitflag context. Every [`ModifierGroup`] maps to exactly one
/// single-bit [`ModifierContext`]; the reverse isn't well-defined
/// (a `ModifierContext` may carry multiple bits or none), so the
/// `From` impl only goes in this direction.
impl From<ModifierGroup> for ModifierContext {
    fn from(group: ModifierGroup) -> Self {
        match group {
            ModifierGroup::Init => ModifierContext::Init,
            ModifierGroup::Update => ModifierContext::Update,
            ModifierGroup::Render => ModifierContext::Render,
        }
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
    F: FnOnce(&mut Vec<BoxedModifier>, &mut Vec<BoxedModifier>, &mut Vec<Box<dyn RenderModifier>>),
{
    let mut init: Vec<BoxedModifier> = asset.init_modifiers().map(|m| m.boxed_clone()).collect();
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
