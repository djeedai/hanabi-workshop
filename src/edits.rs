//! Edit-message scaffolding.
//!
//! See `crate::document` for the architectural commitment. The rule:
//!
//! * UI code emits [`EditRequest`] messages; it never calls `DocumentContent`
//!   mutators directly.
//! * [`apply_edits`] is the **only** caller of `DocumentContent::set_*` and the
//!   only system holding `Query<&mut DocumentContent>` and
//!   `ResMut<Assets<EffectAsset>>` for write access.
//! * [`crate::history::record_history`] maintains the per-document undo stack
//!   from [`EditApplied`] events.

use std::any::TypeId;

use bevy::prelude::*;
use bevy_hanabi::{
    EffectAsset, EffectProperties, EffectSpawner, Expr, ExprHandle, LiteralExpr, ParticleEffect,
    SimulationCondition, SimulationSpace, SpawnerSettings, Value,
};

use crate::document::{DocumentContent, DocumentSceneRoot, ModifierGroup};
use crate::history::EditDirection;
use crate::modifier_ops::{self, BoxedAnyModifier};
use crate::modifier_registry;
use crate::playback::PlaybackCommand;
use crate::proxy::{self, ProxyEffect};

/// A pending mutation to a document, addressed to one document entity.
#[derive(Message, Debug, Clone)]
pub struct EditRequest {
    pub doc: Entity,
    /// Where the request comes from. UI code always emits `Fresh`;
    /// `history_dispatch` rewrites Undo/Redo replays.
    pub direction: EditDirection,
    pub kind: EditKind,
}

impl EditRequest {
    pub fn new(doc: Entity, kind: EditKind) -> Self {
        Self {
            doc,
            direction: EditDirection::Fresh,
            kind,
        }
    }

    /// Flip `direction` to `Undo` (for replays popped from `History.past`).
    pub fn with_undo(mut self) -> Self {
        self.direction = EditDirection::Undo;
        self
    }

    /// Flip `direction` to `Redo` (for replays popped from `History.future`).
    pub fn with_redo(mut self) -> Self {
        self.direction = EditDirection::Redo;
        self
    }
}

/// The actual edit payload. Each variant carries the *new* value;
/// `apply_edits` reads the current value to build the inverse.
#[derive(Debug, Clone)]
pub enum EditKind {
    /// Rename the document (shown in the tab title). Mutates
    /// `DocumentContent.name`, NOT `EffectAsset.name`. Not yet bound
    /// in the UI (Phase 5b will add an inline tab-rename).
    #[allow(dead_code)]
    RenameDocument { new: String },
    /// Set `EffectAsset.name` (the asset's internal identifier; used
    /// when serializing to RON).
    SetEffectName { new: String },
    /// Set `EffectAsset.simulation_space`.
    SetSimulationSpace { new: SimulationSpace },
    /// Set `EffectAsset.simulation_condition`.
    SetSimulationCondition { new: SimulationCondition },
    /// Replace `EffectAsset.spawner` wholesale. Whole-struct is fine —
    /// `SpawnerSettings` is `Copy` and small, and undo's drag-stop
    /// pattern only commits a single value per logical action.
    SetSpawnerSettings { new: SpawnerSettings },
    /// Set `EffectAsset.z_layer_2d`.
    SetZLayer2d { new: f32 },
    /// Replace the [`Value`] of an `Expr::Literal` at the given
    /// canonical `ExprHandle`. Phase 5b "live tweak" path: applied
    /// directly to the canonical asset's `Module`, and uploaded to
    /// the proxy's matching synthetic `Property` via
    /// [`EffectProperties::set_if_changed`] — no shader recompile.
    SetLiteralValue {
        canonical_expr: ExprHandle,
        new: Value,
    },
    /// Add a fresh modifier of a given type (looked up in the
    /// [`AppTypeRegistry`] via its
    /// [`crate::modifier_registry::ReflectModifier`] data) into `group`,
    /// inserted at position `at` (== length means append). UI emits this;
    /// the apply arm allocates fresh literals in the canonical module
    /// before splicing the modifier in.
    AddModifierFromTemplate {
        group: ModifierGroup,
        /// `TypeId` of the Hanabi modifier struct. In-process only —
        /// these edits are never serialized.
        type_id: TypeId,
        at: usize,
    },
    /// Insert a pre-built modifier at position `at`. Used internally
    /// as the inverse of [`EditKind::RemoveModifier`] — undoing a
    /// removal needs to restore the original modifier with its
    /// original `ExprHandle` slots intact.
    AddBoxedModifier {
        group: ModifierGroup,
        at: usize,
        modifier: BoxedAnyModifier,
    },
    /// Remove the modifier at `idx` in `group`.
    RemoveModifier { group: ModifierGroup, idx: usize },
    /// Move the modifier from `from` to `to` within `group`. `to` is
    /// the target index *after* removal of the source slot.
    MoveModifier {
        group: ModifierGroup,
        from: usize,
        to: usize,
    },
    /// Add a brand-new user property to the canonical asset's module.
    /// Inverse: [`EditKind::RemoveProperty`] with the same name. Fails
    /// silently (logs a warning) if `name` is already taken or starts
    /// with the reserved tweak-prop prefix.
    AddProperty { name: String, value: Value },
    /// Remove a user property by name. Bound expression slots are
    /// auto-demoted to `Expr::Literal(default_value)` so the asset
    /// remains valid. Inverse: [`EditKind::RestoreProperty`] carrying
    /// the captured default and the list of demoted handles.
    RemoveProperty { name: String },
    /// Re-add a previously-removed property and re-promote each
    /// `repromote_exprs` slot from literal back to property. Used
    /// only as the inverse of [`EditKind::RemoveProperty`]; not
    /// emitted directly by the UI.
    RestoreProperty {
        name: String,
        value: Value,
        repromote_exprs: Vec<ExprHandle>,
    },
    /// Rename a user property. WGSL identifier name changes too →
    /// triggers a recompile.
    RenameProperty { old: String, new: String },
    /// Replace a user property's initial (default) value. Also pushes
    /// the new value live via `EffectProperties::set_if_changed`, so
    /// the running effect updates without a Respawn.
    SetPropertyDefault { name: String, new: Value },
}

/// Emitted by [`apply_edits`] after a mutation. Carries the inverse edit
/// and the direction flag the history recorder uses.
#[derive(Message, Debug, Clone)]
pub struct EditApplied {
    pub doc: Entity,
    pub inverse: EditRequest,
    pub direction: EditDirection,
    /// True for `SetLiteralValue` (no proxy rebuild needed; value
    /// already uploaded as a property). False for everything else
    /// (proxy must be re-built from canonical to mirror the change).
    pub is_literal_edit: bool,
}

/// User-driven history navigation. Consumed by `crate::history`.
#[derive(Message, Debug, Clone, Copy)]
pub enum HistoryRequest {
    Undo(Entity),
    Redo(Entity),
}

pub struct EditPlugin;

impl Plugin for EditPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<EditRequest>()
            .add_message::<EditApplied>()
            .add_message::<HistoryRequest>()
            .add_systems(
                Update,
                (
                    crate::history::history_dispatch,
                    apply_edits,
                    crate::history::record_history,
                )
                    .chain()
                    .in_set(EditSystems),
            );
    }
}

/// Systems that depend on freshly-applied edits should be ordered
/// `.after(EditSystems)`.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct EditSystems;

/// The single writer of `DocumentContent` for content edits, and of
/// `EffectAsset` for asset-level edits. Touches the document's
/// `ParticleEffect` after every asset mutation to force `bevy_hanabi`'s
/// `compile_effects` to refresh (it reacts to `Ref<ParticleEffect>` change
/// detection, not to `AssetEvent<EffectAsset>`).
pub fn apply_edits(
    mut requests: MessageReader<EditRequest>,
    mut applied: MessageWriter<EditApplied>,
    mut playback: MessageWriter<PlaybackCommand>,
    mut contents: Query<&mut DocumentContent>,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut children_q: Query<&Children>,
    mut scene_roots: Query<(), With<DocumentSceneRoot>>,
    mut particle_effects: Query<&mut ParticleEffect>,
    mut effect_spawners: Query<&mut EffectSpawner>,
    mut proxies: Query<&ProxyEffect>,
    mut effect_props: Query<&mut EffectProperties>,
    type_registry: Res<AppTypeRegistry>,
) {
    for req in requests.read() {
        let Ok(mut content) = contents.get_mut(req.doc) else {
            warn!("edit request for missing document: {:?}", req);
            continue;
        };

        let mut is_literal_edit = false;

        // Each arm returns the inverse `EditKind` (the value to apply
        // to undo this edit). Asset-level arms also touch the doc's
        // ParticleEffect to trigger hanabi recompile.
        let inverse_kind = match &req.kind {
            EditKind::RenameDocument { new } => {
                let old = content.set_name(new.clone());
                EditKind::RenameDocument { new: old }
            }
            EditKind::SetEffectName { new } => {
                let Some(asset) = effects.get_mut(content.effect()) else {
                    warn!("SetEffectName: missing asset for {:?}", req.doc);
                    continue;
                };
                let old = std::mem::replace(&mut asset.name, new.clone());
                content.mark_dirty(true);
                touch_particle_effect(
                    req.doc,
                    children_q.reborrow(),
                    scene_roots.reborrow(),
                    particle_effects.reborrow(),
                );
                EditKind::SetEffectName { new: old }
            }
            EditKind::SetSimulationSpace { new } => {
                let Some(asset) = effects.get_mut(content.effect()) else {
                    warn!("SetSimulationSpace: missing asset for {:?}", req.doc);
                    continue;
                };
                let old = std::mem::replace(&mut asset.simulation_space, *new);
                content.mark_dirty(true);
                touch_particle_effect(
                    req.doc,
                    children_q.reborrow(),
                    scene_roots.reborrow(),
                    particle_effects.reborrow(),
                );
                EditKind::SetSimulationSpace { new: old }
            }
            EditKind::SetSimulationCondition { new } => {
                let Some(asset) = effects.get_mut(content.effect()) else {
                    warn!("SetSimulationCondition: missing asset for {:?}", req.doc);
                    continue;
                };
                let old = std::mem::replace(&mut asset.simulation_condition, *new);
                content.mark_dirty(true);
                touch_particle_effect(
                    req.doc,
                    children_q.reborrow(),
                    scene_roots.reborrow(),
                    particle_effects.reborrow(),
                );
                EditKind::SetSimulationCondition { new: old }
            }
            EditKind::SetSpawnerSettings { new } => {
                let Some(asset) = effects.get_mut(content.effect()) else {
                    warn!("SetSpawnerSettings: missing asset for {:?}", req.doc);
                    continue;
                };
                let old = std::mem::replace(&mut asset.spawner, *new);
                content.mark_dirty(true);
                touch_particle_effect(
                    req.doc,
                    children_q.reborrow(),
                    scene_roots.reborrow(),
                    particle_effects.reborrow(),
                );
                // The live EffectSpawner component is initialised from
                // `asset.spawner` once and never re-read, so we patch it
                // in place. Otherwise the asset edit only takes visible
                // effect after a Respawn.
                patch_effect_spawner(
                    req.doc,
                    *new,
                    children_q.reborrow(),
                    scene_roots.reborrow(),
                    effect_spawners.reborrow(),
                );
                EditKind::SetSpawnerSettings { new: old }
            }
            EditKind::SetZLayer2d { new } => {
                let Some(asset) = effects.get_mut(content.effect()) else {
                    warn!("SetZLayer2d: missing asset for {:?}", req.doc);
                    continue;
                };
                let old = std::mem::replace(&mut asset.z_layer_2d, *new);
                content.mark_dirty(true);
                touch_particle_effect(
                    req.doc,
                    children_q.reborrow(),
                    scene_roots.reborrow(),
                    particle_effects.reborrow(),
                );
                EditKind::SetZLayer2d { new: old }
            }
            EditKind::SetLiteralValue {
                canonical_expr,
                new,
            } => {
                is_literal_edit = true;
                let Some(asset) = effects.get_mut(content.effect()) else {
                    warn!("SetLiteralValue: missing asset for {:?}", req.doc);
                    continue;
                };
                // (1) Mutate the canonical Module's arena slot in place.
                let Some(module) = proxy::module_mut(asset) else {
                    warn!("SetLiteralValue: could not reach &mut Module via reflect");
                    continue;
                };
                let Some(slot) = module.get_mut(*canonical_expr) else {
                    warn!("SetLiteralValue: handle {:?} not in module", canonical_expr);
                    continue;
                };
                let old_value = match slot {
                    Expr::Literal(lit) => proxy::literal_value(lit),
                    _ => None,
                };
                let Some(old_value) = old_value else {
                    warn!(
                        "SetLiteralValue: slot {:?} is not a literal (canonical \
                         expr was promoted/edited externally); skipping",
                        canonical_expr
                    );
                    continue;
                };
                *slot = Expr::Literal(LiteralExpr::new(*new));
                content.mark_dirty(true);
                // NOTE: we deliberately do *not* touch_particle_effect —
                // the canonical asset isn't the one running; the proxy is.
                // The upload below bypasses the shader entirely.

                // (2) Upload to the live proxy's EffectProperties.
                upload_literal_to_proxy(
                    req.doc,
                    *canonical_expr,
                    *new,
                    children_q.reborrow(),
                    scene_roots.reborrow(),
                    proxies.reborrow(),
                    effect_props.reborrow(),
                );

                EditKind::SetLiteralValue {
                    canonical_expr: *canonical_expr,
                    new: old_value,
                }
            }
            EditKind::AddModifierFromTemplate { group, type_id, at } => {
                let Some(asset) = effects.get_mut(content.effect()) else {
                    warn!("AddModifierFromTemplate: missing asset for {:?}", req.doc);
                    continue;
                };
                let type_registry = type_registry.read();
                let Some(kind) = modifier_registry::get_modifier_kind(&type_registry, *type_id)
                else {
                    warn!(
                        "AddModifierFromTemplate: TypeId {:?} not registered or missing \
                         ReflectModifier data",
                        type_id
                    );
                    continue;
                };
                // Allocate fresh literals into the canonical module
                // *before* the rebuild, so they end up in the new
                // asset's module clone (rebuild_with_modifiers clones
                // the module post-allocation).
                let Some(module) = proxy::module_mut(asset) else {
                    warn!("AddModifierFromTemplate: could not reach &mut Module via reflect");
                    continue;
                };
                let modifier = (kind.reflect_modifier.factory)(module);
                drop(type_registry);
                let new_asset = match insert_modifier(asset, *group, *at, modifier) {
                    Ok(a) => a,
                    Err(e) => {
                        warn!("AddModifierFromTemplate: {e}");
                        continue;
                    }
                };
                *asset = new_asset;
                content.mark_dirty(true);
                touch_particle_effect(
                    req.doc,
                    children_q.reborrow(),
                    scene_roots.reborrow(),
                    particle_effects.reborrow(),
                );
                // Structural change → particle attribute layout may
                // differ. Despawn live particles so we don't run a
                // freshly-recompiled update shader against a stale
                // GPU buffer.
                playback.write(PlaybackCommand::Respawn(req.doc));
                EditKind::RemoveModifier {
                    group: *group,
                    idx: *at,
                }
            }
            EditKind::AddBoxedModifier {
                group,
                at,
                modifier,
            } => {
                let Some(asset) = effects.get_mut(content.effect()) else {
                    warn!("AddBoxedModifier: missing asset for {:?}", req.doc);
                    continue;
                };
                let new_asset = match insert_modifier(asset, *group, *at, modifier.clone()) {
                    Ok(a) => a,
                    Err(e) => {
                        warn!("AddBoxedModifier: {e}");
                        continue;
                    }
                };
                *asset = new_asset;
                content.mark_dirty(true);
                touch_particle_effect(
                    req.doc,
                    children_q.reborrow(),
                    scene_roots.reborrow(),
                    particle_effects.reborrow(),
                );
                playback.write(PlaybackCommand::Respawn(req.doc));
                EditKind::RemoveModifier {
                    group: *group,
                    idx: *at,
                }
            }
            EditKind::RemoveModifier { group, idx } => {
                let Some(asset) = effects.get_mut(content.effect()) else {
                    warn!("RemoveModifier: missing asset for {:?}", req.doc);
                    continue;
                };
                let (new_asset, removed) = match remove_modifier(asset, *group, *idx) {
                    Ok(p) => p,
                    Err(e) => {
                        warn!("RemoveModifier: {e}");
                        continue;
                    }
                };
                *asset = new_asset;
                content.mark_dirty(true);
                touch_particle_effect(
                    req.doc,
                    children_q.reborrow(),
                    scene_roots.reborrow(),
                    particle_effects.reborrow(),
                );
                playback.write(PlaybackCommand::Respawn(req.doc));
                EditKind::AddBoxedModifier {
                    group: *group,
                    at: *idx,
                    modifier: removed,
                }
            }
            EditKind::MoveModifier { group, from, to } => {
                let Some(asset) = effects.get_mut(content.effect()) else {
                    warn!("MoveModifier: missing asset for {:?}", req.doc);
                    continue;
                };
                let new_asset = match move_modifier(asset, *group, *from, *to) {
                    Ok(a) => a,
                    Err(e) => {
                        warn!("MoveModifier: {e}");
                        continue;
                    }
                };
                *asset = new_asset;
                content.mark_dirty(true);
                touch_particle_effect(
                    req.doc,
                    children_q.reborrow(),
                    scene_roots.reborrow(),
                    particle_effects.reborrow(),
                );
                // Reorder may or may not change layout, but the
                // recompiled shader binds attribute slots fresh —
                // safest to respawn so live particles don't desync.
                playback.write(PlaybackCommand::Respawn(req.doc));
                EditKind::MoveModifier {
                    group: *group,
                    from: *to,
                    to: *from,
                }
            }
            EditKind::AddProperty { name, value } => {
                if proxy::is_tweak_prop_name(name) {
                    warn!(
                        "AddProperty: name {:?} starts with reserved \
                         tweak prefix; ignoring",
                        name
                    );
                    continue;
                }
                let Some(asset) = effects.get_mut(content.effect()) else {
                    warn!("AddProperty: missing asset for {:?}", req.doc);
                    continue;
                };
                let Some(module) = proxy::module_mut(asset) else {
                    warn!("AddProperty: could not reach &mut Module via reflect");
                    continue;
                };
                if !proxy::add_user_property(module, name, *value) {
                    warn!("AddProperty: name {:?} already exists", name);
                    continue;
                }
                content.mark_dirty(true);
                touch_particle_effect(
                    req.doc,
                    children_q.reborrow(),
                    scene_roots.reborrow(),
                    particle_effects.reborrow(),
                );
                playback.write(PlaybackCommand::Respawn(req.doc));
                EditKind::RemoveProperty { name: name.clone() }
            }
            EditKind::RemoveProperty { name } => {
                let Some(asset) = effects.get_mut(content.effect()) else {
                    warn!("RemoveProperty: missing asset for {:?}", req.doc);
                    continue;
                };
                let Some(module) = proxy::module_mut(asset) else {
                    warn!("RemoveProperty: could not reach &mut Module via reflect");
                    continue;
                };
                let Some((default_value, demoted)) = proxy::remove_user_property(module, name)
                else {
                    warn!("RemoveProperty: name {:?} not found", name);
                    continue;
                };
                content.mark_dirty(true);
                touch_particle_effect(
                    req.doc,
                    children_q.reborrow(),
                    scene_roots.reborrow(),
                    particle_effects.reborrow(),
                );
                playback.write(PlaybackCommand::Respawn(req.doc));
                EditKind::RestoreProperty {
                    name: name.clone(),
                    value: default_value,
                    repromote_exprs: demoted,
                }
            }
            EditKind::RestoreProperty {
                name,
                value,
                repromote_exprs,
            } => {
                let Some(asset) = effects.get_mut(content.effect()) else {
                    warn!("RestoreProperty: missing asset for {:?}", req.doc);
                    continue;
                };
                let Some(module) = proxy::module_mut(asset) else {
                    warn!("RestoreProperty: could not reach &mut Module via reflect");
                    continue;
                };
                if !proxy::restore_property_with_promotions(module, name, *value, repromote_exprs) {
                    warn!("RestoreProperty: name {:?} already exists", name);
                    continue;
                }
                content.mark_dirty(true);
                touch_particle_effect(
                    req.doc,
                    children_q.reborrow(),
                    scene_roots.reborrow(),
                    particle_effects.reborrow(),
                );
                playback.write(PlaybackCommand::Respawn(req.doc));
                EditKind::RemoveProperty { name: name.clone() }
            }
            EditKind::RenameProperty { old, new } => {
                if proxy::is_tweak_prop_name(new) {
                    warn!(
                        "RenameProperty: target name {:?} starts with \
                         reserved tweak prefix; ignoring",
                        new
                    );
                    continue;
                }
                let Some(asset) = effects.get_mut(content.effect()) else {
                    warn!("RenameProperty: missing asset for {:?}", req.doc);
                    continue;
                };
                let Some(module) = proxy::module_mut(asset) else {
                    warn!("RenameProperty: could not reach &mut Module via reflect");
                    continue;
                };
                if !proxy::rename_property(module, old, new) {
                    warn!(
                        "RenameProperty: failed (name {:?} missing or {:?} taken)",
                        old, new
                    );
                    continue;
                }
                content.mark_dirty(true);
                touch_particle_effect(
                    req.doc,
                    children_q.reborrow(),
                    scene_roots.reborrow(),
                    particle_effects.reborrow(),
                );
                playback.write(PlaybackCommand::Respawn(req.doc));
                EditKind::RenameProperty {
                    old: new.clone(),
                    new: old.clone(),
                }
            }
            EditKind::SetPropertyDefault { name, new } => {
                let Some(asset) = effects.get_mut(content.effect()) else {
                    warn!("SetPropertyDefault: missing asset for {:?}", req.doc);
                    continue;
                };
                let Some(module) = proxy::module_mut(asset) else {
                    warn!("SetPropertyDefault: could not reach &mut Module via reflect");
                    continue;
                };
                let Some(old) = proxy::set_property_default(module, name, *new) else {
                    warn!("SetPropertyDefault: name {:?} not found", name);
                    continue;
                };
                content.mark_dirty(true);
                // No shader rebuild needed — only the default value
                // changed. We still upload the new value to the live
                // EffectProperties so the running effect picks it up
                // without a Respawn.
                upload_user_property_to_proxy(
                    req.doc,
                    name,
                    *new,
                    children_q.reborrow(),
                    scene_roots.reborrow(),
                    effect_props.reborrow(),
                );
                // Don't touch_particle_effect — we want this to be a
                // "no recompile" tweak edit. Note we deliberately do
                // *not* set `is_literal_edit = true` though: the
                // canonical asset changed shape (a property's default
                // value mutation), so the proxy must still be rebuilt
                // by `sync_proxy_on_edit_applied` to mirror it. The
                // rebuild is cheap (no shader compile is triggered by
                // it directly — only `touch_particle_effect` does).
                EditKind::SetPropertyDefault {
                    name: name.clone(),
                    new: old,
                }
            }
        };

        applied.write(EditApplied {
            doc: req.doc,
            inverse: EditRequest {
                doc: req.doc,
                direction: req.direction, // unused on inverse, kept for symmetry
                kind: inverse_kind,
            },
            direction: req.direction,
            is_literal_edit,
        });
    }
}

/// Force `bevy_hanabi`'s `compile_effects` to re-process the doc's
/// `ParticleEffect`. We do this after every `EffectAsset` mutation
/// because hanabi reacts to `Ref<ParticleEffect>::is_changed()`, not to
/// `AssetEvent<EffectAsset>::Modified`. The cost is one shader rebuild
/// per commit, which is acceptable at our edit-once-per-drag cadence.
fn touch_particle_effect(
    doc: Entity,
    children_q: Query<&Children>,
    scene_roots: Query<(), With<DocumentSceneRoot>>,
    mut particle_effects: Query<&mut ParticleEffect>,
) {
    let Ok(doc_children) = children_q.get(doc) else {
        return;
    };
    for &child in doc_children {
        if scene_roots.get(child).is_err() {
            continue;
        }
        let Ok(scene_children) = children_q.get(child) else {
            continue;
        };
        for &grandchild in scene_children {
            if let Ok(mut effect) = particle_effects.get_mut(grandchild) {
                effect.set_changed();
                return;
            }
        }
    }
}

/// Push new `SpawnerSettings` onto the live `EffectSpawner` component
/// for the document's effect instance. `bevy_hanabi`'s `tick_spawners`
/// creates `EffectSpawner` once from `asset.spawner` and then never
/// re-reads it, so without this patch the asset edit only takes effect
/// after a Respawn.
fn patch_effect_spawner(
    doc: Entity,
    new: SpawnerSettings,
    children_q: Query<&Children>,
    scene_roots: Query<(), With<DocumentSceneRoot>>,
    mut effect_spawners: Query<&mut EffectSpawner>,
) {
    let Ok(doc_children) = children_q.get(doc) else {
        return;
    };
    for &child in doc_children {
        if scene_roots.get(child).is_err() {
            continue;
        }
        let Ok(scene_children) = children_q.get(child) else {
            continue;
        };
        for &grandchild in scene_children {
            if let Ok(mut spawner) = effect_spawners.get_mut(grandchild) {
                // Only patch `settings`; leave runtime `active` alone
                // — it represents play state, not the startup hint.
                spawner.settings = new;
                return;
            }
        }
    }
}

/// Look up the synthetic property name bound to `canonical_expr` on
/// the document's [`ProxyEffect`], then write the new `Value` into the
/// live proxy entity's `EffectProperties` via `set_if_changed`. This
/// is the "no shader recompile" path for slider tweaks.
///
/// Silently no-ops if the binding is missing (the literal wasn't
/// promoted, e.g. unsupported type), or if `EffectProperties` doesn't
/// exist yet on the proxy (first frame after build — Hanabi will
/// pick up the new value on the next frame regardless because the
/// canonical asset was already mutated and the proxy will be rebuilt).
fn upload_literal_to_proxy(
    doc: Entity,
    canonical_expr: ExprHandle,
    new: Value,
    children_q: Query<&Children>,
    scene_roots: Query<(), With<DocumentSceneRoot>>,
    proxies: Query<&ProxyEffect>,
    mut effect_props: Query<&mut EffectProperties>,
) {
    let Ok(proxy) = proxies.get(doc) else {
        return;
    };
    let Some(binding) = proxy::find_binding(&proxy.bindings, canonical_expr) else {
        return;
    };
    let prop_name = binding.proxy_prop_name.clone();
    let Ok(doc_children) = children_q.get(doc) else {
        return;
    };
    for &child in doc_children {
        if scene_roots.get(child).is_err() {
            continue;
        }
        let Ok(scene_children) = children_q.get(child) else {
            continue;
        };
        for &grandchild in scene_children {
            if let Ok(props) = effect_props.get_mut(grandchild) {
                EffectProperties::set_if_changed(props, &prop_name, new);
                return;
            }
        }
    }
}

/// Push a new value to a (user) property by name on the live proxy's
/// `EffectProperties`. Used by [`EditKind::SetPropertyDefault`] so the
/// running effect reflects the new initial value without a Respawn.
fn upload_user_property_to_proxy(
    doc: Entity,
    name: &str,
    new: Value,
    children_q: Query<&Children>,
    scene_roots: Query<(), With<DocumentSceneRoot>>,
    mut effect_props: Query<&mut EffectProperties>,
) {
    let Ok(doc_children) = children_q.get(doc) else {
        return;
    };
    for &child in doc_children {
        if scene_roots.get(child).is_err() {
            continue;
        }
        let Ok(scene_children) = children_q.get(child) else {
            continue;
        };
        for &grandchild in scene_children {
            if let Ok(props) = effect_props.get_mut(grandchild) {
                EffectProperties::set_if_changed(props, name, new);
                return;
            }
        }
    }
}

/// Insert `modifier` at position `at` in the chosen group's modifier
/// list. Returns a rebuilt `EffectAsset` with the change applied.
///
/// Errors:
/// - `at > current_len`: out-of-range insert.
/// - Group/modifier mismatch: trying to put a plain modifier into the render
///   slot or vice versa.
fn insert_modifier(
    asset: &EffectAsset,
    group: ModifierGroup,
    at: usize,
    modifier: BoxedAnyModifier,
) -> Result<EffectAsset, String> {
    let len = group_len(asset, group);
    if at > len {
        return Err(format!("insert at {at} but group {group:?} has len {len}"));
    }
    match (group, modifier) {
        (ModifierGroup::Render, BoxedAnyModifier::Render(m)) => Ok(
            modifier_ops::rebuild_with_modifiers(asset, |_init, _update, render| {
                render.insert(at, m);
            }),
        ),
        (ModifierGroup::Init | ModifierGroup::Update, BoxedAnyModifier::Plain(m)) => Ok(
            modifier_ops::rebuild_with_modifiers(asset, |init, update, _render| {
                let list = if group == ModifierGroup::Init {
                    init
                } else {
                    update
                };
                list.insert(at, m);
            }),
        ),
        (group, modifier) => Err(format!(
            "modifier kind / group mismatch: {} into {group:?}",
            modifier.short_type_name()
        )),
    }
}

/// Remove and return the modifier at `idx` in the chosen group. Used
/// for `RemoveModifier` (whose inverse must capture the original).
fn remove_modifier(
    asset: &EffectAsset,
    group: ModifierGroup,
    idx: usize,
) -> Result<(EffectAsset, BoxedAnyModifier), String> {
    let len = group_len(asset, group);
    if idx >= len {
        return Err(format!("remove at {idx} but group {group:?} has len {len}"));
    }
    let mut captured: Option<BoxedAnyModifier> = None;
    let new = modifier_ops::rebuild_with_modifiers(asset, |init, update, render| match group {
        ModifierGroup::Init => {
            captured = Some(BoxedAnyModifier::Plain(init.remove(idx)));
        }
        ModifierGroup::Update => {
            captured = Some(BoxedAnyModifier::Plain(update.remove(idx)));
        }
        ModifierGroup::Render => {
            captured = Some(BoxedAnyModifier::Render(render.remove(idx)));
        }
    });
    Ok((new, captured.expect("rebuild closure always runs")))
}

/// Move the modifier at `from` to `to` in the same group. `to` is the
/// post-removal target index — i.e. `to == from + 1` moves it one slot
/// later, `to == from - 1` one slot earlier.
fn move_modifier(
    asset: &EffectAsset,
    group: ModifierGroup,
    from: usize,
    to: usize,
) -> Result<EffectAsset, String> {
    let len = group_len(asset, group);
    if from >= len || to >= len {
        return Err(format!(
            "move {from} -> {to} out of range for group {group:?} (len {len})"
        ));
    }
    if from == to {
        // No-op move; rebuild a clone anyway so the apply path is uniform.
        return Ok(modifier_ops::rebuild_with_modifiers(asset, |_, _, _| {}));
    }
    Ok(modifier_ops::rebuild_with_modifiers(
        asset,
        |init, update, render| match group {
            ModifierGroup::Init => {
                let m = init.remove(from);
                init.insert(to, m);
            }
            ModifierGroup::Update => {
                let m = update.remove(from);
                update.insert(to, m);
            }
            ModifierGroup::Render => {
                let m = render.remove(from);
                render.insert(to, m);
            }
        },
    ))
}

fn group_len(asset: &EffectAsset, group: ModifierGroup) -> usize {
    match group {
        ModifierGroup::Init => asset.init_modifiers().count(),
        ModifierGroup::Update => asset.update_modifiers().count(),
        ModifierGroup::Render => asset.render_modifiers().count(),
    }
}
