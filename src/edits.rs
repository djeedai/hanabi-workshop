//! Edit-message scaffolding.
//!
//! See `crate::document` for the architectural commitment. The rule:
//!
//! * UI code emits [`EditRequest`] messages; it never calls
//!   `DocumentContent` mutators directly.
//! * [`apply_edits`] is the **only** caller of `DocumentContent::set_*`
//!   and the only system holding `Query<&mut DocumentContent>` and
//!   `ResMut<Assets<EffectAsset>>` for write access.
//! * [`crate::history::record_history`] maintains the per-document undo
//!   stack from [`EditApplied`] events.

use bevy::prelude::*;
use bevy_hanabi::{EffectAsset, EffectSpawner, ParticleEffect, SimulationCondition, SimulationSpace, SpawnerSettings};

use crate::document::{DocumentContent, DocumentSceneRoot};
use crate::history::EditDirection;

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
}

/// Emitted by [`apply_edits`] after a mutation. Carries the inverse edit
/// and the direction flag the history recorder uses.
#[derive(Message, Debug, Clone)]
pub struct EditApplied {
    pub doc: Entity,
    pub inverse: EditRequest,
    pub direction: EditDirection,
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
    mut contents: Query<&mut DocumentContent>,
    mut effects: ResMut<Assets<EffectAsset>>,
    children_q: Query<&Children>,
    scene_roots: Query<(), With<DocumentSceneRoot>>,
    mut particle_effects: Query<&mut ParticleEffect>,
    mut effect_spawners: Query<&mut EffectSpawner>,
) {
    for req in requests.read() {
        let Ok(mut content) = contents.get_mut(req.doc) else {
            warn!("edit request for missing document: {:?}", req);
            continue;
        };

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
                touch_particle_effect(req.doc, &children_q, &scene_roots, &mut particle_effects);
                EditKind::SetEffectName { new: old }
            }
            EditKind::SetSimulationSpace { new } => {
                let Some(asset) = effects.get_mut(content.effect()) else {
                    warn!("SetSimulationSpace: missing asset for {:?}", req.doc);
                    continue;
                };
                let old = std::mem::replace(&mut asset.simulation_space, *new);
                content.mark_dirty(true);
                touch_particle_effect(req.doc, &children_q, &scene_roots, &mut particle_effects);
                EditKind::SetSimulationSpace { new: old }
            }
            EditKind::SetSimulationCondition { new } => {
                let Some(asset) = effects.get_mut(content.effect()) else {
                    warn!("SetSimulationCondition: missing asset for {:?}", req.doc);
                    continue;
                };
                let old = std::mem::replace(&mut asset.simulation_condition, *new);
                content.mark_dirty(true);
                touch_particle_effect(req.doc, &children_q, &scene_roots, &mut particle_effects);
                EditKind::SetSimulationCondition { new: old }
            }
            EditKind::SetSpawnerSettings { new } => {
                let Some(asset) = effects.get_mut(content.effect()) else {
                    warn!("SetSpawnerSettings: missing asset for {:?}", req.doc);
                    continue;
                };
                let old = std::mem::replace(&mut asset.spawner, *new);
                content.mark_dirty(true);
                touch_particle_effect(req.doc, &children_q, &scene_roots, &mut particle_effects);
                // The live EffectSpawner component is initialised from
                // `asset.spawner` once and never re-read, so we patch it
                // in place. Otherwise the asset edit only takes visible
                // effect after a Respawn.
                patch_effect_spawner(
                    req.doc,
                    *new,
                    &children_q,
                    &scene_roots,
                    &mut effect_spawners,
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
                touch_particle_effect(req.doc, &children_q, &scene_roots, &mut particle_effects);
                EditKind::SetZLayer2d { new: old }
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
    children_q: &Query<&Children>,
    scene_roots: &Query<(), With<DocumentSceneRoot>>,
    particle_effects: &mut Query<&mut ParticleEffect>,
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
    children_q: &Query<&Children>,
    scene_roots: &Query<(), With<DocumentSceneRoot>>,
    effect_spawners: &mut Query<&mut EffectSpawner>,
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
