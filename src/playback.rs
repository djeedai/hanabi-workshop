//! Per-document particle playback: play/pause, restart, respawn.
//!
//! Each document entity carries a [`PlaybackState`] component. The UI mutates
//! `playing` directly (it's plain state). One system pushes the **active
//! document's** `playing` into the global
//! `Time<EffectSimulation>::set_relative_speed` (1.0 if playing, 0.0 if
//! paused).
//!
//! ## Upstream gap
//!
//! `bevy_hanabi` 0.18 does **not** expose any per-emitter time scale. There is
//! only one global clock (`Time<EffectSimulation>`) that drives `tick_spawners`
//! and the GPU simulation. Pausing it freezes every open document at once. We
//! work around this by gating the global clock on the *active* document —
//! switching tabs effectively switches whose playback state owns the clock.
//! This matches a single-doc-at-a-time workflow. A real per-document timeline
//! would need an upstream feature (e.g. an `EffectTimeScale(f32)` component
//! that `tick_spawners` multiplies into `dt`, plus a per-instance time uniform
//! in the shader).
//!
//! [`PlaybackCommand`] is reserved for actions the UI cannot perform itself:
//! `Restart` (reset cycle time) and `Respawn` (despawn the scene root so
//! reconciliation rebuilds it). By convention, *state* lives on the component
//! and is mutated directly; *actions* go through the message channel.

use std::collections::HashSet;

use bevy::prelude::*;
use bevy_hanabi::{EffectSimulation, EffectSimulationTime, EffectSpawner, ShaderCache};

use crate::document::{
    ActiveDocument, DocumentContent, DocumentSceneRoot, EmitterSceneEntities, RenderLayerPool,
};

/// Per-document playback state.
///
/// The active document's `playing` value is mirrored into the global
/// `Time<EffectSimulation>` speed each frame.
#[derive(Component, Debug, Clone, Copy)]
pub struct PlaybackState {
    pub playing: bool,
}

impl Default for PlaybackState {
    fn default() -> Self {
        Self { playing: true }
    }
}

/// One-shot playback actions, addressed to a document entity.
///
/// Only includes actions that require `Commands` or the `EffectSpawner` query
/// (i.e. that the UI cannot perform itself).
#[derive(Message, Debug, Clone, Copy)]
pub enum PlaybackCommand {
    /// Reset every CPU-rooted emitter's spawner in the document — clears
    /// accumulated cycle time so emission restarts cleanly. GPU-driven child
    /// emitters are untouched: they carry only an inert spawner and are
    /// driven purely by their parent's spawn events. Does not despawn
    /// existing particles.
    Restart(Entity),
    /// Debug action: despawn the document's scene root so reconciliation
    /// recreates it on the next frame, giving a fresh `ParticleEffect`.
    /// Useful when in-place asset edits aren't picked up correctly.
    Respawn(Entity),
    /// Safely tear down and close a document.
    CloseDocument(Entity),
}

pub struct PlaybackPlugin;

impl Plugin for PlaybackPlugin {
    fn build(&self, app: &mut App) {
        // `apply_playback_commands` must observe every Respawn emitted by
        // `apply_edits` in the same Update, before Hanabi's PostUpdate systems
        // see the new assets.
        app.add_message::<PlaybackCommand>().add_systems(
            Update,
            (
                apply_playback_commands
                    .after(crate::edits::EditSystems)
                    .after(crate::app_commands::apply_app_commands),
                drive_effect_simulation_clock,
            ),
        );
    }
}

/// Every CPU-rooted emitter's preview entity for a document, i.e. every
/// [`SceneEmitter`] whose canonical [`crate::document::EmitterRecord::parent`]
/// is `None`.
///
/// A GPU-driven child emitter carries an (inert) `EffectSpawner` too — every
/// baked emitter does, see `bake::inert_spawner` — but only a CPU root's
/// spawner should ever be reset directly: children are driven purely by
/// spawn events from their parent, not their own cycle time.
///
/// [`SceneEmitter`]: crate::document::SceneEmitter
fn cpu_root_spawner_entities(
    doc: Entity,
    content: &DocumentContent,
    children_q: &Query<&Children>,
    scene_roots: &Query<&EmitterSceneEntities, With<DocumentSceneRoot>>,
) -> Vec<Entity> {
    let Ok(doc_children) = children_q.get(doc) else {
        return Vec::new();
    };
    for &child in doc_children {
        let Ok(scene_entities) = scene_roots.get(child) else {
            continue;
        };
        return content
            .preview_emitter_ids()
            .filter(|id| content.emitter_parent(*id).is_none())
            .filter_map(|id| scene_entities.get(id))
            .collect();
    }
    Vec::new()
}

fn apply_playback_commands(
    mut commands: Commands,
    mut reader: MessageReader<PlaybackCommand>,
    mut spawner_q: Query<&mut EffectSpawner>,
    docs: Query<&DocumentContent>,
    children_q: Query<&Children>,
    scene_roots: Query<&EmitterSceneEntities, With<DocumentSceneRoot>>,
    mut shader_cache: ResMut<ShaderCache>,
    mut layer_pool: ResMut<RenderLayerPool>,
    mut active: ResMut<ActiveDocument>,
) {
    let incoming: Vec<PlaybackCommand> = reader.read().copied().collect();
    let close_requests: HashSet<Entity> = incoming
        .iter()
        .filter_map(|command| match command {
            PlaybackCommand::CloseDocument(doc) => Some(*doc),
            _ => None,
        })
        .collect();
    let mut released_shaders = false;

    // A single frame can emit several `Respawn`s for one document (e.g. an edit
    // that both adds a modifier and links a node). The child queries are
    // snapshotted at system start, so despawning the scene root more than once
    // would target an already-despawned entity. Collapse to one respawn per doc.
    let mut respawned: HashSet<Entity> = HashSet::new();
    for cmd in incoming {
        match cmd {
            PlaybackCommand::Restart(doc) => {
                let Ok(content) = docs.get(doc) else {
                    continue;
                };
                for spawner_entity in
                    cpu_root_spawner_entities(doc, content, &children_q, &scene_roots)
                {
                    if let Ok(mut spawner) = spawner_q.get_mut(spawner_entity) {
                        spawner.reset();
                    }
                }
            }
            PlaybackCommand::Respawn(doc) | PlaybackCommand::CloseDocument(doc) => {
                if !respawned.insert(doc) {
                    continue;
                }
                if matches!(cmd, PlaybackCommand::Respawn(_)) && close_requests.contains(&doc) {
                    continue;
                }
                let Ok(doc_children) = children_q.get(doc) else {
                    if matches!(cmd, PlaybackCommand::CloseDocument(_)) {
                        if let Ok(content) = docs.get(doc) {
                            layer_pool.free(content.render_layer());
                        }
                        commands.entity(doc).despawn();
                        if active.0 == Some(doc) {
                            active.0 = None;
                        }
                    }
                    continue;
                };
                match cmd {
                    PlaybackCommand::Respawn(_) => {
                        for &child in doc_children {
                            commands.entity(child).despawn();
                            released_shaders = true;
                        }
                    }
                    PlaybackCommand::CloseDocument(_) => {
                        if let Ok(content) = docs.get(doc) {
                            layer_pool.free(content.render_layer());
                        }
                        commands.entity(doc).despawn();
                        if active.0 == Some(doc) {
                            active.0 = None;
                        }
                        released_shaders = true;
                    }
                    PlaybackCommand::Restart(_) => unreachable!(),
                }
            }
        }
    }

    if released_shaders {
        // `ShaderCache` never evicts variants. Drop its strong handles only
        // once the corresponding scene teardown actually occurs.
        *shader_cache = ShaderCache::default();
    }
}

/// Drive the global sim clock from the active document's play state.
///
/// Sets `Time<EffectSimulation>` speed to follow the active document's
/// `PlaybackState.playing`: 1.0 when playing, 0.0 when paused. When no active
/// document, defaults to playing (1.0).
///
/// We use `set_relative_speed` rather than `pause()`/`unpause()` because the
/// latter are defined on `Time<Virtual>` only, not on the generic `Time<T>`
/// that `Time<EffectSimulation>` instantiates. Effect is identical: speed 0
/// means `delta_secs() == 0`, freezing the GPU sim time uniform and
/// `tick_spawners`' cycle accumulation.
///
/// This is the "follows active doc" workaround for hanabi's missing per-emitter
/// time control (see module docs).
fn drive_effect_simulation_clock(
    active: Res<ActiveDocument>,
    playback: Query<&PlaybackState>,
    mut sim_time: ResMut<Time<EffectSimulation>>,
) {
    let playing = active
        .0
        .and_then(|e| playback.get(e).ok())
        .map(|p| p.playing)
        .unwrap_or(true);
    let target = if playing { 1.0 } else { 0.0 };
    if (sim_time.relative_speed_f64() - target).abs() > f64::EPSILON {
        sim_time.set_relative_speed_f64(target);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use bevy_hanabi::{EffectParent, ParticleEffect};
    use hanabi_effect_graph::model::EmitterId;

    use super::*;
    use crate::document::SceneEmitter;

    fn gpu_hierarchy_app() -> (App, Entity, Entity, Entity) {
        let mut app = App::new();
        app.add_message::<PlaybackCommand>()
            .init_resource::<ShaderCache>()
            .init_resource::<RenderLayerPool>()
            .init_resource::<ActiveDocument>()
            .add_systems(Update, apply_playback_commands);

        let parent_emitter = EmitterId::new(1).unwrap();
        let child_emitter = EmitterId::new(2).unwrap();
        let doc = app.world_mut().spawn_empty().id();
        let scene_root = app
            .world_mut()
            .spawn((DocumentSceneRoot, EmitterSceneEntities(HashMap::new())))
            .id();
        let parent = app
            .world_mut()
            .spawn((
                SceneEmitter(parent_emitter),
                ParticleEffect::new(Handle::default()),
                Visibility::Visible,
            ))
            .id();
        let child = app
            .world_mut()
            .spawn((
                SceneEmitter(child_emitter),
                ParticleEffect::new(Handle::default()),
                EffectParent::new(parent),
                Visibility::Visible,
            ))
            .id();
        app.world_mut()
            .entity_mut(scene_root)
            .add_children(&[parent, child]);
        app.world_mut().entity_mut(doc).add_child(scene_root);
        app.world_mut()
            .entity_mut(scene_root)
            .get_mut::<EmitterSceneEntities>()
            .unwrap()
            .0
            .extend([(parent_emitter, parent), (child_emitter, child)]);
        (app, doc, scene_root, child)
    }

    #[test]
    fn gpu_hierarchy_respawn_despawns_immediately() {
        let (mut app, doc, scene_root, child) = gpu_hierarchy_app();

        app.world_mut().write_message(PlaybackCommand::Respawn(doc));
        app.update();

        assert!(app.world().get_entity(scene_root).is_err());
        assert!(app.world().get_entity(child).is_err());
        assert!(app.world().get_entity(doc).is_ok());
    }

    #[test]
    fn close_despawns_gpu_hierarchy_immediately() {
        let (mut app, doc, scene_root, _) = gpu_hierarchy_app();
        app.world_mut().resource_mut::<ActiveDocument>().0 = Some(doc);

        app.world_mut()
            .write_message(PlaybackCommand::CloseDocument(doc));
        app.update();

        assert!(app.world().get_entity(doc).is_err());
        assert!(app.world().get_entity(scene_root).is_err());
        assert_eq!(app.world().resource::<ActiveDocument>().0, None);
    }
}
