//! Per-document particle playback: play/pause, restart, respawn.
//!
//! Each document entity carries a [`PlaybackState`] component. The UI
//! mutates `playing` directly (it's plain state). One system pushes
//! the **active document's** `playing` into the global
//! `Time<EffectSimulation>::set_relative_speed` (1.0 if playing, 0.0
//! if paused).
//!
//! ## Upstream gap
//!
//! `bevy_hanabi` 0.18 does **not** expose any per-effect time scale.
//! There is only one global clock (`Time<EffectSimulation>`) that
//! drives `tick_spawners` and the GPU simulation. Pausing it freezes
//! every open document at once. We work around this by gating the
//! global clock on the *active* document — switching tabs effectively
//! switches whose playback state owns the clock. This matches a
//! single-doc-at-a-time workflow. A real per-document timeline would
//! need an upstream feature (e.g. an `EffectTimeScale(f32)` component
//! that `tick_spawners` multiplies into `dt`, plus a per-instance
//! time uniform in the shader). Worth filing as an issue when we have
//! a multi-doc need.
//!
//! [`PlaybackCommand`] is reserved for actions the UI cannot perform
//! itself: `Restart` (reset cycle time) and `Respawn` (despawn the
//! scene root so reconciliation rebuilds it). Per §6 coding
//! conventions, *state* lives on the component and is mutated
//! directly; *actions* go through the message channel.

use bevy::prelude::*;
use bevy_hanabi::{EffectSimulation, EffectSimulationTime, EffectSpawner};

use crate::document::{ActiveDocument, DocumentSceneRoot};

/// Per-document playback state. The active document's `playing`
/// value is mirrored into the global `Time<EffectSimulation>` speed
/// each frame.
#[derive(Component, Debug, Clone, Copy)]
pub struct PlaybackState {
    pub playing: bool,
}

impl Default for PlaybackState {
    fn default() -> Self {
        Self { playing: true }
    }
}

/// One-shot playback actions, addressed to a document entity. Only
/// includes actions that require `Commands` or the `EffectSpawner`
/// query (i.e. that the UI cannot perform itself).
#[derive(Message, Debug, Clone, Copy)]
pub enum PlaybackCommand {
    /// Reset the document's spawner — clears accumulated cycle time so
    /// emission restarts cleanly. Does not despawn existing particles.
    Restart(Entity),
    /// Debug action: despawn the document's scene root so reconciliation
    /// recreates it on the next frame, giving a fresh `ParticleEffect`.
    /// Useful when in-place asset edits aren't picked up correctly.
    Respawn(Entity),
}

pub struct PlaybackPlugin;

impl Plugin for PlaybackPlugin {
    fn build(&self, app: &mut App) {
        // `apply_playback_commands` MUST run after `apply_edits` (via
        // `EditSystems`) so that a `Respawn` written by an edit lands
        // and despawns the scene root in the *same* frame's
        // command-flush at end of `Update`. Otherwise hanabi's
        // `compile_effects` runs in `PostUpdate` against the
        // already-mutated asset while the old `ParticleEffect` entity
        // (and its `CachedPipelines` keyed on the previous property
        // layout) is still alive — producing a wgpu validation crash
        // when the property buffer size shifts (e.g. adding a user
        // property). See `hanabi_gaps.md` §1.6.
        app.add_message::<PlaybackCommand>().add_systems(
            Update,
            (
                apply_playback_commands.after(crate::edits::EditSystems),
                drive_effect_simulation_clock,
            ),
        );
    }
}

fn find_spawner_entity(
    doc: Entity,
    children_q: &Query<&Children>,
    scene_roots: &Query<(), With<DocumentSceneRoot>>,
    spawners: &Query<(), With<EffectSpawner>>,
) -> Option<Entity> {
    let doc_children = children_q.get(doc).ok()?;
    for &child in doc_children {
        if scene_roots.get(child).is_err() {
            continue;
        }
        let scene_children = children_q.get(child).ok()?;
        for &grandchild in scene_children {
            if spawners.get(grandchild).is_ok() {
                return Some(grandchild);
            }
        }
    }
    None
}

pub fn apply_playback_commands(
    mut commands: Commands,
    mut reader: MessageReader<PlaybackCommand>,
    mut spawner_q: Query<&mut EffectSpawner>,
    children_q: Query<&Children>,
    scene_roots: Query<(), With<DocumentSceneRoot>>,
    scene_root_children: Query<Entity, With<DocumentSceneRoot>>,
    spawners: Query<(), With<EffectSpawner>>,
) {
    for cmd in reader.read() {
        match *cmd {
            PlaybackCommand::Restart(doc) => {
                if let Some(spawner_entity) =
                    find_spawner_entity(doc, &children_q, &scene_roots, &spawners)
                {
                    if let Ok(mut spawner) = spawner_q.get_mut(spawner_entity) {
                        spawner.reset();
                    }
                }
            }
            PlaybackCommand::Respawn(doc) => {
                let Ok(doc_children) = children_q.get(doc) else {
                    continue;
                };
                for &child in doc_children {
                    if scene_root_children.get(child).is_ok() {
                        commands.entity(child).despawn();
                    }
                }
            }
        }
    }
}

/// Sets `Time<EffectSimulation>` speed to follow the active document's
/// `PlaybackState.playing`: 1.0 when playing, 0.0 when paused. When no
/// active document, defaults to playing (1.0).
///
/// We use `set_relative_speed` rather than `pause()`/`unpause()`
/// because the latter are defined on `Time<Virtual>` only, not on
/// the generic `Time<T>` that `Time<EffectSimulation>` instantiates.
/// Effect is identical: speed 0 means `delta_secs() == 0`, freezing
/// the GPU sim time uniform and `tick_spawners`' cycle accumulation.
///
/// This is the "follows active doc" workaround for hanabi's missing
/// per-effect time control (see module docs).
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
