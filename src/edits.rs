//! Edit-message scaffolding.
//!
//! See `crate::document` for the architectural commitment. The rule:
//!
//! * UI code emits [`EditRequest`] messages; it never calls
//!   `DocumentContent` mutators directly.
//! * [`apply_edits`] is the **only** caller of `DocumentContent::set_*` and
//!   the only system holding `Query<&mut DocumentContent>` for write access.
//! * [`record_history`] (later) maintains the per-document undo stack.

use bevy::prelude::*;

use crate::document::DocumentContent;

/// A pending mutation to a document, dispatched by a UI panel.
#[derive(Message, Debug, Clone)]
pub enum EditRequest {
    RenameEffect { doc: Entity, new_name: String },
    /// Placeholder until Phase 3+ wires real `EffectAsset` field edits.
    SetSpawnerRate { doc: Entity, new_rate: f32 },
}

/// Emitted by [`apply_edits`] after a mutation. Carries the inverse edit.
#[derive(Message, Debug, Clone)]
pub struct EditApplied {
    pub doc: Entity,
    pub inverse: EditRequest,
    pub from_history: bool,
}

/// User-driven history navigation. Wired but not yet bound.
#[derive(Message, Debug, Clone)]
#[allow(dead_code)]
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
                (apply_edits, record_history)
                    .chain()
                    .in_set(EditSystems),
            );
    }
}

/// Systems that depend on freshly-applied edits should be ordered
/// `.after(EditSystems)`.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct EditSystems;

/// The single writer of `DocumentContent` for content edits.
pub fn apply_edits(
    mut requests: MessageReader<EditRequest>,
    mut applied: MessageWriter<EditApplied>,
    mut contents: Query<&mut DocumentContent>,
) {
    for req in requests.read() {
        let target = match req {
            EditRequest::RenameEffect { doc, .. } => *doc,
            EditRequest::SetSpawnerRate { doc, .. } => *doc,
        };
        let Ok(mut content) = contents.get_mut(target) else {
            warn!("edit request for missing document: {:?}", req);
            continue;
        };
        let inverse = match req {
            EditRequest::RenameEffect { doc, new_name } => {
                let old = content.set_name(new_name.clone());
                EditRequest::RenameEffect {
                    doc: *doc,
                    new_name: old,
                }
            }
            EditRequest::SetSpawnerRate { doc, new_rate: _ } => {
                // TODO Phase 3+: mutate the actual EffectAsset spawner.
                content.mark_dirty(true);
                EditRequest::SetSpawnerRate {
                    doc: *doc,
                    new_rate: 0.0,
                }
            }
        };
        applied.write(EditApplied {
            doc: target,
            inverse,
            from_history: false,
        });
    }
}

/// Reads applied edits and (later) appends inverses to a per-document
/// undo stack. Wired but inert for Phase 0.
pub fn record_history(mut applied: MessageReader<EditApplied>) {
    for ev in applied.read() {
        // TODO Phase 5: push `ev.inverse` onto a per-document history stack.
        let _ = ev;
    }
}
