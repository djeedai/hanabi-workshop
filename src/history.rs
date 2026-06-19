//! Per-document undo/redo history.
//!
//! Each document entity carries a [`History`] component containing two
//! bounded queues:
//!
//! - `past`: inverse edits that, when re-applied via [`EditRequest`], will move
//!   the document one step backwards.
//! - `future`: edits that, when re-applied, will redo a previously undone step.
//!
//! ## Pipeline
//!
//! ```text
//! UI            HistoryRequest               EditRequest (from_history=true)
//!  └─ Ctrl-Z  ──────────────▶  history_dispatch  ───────────────▶  apply_edits
//!                                   │                                   │
//!                                   │ (pops the right queue,            │ (mutates the
//!                                   │  hands inverse to apply)          │  document; emits
//!                                   ▼                                   ▼  EditApplied)
//!                              History.past/future            record_history
//!                                                                       │
//!                              (writes back the new inverse,            ▼
//!                               flipping past<->future based      History.past/future
//!                               on EditApplied.from_history)
//! ```
//!
//! Invariant: `record_history` is the *only* writer of `History`. The
//! `history_dispatch` system only reads it (to know what to replay).

use std::collections::VecDeque;

use bevy::prelude::*;

use crate::edits::{EditApplied, EditRequest, HistoryRequest};

/// Maximum number of undoable steps retained per document.
///
/// Older steps are dropped from the back of `past` when this is exceeded.
pub const HISTORY_CAP: usize = 100;

/// Per-document undo stack.
///
/// `past` is read back-to-front by Undo; `future` is read back-to-front by
/// Redo.
#[derive(Component, Default, Debug)]
pub struct History {
    pub past: VecDeque<EditRequest>,
    pub future: VecDeque<EditRequest>,
}

/// Translate [`HistoryRequest`]s into flagged [`EditRequest`]s.
///
/// Sets the appropriate `direction`. Runs before [`crate::edits::apply_edits`].
pub fn history_dispatch(
    mut reqs: MessageReader<HistoryRequest>,
    mut edits: MessageWriter<EditRequest>,
    mut histories: Query<&mut History>,
) {
    for req in reqs.read() {
        match *req {
            HistoryRequest::Undo(doc) => {
                if let Ok(mut history) = histories.get_mut(doc)
                    && let Some(inverse) = history.past.pop_back()
                {
                    edits.write(inverse.with_undo());
                }
            }
            HistoryRequest::Redo(doc) => {
                if let Ok(mut history) = histories.get_mut(doc)
                    && let Some(inverse) = history.future.pop_back()
                {
                    edits.write(inverse.with_redo());
                }
            }
        }
    }
}

/// Push edit inverses onto the right history queue.
///
/// - A fresh edit (`Fresh`) pushes onto `past` AND clears `future` (the classic
///   "branching invalidates redo" semantics).
/// - An Undo replay (`Undo`) pushes its own inverse onto `future` so Redo can
///   replay it.
/// - A Redo replay (`Redo`) pushes its own inverse back onto `past`.
pub fn record_history(mut applied: MessageReader<EditApplied>, mut histories: Query<&mut History>) {
    for ev in applied.read() {
        let Ok(mut history) = histories.get_mut(ev.doc) else {
            continue;
        };
        match ev.direction {
            EditDirection::Fresh => {
                push_bounded(&mut history.past, ev.inverse.clone());
                history.future.clear();
            }
            EditDirection::Undo => {
                push_bounded(&mut history.future, ev.inverse.clone());
            }
            EditDirection::Redo => {
                push_bounded(&mut history.past, ev.inverse.clone());
            }
        }
    }
}

fn push_bounded(queue: &mut VecDeque<EditRequest>, item: EditRequest) {
    if queue.len() >= HISTORY_CAP {
        queue.pop_front();
    }
    queue.push_back(item);
}

/// Distinguishes the three kinds of applied edits for the history recorder.
///
/// Embedded in [`EditApplied`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditDirection {
    /// User-originated edit. Pushes inverse onto `past`, clears `future`.
    Fresh,
    /// Replay of a `past` entry (Ctrl-Z). Pushes inverse onto `future`.
    Undo,
    /// Replay of a `future` entry (Ctrl-Shift-Z / Ctrl-Y). Pushes
    /// inverse onto `past`.
    Redo,
}
