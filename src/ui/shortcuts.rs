//! Keyboard shortcut handling for the editor.
//!
//! Runs as a Bevy system (NOT inside egui) so we can read raw
//! `ButtonInput<KeyCode>` and gate on `!egui_ctx.wants_keyboard_input()`.
//! That way text fields keep native Ctrl-Z behaviour during editing.

use bevy::prelude::*;
use bevy_egui::EguiContexts;

use crate::{document::ActiveDocument, edits::HistoryRequest};

/// Read undo/redo shortcuts and emit a `HistoryRequest` for the active doc.
///
/// Reads Ctrl-Z / Ctrl-Shift-Z / Ctrl-Y, unless egui currently owns keyboard
/// focus (a text field is being edited).
pub fn handle_history_shortcuts(
    mut contexts: EguiContexts,
    keys: Res<ButtonInput<KeyCode>>,
    active: Res<ActiveDocument>,
    mut history: MessageWriter<HistoryRequest>,
) -> Result {
    let Some(doc) = active.0 else {
        return Ok(());
    };
    let ctx = contexts.ctx_mut()?;
    if ctx.wants_keyboard_input() {
        return Ok(());
    }

    let ctrl = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
    if !ctrl {
        return Ok(());
    }
    let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);

    if keys.just_pressed(KeyCode::KeyZ) {
        if shift {
            history.write(HistoryRequest::Redo(doc));
        } else {
            history.write(HistoryRequest::Undo(doc));
        }
    } else if keys.just_pressed(KeyCode::KeyY) {
        history.write(HistoryRequest::Redo(doc));
    }

    Ok(())
}
