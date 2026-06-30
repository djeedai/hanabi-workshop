//! Keyboard shortcut handling for the editor.
//!
//! Runs as a Bevy system (NOT inside egui) so we can read raw
//! `ButtonInput<KeyCode>` and gate on `!egui_ctx.egui_wants_keyboard_input()`.
//! That way text fields keep native Ctrl-Z behaviour during editing.

use bevy::prelude::*;
use bevy_egui::EguiContexts;

use crate::{
    app_commands::{AppCommand, DialogKind, PendingFileDialogs},
    document::{ActiveDocument, DocumentContent},
    edits::HistoryRequest,
};

/// Whether the platform command modifier is held — Cmd on macOS, Ctrl elsewhere.
fn command_modifier(keys: &ButtonInput<KeyCode>) -> bool {
    if cfg!(target_os = "macos") {
        keys.pressed(KeyCode::SuperLeft) || keys.pressed(KeyCode::SuperRight)
    } else {
        keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight)
    }
}

/// Read undo/redo shortcuts and emit a `HistoryRequest` for the active doc.
///
/// Reads the platform command modifier with Z / Shift-Z / Y — Cmd on macOS,
/// Ctrl elsewhere — unless egui currently owns keyboard focus (a text field is
/// being edited).
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
    if ctx.egui_wants_keyboard_input() {
        return Ok(());
    }

    if !command_modifier(&keys) {
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

/// Read the Save shortcut (command-S) and save the active document.
///
/// Mirrors the File menu's Save semantics but adds the missing-path fallback:
/// a document that has never been saved has no path, so this pops the native
/// Save As dialog instead of silently doing nothing.
pub fn handle_save_shortcut(
    mut contexts: EguiContexts,
    keys: Res<ButtonInput<KeyCode>>,
    active: Res<ActiveDocument>,
    docs: Query<&DocumentContent>,
    mut app: MessageWriter<AppCommand>,
    mut pending: ResMut<PendingFileDialogs>,
) -> Result {
    let Some(doc) = active.0 else {
        return Ok(());
    };
    let ctx = contexts.ctx_mut()?;
    if ctx.egui_wants_keyboard_input() {
        return Ok(());
    }

    if !command_modifier(&keys) || !keys.just_pressed(KeyCode::KeyS) {
        return Ok(());
    }

    let has_path = docs.get(doc).is_ok_and(|c| c.path().is_some());
    if has_path {
        app.write(AppCommand::SaveActive);
    } else {
        pending.spawn(DialogKind::SaveAs);
    }

    Ok(())
}
