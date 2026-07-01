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

/// Whether the platform command modifier is held — Cmd on macOS, Ctrl
/// elsewhere.
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

/// Read file shortcuts and act on them.
///
/// Uses the platform command modifier — Cmd on macOS, Ctrl elsewhere — unless
/// egui currently owns keyboard focus (a text field is being edited):
///
/// - N: New, O: Open, Shift-O: Import, Q: Quit — document-independent.
/// - W: Close the active document (no-op when none is open).
/// - S: Save the active document, falling back to the Save As dialog when it
///   has no path yet; Shift-S always pops Save As.
pub fn handle_file_shortcuts(
    mut contexts: EguiContexts,
    keys: Res<ButtonInput<KeyCode>>,
    active: Res<ActiveDocument>,
    docs: Query<&DocumentContent>,
    mut app: MessageWriter<AppCommand>,
    mut pending: ResMut<PendingFileDialogs>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    if ctx.egui_wants_keyboard_input() {
        return Ok(());
    }

    if !command_modifier(&keys) {
        return Ok(());
    }
    let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);

    if keys.just_pressed(KeyCode::KeyN) && !shift {
        app.write(AppCommand::NewDocument);
    } else if keys.just_pressed(KeyCode::KeyO) {
        pending.spawn(if shift {
            DialogKind::Import
        } else {
            DialogKind::Open
        });
    } else if keys.just_pressed(KeyCode::KeyS) {
        if let Some(doc) = active.0 {
            let has_path = docs.get(doc).is_ok_and(|c| c.path().is_some());
            if has_path && !shift {
                app.write(AppCommand::SaveActive);
            } else {
                pending.spawn(DialogKind::SaveAs);
            }
        }
    } else if keys.just_pressed(KeyCode::KeyW) && !shift {
        if let Some(doc) = active.0 {
            app.write(AppCommand::CloseDocument(doc));
        }
    } else if keys.just_pressed(KeyCode::KeyQ) && !shift {
        std::process::exit(0);
    }

    Ok(())
}
