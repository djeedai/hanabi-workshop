//! Unsaved-changes confirmation guard for closing documents and quitting.
//!
//! Closing a document or quitting the app can discard unsaved edits. Instead
//! of acting immediately, the menu, keyboard shortcuts and tab close button
//! emit [`AppCommand::RequestCloseDocument`] / [`AppCommand::RequestQuit`],
//! which this module intercepts. Clean documents are closed straight away;
//! dirty ones raise a modal offering **Save**, **Don't Save** or **Cancel**.
//!
//! Quitting with several dirty documents queues them and prompts for each in
//! turn. Saving an untitled document opens the native Save As dialog and
//! resumes once it resolves; cancelling that dialog aborts the whole operation
//! so no work is lost.
//!
//! [`AppCommand::RequestCloseDocument`]: crate::app_commands::AppCommand::RequestCloseDocument
//! [`AppCommand::RequestQuit`]: crate::app_commands::AppCommand::RequestQuit

use std::{collections::VecDeque, path::PathBuf};

use bevy::{
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future},
    window::WindowCloseRequested,
};
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};

use crate::{app_commands::AppCommand, document::DocumentContent};

/// Wires up the unsaved-changes confirmation guard.
pub struct ConfirmPlugin;

impl Plugin for ConfirmPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ConfirmState>()
            .add_systems(
                Update,
                (begin_confirm, on_window_close_requested, poll_guard_save),
            )
            .add_systems(EguiPrimaryContextPass, draw_confirm_dialog);

        #[cfg(target_os = "macos")]
        app.add_systems(Update, patch_macos_quit_menu);
    }
}

/// Retarget the macOS default-menu Quit item so Cmd+Q honours the guard.
///
/// winit installs an App menu whose Quit item binds Cmd+Q to AppKit's
/// `terminate:`, exiting the process before Bevy runs. Repointing the item's
/// action to `performClose:` routes both the Cmd+Q key equivalent and a mouse
/// click through winit's `WindowCloseRequested` into
/// [`on_window_close_requested`], and thus the unsaved-changes guard, while
/// keeping the native Cmd+Q hint. Runs each frame until the menu exists and is
/// patched, then does nothing.
#[cfg(target_os = "macos")]
#[allow(
    unsafe_code,
    reason = "retargeting AppKit's menu item requires objc2's unsafe setters"
)]
fn patch_macos_quit_menu(_main_thread: bevy::ecs::system::NonSendMarker, mut done: Local<bool>) {
    use objc2::{MainThreadMarker, sel};
    use objc2_app_kit::NSApplication;

    if *done {
        return;
    }
    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    let app = NSApplication::sharedApplication(mtm);
    let Some(main_menu) = app.mainMenu() else {
        return; // menu not built yet; retry next frame
    };

    for i in 0..main_menu.numberOfItems() {
        let Some(submenu) = main_menu.itemAtIndex(i).and_then(|item| item.submenu()) else {
            continue;
        };
        for j in 0..submenu.numberOfItems() {
            let Some(item) = submenu.itemAtIndex(j) else {
                continue;
            };
            if item.action() != Some(sel!(terminate:)) {
                continue;
            }
            // SAFETY: called on the main thread with a live menu item.
            // `performClose:` is answered by the responder chain's key window,
            // which winit's delegate handles by emitting `WindowCloseRequested`
            // without actually closing.
            unsafe {
                item.setTarget(None);
                item.setAction(Some(sel!(performClose:)));
            }
            *done = true;
            return;
        }
    }
}

/// The in-flight confirmation prompt, if any.
///
/// A single prompt walks a queue of dirty documents front-to-back. For a
/// close request the queue holds just the one document; for a quit it holds
/// every dirty document.
#[derive(Resource, Default)]
pub struct ConfirmState {
    prompt: Option<Prompt>,
}

struct Prompt {
    /// Dirty documents still awaiting a decision; the front is shown.
    queue: VecDeque<Entity>,
    /// Whether this prompt should quit the app once the queue drains.
    quitting: bool,
    /// A native Save As dialog running for the front document.
    ///
    /// While set, the modal is hidden and [`poll_guard_save`] drives the
    /// prompt forward when the dialog resolves.
    save_task: Option<Task<Option<PathBuf>>>,
}

/// Route [`AppCommand::RequestCloseDocument`] / [`RequestQuit`] into the guard.
///
/// Clean targets are acted on immediately; dirty ones open (or, if a prompt is
/// already active, are ignored to avoid stacking modals).
///
/// [`RequestQuit`]: AppCommand::RequestQuit
fn begin_confirm(
    // Reading and writing `AppCommand` in one system conflicts on the message
    // buffer, so gate both behind a `ParamSet`.
    mut messages: ParamSet<(MessageReader<AppCommand>, MessageWriter<AppCommand>)>,
    mut state: ResMut<ConfirmState>,
    docs: Query<(Entity, &DocumentContent)>,
) {
    let requests: Vec<Request> = messages
        .p0()
        .read()
        .filter_map(|cmd| match cmd {
            AppCommand::RequestCloseDocument(entity) => Some(Request::Close(*entity)),
            AppCommand::RequestQuit => Some(Request::Quit),
            _ => None,
        })
        .collect();

    for request in requests {
        // A prompt is already up: ignore further lifecycle requests so we
        // never stack modals.
        if state.prompt.is_some() {
            break;
        }
        match request {
            Request::Close(entity) => match docs.get(entity) {
                Ok((_, content)) if content.dirty() => {
                    state.prompt = Some(Prompt {
                        queue: VecDeque::from([entity]),
                        quitting: false,
                        save_task: None,
                    });
                }
                // Clean (or already gone): close without asking.
                _ => {
                    messages.p1().write(AppCommand::CloseDocument(entity));
                }
            },
            Request::Quit => {
                let dirty: VecDeque<Entity> = docs
                    .iter()
                    .filter(|(_, content)| content.dirty())
                    .map(|(entity, _)| entity)
                    .collect();
                if dirty.is_empty() {
                    std::process::exit(0);
                }
                state.prompt = Some(Prompt {
                    queue: dirty,
                    quitting: true,
                    save_task: None,
                });
            }
        }
    }
}

enum Request {
    Close(Entity),
    Quit,
}

/// Route the OS window-close request through the unsaved-changes guard.
///
/// Bevy's own close handler is disabled (`close_when_requested = false`), so
/// closing the window behaves like [`AppCommand::RequestQuit`].
fn on_window_close_requested(
    mut closed: MessageReader<WindowCloseRequested>,
    mut app: MessageWriter<AppCommand>,
) {
    if closed.read().next().is_some() {
        closed.clear();
        app.write(AppCommand::RequestQuit);
    }
}

/// Draw the confirmation modal for the front document of the active prompt.
fn draw_confirm_dialog(
    mut contexts: EguiContexts,
    mut state: ResMut<ConfirmState>,
    mut app: MessageWriter<AppCommand>,
    docs: Query<(Entity, &DocumentContent)>,
) -> Result {
    let Some(prompt) = state.prompt.as_ref() else {
        return Ok(());
    };
    // A native Save As dialog is up for the front document; poll_guard_save
    // owns the prompt until it resolves.
    if prompt.save_task.is_some() {
        return Ok(());
    }
    let Some(&entity) = prompt.queue.front() else {
        return Ok(());
    };
    // The document vanished from under us (shouldn't happen): drop it.
    let Ok((_, content)) = docs.get(entity) else {
        advance(&mut state, entity);
        return Ok(());
    };
    let name = content.name().to_string();
    let has_path = content.path().is_some();

    let ctx = contexts.ctx_mut()?;
    // Match the menu popups: darker fill, no border.
    let style = ctx.global_style();
    let frame = egui::Frame::popup(&style)
        .fill(crate::ui::darken_popup_fill(style.visuals.window_fill))
        .stroke(egui::Stroke::NONE);
    let mut cancel = false;
    let mut decision = None;
    let response = egui::Modal::new(egui::Id::new("confirm_unsaved"))
        .frame(frame)
        .show(ctx, |ui| {
            ui.set_min_width(340.0);
            ui.heading("Unsaved changes");
            ui.add_space(8.0);
            ui.label(format!(
                "\"{name}\" has unsaved changes. Do you want to save them?"
            ));
            ui.add_space(16.0);
            let button_size = egui::vec2(96.0, 28.0);
            let gap = 12.0;
            let total = button_size.x * 3.0 + gap * 2.0;
            let indent = ((ui.available_width() - total).max(0.0)) / 2.0;
            ui.horizontal(|ui| {
                ui.add_space(indent);
                ui.spacing_mut().item_spacing.x = gap;
                if ui
                    .add_sized(button_size, egui::Button::new("Save"))
                    .clicked()
                {
                    decision = Some(Action::Save);
                }
                if ui
                    .add_sized(button_size, egui::Button::new("Don't Save"))
                    .clicked()
                {
                    decision = Some(Action::DontSave);
                }
                if ui
                    .add_sized(button_size, egui::Button::new("Cancel"))
                    .clicked()
                {
                    cancel = true;
                }
            });
        });
    let mut action = if cancel {
        Some(Action::Cancel)
    } else {
        decision
    };
    if response.should_close() {
        action = Some(Action::Cancel);
    }

    match action {
        None => {}
        Some(Action::Cancel) => state.prompt = None,
        Some(Action::DontSave) => {
            if !state.prompt.as_ref().unwrap().quitting {
                app.write(AppCommand::CloseDocument(entity));
            }
            advance(&mut state, entity);
        }
        Some(Action::Save) => {
            if has_path {
                app.write(AppCommand::SaveDocument(entity));
                if !state.prompt.as_ref().unwrap().quitting {
                    app.write(AppCommand::CloseDocument(entity));
                }
                advance(&mut state, entity);
            } else {
                // Untitled: pick a path, then resume in poll_guard_save.
                let pool = AsyncComputeTaskPool::get();
                let task = pool.spawn(async {
                    rfd::AsyncFileDialog::new()
                        .add_filter("Effect Graph", &["hnb"])
                        .save_file()
                        .await
                        .map(|h| h.path().to_path_buf())
                });
                state.prompt.as_mut().unwrap().save_task = Some(task);
            }
        }
    }
    Ok(())
}

/// Resume a prompt once its front document's native Save As dialog resolves.
///
/// A chosen path saves (and closes, for a close request) then advances; a
/// cancelled dialog aborts the whole prompt so nothing is lost.
fn poll_guard_save(mut state: ResMut<ConfirmState>, mut app: MessageWriter<AppCommand>) {
    let Some(prompt) = state.prompt.as_mut() else {
        return;
    };
    let Some(task) = prompt.save_task.as_mut() else {
        return;
    };
    let Some(result) = block_on(future::poll_once(task)) else {
        return; // dialog still open
    };
    prompt.save_task = None;
    let Some(&entity) = prompt.queue.front() else {
        state.prompt = None;
        return;
    };
    match result {
        Some(path) => {
            app.write(AppCommand::SaveDocumentAs(entity, path));
            if !prompt.quitting {
                app.write(AppCommand::CloseDocument(entity));
            }
            advance(&mut state, entity);
        }
        // Cancelled the Save As dialog: abort, keeping all work.
        None => state.prompt = None,
    }
}

/// Pop the front document and either advance or finish the prompt.
///
/// When the queue drains, a quitting prompt exits the process; otherwise the
/// prompt is cleared.
fn advance(state: &mut ConfirmState, entity: Entity) {
    let Some(prompt) = state.prompt.as_mut() else {
        return;
    };
    if prompt.queue.front() == Some(&entity) {
        prompt.queue.pop_front();
    }
    if prompt.queue.is_empty() {
        if prompt.quitting {
            std::process::exit(0);
        }
        state.prompt = None;
    }
}

enum Action {
    Cancel,
    DontSave,
    Save,
}
