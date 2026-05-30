//! Top-level editor UI: menu bar + nested document dock.

use std::collections::{HashMap, HashSet};

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use egui_dock::{DockArea, DockState, Style};

mod document_tabs;
pub mod icons;
pub mod modifier_names;
mod panels;
mod shortcuts;

pub use shortcuts::handle_history_shortcuts;

use crate::document::{
    ActiveDocument, DocumentRoot, DocumentViewports, ViewportSizeRequests,
};

/// Outer dock that hosts one tab per open document. Tabs may be torn off
/// into floating windows for side-by-side document comparison.
#[derive(Resource)]
pub struct DocumentDock {
    pub state: DockState<Entity>,
}

impl Default for DocumentDock {
    fn default() -> Self {
        Self {
            state: DockState::new(Vec::new()),
        }
    }
}

/// Egui pass: draws the menu and the outer document dock.
///
/// The inner [`DocumentTabViewer`] holds `&mut` references to the
/// queries — each call to `TabViewer::ui()` / `TabViewer::title()`
/// acquires its own short-lived `Mut<DocumentUi>` / `Mut<PlaybackState>`
/// for the tab being rendered and drops it on return, so successive
/// tabs don't conflict.
pub fn draw_editor_ui(
    mut contexts: EguiContexts,
    mut document_dock: ResMut<DocumentDock>,
    viewports: Res<DocumentViewports>,
    mut size_requests: ResMut<ViewportSizeRequests>,
    document_root: Option<Res<DocumentRoot>>,
    active: ResMut<ActiveDocument>,
    mut pending_dialogs: ResMut<crate::app_commands::PendingFileDialogs>,
    root_children: Query<&Children>,
    mut tab_data: document_tabs::TabViewerData,
    mut edit_writer: bevy::ecs::message::MessageWriter<crate::edits::EditRequest>,
    mut app_writer: bevy::ecs::message::MessageWriter<crate::app_commands::AppCommand>,
    mut playback_writer: bevy::ecs::message::MessageWriter<crate::playback::PlaybackCommand>,
    mut history_writer: bevy::ecs::message::MessageWriter<crate::edits::HistoryRequest>,
    mut cam_writer: bevy::ecs::message::MessageWriter<
        crate::plugins::camera_control::CameraControlMessage,
    >,
) -> Result {
    let Some(root) = document_root else {
        return Ok(());
    };
    let ordered_docs: Vec<Entity> = root_children
        .get(root.0)
        .map(|c| c.iter().collect())
        .unwrap_or_default();

    sync_document_tabs(&mut document_dock, &ordered_docs);

    let viewport_textures = resolve_viewport_textures(&mut contexts, &viewports);

    let active_has_path = active
        .0
        .and_then(|e| tab_data.docs.get(e).ok())
        .map(|(c, _, _)| c.path().is_some())
        .unwrap_or(false);

    let ctx = contexts.ctx_mut()?;
    draw_menu_bar(
        ctx,
        &mut app_writer,
        &mut pending_dialogs,
        &mut history_writer,
        active.0,
        active_has_path,
    );

    let mut tab_viewer = document_tabs::DocumentTabViewer {
        data: &mut tab_data,
        viewport_textures: &viewport_textures,
        size_requests: &mut size_requests,
        edits: &mut edit_writer,
        playback: &mut playback_writer,
        cam_msgs: &mut cam_writer,
    };
    DockArea::new(&mut document_dock.state)
        .style(Style::from_egui(ctx.style().as_ref()))
        .show_leaf_collapse_buttons(false)
        .show_leaf_close_all_buttons(false)
        .show(ctx, &mut tab_viewer);

    // Sync the focused outer tab into ActiveDocument.
    let focused = document_dock
        .state
        .find_active_focused()
        .map(|(_, tab)| *tab);
    let mut active = active;
    if active.0 != focused {
        active.0 = focused;
    }

    Ok(())
}

fn draw_menu_bar(
    ctx: &egui::Context,
    app: &mut bevy::ecs::message::MessageWriter<crate::app_commands::AppCommand>,
    pending: &mut crate::app_commands::PendingFileDialogs,
    history: &mut bevy::ecs::message::MessageWriter<crate::edits::HistoryRequest>,
    active: Option<Entity>,
    active_has_path: bool,
) {
    use crate::app_commands::{AppCommand, DialogKind};
    use crate::edits::HistoryRequest;

    egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("New").clicked() {
                    app.write(AppCommand::NewDocument);
                    ui.close();
                }
                if ui.button("Open…").clicked() {
                    pending.spawn(DialogKind::Open);
                    ui.close();
                }
                ui.add_enabled_ui(active.is_some(), |ui| {
                    let save_btn = ui.add_enabled(active_has_path, egui::Button::new("Save"));
                    if save_btn.clicked() {
                        app.write(AppCommand::SaveActive);
                        ui.close();
                    }
                    if ui.button("Save As…").clicked() {
                        pending.spawn(DialogKind::SaveAs);
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Close").clicked() {
                        if let Some(e) = active {
                            app.write(AppCommand::CloseDocument(e));
                        }
                        ui.close();
                    }
                });
                ui.separator();
                if ui.button("Exit").clicked() {
                    std::process::exit(0);
                }
            });
            ui.menu_button("Edit", |ui| {
                ui.add_enabled_ui(active.is_some(), |ui| {
                    if ui
                        .add(egui::Button::new("Undo").shortcut_text("Ctrl+Z"))
                        .clicked()
                    {
                        if let Some(e) = active {
                            history.write(HistoryRequest::Undo(e));
                        }
                        ui.close();
                    }
                    if ui
                        .add(egui::Button::new("Redo").shortcut_text("Ctrl+Shift+Z"))
                        .clicked()
                    {
                        if let Some(e) = active {
                            history.write(HistoryRequest::Redo(e));
                        }
                        ui.close();
                    }
                });
            });
            ui.menu_button("View", |ui| {
                ui.label("(layout reset TBD)");
            });
        });
    });
}

/// Ensures the outer dock's tabs match the current set of documents.
fn sync_document_tabs(dock: &mut DocumentDock, ordered: &[Entity]) {
    let current: HashSet<Entity> = dock.state.iter_all_tabs().map(|(_, e)| *e).collect();
    let wanted: HashSet<Entity> = ordered.iter().copied().collect();

    for doc in ordered {
        if !current.contains(doc) {
            dock.state.push_to_focused_leaf(*doc);
        }
    }

    let stale: Vec<Entity> = current.difference(&wanted).copied().collect();
    for doc in stale {
        let locations: Vec<_> = dock
            .state
            .iter_all_tabs()
            .filter(|(_, t)| **t == doc)
            .map(|(loc, _)| loc)
            .collect();
        for (surface, node) in locations {
            let _ = dock.state.remove_tab((surface, node, egui_dock::TabIndex(0)));
        }
    }
}

fn resolve_viewport_textures(
    contexts: &mut EguiContexts,
    viewports: &DocumentViewports,
) -> HashMap<(Entity, usize), egui::TextureId> {
    let mut out = HashMap::new();
    for (doc, slots) in &viewports.by_doc {
        for (vp_idx, handle) in &slots.images {
            if let Some(tex) = contexts.image_id(handle) {
                out.insert((*doc, *vp_idx), tex);
            }
        }
    }
    out
}
