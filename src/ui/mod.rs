//! Top-level editor UI: menu bar + nested document dock.

use std::collections::{HashMap, HashSet};

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use egui_dock::{DockArea, DockState, Style};

mod document_tabs;
mod panels;

use crate::document::{DocumentContent, DocumentRoot, DocumentUi, DocumentViewports, PanelKind, ViewportSizeRequests};

/// Per-document working snapshot used by the outer TabViewer during a single
/// egui pass. The inner dock is *moved out* of the live `DocumentUi`
/// component for the duration of the pass, then moved back afterward.
pub struct DocTabState {
    pub entity: Entity,
    pub name: String,
    pub dirty: bool,
    pub dock: DockState<PanelKind>,
}

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
pub fn draw_editor_ui(
    mut contexts: EguiContexts,
    mut document_dock: ResMut<DocumentDock>,
    viewports: Res<DocumentViewports>,
    mut size_requests: ResMut<ViewportSizeRequests>,
    document_root: Option<Res<DocumentRoot>>,
    active: Res<crate::document::ActiveDocument>,
    mut pending_dialogs: ResMut<crate::app_commands::PendingFileDialogs>,
    root_children: Query<&Children>,
    mut docs: Query<(Entity, &DocumentContent, &mut DocumentUi)>,
    mut edit_writer: bevy::ecs::message::MessageWriter<crate::edits::EditRequest>,
    mut app_writer: bevy::ecs::message::MessageWriter<crate::app_commands::AppCommand>,
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
        .and_then(|e| docs.get(e).ok())
        .map(|(_, c, _)| c.path().is_some())
        .unwrap_or(false);

    // Snapshot each document, taking its inner dock out so the TabViewer
    // doesn't need to hold a Query borrow across the egui pass.
    let mut tab_states: HashMap<Entity, DocTabState> = HashMap::new();
    for (entity, content, mut ui_state) in docs.iter_mut() {
        let dock = std::mem::replace(&mut ui_state.dock, DockState::new(Vec::new()));
        tab_states.insert(
            entity,
            DocTabState {
                entity,
                name: content.name().to_string(),
                dirty: content.dirty(),
                dock,
            },
        );
    }

    let ctx = contexts.ctx_mut()?;
    draw_menu_bar(ctx, &mut app_writer, &mut pending_dialogs, active.0, active_has_path);

    let mut tab_viewer = document_tabs::DocumentTabViewer {
        tab_states: &mut tab_states,
        viewport_textures: &viewport_textures,
        size_requests: &mut size_requests,
        edits: &mut edit_writer,
    };
    DockArea::new(&mut document_dock.state)
        .style(Style::from_egui(ctx.style().as_ref()))
        .show(ctx, &mut tab_viewer);

    // Move docks back into the live components.
    for (entity, _content, mut ui_state) in docs.iter_mut() {
        if let Some(state) = tab_states.remove(&entity) {
            ui_state.dock = state.dock;
        }
    }

    Ok(())
}

fn draw_menu_bar(
    ctx: &egui::Context,
    app: &mut bevy::ecs::message::MessageWriter<crate::app_commands::AppCommand>,
    pending: &mut crate::app_commands::PendingFileDialogs,
    active: Option<Entity>,
    active_has_path: bool,
) {
    use crate::app_commands::{AppCommand, DialogKind};

    egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
        egui::menu::bar(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("New").clicked() {
                    app.write(AppCommand::NewDocument);
                    ui.close_menu();
                }
                if ui.button("Open…").clicked() {
                    pending.spawn(DialogKind::Open);
                    ui.close_menu();
                }
                ui.add_enabled_ui(active.is_some(), |ui| {
                    let save_btn = ui.add_enabled(active_has_path, egui::Button::new("Save"));
                    if save_btn.clicked() {
                        app.write(AppCommand::SaveActive);
                        ui.close_menu();
                    }
                    if ui.button("Save As…").clicked() {
                        pending.spawn(DialogKind::SaveAs);
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Close").clicked() {
                        if let Some(e) = active {
                            app.write(AppCommand::CloseDocument(e));
                        }
                        ui.close_menu();
                    }
                });
                ui.separator();
                if ui.button("Exit").clicked() {
                    std::process::exit(0);
                }
            });
            ui.menu_button("Edit", |ui| {
                if ui.button("Undo").clicked() {
                    ui.close_menu();
                }
                if ui.button("Redo").clicked() {
                    ui.close_menu();
                }
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
