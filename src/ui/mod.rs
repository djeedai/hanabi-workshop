use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use egui_dock::{DockArea, Style};

use crate::plugins::editor::ViewportImages;

mod dock;
mod panels;

pub use dock::{EditorTab, EditorUiState};

/// Top-level egui system: draws the menu bar and the docking area.
pub fn draw_editor_ui(
    mut contexts: EguiContexts,
    mut ui_state: ResMut<EditorUiState>,
    viewport_images: Res<ViewportImages>,
) -> Result {
    // Resolve viewport image handles into egui texture ids before we borrow the context mutably.
    let viewport_textures: Vec<egui::TextureId> = viewport_images
        .0
        .iter()
        .map(|h| contexts.image_id(h).expect("viewport image not registered"))
        .collect();

    let ctx = contexts.ctx_mut()?;

    egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
        egui::menu::bar(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Open…").clicked() {
                    ui.close_menu();
                }
                if ui.button("Save").clicked() {
                    ui.close_menu();
                }
                if ui.button("Save As…").clicked() {
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("Exit").clicked() {
                    std::process::exit(0);
                }
            });

            ui.menu_button("View", |ui| {
                if ui.button("Reset Layout").clicked() {
                    ui_state.reset_layout();
                    ui.close_menu();
                }
            });
        });
    });

    let mut tab_viewer = panels::EditorTabViewer {
        viewport_textures: &viewport_textures,
    };
    DockArea::new(&mut ui_state.dock_state)
        .style(Style::from_egui(ctx.style().as_ref()))
        .show(ctx, &mut tab_viewer);

    Ok(())
}
