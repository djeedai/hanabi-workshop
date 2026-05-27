use bevy::prelude::*;
use bevy_egui::egui;

/// Outline (modifier list) panel. Phase 5 fleshes this out.
pub fn show(ui: &mut egui::Ui, _doc_entity: Entity) {
    ui.heading("Outline");
    ui.separator();
    ui.label("(modifier list — Phase 5)");
}
