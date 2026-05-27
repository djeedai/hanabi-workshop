use bevy::prelude::*;
use bevy_egui::egui;

use crate::edits::EditRequest;

/// Properties panel for the given document. Phase 5 fleshes this out.
pub fn show(
    ui: &mut egui::Ui,
    _doc_entity: Entity,
    _edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
) {
    ui.heading("Properties");
    ui.separator();
    ui.label("(modifier property editing — Phase 5)");
}
