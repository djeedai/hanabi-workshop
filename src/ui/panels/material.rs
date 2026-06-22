//! Material panel: the effect's texture slots.
//!
//! Lists every [`TextureSlotDef`] on the canonical [`EffectGraph`] in sampling-
//! index order. Each row offers rename, image bind/clear (via a native file
//! dialog), reorder, and remove, plus a count of the image nodes referencing
//! the slot. Slots are addressed by stable [`SlotId`], so renames and reorders
//! never break edits in flight. All mutations are emitted as [`EditRequest`];
//! the panel never touches the graph directly.
//!
//! [`EffectGraph`]: crate::effect_graph::model::EffectGraph
//! [`TextureSlotDef`]: crate::effect_graph::model::TextureSlotDef
//! [`SlotId`]: crate::effect_graph::model::SlotId

use bevy::prelude::*;
use bevy_egui::egui;

use crate::{
    app_commands::{DialogKind, PendingFileDialogs},
    edits::{EditKind, EditRequest},
    effect_graph::model::{
        EffectGraph, ExprNode, NodePayload, SlotId, TextureSlotDef, TextureValue,
    },
    ui::icons::{ICON_ARROW_DOWN, ICON_ARROW_UP, ICON_FOLDER_OPEN, ICON_PLUS, ICON_XMARK},
};

/// Top-level entry point for the Material tab.
///
/// Lists every texture slot; a pure-UI helper that never mutates the graph
/// directly — it only emits [`EditRequest`] and pops file dialogs.
pub fn show_panel(
    ui: &mut egui::Ui,
    doc: Entity,
    graph: &EffectGraph,
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
    pending: &mut PendingFileDialogs,
) {
    let slots = &graph.textures;
    egui::ScrollArea::vertical().show(ui, |ui| {
        if slots.is_empty() {
            ui.weak("(no texture slots)");
        }
        let count = slots.len();
        for (index, slot) in slots.iter().enumerate() {
            let refs = reference_count(graph, slot.id);
            slot_row(ui, doc, slot, index, count, refs, edits, pending);
        }
        ui.add_space(4.0);
        if ui
            .button(format!("{ICON_PLUS}  Add slot"))
            .on_hover_text("Add an unbound texture slot")
            .clicked()
        {
            edits.write(EditRequest::new(doc, EditKind::AddTextureSlot));
        }
    });
}

/// How many image nodes reference the slot `id`.
fn reference_count(graph: &EffectGraph, id: SlotId) -> usize {
    graph
        .nodes
        .iter()
        .filter(|n| matches!(&n.payload, NodePayload::Expr(ExprNode::Image(s)) if *s == id))
        .count()
}

#[allow(clippy::too_many_arguments)]
fn slot_row(
    ui: &mut egui::Ui,
    doc: Entity,
    slot: &TextureSlotDef,
    index: usize,
    count: usize,
    refs: usize,
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
    pending: &mut PendingFileDialogs,
) {
    let id = slot.id;
    ui.group(|ui| {
        ui.horizontal(|ui| {
            // Sampling index: the slot's position in the list and its shader
            // binding point.
            ui.monospace(format!("[{index}]"));

            // Name (rename on lost_focus). Draft keyed by the stable slot id so
            // it survives the rename round-trip.
            let draft_id = egui::Id::new(("slot-name", doc, id));
            let mut draft: String = ui.ctx().data_mut(|d| {
                d.get_temp::<String>(draft_id)
                    .unwrap_or_else(|| slot.name.to_string())
            });
            let resp = ui.add(egui::TextEdit::singleline(&mut draft).desired_width(120.0));
            if resp.has_focus() || resp.changed() {
                ui.ctx()
                    .data_mut(|d| d.insert_temp(draft_id, draft.clone()));
            }
            if resp.lost_focus() {
                let trimmed = draft.trim().to_string();
                ui.ctx().data_mut(|d| d.remove::<String>(draft_id));
                if !trimmed.is_empty() && trimmed != *slot.name {
                    edits.write(EditRequest::new(
                        doc,
                        EditKind::RenameTextureSlot {
                            id,
                            new: trimmed.into(),
                        },
                    ));
                }
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // Removal is only safe when no image node points at the slot;
                // otherwise those nodes would dangle.
                let removable = refs == 0;
                let remove = ui
                    .add_enabled(removable, egui::Button::new(ICON_XMARK.to_string()))
                    .on_hover_text("Remove this slot")
                    .on_disabled_hover_text("In use by an image node");
                if remove.clicked() {
                    edits.write(EditRequest::new(doc, EditKind::RemoveTextureSlot { id }));
                }

                // Reorder. Moving a slot reassigns sampling indices, so any
                // raw-`u32` sampler link now reads a different slot.
                let can_down = index + 1 < count;
                if ui
                    .add_enabled(can_down, egui::Button::new(ICON_ARROW_DOWN.to_string()))
                    .on_hover_text("Move down (higher index)")
                    .clicked()
                {
                    edits.write(EditRequest::new(
                        doc,
                        EditKind::ReorderTextureSlot { id, to: index + 1 },
                    ));
                }
                let can_up = index > 0;
                if ui
                    .add_enabled(can_up, egui::Button::new(ICON_ARROW_UP.to_string()))
                    .on_hover_text("Move up (lower index)")
                    .clicked()
                {
                    edits.write(EditRequest::new(
                        doc,
                        EditKind::ReorderTextureSlot { id, to: index - 1 },
                    ));
                }
            });
        });

        // Image binding row.
        ui.horizontal(|ui| {
            ui.label("image:");
            match &slot.image {
                TextureValue::Asset(path) => {
                    ui.monospace(path.to_string());
                }
                TextureValue::Slot { name } => {
                    ui.weak(format!("(unbound: {name})"));
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if matches!(slot.image, TextureValue::Asset(_))
                    && ui
                        .button("Clear")
                        .on_hover_text("Unbind this image")
                        .clicked()
                {
                    edits.write(EditRequest::new(
                        doc,
                        EditKind::SetTextureSlotImage {
                            id,
                            image: TextureValue::default(),
                        },
                    ));
                }
                if ui
                    .button(format!("{ICON_FOLDER_OPEN}  Bind…"))
                    .on_hover_text("Pick an image file")
                    .clicked()
                {
                    pending.spawn(DialogKind::BindTexture { doc, slot: id });
                }
            });
        });

        // Reference count / orphan indicator.
        if refs == 0 {
            ui.weak("orphan — no image node references this slot");
        } else {
            ui.weak(format!(
                "{refs} reference{}",
                if refs == 1 { "" } else { "s" }
            ));
        }
    });
}
