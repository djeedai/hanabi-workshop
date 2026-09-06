//! Material panel: the emitter's texture slots.
//!
//! Lists every [`TextureSlotDef`] on the canonical [`EmitterGraph`] in
//! sampling- index order. Each row offers rename, reorder, and remove, plus a
//! count of the image nodes bound to the slot. Slots are addressed by stable
//! [`SlotId`], so renames and reorders never break bindings in flight.
//! Asset-bound images live on their image nodes, not here. All mutations are
//! emitted as [`EditRequest`]; the panel never touches the graph directly.
//!
//! [`EmitterGraph`]: crate::effect_graph::model::EmitterGraph
//! [`TextureSlotDef`]: crate::effect_graph::model::TextureSlotDef
//! [`SlotId`]: crate::effect_graph::model::SlotId

use bevy::prelude::*;
use bevy_egui::egui;

use crate::{
    edits::{EditKind, EditRequest},
    effect_graph::model::{
        EffectGraph, EmitterGraph, EmitterId, ExprNode, ImageBinding, NodePayload, SlotId,
        TextureSlotDef,
    },
    ui::icons::{ICON_ARROW_DOWN, ICON_ARROW_UP, ICON_PLUS, ICON_XMARK},
};

/// Top-level entry point for the Material tab.
///
/// Lists every texture slot; a pure-UI helper that never mutates the
/// graph directly — it only emits [`EditRequest`].
pub fn show_panel(
    ui: &mut egui::Ui,
    doc: Entity,
    effect_graph: &EffectGraph,
    emitter: EmitterId,
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
) {
    let Some(graph) = effect_graph.emitter(emitter) else {
        ui.weak("(no emitter selected)");
        return;
    };
    let slots = &graph.texture_slots;
    egui::ScrollArea::vertical().show(ui, |ui| {
        if slots.is_empty() {
            ui.weak("(no texture slots)");
        }
        let count = slots.len();
        for (index, slot) in slots.iter().enumerate() {
            let refs = reference_count(graph, slot.id);
            slot_row(ui, doc, emitter, slot, index, count, refs, edits);
        }
        ui.add_space(4.0);
        if ui
            .button(format!("{ICON_PLUS}  Add slot"))
            .on_hover_text("Add a host-supplied texture slot")
            .clicked()
        {
            edits.write(EditRequest::new(doc, EditKind::AddTextureSlot { emitter }));
        }
    });
}

/// How many image nodes are bound to the texture slot `id`.
fn reference_count(graph: &EmitterGraph, id: SlotId) -> usize {
    graph
        .nodes
        .iter()
        .filter(|n| {
            matches!(
                &n.payload,
                NodePayload::Expr(ExprNode::Image(ImageBinding::Slot(s))) if *s == id
            )
        })
        .count()
}

#[allow(clippy::too_many_arguments)]
fn slot_row(
    ui: &mut egui::Ui,
    doc: Entity,
    emitter: EmitterId,
    slot: &TextureSlotDef,
    index: usize,
    count: usize,
    refs: usize,
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
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
                            emitter,
                            id,
                            new: trimmed.into(),
                        },
                    ));
                }
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // Removal is only safe when no image node is bound to the slot;
                // otherwise those nodes would dangle.
                let removable = refs == 0;
                let remove = ui
                    .add_enabled(removable, egui::Button::new(ICON_XMARK.to_string()))
                    .on_hover_text("Remove this slot")
                    .on_disabled_hover_text("In use by an image node");
                if remove.clicked() {
                    edits.write(EditRequest::new(
                        doc,
                        EditKind::RemoveTextureSlot { emitter, id },
                    ));
                }

                // Reorder. Moving a slot reassigns sampling indices, so any
                // its slot binding now targets a different slot.
                let can_down = index + 1 < count;
                if ui
                    .add_enabled(can_down, egui::Button::new(ICON_ARROW_DOWN.to_string()))
                    .on_hover_text("Move down (higher index)")
                    .clicked()
                {
                    edits.write(EditRequest::new(
                        doc,
                        EditKind::ReorderTextureSlot {
                            emitter,
                            id,
                            to: index + 1,
                        },
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
                        EditKind::ReorderTextureSlot {
                            emitter,
                            id,
                            to: index - 1,
                        },
                    ));
                }
            });
        });

        // Reference count / orphan indicator.
        if refs == 0 {
            ui.weak("orphan — no image node bound to this slot");
        } else {
            ui.weak(format!(
                "{refs} binding{}",
                if refs == 1 { "" } else { "s" }
            ));
        }
    });
}
