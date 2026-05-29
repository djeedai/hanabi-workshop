//! Outline (modifier list) panel.
//!
//! Lists `init` / `update` / `render` modifiers, with per-row remove
//! (`✕`) and reorder (`↑` / `↓`) buttons, and a per-section `+`
//! button that opens a popup of curated modifier templates to insert.
//! Clicking a row writes the selection into `DocumentUi.selected_modifier`
//! so the Properties panel can show field editors for it.

use bevy::prelude::*;
use bevy_egui::egui;
use bevy_hanabi::EffectAsset;

use crate::document::{ModifierGroup, ModifierSelection};
use crate::edits::{EditKind, EditRequest};
use crate::modifier_ops::AddModifierKind;

pub fn show(
    ui: &mut egui::Ui,
    doc_entity: Entity,
    effects: &Assets<EffectAsset>,
    effect_handle: &Handle<EffectAsset>,
    selected: &mut Option<ModifierSelection>,
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
) {
    ui.heading("Outline");
    ui.separator();

    let Some(asset) = effects.get(effect_handle) else {
        ui.label("(effect asset not loaded yet)");
        return;
    };

    let init: Vec<String> = asset.init_modifiers().map(modifier_label).collect();
    let update: Vec<String> = asset.update_modifiers().map(modifier_label).collect();
    let render: Vec<String> = asset
        .render_modifiers()
        .map(|m| modifier_label(m.as_modifier()))
        .collect();

    egui::ScrollArea::vertical().show(ui, |ui| {
        section(ui, doc_entity, ModifierGroup::Init, &init, selected, edits);
        section(ui, doc_entity, ModifierGroup::Update, &update, selected, edits);
        section(ui, doc_entity, ModifierGroup::Render, &render, selected, edits);
    });
}

fn section(
    ui: &mut egui::Ui,
    doc_entity: Entity,
    group: ModifierGroup,
    labels: &[String],
    selected: &mut Option<ModifierSelection>,
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
) {
    let len = labels.len();
    egui::CollapsingHeader::new(format!("{} ({})", group.label(), len))
        .id_salt(("outline-group", group as u8))
        .default_open(true)
        .show(ui, |ui| {
            for (idx, label) in labels.iter().enumerate() {
                ui.horizontal(|ui| {
                    let is_selected = matches!(
                        selected,
                        Some(s) if s.group == group && s.idx == idx
                    );
                    // Take most of the row width for the label so the
                    // action buttons hug the right edge.
                    let resp = ui.add(egui::Button::selectable(is_selected, label));
                    if resp.clicked() {
                        *selected = Some(ModifierSelection { group, idx });
                    }

                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            // Remove.
                            let remove_btn = ui
                                .small_button("✕")
                                .on_hover_text("Remove this modifier");
                            if remove_btn.clicked() {
                                edits.write(EditRequest::new(
                                    doc_entity,
                                    EditKind::RemoveModifier { group, idx },
                                ));
                                // Clear selection if it pointed here.
                                if matches!(
                                    selected,
                                    Some(s) if s.group == group && s.idx == idx
                                ) {
                                    *selected = None;
                                }
                            }
                            // Move down.
                            let can_down = idx + 1 < len;
                            let down_resp = ui.add_enabled(
                                can_down,
                                egui::Button::new("↓").small(),
                            );
                            if down_resp.clicked() {
                                edits.write(EditRequest::new(
                                    doc_entity,
                                    EditKind::MoveModifier {
                                        group,
                                        from: idx,
                                        to: idx + 1,
                                    },
                                ));
                                // Track the selection along with the move.
                                if let Some(s) = selected.as_mut() {
                                    if s.group == group && s.idx == idx {
                                        s.idx = idx + 1;
                                    } else if s.group == group && s.idx == idx + 1 {
                                        s.idx = idx;
                                    }
                                }
                            }
                            // Move up.
                            let can_up = idx > 0;
                            let up_resp = ui.add_enabled(
                                can_up,
                                egui::Button::new("↑").small(),
                            );
                            if up_resp.clicked() {
                                edits.write(EditRequest::new(
                                    doc_entity,
                                    EditKind::MoveModifier {
                                        group,
                                        from: idx,
                                        to: idx - 1,
                                    },
                                ));
                                if let Some(s) = selected.as_mut() {
                                    if s.group == group && s.idx == idx {
                                        s.idx = idx - 1;
                                    } else if s.group == group && s.idx == idx - 1 {
                                        s.idx = idx;
                                    }
                                }
                            }
                        },
                    );
                });
            }
            if labels.is_empty() {
                ui.weak("(none)");
            }

            // Per-section Add menu. Append-only (at == current len).
            ui.add_space(2.0);
            ui.menu_button("+ Add modifier…", |ui| {
                for kind in AddModifierKind::options_for(group) {
                    if ui.button(kind.label()).clicked() {
                        edits.write(EditRequest::new(
                            doc_entity,
                            EditKind::AddModifierFromTemplate {
                                group,
                                kind: *kind,
                                at: len,
                            },
                        ));
                        ui.close();
                    }
                }
            });
        });
}

/// Short label for a modifier. Uses the type name with namespace stripped,
/// since `Reflect::reflect_type_path` returns the fully-qualified path.
fn modifier_label(m: &dyn bevy_hanabi::Modifier) -> String {
    let full = m.reflect_type_path();
    full.rsplit("::").next().unwrap_or(full).to_string()
}

