//! Effect panel.
//!
//! Top-level view of the current effect: the particle layout
//! (attribute strip) followed by the `init` / `update` / `render`
//! modifier list. Per-row remove (`✕`) and reorder (`↑` / `↓`)
//! buttons, and a per-section `+` button to insert curated modifier
//! templates. Clicking a row writes the selection into
//! `DocumentUi.selected_modifier` so the Properties panel can show
//! field editors for it.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy_egui::egui;
use bevy_hanabi::{Attribute, EffectAsset, SetAttributeModifier};

use crate::document::{ModifierGroup, ModifierSelection};
use crate::edits::{EditKind, EditRequest};
use crate::modifier_registry;

/// For each `SetAttributeModifier` index in an ordered list of
/// modifiers, returns `(shadower_idx, attr_name)` when a *later*
/// modifier in the same list also writes the same attribute — making
/// the earlier modifier's write a no-op.
///
/// Only meaningful within a single group (Init or Update): both run
/// strictly in order, with each writer overwriting any previous
/// per-particle value. Cross-group interactions (Update running after
/// Init every frame) aren't flagged here — Init-only "spawn-time"
/// values are still observable on frame 0.
fn shadowed_set_attributes(asset: &EffectAsset, group: ModifierGroup) -> HashMap<usize, (usize, &'static str)> {
    let mut last_writer: HashMap<Attribute, usize> = HashMap::default();
    let mut shadow: HashMap<usize, (usize, &'static str)> = HashMap::default();
    let mods: Box<dyn Iterator<Item = &dyn bevy::reflect::Reflect>> = match group {
        ModifierGroup::Init => Box::new(asset.init_modifiers().map(|m| m.as_reflect())),
        ModifierGroup::Update => Box::new(asset.update_modifiers().map(|m| m.as_reflect())),
        ModifierGroup::Render => return shadow,
    };
    for (i, m) in mods.enumerate() {
        if let Some(sam) = m.downcast_ref::<SetAttributeModifier>()
            && let Some(prev) = last_writer.insert(sam.attribute, i)
        {
            shadow.insert(prev, (i, sam.attribute.name()));
        }
    }
    shadow
}

pub fn show(
    ui: &mut egui::Ui,
    doc_entity: Entity,
    effects: &Assets<EffectAsset>,
    effect_handle: &Handle<EffectAsset>,
    selected: &mut Option<ModifierSelection>,
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
    type_registry: &AppTypeRegistry,
) {
    let Some(asset) = effects.get(effect_handle) else {
        ui.label("(effect asset not loaded yet)");
        return;
    };

    let init: Vec<String> = asset
        .init_modifiers()
        .map(|m| crate::ui::modifier_names::display_name_for_modifier(m).into_owned())
        .collect();
    let update: Vec<String> = asset
        .update_modifiers()
        .map(|m| crate::ui::modifier_names::display_name_for_modifier(m).into_owned())
        .collect();
    let render: Vec<String> = asset
        .render_modifiers()
        .map(|m| crate::ui::modifier_names::display_name_for_modifier(m.as_modifier()).into_owned())
        .collect();

    egui::ScrollArea::vertical().show(ui, |ui| {
        super::debug::layout_section(ui, asset);
        ui.add_space(4.0);
        ui.separator();
        section(
            ui,
            doc_entity,
            ModifierGroup::Init,
            &init,
            &shadowed_set_attributes(asset, ModifierGroup::Init),
            selected,
            edits,
            type_registry,
        );
        section(
            ui,
            doc_entity,
            ModifierGroup::Update,
            &update,
            &shadowed_set_attributes(asset, ModifierGroup::Update),
            selected,
            edits,
            type_registry,
        );
        section(
            ui,
            doc_entity,
            ModifierGroup::Render,
            &render,
            &HashMap::default(),
            selected,
            edits,
            type_registry,
        );
    });
}

fn section(
    ui: &mut egui::Ui,
    doc_entity: Entity,
    group: ModifierGroup,
    labels: &[String],
    shadowed: &HashMap<usize, (usize, &'static str)>,
    selected: &mut Option<ModifierSelection>,
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
    type_registry: &AppTypeRegistry,
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

                    if let Some((shadower, attr_name)) = shadowed.get(&idx) {
                        let warn = ui.label(
                            egui::RichText::new(
                                crate::ui::icons::ICON_TRIANGLE_EXCLAMATION.to_string(),
                            )
                            .color(egui::Color32::from_rgb(255, 180, 50)),
                        );
                        warn.on_hover_text(format!(
                            "This SetAttributeModifier writes `{attr_name}`, but \
                             #{shadower} in the same group writes the same \
                             attribute later and overwrites it. This modifier \
                             has no effect."
                        ));
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        // Remove.
                        let remove_btn = ui
                            .small_button(crate::ui::icons::ICON_XMARK.to_string())
                            .on_hover_text("Remove this modifier");
                        if remove_btn.clicked() {
                            edits.write(EditRequest::new(
                                doc_entity,
                                EditKind::RemoveModifier { group, idx },
                            ));
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
                            egui::Button::new(crate::ui::icons::ICON_CHEVRON_DOWN.to_string())
                                .small(),
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
                            egui::Button::new(crate::ui::icons::ICON_CHEVRON_UP.to_string())
                                .small(),
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
                    });
                });
            }
            if labels.is_empty() {
                ui.weak("(none)");
            }

            // Per-section Add menu. Append-only (at == current len).
            ui.add_space(2.0);
            ui.menu_button("+ Add modifier...", |ui| {
                let type_registry = type_registry.read();
                for kind in modifier_registry::iter_modifier_kinds_for(&type_registry, group) {
                    if ui.button(kind.display_name()).clicked() {
                        edits.write(EditRequest::new(
                            doc_entity,
                            EditKind::AddModifierFromTemplate {
                                group,
                                type_id: kind.type_id,
                                at: len,
                            },
                        ));
                        ui.close();
                    }
                }
            });
        });
}
