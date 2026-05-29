//! Outline (modifier list) panel.
//!
//! Phase 5a: read-only listing of `init` / `update` / `render` modifiers.
//! Clicking a row writes the selection into `DocumentUi.selected_modifier`
//! so the Properties panel can show field editors for it (Phase 5b).

use bevy::prelude::*;
use bevy_egui::egui;
use bevy_hanabi::EffectAsset;

use crate::document::{ModifierGroup, ModifierSelection};

pub fn show(
    ui: &mut egui::Ui,
    _doc_entity: Entity,
    effects: &Assets<EffectAsset>,
    effect_handle: &Handle<EffectAsset>,
    selected: &mut Option<ModifierSelection>,
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
        section(ui, ModifierGroup::Init, &init, selected);
        section(ui, ModifierGroup::Update, &update, selected);
        section(ui, ModifierGroup::Render, &render, selected);
    });

    ui.add_space(8.0);
    ui.weak("[+ Add / Remove / Reorder coming in Phase 5b]");
}

fn section(
    ui: &mut egui::Ui,
    group: ModifierGroup,
    labels: &[String],
    selected: &mut Option<ModifierSelection>,
) {
    egui::CollapsingHeader::new(format!("{} ({})", group.label(), labels.len()))
        .id_salt(("outline-group", group as u8))
        .default_open(true)
        .show(ui, |ui| {
            if labels.is_empty() {
                ui.weak("(none)");
                return;
            }
            for (idx, label) in labels.iter().enumerate() {
                let is_selected = matches!(
                    selected,
                    Some(s) if s.group == group && s.idx == idx
                );
                if ui.selectable_label(is_selected, label).clicked() {
                    *selected = Some(ModifierSelection { group, idx });
                }
            }
        });
}

/// Short label for a modifier. Uses the type name with namespace stripped,
/// since `Reflect::reflect_type_path` returns the fully-qualified path.
fn modifier_label(m: &dyn bevy_hanabi::Modifier) -> String {
    let full = m.reflect_type_path();
    full.rsplit("::").next().unwrap_or(full).to_string()
}
