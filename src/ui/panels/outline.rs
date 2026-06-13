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
use bevy_hanabi::{Attribute, EffectAsset};

use crate::document::{ModifierGroup, ModifierSelection};
use crate::edits::{EditKind, EditRequest};
use crate::effect_graph::model::{
    EditValue, EffectGraph, GraphNode, ModifierNodeData, NodePayload,
};
use crate::modifier_registry::{self, ReflectModifier};
use crate::ui::modifier_names::display_name_for_type;

/// For each modifier index in `group`, returns the list of
/// `(attribute, shadower_idx)` pairs explaining why it has no effect:
/// every attribute the modifier overwrites is *also* overwritten by
/// some later modifier in the same group, making the earlier
/// modifier's writes dead.
///
/// Built on the [`ReflectModifier::overwrites`] type-data callback,
/// which gives the per-instance set of attributes a modifier *fully
/// assigns to* (distinct from `Modifier::attributes()`, which mixes
/// reads and writes in upstream bevy_hanabi 0.18).
///
/// Only meaningful within a single group (Init / Update): each runs
/// strictly in order, with subsequent overwrites discarding any
/// previous per-particle value. The Render group is currently a
/// no-op — render modifiers write vertex-shader variables rather
/// than particle attributes, so they need a separate model.
fn shadowed_modifiers(
    asset: &EffectAsset,
    group: ModifierGroup,
    type_registry: &AppTypeRegistry,
) -> HashMap<usize, Vec<(Attribute, usize)>> {
    let mut out: HashMap<usize, Vec<(Attribute, usize)>> = HashMap::default();
    if matches!(group, ModifierGroup::Render) {
        return out;
    }
    let registry = type_registry.read();

    // Per-modifier set of attributes the modifier fully overwrites.
    let overwrites: Vec<Vec<Attribute>> = match group {
        ModifierGroup::Init => asset
            .init_modifiers()
            .map(|m| overwrites_for(m.as_reflect(), &registry))
            .collect(),
        ModifierGroup::Update => asset
            .update_modifiers()
            .map(|m| overwrites_for(m.as_reflect(), &registry))
            .collect(),
        ModifierGroup::Render => return out,
    };

    // For each modifier i, look forward for the *earliest* later j
    // that overwrites each attribute in i's set. If every attribute
    // is covered, i is fully shadowed.
    for (i, w_i) in overwrites.iter().enumerate() {
        if w_i.is_empty() {
            continue;
        }
        let mut hits: Vec<(Attribute, usize)> = Vec::with_capacity(w_i.len());
        for &attr in w_i {
            let earliest = overwrites
                .iter()
                .enumerate()
                .skip(i + 1)
                .find(|(_, w_j)| w_j.contains(&attr))
                .map(|(j, _)| j);
            match earliest {
                Some(j) => hits.push((attr, j)),
                None => {
                    // At least one produced attribute survives — not
                    // shadowed.
                    hits.clear();
                    break;
                }
            }
        }
        if !hits.is_empty() {
            out.insert(i, hits);
        }
    }
    out
}

/// Per-instance overwrite set lookup. Falls back to empty if the
/// modifier's type isn't registered (e.g. third-party type that
/// didn't ship a [`ReflectModifier`]) — we conservatively skip
/// shadow analysis rather than risk a false positive.
fn overwrites_for(
    m: &dyn bevy::reflect::Reflect,
    registry: &bevy::reflect::TypeRegistry,
) -> Vec<Attribute> {
    let Some(reg) = registry.get(std::any::Any::type_id(m)) else {
        return Vec::new();
    };
    let Some(rm) = reg.data::<ReflectModifier>() else {
        return Vec::new();
    };
    (rm.overwrites)(m)
}

/// Display label for a stack member node, mirroring
/// [`crate::ui::modifier_names::display_name_for_modifier`] but reading
/// the graph node rather than a baked modifier instance.
fn node_label(node: &GraphNode) -> String {
    match &node.payload {
        NodePayload::Modifier(ModifierNodeData::Known { type_path, config }) => {
            let short = short_type_name(type_path);
            if short == "SetAttributeModifier"
                && let Some(EditValue::Attribute(attr)) = config.get("attribute")
            {
                return format!("Set Attribute ({})", attr.name());
            }
            display_name_for_type(short).into_owned()
        }
        NodePayload::Modifier(ModifierNodeData::Unknown { type_path, .. }) => {
            format!("{} (?)", short_type_name(type_path))
        }
        NodePayload::Expr(_) => "(non-modifier node)".to_string(),
    }
}

/// Display labels for every member of `group`'s stack, in execution
/// order. A missing node id yields a placeholder so list indices stay
/// aligned with the stack — the same index space the Details panel
/// reads.
fn group_labels(graph: &EffectGraph, group: ModifierGroup) -> Vec<String> {
    let Some(stack) = graph.stack(group) else {
        return Vec::new();
    };
    stack
        .members
        .iter()
        .map(|&id| {
            graph
                .node(id)
                .map(node_label)
                .unwrap_or_else(|| "(missing)".to_string())
        })
        .collect()
}

fn short_type_name(type_path: &str) -> &str {
    let head = type_path.split('<').next().unwrap_or(type_path);
    head.rsplit("::").next().unwrap_or(head)
}

pub fn show(
    ui: &mut egui::Ui,
    doc_entity: Entity,
    effects: &Assets<EffectAsset>,
    effect_handle: &Handle<EffectAsset>,
    graph: &EffectGraph,
    selected: &mut Option<ModifierSelection>,
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
    type_registry: &AppTypeRegistry,
) {
    // The modifier list is read from the canonical graph so its indices
    // match the Details panel. The preview asset (re-baked on every edit)
    // is used only for the cosmetic layout strip and shadow analysis.
    let asset = effects.get(effect_handle);

    let init = group_labels(graph, ModifierGroup::Init);
    let update = group_labels(graph, ModifierGroup::Update);
    let render = group_labels(graph, ModifierGroup::Render);

    egui::ScrollArea::vertical().show(ui, |ui| {
        if let Some(asset) = asset {
            super::shaders::layout_section(ui, asset);
            ui.add_space(4.0);
            ui.separator();
        }
        let init_shadow = asset
            .map(|a| shadowed_modifiers(a, ModifierGroup::Init, type_registry))
            .unwrap_or_default();
        let update_shadow = asset
            .map(|a| shadowed_modifiers(a, ModifierGroup::Update, type_registry))
            .unwrap_or_default();
        section(
            ui,
            doc_entity,
            ModifierGroup::Init,
            &init,
            &init_shadow,
            selected,
            edits,
            type_registry,
        );
        section(
            ui,
            doc_entity,
            ModifierGroup::Update,
            &update,
            &update_shadow,
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
    shadowed: &HashMap<usize, Vec<(Attribute, usize)>>,
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

                    if let Some(hits) = shadowed.get(&idx) {
                        let warn = ui.label(
                            egui::RichText::new(
                                crate::ui::icons::ICON_TRIANGLE_EXCLAMATION.to_string(),
                            )
                            .color(egui::Color32::from_rgb(255, 180, 50)),
                        );
                        let mut tip = String::from(
                            "This modifier has no effect: every attribute \
                             it writes is overwritten by a later modifier \
                             in the same group.\n",
                        );
                        for (attr, j) in hits {
                            tip.push_str(&format!("  • `{}` → overwritten by #{}\n", attr.name(), j));
                        }
                        warn.on_hover_text(tip);
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
