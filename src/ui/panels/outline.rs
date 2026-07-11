//! Effect panel.
//!
//! Top of the panel edits the effect-level fields (`EffectAsset.name`,
//! `simulation_space`, `simulation_condition`, `z_layer_2d`) and the
//! `SpawnerSettings` (count, period, cycle_count, starts_active). Below
//! that sits the read-only particle-layout strip.
//!
//! Modifier editing lives in the Graph panel — each modifier is a stacked
//! node there, with a per-node close button to remove it and a warning
//! badge when its writes are shadowed by a later modifier.
//!
//! ## Local-draft pattern
//!
//! Continuous edits (text typing, drag-value scrubbing) keep an in-flight
//! draft value in egui's per-id memory. A single [`EditRequest`] is
//! committed on `lost_focus()` (text) or `drag_stopped()` (numeric), with
//! the captured old/new values. This collapses each user gesture into one
//! undoable step and one bevy_hanabi shader rebuild.

use bevy::prelude::*;
use bevy_egui::egui;
use bevy_hanabi::{CpuValue, EffectAsset, SimulationCondition, SimulationSpace, SpawnerSettings};

use super::collapsing;
use crate::{
    edits::{EditKind, EditRequest},
    effect_graph::model::{EffectGraph, EffectHeader},
};

pub fn show(
    ui: &mut egui::Ui,
    doc: Entity,
    graph: &EffectGraph,
    effects: &Assets<EffectAsset>,
    effect_handle: &Handle<EffectAsset>,
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
) {
    // The inner dock inherits the document tab body's zeroed vertical item
    // spacing; restore the theme default so consecutive field rows don't touch.
    let default_spacing_y = ui.ctx().global_style().spacing.item_spacing.y;
    ui.spacing_mut().item_spacing.y = default_spacing_y;

    egui::ScrollArea::vertical().show(ui, |ui| {
        // All sections share one resizable label column, persisted per
        // document so labels and value widgets stay aligned across sections.
        let width_id = egui::Id::new(("effect-label-width", doc));
        let available_width = ui.available_width();
        let max_label_w = (available_width - 80.0).max(MIN_LABEL_WIDTH);
        let default_label_w = available_width * DEFAULT_LABEL_WIDTH_FRACTION;
        let label_w = ui
            .ctx()
            .data_mut(|d| d.get_temp::<f32>(width_id).unwrap_or(default_label_w))
            .clamp(MIN_LABEL_WIDTH, max_label_w);

        let fields = ui
            .vertical(|ui| {
                effect_fields(ui, doc, &graph.header, label_w, edits);
                ui.add_space(8.0);
                spawner_fields(ui, doc, graph.header.spawner, label_w, edits);
            })
            .response;

        if let Some(new_w) = column_divider(ui, width_id.with("split"), fields.rect, label_w) {
            let new_w = new_w.clamp(MIN_LABEL_WIDTH, max_label_w);
            ui.ctx().data_mut(|d| d.insert_temp(width_id, new_w));
        }

        ui.add_space(8.0);
        if let Some(asset) = effects.get(effect_handle) {
            super::shaders::layout_section(ui, asset);
        } else {
            ui.weak("(effect not loaded)");
        }
    });
}

/// Minimum width the shared label column can be shrunk to.
const MIN_LABEL_WIDTH: f32 = 48.0;

/// Initial fraction of the panel occupied by the shared label column.
const DEFAULT_LABEL_WIDTH_FRACTION: f32 = 0.45;

fn effect_fields(
    ui: &mut egui::Ui,
    doc: Entity,
    header: &EffectHeader,
    label_w: f32,
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
) {
    collapsing(ui, ("effect-section", "Effect"), "Effect", |ui| {
        // Name: text field, committed on lost_focus.
        let id = egui::Id::new(("prop-effect-name", doc));
        let mut draft: String = ui.ctx().data_mut(|d| {
            d.get_temp::<String>(id)
                .unwrap_or_else(|| header.name.to_string())
        });
        let resp = field_row(ui, label_w, "Name", |ui| {
            ui.add(egui::TextEdit::singleline(&mut draft).desired_width(ui.available_width()))
        });
        if resp.has_focus() || resp.changed() {
            ui.ctx().data_mut(|d| d.insert_temp(id, draft.clone()));
        }
        if resp.lost_focus() {
            if draft != *header.name {
                edits.write(EditRequest::new(
                    doc,
                    EditKind::SetEffectName { new: draft.clone() },
                ));
            }
            ui.ctx().data_mut(|d| d.remove::<String>(id));
        }

        // Simulation space.
        let mut sim_space = header.simulation_space;
        field_row(ui, label_w, "Simulation space", |ui| {
            egui::ComboBox::from_id_salt(("prop-effect-sim-space", doc))
                .width(ui.available_width())
                .truncate()
                .selected_text(format!("{sim_space:?}"))
                .show_ui(ui, |ui| {
                    for option in [SimulationSpace::Global, SimulationSpace::Local] {
                        ui.selectable_value(&mut sim_space, option, format!("{option:?}"));
                    }
                });
        });
        if sim_space != header.simulation_space {
            edits.write(EditRequest::new(
                doc,
                EditKind::SetSimulationSpace { new: sim_space },
            ));
        }

        // Simulation condition.
        let mut sim_cond = header.simulation_condition;
        field_row(ui, label_w, "Simulation condition", |ui| {
            egui::ComboBox::from_id_salt(("prop-effect-sim-cond", doc))
                .width(ui.available_width())
                .truncate()
                .selected_text(format!("{sim_cond:?}"))
                .show_ui(ui, |ui| {
                    for option in [
                        SimulationCondition::Always,
                        SimulationCondition::WhenVisible,
                    ] {
                        ui.selectable_value(&mut sim_cond, option, format!("{option:?}"));
                    }
                });
        });
        if sim_cond != header.simulation_condition {
            edits.write(EditRequest::new(
                doc,
                EditKind::SetSimulationCondition { new: sim_cond },
            ));
        }

        // Z layer (2D only, but always shown for inspection).
        if let Some(new_z) = drag_f32(
            ui,
            label_w,
            ("prop-effect-zlayer", doc),
            "Z layer (2D)",
            header.z_layer_2d,
            f32::MIN..=f32::MAX,
            0.01,
        ) {
            edits.write(EditRequest::new(doc, EditKind::SetZLayer2d { new: new_z }));
        }

        // Capacity: max live particle count. Editing it re-bakes the asset
        // (forces a particle-buffer reallocation on the next reconcile).
        if let Some(new_capacity) = drag_u32(
            ui,
            label_w,
            ("prop-effect-capacity", doc),
            "Capacity",
            header.capacity,
        ) {
            edits.write(EditRequest::new(
                doc,
                EditKind::SetCapacity {
                    new: new_capacity.max(1),
                },
            ));
        }
    });
}

fn spawner_fields(
    ui: &mut egui::Ui,
    doc: Entity,
    current: SpawnerSettings,
    label_w: f32,
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
) {
    collapsing(ui, ("effect-section", "Spawner"), "Spawner", |ui| {
        // We expose the most useful subset of SpawnerSettings.
        // `spawn_duration` and `emit_on_start` aren't shown here (no
        // public getter for the latter in bevy_hanabi 0.18); they are
        // preserved by reading the current value when committing.
        //
        // Each numeric uses a draft cached in egui per-id memory so
        // that mid-drag values aren't clobbered by the per-frame
        // re-snapshot of the asset.
        let current_count = cpu_value_scalar(current.count());
        let current_period = cpu_value_scalar(current.period());
        let current_cycle = current.cycle_count();
        let current_active = current.starts_active();

        let count = drag_f32(
            ui,
            label_w,
            ("prop-spawner-count", doc),
            "Count",
            current_count,
            0.0..=f32::MAX,
            0.5,
        );
        let period = drag_f32(
            ui,
            label_w,
            ("prop-spawner-period", doc),
            "Period (s)",
            current_period,
            0.001..=f32::MAX,
            0.01,
        );
        let cycle_count = drag_u32(
            ui,
            label_w,
            ("prop-spawner-cycle", doc),
            "Cycle count (0 = infinite)",
            current_cycle,
        );

        let mut starts_active = current_active;
        let active_changed = field_row(ui, label_w, "Starts active", |ui| {
            ui.checkbox(&mut starts_active, "").changed()
        });

        let changed =
            count.is_some() || period.is_some() || cycle_count.is_some() || active_changed;

        if changed {
            let final_count = count.unwrap_or(current_count);
            let final_period = period.unwrap_or(current_period).max(0.001);
            let final_cycle = cycle_count.unwrap_or(current_cycle);

            // `final_period` is clamped positive and finite, so `try_new`
            // accepts any cycle count; bail out (rather than panic) on the
            // off chance the inputs are still degenerate.
            match SpawnerSettings::try_new(
                final_count.into(),
                current.spawn_duration(),
                final_period.into(),
                final_cycle,
            ) {
                Ok(new) => {
                    let new = new.with_starts_active(starts_active);
                    edits.write(EditRequest::new(doc, EditKind::SetSpawnerSettings { new }));
                }
                Err(err) => warn!("ignoring invalid spawner settings: {err}"),
            }
        }
    });
}

/// Render a labelled `DragValue<f32>` backed by an egui-memory draft.
///
/// Emits one label/value row. Returns `Some(new_value)` on the frame the user
/// commits the edit (drag released or text-edit focus lost), or `None`
/// otherwise. The draft is cleared on commit so the next frame re-snapshots
/// from the asset.
fn drag_f32(
    ui: &mut egui::Ui,
    label_w: f32,
    id_src: impl std::hash::Hash,
    label: &str,
    current: f32,
    range: std::ops::RangeInclusive<f32>,
    speed: f32,
) -> Option<f32> {
    let id = egui::Id::new(id_src);
    let mut value: f32 = ui
        .ctx()
        .data_mut(|d| d.get_temp::<f32>(id).unwrap_or(current));
    let size = egui::vec2(drag_field_width(ui), ui.spacing().interact_size.y);
    let resp = field_row(ui, label_w, label, |ui| {
        ui.add_sized(
            size,
            egui::DragValue::new(&mut value).range(range).speed(speed),
        )
    });
    if resp.dragged() || resp.has_focus() || resp.changed() {
        ui.ctx().data_mut(|d| d.insert_temp(id, value));
    }
    if resp.drag_stopped() || resp.lost_focus() {
        ui.ctx().data_mut(|d| d.remove::<f32>(id));
        if value != current {
            return Some(value);
        }
    }
    None
}

fn drag_u32(
    ui: &mut egui::Ui,
    label_w: f32,
    id_src: impl std::hash::Hash,
    label: &str,
    current: u32,
) -> Option<u32> {
    let id = egui::Id::new(id_src);
    let mut value: u32 = ui
        .ctx()
        .data_mut(|d| d.get_temp::<u32>(id).unwrap_or(current));
    let size = egui::vec2(drag_field_width(ui), ui.spacing().interact_size.y);
    let resp = field_row(ui, label_w, label, |ui| {
        ui.add_sized(size, egui::DragValue::new(&mut value).range(0..=u32::MAX))
    });
    if resp.dragged() || resp.has_focus() || resp.changed() {
        ui.ctx().data_mut(|d| d.insert_temp(id, value));
    }
    if resp.drag_stopped() || resp.lost_focus() {
        ui.ctx().data_mut(|d| d.remove::<u32>(id));
        if value != current {
            return Some(value);
        }
    }
    None
}

/// Width of a numeric drag field.
///
/// A little wider than egui's default so values aren't cramped in the roomy
/// value column.
fn drag_field_width(ui: &egui::Ui) -> f32 {
    ui.spacing().interact_size.x * 1.3
}

/// Best-effort scalar extraction from a `CpuValue<f32>`.
///
/// `Single` returns the value; uniform ranges fall back to the midpoint.
fn cpu_value_scalar(v: CpuValue<f32>) -> f32 {
    match v {
        CpuValue::Single(s) => s,
        CpuValue::Uniform((a, b)) => 0.5 * (a + b),
        _ => 0.0,
    }
}

/// Lay out one label/value row sharing the panel's resizable label column.
///
/// The label occupies a fixed-width left cell, so labels and value widgets line
/// up across every section. `add` populates the value cell and its return value
/// is forwarded to the caller.
fn field_row<R>(
    ui: &mut egui::Ui,
    label_w: f32,
    label: &str,
    add: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    ui.horizontal(|ui| {
        // Reserve a full-height label cell so consecutive rows don't overlap
        // and every value widget starts at the same x.
        let h = ui.spacing().interact_size.y;
        ui.set_min_height(h);
        let cell = egui::vec2(label_w, h);
        ui.allocate_ui_with_layout(
            cell,
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.set_min_size(cell);
                // Truncate the label inside a slightly narrower cell so its
                // text keeps a small gap from the divider, matching the
                // panel's left content margin.
                let text = egui::vec2((label_w - LABEL_INNER_PAD).max(0.0), h);
                ui.allocate_ui_with_layout(
                    text,
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| {
                        ui.set_min_size(text);
                        ui.add(egui::Label::new(label).truncate());
                    },
                );
            },
        );
        add(ui)
    })
    .inner
}

/// Right-hand gap kept between a truncated label and the column divider.
const LABEL_INNER_PAD: f32 = 4.0;

/// Draggable vertical divider that sets the shared label-column width.
///
/// Spans `area` at the current split position. The guide line is only painted
/// while the handle is hovered or dragged. Returns the proposed new label width
/// while the user drags the handle, or `None` otherwise.
fn column_divider(ui: &mut egui::Ui, id: egui::Id, area: egui::Rect, label_w: f32) -> Option<f32> {
    let split_x = area.left() + label_w;
    let handle = egui::Rect::from_x_y_ranges((split_x - 3.0)..=(split_x + 3.0), area.y_range());
    let resp = ui.interact(handle, id, egui::Sense::drag());
    if resp.hovered() || resp.dragged() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
        let stroke = egui::Stroke::new(2.0, ui.visuals().widgets.hovered.fg_stroke.color);
        ui.painter().vline(split_x, area.y_range(), stroke);
    }
    resp.dragged().then(|| label_w + resp.drag_delta().x)
}
