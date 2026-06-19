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
    egui::ScrollArea::vertical().show(ui, |ui| {
        effect_fields(ui, doc, &graph.header, edits);
        ui.add_space(8.0);
        spawner_fields(ui, doc, graph.header.spawner, edits);
        ui.add_space(8.0);
        ui.separator();
        if let Some(asset) = effects.get(effect_handle) {
            super::shaders::layout_section(ui, asset);
        } else {
            ui.weak("(effect not loaded)");
        }
    });
}

fn effect_fields(
    ui: &mut egui::Ui,
    doc: Entity,
    header: &EffectHeader,
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
) {
    collapsing(ui, "Effect", |ui| {
        // Name: text field, committed on lost_focus.
        let id = egui::Id::new(("prop-effect-name", doc));
        let mut draft: String = ui.ctx().data_mut(|d| {
            d.get_temp::<String>(id)
                .unwrap_or_else(|| header.name.to_string())
        });
        let resp = ui
            .horizontal(|ui| {
                ui.label("Name");
                ui.add(egui::TextEdit::singleline(&mut draft).desired_width(200.0))
            })
            .inner;
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
        egui::ComboBox::from_label("Simulation space")
            .selected_text(format!("{sim_space:?}"))
            .show_ui(ui, |ui| {
                for option in [SimulationSpace::Global, SimulationSpace::Local] {
                    ui.selectable_value(&mut sim_space, option, format!("{option:?}"));
                }
            });
        if sim_space != header.simulation_space {
            edits.write(EditRequest::new(
                doc,
                EditKind::SetSimulationSpace { new: sim_space },
            ));
        }

        // Simulation condition.
        let mut sim_cond = header.simulation_condition;
        egui::ComboBox::from_label("Simulation condition")
            .selected_text(format!("{sim_cond:?}"))
            .show_ui(ui, |ui| {
                for option in [
                    SimulationCondition::Always,
                    SimulationCondition::WhenVisible,
                ] {
                    ui.selectable_value(&mut sim_cond, option, format!("{option:?}"));
                }
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
            ("prop-effect-zlayer", doc),
            "Z layer (2D)",
            header.z_layer_2d,
            f32::MIN..=f32::MAX,
            0.01,
        ) {
            edits.write(EditRequest::new(doc, EditKind::SetZLayer2d { new: new_z }));
        }

        // Capacity: read-only (set at bake time from the header).
        ui.horizontal(|ui| {
            ui.label("Capacity");
            ui.weak(format!("{}", header.capacity));
            ui.weak("(read-only)");
        });
    });
}

fn spawner_fields(
    ui: &mut egui::Ui,
    doc: Entity,
    current: SpawnerSettings,
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
) {
    collapsing(ui, "Spawner", |ui| {
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
            ("prop-spawner-count", doc),
            "Count",
            current_count,
            0.0..=f32::MAX,
            0.5,
        );
        let period = drag_f32(
            ui,
            ("prop-spawner-period", doc),
            "Period (s)",
            current_period,
            0.001..=f32::MAX,
            0.01,
        );
        let cycle_count = drag_u32(
            ui,
            ("prop-spawner-cycle", doc),
            "Cycle count (0 = infinite)",
            current_cycle,
        );

        let mut starts_active = current_active;
        let active_changed = ui.checkbox(&mut starts_active, "Starts active").changed();

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
/// Returns `Some(new_value)` on the frame the user commits the edit (drag
/// released or text-edit focus lost), or `None` otherwise. The draft is cleared
/// on commit so the next frame re-snapshots from the asset.
fn drag_f32(
    ui: &mut egui::Ui,
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
    let resp = ui
        .horizontal(|ui| {
            ui.label(label);
            ui.add(egui::DragValue::new(&mut value).range(range).speed(speed))
        })
        .inner;
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
    id_src: impl std::hash::Hash,
    label: &str,
    current: u32,
) -> Option<u32> {
    let id = egui::Id::new(id_src);
    let mut value: u32 = ui
        .ctx()
        .data_mut(|d| d.get_temp::<u32>(id).unwrap_or(current));
    let resp = ui
        .horizontal(|ui| {
            ui.label(label);
            ui.add(egui::DragValue::new(&mut value).range(0..=u32::MAX))
        })
        .inner;
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

fn collapsing<R>(ui: &mut egui::Ui, label: &str, add: impl FnOnce(&mut egui::Ui) -> R) {
    egui::CollapsingHeader::new(label)
        .default_open(true)
        .show(ui, add);
}
