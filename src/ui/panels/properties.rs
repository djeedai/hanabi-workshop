//! Properties panel.
//!
//! Phase 5a: edits the effect-level fields
//! (`EffectAsset.name`, `simulation_space`, `simulation_condition`,
//! `z_layer_2d`) and `SpawnerSettings` (count, period, cycle_count,
//! starts_active).
//!
//! Per-modifier field editing is deferred to Phase 5b — when a modifier
//! is selected, this panel shows its type name and a placeholder.
//!
//! ## Local-draft pattern
//!
//! Continuous edits (text typing, drag-value scrubbing) keep an
//! in-flight draft value in egui's per-id memory. A single
//! [`EditRequest`] is committed on `lost_focus()` (text) or
//! `drag_stopped()` (numeric), with the captured old/new values. This
//! collapses each user gesture into one undoable step and one
//! bevy_hanabi shader rebuild.

use bevy::prelude::*;
use bevy_egui::egui;
use bevy_hanabi::{
    CpuValue, EffectAsset, Expr, ScalarValue, SimulationCondition, SimulationSpace,
    SpawnerSettings, Value,
};

use crate::document::ModifierSelection;
use crate::edits::{EditKind, EditRequest};
use crate::proxy::{self, LiteralBinding};

pub fn show(
    ui: &mut egui::Ui,
    doc: Entity,
    effects: &Assets<EffectAsset>,
    effect_handle: &Handle<EffectAsset>,
    selection: Option<ModifierSelection>,
    bindings: &[LiteralBinding],
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
) {
    ui.heading("Properties");
    ui.separator();

    let Some(asset) = effects.get(effect_handle) else {
        ui.label("(effect asset not loaded yet)");
        return;
    };

    egui::ScrollArea::vertical().show(ui, |ui| {
        effect_fields(ui, doc, asset, edits);
        ui.add_space(8.0);
        spawner_fields(ui, doc, asset.spawner, edits);
        ui.add_space(8.0);
        live_tweakers(ui, doc, asset, bindings, edits);
        ui.add_space(8.0);
        ui.separator();
        modifier_section(ui, asset, selection);
    });
}

fn effect_fields(
    ui: &mut egui::Ui,
    doc: Entity,
    asset: &EffectAsset,
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
) {
    collapsing(ui, "Effect", |ui| {
        // Name: text field, committed on lost_focus.
        let id = egui::Id::new(("prop-effect-name", doc));
        let mut draft: String = ui
            .ctx()
            .data_mut(|d| d.get_temp::<String>(id).unwrap_or_else(|| asset.name.clone()));
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
            if draft != asset.name {
                edits.write(EditRequest::new(
                    doc,
                    EditKind::SetEffectName { new: draft.clone() },
                ));
            }
            ui.ctx().data_mut(|d| d.remove::<String>(id));
        }

        // Simulation space.
        let mut sim_space = asset.simulation_space;
        egui::ComboBox::from_label("Simulation space")
            .selected_text(format!("{sim_space:?}"))
            .show_ui(ui, |ui| {
                for option in [SimulationSpace::Global, SimulationSpace::Local] {
                    ui.selectable_value(&mut sim_space, option, format!("{option:?}"));
                }
            });
        if sim_space != asset.simulation_space {
            edits.write(EditRequest::new(
                doc,
                EditKind::SetSimulationSpace { new: sim_space },
            ));
        }

        // Simulation condition.
        let mut sim_cond = asset.simulation_condition;
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
        if sim_cond != asset.simulation_condition {
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
            asset.z_layer_2d,
            f32::MIN..=f32::MAX,
            0.01,
        ) {
            edits.write(EditRequest::new(doc, EditKind::SetZLayer2d { new: new_z }));
        }

        // Capacity: read-only (no public setter on EffectAsset 0.18).
        ui.horizontal(|ui| {
            ui.label("Capacity");
            ui.weak(format!("{}", asset.capacity()));
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
        let active_changed = ui
            .checkbox(&mut starts_active, "Starts active")
            .changed();

        let changed = count.is_some()
            || period.is_some()
            || cycle_count.is_some()
            || active_changed;

        if changed {
            let final_count = count.unwrap_or(current_count);
            let final_period = period.unwrap_or(current_period).max(0.001);
            let final_cycle = cycle_count.unwrap_or(current_cycle);

            // SpawnerSettings::new panics on cycle_count != 1 with a
            // degenerate period. Build with cycle_count=1 and a safe
            // period, then override cycle_count via the setter.
            let mut new = SpawnerSettings::new(
                final_count.into(),
                current.spawn_duration(),
                final_period.into(),
                1,
            )
            .with_starts_active(starts_active);
            new.set_cycle_count(final_cycle);
            edits.write(EditRequest::new(doc, EditKind::SetSpawnerSettings { new }));
        }
    });
}

/// Render a labelled `DragValue<f32>` backed by an egui-memory draft.
/// Returns `Some(new_value)` on the frame the user commits the edit
/// (drag released or text-edit focus lost), or `None` otherwise. The
/// draft is cleared on commit so the next frame re-snapshots from the
/// asset.
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

/// Phase 5b debug surface: list every literal that the proxy has
/// promoted to a synthetic property, and let the user tweak it live.
/// In the final UI these widgets will be embedded inline per modifier
/// field; this section is the temporary "raw bindings" view.
fn live_tweakers(
    ui: &mut egui::Ui,
    doc: Entity,
    asset: &EffectAsset,
    bindings: &[LiteralBinding],
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
) {
    if bindings.is_empty() {
        return;
    }
    collapsing(ui, "Live tweakers (debug)", |ui| {
        let module = asset.module();
        for binding in bindings {
            // Read the *current* canonical value from the module
            // arena — the binding's `last_value` is only a fallback
            // for property demotion.
            let current = match module.get(binding.canonical_expr) {
                Some(Expr::Literal(lit)) => proxy::literal_value(lit),
                _ => None,
            };
            let Some(current) = current else {
                ui.label(format!("{}: (slot no longer a literal)", binding.label));
                continue;
            };
            match current {
                Value::Scalar(ScalarValue::Float(f)) => {
                    if let Some(new) = drag_f32(
                        ui,
                        ("tweak-f32", doc, binding.canonical_expr),
                        &binding.label,
                        f,
                        f32::MIN..=f32::MAX,
                        0.01,
                    ) {
                        edits.write(EditRequest::new(
                            doc,
                            EditKind::SetLiteralValue {
                                canonical_expr: binding.canonical_expr,
                                new: Value::Scalar(ScalarValue::Float(new)),
                            },
                        ));
                    }
                }
                other => {
                    ui.label(format!("{}: {:?} (no editor yet)", binding.label, other));
                }
            }
        }
    });
}

fn modifier_section(
    ui: &mut egui::Ui,
    asset: &EffectAsset,
    selection: Option<ModifierSelection>,
) {
    ui.heading("Modifier");
    let Some(sel) = selection else {
        ui.weak("(select a modifier in the Outline panel)");
        return;
    };
    let label = match sel.group {
        crate::document::ModifierGroup::Init => asset
            .init_modifiers()
            .nth(sel.idx)
            .map(|m| short_type_name(m.reflect_type_path())),
        crate::document::ModifierGroup::Update => asset
            .update_modifiers()
            .nth(sel.idx)
            .map(|m| short_type_name(m.reflect_type_path())),
        crate::document::ModifierGroup::Render => asset
            .render_modifiers()
            .nth(sel.idx)
            .map(|m| short_type_name(m.as_modifier().reflect_type_path())),
    };
    match label {
        Some(name) => {
            ui.label(format!("{} [{}#{}]", name, sel.group.label(), sel.idx));
        }
        None => {
            ui.weak("(selected modifier no longer exists)");
        }
    }
    ui.weak("[per-field editing coming in Phase 5b]");
}

fn short_type_name(full: &str) -> String {
    full.rsplit("::").next().unwrap_or(full).to_string()
}

/// Best-effort scalar extraction from a `CpuValue<f32>`. `Single`
/// returns the value; uniform ranges fall back to the midpoint.
/// True range/random editing arrives in Phase 5b.
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
