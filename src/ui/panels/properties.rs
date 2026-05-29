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

use bevy::math::{Vec2, Vec3, Vec4};
use bevy::prelude::*;
use bevy_egui::egui;
use bevy_hanabi::graph::expr::PropertyExpr;
use bevy_hanabi::{
    CpuValue, EffectAsset, Expr, ExprHandle, Module, ScalarValue, SimulationCondition,
    SimulationSpace, SpawnerSettings, Value, VectorValue,
};

use crate::document::{ModifierGroup, ModifierSelection};
use crate::edits::{EditKind, EditRequest};
use crate::proxy;

pub fn show(
    ui: &mut egui::Ui,
    doc: Entity,
    effects: &Assets<EffectAsset>,
    effect_handle: &Handle<EffectAsset>,
    selection: Option<ModifierSelection>,
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
        ui.separator();
        modifier_section(ui, doc, asset, selection, edits);
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
fn modifier_section(
    ui: &mut egui::Ui,
    doc: Entity,
    asset: &EffectAsset,
    selection: Option<ModifierSelection>,
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
) {
    ui.heading("Modifier");
    let Some(sel) = selection else {
        ui.weak("(select a modifier in the Outline panel)");
        return;
    };

    // Resolve the selected modifier as a `&dyn Reflect`. Render
    // modifiers must be upcast via `as_modifier()`. We also keep its
    // type label for display.
    let modifier_ref: Option<&dyn bevy::reflect::Reflect> = match sel.group {
        ModifierGroup::Init => asset
            .init_modifiers()
            .nth(sel.idx)
            .map(|m| m.as_reflect()),
        ModifierGroup::Update => asset
            .update_modifiers()
            .nth(sel.idx)
            .map(|m| m.as_reflect()),
        ModifierGroup::Render => asset
            .render_modifiers()
            .nth(sel.idx)
            .map(|m| m.as_modifier().as_reflect()),
    };
    let Some(modifier) = modifier_ref else {
        ui.weak("(selected modifier no longer exists)");
        return;
    };

    let type_name = short_type_name(modifier.reflect_type_path());
    ui.label(format!(
        "{} [{}#{}]",
        type_name,
        sel.group.label(),
        sel.idx
    ));
    ui.add_space(4.0);

    // For SetAttributeModifier, annotate which attribute is being
    // initialised since the `value` field's type depends on it.
    if type_name == "SetAttributeModifier" {
        if let bevy::reflect::ReflectRef::Struct(s) = modifier.reflect_ref() {
            if let Some(attr) = s
                .field("attribute")
                .and_then(|f| f.try_downcast_ref::<bevy_hanabi::Attribute>())
            {
                ui.horizontal(|ui| {
                    ui.label("attribute");
                    ui.weak(attr.name());
                });
            }
        }
    }

    let module = asset.module();
    let fields = proxy::modifier_expr_fields(modifier);
    if fields.is_empty() {
        ui.weak(
            "(no editable expression fields — non-expression \
             fields are coming soon)",
        );
        return;
    }
    for (name, handle) in fields {
        expr_field_editor(ui, doc, &name, handle, module, edits);
    }
}

/// Render an editor for a single `ExprHandle` modifier field, with the
/// three-state UX:
/// - `Expr::Literal(_)` → typed inline editor (drag values / color);
/// - `Expr::Property(_)` → "(bound to <name>)" label (binding UI in
///   `5b-bind-field-to-property`);
/// - anything else → "(complex expression)".
fn expr_field_editor(
    ui: &mut egui::Ui,
    doc: Entity,
    field_name: &str,
    handle: ExprHandle,
    module: &Module,
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
) {
    let Some(expr) = module.get(handle) else {
        ui.horizontal(|ui| {
            ui.label(field_name);
            ui.weak("(invalid handle)");
        });
        return;
    };
    match expr {
        Expr::Literal(lit) => {
            let Some(value) = proxy::literal_value(lit) else {
                ui.horizontal(|ui| {
                    ui.label(field_name);
                    ui.weak("(unreadable literal)");
                });
                return;
            };
            literal_editor(ui, doc, field_name, handle, value, edits);
        }
        Expr::Property(pe) => {
            ui.horizontal(|ui| {
                ui.label(field_name);
                let name = property_name_for(module, pe).unwrap_or_else(|| "?".into());
                ui.weak(format!("← property {name}"));
            });
        }
        _ => {
            ui.horizontal(|ui| {
                ui.label(field_name);
                ui.weak("(complex expression)");
            });
        }
    }
}

fn property_name_for(module: &Module, pe: &PropertyExpr) -> Option<String> {
    let handle = proxy::property_handle_of(pe)?;
    module.get_property(handle).map(|p| p.name().to_string())
}

/// Type-dispatched editor for a `Value` literal. Emits a single
/// `SetLiteralValue` on commit.
fn literal_editor(
    ui: &mut egui::Ui,
    doc: Entity,
    label: &str,
    handle: ExprHandle,
    current: Value,
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
) {
    let id_base = ("modifier-lit", doc, handle);
    let emit = |edits: &mut bevy::ecs::message::MessageWriter<EditRequest>, new: Value| {
        edits.write(EditRequest::new(
            doc,
            EditKind::SetLiteralValue {
                canonical_expr: handle,
                new,
            },
        ));
    };

    match current {
        Value::Scalar(ScalarValue::Float(f)) => {
            if let Some(v) = drag_f32(ui, (id_base, "f"), label, f, f32::MIN..=f32::MAX, 0.01) {
                emit(edits, Value::Scalar(ScalarValue::Float(v)));
            }
        }
        Value::Scalar(ScalarValue::Int(i)) => {
            if let Some(v) = drag_i32(ui, (id_base, "i"), label, i) {
                emit(edits, Value::Scalar(ScalarValue::Int(v)));
            }
        }
        Value::Scalar(ScalarValue::Uint(u)) => {
            if let Some(v) = drag_u32(ui, (id_base, "u"), label, u) {
                emit(edits, Value::Scalar(ScalarValue::Uint(v)));
            }
        }
        Value::Scalar(ScalarValue::Bool(b)) => {
            let mut val = b;
            if ui
                .horizontal(|ui| {
                    ui.label(label);
                    ui.checkbox(&mut val, "")
                })
                .inner
                .changed()
                && val != b
            {
                emit(edits, Value::Scalar(ScalarValue::Bool(val)));
            }
        }
        Value::Vector(vv) => {
            vector_editor(ui, doc, label, handle, vv, edits);
        }
        Value::Matrix(_) => {
            ui.horizontal(|ui| {
                ui.label(label);
                ui.weak("(matrix literal — no editor yet)");
            });
        }
        _ => {
            ui.horizontal(|ui| {
                ui.label(label);
                ui.weak("(unsupported literal kind)");
            });
        }
    }
}

fn vector_editor(
    ui: &mut egui::Ui,
    doc: Entity,
    label: &str,
    handle: ExprHandle,
    vv: VectorValue,
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
) {
    use bevy_hanabi::attributes::VectorType;
    let id_base = ("modifier-lit-vec", doc, handle);
    match vv.vector_type() {
        VectorType::VEC2F => {
            let cur = vv.as_vec2();
            if let Some(v) = drag_vec_n(ui, id_base, label, &[cur.x, cur.y]) {
                emit_vec(edits, doc, handle, Value::Vector(VectorValue::new_vec2(Vec2::new(v[0], v[1]))));
            }
        }
        VectorType::VEC3F => {
            let cur = vv.as_vec3();
            if let Some(v) = drag_vec_n(ui, id_base, label, &[cur.x, cur.y, cur.z]) {
                emit_vec(
                    edits,
                    doc,
                    handle,
                    Value::Vector(VectorValue::new_vec3(Vec3::new(v[0], v[1], v[2]))),
                );
            }
        }
        VectorType::VEC4F => {
            let cur = vv.as_vec4();
            if let Some(v) = drag_vec_n(ui, id_base, label, &[cur.x, cur.y, cur.z, cur.w]) {
                emit_vec(
                    edits,
                    doc,
                    handle,
                    Value::Vector(VectorValue::new_vec4(Vec4::new(v[0], v[1], v[2], v[3]))),
                );
            }
        }
        other => {
            ui.horizontal(|ui| {
                ui.label(label);
                ui.weak(format!("({other:?} — no editor yet)"));
            });
        }
    }
}

fn emit_vec(
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
    doc: Entity,
    canonical_expr: ExprHandle,
    new: Value,
) {
    edits.write(EditRequest::new(
        doc,
        EditKind::SetLiteralValue {
            canonical_expr,
            new,
        },
    ));
}

/// Multi-component DragValue editor for `[f32; N]`. Returns `Some` on
/// commit (drag-stopped or focus-lost on any component changed since
/// the snapshot). Cached drafts use `(id_src, component_index)`.
fn drag_vec_n(
    ui: &mut egui::Ui,
    id_src: impl std::hash::Hash + Copy,
    label: &str,
    current: &[f32],
) -> Option<Vec<f32>> {
    let mut drafts: Vec<f32> = current.to_vec();
    let mut committed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        for (i, val) in drafts.iter_mut().enumerate() {
            let id = egui::Id::new((id_src, "comp", i));
            let mut cur: f32 = ui
                .ctx()
                .data_mut(|d| d.get_temp::<f32>(id).unwrap_or(*val));
            let resp = ui.add(egui::DragValue::new(&mut cur).speed(0.01));
            if resp.dragged() || resp.has_focus() || resp.changed() {
                ui.ctx().data_mut(|d| d.insert_temp(id, cur));
            }
            if resp.drag_stopped() || resp.lost_focus() {
                ui.ctx().data_mut(|d| d.remove::<f32>(id));
                if cur != *val {
                    committed = true;
                }
            }
            *val = cur;
        }
    });
    if committed {
        Some(drafts)
    } else {
        None
    }
}

fn drag_i32(
    ui: &mut egui::Ui,
    id_src: impl std::hash::Hash,
    label: &str,
    current: i32,
) -> Option<i32> {
    let id = egui::Id::new(id_src);
    let mut value: i32 = ui
        .ctx()
        .data_mut(|d| d.get_temp::<i32>(id).unwrap_or(current));
    let resp = ui
        .horizontal(|ui| {
            ui.label(label);
            ui.add(egui::DragValue::new(&mut value))
        })
        .inner;
    if resp.dragged() || resp.has_focus() || resp.changed() {
        ui.ctx().data_mut(|d| d.insert_temp(id, value));
    }
    if resp.drag_stopped() || resp.lost_focus() {
        ui.ctx().data_mut(|d| d.remove::<i32>(id));
        if value != current {
            return Some(value);
        }
    }
    None
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
