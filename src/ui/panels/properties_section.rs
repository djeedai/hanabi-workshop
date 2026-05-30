//! User-properties section inside the Effect panel.
//!
//! Lists every user-defined property on the canonical asset's
//! `Module` (skipping the synthetic `hwk_tweak_*` ones the proxy
//! injects). Each row offers rename, initial-value editing, and
//! remove. An "Add property" row at the bottom takes a name and
//! type and emits an `AddProperty` edit.

use bevy::math::{Vec2, Vec3, Vec4};
use bevy::prelude::*;
use bevy_egui::egui;
use bevy_hanabi::{EffectAsset, ScalarValue, Value, VectorValue};

use crate::edits::{EditKind, EditRequest};
use crate::proxy;

/// Top-level entry point for the standalone Properties tab. Wraps
/// [`show`] in a vertical scroll area; resolves the asset lazily so a
/// not-yet-loaded handle just renders a placeholder.
pub fn show_panel(
    ui: &mut egui::Ui,
    doc: Entity,
    effects: &Assets<EffectAsset>,
    effect_handle: &Handle<EffectAsset>,
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
) {
    let Some(asset) = effects.get(effect_handle) else {
        ui.label("(effect asset not loaded yet)");
        return;
    };
    egui::ScrollArea::vertical().show(ui, |ui| {
        show(ui, doc, asset, edits);
    });
}

/// Render the "Properties" collapsing section. Pure-UI helper; never
/// mutates the asset directly — only emits [`EditRequest`].
pub fn show(
    ui: &mut egui::Ui,
    doc: Entity,
    asset: &EffectAsset,
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
) {
    let props = proxy::user_properties(asset.module());

    egui::CollapsingHeader::new(format!("Properties ({})", props.len()))
        .id_salt(("effect-properties", doc))
        .default_open(true)
        .show(ui, |ui| {
            for (name, value) in &props {
                property_row(ui, doc, name, *value, edits);
            }
            if props.is_empty() {
                ui.weak("(none)");
            }
            ui.add_space(4.0);
            add_property_row(ui, doc, &props, edits);
        });
}

fn property_row(
    ui: &mut egui::Ui,
    doc: Entity,
    name: &str,
    value: Value,
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
) {
    ui.group(|ui| {
        ui.horizontal(|ui| {
            // Name (rename on lost_focus). Per-row id keyed by the
            // *current* name so renaming swaps the draft slot.
            let id = egui::Id::new(("prop-name", doc, name));
            let mut draft: String = ui
                .ctx()
                .data_mut(|d| d.get_temp::<String>(id).unwrap_or_else(|| name.to_string()));
            let resp = ui.add(egui::TextEdit::singleline(&mut draft).desired_width(140.0));
            if resp.has_focus() || resp.changed() {
                ui.ctx().data_mut(|d| d.insert_temp(id, draft.clone()));
            }
            if resp.lost_focus() {
                let trimmed = draft.trim().to_string();
                ui.ctx().data_mut(|d| d.remove::<String>(id));
                if !trimmed.is_empty() && trimmed != name {
                    edits.write(EditRequest::new(
                        doc,
                        EditKind::RenameProperty {
                            old: name.to_string(),
                            new: trimmed,
                        },
                    ));
                }
            }

            ui.weak(format!("[{}]", value_type_label(value)));

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let remove = ui
                    .small_button(crate::ui::icons::ICON_XMARK.to_string())
                    .on_hover_text("Remove this property");
                if remove.clicked() {
                    edits.write(EditRequest::new(
                        doc,
                        EditKind::RemoveProperty {
                            name: name.to_string(),
                        },
                    ));
                }
            });
        });

        // Initial-value editor — typed by the current Value kind.
        value_editor(ui, doc, name, value, edits);
    });
}

fn add_property_row(
    ui: &mut egui::Ui,
    doc: Entity,
    existing: &[(String, Value)],
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
) {
    // Per-doc draft state for the add-row.
    let name_id = egui::Id::new(("add-prop-name", doc));
    let kind_id = egui::Id::new(("add-prop-kind", doc));
    let mut name: String = ui
        .ctx()
        .data_mut(|d| d.get_temp::<String>(name_id).unwrap_or_default());
    let mut kind: AddKind = ui
        .ctx()
        .data_mut(|d| d.get_temp::<AddKind>(kind_id).unwrap_or(AddKind::Float));

    let mut submit = false;
    ui.horizontal(|ui| {
        ui.label("Add:");
        let resp = ui.add(
            egui::TextEdit::singleline(&mut name)
                .desired_width(140.0)
                .hint_text("name"),
        );
        if resp.changed() {
            ui.ctx().data_mut(|d| d.insert_temp(name_id, name.clone()));
        }
        egui::ComboBox::from_id_salt(("add-prop-kind-combo", doc))
            .selected_text(kind.label())
            .show_ui(ui, |ui| {
                for opt in AddKind::ALL {
                    if ui.selectable_label(kind == *opt, opt.label()).clicked() {
                        kind = *opt;
                        ui.ctx().data_mut(|d| d.insert_temp(kind_id, kind));
                    }
                }
            });
        let trimmed = name.trim().to_string();
        let valid = !trimmed.is_empty()
            && !proxy::is_tweak_prop_name(&trimmed)
            && !existing.iter().any(|(n, _)| n == &trimmed);
        if ui
            .add_enabled(valid, egui::Button::new("+"))
            .on_disabled_hover_text(if trimmed.is_empty() {
                "Enter a name first"
            } else if proxy::is_tweak_prop_name(&trimmed) {
                "Name uses the reserved 'hwk_tweak_' prefix"
            } else {
                "A property with that name already exists"
            })
            .clicked()
        {
            submit = true;
        }
    });

    if submit {
        let trimmed = name.trim().to_string();
        edits.write(EditRequest::new(
            doc,
            EditKind::AddProperty {
                name: trimmed,
                value: kind.default_value(),
            },
        ));
        // Clear the draft so the field is empty for the next add.
        ui.ctx().data_mut(|d| d.remove::<String>(name_id));
    }
}

/// Type-dispatched editor for a property's initial value. Same shape
/// as the modifier-literal editor in `properties.rs`. Emits
/// [`EditKind::SetPropertyDefault`] on drag-stop / focus-loss.
fn value_editor(
    ui: &mut egui::Ui,
    doc: Entity,
    name: &str,
    current: Value,
    edits: &mut bevy::ecs::message::MessageWriter<EditRequest>,
) {
    let id_base = ("prop-value", doc, name);
    let emit = |edits: &mut bevy::ecs::message::MessageWriter<EditRequest>, new: Value| {
        edits.write(EditRequest::new(
            doc,
            EditKind::SetPropertyDefault {
                name: name.to_string(),
                new,
            },
        ));
    };
    match current {
        Value::Scalar(ScalarValue::Float(f)) => {
            if let Some(v) = drag_f32(ui, (id_base, "f"), "value", f, 0.01) {
                emit(edits, Value::Scalar(ScalarValue::Float(v)));
            }
        }
        Value::Scalar(ScalarValue::Int(i)) => {
            if let Some(v) = drag_i32(ui, (id_base, "i"), "value", i) {
                emit(edits, Value::Scalar(ScalarValue::Int(v)));
            }
        }
        Value::Scalar(ScalarValue::Uint(u)) => {
            if let Some(v) = drag_u32(ui, (id_base, "u"), "value", u) {
                emit(edits, Value::Scalar(ScalarValue::Uint(v)));
            }
        }
        Value::Scalar(ScalarValue::Bool(b)) => {
            let mut val = b;
            if ui
                .horizontal(|ui| {
                    ui.label("value");
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
            use bevy_hanabi::attributes::VectorType;
            match vv.vector_type() {
                VectorType::VEC2F => {
                    let c = vv.as_vec2();
                    if let Some(v) = drag_vec_n(ui, id_base, "value", &[c.x, c.y]) {
                        emit(
                            edits,
                            Value::Vector(VectorValue::new_vec2(Vec2::new(v[0], v[1]))),
                        );
                    }
                }
                VectorType::VEC3F => {
                    let c = vv.as_vec3();
                    if let Some(v) = drag_vec_n(ui, id_base, "value", &[c.x, c.y, c.z]) {
                        emit(
                            edits,
                            Value::Vector(VectorValue::new_vec3(Vec3::new(v[0], v[1], v[2]))),
                        );
                    }
                }
                VectorType::VEC4F => {
                    let c = vv.as_vec4();
                    if let Some(v) = drag_vec_n(ui, id_base, "value", &[c.x, c.y, c.z, c.w]) {
                        emit(
                            edits,
                            Value::Vector(VectorValue::new_vec4(Vec4::new(v[0], v[1], v[2], v[3]))),
                        );
                    }
                }
                other => {
                    ui.weak(format!("({other:?} — no editor yet)"));
                }
            }
        }
        _ => {
            ui.weak("(no editor for this value type yet)");
        }
    }
}

fn value_type_label(v: Value) -> &'static str {
    match v {
        Value::Scalar(ScalarValue::Float(_)) => "f32",
        Value::Scalar(ScalarValue::Int(_)) => "i32",
        Value::Scalar(ScalarValue::Uint(_)) => "u32",
        Value::Scalar(ScalarValue::Bool(_)) => "bool",
        Value::Vector(vv) => {
            use bevy_hanabi::attributes::VectorType;
            match vv.vector_type() {
                VectorType::VEC2F => "vec2<f32>",
                VectorType::VEC3F => "vec3<f32>",
                VectorType::VEC4F => "vec4<f32>",
                _ => "vec?",
            }
        }
        Value::Matrix(_) => "mat",
        _ => "?",
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum AddKind {
    #[default]
    Float,
    Int,
    Uint,
    Bool,
    Vec2,
    Vec3,
    Vec4,
}

impl AddKind {
    const ALL: &'static [AddKind] = &[
        AddKind::Float,
        AddKind::Int,
        AddKind::Uint,
        AddKind::Bool,
        AddKind::Vec2,
        AddKind::Vec3,
        AddKind::Vec4,
    ];

    fn label(self) -> &'static str {
        match self {
            AddKind::Float => "f32",
            AddKind::Int => "i32",
            AddKind::Uint => "u32",
            AddKind::Bool => "bool",
            AddKind::Vec2 => "vec2<f32>",
            AddKind::Vec3 => "vec3<f32>",
            AddKind::Vec4 => "vec4<f32>",
        }
    }

    fn default_value(self) -> Value {
        match self {
            AddKind::Float => Value::Scalar(ScalarValue::Float(0.0)),
            AddKind::Int => Value::Scalar(ScalarValue::Int(0)),
            AddKind::Uint => Value::Scalar(ScalarValue::Uint(0)),
            AddKind::Bool => Value::Scalar(ScalarValue::Bool(false)),
            AddKind::Vec2 => Value::Vector(VectorValue::new_vec2(Vec2::ZERO)),
            AddKind::Vec3 => Value::Vector(VectorValue::new_vec3(Vec3::ZERO)),
            AddKind::Vec4 => Value::Vector(VectorValue::new_vec4(Vec4::ZERO)),
        }
    }
}

// ---------------------------------------------------------------------------
// Local copies of the drag-with-draft helpers. Kept private here so the
// properties panel module can remain self-contained; the equivalents
// in `properties.rs` aren't pub.
// ---------------------------------------------------------------------------

fn drag_f32(
    ui: &mut egui::Ui,
    id_src: impl std::hash::Hash,
    label: &str,
    current: f32,
    speed: f32,
) -> Option<f32> {
    let id = egui::Id::new(id_src);
    let mut value: f32 = ui
        .ctx()
        .data_mut(|d| d.get_temp::<f32>(id).unwrap_or(current));
    let resp = ui
        .horizontal(|ui| {
            ui.label(label);
            ui.add(egui::DragValue::new(&mut value).speed(speed))
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
            let mut cur: f32 = ui.ctx().data_mut(|d| d.get_temp::<f32>(id).unwrap_or(*val));
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
    if committed { Some(drafts) } else { None }
}
