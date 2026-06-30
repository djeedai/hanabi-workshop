//! Type-dispatched editors for a `bevy_hanabi::Value`.
//!
//! Each continuous editor keeps an in-flight draft in egui per-id memory and
//! returns `Some(new)` only on the frame the gesture commits (drag released or
//! text-edit focus lost), so a single user gesture collapses into one edit.

use bevy::math::{Vec2, Vec3, Vec4};
use bevy_egui::egui;
use bevy_hanabi::{ScalarValue, Value, VectorValue};

/// Accent color mapping a vector component to its spatial/color axis.
///
/// Shared with the viewport's axis gizmo so X/Y/Z read as the same
/// red/green/blue everywhere; the `W` (alpha) component uses a neutral grey.
pub const AXIS_X_COLOR: egui::Color32 = egui::Color32::from_rgb(232, 91, 91);
pub const AXIS_Y_COLOR: egui::Color32 = egui::Color32::from_rgb(126, 207, 80);
pub const AXIS_Z_COLOR: egui::Color32 = egui::Color32::from_rgb(70, 132, 232);
pub const AXIS_W_COLOR: egui::Color32 = egui::Color32::from_rgb(150, 150, 150);

/// Render an editor for `current`, dispatched on its value kind.
///
/// Returns `Some(new_value)` on the frame the user commits a change. `id_base`
/// must be stable for the edited target so drafts survive across frames.
pub fn value_editor(
    ui: &mut egui::Ui,
    id_base: impl std::hash::Hash + Copy,
    current: Value,
) -> Option<Value> {
    value_editor_impl(ui, id_base, current, None)
}

/// Like [`value_editor`] but lays the control out at `size` for a value chip.
///
/// Overlays directly on a node-graph value chip. Only scalar and bool values
/// are supported inline here; vec3/vec4 use the graph panel's stacked
/// per-component editor, and other vectors are edited via the popup
/// `value_editor`.
pub fn inline_value_editor(
    ui: &mut egui::Ui,
    id_base: impl std::hash::Hash + Copy,
    current: Value,
    size: egui::Vec2,
) -> Option<Value> {
    value_editor_impl(ui, id_base, current, Some(size))
}

fn value_editor_impl(
    ui: &mut egui::Ui,
    id_base: impl std::hash::Hash + Copy,
    current: Value,
    size: Option<egui::Vec2>,
) -> Option<Value> {
    match current {
        Value::Scalar(ScalarValue::Float(f)) => drag_f32(ui, (id_base, "f"), f, 0.01, size)
            .map(|v| Value::Scalar(ScalarValue::Float(v))),
        Value::Scalar(ScalarValue::Int(i)) => {
            drag_i32(ui, (id_base, "i"), i, size).map(|v| Value::Scalar(ScalarValue::Int(v)))
        }
        Value::Scalar(ScalarValue::Uint(u)) => {
            drag_u32(ui, (id_base, "u"), u, size).map(|v| Value::Scalar(ScalarValue::Uint(v)))
        }
        Value::Scalar(ScalarValue::Bool(b)) => {
            let toggled = match size {
                Some(s) => {
                    add_sized_left(ui, s, egui::Button::new(if b { "true" } else { "false" }))
                        .clicked()
                }
                None => {
                    let mut val = b;
                    ui.checkbox(&mut val, "").changed()
                }
            };
            if toggled {
                Some(Value::Scalar(ScalarValue::Bool(!b)))
            } else {
                None
            }
        }
        Value::Vector(vv) => {
            use bevy_hanabi::attributes::VectorType;
            match vv.vector_type() {
                VectorType::VEC2F => {
                    let c = vv.as_vec2();
                    drag_vec_n(ui, id_base, &[c.x, c.y])
                        .map(|v| Value::Vector(VectorValue::new_vec2(Vec2::new(v[0], v[1]))))
                }
                VectorType::VEC3F => {
                    let c = vv.as_vec3();
                    drag_vec_n(ui, id_base, &[c.x, c.y, c.z])
                        .map(|v| Value::Vector(VectorValue::new_vec3(Vec3::new(v[0], v[1], v[2]))))
                }
                VectorType::VEC4F => {
                    let c = vv.as_vec4();
                    drag_vec_n(ui, id_base, &[c.x, c.y, c.z, c.w]).map(|v| {
                        Value::Vector(VectorValue::new_vec4(Vec4::new(v[0], v[1], v[2], v[3])))
                    })
                }
                other => {
                    ui.weak(format!("({other:?} — no editor yet)"));
                    None
                }
            }
        }
        _ => {
            ui.weak("(no editor for this value type yet)");
            None
        }
    }
}

/// Like [`egui::Ui::add_sized`] but anchored to the region's left edge.
///
/// `add_sized` justifies only along the parent's main axis; inside the inline
/// chip's vertical layout that means horizontal centering, so a widget wider
/// than `size` spills equally left and right — over the adjacent port label.
/// Justifying horizontally and aligning left keeps any overflow on the right,
/// clear of the label.
fn add_sized_left(
    ui: &mut egui::Ui,
    size: egui::Vec2,
    widget: impl egui::Widget,
) -> egui::Response {
    let layout = egui::Layout::left_to_right(egui::Align::Center).with_main_justify(true);
    ui.allocate_ui_with_layout(size, layout, |ui| ui.add(widget))
        .inner
}

fn drag_f32(
    ui: &mut egui::Ui,
    id_src: impl std::hash::Hash,
    current: f32,
    speed: f32,
    size: Option<egui::Vec2>,
) -> Option<f32> {
    let id = egui::Id::new(id_src);
    let mut value: f32 = ui
        .ctx()
        .data_mut(|d| d.get_temp::<f32>(id).unwrap_or(current));
    let dv = egui::DragValue::new(&mut value).speed(speed);
    let resp = match size {
        Some(s) => add_sized_left(ui, s, dv),
        None => ui.add(dv),
    };
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
    current: i32,
    size: Option<egui::Vec2>,
) -> Option<i32> {
    let id = egui::Id::new(id_src);
    let mut value: i32 = ui
        .ctx()
        .data_mut(|d| d.get_temp::<i32>(id).unwrap_or(current));
    let dv = egui::DragValue::new(&mut value);
    let resp = match size {
        Some(s) => add_sized_left(ui, s, dv),
        None => ui.add(dv),
    };
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
    current: u32,
    size: Option<egui::Vec2>,
) -> Option<u32> {
    let id = egui::Id::new(id_src);
    let mut value: u32 = ui
        .ctx()
        .data_mut(|d| d.get_temp::<u32>(id).unwrap_or(current));
    let dv = egui::DragValue::new(&mut value).range(0..=u32::MAX);
    let resp = match size {
        Some(s) => add_sized_left(ui, s, dv),
        None => ui.add(dv),
    };
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

/// Multi-component `[f32; N]` editor laid out horizontally.
///
/// Commits when any component's gesture ends and the value changed since the
/// snapshot.
fn drag_vec_n(
    ui: &mut egui::Ui,
    id_src: impl std::hash::Hash + Copy,
    current: &[f32],
) -> Option<Vec<f32>> {
    let mut drafts: Vec<f32> = current.to_vec();
    let mut committed = false;
    ui.horizontal(|ui| {
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
