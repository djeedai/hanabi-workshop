//! Hand-editable curve and gradient widgets for keyframe authoring.
//!
//! Both edit a piecewise-linear sequence of keys `{ ratio ∈ [0,1], value }`:
//! [`CurveEditor`] over a numeric XY graph (e.g. size over lifetime) and
//! [`GradientBar`] over a color stop strip (e.g. color over lifetime). They are
//! stateless and mutate caller-owned key vectors in place, reporting `changed`
//! while a gesture is live and `committed` on the frame it ends, so a single
//! drag collapses into one persisted edit.
//!
//! [`CurveEditor`]: crate::curve::CurveEditor
//! [`GradientBar`]: crate::curve::GradientBar

use egui::{Color32, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};

/// Outcome of one widget interaction.
///
/// `changed` is set on any value mutation this frame; `committed` only on the
/// frame a gesture finishes (drag released, key added, or key removed) so the
/// caller can funnel one edit per gesture.
#[derive(Clone, Copy, Default)]
pub struct CurveResponse {
    pub changed: bool,
    pub committed: bool,
}

/// Radius (points) of a draggable key handle.
const HANDLE: f32 = 5.0;
/// Pointer pick distance (points) for grabbing a handle.
const PICK: f32 = 11.0;

/// Sort keys by ratio and pin the first/last to the 0 and 1 bounds.
///
/// Hanabi samples a gradient as constant outside `[0, 1]`, so the endpoints are
/// kept anchored and only interior keys move freely along the ratio axis.
fn normalize<T>(keys: &mut [(f32, T)]) {
    keys.sort_by(|a, b| a.0.total_cmp(&b.0));
    if let Some(first) = keys.first_mut() {
        first.0 = 0.0;
    }
    if let Some(last) = keys.last_mut() {
        last.0 = 1.0;
    }
}

/// Index of the color stop whose marker is nearest `p` along x, within
/// [`PICK`].
fn nearest_stop(keys: &[(f32, [f32; 4])], rect: Rect, p: Pos2) -> Option<usize> {
    let to_x = |ratio: f32| rect.left() + ratio.clamp(0.0, 1.0) * rect.width();
    keys.iter()
        .enumerate()
        .map(|(i, k)| (i, (to_x(k.0) - p.x).abs()))
        .filter(|(_, d)| *d <= PICK)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
}

/// XY graph editor for a numeric over-lifetime curve.
///
/// Keys are `(ratio, value)`. Drag a handle to move it, double-click empty
/// space to insert, right-click a handle to delete. The first and last keys
/// stay pinned to ratio 0 and 1; interior keys move on both axes.
pub struct CurveEditor<'k> {
    keys: &'k mut Vec<(f32, f32)>,
    y_min: f32,
    y_max: f32,
    height: f32,
}

impl<'k> CurveEditor<'k> {
    /// Edit `keys` (a `(ratio, value)` list); at least one key is kept.
    pub fn new(keys: &'k mut Vec<(f32, f32)>) -> Self {
        Self {
            keys,
            y_min: 0.0,
            y_max: 1.0,
            height: 120.0,
        }
    }

    /// Set the visible value range mapped to the graph's vertical extent.
    pub fn y_range(mut self, min: f32, max: f32) -> Self {
        self.y_min = min;
        self.y_max = max.max(min + f32::EPSILON);
        self
    }

    /// Set the graph height in points.
    pub fn height(mut self, height: f32) -> Self {
        self.height = height;
        self
    }

    /// Render the editor and apply this frame's input.
    pub fn show(self, ui: &mut egui::Ui) -> CurveResponse {
        let mut out = CurveResponse::default();
        if self.keys.is_empty() {
            self.keys.push((0.0, self.y_min));
            self.keys.push((1.0, self.y_max));
            out.changed = true;
        }
        let width = ui.available_width().max(80.0);
        let (rect, resp) =
            ui.allocate_exact_size(Vec2::new(width, self.height), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        let vis = ui.visuals();
        painter.rect_filled(rect, 4.0, vis.extreme_bg_color);
        painter.rect_stroke(
            rect,
            4.0,
            vis.widgets.noninteractive.bg_stroke,
            StrokeKind::Inside,
        );

        let to_screen = |ratio: f32, value: f32| -> Pos2 {
            let nx = ratio.clamp(0.0, 1.0);
            let ny = ((value - self.y_min) / (self.y_max - self.y_min)).clamp(0.0, 1.0);
            Pos2::new(
                rect.left() + nx * rect.width(),
                rect.bottom() - ny * rect.height(),
            )
        };
        let from_screen = |p: Pos2| -> (f32, f32) {
            let nx = ((p.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
            let ny = ((rect.bottom() - p.y) / rect.height()).clamp(0.0, 1.0);
            (nx, self.y_min + ny * (self.y_max - self.y_min))
        };

        for i in 0..self.keys.len().saturating_sub(1) {
            let a = to_screen(self.keys[i].0, self.keys[i].1);
            let b = to_screen(self.keys[i + 1].0, self.keys[i + 1].1);
            painter.line_segment([a, b], Stroke::new(2.0, vis.widgets.active.fg_stroke.color));
        }

        let active_id = ui.id().with("curve-drag");
        let mut dragging: Option<usize> = ui.ctx().data(|d| d.get_temp::<usize>(active_id));

        if resp.drag_started()
            && let Some(p) = resp.interact_pointer_pos()
        {
            dragging = self
                .keys
                .iter()
                .enumerate()
                .map(|(i, k)| (i, to_screen(k.0, k.1).distance(p)))
                .filter(|(_, d)| *d <= PICK)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(i, _)| i);
            match dragging {
                Some(i) => ui.ctx().data_mut(|d| {
                    d.insert_temp(active_id, i);
                }),
                None => ui.ctx().data_mut(|d| {
                    d.remove::<usize>(active_id);
                }),
            }
        }
        if resp.dragged()
            && let (Some(i), Some(p)) = (dragging, resp.interact_pointer_pos())
        {
            let (r, v) = from_screen(p);
            let last = self.keys.len() - 1;
            self.keys[i].1 = v;
            if i != 0 && i != last {
                self.keys[i].0 = r;
            }
            out.changed = true;
        }
        if resp.drag_stopped() {
            if dragging.is_some() {
                normalize(self.keys);
                out.committed = true;
            }
            ui.ctx().data_mut(|d| d.remove::<usize>(active_id));
            dragging = None;
        }

        if resp.double_clicked()
            && let Some(p) = resp.interact_pointer_pos()
        {
            let on_handle = self
                .keys
                .iter()
                .any(|k| to_screen(k.0, k.1).distance(p) <= PICK);
            if !on_handle {
                let (r, v) = from_screen(p);
                self.keys.push((r.clamp(0.001, 0.999), v));
                normalize(self.keys);
                out.committed = true;
                out.changed = true;
            }
        }
        if resp.secondary_clicked()
            && self.keys.len() > 1
            && let Some(p) = resp.interact_pointer_pos()
            && let Some((i, d)) = self
                .keys
                .iter()
                .enumerate()
                .map(|(i, k)| (i, to_screen(k.0, k.1).distance(p)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
            && d <= PICK
        {
            self.keys.remove(i);
            normalize(self.keys);
            out.committed = true;
            out.changed = true;
        }

        for (i, k) in self.keys.iter().enumerate() {
            let c = to_screen(k.0, k.1);
            let hot = dragging == Some(i);
            painter.circle_filled(
                c,
                if hot { HANDLE + 1.0 } else { HANDLE },
                vis.selection.bg_fill,
            );
            painter.circle_stroke(c, HANDLE, Stroke::new(1.0, vis.extreme_bg_color));
        }

        out
    }
}

/// Color stop strip editor for a color over-lifetime gradient.
///
/// Keys are `(ratio, [r, g, b, a])`. Click a stop to select, drag to move,
/// double-click empty space to insert, right-click to delete; the selected
/// stop's color is edited with a swatch picker below the strip. The first and
/// last stops stay pinned to ratio 0 and 1.
pub struct GradientBar<'k> {
    keys: &'k mut Vec<(f32, [f32; 4])>,
    height: f32,
}

impl<'k> GradientBar<'k> {
    /// Edit `keys` (a `(ratio, rgba)` list); at least one stop is kept.
    pub fn new(keys: &'k mut Vec<(f32, [f32; 4])>) -> Self {
        Self { keys, height: 28.0 }
    }

    /// Set the gradient strip height in points.
    pub fn height(mut self, height: f32) -> Self {
        self.height = height;
        self
    }

    /// Render the strip plus the selected-stop color picker.
    pub fn show(self, ui: &mut egui::Ui) -> CurveResponse {
        let mut out = CurveResponse::default();
        if self.keys.is_empty() {
            self.keys.push((0.0, [0.0, 0.0, 0.0, 1.0]));
            self.keys.push((1.0, [1.0, 1.0, 1.0, 1.0]));
            out.changed = true;
        }
        let width = ui.available_width().max(80.0);
        let (rect, resp) =
            ui.allocate_exact_size(Vec2::new(width, self.height), Sense::click_and_drag());
        let painter = ui.painter_at(rect);

        let to_x = |ratio: f32| rect.left() + ratio.clamp(0.0, 1.0) * rect.width();
        let to_ratio = |x: f32| ((x - rect.left()) / rect.width()).clamp(0.0, 1.0);
        let col = |c: &[f32; 4]| {
            Color32::from(egui::Rgba::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]))
        };
        for i in 0..self.keys.len().saturating_sub(1) {
            let a = &self.keys[i];
            let b = &self.keys[i + 1];
            let mut mesh = egui::Mesh::default();
            let (ca, cb) = (col(&a.1), col(&b.1));
            let (xa, xb) = (to_x(a.0), to_x(b.0));
            mesh.colored_vertex(Pos2::new(xa, rect.top()), ca);
            mesh.colored_vertex(Pos2::new(xa, rect.bottom()), ca);
            mesh.colored_vertex(Pos2::new(xb, rect.top()), cb);
            mesh.colored_vertex(Pos2::new(xb, rect.bottom()), cb);
            mesh.add_triangle(0, 1, 2);
            mesh.add_triangle(2, 1, 3);
            painter.add(mesh);
        }
        painter.rect_stroke(
            rect,
            2.0,
            ui.visuals().widgets.noninteractive.bg_stroke,
            StrokeKind::Inside,
        );

        let sel_id = ui.id().with("grad-sel");
        let mut selected: usize = ui.ctx().data(|d| d.get_temp(sel_id).unwrap_or(0));
        let drag_id = ui.id().with("grad-drag");
        let mut dragging: Option<usize> = ui.ctx().data(|d| d.get_temp::<usize>(drag_id));

        if resp.drag_started()
            && let Some(p) = resp.interact_pointer_pos()
        {
            dragging = nearest_stop(self.keys, rect, p);
            if let Some(i) = dragging {
                selected = i;
                ui.ctx().data_mut(|d| {
                    d.insert_temp(drag_id, i);
                });
            } else {
                ui.ctx().data_mut(|d| {
                    d.remove::<usize>(drag_id);
                });
            }
        }
        if resp.dragged()
            && let (Some(i), Some(p)) = (dragging, resp.interact_pointer_pos())
        {
            let last = self.keys.len() - 1;
            if i != 0 && i != last {
                self.keys[i].0 = to_ratio(p.x);
                out.changed = true;
            }
        }
        if resp.drag_stopped() {
            if let Some(i) = dragging {
                let val = self.keys[i];
                normalize(self.keys);
                selected = self.keys.iter().position(|k| *k == val).unwrap_or(0);
                out.committed = true;
            }
            ui.ctx().data_mut(|d| {
                d.remove::<usize>(drag_id);
            });
        }
        // A plain click (no drag) selects the nearest stop.
        if resp.clicked()
            && let Some(i) = resp
                .interact_pointer_pos()
                .and_then(|p| nearest_stop(self.keys, rect, p))
        {
            selected = i;
        }
        if resp.double_clicked()
            && let Some(p) = resp.interact_pointer_pos()
            && nearest_stop(self.keys, rect, p).is_none()
        {
            let r = to_ratio(p.x).clamp(0.001, 0.999);
            self.keys.push((r, [1.0, 1.0, 1.0, 1.0]));
            normalize(self.keys);
            selected = self.keys.iter().position(|k| k.0 == r).unwrap_or(0);
            out.committed = true;
            out.changed = true;
        }
        if resp.secondary_clicked()
            && self.keys.len() > 1
            && let Some(i) = resp
                .interact_pointer_pos()
                .and_then(|p| nearest_stop(self.keys, rect, p))
        {
            self.keys.remove(i);
            out.committed = true;
            out.changed = true;
        }
        selected = selected.min(self.keys.len() - 1);

        for (i, k) in self.keys.iter().enumerate() {
            let x = to_x(k.0);
            let r = Rect::from_center_size(
                Pos2::new(x, rect.bottom() - HANDLE),
                Vec2::new(HANDLE * 2.0, HANDLE * 2.0),
            );
            let outline = if i == selected {
                ui.visuals().selection.stroke.color
            } else {
                ui.visuals().extreme_bg_color
            };
            painter.rect_filled(r, 1.0, col(&k.1));
            painter.rect_stroke(r, 1.0, Stroke::new(1.5, outline), StrokeKind::Outside);
        }

        let mut rgba = self.keys[selected].1;
        if ui.color_edit_button_rgba_unmultiplied(&mut rgba).changed() {
            self.keys[selected].1 = rgba;
            out.changed = true;
            out.committed = true;
        }

        ui.ctx().data_mut(|d| d.insert_temp(sel_id, selected));
        out
    }
}
