use std::collections::HashMap;

use bevy::prelude::*;
use bevy_egui::egui;

use crate::{
    document::ViewportSizeRequests,
    plugins::camera_control::{
        CameraControl, CameraControlMessage, ORBIT_RAD_PER_PIXEL, PAN_FRACTION_PER_PIXEL,
        ZOOM_PER_NOTCH,
    },
};

/// Render the viewport panel and its controls.
///
/// Displays the render-target image, records the panel's pixel size, interprets
/// pointer input for orbit-camera control, and overlays a grid toggle and
/// Blender-style axis gizmo.
pub fn show(
    ui: &mut egui::Ui,
    doc_entity: Entity,
    viewport_index: usize,
    viewport_textures: &HashMap<(Entity, usize), egui::TextureId>,
    size_requests: &mut ViewportSizeRequests,
    cam_msgs: &mut bevy::ecs::message::MessageWriter<CameraControlMessage>,
    cameras: &Query<(&'static crate::document::ViewportCamera, &'static ChildOf)>,
    show_grid: &mut bool,
) {
    let Some(tex) = viewport_textures
        .get(&(doc_entity, viewport_index))
        .copied()
    else {
        ui.centered_and_justified(|ui| {
            ui.label("(waiting for render target)");
        });
        return;
    };
    let size = ui.available_size();
    let resp = ui.add(
        egui::Image::new(egui::load::SizedTexture::new(tex, size))
            .sense(egui::Sense::click_and_drag()),
    );

    let pixels_per_point = ui.ctx().pixels_per_point();
    let px = UVec2::new(
        (size.x * pixels_per_point).max(1.0) as u32,
        (size.y * pixels_per_point).max(1.0) as u32,
    );
    size_requests.0.insert((doc_entity, viewport_index), px);

    // === Orbit (LMB drag) / Pan (RMB drag) ===
    let drag = resp.drag_delta();
    if drag != egui::Vec2::ZERO {
        if resp.dragged_by(egui::PointerButton::Primary) {
            cam_msgs.write(CameraControlMessage {
                doc: doc_entity,
                viewport_index,
                control: CameraControl::Orbit {
                    yaw: drag.x * ORBIT_RAD_PER_PIXEL,
                    pitch: drag.y * ORBIT_RAD_PER_PIXEL,
                },
            });
        } else if resp.dragged_by(egui::PointerButton::Secondary) {
            cam_msgs.write(CameraControlMessage {
                doc: doc_entity,
                viewport_index,
                control: CameraControl::Pan {
                    dx: drag.x * PAN_FRACTION_PER_PIXEL,
                    dy: drag.y * PAN_FRACTION_PER_PIXEL,
                },
            });
        }
    }

    // === Zoom (scroll wheel while hovered) ===
    if resp.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            let notches = scroll / 50.0;
            let factor = ZOOM_PER_NOTCH.powf(notches);
            cam_msgs.write(CameraControlMessage {
                doc: doc_entity,
                viewport_index,
                control: CameraControl::Zoom { factor },
            });
        }
    }

    draw_grid_toggle(ui, resp.rect, viewport_index, show_grid);

    // === Axis gizmo overlay (top-right corner) ===
    // Look up this viewport's camera directly via the ECS query.
    let basis = cameras
        .iter()
        .find(|(cam, child_of)| {
            child_of.parent() == doc_entity && cam.viewport_index == viewport_index
        })
        .map(|(cam, _)| cam.basis());
    if let Some(basis) = basis {
        draw_axis_gizmo(ui, resp.rect, basis, doc_entity, viewport_index, cam_msgs);
    }
}

fn draw_grid_toggle(
    ui: &mut egui::Ui,
    viewport_rect: egui::Rect,
    viewport_index: usize,
    show_grid: &mut bool,
) {
    use egui::{Align2, Color32, CornerRadius, FontId, Sense, Stroke};

    const SIZE: f32 = 34.0;
    const MARGIN: f32 = 8.0;
    const ICON_Y_OFFSET: f32 = -1.0;

    let rect = egui::Rect::from_min_size(
        viewport_rect.left_top() + egui::vec2(MARGIN, MARGIN),
        egui::Vec2::splat(SIZE),
    );
    let response = ui.interact(
        rect,
        ui.id().with(("viewport-grid-toggle", viewport_index)),
        Sense::click(),
    );
    if response.clicked() {
        *show_grid = !*show_grid;
    }

    let painter = ui.painter_at(viewport_rect);
    painter.rect_filled(
        rect,
        CornerRadius::same(5),
        if response.hovered() {
            Color32::from_rgba_unmultiplied(80, 80, 80, 180)
        } else {
            Color32::from_black_alpha(96)
        },
    );

    painter.text(
        rect.center() + egui::vec2(0.0, ICON_Y_OFFSET),
        Align2::CENTER_CENTER,
        crate::ui::icons::ICON_BORDER_ALL,
        FontId::proportional(18.0),
        if *show_grid {
            Color32::WHITE
        } else {
            Color32::from_gray(170)
        },
    );

    if *show_grid {
        painter.rect_stroke(
            rect.shrink(0.5),
            CornerRadius::same(5),
            Stroke::new(1.0_f32, Color32::from_gray(190)),
            egui::StrokeKind::Inside,
        );
    }

    response.on_hover_text(if *show_grid { "Hide grid" } else { "Show grid" });
}

/// Blender-style XYZ axis gizmo.
///
/// The 3 world axes are projected through the camera's `(right, up, forward)`
/// basis (`basis` columns):
/// - screen x = world_axis · right
/// - screen y = - world_axis · up (egui's y points down)
/// - depth   = world_axis · forward (larger = farther from camera)
///
/// Endpoints are sorted by depth so the closest ones render last (on top).
/// Positive ends draw a filled disc with a colored letter; negative ends
/// draw a hollow ring. Background axis lines run from origin to the
/// positive end only, matching Blender's gizmo.
fn draw_axis_gizmo(
    ui: &mut egui::Ui,
    viewport_rect: egui::Rect,
    basis: Mat3,
    doc_entity: Entity,
    viewport_index: usize,
    cam_msgs: &mut bevy::ecs::message::MessageWriter<CameraControlMessage>,
) {
    use egui::{Color32, CursorIcon, Pos2, Sense, Stroke};

    const RADIUS: f32 = 26.0;
    const MARGIN: f32 = 8.0;
    const DOT_RADIUS: f32 = 9.0;
    const LINE_WIDTH: f32 = 2.0;
    // 10 px padding past the furthest endpoint disc keeps the discs
    // comfortably inside the background even as anti-aliased subpixel
    // positions shift while orbiting.
    const BG_RADIUS: f32 = RADIUS + DOT_RADIUS + 10.0;

    let center = Pos2::new(
        viewport_rect.right() - MARGIN - BG_RADIUS,
        viewport_rect.top() + MARGIN + BG_RADIUS,
    );

    // Axis colors (Blender-ish), shared with the inline vector editor.
    let x_col = super::value_edit::AXIS_X_COLOR;
    let y_col = super::value_edit::AXIS_Y_COLOR;
    let z_col = super::value_edit::AXIS_Z_COLOR;

    // (world_axis, label, color, is_positive, interaction_id)
    let axes: [(Vec3, &str, Color32, bool, &str); 6] = [
        (Vec3::X, "X", x_col, true, "+X"),
        (Vec3::NEG_X, "X", x_col, false, "-X"),
        (Vec3::Y, "Y", y_col, true, "+Y"),
        (Vec3::NEG_Y, "Y", y_col, false, "-Y"),
        (Vec3::Z, "Z", z_col, true, "+Z"),
        (Vec3::NEG_Z, "Z", z_col, false, "-Z"),
    ];

    // Project to (screen_offset, depth).
    let right = basis.col(0);
    let up = basis.col(1);
    let forward = basis.col(2);
    let mut projected: Vec<(Vec3, egui::Vec2, f32, &str, Color32, bool, &str)> = axes
        .iter()
        .map(|(v, lbl, col, pos, id)| {
            let sx = v.dot(right);
            let sy = -v.dot(up);
            let depth = v.dot(forward);
            (
                *v,
                egui::vec2(sx, sy) * RADIUS,
                depth,
                *lbl,
                *col,
                *pos,
                *id,
            )
        })
        .collect();
    // Back-to-front: larger depth lies farther along the viewing direction and
    // draws first; smaller depth is closer and draws last.
    projected.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let painter = ui.painter_at(viewport_rect);
    let hovered_axis = ui.input(|input| {
        let pointer = input.pointer.hover_pos()?;
        projected
            .iter()
            .enumerate()
            .rev()
            .find_map(|(index, (_, offset, _, _, _, _, _))| {
                (pointer.distance(center + *offset) <= DOT_RADIUS * 1.2).then_some(index)
            })
    });

    // Translucent background circle. Sized to fully contain the
    // furthest endpoint disc (`RADIUS + DOT_RADIUS`) plus a small pad.
    painter.circle_filled(center, BG_RADIUS, Color32::from_black_alpha(96));

    for (index, (direction, offset, _depth, label, color, is_positive, interaction_id)) in
        projected.into_iter().enumerate()
    {
        let endpoint = center + offset;
        let pointer_over_circle = hovered_axis == Some(index);
        if pointer_over_circle {
            ui.ctx().set_cursor_icon(CursorIcon::PointingHand);
        }
        let clicked = pointer_over_circle
            && ui
                .interact(
                    egui::Rect::from_center_size(endpoint, egui::Vec2::splat(DOT_RADIUS * 2.4)),
                    ui.id().with(("axis-gizmo", viewport_index, interaction_id)),
                    Sense::click(),
                )
                .clicked();
        if clicked {
            cam_msgs.write(CameraControlMessage {
                doc: doc_entity,
                viewport_index,
                control: CameraControl::Align { direction },
            });
        }

        if is_positive {
            // Line from origin to positive end.
            painter.line_segment([center, endpoint], Stroke::new(LINE_WIDTH, color));
            painter.circle_filled(endpoint, DOT_RADIUS, color);
            // Letter label centered in the disc.
            painter.text(
                endpoint,
                egui::Align2::CENTER_CENTER,
                label,
                egui::FontId::proportional(11.0),
                Color32::BLACK,
            );
        } else {
            // Negative end: hollow ring, no line.
            painter.circle_stroke(endpoint, DOT_RADIUS - 1.0, Stroke::new(1.5_f32, color));
        }
        if pointer_over_circle {
            painter.circle_stroke(
                endpoint,
                DOT_RADIUS + 2.0,
                Stroke::new(1.5_f32, Color32::WHITE),
            );
        }
    }
}
