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

/// Render the viewport panel: image, camera input, and axis gizmo.
///
/// Displays the render-target image, records the panel's pixel size, interprets
/// pointer input for orbit-camera control, and overlays a Blender-style axis
/// gizmo in the top-right corner.
pub fn show(
    ui: &mut egui::Ui,
    doc_entity: Entity,
    viewport_index: usize,
    viewport_textures: &HashMap<(Entity, usize), egui::TextureId>,
    size_requests: &mut ViewportSizeRequests,
    cam_msgs: &mut bevy::ecs::message::MessageWriter<CameraControlMessage>,
    cameras: &Query<(&'static crate::document::ViewportCamera, &'static ChildOf)>,
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

    // === Axis gizmo overlay (top-right corner) ===
    // Look up this viewport's camera directly via the ECS query.
    let basis = cameras
        .iter()
        .find(|(cam, child_of)| {
            child_of.parent() == doc_entity && cam.viewport_index == viewport_index
        })
        .map(|(cam, _)| {
            let eye = cam.eye();
            let forward = (cam.target - eye).normalize_or_zero();
            let right = forward.cross(Vec3::Y).normalize_or_zero();
            let up = right.cross(forward).normalize_or_zero();
            Mat3::from_cols(right, up, forward)
        });
    if let Some(basis) = basis {
        draw_axis_gizmo(ui, resp.rect, basis);
    }
}

/// Blender-style XYZ axis gizmo.
///
/// The 3 world axes are projected through the camera's `(right, up, forward)`
/// basis (`basis` columns):
/// - screen x = world_axis · right
/// - screen y = - world_axis · up (egui's y points down)
/// - depth   = world_axis · forward (positive = in front of camera)
///
/// Endpoints are sorted by depth so the closest ones render last (on top).
/// Positive ends draw a filled disc with a colored letter; negative ends
/// draw a hollow ring. Background axis lines run from origin to the
/// positive end only, matching Blender's gizmo.
fn draw_axis_gizmo(ui: &mut egui::Ui, viewport_rect: egui::Rect, basis: Mat3) {
    use egui::{Color32, Pos2, Stroke};

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

    // Axis colors (Blender-ish).
    let x_col = Color32::from_rgb(232, 91, 91);
    let y_col = Color32::from_rgb(126, 207, 80);
    let z_col = Color32::from_rgb(70, 132, 232);

    // (world_axis, label, color, is_positive)
    let axes: [(Vec3, &str, Color32, bool); 6] = [
        (Vec3::X, "X", x_col, true),
        (Vec3::NEG_X, "X", x_col, false),
        (Vec3::Y, "Y", y_col, true),
        (Vec3::NEG_Y, "Y", y_col, false),
        (Vec3::Z, "Z", z_col, true),
        (Vec3::NEG_Z, "Z", z_col, false),
    ];

    // Project to (screen_offset, depth).
    let right = basis.col(0);
    let up = basis.col(1);
    let forward = basis.col(2);
    let mut projected: Vec<(egui::Vec2, f32, &str, Color32, bool)> = axes
        .iter()
        .map(|(v, lbl, col, pos)| {
            let sx = v.dot(right);
            let sy = -v.dot(up);
            let depth = v.dot(forward);
            (egui::vec2(sx, sy) * RADIUS, depth, *lbl, *col, *pos)
        })
        .collect();
    // Back-to-front: smaller depth (further behind camera = more negative
    // dot with forward) draws first, larger depth draws last.
    projected.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let painter = ui.painter_at(viewport_rect);

    // Translucent background circle. Sized to fully contain the
    // furthest endpoint disc (`RADIUS + DOT_RADIUS`) plus a small pad.
    painter.circle_filled(center, BG_RADIUS, Color32::from_black_alpha(96));

    for (offset, _depth, label, color, is_positive) in projected {
        let endpoint = center + offset;
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
            painter.circle_stroke(endpoint, DOT_RADIUS - 1.0, Stroke::new(1.5, color));
        }
    }
}
