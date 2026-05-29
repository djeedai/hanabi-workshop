//! 3D-viewport orbit camera controls.
//!
//! The viewport panel ([`crate::ui::panels::viewport`]) interprets egui
//! pointer input on the displayed image and publishes
//! [`CameraControlMessage`]s. A single Update system,
//! [`apply_camera_controls`], consumes those messages: it locates the
//! matching [`ViewportCamera`] entity, updates the orbit state in place,
//! and re-derives the camera `Transform`.
//!
//! Conventions:
//! - **Left-mouse drag** on the viewport → orbit (yaw + pitch).
//! - **Right-mouse drag** → pan the target point parallel to the camera plane.
//! - **Scroll wheel** → log-zoom (multiplies/divides `distance`).
//!
//! Sensitivity constants live below and are intentionally simple; we can
//! tune (or expose in preferences) later. The system runs after
//! [`crate::ui::draw_editor_ui`] so messages from frame `N` apply on the
//! same frame they were produced.

use std::f32::consts::FRAC_PI_2;

use bevy::prelude::*;

use crate::document::ViewportCamera;

/// One unit of user input on a specific viewport's orbit camera.
#[derive(Clone, Copy, Debug, Message)]
pub struct CameraControlMessage {
    pub doc: Entity,
    pub viewport_index: usize,
    pub control: CameraControl,
}

#[derive(Clone, Copy, Debug)]
pub enum CameraControl {
    /// Orbit by `(yaw_delta, pitch_delta)`, both in radians.
    Orbit { yaw: f32, pitch: f32 },
    /// Pan target by `(dx, dy)` in *screen* fractions of the viewport
    /// (-1..=1 across the visible image). Scaled by current distance so
    /// the pinned-under-cursor feel is approximately preserved.
    Pan { dx: f32, dy: f32 },
    /// Multiplicative zoom factor: `distance *= factor`.
    Zoom { factor: f32 },
}

/// Just-below-π/2 clamp for pitch to avoid the looking-straight-up gimbal flip.
const PITCH_LIMIT: f32 = FRAC_PI_2 - 0.01;
const MIN_DISTANCE: f32 = 0.1;
const MAX_DISTANCE: f32 = 1.0e4;

/// Public sensitivities. Callers (the viewport panel) scale the raw
/// pixel deltas into these "natural" units before sending the message.
pub const ORBIT_RAD_PER_PIXEL: f32 = 0.005;
pub const PAN_FRACTION_PER_PIXEL: f32 = 0.0015;
/// `factor = ZOOM_PER_NOTCH ^ wheel_delta_units`. Below 1 zooms in.
pub const ZOOM_PER_NOTCH: f32 = 0.9;

pub struct CameraControlPlugin;

impl Plugin for CameraControlPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<CameraControlMessage>()
            .add_systems(Update, apply_camera_controls);
    }
}

fn apply_camera_controls(
    mut messages: MessageReader<CameraControlMessage>,
    mut cameras: Query<(&mut ViewportCamera, &mut Transform, &ChildOf)>,
) {
    // Group messages by (doc, viewport_index) so we don't recompute the
    // transform per message when many small deltas come in on one frame.
    use std::collections::HashMap;
    #[derive(Default, Clone, Copy)]
    struct Accum {
        yaw: f32,
        pitch: f32,
        pan_x: f32,
        pan_y: f32,
        zoom_factor: f32,
    }
    let mut acc: HashMap<(Entity, usize), Accum> = HashMap::new();
    for msg in messages.read() {
        let entry = acc.entry((msg.doc, msg.viewport_index)).or_insert(Accum {
            zoom_factor: 1.0,
            ..Default::default()
        });
        match msg.control {
            CameraControl::Orbit { yaw, pitch } => {
                entry.yaw += yaw;
                entry.pitch += pitch;
            }
            CameraControl::Pan { dx, dy } => {
                entry.pan_x += dx;
                entry.pan_y += dy;
            }
            CameraControl::Zoom { factor } => {
                entry.zoom_factor *= factor;
            }
        }
    }

    if acc.is_empty() {
        return;
    }

    for (mut cam, mut tf, child_of) in &mut cameras {
        let key = (child_of.parent(), cam.viewport_index);
        let Some(a) = acc.get(&key) else { continue };

        cam.yaw -= a.yaw;
        cam.pitch = (cam.pitch + a.pitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        cam.distance = (cam.distance * a.zoom_factor).clamp(MIN_DISTANCE, MAX_DISTANCE);

        if a.pan_x != 0.0 || a.pan_y != 0.0 {
            // Build a screen-space basis from the current orientation.
            let forward = (cam.target - cam.eye()).normalize_or_zero();
            let right = forward.cross(Vec3::Y).normalize_or_zero();
            let up = right.cross(forward).normalize_or_zero();
            // Pan amount in world units scales with current distance so
            // the cursor-tracked point feels stable across zoom levels.
            let scale = cam.distance;
            cam.target += (-right * a.pan_x + up * a.pan_y) * scale;
        }

        let eye = cam.eye();
        *tf = Transform::from_translation(eye).looking_at(cam.target, Vec3::Y);
    }
}
