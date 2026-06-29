//! Resize per-viewport render-target images to the UI-requested panel size.
//!
//! Uses the panel size requested by the UI on the previous frame.

use bevy::{prelude::*, render::render_resource::Extent3d};

use crate::document::{DocumentViewports, ViewportSizeRequests};

const RESIZE_HYSTERESIS_PX: u32 = 4;

pub fn apply_viewport_resizes(
    requests: Res<ViewportSizeRequests>,
    viewports: Res<DocumentViewports>,
    mut images: ResMut<Assets<Image>>,
) {
    for ((doc, vp_idx), desired) in &requests.0 {
        let Some(slots) = viewports.by_doc.get(doc) else {
            continue;
        };
        let Some(handle) = slots.images.get(vp_idx) else {
            continue;
        };
        let Some(mut image) = images.get_mut(handle) else {
            continue;
        };
        let current = UVec2::new(
            image.texture_descriptor.size.width,
            image.texture_descriptor.size.height,
        );
        let target_w = desired.x.max(1);
        let target_h = desired.y.max(1);
        if current.x.abs_diff(target_w) < RESIZE_HYSTERESIS_PX
            && current.y.abs_diff(target_h) < RESIZE_HYSTERESIS_PX
        {
            continue;
        }
        image.resize(Extent3d {
            width: target_w,
            height: target_h,
            depth_or_array_layers: 1,
        });
    }
}
