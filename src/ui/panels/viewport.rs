use std::collections::HashMap;

use bevy::prelude::*;
use bevy_egui::egui;

use crate::document::ViewportSizeRequests;

/// Display the viewport's render-target image, and record the panel's pixel
/// size so the resize-to-fit system can match the image extent.
pub fn show(
    ui: &mut egui::Ui,
    doc_entity: Entity,
    viewport_index: usize,
    viewport_textures: &HashMap<(Entity, usize), egui::TextureId>,
    size_requests: &mut ViewportSizeRequests,
) {
    let Some(tex) = viewport_textures.get(&(doc_entity, viewport_index)).copied() else {
        ui.centered_and_justified(|ui| {
            ui.label("(waiting for render target)");
        });
        return;
    };
    let size = ui.available_size();
    ui.image(egui::load::SizedTexture::new(tex, size));

    let pixels_per_point = ui.ctx().pixels_per_point();
    let px = UVec2::new(
        (size.x * pixels_per_point).max(1.0) as u32,
        (size.y * pixels_per_point).max(1.0) as u32,
    );
    size_requests.0.insert((doc_entity, viewport_index), px);
}
