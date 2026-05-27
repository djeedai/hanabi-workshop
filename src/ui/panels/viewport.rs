use bevy_egui::egui;

/// Displays the given egui texture (a Bevy render-to-image target) filling the panel.
pub fn show(ui: &mut egui::Ui, tex: egui::TextureId) {
    let size = ui.available_size();
    ui.image(egui::load::SizedTexture::new(tex, size));
}
