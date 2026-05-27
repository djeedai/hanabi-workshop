use bevy_egui::egui;

/// Displays and edits properties of the currently selected `EffectAsset`.
///
/// TODO: Accept the selected `EffectAsset` (or a mutable borrow of it)
/// and render its spawner settings, modifiers, and attribute bindings.
pub fn show(ui: &mut egui::Ui) {
    ui.heading("Properties");
    ui.separator();
    ui.label("Select an effect to inspect its properties.");
}
