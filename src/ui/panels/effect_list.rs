use bevy_egui::egui;

/// Lists loaded `EffectAsset`s and lets the user select one.
///
/// TODO: Accept a `ResMut<Assets<EffectAsset>>` and the currently
/// selected handle, then render a selectable list.
pub fn show(ui: &mut egui::Ui) {
    ui.heading("Effects");
    ui.separator();

    // Placeholder rows
    for name in ["(no effects loaded)"] {
        ui.selectable_label(false, name);
    }

    ui.separator();
    if ui.button("+ New Effect").clicked() {
        // TODO: create a blank EffectAsset and add it to the asset list
    }
}
