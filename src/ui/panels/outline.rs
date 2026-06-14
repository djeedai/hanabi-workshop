//! Effect panel.
//!
//! Top-level view of the current effect's particle layout (the attribute
//! strip). Modifier editing lives in the Graph panel — each modifier is a
//! stacked node there, with a per-node close button to remove it and a
//! warning badge when its writes are shadowed by a later modifier.

use bevy::prelude::*;
use bevy_hanabi::EffectAsset;

pub fn show(ui: &mut egui::Ui, effects: &Assets<EffectAsset>, effect_handle: &Handle<EffectAsset>) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        if let Some(asset) = effects.get(effect_handle) {
            super::shaders::layout_section(ui, asset);
        } else {
            ui.weak("(effect not loaded)");
        }
    });
}
