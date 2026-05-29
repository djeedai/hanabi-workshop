use bevy::app::{PluginGroup, PluginGroupBuilder};
use bevy_egui::EguiPlugin;
use bevy_hanabi::HanabiPlugin;

pub mod camera_control;
pub mod editor;
pub mod reconcile;
pub mod viewport_resize;

pub use editor::EditorPlugin;

/// Bundles all editor plugins together.
pub struct EditorPlugins;

impl PluginGroup for EditorPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(HanabiPlugin)
            .add(EguiPlugin::default())
            .add(EditorPlugin)
    }
}
