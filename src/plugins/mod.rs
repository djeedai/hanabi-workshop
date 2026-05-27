use bevy::app::{PluginGroup, PluginGroupBuilder};
use bevy_egui::EguiPlugin;

pub mod editor;
pub use editor::{EditorPlugin, ViewportImages};

/// Bundles all editor plugins together.
pub struct EditorPlugins;

impl PluginGroup for EditorPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(EguiPlugin::default())
            .add(EditorPlugin)
    }
}
