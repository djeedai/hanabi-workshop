//! Asset browser panel.
//!
//! Displays the global texture catalog and starts typed texture drags consumed
//! by the document's graph panel.

use bevy::{ecs::message::MessageWriter, prelude::AssetServer};
use bevy_egui::egui;

use crate::{
    app_commands::PendingFileDialogs,
    asset_library::{TextureCatalog, TextureLibraryCommand, TextureLibrarySettings},
    texture_preview::TexturePreviewCache,
    ui::asset_browser::{self, BrowserOptions},
};

/// Draw the asset browser panel.
#[allow(clippy::too_many_arguments)]
pub fn show(
    ui: &mut egui::Ui,
    catalog: &TextureCatalog,
    settings: &TextureLibrarySettings,
    previews: &mut TexturePreviewCache,
    asset_server: &AssetServer,
    commands: &mut MessageWriter<TextureLibraryCommand>,
    pending: &mut PendingFileDialogs,
) {
    let _ = asset_browser::show(
        ui,
        catalog,
        settings,
        previews,
        asset_server,
        commands,
        pending,
        BrowserOptions {
            id_salt: egui::Id::new("assets-panel"),
            manage_sources: true,
            draggable: true,
            selectable: false,
        },
    );
}
