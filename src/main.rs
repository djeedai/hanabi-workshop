use bevy::{asset::UnapprovedPathMode, prelude::*};

#[allow(non_snake_case, dead_code)]
mod IconsFontAwesome7;
mod app_commands;
mod confirm;
mod document;
mod edits;
mod effect_graph;
mod effect_library;
mod history;
mod playback;
mod plugins;
mod proxy;
mod thumbnail;
mod ui;

use hanabi_effect_graph::modifier_registry;

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Hanabi Workshop".into(),
                        ..default()
                    }),
                    // The unsaved-changes guard (`crate::confirm`) intercepts
                    // the OS window-close request, so Bevy must not close the
                    // window itself.
                    close_when_requested: false,
                    ..default()
                })
                // Texture bindings are absolute paths the artist picks from
                // anywhere on disk, outside the `assets/` folder. Deny (rather
                // than the default Forbid) so those specific loads can opt in
                // via `AssetServer::load_override`, while every other
                // unapproved path stays rejected.
                .set(AssetPlugin {
                    unapproved_path_mode: UnapprovedPathMode::Deny,
                    ..default()
                }),
        )
        .add_plugins(plugins::EditorPlugins)
        .run();
}
