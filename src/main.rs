
use bevy::prelude::*;

#[allow(non_snake_case, dead_code)]
mod IconsFontAwesome7;
mod app_commands;
mod document;
mod edits;
mod effect_graph;
mod history;
mod playback;
mod plugins;
mod proxy;
mod ui;

use hanabi_effect_graph::modifier_registry;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Hanabi Workshop".into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(plugins::EditorPlugins)
        .run();
}
