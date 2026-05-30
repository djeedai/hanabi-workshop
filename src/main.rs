use bevy::prelude::*;

mod app_commands;
mod demo_effect;
mod document;
mod edits;
mod history;
#[allow(non_snake_case, dead_code)]
mod IconsFontAwesome7;
mod modifier_ops;
mod modifier_registry;
mod playback;
mod plugins;
mod proxy;
mod ui;

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
