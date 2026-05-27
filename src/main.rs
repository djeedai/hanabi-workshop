use bevy::prelude::*;

mod plugins;
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
