//! Runtime load: prove the `.hnb` [`AssetLoader`] works end to end.
//!
//! Serves the built-in demo graph as a `.hnb` file from an *in-memory* asset
//! source (nothing touches disk), then loads it through the Bevy
//! [`AssetServer`] — exercising [`EffectGraphLoader`] exactly as a game would
//! when loading unbaked graphs during development — and bakes the loaded graph
//! in-process into a `bevy_hanabi` [`EffectAsset`].
//!
//! ```sh
//! cargo run -p hanabi_effect_graph --example runtime_load
//! ```
//!
//! [`AssetLoader`]: bevy::asset::AssetLoader
//! [`EffectGraphLoader`]: hanabi_effect_graph::EffectGraphLoader
//! [`EffectAsset`]: bevy_hanabi::EffectAsset

use std::path::Path;

use bevy::asset::AssetApp;
use bevy::asset::io::memory::{Dir, MemoryAssetReader};
use bevy::asset::io::{AssetSourceBuilder, AssetSourceId};
use bevy::prelude::*;
use hanabi_effect_graph::model::{EffectGraphAsset, FORMAT_VERSION};
use hanabi_effect_graph::modifier_registry::ModifierRegistryPlugin;
use hanabi_effect_graph::{EffectGraphPlugin, bake, demo, to_ron_string};

fn main() {
    // Serve a `.hnb` file from an in-memory directory so the `AssetServer` can
    // resolve it through a custom source — no filesystem involved.
    let staged = EffectGraphAsset {
        version: FORMAT_VERSION,
        graph: demo::demo_graph(),
        layout: None,
    };
    let ron = to_ron_string(&staged).expect("serialize EffectGraphAsset");
    let dir = Dir::default();
    dir.insert_asset(Path::new("effect.hnb"), ron.into_bytes());

    let mut app = App::new();
    // The custom source must be registered before `AssetPlugin` reads it.
    app.register_asset_source(
        AssetSourceId::Default,
        AssetSourceBuilder::new(move || Box::new(MemoryAssetReader { root: dir.clone() })),
    )
    .add_plugins((
        MinimalPlugins,
        AssetPlugin::default(),
        // Registers the asset type + `.hnb` loader, and the modifier types the
        // bake resolves by type path.
        EffectGraphPlugin,
        ModifierRegistryPlugin,
    ));

    let handle: Handle<EffectGraphAsset> = app.world().resource::<AssetServer>().load("effect.hnb");

    // Pump the app until the async load resolves (or give up).
    let mut frames = 0;
    loop {
        app.update();
        match app.world().resource::<AssetServer>().load_state(&handle) {
            bevy::asset::LoadState::Loaded => break,
            bevy::asset::LoadState::Failed(error) => {
                eprintln!("failed to load effect.hnb: {error}");
                std::process::exit(1);
            }
            _ => {}
        }
        frames += 1;
        if frames > 1000 {
            eprintln!("timed out waiting for effect.hnb to load");
            std::process::exit(1);
        }
    }

    let world = app.world();
    let loaded = world
        .resource::<Assets<EffectGraphAsset>>()
        .get(&handle)
        .expect("loaded EffectGraphAsset");
    println!("loaded effect.hnb via AssetServer in {frames} frame(s)");

    let registry = world.resource::<AppTypeRegistry>().read();
    match bake::bake(&loaded.graph, &registry) {
        Ok(effect) => {
            println!("baked '{}' in-process:", effect.name);
            println!("  capacity:         {}", effect.capacity());
            println!("  init modifiers:   {}", effect.init_modifiers().count());
            println!("  update modifiers: {}", effect.update_modifiers().count());
            println!("  render modifiers: {}", effect.render_modifiers().count());
        }
        Err(errors) => {
            eprintln!("bake failed:");
            for e in errors {
                eprintln!("  - {e:?}");
            }
            std::process::exit(1);
        }
    }
}
