//! Runtime load: prove the `.hnb` [`AssetLoader`] works end to end.
//!
//! Serves the built-in demo document as a `.hnb` file from an *in-memory*
//! asset source (nothing touches disk), then loads it through the Bevy
//! [`AssetServer`] — exercising [`EffectGraphLoader`] exactly as a game would
//! when loading unbaked graphs during development — and bakes the loaded
//! document in-process into its `bevy_hanabi` [`EffectAsset`]s.
//!
//! ```sh
//! cargo run -p hanabi_effect_graph --example runtime_load
//! ```
//!
//! [`AssetLoader`]: bevy::asset::AssetLoader
//! [`EffectGraphLoader`]: hanabi_effect_graph::EffectGraphLoader
//! [`EffectAsset`]: bevy_hanabi::EffectAsset

use std::path::Path;

use bevy::{
    asset::{
        AssetApp,
        io::{
            AssetSourceBuilder, AssetSourceId,
            memory::{Dir, MemoryAssetReader},
        },
    },
    prelude::*,
};
use hanabi_effect_graph::{
    EffectGraphPlugin, bake, demo,
    model::{EffectGraphAsset, FORMAT_VERSION},
    modifier_registry::ModifierRegistryPlugin,
    to_ron_string,
};

fn main() {
    // Serve a `.hnb` file from an in-memory directory so the `AssetServer` can
    // resolve it through a custom source — no filesystem involved.
    let staged = EffectGraphAsset {
        version: FORMAT_VERSION,
        graph: demo::demo_effect(),
        layout: None,
    };
    let ron = to_ron_string(&staged).expect("serialize EffectGraphAsset");
    let dir = Dir::default();
    dir.insert_asset(Path::new("emitter.hnb"), ron.into_bytes());

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

    let handle: Handle<EffectGraphAsset> =
        app.world().resource::<AssetServer>().load("emitter.hnb");

    // Pump the app until the async load resolves (or give up).
    let mut frames = 0;
    loop {
        app.update();
        match app.world().resource::<AssetServer>().load_state(&handle) {
            bevy::asset::LoadState::Loaded => break,
            bevy::asset::LoadState::Failed(error) => {
                eprintln!("failed to load emitter.hnb: {error}");
                std::process::exit(1);
            }
            _ => {}
        }
        frames += 1;
        if frames > 1000 {
            eprintln!("timed out waiting for emitter.hnb to load");
            std::process::exit(1);
        }
    }

    let world = app.world();
    let loaded = world
        .resource::<Assets<EffectGraphAsset>>()
        .get(&handle)
        .expect("loaded EffectGraphAsset");
    println!("loaded emitter.hnb via AssetServer in {frames} frame(s)");

    let registry = world.resource::<AppTypeRegistry>().read();
    match bake::bake_effect(&loaded.graph, &registry) {
        Ok(baked) => {
            for emitter in &baked.emitters {
                println!(
                    "baked '{}' in-process (parent: {:?}):",
                    emitter.asset.name, emitter.parent
                );
                println!("  capacity:         {}", emitter.asset.capacity());
                println!(
                    "  init modifiers:   {}",
                    emitter.asset.init_modifiers().count()
                );
                println!(
                    "  update modifiers: {}",
                    emitter.asset.update_modifiers().count()
                );
                println!(
                    "  render modifiers: {}",
                    emitter.asset.render_modifiers().count()
                );
            }
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
