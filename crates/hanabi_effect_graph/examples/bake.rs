//! Offline bake: a `.hnb` [`EffectGraphAsset`] → baked [`EffectAsset`] RON.
//!
//! Reads an [`EffectGraphAsset`] `.hnb` file (or the built-in demo graph) and
//! emits the baked `bevy_hanabi` [`EffectAsset`] as RON. This is the building
//! block an [`AssetProcessor`] would call to "bake" effects in batch:
//! deserialize the graph, register the modifier types, [`bake()`], then
//! serialize the result.
//!
//! ```sh
//! cargo run -p hanabi_effect_graph --example bake -- path/to/effect.hnb
//! cargo run -p hanabi_effect_graph --example bake            # bakes the demo graph
//! ```
//!
//! [`AssetProcessor`]: bevy::asset::processor::AssetProcessor
//! [`bake()`]: hanabi_effect_graph::bake::bake

use bevy::asset::AssetPlugin;
use bevy::prelude::*;
use hanabi_effect_graph::model::{EffectGraph, EffectGraphAsset};
use hanabi_effect_graph::modifier_registry::ModifierRegistryPlugin;
use hanabi_effect_graph::{bake, demo};

fn main() {
    let graph: EffectGraph = match std::env::args().nth(1) {
        Some(path) => {
            let bytes = std::fs::read(&path).expect("read input file");
            let asset: EffectGraphAsset =
                ron::de::from_bytes(&bytes).expect("deserialize EffectGraphAsset");
            asset.graph
        }
        None => demo::demo_graph(),
    };

    // Register the modifier types so the bake can resolve them by type path.
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        AssetPlugin::default(),
        ModifierRegistryPlugin,
    ));
    let registry = app.world().resource::<AppTypeRegistry>().read();

    match bake::bake(&graph, &registry) {
        Ok(effect) => {
            let ron = effect.serialize(&registry).expect("serialize EffectAsset");
            println!("{ron}");
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
