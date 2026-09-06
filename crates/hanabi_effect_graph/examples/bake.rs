//! Offline bake: a `.hnb` [`EffectGraphAsset`] → baked [`EffectAsset`](s) RON.
//!
//! Reads a [`EffectGraphAsset`] `.hnb` file (or the built-in demo document) and
//! emits every contained emitter's baked `bevy_hanabi` [`EffectAsset`] as RON.
//! This is the building block an [`AssetProcessor`] would call to "bake"
//! emitters in batch: deserialize the document, register the modifier types,
//! [`bake_effect()`], then serialize each result.
//!
//! ```sh
//! cargo run -p hanabi_effect_graph --example bake -- path/to/emitter.hnb
//! cargo run -p hanabi_effect_graph --example bake            # bakes the demo document
//! ```
//!
//! [`AssetProcessor`]: bevy::asset::processor::AssetProcessor
//! [`bake_effect()`]: hanabi_effect_graph::bake::bake_effect

use bevy::{asset::AssetPlugin, prelude::*};
use hanabi_effect_graph::{
    bake, demo,
    model::{EffectGraph, EffectGraphAsset},
    modifier_registry::ModifierRegistryPlugin,
};

fn main() {
    let graph: EffectGraph = match std::env::args().nth(1) {
        Some(path) => {
            let bytes = std::fs::read(&path).expect("read input file");
            let asset: EffectGraphAsset =
                ron::de::from_bytes(&bytes).expect("deserialize EffectGraphAsset");
            asset.graph
        }
        None => demo::demo_effect(),
    };

    // Register the modifier types so the bake can resolve them by type path.
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        AssetPlugin::default(),
        ModifierRegistryPlugin,
    ));
    let registry = app.world().resource::<AppTypeRegistry>().read();

    match bake::bake_effect(&graph, &registry) {
        Ok(baked) => {
            for emitter in &baked.emitters {
                let ron = emitter
                    .asset
                    .serialize(&registry)
                    .expect("serialize EffectAsset");
                println!(
                    "# emitter {:?} (parent: {:?})",
                    emitter.emitter, emitter.parent
                );
                println!("{ron}");
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
