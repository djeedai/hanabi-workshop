//! Bakes `.hnb` [`EffectGraphAsset`] files into runtime [`EffectAsset`]s.
//!
//! With Bevy's [`AssetProcessor`] (a game running under
//! `AssetMode::Processed`), a `.hnb` graph is loaded, transformed by baking,
//! and saved as an `EffectAsset` RON. The deployed game then loads the baked
//! output through [`EffectAssetLoader`] without ever seeing the source graph or
//! needing this crate's baking code. The same baking step is available
//! in-process for development via [`crate::loader::EffectGraphPlugin`].
//!
//! The pipeline is the idiomatic [`LoadTransformAndSave`] composed of:
//! - [`EffectGraphLoader`] — reads `.hnb` into a [`EffectGraphAsset`],
//! - [`EffectGraphBaker`] — bakes the graph into an [`EffectAsset`],
//! - [`EffectAssetSaver`] — serializes the `EffectAsset` to RON for
//!   [`EffectAssetLoader`].
//!
//! [`EffectAsset`]: bevy_hanabi::EffectAsset
//! [`AssetProcessor`]: bevy::asset::processor::AssetProcessor

use bevy::{
    app::{App, Plugin},
    asset::{
        AssetApp,
        io::{AsyncWriteExt, Writer},
        processor::LoadTransformAndSave,
        saver::{AssetSaver, SavedAsset},
        transformer::{AssetTransformer, TransformedAsset},
    },
    ecs::reflect::AppTypeRegistry,
    reflect::{TypePath, TypeRegistryArc},
};
use bevy_hanabi::{EffectAsset, EffectAssetLoader};
use thiserror::Error;

use crate::{
    bake::{self, BakeError},
    loader::EffectGraphLoader,
    model::EffectGraphAsset,
    modifier_registry::ModifierRegistryPlugin,
};

/// Full `.hnb` → baked `EffectAsset` processor pipeline.
///
/// Register it with [`EffectGraphProcessorPlugin`], or build one directly with
/// [`new`] and pass it to [`App::register_asset_processor`].
///
/// [`new`]: EffectGraphProcessor::new
/// [`App::register_asset_processor`]: bevy::asset::AssetApp::register_asset_processor
pub type EffectGraphProcessor =
    LoadTransformAndSave<EffectGraphLoader, EffectGraphBaker, EffectAssetSaver>;

/// Bakes a loaded [`EffectGraphAsset`] into a runtime [`EffectAsset`].
///
/// Holds a shared handle to the app's type registry so the bake can resolve
/// modifier types by reflection. The registry must contain the modifier types
/// (see [`ModifierRegistryPlugin`]); the shared handle picks up that
/// registration regardless of plugin order.
#[derive(TypePath)]
pub struct EffectGraphBaker {
    type_registry: TypeRegistryArc,
}

/// Error raised when a graph cannot be baked during processing.
#[derive(Debug, Error)]
#[error("effect graph failed to bake: {0:?}")]
pub struct BakeTransformError(pub Vec<BakeError>);

impl AssetTransformer for EffectGraphBaker {
    type AssetInput = EffectGraphAsset;
    type AssetOutput = EffectAsset;
    type Settings = ();
    type Error = BakeTransformError;

    async fn transform<'a>(
        &'a self,
        asset: TransformedAsset<Self::AssetInput>,
        _settings: &'a Self::Settings,
    ) -> Result<TransformedAsset<Self::AssetOutput>, Self::Error> {
        let registry = self.type_registry.read();
        let emitter = bake::bake(&asset.get().graph, &registry).map_err(BakeTransformError)?;
        Ok(asset.replace_asset(emitter))
    }
}

/// Serializes a baked [`EffectAsset`] for [`EffectAssetLoader`].
///
/// Holds the type registry handle used to serialize modifier types.
#[derive(TypePath)]
pub struct EffectAssetSaver {
    type_registry: TypeRegistryArc,
}

/// Error raised when a baked [`EffectAsset`] cannot be written.
#[derive(Debug, Error)]
pub enum EffectAssetSaveError {
    /// The asset could not be serialized to RON.
    #[error("failed to serialize EffectAsset to RON: {0}")]
    Serialize(#[from] ron::Error),
    /// Writing the serialized bytes failed.
    #[error("failed to write EffectAsset bytes: {0}")]
    Io(#[from] std::io::Error),
}

impl AssetSaver for EffectAssetSaver {
    type Asset = EffectAsset;
    type Settings = ();
    type OutputLoader = EffectAssetLoader;
    type Error = EffectAssetSaveError;

    async fn save(
        &self,
        writer: &mut Writer,
        asset: SavedAsset<'_, '_, Self::Asset>,
        _settings: &Self::Settings,
        _path: bevy::asset::AssetPath<'_>,
    ) -> Result<(), Self::Error> {
        let ron = {
            let registry = self.type_registry.read();
            asset.get().serialize(&registry)?
        };
        writer.write_all(ron.as_bytes()).await?;
        Ok(())
    }
}

/// Registers the `.hnb` → baked `EffectAsset` processor pipeline.
///
/// Pulls in [`ModifierRegistryPlugin`] (so the bake can resolve modifier types)
/// and [`EffectGraphLoader`] for the graph asset. Add this to a tool or game
/// running under `AssetMode::Processed`; processed `.hnb` files are then served
/// as baked `EffectAsset`s loaded by [`EffectAssetLoader`].
pub struct EffectGraphProcessorPlugin;

impl Plugin for EffectGraphProcessorPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<ModifierRegistryPlugin>() {
            app.add_plugins(ModifierRegistryPlugin);
        }
        app.init_asset::<EffectGraphAsset>()
            .init_asset_loader::<EffectGraphLoader>();

        let type_registry = app.world().resource::<AppTypeRegistry>().0.clone();
        let processor = EffectGraphProcessor::new(
            EffectGraphBaker {
                type_registry: type_registry.clone(),
            },
            EffectAssetSaver { type_registry },
        );
        app.register_asset_processor::<EffectGraphProcessor>(processor)
            .set_default_asset_processor::<EffectGraphProcessor>("hnb");
    }
}

#[cfg(test)]
mod tests {
    use bevy::{
        asset::{ErasedLoadedAsset, LoadedAsset, saver::SavedAsset, transformer::TransformedAsset},
        ecs::reflect::AppTypeRegistry,
        tasks::block_on,
    };
    use bevy_hanabi::EffectAsset;

    use super::*;
    use crate::{
        bake, demo,
        model::{EffectGraphAsset, FORMAT_VERSION, SourceContext, SourceKind, SourceLink},
    };

    fn test_registry() -> AppTypeRegistry {
        let registry = AppTypeRegistry::default();
        bevy_hanabi::register_modifiers(&registry);
        registry
    }

    /// A minimal single-emitter, CPU-connected [`EffectGraphAsset`] wrapping
    /// [`demo::build_demo_emitter`] — the shape the strict single-emitter
    /// [`bake::bake`] accepts, as opposed to [`demo::demo_effect`]'s
    /// multi-emitter document, which it must reject.
    fn demo_graph_asset() -> EffectGraphAsset {
        let mut effect_graph = crate::model::EffectGraph::empty();
        let emitter_id = effect_graph.alloc_emitter_id();
        // Thread the actual `EffectGraph` allocator through `build_demo_emitter` (as
        // `demo_effect` does) so every id it mints advances `effect_graph.next_id`,
        // guaranteeing `alloc_source_id` below can never collide with it.
        let emitter = demo::build_demo_emitter(emitter_id, &mut effect_graph.next_id);
        let source_id = effect_graph.alloc_source_id();
        effect_graph.sources.push(SourceContext {
            id: source_id,
            kind: SourceKind::CpuSpawner {
                settings: bevy_hanabi::SpawnerSettings::rate(120.0.into()),
            },
        });
        effect_graph.source_links.push(SourceLink {
            source: source_id,
            emitter: emitter_id,
        });
        effect_graph.emitters.push(emitter);

        EffectGraphAsset {
            version: FORMAT_VERSION,
            graph: effect_graph,
            layout: None,
        }
    }

    /// Drive the real transformer + saver and verify the saved bytes reload.
    ///
    /// Confirms the saved bytes load back through the same path
    /// [`EffectAssetLoader`] uses, baking to the same shape as a direct
    /// in-process bake.
    #[test]
    fn processor_bakes_and_saves_loadable_effect_asset() {
        let app_registry = test_registry();
        let registry_arc = app_registry.0.clone();
        let transformer = EffectGraphBaker {
            type_registry: registry_arc.clone(),
        };
        let saver = EffectAssetSaver {
            type_registry: registry_arc,
        };

        let erased: ErasedLoadedAsset = LoadedAsset::from(demo_graph_asset()).into();
        let input = TransformedAsset::<EffectGraphAsset>::from_loaded(erased).unwrap();

        let output = block_on(transformer.transform(input, &())).expect("transform");

        let mut bytes: Vec<u8> = Vec::new();
        let path = bevy::asset::AssetPath::from("test.emitter.ron");
        block_on(saver.save(&mut bytes, SavedAsset::from_transformed(&output), &(), path))
            .expect("save");

        let ron = String::from_utf8(bytes).expect("utf8");
        let registry = app_registry.read();
        let loaded = EffectAsset::deserialize(&ron, &registry).expect("loader deserialize");

        let expected = bake::bake(&demo_graph_asset().graph, &registry).expect("direct bake");
        assert_eq!(loaded.name, expected.name);
        assert_eq!(loaded.capacity(), expected.capacity());
        assert_eq!(
            loaded.init_modifiers().count(),
            expected.init_modifiers().count()
        );
        assert_eq!(
            loaded.update_modifiers().count(),
            expected.update_modifiers().count()
        );
        assert_eq!(
            loaded.render_modifiers().count(),
            expected.render_modifiers().count()
        );
    }

    /// A multi-emitter document (e.g. [`demo::demo_effect`]'s
    /// CPU-root/GPU-child pair) is rejected by the transformer, directing
    /// the caller to [`bake::bake_effect`] instead of attempting a partial
    /// or best-guess bake.
    #[test]
    fn processor_rejects_multi_emitter_effect() {
        let app_registry = test_registry();
        let transformer = EffectGraphBaker {
            type_registry: app_registry.0.clone(),
        };

        let asset = EffectGraphAsset {
            version: FORMAT_VERSION,
            graph: demo::demo_effect(),
            layout: None,
        };
        let erased: ErasedLoadedAsset = LoadedAsset::from(asset).into();
        let input = TransformedAsset::<EffectGraphAsset>::from_loaded(erased).unwrap();

        let error = match block_on(transformer.transform(input, &())) {
            Err(e) => e,
            Ok(_) => panic!("multi-emitter document must be rejected"),
        };
        assert!(
            error.0.iter().any(|e| e.message.contains("bake_effect")),
            "rejection message should direct the caller to bake_effect: {:?}",
            error.0
        );
    }

    /// `demo_graph_asset`'s single emitter must be built entirely from the
    /// wrapping `EffectGraph`'s own allocator, with no id collision between the
    /// emitter's internal ids (properties/nodes/stacks) and its source id.
    ///
    /// Regression test for the removed `effect_graph.next_id = 1000` hack: that
    /// hack was only needed because `demo::demo_emitter()` minted ids from
    /// a counter disjoint from the wrapping `EffectGraph`;
    /// `demo_graph_asset` now threads `demo::build_demo_emitter` through
    /// the real `effect_graph.next_id` counter instead, so no manual jump
    /// is needed and no collision is possible.
    #[test]
    fn demo_graph_asset_has_no_id_collisions() {
        let asset = demo_graph_asset();
        let effect_graph = asset.graph;
        let emitter = &effect_graph.emitters[0];

        let mut ids: Vec<u32> = vec![emitter.id.get()];
        ids.extend(emitter.properties.iter().map(|p| p.id.get()));
        ids.extend(emitter.texture_slots.iter().map(|s| s.id.get()));
        ids.extend(emitter.nodes.iter().map(|n| n.id.get()));
        ids.extend(emitter.stacks.iter().map(|s| s.id.get()));
        ids.extend(effect_graph.sources.iter().map(|s| s.id.get()));

        let unique: std::collections::HashSet<u32> = ids.iter().copied().collect();
        assert_eq!(
            unique.len(),
            ids.len(),
            "duplicate id found in demo_graph_asset: {ids:?}"
        );
        let max_id = ids
            .into_iter()
            .max()
            .expect("demo_graph_asset mints some ids");
        assert!(
            effect_graph.next_id > max_id,
            "next_id ({}) must be strictly greater than every id in use (max {max_id})",
            effect_graph.next_id
        );
    }
}
