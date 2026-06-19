//! Asset processor that bakes `.hnb` [`EffectGraphAsset`] files into runtime
//! [`EffectAsset`]s at processing time.
//!
//! With Bevy's [`AssetProcessor`] (a
//! game running under `AssetMode::Processed`), a `.hnb` graph is loaded,
//! transformed by baking, and saved as an `EffectAsset` RON. The deployed game
//! then loads the baked output through [`EffectAssetLoader`] without ever seeing
//! the source graph or needing this crate's baking code. The same baking step
//! is available in-process for development via [`crate::loader::EffectGraphPlugin`].
//!
//! The pipeline is the idiomatic
//! [`LoadTransformAndSave`] composed of:
//! - [`EffectGraphLoader`] — reads `.hnb` into an [`EffectGraphAsset`],
//! - [`EffectGraphBaker`] — bakes the graph into an [`EffectAsset`],
//! - [`EffectAssetSaver`] — serializes the `EffectAsset` to RON for
//!   [`EffectAssetLoader`].
//!
//! [`EffectAsset`]: bevy_hanabi::EffectAsset
//! [`AssetProcessor`]: bevy::asset::processor::AssetProcessor

use bevy::app::{App, Plugin};
use bevy::asset::io::{AsyncWriteExt, Writer};
use bevy::asset::processor::LoadTransformAndSave;
use bevy::asset::saver::{AssetSaver, SavedAsset};
use bevy::asset::transformer::{AssetTransformer, TransformedAsset};
use bevy::asset::AssetApp;
use bevy::ecs::reflect::AppTypeRegistry;
use bevy::reflect::{TypePath, TypeRegistryArc};
use bevy_hanabi::{EffectAsset, EffectAssetLoader};
use thiserror::Error;

use crate::bake::{self, BakeError};
use crate::loader::EffectGraphLoader;
use crate::model::EffectGraphAsset;
use crate::modifier_registry::ModifierRegistryPlugin;

/// Full `.hnb` → baked `EffectAsset` processor pipeline.
///
/// Register it with [`EffectGraphProcessorPlugin`], or build one directly with
/// [`new`] and pass it to
/// [`App::register_asset_processor`].
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
        let effect = bake::bake(&asset.get().graph, &registry).map_err(BakeTransformError)?;
        Ok(asset.replace_asset(effect))
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
        asset: SavedAsset<'_, Self::Asset>,
        _settings: &Self::Settings,
    ) -> Result<(), Self::Error> {
        let ron = {
            let registry = self.type_registry.read();
            asset.get().serialize(&registry)?
        };
        writer.write_all(ron.as_bytes()).await?;
        Ok(())
    }
}

/// Registers the `.hnb` → baked `EffectAsset` [`EffectGraphProcessor`] and sets
/// it as the default processor for the `hnb` extension.
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
    use super::*;
    use bevy::asset::saver::SavedAsset;
    use bevy::asset::transformer::TransformedAsset;
    use bevy::asset::{ErasedLoadedAsset, LoadedAsset};
    use bevy::ecs::reflect::AppTypeRegistry;
    use bevy::tasks::block_on;
    use bevy_hanabi::EffectAsset;

    use crate::model::{EffectGraphAsset, FORMAT_VERSION};
    use crate::{bake, demo};

    fn test_registry() -> AppTypeRegistry {
        let registry = AppTypeRegistry::default();
        bevy_hanabi::register_modifiers(&registry);
        registry
    }

    fn demo_graph_asset() -> EffectGraphAsset {
        EffectGraphAsset {
            version: FORMAT_VERSION,
            graph: demo::demo_graph(),
            layout: None,
        }
    }

    /// Drive the real transformer + saver and confirm the saved bytes load back
    /// through the same path [`EffectAssetLoader`] uses, baking to the same
    /// shape as a direct in-process bake.
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
        block_on(saver.save(&mut bytes, SavedAsset::from_transformed(&output), &())).expect("save");

        let ron = String::from_utf8(bytes).expect("utf8");
        let registry = app_registry.read();
        let loaded = EffectAsset::deserialize(&ron, &registry).expect("loader deserialize");

        let expected = bake::bake(&demo::demo_graph(), &registry).expect("direct bake");
        assert_eq!(loaded.name, expected.name);
        assert_eq!(loaded.capacity(), expected.capacity());
        assert_eq!(loaded.init_modifiers().count(), expected.init_modifiers().count());
        assert_eq!(
            loaded.update_modifiers().count(),
            expected.update_modifiers().count()
        );
        assert_eq!(
            loaded.render_modifiers().count(),
            expected.render_modifiers().count()
        );
    }
}
