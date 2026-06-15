//! Bevy [`AssetLoader`] and plugin for [`EffectGraphAsset`].
//!
//! Loads `.hnb` RON files into an [`EffectGraphAsset`] held by handle. The
//! asset can then be baked into a runtime
//! [`EffectAsset`](bevy_hanabi::EffectAsset) (see [`crate::bake`]) — in-process
//! during development, or offline through an
//! [`AssetProcessor`](bevy::asset::processor::AssetProcessor).

use bevy::app::{App, Plugin};
use bevy::asset::io::Reader;
use bevy::asset::{AssetApp, AssetLoader, LoadContext};
use thiserror::Error;

use crate::model::{EffectGraphAsset, FORMAT_VERSION};

/// Registers [`EffectGraphAsset`] and its [`EffectGraphLoader`].
pub struct EffectGraphPlugin;

impl Plugin for EffectGraphPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<EffectGraphAsset>()
            .init_asset_loader::<EffectGraphLoader>();
    }
}

/// Parse an [`EffectGraphAsset`] from RON bytes, rejecting future schema
/// versions. The single source of truth for the `.hnb` on-disk format, shared
/// by [`EffectGraphLoader`] and synchronous editor saves/loads.
pub fn from_ron_bytes(bytes: &[u8]) -> Result<EffectGraphAsset, EffectGraphLoaderError> {
    let asset: EffectGraphAsset = ron::de::from_bytes(bytes)?;
    if asset.version > FORMAT_VERSION {
        return Err(EffectGraphLoaderError::UnsupportedVersion {
            found: asset.version,
            supported: FORMAT_VERSION,
        });
    }
    Ok(asset)
}

/// Serialize an [`EffectGraphAsset`] to pretty RON for writing a `.hnb` file.
pub fn to_ron_string(asset: &EffectGraphAsset) -> Result<String, ron::Error> {
    ron::ser::to_string_pretty(asset, ron::ser::PrettyConfig::default())
}

/// Loads `.hnb` RON files into an [`EffectGraphAsset`].
#[derive(Default, bevy::reflect::TypePath)]
pub struct EffectGraphLoader;

/// Errors produced while loading an [`EffectGraphAsset`].
#[derive(Debug, Error)]
pub enum EffectGraphLoaderError {
    #[error("failed to read asset bytes: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to deserialize EffectGraphAsset RON: {0}")]
    Ron(#[from] ron::error::SpannedError),
    #[error("unsupported EffectGraphAsset version {found}; this build supports up to {supported}")]
    UnsupportedVersion { found: u32, supported: u32 },
}

impl AssetLoader for EffectGraphLoader {
    type Asset = EffectGraphAsset;
    type Settings = ();
    type Error = EffectGraphLoaderError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &Self::Settings,
        _load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        from_ron_bytes(&bytes)
    }

    fn extensions(&self) -> &[&str] {
        &["hnb"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{EffectGraph, EffectGraphAsset, GraphLayout};

    fn sample_asset() -> EffectGraphAsset {
        EffectGraphAsset {
            version: FORMAT_VERSION,
            graph: EffectGraph::empty(),
            layout: Some(GraphLayout {
                pan: (1.0, -2.0),
                zoom: 1.5,
                node_pos: Vec::new(),
                stack_pos: Vec::new(),
            }),
        }
    }

    #[test]
    fn ron_round_trips_through_helpers() {
        let asset = sample_asset();
        let text = to_ron_string(&asset).expect("serialize");
        let back = from_ron_bytes(text.as_bytes()).expect("deserialize");
        assert_eq!(asset, back);
    }

    #[test]
    fn rejects_future_version() {
        let mut asset = sample_asset();
        asset.version = FORMAT_VERSION + 1;
        let text = to_ron_string(&asset).expect("serialize");
        assert!(matches!(
            from_ron_bytes(text.as_bytes()),
            Err(EffectGraphLoaderError::UnsupportedVersion { .. })
        ));
    }
}
