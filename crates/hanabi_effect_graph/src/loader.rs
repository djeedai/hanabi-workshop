//! Bevy [`AssetLoader`] and plugin for [`EffectGraphAsset`].
//!
//! Loads `.hnb` RON files into an [`EffectGraphAsset`] held by handle. The
//! asset can then be baked into a runtime [`EffectAsset`] (see [`crate::bake`])
//! — in-process during development, or offline through an [`AssetProcessor`].
//!
//! # On-disk format
//!
//! A `.hnb` file is a RON-serialized [`EffectGraphAsset`] whose first field is
//! a [`FORMAT_VERSION`] stamp, prefixed by the [`MAGIC_HEADER`] comment line
//! for content detection. [`from_ron_bytes`] and [`to_ron_string`] are the
//! single read/write funnel, shared by the [`EffectGraphLoader`] and the
//! editor's synchronous saves/loads.
//!
//! # Evolving the schema
//!
//! - **Additive, backward-compatible change** (a new optional field, a widened
//!   default): give the field `#[serde(default)]` (or `#[serde(alias = "...")]`
//!   for a rename) and *do not* bump [`FORMAT_VERSION`]. Old files parse
//!   unchanged.
//! - **Breaking change** (a removed/retyped field, changed semantics): bump
//!   [`FORMAT_VERSION`], add one guarded step to `apply_migrations` upgrading
//!   the previous version's untyped tree, and freeze a fixture of the old
//!   format so the migration stays covered.
//!
//! The writer always stamps the current [`FORMAT_VERSION`]; the reader rejects
//! anything newer and walks the migration ladder for anything older.
//!
//! [`EffectAsset`]: bevy_hanabi::EffectAsset
//! [`AssetProcessor`]: bevy::asset::processor::AssetProcessor
//! [`FORMAT_VERSION`]: crate::model::FORMAT_VERSION

use bevy::{
    app::{App, Plugin},
    asset::{AssetApp, AssetLoader, LoadContext, io::Reader},
};
use serde::Deserialize;
use thiserror::Error;

use crate::model::{EffectGraphAsset, FORMAT_VERSION};

/// Leading marker line stamped on every `.hnb` file for content detection.
///
/// A RON line comment, so deserialization ignores it and files written by
/// older builds without it still load. It lets `file`-style tools and humans
/// recognize a `.hnb` graph by its first line without parsing the whole
/// document.
pub const MAGIC_HEADER: &str =
    "// Hanabi effect graph - https://github.com/djeedai/hanabi-workshop";

/// Registers [`EffectGraphAsset`] and its [`EffectGraphLoader`].
pub struct EffectGraphPlugin;

impl Plugin for EffectGraphPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<EffectGraphAsset>()
            .init_asset_loader::<EffectGraphLoader>();
    }
}

/// Parse an [`EffectGraphAsset`] from RON bytes.
///
/// Peeks the schema version, rejects versions newer than [`FORMAT_VERSION`],
/// and upgrades older ones through the migration ladder before deserializing.
/// The single source of truth for the `.hnb` on-disk format, shared by
/// [`EffectGraphLoader`] and synchronous editor saves/loads.
pub fn from_ron_bytes(bytes: &[u8]) -> Result<EffectGraphAsset, EffectGraphLoaderError> {
    // Read only the version first; RON ignores the leading magic-header comment
    // and any fields beyond `version`.
    let AssetVersion { version } = ron::de::from_bytes(bytes)?;
    if version > FORMAT_VERSION {
        return Err(EffectGraphLoaderError::UnsupportedVersion {
            found: version,
            supported: FORMAT_VERSION,
        });
    }
    if version == FORMAT_VERSION {
        return Ok(ron::de::from_bytes(bytes)?);
    }
    migrate_from(bytes, version)
}

/// Serialize an [`EffectGraphAsset`] to pretty RON for writing a `.hnb` file.
///
/// Prepends [`MAGIC_HEADER`] so every written file is content-detectable.
pub fn to_ron_string(asset: &EffectGraphAsset) -> Result<String, ron::Error> {
    let body = ron::ser::to_string_pretty(asset, ron::ser::PrettyConfig::default())?;
    Ok(format!("{MAGIC_HEADER}\n{body}"))
}

/// The subset of an [`EffectGraphAsset`] needed to branch on schema version.
#[derive(Deserialize)]
struct AssetVersion {
    version: u32,
}

/// Deserialize and upgrade an asset saved under an older `version`.
///
/// Parses the document into an untyped RON tree, walks the migration ladder up
/// to [`FORMAT_VERSION`], then deserializes the upgraded tree. Reached only for
/// `version < FORMAT_VERSION`; the ladder is empty until the first breaking
/// schema change, so no such files exist yet.
fn migrate_from(bytes: &[u8], version: u32) -> Result<EffectGraphAsset, EffectGraphLoaderError> {
    let value: ron::Value = ron::de::from_bytes(bytes)?;
    let value = apply_migrations(value, version);
    value
        .into_rust()
        .map_err(|source| EffectGraphLoaderError::Migration {
            from: version,
            source,
        })
}

/// Apply ordered `v→v+1` transforms to an untyped asset tree.
///
/// Each breaking [`FORMAT_VERSION`] bump appends one guarded upgrade step, e.g.
/// `if from < 2 { value = migrate_v1_to_v2(value); }`. No migrations are
/// defined yet.
fn apply_migrations(value: ron::Value, from: u32) -> ron::Value {
    let _ = from;
    value
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
    #[error("failed to migrate EffectGraphAsset from version {from}: {source}")]
    Migration { from: u32, source: ron::Error },
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
    fn writes_magic_header() {
        let text = to_ron_string(&sample_asset()).expect("serialize");
        assert!(text.starts_with(MAGIC_HEADER));
    }

    #[test]
    fn loads_legacy_file_without_header() {
        // A file written before the header existed: plain RON, no leading
        // comment. It must still deserialize.
        let asset = sample_asset();
        let body = ron::ser::to_string_pretty(&asset, ron::ser::PrettyConfig::default())
            .expect("serialize");
        assert!(!body.starts_with(MAGIC_HEADER));
        let back = from_ron_bytes(body.as_bytes()).expect("deserialize");
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
