//! Lazy loading and egui registration for image previews.
//!
//! UI systems request previews by their persisted [`AssetPath`]. The cache
//! keeps each image alive, tracks asynchronous load completion, and registers a
//! loaded image with [`EguiUserTextures`] exactly once. Relative paths use
//! Bevy's normal approved load path, while absolute paths explicitly opt into
//! unapproved loads.
//!
//! [`AssetPath`]: bevy::asset::AssetPath
//! [`EguiUserTextures`]: bevy_egui::EguiUserTextures

use std::{
    collections::HashMap,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

use bevy::{
    asset::{AssetLoadError, AssetPath, LoadState},
    prelude::*,
};
use bevy_egui::{EguiTextureHandle, EguiUserTextures};
use egui::TextureId;

/// Installs lazy texture preview caching.
///
/// Add this plugin alongside [`EguiPlugin`]. UI systems can order
/// themselves after [`TexturePreviewSystems`] to observe load transitions made
/// during the current frame.
///
/// [`EguiPlugin`]: bevy_egui::EguiPlugin
/// [`TexturePreviewSystems`]: crate::texture_preview::TexturePreviewSystems
pub struct TexturePreviewPlugin;

impl Plugin for TexturePreviewPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TexturePreviewCache>().add_systems(
            Update,
            (invalidate_failed_previews, update_texture_previews)
                .chain()
                .in_set(TexturePreviewSystems),
        );
    }
}

/// Labels the system that advances texture preview load states.
///
/// UI systems may order themselves after this set when same-frame visibility of
/// newly completed loads matters.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TexturePreviewSystems;

/// Stable cache identity for a persisted image asset path.
///
/// Construction lexically removes `.` segments and collapses cancellable `..`
/// segments without touching the filesystem. The asset source and sub-asset
/// label are preserved.
///
/// [`AssetPath`]: bevy::asset::AssetPath
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TexturePreviewKey(AssetPath<'static>);

impl TexturePreviewKey {
    /// Creates a normalized preview key.
    ///
    /// The path is converted to owned storage so the key can outlive the caller
    /// and be retained in a Bevy resource.
    pub fn new(path: impl Into<AssetPath<'static>>) -> Self {
        Self::from_asset_path(&path.into())
    }

    /// Creates a normalized preview key from a borrowed asset path.
    ///
    /// This accepts any [`AssetPath`] lifetime and always returns an owned key.
    ///
    /// [`AssetPath`]: bevy::asset::AssetPath
    pub fn from_asset_path(path: &AssetPath<'_>) -> Self {
        let source = path.source().clone_owned();
        let label = path.label().map(str::to_owned);
        let mut normalized =
            AssetPath::from_path_buf(normalize_path(path.path())).with_source(source);
        if let Some(label) = label {
            normalized = normalized.with_label(label);
        }
        Self(normalized)
    }

    /// Returns the normalized owned asset path.
    ///
    /// This is the exact path used to load the image.
    pub fn asset_path(&self) -> &AssetPath<'static> {
        &self.0
    }
}

impl From<AssetPath<'static>> for TexturePreviewKey {
    fn from(path: AssetPath<'static>) -> Self {
        Self::new(path)
    }
}

impl From<PathBuf> for TexturePreviewKey {
    fn from(path: PathBuf) -> Self {
        Self::new(path)
    }
}

impl From<String> for TexturePreviewKey {
    fn from(path: String) -> Self {
        Self::new(path)
    }
}

/// A loaded image ready for use in egui.
///
/// The strong image handle is retained both here and by Bevy-egui's texture
/// registry, while `texture_id` can be passed directly to egui image widgets.
#[derive(Debug, Clone)]
pub struct ReadyTexturePreview {
    /// Strong handle keeping the Bevy image asset alive.
    _image: Handle<Image>,
    /// Registered texture identity for egui image widgets.
    pub texture_id: TextureId,
}

/// Current asynchronous state of one texture preview.
///
/// Every variant retains the strong image handle created for the request. A
/// failed entry remains cached and will not be loaded repeatedly by UI frames.
#[derive(Debug, Clone)]
pub enum TexturePreviewState {
    /// The image asset is still loading.
    Loading { image: Handle<Image> },
    /// The image is loaded and registered with egui.
    Ready(ReadyTexturePreview),
    /// Bevy's asset loader rejected or failed to decode the image.
    Failed {
        _image: Handle<Image>,
        _error: Arc<AssetLoadError>,
    },
}

/// Lazily populated image preview states.
///
/// Call [`request`] or [`request_path`] from a UI system. Repeated requests are
/// cheap and return the existing state without restarting the asset load.
///
/// [`request`]: TexturePreviewCache::request
/// [`request_path`]: TexturePreviewCache::request_path
#[derive(Resource, Default)]
pub struct TexturePreviewCache {
    entries: HashMap<TexturePreviewKey, TexturePreviewState>,
}

impl TexturePreviewCache {
    /// Requests a preview and returns its current state.
    ///
    /// Relative paths go through [`load`], preserving Bevy's approved-path
    /// checks. Absolute paths use [`override_unapproved`] for explicit
    /// user-picked files outside the asset root.
    ///
    /// [`load`]: AssetServer::load
    /// [`override_unapproved`]: bevy::asset::LoadBuilder::override_unapproved
    pub fn request(
        &mut self,
        asset_server: &AssetServer,
        key: impl Into<TexturePreviewKey>,
    ) -> &TexturePreviewState {
        self.entries.entry(key.into()).or_insert_with_key(|key| {
            let path = key.asset_path().clone_owned();
            let image = if path.path().is_absolute() {
                asset_server.load_builder().override_unapproved().load(path)
            } else {
                asset_server.load(path)
            };
            TexturePreviewState::Loading { image }
        })
    }

    /// Requests a preview directly from a persisted asset path.
    ///
    /// This convenience method normalizes and owns the borrowed path before
    /// forwarding to [`request`].
    ///
    /// [`request`]: TexturePreviewCache::request
    pub fn request_path(
        &mut self,
        asset_server: &AssetServer,
        path: &AssetPath<'_>,
    ) -> &TexturePreviewState {
        self.request(asset_server, TexturePreviewKey::from_asset_path(path))
    }
}

fn invalidate_failed_previews(
    catalog: Res<crate::asset_library::TextureCatalog>,
    mut cache: ResMut<TexturePreviewCache>,
) {
    if catalog.is_changed() {
        cache
            .entries
            .retain(|_, state| !matches!(state, TexturePreviewState::Failed { .. }));
    }
}

/// Advances loading previews to ready or failed states.
///
/// A successfully loaded image is registered with [`EguiUserTextures`] only
/// during its single `Loading` to `Ready` transition.
///
/// [`EguiUserTextures`]: bevy_egui::EguiUserTextures
pub fn update_texture_previews(
    asset_server: Res<AssetServer>,
    mut egui_textures: ResMut<EguiUserTextures>,
    mut cache: ResMut<TexturePreviewCache>,
) {
    for state in cache.entries.values_mut() {
        let TexturePreviewState::Loading { image } = state else {
            continue;
        };
        let image = image.clone();
        match asset_server.get_load_state(image.id()) {
            Some(LoadState::Loaded) => {
                let texture_id = egui_textures.add_image(EguiTextureHandle::Strong(image.clone()));
                *state = TexturePreviewState::Ready(ReadyTexturePreview {
                    _image: image,
                    texture_id,
                });
            }
            Some(LoadState::Failed(error)) => {
                *state = TexturePreviewState::Failed {
                    _image: image,
                    _error: error,
                };
            }
            Some(LoadState::NotLoaded | LoadState::Loading) | None => {}
        }
    }
}

fn normalize_path(path: &Path) -> PathBuf {
    let absolute = path.is_absolute();
    let mut normalized = PathBuf::new();

    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(Path::new(std::path::MAIN_SEPARATOR_STR)),
            Component::CurDir => {}
            Component::ParentDir => {
                let can_pop = matches!(
                    normalized.components().next_back(),
                    Some(Component::Normal(_))
                );
                if can_pop {
                    normalized.pop();
                } else if !absolute {
                    normalized.push(Component::ParentDir.as_os_str());
                }
            }
            Component::Normal(segment) => normalized.push(segment),
        }
    }

    normalized
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_normalizes_lexically_equivalent_paths() {
        let plain = TexturePreviewKey::new("textures/smoke.png");
        let dotted = TexturePreviewKey::new("textures/./effects/../smoke.png");

        assert_eq!(plain, dotted);
    }

    #[test]
    fn key_preserves_source_label_and_leading_parents() {
        let key = TexturePreviewKey::new("catalog://../../textures/./smoke.png#preview");

        assert_eq!(
            key.asset_path(),
            &AssetPath::parse("catalog://../../textures/smoke.png#preview")
        );
    }

    #[cfg(unix)]
    #[test]
    fn key_preserves_absolute_paths() {
        let key = TexturePreviewKey::new(PathBuf::from("/art/./effects/../smoke.png"));

        assert_eq!(key.asset_path().path(), Path::new("/art/smoke.png"));
        assert!(key.asset_path().path().is_absolute());
    }
}
