//! Runtime thumbnail generation and caching for the Home browser.
//!
//! Each browsable effect is previewed by an image rendered off-screen: the
//! `.hnb` graph is baked, spawned onto a dedicated camera + render layer,
//! simulated for a few warm-up frames, then captured with [`Screenshot`] and
//! written to a content-addressed PNG cache under the OS cache dir. Cached PNGs
//! are reused across runs and invalidated automatically when a file's contents
//! change (the cache key is a hash of the file bytes).
//!
//! The UI reads [`ThumbnailCache`] for display and emits [`ThumbnailRequest`]
//! for effects it wants previewed.
//!
//! [`Screenshot`]: bevy::render::view::screenshot::Screenshot

use std::{
    collections::{HashMap, VecDeque},
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
};

use bevy::{
    camera::{Hdr, RenderTarget, visibility::RenderLayers},
    post_process::bloom::Bloom,
    prelude::*,
    render::{
        render_resource::{
            Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
        },
        view::screenshot::{Screenshot, ScreenshotCaptured, save_to_disk},
    },
};
use bevy_egui::{EguiTextureHandle, EguiUserTextures};
use bevy_hanabi::{EffectAsset, EffectMaterial, EffectProperties, ParticleEffect};

use crate::{
    document::{RenderLayerPool, next_preview_tag},
    effect_graph::bake::{PlannedImage, bake_preview_with_provenance},
    plugins::reconcile::TexturePlaceholder,
};

/// Side length in pixels of a generated thumbnail.
///
/// A multiple of 64 keeps `width * 4` aligned to wgpu's 256-byte row stride, so
/// the captured image has no row padding.
const THUMB_SIZE: u32 = 256;

/// Frames an effect simulates off-screen before its thumbnail is captured.
const WARMUP_FRAMES: u32 = 40;

/// Maximum number of thumbnails rendering at once (bounds render-layer/GPU
/// use).
const MAX_IN_FLIGHT: usize = 2;

/// Soft cap on total on-disk thumbnail cache size; oldest PNGs are evicted once
/// a freshly written thumbnail pushes the directory past this.
const CACHE_CAP_BYTES: u64 = 64 * 1024 * 1024;

/// Display state of a single effect's thumbnail.
pub enum ThumbState {
    /// A preview is being rendered; show a placeholder meanwhile.
    Generating,
    /// A preview image is ready and registered with egui.
    Ready(Handle<Image>),
    /// Generation failed (unreadable/unparseable file); show a placeholder.
    Failed,
}

/// Per-effect thumbnail states, keyed by `.hnb` path.
#[derive(Resource, Default)]
pub struct ThumbnailCache {
    states: HashMap<PathBuf, ThumbState>,
}

impl ThumbnailCache {
    /// Iterate effects whose thumbnail is ready, with their image handles.
    pub fn ready_handles(&self) -> impl Iterator<Item = (&PathBuf, &Handle<Image>)> {
        self.states.iter().filter_map(|(path, state)| match state {
            ThumbState::Ready(handle) => Some((path, handle)),
            _ => None,
        })
    }
}

/// Pending and in-flight generation bookkeeping.
#[derive(Resource, Default)]
struct ThumbnailWork {
    queue: VecDeque<GenJob>,
    in_flight: usize,
}

/// A queued generation job: the source effect and its target cache PNG.
struct GenJob {
    path: PathBuf,
    png: PathBuf,
}

/// A request to (lazily) generate a thumbnail for an effect path.
#[derive(Message, Debug, Clone)]
pub struct ThumbnailRequest(pub PathBuf);

/// A request to drop all cached thumbnails (in-memory states and on-disk PNGs).
///
/// Effects still visible in the browser re-request generation on the next
/// frame, so the cache repopulates lazily.
#[derive(Message, Debug, Clone)]
pub struct ClearThumbnailCache;

/// Marks the root of an off-screen thumbnail render scene.
///
/// Carries what the capture step needs: the source path, the target PNG, the
/// allocated render layer, the render-target image, and the warm-up countdown.
#[derive(Component)]
struct ThumbnailJob {
    path: PathBuf,
    png: PathBuf,
    layer: usize,
    image: Handle<Image>,
    warmup: u32,
    /// Set once the [`Screenshot`] has been requested, so we don't request it
    /// again on subsequent frames.
    awaiting_capture: bool,
}

pub struct ThumbnailPlugin;

impl Plugin for ThumbnailPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ThumbnailCache>()
            .init_resource::<ThumbnailWork>()
            .add_message::<ThumbnailRequest>()
            .add_message::<ClearThumbnailCache>()
            .add_systems(
                Update,
                (
                    (handle_thumbnail_requests, drive_thumbnail_generation).chain(),
                    advance_thumbnail_jobs,
                    clear_thumbnail_cache,
                ),
            );
    }
}

/// Resolve requests into cache hits or queued generation jobs.
fn handle_thumbnail_requests(
    mut requests: MessageReader<ThumbnailRequest>,
    mut cache: ResMut<ThumbnailCache>,
    mut work: ResMut<ThumbnailWork>,
    asset_server: Res<AssetServer>,
    mut egui_textures: ResMut<EguiUserTextures>,
) {
    for ThumbnailRequest(path) in requests.read() {
        if cache.states.contains_key(path) {
            continue;
        }
        let Some(key) = cache_key(path) else {
            cache.states.insert(path.clone(), ThumbState::Failed);
            continue;
        };
        let Some(png) = thumbs_dir().map(|d| d.join(format!("{key}.png"))) else {
            cache.states.insert(path.clone(), ThumbState::Failed);
            continue;
        };
        if png.exists() {
            let handle = load_registered(&asset_server, &mut egui_textures, &png);
            cache.states.insert(path.clone(), ThumbState::Ready(handle));
        } else {
            cache.states.insert(path.clone(), ThumbState::Generating);
            work.queue.push_back(GenJob {
                path: path.clone(),
                png,
            });
        }
    }
}

/// Start rendering queued jobs, up to [`MAX_IN_FLIGHT`] at a time.
fn drive_thumbnail_generation(
    mut commands: Commands,
    mut work: ResMut<ThumbnailWork>,
    mut cache: ResMut<ThumbnailCache>,
    registry: Res<AppTypeRegistry>,
    mut effect_assets: ResMut<Assets<EffectAsset>>,
    mut images: ResMut<Assets<Image>>,
    mut layer_pool: ResMut<RenderLayerPool>,
    placeholder: Res<TexturePlaceholder>,
    asset_server: Res<AssetServer>,
) {
    while work.in_flight < MAX_IN_FLIGHT {
        let Some(job) = work.queue.pop_front() else {
            break;
        };
        let Some(graph) = load_graph(&job.path) else {
            cache.states.insert(job.path.clone(), ThumbState::Failed);
            continue;
        };

        let (asset, provenance) = {
            let registry = registry.read();
            bake_preview_with_provenance(&graph, &registry, next_preview_tag())
        };
        let effect_handle = effect_assets.add(asset);

        let material_images: Vec<Handle<Image>> = provenance
            .texture_plan
            .iter()
            .map(|planned| match planned {
                PlannedImage::Asset(path) => asset_server
                    .load_builder()
                    .override_unapproved()
                    .load(path.clone()),
                PlannedImage::Runtime(_) | PlannedImage::Unbound => placeholder.0.clone(),
            })
            .collect();

        let layer = layer_pool.allocate();
        let layers = RenderLayers::layer(layer);
        let target = make_thumb_target(&mut images);

        spawn_thumbnail_scene(
            &mut commands,
            &layers,
            effect_handle,
            material_images,
            target.clone(),
            ThumbnailJob {
                path: job.path,
                png: job.png,
                layer,
                image: target,
                warmup: WARMUP_FRAMES,
                awaiting_capture: false,
            },
        );
        work.in_flight += 1;
    }
}

/// Count down each job's warm-up, then request its screenshot capture.
fn advance_thumbnail_jobs(mut commands: Commands, mut jobs: Query<(Entity, &mut ThumbnailJob)>) {
    for (scene, mut job) in &mut jobs {
        if job.awaiting_capture {
            continue;
        }
        if job.warmup > 0 {
            job.warmup -= 1;
            continue;
        }
        job.awaiting_capture = true;

        let path = job.path.clone();
        let png = job.png.clone();
        let layer = job.layer;
        let image = job.image.clone();

        commands.spawn(Screenshot::image(image)).observe(
            move |captured: On<ScreenshotCaptured>,
                  mut commands: Commands,
                  mut cache: ResMut<ThumbnailCache>,
                  mut work: ResMut<ThumbnailWork>,
                  mut layer_pool: ResMut<RenderLayerPool>,
                  asset_server: Res<AssetServer>,
                  mut egui_textures: ResMut<EguiUserTextures>| {
                // `save_to_disk` is bevy's own PNG writer; it does not create
                // parent dirs, so ensure the cache dir exists first.
                if let Some(parent) = png.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                save_to_disk(png.clone())(captured);

                let state = if png.exists() {
                    // Re-read the freshly written file even if this path was
                    // loaded earlier this session (Regenerate rewrites the same
                    // content-hash path); without a file watcher the AssetServer
                    // would otherwise keep serving the stale cached image.
                    asset_server.reload(png.clone());
                    let handle = load_registered(&asset_server, &mut egui_textures, &png);
                    if let Some(dir) = thumbs_dir() {
                        enforce_cache_cap(&dir, CACHE_CAP_BYTES);
                    }
                    ThumbState::Ready(handle)
                } else {
                    warn!("failed to write thumbnail {}", png.display());
                    ThumbState::Failed
                };
                cache.states.insert(path.clone(), state);

                commands.entity(scene).despawn();
                layer_pool.free(layer);
                work.in_flight = work.in_flight.saturating_sub(1);
            },
        );
    }
}

/// Drop all cached thumbnails on request: clear in-memory states and delete the
/// on-disk PNGs. Visible browser cards re-request generation next frame.
fn clear_thumbnail_cache(
    mut requests: MessageReader<ClearThumbnailCache>,
    mut cache: ResMut<ThumbnailCache>,
) {
    if requests.read().count() == 0 {
        return;
    }
    cache.states.clear();
    if let Some(dir) = thumbs_dir() {
        for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "png") {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
}

/// Evict the oldest cached PNGs until the directory is within `cap_bytes`.
fn enforce_cache_cap(dir: &Path, cap_bytes: u64) {
    let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if path.extension().is_none_or(|ext| ext != "png") {
                return None;
            }
            let meta = entry.metadata().ok()?;
            Some((meta.modified().ok()?, meta.len(), path))
        })
        .collect();

    let total: u64 = files.iter().map(|(_, size, _)| *size).sum();
    if total <= cap_bytes {
        return;
    }

    // Oldest first, so eviction removes least-recently-written thumbnails.
    files.sort_by_key(|(mtime, _, _)| *mtime);
    let mut over = total - cap_bytes;
    for (_, size, path) in files {
        if over == 0 {
            break;
        }
        if std::fs::remove_file(&path).is_ok() {
            over = over.saturating_sub(size);
        }
    }
}

/// Spawn the off-screen scene: light, effect, and a camera rendering to
/// `target`.
fn spawn_thumbnail_scene(
    commands: &mut Commands,
    layers: &RenderLayers,
    effect: Handle<EffectAsset>,
    material_images: Vec<Handle<Image>>,
    target: Handle<Image>,
    job: ThumbnailJob,
) {
    commands
        .spawn((Transform::default(), Visibility::default(), job))
        .with_children(|p| {
            p.spawn((
                DirectionalLight {
                    illuminance: 10_000.0,
                    ..default()
                },
                Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.6, 0.4, 0.0)),
                layers.clone(),
            ));
            let mut effect = p.spawn((
                ParticleEffect::new(effect),
                EffectProperties::default(),
                Transform::IDENTITY,
                layers.clone(),
            ));
            if !material_images.is_empty() {
                effect.insert(EffectMaterial {
                    images: material_images,
                });
            }
            // Fixed three-quarter framing looking at the origin.
            let eye = Vec3::new(2.6, 1.8, 3.8);
            p.spawn((
                Camera3d::default(),
                Camera {
                    order: -1,
                    clear_color: ClearColorConfig::Custom(Color::BLACK),
                    ..default()
                },
                // Match the live viewport: HDR + bloom for glowing particle
                // cores instead of flat quads.
                Hdr,
                Bloom {
                    intensity: 0.25,
                    ..default()
                },
                RenderTarget::Image(target.into()),
                Transform::from_translation(eye).looking_at(Vec3::ZERO, Vec3::Y),
                layers.clone(),
            ));
        });
}

/// Create a square off-screen render target for a thumbnail camera.
fn make_thumb_target(images: &mut Assets<Image>) -> Handle<Image> {
    let size = Extent3d {
        width: THUMB_SIZE,
        height: THUMB_SIZE,
        ..default()
    };
    let mut image = Image {
        texture_descriptor: TextureDescriptor {
            label: Some("thumbnail-target"),
            size,
            dimension: TextureDimension::D2,
            format: TextureFormat::Bgra8UnormSrgb,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::COPY_SRC
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
        ..default()
    };
    image.resize(size);
    images.add(image)
}

/// Load a cached PNG via the asset server and register it with egui.
///
/// The cache dir is outside `assets/`, so the load opts past the asset server's
/// unapproved-path policy.
fn load_registered(
    asset_server: &AssetServer,
    egui_textures: &mut EguiUserTextures,
    png: &Path,
) -> Handle<Image> {
    let handle: Handle<Image> = asset_server
        .load_builder()
        .override_unapproved()
        .load(png.to_path_buf());
    egui_textures.add_image(EguiTextureHandle::Strong(handle.clone()));
    handle
}

/// Load an effect graph from a `.hnb` file, or `None` on read/parse error.
fn load_graph(path: &Path) -> Option<crate::effect_graph::model::EffectGraph> {
    let bytes = std::fs::read(path).ok()?;
    hanabi_effect_graph::from_ron_bytes(&bytes)
        .ok()
        .map(|asset| asset.graph)
}

/// Content-addressed cache key: a hash of the file bytes, or `None` if
/// unreadable.
fn cache_key(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    Some(format!("{:016x}", hasher.finish()))
}

/// The thumbnail cache directory under the OS cache dir.
fn thumbs_dir() -> Option<PathBuf> {
    dirs::cache_dir().map(|d| d.join("hanabi-workshop").join("thumbs"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_is_stable_and_content_sensitive() {
        let dir = std::env::temp_dir().join(format!("hwk-thumb-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.hnb");
        let b = dir.join("b.hnb");
        std::fs::write(&a, b"hello").unwrap();
        std::fs::write(&b, b"hello").unwrap();
        // Same content => same key; regardless of path.
        assert_eq!(cache_key(&a), cache_key(&b));
        // Changed content => different key.
        let before = cache_key(&a);
        std::fs::write(&a, b"world").unwrap();
        assert_ne!(before, cache_key(&a));
        // Missing file => no key.
        assert!(cache_key(&dir.join("missing.hnb")).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_cap_evicts_down_to_limit() {
        let dir = std::env::temp_dir().join(format!("hwk-cap-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Five 1 KiB PNGs = 5 KiB on disk; a 3 KiB cap must evict at least two.
        for i in 0..5 {
            std::fs::write(dir.join(format!("t{i}.png")), vec![0u8; 1024]).unwrap();
        }
        // A non-PNG file must never be touched by the cap.
        std::fs::write(dir.join("keep.txt"), vec![0u8; 4096]).unwrap();

        enforce_cache_cap(&dir, 3 * 1024);

        let png_bytes: u64 = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "png"))
            .map(|e| e.metadata().unwrap().len())
            .sum();
        assert!(png_bytes <= 3 * 1024, "cache still over cap: {png_bytes}");
        assert!(dir.join("keep.txt").exists(), "non-PNG was evicted");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
