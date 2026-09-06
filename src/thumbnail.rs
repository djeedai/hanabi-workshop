//! Runtime thumbnail generation and caching for the Home browser.
//!
//! Each browsable emitter is previewed by an image rendered off-screen: the
//! `.hnb` graph is baked, spawned onto a dedicated camera + render layer,
//! simulated for a few warm-up frames, then captured with [`Screenshot`] and
//! written to a content-addressed PNG cache under the OS cache dir. Cached PNGs
//! are reused across runs and invalidated automatically when a file's contents
//! change (the cache key is a hash of the file bytes).
//!
//! The UI reads [`ThumbnailCache`] for display and emits [`ThumbnailRequest`]
//! for emitters it wants previewed.
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
use hanabi_effect_graph::model::{EffectGraph, EmitterId};

use crate::{
    document::{EmitterRecord, RenderLayerPool, bake_effect_records, next_preview_tag},
    effect_graph::bake::PlannedImage,
    playback::TeardownEffect,
    plugins::reconcile::TexturePlaceholder,
};

/// Side length in pixels of a generated thumbnail.
///
/// A multiple of 64 keeps `width * 4` aligned to wgpu's 256-byte row stride, so
/// the captured image has no row padding.
const THUMB_SIZE: u32 = 256;

/// Frames an emitter simulates off-screen before its thumbnail is captured.
const WARMUP_FRAMES: u32 = 40;

/// Maximum number of thumbnails rendering at once (bounds render-layer/GPU
/// use).
const MAX_IN_FLIGHT: usize = 2;

/// Soft cap on total on-disk thumbnail cache size; oldest PNGs are evicted once
/// a freshly written thumbnail pushes the directory past this.
const CACHE_CAP_BYTES: u64 = 64 * 1024 * 1024;

/// Display state of a single emitter's thumbnail.
pub enum ThumbState {
    /// A preview is being rendered; show a placeholder meanwhile.
    Generating,
    /// A preview image is ready and registered with egui.
    Ready(Handle<Image>),
    /// Generation failed (unreadable/unparseable file); show a placeholder.
    Failed,
}

/// Per-emitter thumbnail states, keyed by `.hnb` path.
#[derive(Resource, Default)]
pub struct ThumbnailCache {
    states: HashMap<PathBuf, ThumbState>,
}

impl ThumbnailCache {
    /// Iterate emitters whose thumbnail is ready, with their image handles.
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

/// A queued generation job: the source emitter and its target cache PNG.
struct GenJob {
    path: PathBuf,
    png: PathBuf,
}

/// A request to (lazily) generate a thumbnail for an emitter path.
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

/// Particle entities belonging to one thumbnail scene.
#[derive(Component)]
struct ThumbnailEmitters(Vec<Entity>);

/// Delays thumbnail scene destruction until detached GPU events are released.
#[derive(Component)]
struct PendingThumbnailCleanup {
    layer: usize,
    armed: bool,
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
                    cleanup_thumbnail_scenes.after(advance_thumbnail_jobs),
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
    mut emitter_assets: ResMut<Assets<EffectAsset>>,
    mut images: ResMut<Assets<Image>>,
    mut layer_pool: ResMut<RenderLayerPool>,
    placeholder: Res<TexturePlaceholder>,
    asset_server: Res<AssetServer>,
) {
    while work.in_flight < MAX_IN_FLIGHT {
        let Some(job) = work.queue.pop_front() else {
            break;
        };
        let Some(effect_graph) = load_graph(&job.path) else {
            cache.states.insert(job.path.clone(), ThumbState::Failed);
            continue;
        };

        let records = {
            let registry = registry.read();
            bake_effect_records(
                &effect_graph,
                &registry,
                next_preview_tag(),
                &mut emitter_assets,
            )
        };
        let records = match records {
            Ok(records) => records,
            Err(errors) => {
                warn!(
                    "thumbnail bake failed for {}: {errors:?}",
                    job.path.display()
                );
                cache.states.insert(job.path.clone(), ThumbState::Failed);
                continue;
            }
        };

        let layer = layer_pool.allocate();
        let layers = RenderLayers::layer(layer);
        let target = make_thumb_target(&mut images);

        spawn_thumbnail_scene(
            &mut commands,
            &layers,
            &effect_graph,
            records,
            &asset_server,
            &placeholder,
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
                  mut egui_textures: ResMut<EguiUserTextures>,
                  thumbnail_emitters: Query<&ThumbnailEmitters>,
                  effect_parents: Query<(), With<bevy_hanabi::EffectParent>>,
                  mut particle_effects: Query<&mut ParticleEffect>,
                  teardown_effect: Res<TeardownEffect>| {
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

                let gpu_hierarchy = thumbnail_emitters.get(scene).is_ok_and(|emitters| {
                    emitters
                        .0
                        .iter()
                        .any(|entity| effect_parents.contains(*entity))
                });
                if gpu_hierarchy {
                    if let Ok(emitters) = thumbnail_emitters.get(scene) {
                        for &emitter_entity in &emitters.0 {
                            if effect_parents.contains(emitter_entity) {
                                commands
                                    .entity(emitter_entity)
                                    .remove::<bevy_hanabi::EffectParent>();
                            }
                            if let Ok(mut emitter) = particle_effects.get_mut(emitter_entity) {
                                emitter.handle = teardown_effect.0.clone();
                            }
                        }
                    }
                    commands.entity(scene).insert(PendingThumbnailCleanup {
                        layer,
                        armed: false,
                    });
                } else {
                    commands.entity(scene).despawn();
                    layer_pool.free(layer);
                    work.in_flight = work.in_flight.saturating_sub(1);
                }
            },
        );
    }
}

/// Destroy detached thumbnail scenes after a complete render frame.
fn cleanup_thumbnail_scenes(
    mut commands: Commands,
    mut scenes: Query<(Entity, &mut PendingThumbnailCleanup)>,
    mut layer_pool: ResMut<RenderLayerPool>,
    mut work: ResMut<ThumbnailWork>,
) {
    for (scene, mut cleanup) in &mut scenes {
        if !cleanup.armed {
            cleanup.armed = true;
            continue;
        }
        layer_pool.free(cleanup.layer);
        work.in_flight = work.in_flight.saturating_sub(1);
        commands.entity(scene).despawn();
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

/// Spawn the off-screen scene: light, camera rendering to `target`, and one
/// `ParticleEffect` per baked emitter in `effect_graph`, with `EffectParent`
/// wiring for GPU-driven children — mirrors
/// [`plugins::reconcile::ensure_scene_root`](crate::plugins::reconcile).
fn spawn_thumbnail_scene(
    commands: &mut Commands,
    layers: &RenderLayers,
    effect_graph: &EffectGraph,
    records: HashMap<EmitterId, EmitterRecord>,
    asset_server: &AssetServer,
    placeholder: &TexturePlaceholder,
    target: Handle<Image>,
    job: ThumbnailJob,
) {
    let scene = commands
        .spawn((Transform::default(), Visibility::default(), job))
        .id();
    commands.entity(scene).with_children(|p| {
        p.spawn((
            DirectionalLight {
                illuminance: 10_000.0,
                ..default()
            },
            Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.6, 0.4, 0.0)),
            layers.clone(),
        ));
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

    // First pass: one `ParticleEffect` per baked emitter, in document order.
    let emitter_ids: Vec<EmitterId> = effect_graph.emitters.iter().map(|e| e.id).collect();
    let mut entity_map: HashMap<EmitterId, Entity> = HashMap::with_capacity(emitter_ids.len());
    for &emitter_id in &emitter_ids {
        let Some(record) = records.get(&emitter_id) else {
            continue;
        };
        let images: Vec<Handle<Image>> = record
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

        let mut emitter_cmds = commands.spawn((
            ParticleEffect::new(record.asset.clone()),
            EffectProperties::default(),
            Transform::IDENTITY,
            layers.clone(),
        ));
        if !images.is_empty() {
            emitter_cmds.insert(EffectMaterial { images });
        }
        let entity = emitter_cmds.id();
        commands.entity(scene).add_child(entity);
        entity_map.insert(emitter_id, entity);
    }

    // Second pass: `EffectParent` is order-independent, but every sibling
    // must already exist.
    for &emitter_id in &emitter_ids {
        if let Some(parent_id) = records.get(&emitter_id).and_then(|r| r.parent)
            && let (Some(&child), Some(&parent)) =
                (entity_map.get(&emitter_id), entity_map.get(&parent_id))
        {
            commands
                .entity(child)
                .insert(bevy_hanabi::EffectParent::new(parent));
        }
    }
    commands
        .entity(scene)
        .insert(ThumbnailEmitters(entity_map.into_values().collect()));
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

/// Load a document's `EffectGraph` from a `.hnb` file, or `None` on read/parse
/// error.
fn load_graph(path: &Path) -> Option<EffectGraph> {
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
