//! App-level command channel: file operations (new/open/save/close).
//!
//! Mirrors the `EditRequest` design (see `crate::edits`), but for
//! operations that don't fit the per-document edit model — creating
//! and destroying documents, and persisting them to disk.
//!
//! UI code (menu bar) emits [`AppCommand`] messages. A single
//! [`apply_app_commands`] system consumes them and is the **only**
//! site that spawns/despawns document entities or touches the file
//! system. Path-picking dialogs are popped on the UI side (via
//! `rfd`) before emitting the command, so this system stays pure
//! and synchronous.

use std::path::PathBuf;

use bevy::prelude::*;
use bevy::tasks::{block_on, futures_lite::future, AsyncComputeTaskPool, Task};
use bevy_hanabi::EffectAsset;

use crate::document::{
    ActiveDocument, DocumentContent, DocumentRoot, DocumentUi, RenderLayerPool,
};

/// File / document operations.
#[derive(Message, Debug, Clone)]
pub enum AppCommand {
    /// Create a new, empty `EffectAsset` document.
    NewDocument,
    /// Load an `EffectAsset` from a RON file and open it as a document.
    OpenFile(PathBuf),
    /// Save the active document. If it has no path yet, this is a no-op
    /// (UI should have popped a dialog and sent `SaveActiveAs` instead).
    SaveActive,
    /// Save the active document to the given path.
    SaveActiveAs(PathBuf),
    /// Close the given document. **No confirmation** in v1; UI is
    /// responsible for any "discard unsaved changes?" prompt.
    CloseDocument(Entity),
}

pub struct AppCommandPlugin;

impl Plugin for AppCommandPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<AppCommand>()
            .init_resource::<PendingFileDialogs>()
            .add_systems(Update, (poll_file_dialogs, apply_app_commands).chain());
    }
}

/// What to do once a pending file dialog resolves.
#[derive(Debug, Clone, Copy)]
pub enum DialogKind {
    Open,
    SaveAs,
}

/// A native file dialog spawned on the async compute task pool. Polled
/// each frame; on completion the selected path becomes an [`AppCommand`].
pub struct PendingDialog {
    pub kind: DialogKind,
    pub task: Task<Option<PathBuf>>,
}

#[derive(Resource, Default)]
pub struct PendingFileDialogs {
    pub dialogs: Vec<PendingDialog>,
}

impl PendingFileDialogs {
    /// Spawn a new native dialog on the async compute pool. Returns
    /// immediately; result arrives a few frames later.
    pub fn spawn(&mut self, kind: DialogKind) {
        let pool = AsyncComputeTaskPool::get();
        let task = match kind {
            DialogKind::Open => pool.spawn(async {
                rfd::AsyncFileDialog::new()
                    .add_filter("EffectAsset (RON)", &["ron"])
                    .pick_file()
                    .await
                    .map(|h| h.path().to_path_buf())
            }),
            DialogKind::SaveAs => pool.spawn(async {
                rfd::AsyncFileDialog::new()
                    .add_filter("EffectAsset (RON)", &["ron"])
                    .save_file()
                    .await
                    .map(|h| h.path().to_path_buf())
            }),
        };
        self.dialogs.push(PendingDialog { kind, task });
    }
}

/// Polls all pending dialogs; emits the matching [`AppCommand`] when
/// a dialog completes.
pub fn poll_file_dialogs(
    mut pending: ResMut<PendingFileDialogs>,
    mut app: MessageWriter<AppCommand>,
) {
    pending.dialogs.retain_mut(|dialog| {
        let Some(result) = block_on(future::poll_once(&mut dialog.task)) else {
            return true; // keep, not ready
        };
        if let Some(path) = result {
            match dialog.kind {
                DialogKind::Open => app.write(AppCommand::OpenFile(path)),
                DialogKind::SaveAs => app.write(AppCommand::SaveActiveAs(path)),
            };
        }
        false // drop
    });
}

/// Single consumer of [`AppCommand`]s — the only system that spawns or
/// despawns document entities, or reads/writes effect files.
pub fn apply_app_commands(
    mut commands: Commands,
    mut reader: MessageReader<AppCommand>,
    mut effect_assets: ResMut<Assets<EffectAsset>>,
    mut layer_pool: ResMut<RenderLayerPool>,
    mut active: ResMut<ActiveDocument>,
    root: Option<Res<DocumentRoot>>,
    mut docs: Query<&mut DocumentContent>,
) {
    let Some(root) = root else {
        return;
    };

    for cmd in reader.read() {
        match cmd {
            AppCommand::NewDocument => {
                let asset = effect_assets.add(EffectAsset::default());
                let entity = spawn_document(
                    &mut commands,
                    &mut layer_pool,
                    root.0,
                    "Untitled".to_string(),
                    None,
                    asset,
                );
                active.0 = Some(entity);
            }
            AppCommand::OpenFile(path) => match load_effect_from_disk(path) {
                Ok(asset) => {
                    let name = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("Untitled")
                        .to_string();
                    let handle = effect_assets.add(asset);
                    let entity = spawn_document(
                        &mut commands,
                        &mut layer_pool,
                        root.0,
                        name,
                        Some(path.clone()),
                        handle,
                    );
                    active.0 = Some(entity);
                }
                Err(e) => {
                    error!("failed to open {}: {e}", path.display());
                }
            },
            AppCommand::SaveActive => {
                let Some(entity) = active.0 else { continue };
                let Ok(content) = docs.get(entity) else { continue };
                let Some(path) = content.path().map(|p| p.to_path_buf()) else {
                    warn!("SaveActive with no path; UI should have used SaveActiveAs");
                    continue;
                };
                save_document(docs.reborrow(), entity, &path, &effect_assets);
            }
            AppCommand::SaveActiveAs(path) => {
                let Some(entity) = active.0 else { continue };
                save_document(docs.reborrow(), entity, path, &effect_assets);
            }
            AppCommand::CloseDocument(entity) => {
                commands.entity(*entity).despawn();
                if active.0 == Some(*entity) {
                    active.0 = None;
                }
            }
        }
    }
}

fn save_document(
    mut docs: Query<&mut DocumentContent>,
    entity: Entity,
    path: &std::path::Path,
    effect_assets: &Assets<EffectAsset>,
) {
    let Ok(mut content) = docs.get_mut(entity) else {
        return;
    };
    let Some(asset) = effect_assets.get(content.effect()) else {
        error!("save: effect asset missing for document {entity:?}");
        return;
    };
    match write_effect_to_disk(asset, path) {
        Ok(()) => {
            content.set_path(Some(path.to_path_buf()));
            content.mark_dirty(false);
            info!("saved {} to {}", content.name(), path.display());
        }
        Err(e) => error!("failed to save {}: {e}", path.display()),
    }
}

fn load_effect_from_disk(path: &std::path::Path) -> Result<EffectAsset, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    ron::de::from_bytes::<EffectAsset>(&bytes).map_err(|e| e.to_string())
}

fn write_effect_to_disk(asset: &EffectAsset, path: &std::path::Path) -> Result<(), String> {
    let pretty = ron::ser::PrettyConfig::default();
    let text = ron::ser::to_string_pretty(asset, pretty).map_err(|e| e.to_string())?;
    std::fs::write(path, text).map_err(|e| e.to_string())
}

/// Spawns a new document entity as a child of the document root.
/// Shared by `NewDocument` and `OpenFile` (and by the startup seed).
pub fn spawn_document(
    commands: &mut Commands,
    layer_pool: &mut RenderLayerPool,
    root: Entity,
    name: String,
    path: Option<PathBuf>,
    effect: Handle<EffectAsset>,
) -> Entity {
    let layer = layer_pool.allocate();
    let entity = commands
        .spawn((
            DocumentContent::new(name, path, effect, layer),
            DocumentUi::default(),
            Transform::default(),
            Visibility::default(),
        ))
        .id();
    commands.entity(root).add_child(entity);
    entity
}
