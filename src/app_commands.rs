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

use std::{collections::HashMap, path::PathBuf};

use bevy::{
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future},
};
use bevy_hanabi::EffectAsset;
use hanabi_effect_graph::model::{EffectGraph, EffectGraphAsset, EmitterId, FORMAT_VERSION};
use hanabi_node_graph::GraphView;

use crate::{
    document::{
        ActiveDocument, DocumentContent, DocumentRoot, DocumentUi, EmitterRecord, FocusDocument,
        RenderLayerPool, bake_effect_records, graph_view_from_layout, graph_view_to_layout,
    },
    edits::{EditKind, EditRequest},
    effect_graph::model::{ImageBinding, NodeId, SharedStr},
    playback::PlaybackCommand,
};

/// File / document operations.
#[derive(Message, Debug, Clone)]
pub enum AppCommand {
    /// Create a new document seeded with the demo graph.
    NewDocument,
    /// Load a [`EffectGraphAsset`] from a `.hnb` file and open it as a
    /// document.
    OpenFile(PathBuf),
    /// Import a baked [`EffectAsset`] from a `.ron` file, reverse it into a
    /// [`EffectGraph`] (best-effort), and open it as a new untitled document.
    ImportFile(PathBuf),
    /// Save the active document. If it has no path yet, this is a no-op
    /// (UI should have popped a dialog and sent `SaveActiveAs` instead).
    SaveActive,
    /// Save the active document to the given path.
    SaveActiveAs(PathBuf),
    /// Save the given document to its current path. No-op if it has none
    /// (caller should route through the Save As dialog instead).
    SaveDocument(Entity),
    /// Save the given document to the given path.
    SaveDocumentAs(Entity, PathBuf),
    /// Close the given document. The caller is responsible for any
    /// "discard unsaved changes?" prompt; route through
    /// [`RequestCloseDocument`] to get one.
    ///
    /// [`RequestCloseDocument`]: AppCommand::RequestCloseDocument
    CloseDocument(Entity),
    /// Request to close the given document, going through the unsaved-changes
    /// guard (see [`crate::confirm`]). Handled there, not by
    /// [`apply_app_commands`].
    RequestCloseDocument(Entity),
    /// Request to quit the app, going through the unsaved-changes guard (see
    /// [`crate::confirm`]). Handled there, not by [`apply_app_commands`].
    RequestQuit,
}

pub struct AppCommandPlugin;

impl Plugin for AppCommandPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<AppCommand>()
            .init_resource::<PendingFileDialogs>()
            .add_message::<FocusDocument>()
            .add_systems(Update, (poll_file_dialogs, apply_app_commands).chain());
    }
}

/// What to do once a pending file dialog resolves.
#[derive(Debug, Clone)]
pub enum DialogKind {
    Open,
    Import,
    SaveAs,
    /// Add a recursively scanned folder to the global Assets panel.
    AddTextureFolder,
    /// Bind an image asset to an image source. With `port` set it targets a
    /// consumer's inline image input (a [`SetInputImageBinding`] edit);
    /// without, an Image node (a [`SetImageNodeBinding`] edit).
    ///
    /// [`SetImageNodeBinding`]: crate::edits::EditKind::SetImageNodeBinding
    /// [`SetInputImageBinding`]: crate::edits::EditKind::SetInputImageBinding
    BindImageNode {
        doc: Entity,
        node: NodeId,
        port: Option<SharedStr>,
    },
}

/// A native file dialog spawned on the async compute task pool.
///
/// Polled each frame; on completion the selected path becomes an
/// [`AppCommand`].
pub struct PendingDialog {
    pub kind: DialogKind,
    pub task: Task<Option<PathBuf>>,
}

#[derive(Resource, Default)]
pub struct PendingFileDialogs {
    pub dialogs: Vec<PendingDialog>,
}

impl PendingFileDialogs {
    /// Spawn a new native dialog on the async compute pool.
    ///
    /// Returns immediately; result arrives a few frames later.
    pub fn spawn(&mut self, kind: DialogKind) {
        let pool = AsyncComputeTaskPool::get();
        let task = match kind {
            DialogKind::Open => pool.spawn(async {
                rfd::AsyncFileDialog::new()
                    .add_filter("Effect Graph", &["hnb"])
                    .pick_file()
                    .await
                    .map(|h| h.path().to_path_buf())
            }),
            DialogKind::Import => pool.spawn(async {
                rfd::AsyncFileDialog::new()
                    .add_filter("Effect Asset", &["ron"])
                    .pick_file()
                    .await
                    .map(|h| h.path().to_path_buf())
            }),
            DialogKind::SaveAs => pool.spawn(async {
                rfd::AsyncFileDialog::new()
                    .add_filter("Effect Graph", &["hnb"])
                    .save_file()
                    .await
                    .map(|h| h.path().to_path_buf())
            }),
            DialogKind::AddTextureFolder => pool.spawn(async {
                rfd::AsyncFileDialog::new()
                    .pick_folder()
                    .await
                    .map(|h| h.path().to_path_buf())
            }),
            DialogKind::BindImageNode { .. } => pool.spawn(async {
                rfd::AsyncFileDialog::new()
                    .add_filter(
                        "Image",
                        &["png", "jpg", "jpeg", "ktx2", "basis", "exr", "hdr"],
                    )
                    .pick_file()
                    .await
                    .map(|h| h.path().to_path_buf())
            }),
        };
        self.dialogs.push(PendingDialog { kind, task });
    }
}

/// Poll all pending dialogs and emit the matching [`AppCommand`] on completion.
pub fn poll_file_dialogs(
    mut pending: ResMut<PendingFileDialogs>,
    mut app: MessageWriter<AppCommand>,
    mut edits: MessageWriter<EditRequest>,
    mut texture_library: MessageWriter<crate::asset_library::TextureLibraryCommand>,
    docs: Query<&DocumentContent>,
) {
    pending.dialogs.retain_mut(|dialog| {
        let Some(result) = block_on(future::poll_once(&mut dialog.task)) else {
            return true; // keep, not ready
        };
        if let Some(path) = result {
            match &dialog.kind {
                DialogKind::Open => {
                    app.write(AppCommand::OpenFile(path));
                }
                DialogKind::Import => {
                    app.write(AppCommand::ImportFile(path));
                }
                DialogKind::SaveAs => {
                    app.write(AppCommand::SaveActiveAs(path));
                }
                DialogKind::AddTextureFolder => {
                    texture_library
                        .write(crate::asset_library::TextureLibraryCommand::AddExternalRoot(path));
                }
                DialogKind::BindImageNode { doc, node, port } => {
                    // The node id alone resolves its owning emitter unambiguously
                    // (ids are unique across the whole document), so no extra
                    // `active_emitter` plumbing is needed for this dialog.
                    let Some(emitter) = docs
                        .get(*doc)
                        .ok()
                        .and_then(|content| content.effect_graph().emitter_owning_node(*node))
                    else {
                        warn!("BindImageNode: node {node:?} not found in document {doc:?}");
                        return false;
                    };
                    let asset = crate::asset_library::persisted_texture_asset_path(&path);
                    let binding = ImageBinding::Asset(asset);
                    let kind = match port {
                        Some(port) => EditKind::SetInputImageBinding {
                            emitter,
                            node: *node,
                            port: port.clone(),
                            binding,
                        },
                        None => EditKind::SetImageNodeBinding {
                            emitter,
                            node: *node,
                            binding,
                        },
                    };
                    edits.write(EditRequest::new(*doc, kind));
                }
            }
        }
        false // drop
    });
}

/// Single consumer of [`AppCommand`]s.
///
/// The only system that spawns or despawns document entities, or reads/writes
/// emitter files.
pub fn apply_app_commands(
    mut commands: Commands,
    mut reader: MessageReader<AppCommand>,
    mut emitter_assets: ResMut<Assets<EffectAsset>>,
    mut layer_pool: ResMut<RenderLayerPool>,
    mut active: ResMut<ActiveDocument>,
    mut focus: MessageWriter<FocusDocument>,
    mut playback: MessageWriter<PlaybackCommand>,
    mut recents: ResMut<crate::effect_library::RecentFiles>,
    registry: Res<AppTypeRegistry>,
    root: Option<Res<DocumentRoot>>,
    mut docs: Query<(Entity, &mut DocumentContent, &DocumentUi)>,
) {
    let Some(root) = root else {
        return;
    };

    for cmd in reader.read() {
        match cmd {
            AppCommand::NewDocument => {
                let effect_graph = hanabi_effect_graph::demo::demo_effect();
                let preview_tag = crate::document::next_preview_tag();
                let records = {
                    let registry = registry.read();
                    bake_effect_records(&effect_graph, &registry, preview_tag, &mut emitter_assets)
                };
                match records {
                    Ok(records) => {
                        let Some(entity) = spawn_document(
                            &mut commands,
                            &mut layer_pool,
                            root.0,
                            "Untitled".to_string(),
                            None,
                            effect_graph,
                            records,
                            Vec::new(),
                            preview_tag,
                            GraphView::default(),
                        ) else {
                            error!("new document has no emitter pipeline; refusing to open");
                            continue;
                        };
                        active.0 = Some(entity);
                        focus.write(FocusDocument(entity));
                    }
                    Err(errors) => {
                        error!(
                            "failed to bake new document ({} error(s)): {errors:?}",
                            errors.len()
                        );
                    }
                }
            }
            AppCommand::OpenFile(path) => {
                // Don't open the same file twice: a second document sharing the
                // path would be a competing source of truth (both Save to it).
                // Focus the already-open document instead.
                if let Some(existing) = docs
                    .iter()
                    .find(|(_, c, _)| c.path().is_some_and(|p| same_file(p, path)))
                    .map(|(e, _, _)| e)
                {
                    info!("{} is already open; focusing it", path.display());
                    active.0 = Some(existing);
                    focus.write(FocusDocument(existing));
                    recents.record(path);
                    crate::effect_library::save_recent_files(&recents);
                    continue;
                }
                match load_graph_from_disk(path) {
                    Ok(loaded) => {
                        let name = path
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("Untitled")
                            .to_string();
                        let preview_tag = crate::document::next_preview_tag();
                        let records = {
                            let registry = registry.read();
                            bake_effect_records(
                                &loaded.graph,
                                &registry,
                                preview_tag,
                                &mut emitter_assets,
                            )
                        };
                        let (records, bake_errors) = match records {
                            Ok(records) => (records, Vec::new()),
                            Err(errors) => {
                                warn!(
                                    "opened {} with preview disabled by {} bake error(s): {errors:?}",
                                    path.display(),
                                    errors.len()
                                );
                                (HashMap::new(), errors)
                            }
                        };
                        let graph_view = loaded
                            .layout
                            .as_ref()
                            .map(graph_view_from_layout)
                            .unwrap_or_default();
                        let Some(entity) = spawn_document(
                            &mut commands,
                            &mut layer_pool,
                            root.0,
                            name,
                            Some(path.clone()),
                            loaded.graph,
                            records,
                            bake_errors,
                            preview_tag,
                            graph_view,
                        ) else {
                            error!(
                                "{} has no emitter pipeline; refusing to open",
                                path.display()
                            );
                            continue;
                        };
                        active.0 = Some(entity);
                        focus.write(FocusDocument(entity));
                        recents.record(path);
                        crate::effect_library::save_recent_files(&recents);
                    }
                    Err(e) => {
                        error!("failed to open {}: {e}", path.display());
                    }
                }
            }
            AppCommand::ImportFile(path) => {
                let loaded = {
                    let registry = registry.read();
                    load_effect_asset_from_disk(path, &registry)
                };
                match loaded {
                    Ok(asset) => {
                        let (effect_graph, warnings) =
                            hanabi_effect_graph::import::import_effect(&asset);
                        for w in &warnings {
                            warn!("import {}: {w}", path.display());
                        }
                        let name = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("Imported")
                            .to_string();
                        let preview_tag = crate::document::next_preview_tag();
                        let records = {
                            let registry = registry.read();
                            bake_effect_records(
                                &effect_graph,
                                &registry,
                                preview_tag,
                                &mut emitter_assets,
                            )
                        };
                        match records {
                            Ok(records) => {
                                // No path: the source `.ron` is a baked artifact,
                                // not the canonical graph, so Save must prompt for
                                // a new `.hnb`.
                                let Some(entity) = spawn_document(
                                    &mut commands,
                                    &mut layer_pool,
                                    root.0,
                                    name,
                                    None,
                                    effect_graph,
                                    records,
                                    Vec::new(),
                                    preview_tag,
                                    GraphView::default(),
                                ) else {
                                    error!(
                                        "imported {} has no emitter pipeline; refusing to open",
                                        path.display()
                                    );
                                    continue;
                                };
                                active.0 = Some(entity);
                                focus.write(FocusDocument(entity));
                            }
                            Err(errors) => {
                                error!(
                                    "failed to bake imported {} ({} error(s)): {errors:?}",
                                    path.display(),
                                    errors.len()
                                );
                            }
                        }
                    }
                    Err(e) => {
                        error!("failed to import {}: {e}", path.display());
                    }
                }
            }
            AppCommand::SaveActive => {
                let Some(entity) = active.0 else { continue };
                let Ok((_, content, _)) = docs.get(entity) else {
                    continue;
                };
                let Some(path) = content.path().map(|p| p.to_path_buf()) else {
                    warn!("SaveActive with no path; UI should have used SaveActiveAs");
                    continue;
                };
                save_document(docs.reborrow(), entity, &path);
                recents.record(&path);
                crate::effect_library::save_recent_files(&recents);
            }
            AppCommand::SaveActiveAs(path) => {
                let Some(entity) = active.0 else { continue };
                save_document(docs.reborrow(), entity, path);
                recents.record(path);
                crate::effect_library::save_recent_files(&recents);
            }
            AppCommand::SaveDocument(entity) => {
                let Ok((_, content, _)) = docs.get(*entity) else {
                    continue;
                };
                let Some(path) = content.path().map(|p| p.to_path_buf()) else {
                    warn!("SaveDocument with no path; caller should use SaveDocumentAs");
                    continue;
                };
                save_document(docs.reborrow(), *entity, &path);
                recents.record(&path);
                crate::effect_library::save_recent_files(&recents);
            }
            AppCommand::SaveDocumentAs(entity, path) => {
                save_document(docs.reborrow(), *entity, path);
                recents.record(path);
                crate::effect_library::save_recent_files(&recents);
            }
            AppCommand::CloseDocument(entity) => {
                playback.write(PlaybackCommand::CloseDocument(*entity));
            }
            // Guarded lifecycle requests are handled by `crate::confirm`.
            AppCommand::RequestCloseDocument(_) | AppCommand::RequestQuit => {}
        }
    }
}

/// Whether two paths point at the same file.
///
/// Compares canonicalized forms when both resolve on disk (so `./a.hnb` and
/// `a.hnb` match), else a plain compare.
fn same_file(a: &std::path::Path, b: &std::path::Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(ca), Ok(cb)) => ca == cb,
        _ => a == b,
    }
}

fn save_document(
    mut docs: Query<(Entity, &mut DocumentContent, &DocumentUi)>,
    entity: Entity,
    path: &std::path::Path,
) {
    let Ok((_, mut content, ui)) = docs.get_mut(entity) else {
        return;
    };
    let asset = EffectGraphAsset {
        version: FORMAT_VERSION,
        graph: content.effect_graph().clone(),
        layout: Some(graph_view_to_layout(&ui.graph_view, content.effect_graph())),
    };
    match write_graph_to_disk(&asset, path) {
        Ok(()) => {
            // Re-derive the tab name from the saved file (an untitled document
            // saved for the first time should adopt its new file name). The
            // extension is kept so the name reads as the on-disk `.hnb` file.
            if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                content.set_name(name.to_string());
            }
            content.set_path(Some(path.to_path_buf()));
            content.mark_dirty(false);
            info!("saved {} to {}", content.name(), path.display());
        }
        Err(e) => error!("failed to save {}: {e}", path.display()),
    }
}

fn load_graph_from_disk(path: &std::path::Path) -> Result<EffectGraphAsset, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    hanabi_effect_graph::from_ron_bytes(&bytes).map_err(|e| e.to_string())
}

fn load_effect_asset_from_disk(
    path: &std::path::Path,
    registry: &bevy::reflect::TypeRegistry,
) -> Result<EffectAsset, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    EffectAsset::deserialize(&text, registry).map_err(|e| e.to_string())
}

fn write_graph_to_disk(asset: &EffectGraphAsset, path: &std::path::Path) -> Result<(), String> {
    let text = hanabi_effect_graph::to_ron_string(asset).map_err(|e| e.to_string())?;
    std::fs::write(path, text).map_err(|e| e.to_string())
}

/// Spawn a new document entity as a child of the document root.
///
/// Shared by `NewDocument`, `OpenFile`, and `ImportFile`. `emitters` is
/// normally contains the baked-and-registered derivative of `effect_graph` (see
/// [`bake_effect_records`]). It may be empty when opening an incomplete saved
/// graph; `bake_errors` then explains why its preview is disabled while the
/// graph remains editable and saveable. `graph_view` seeds the node-graph
/// panel's pan/zoom/node and source-context positions.
///
/// Returns `None` without spawning anything if `effect_graph` has no emitter
/// pipeline at all: [`DocumentUi`] always needs a valid `active_emitter` to
/// focus, and an empty document has none to offer.
pub fn spawn_document(
    commands: &mut Commands,
    layer_pool: &mut RenderLayerPool,
    root: Entity,
    name: String,
    path: Option<PathBuf>,
    effect_graph: EffectGraph,
    emitters: HashMap<EmitterId, EmitterRecord>,
    bake_errors: Vec<hanabi_effect_graph::bake::EffectBakeError>,
    preview_tag: u64,
    graph_view: GraphView,
) -> Option<Entity> {
    let active_emitter = effect_graph.emitters.first().map(|e| e.id)?;
    let layer = layer_pool.allocate();
    let mut content = DocumentContent::new(name, path, effect_graph, emitters, layer, preview_tag);
    content.set_bake_errors(bake_errors);
    let entity = commands
        .spawn((
            content,
            DocumentUi {
                graph_view,
                ..DocumentUi::new(active_emitter)
            },
            crate::playback::PlaybackState::default(),
            crate::history::History::default(),
            crate::plugins::shader_errors::ShaderErrors::default(),
            Transform::default(),
            Visibility::default(),
        ))
        .id();
    commands.entity(root).add_child(entity);
    Some(entity)
}
