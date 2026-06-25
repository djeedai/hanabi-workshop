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

use bevy::{
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future},
};
use bevy_hanabi::EffectAsset;
use hanabi_effect_graph::model::{EffectGraphAsset, FORMAT_VERSION};
use hanabi_node_graph::GraphView;

use crate::{
    document::{
        ActiveDocument, DocumentContent, DocumentRoot, DocumentUi, FocusDocument, RenderLayerPool,
        graph_view_from_layout, graph_view_to_layout,
    },
    edits::{EditKind, EditRequest},
    effect_graph::model::{EffectGraph, ImageBinding, NodeId, SharedStr},
};

/// File / document operations.
#[derive(Message, Debug, Clone)]
pub enum AppCommand {
    /// Create a new document seeded with the demo graph.
    NewDocument,
    /// Load an [`EffectGraphAsset`] from a `.hnb` file and open it as a
    /// document.
    OpenFile(PathBuf),
    /// Import a baked [`EffectAsset`] from a `.ron` file, reverse it into an
    /// [`EffectGraph`] (best-effort), and open it as a new untitled document.
    ImportFile(PathBuf),
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
                DialogKind::BindImageNode { doc, node, port } => {
                    let asset = bevy::asset::AssetPath::from(path.to_string_lossy().into_owned());
                    let binding = ImageBinding::Asset(asset);
                    let kind = match port {
                        Some(port) => EditKind::SetInputImageBinding {
                            node: *node,
                            port: port.clone(),
                            binding,
                        },
                        None => EditKind::SetImageNodeBinding {
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
/// effect files.
pub fn apply_app_commands(
    mut commands: Commands,
    mut reader: MessageReader<AppCommand>,
    mut effect_assets: ResMut<Assets<EffectAsset>>,
    mut layer_pool: ResMut<RenderLayerPool>,
    mut active: ResMut<ActiveDocument>,
    mut focus: MessageWriter<FocusDocument>,
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
                let graph = crate::effect_graph::demo::demo_graph();
                let preview_tag = crate::document::next_preview_tag();
                let (asset, provenance) = {
                    let registry = registry.read();
                    crate::effect_graph::bake::bake_preview_with_provenance(
                        &graph,
                        &registry,
                        preview_tag,
                    )
                };
                let handle = effect_assets.add(asset);
                let entity = spawn_document(
                    &mut commands,
                    &mut layer_pool,
                    root.0,
                    "Untitled".to_string(),
                    None,
                    graph,
                    handle,
                    preview_tag,
                    GraphView::default(),
                    provenance.literal_sites,
                    provenance.texture_plan,
                );
                active.0 = Some(entity);
                focus.write(FocusDocument(entity));
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
                    continue;
                }
                match load_graph_from_disk(path) {
                    Ok(loaded) => {
                        let name = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("Untitled")
                            .to_string();
                        let preview_tag = crate::document::next_preview_tag();
                        let (asset, provenance) = {
                            let registry = registry.read();
                            crate::effect_graph::bake::bake_preview_with_provenance(
                                &loaded.graph,
                                &registry,
                                preview_tag,
                            )
                        };
                        let handle = effect_assets.add(asset);
                        let graph_view = loaded
                            .layout
                            .as_ref()
                            .map(graph_view_from_layout)
                            .unwrap_or_default();
                        let entity = spawn_document(
                            &mut commands,
                            &mut layer_pool,
                            root.0,
                            name,
                            Some(path.clone()),
                            loaded.graph,
                            handle,
                            preview_tag,
                            graph_view,
                            provenance.literal_sites,
                            provenance.texture_plan,
                        );
                        active.0 = Some(entity);
                        focus.write(FocusDocument(entity));
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
                        let (graph, warnings) = hanabi_effect_graph::import::import(&asset);
                        for w in &warnings {
                            warn!("import {}: {w}", path.display());
                        }
                        let name = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("Imported")
                            .to_string();
                        let preview_tag = crate::document::next_preview_tag();
                        let (preview, provenance) = {
                            let registry = registry.read();
                            crate::effect_graph::bake::bake_preview_with_provenance(
                                &graph,
                                &registry,
                                preview_tag,
                            )
                        };
                        let handle = effect_assets.add(preview);
                        // No path: the source `.ron` is a baked artifact, not the
                        // canonical graph, so Save must prompt for a new `.hnb`.
                        let entity = spawn_document(
                            &mut commands,
                            &mut layer_pool,
                            root.0,
                            name,
                            None,
                            graph,
                            handle,
                            preview_tag,
                            GraphView::default(),
                            provenance.literal_sites,
                            provenance.texture_plan,
                        );
                        active.0 = Some(entity);
                        focus.write(FocusDocument(entity));
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
            }
            AppCommand::SaveActiveAs(path) => {
                let Some(entity) = active.0 else { continue };
                save_document(docs.reborrow(), entity, path);
            }
            AppCommand::CloseDocument(entity) => {
                if let Ok((_, content, _)) = docs.get(*entity) {
                    layer_pool.free(content.render_layer());
                }
                commands.entity(*entity).despawn();
                if active.0 == Some(*entity) {
                    active.0 = None;
                }
            }
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
        graph: content.graph().clone(),
        layout: Some(graph_view_to_layout(&ui.graph_view)),
    };
    match write_graph_to_disk(&asset, path) {
        Ok(()) => {
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
/// Shared by `NewDocument` and `OpenFile` (and by the startup seed). The
/// `effect` handle is expected to be the baked derivative of `graph`, and
/// `graph_view` seeds the node-graph panel's pan/zoom/positions (default for a
/// new document, restored from the saved layout when opening a file).
pub fn spawn_document(
    commands: &mut Commands,
    layer_pool: &mut RenderLayerPool,
    root: Entity,
    name: String,
    path: Option<PathBuf>,
    graph: EffectGraph,
    effect: Handle<EffectAsset>,
    preview_tag: u64,
    graph_view: GraphView,
    literal_sites: hanabi_effect_graph::bake::LiteralSites,
    texture_plan: hanabi_effect_graph::bake::TexturePlan,
) -> Entity {
    let layer = layer_pool.allocate();
    let entity = commands
        .spawn((
            DocumentContent::new(
                name,
                path,
                graph,
                effect,
                layer,
                preview_tag,
                literal_sites,
                texture_plan,
            ),
            DocumentUi {
                dock: crate::document::default_dock(),
                graph_view,
            },
            crate::playback::PlaybackState::default(),
            crate::history::History::default(),
            crate::plugins::shader_errors::ShaderErrors::default(),
            Transform::default(),
            Visibility::default(),
        ))
        .id();
    commands.entity(root).add_child(entity);
    entity
}
