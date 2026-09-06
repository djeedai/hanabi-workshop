//! Document-centric data model — ECS form.
//!
//! Each open document is an **entity** carrying [`DocumentContent`] and
//! [`DocumentUi`] components. Document entities are children of the singleton
//! [`DocumentRoot`] entity so that `Children` provides stable tab ordering.
//! Each document owns exactly one [`EffectGraph`], which is the complete
//! artist-authored effect and may contain multiple
//! [`hanabi_effect_graph::model::EmitterGraph`] pipelines.
//!
//! ## Edit boundary
//!
//! `DocumentContent` fields are private and are only mutated via
//! `pub(crate)` setter methods. The convention is that **only the
//! `apply_edits` system in `crate::edits` calls those setters.** Code
//! review enforces this — the setters are intentionally not public.
//! `DocumentUi` is freely mutable: UI state isn't undoable.

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use bevy::prelude::*;
use bevy_hanabi::EffectAsset;
use egui_dock::{DockState, NodeIndex};
pub use hanabi_effect_graph::ModifierGroup;
use hanabi_effect_graph::{
    bake::{BakedEmitter, EffectBakeError, LiteralSites, TexturePlan, bake_effect},
    model::{EffectGraph, EmitterId},
};

/// Snapshot the node-graph panel's [`GraphView`] into a [`GraphLayout`].
///
/// Captures pan/zoom and world positions for saving. Entries are sorted by id
/// so saved files are diff-stable. `effect_graph` disambiguates a widget
/// position entry that is actually a spawn-source context
/// (`CpuSpawner`/`GpuEvent`) from an ordinary expression/modifier node: both
/// kinds of id are minted from the same document-wide allocator and rendered as
/// plain widget [`NodeId`]s (the widget itself has no concept of a source
/// context — see `crate::effect_graph::view`), so only cross-checking against
/// the model can tell them apart.
///
/// [`GraphView`]: hanabi_node_graph::GraphView
/// [`GraphLayout`]: hanabi_effect_graph::model::GraphLayout
/// [`NodeId`]: hanabi_node_graph::NodeId
pub fn graph_view_to_layout(
    view: &hanabi_node_graph::GraphView,
    effect_graph: &EffectGraph,
) -> hanabi_effect_graph::model::GraphLayout {
    use hanabi_effect_graph::model::{
        GraphLayout, NodeId as MNodeId, SourceId as MSourceId, StackId as MStackId,
    };

    let mut node_pos: Vec<(MNodeId, (f64, f64))> = Vec::new();
    let mut source_pos: Vec<(MSourceId, (f64, f64))> = Vec::new();
    for (id, p) in &view.positions {
        let raw = id.get();
        if let Some(sid) = MSourceId::new(raw)
            && effect_graph.source(sid).is_some()
        {
            source_pos.push((sid, (p.x, p.y)));
        } else if let Some(nid) = MNodeId::new(raw)
            && effect_graph.emitter_owning_node(nid).is_some()
        {
            node_pos.push((nid, (p.x, p.y)));
        }
    }
    node_pos.sort_by_key(|(id, _)| id.get());
    source_pos.sort_by_key(|(id, _)| id.get());

    let mut stack_pos: Vec<(MStackId, (f64, f64))> = view
        .stack_positions
        .iter()
        .filter_map(|(id, p)| MStackId::new(id.get()).map(|m| (m, (p.x, p.y))))
        .collect();
    stack_pos.sort_by_key(|(id, _)| id.get());

    GraphLayout {
        pan: (view.pan.x, view.pan.y),
        zoom: view.zoom,
        node_pos,
        stack_pos,
        source_pos,
    }
}

/// Rebuild a [`GraphView`] from a persisted [`GraphLayout`].
///
/// Any node/stack not in the layout is left unplaced for the panel's
/// auto-layout to seed. Source-context positions merge into the same
/// [`GraphView::positions`] map as ordinary node positions — see
/// [`graph_view_to_layout`] for why the widget can't (and doesn't need to)
/// tell the two kinds of id apart.
///
/// [`GraphView`]: hanabi_node_graph::GraphView
/// [`GraphLayout`]: hanabi_effect_graph::model::GraphLayout
/// [`GraphView::positions`]: hanabi_node_graph::GraphView::positions
pub fn graph_view_from_layout(
    layout: &hanabi_effect_graph::model::GraphLayout,
) -> hanabi_node_graph::GraphView {
    use hanabi_node_graph::{GraphView, NodeId as WNodeId, StackId as WStackId};

    let mut view = GraphView::default();
    view.pan = glam::DVec2::new(layout.pan.0, layout.pan.1);
    if layout.zoom > 0.0 {
        view.zoom = layout.zoom;
    }
    for (id, (x, y)) in &layout.node_pos {
        if let Some(w) = WNodeId::new(id.get()) {
            view.positions.insert(w, glam::DVec2::new(*x, *y));
        }
    }
    for (id, (x, y)) in &layout.stack_pos {
        if let Some(w) = WStackId::new(id.get()) {
            view.stack_positions.insert(w, glam::DVec2::new(*x, *y));
        }
    }
    for (id, (x, y)) in &layout.source_pos {
        if let Some(w) = WNodeId::new(id.get()) {
            view.positions.insert(w, glam::DVec2::new(*x, *y));
        }
    }
    view
}

/// Source of process-unique [`DocumentContent::preview_tag`] values.
///
/// Monotonic and never reused, so two open documents — even ones baked from
/// byte-identical graphs — get distinct preview-asset names. This keeps each
/// document's baked [`bevy_hanabi::EffectAsset`] individually identifiable
/// (e.g. in the debug inspector) rather than aliasing on a shared name.
static NEXT_PREVIEW_TAG: AtomicU64 = AtomicU64::new(1);

/// Allocate a fresh, process-unique preview tag for a new document.
pub fn next_preview_tag() -> u64 {
    NEXT_PREVIEW_TAG.fetch_add(1, Ordering::Relaxed)
}

/// Document- and emitter-unique preview asset name: `{base}~{tag}~{emitter}`.
///
/// `preview_tag` alone (the pre-multi-emitter scheme) is only document-unique:
/// two emitter pipelines in the same [`EffectGraph`] can share the same name
/// (e.g. two untouched `"untitled"` pipelines), which would otherwise alias
/// one `hanabi/{name}_…` shader-cache path across both. Appending the
/// [`EmitterId`] makes every emitter's preview asset name unique even within
/// one document.
pub fn preview_asset_name(base: &str, preview_tag: u64, emitter: EmitterId) -> String {
    format!("{base}~{preview_tag}~{}", emitter.get())
}

/// Bake every emitter of `effect_graph` and register each as a fresh preview
/// [`EffectAsset`] in `assets`, keyed by [`EmitterId`].
///
/// Fails atomically: [`bake_effect`] validates topology and bakes every emitter
/// before this function adds anything to `assets`, so a caller never ends up
/// publishing a document with some emitters baked and others missing (see
/// [`DocumentContent::set_emitter_records`]). Each asset's name is made
/// document- and emitter-unique by [`preview_asset_name`] so two open
/// documents, or two emitters within one document, never alias a shader-cache
/// entry.
///
/// Used by New/Open/Import and every whole-document structural rebake. Pure
/// live-value edits bypass baking and upload directly to the matching proxy
/// instance.
pub fn bake_effect_records(
    effect_graph: &EffectGraph,
    registry: &bevy::reflect::TypeRegistry,
    preview_tag: u64,
    assets: &mut Assets<EffectAsset>,
) -> Result<HashMap<EmitterId, EmitterRecord>, Vec<EffectBakeError>> {
    let baked = bake_effect(effect_graph, registry)?;
    let mut out = HashMap::with_capacity(baked.emitters.len());
    for BakedEmitter {
        emitter,
        mut asset,
        provenance,
        parent,
    } in baked.emitters
    {
        asset.name = preview_asset_name(&asset.name, preview_tag, emitter);
        let handle = assets.add(asset);
        out.insert(
            emitter,
            EmitterRecord {
                asset: handle,
                literal_sites: provenance.literal_sites,
                texture_plan: provenance.texture_plan,
                parent,
            },
        );
    }
    Ok(out)
}

// ============================================================================
// Components
// ============================================================================

/// One emitter pipeline's canonical runtime record within a
/// [`DocumentContent`].
///
/// Produced by baking the matching [`hanabi_effect_graph::model::EmitterGraph`]
/// inside the document's [`EffectGraph`] (see [`bake_effect_records`]);
/// replaced wholesale on a structural/topology rebake. `asset` is the
/// *canonical* preview [`bevy_hanabi::EffectAsset`] — never instantiated
/// directly; see [`crate::proxy::ProxyEmitters`] for the promoted-literal
/// derivative that actually drives the viewport.
pub struct EmitterRecord {
    /// Canonical preview asset handle for this emitter. Its name is
    /// document/emitter-unique (see [`preview_asset_name`]).
    pub asset: Handle<EffectAsset>,
    /// Provenance of every promotable literal in the current bake: maps each
    /// [`LiteralSite`] (a graph node or inline port default) to the
    /// `ExprHandle` it produced in `asset`. Drives the live literal-tweak fast
    /// path (see [`crate::proxy::ProxyEmitters`]).
    ///
    /// [`LiteralSite`]: crate::effect_graph::bake::LiteralSite
    pub literal_sites: LiteralSites,
    /// Resolved texture slots of the current bake, ordered by sampling index.
    /// The renderer builds this emitter's [`bevy_hanabi::EffectMaterial`] from
    /// it.
    pub texture_plan: TexturePlan,
    /// The parent emitter that spawns particles into this one via a GPU source
    /// context, if any — mirrors [`EffectGraph::parent_emitter`] as of this
    /// record's last bake. `None` for a CPU-rooted emitter.
    pub parent: Option<EmitterId>,
}

/// Content of a document.
///
/// Fields are private; mutation goes through `pub(crate)` setters used by
/// `apply_edits` only.
#[derive(Component)]
pub struct DocumentContent {
    name: String,
    path: Option<PathBuf>,
    /// The canonical, edited, saved effect graph.
    ///
    /// Contains every emitter pipeline plus the spawn sources and topology
    /// links that drive them.
    /// Every contained emitter's runtime bake output lives in `emitters`,
    /// keyed by its stable [`EmitterId`].
    effect_graph: EffectGraph,
    /// Canonical runtime record per emitter pipeline in `effect_graph`. Kept in
    /// sync with `effect_graph` by `apply_edits`: every emitter id in
    /// `effect_graph.emitters` has a matching entry here once it has baked
    /// successfully at least once (see module docs on partial-failure
    /// handling).
    emitters: HashMap<EmitterId, EmitterRecord>,
    /// Errors from the most recent strict bake attempt.
    ///
    /// The authored graph remains saveable while these are present. The
    /// preview may contain a reduced, valid projection that omits incomplete
    /// GPU-event branches.
    bake_errors: Vec<EffectBakeError>,
    dirty: bool,
    render_layer: usize,
    /// Process-unique tag baked into every emitter's preview asset name (see
    /// [`preview_asset_name`]) to give this document's assets distinct names
    /// from other open documents'. See [`next_preview_tag`].
    preview_tag: u64,
}

impl DocumentContent {
    pub fn new(
        name: String,
        path: Option<PathBuf>,
        effect_graph: EffectGraph,
        emitters: HashMap<EmitterId, EmitterRecord>,
        render_layer: usize,
        preview_tag: u64,
    ) -> Self {
        Self {
            name,
            path,
            effect_graph,
            emitters,
            bake_errors: Vec::new(),
            dirty: false,
            render_layer,
            preview_tag,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn path(&self) -> Option<&std::path::Path> {
        self.path.as_deref()
    }
    /// The canonical effect graph: emitter pipelines, spawn source
    /// contexts, and the topology links between them.
    pub fn effect_graph(&self) -> &EffectGraph {
        &self.effect_graph
    }
    /// Mutable access to the canonical `EffectGraph`.
    ///
    /// Only callable from [`crate::edits::apply_edits`] (the single edit
    /// writer).
    pub(crate) fn effect_graph_mut(&mut self) -> &mut EffectGraph {
        &mut self.effect_graph
    }
    /// Every emitter pipeline id in this document, in stable document order
    /// (the order `EffectGraph::emitters` stores them — insertion order, so it
    /// stays diff-stable across saves).
    pub fn emitter_ids(&self) -> impl Iterator<Item = EmitterId> + '_ {
        self.effect_graph.emitters.iter().map(|e| e.id)
    }
    /// Emitter ids currently present in the valid runtime preview projection.
    pub fn preview_emitter_ids(&self) -> impl Iterator<Item = EmitterId> + '_ {
        self.effect_graph
            .emitters
            .iter()
            .map(|emitter| emitter.id)
            .filter(|emitter| self.emitters.contains_key(emitter))
    }
    /// Errors from the latest strict bake of the authored graph.
    pub fn bake_errors(&self) -> &[EffectBakeError] {
        &self.bake_errors
    }
    /// The canonical runtime record for one emitter, if it has baked
    /// successfully at least once.
    pub fn emitter_record(&self, emitter: EmitterId) -> Option<&EmitterRecord> {
        self.emitters.get(&emitter)
    }
    /// The canonical preview asset handle for one emitter.
    pub fn emitter_asset(&self, emitter: EmitterId) -> Option<&Handle<EffectAsset>> {
        self.emitters.get(&emitter).map(|r| &r.asset)
    }
    /// Literal provenance of one emitter's current canonical bake.
    pub fn literal_sites(&self, emitter: EmitterId) -> Option<&LiteralSites> {
        self.emitters.get(&emitter).map(|r| &r.literal_sites)
    }
    /// The parent emitter driving `emitter` via a GPU source context, if any.
    pub fn emitter_parent(&self, emitter: EmitterId) -> Option<EmitterId> {
        self.emitters.get(&emitter).and_then(|r| r.parent)
    }
    pub fn dirty(&self) -> bool {
        self.dirty
    }
    pub fn render_layer(&self) -> usize {
        self.render_layer
    }
    /// Process-unique preview tag; baked into every emitter's preview asset
    /// name.
    pub fn preview_tag(&self) -> u64 {
        self.preview_tag
    }

    // --- Mutators below: ONLY callable from `crate::edits::apply_edits`. ---

    pub(crate) fn set_name(&mut self, new: String) -> String {
        let old = std::mem::replace(&mut self.name, new);
        self.dirty = true;
        old
    }

    pub(crate) fn set_path(&mut self, new: Option<PathBuf>) {
        self.path = new;
    }

    pub(crate) fn mark_dirty(&mut self, dirty: bool) {
        self.dirty = dirty;
    }

    /// Replace the whole per-emitter canonical record collection.
    ///
    /// Called after a transactional whole-document rebake (a structural or
    /// topology edit, or a load) produced by [`bake_effect_records`]. Replacing
    /// every record together — rather than merging — is deliberate: a partial
    /// update could pair one emitter's freshly-baked record with another's
    /// stale one, and [`bake_effect_records`] never returns a collection
    /// missing an emitter that exists in `effect_graph`.
    pub(crate) fn set_emitter_records(&mut self, records: HashMap<EmitterId, EmitterRecord>) {
        self.emitters = records;
        self.bake_errors.clear();
    }

    /// Record a failed strict bake while retaining or replacing the preview.
    pub(crate) fn set_bake_errors(&mut self, errors: Vec<EffectBakeError>) {
        self.bake_errors = errors;
    }
}

/// Per-document UI state.
///
/// Freely mutable — not part of the edit channel.
#[derive(Component)]
pub struct DocumentUi {
    pub dock: DockState<PanelKind>,
    /// Persistable view state for the node-graph panel (pan/zoom/positions).
    pub graph_view: hanabi_node_graph::GraphView,
    /// Modifier node currently requesting contextual viewport help.
    pub modifier_gizmo_node: Option<crate::effect_graph::model::NodeId>,
    /// Render frame in which the Graph panel last refreshed the gizmo target.
    pub modifier_gizmo_frame: u32,
    /// Whether the horizontal grid is visible in this document's viewports.
    pub show_viewport_grid: bool,
    /// Emitter the Emitter/Properties/Material/Shaders/Graph panels
    /// currently operate on.
    ///
    /// Updated by the UI whenever the user interacts with any source, stack,
    /// modifier, or expression belonging to a different emitter; falls back to
    /// whichever emitter the canvas last focused when selection is empty or
    /// spans emitters. Pure UI focus state — not part of the edit channel, so
    /// switching it is never undoable. Always one of the owning document's
    /// `EffectGraph::emitters`; there is no meaningful default construction for
    /// this type (an `EmitterId` has no zero/sentinel value, and an empty
    /// document has no emitter to default to), so it is threaded through at
    /// construction instead — see [`DocumentUi::new`].
    pub active_emitter: EmitterId,
}

impl DocumentUi {
    /// A fresh per-document UI state, focused on `active_emitter`.
    pub fn new(active_emitter: EmitterId) -> Self {
        Self {
            dock: default_dock(),
            graph_view: hanabi_node_graph::GraphView::default(),
            modifier_gizmo_node: None,
            modifier_gizmo_frame: 0,
            show_viewport_grid: true,
            active_emitter,
        }
    }
}

/// Build the default per-document dock layout.
///
/// Three columns left-to-right: `[(Viewport on top, Effect / Material /
/// Properties tabbed below) ≈28% | Graph (Shaders tabbed behind) ≈54% | Assets
/// ≈18%]`. The Viewport is sized to be roughly square.
pub fn default_dock() -> DockState<PanelKind> {
    // The center hosts the Graph (visible) with Shaders tabbed behind it.
    let mut dock = DockState::new(vec![PanelKind::Graph, PanelKind::Shaders]);
    let surface = dock.main_surface_mut();
    // Left column: Viewport on top (≈ square), inspector panels below.
    let [graph_node, left_node] =
        surface.split_left(NodeIndex::root(), 0.28, vec![PanelKind::Viewport(0)]);
    surface.split_below(
        left_node,
        0.5,
        vec![
            PanelKind::Emitter,
            PanelKind::Material,
            PanelKind::Properties,
        ],
    );
    surface.split_right(graph_node, 0.75, vec![PanelKind::Assets]);
    dock
}

/// Panel kinds that may appear inside a document's dock area.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PanelKind {
    Viewport(usize),
    /// Outline of the emitter: particle layout + modifier groups.
    Emitter,
    /// User-defined properties on the emitter's `Module`.
    Properties,
    /// Texture slots (the emitter's material image bindings).
    Material,
    /// Browsable project, preset, and external assets.
    Assets,
    /// Generated WGSL shaders (init / update / render) baked by hanabi.
    Shaders,
    /// Node-graph editor canvas (PoC).
    Graph,
}

/// Component on the per-document camera entity.
///
/// Stores the local viewport index, the render-target image handle, and the
/// orbit-camera state (target/yaw/pitch/distance). `Transform` is derived from
/// the orbit state by the `apply_camera_controls` system.
#[derive(Component)]
pub struct ViewportCamera {
    pub viewport_index: usize,
    pub image: Handle<Image>,
    /// Point the camera orbits around.
    pub target: Vec3,
    /// Rotation around the world `Y` axis, in radians. 0 looks down `-Z`.
    pub yaw: f32,
    /// Elevation above the equatorial plane, in radians. Clamped to
    /// `(-π/2 + ε, π/2 - ε)` to avoid gimbal flip.
    pub pitch: f32,
    /// Distance from `target` to the camera, in world units. Clamped
    /// to `[0.1, 1e4]`.
    pub distance: f32,
}

impl ViewportCamera {
    /// Compute the camera world position from orbit state.
    pub fn eye(&self) -> Vec3 {
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        self.target
            + Vec3::new(
                self.distance * cp * sy,
                self.distance * sp,
                self.distance * cp * cy,
            )
    }

    /// Compute the camera's right, up, and forward basis vectors.
    pub fn basis(&self) -> Mat3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let right = Vec3::new(cos_yaw, 0.0, -sin_yaw);
        let forward = Vec3::new(-cos_pitch * sin_yaw, -sin_pitch, -cos_pitch * cos_yaw);
        let up = right.cross(forward);
        Mat3::from_cols(right, up, forward)
    }

    /// Derive the camera transform from its orbit state.
    pub fn transform(&self) -> Transform {
        let basis = self.basis();
        Transform::from_translation(self.eye()).looking_to(basis.col(2), basis.col(1))
    }
}

/// Marker for the (single) scene root of a document.
///
/// Children of this entity are the visible scene content (light, one
/// `ParticleEffect` entity per baked emitter in the document's `EffectGraph`,
/// ...).
#[derive(Component)]
pub struct DocumentSceneRoot;

/// Marker on a per-emitter preview `ParticleEffect` entity naming which
/// canonical emitter pipeline it instances.
///
/// Inserted by `crate::plugins::reconcile` alongside the entity's
/// `ParticleEffect`/`EffectProperties`; read back by shader-error attribution
/// and live-value routing so a GPU-side emitter instance always resolves to
/// its owning [`EmitterId`] without needing the [`EmitterSceneEntities`] map.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneEmitter(pub EmitterId);

/// `EmitterId → Entity` map of a document's preview instances, one entry per
/// baked emitter currently spawned under its [`DocumentSceneRoot`].
///
/// Lives on the `DocumentSceneRoot` entity itself, rebuilt by
/// `crate::plugins::reconcile::reconcile_documents` whenever the scene is
/// (re)spawned. The canonical lookup used by playback (restart every CPU-root
/// spawner), live-value upload routing, and shader-error attribution — all of
/// which need to go from "which emitter" to "which live ECS entity" without
/// re-deriving it from `Children` themselves.
#[derive(Component, Debug, Clone, Default)]
pub struct EmitterSceneEntities(pub HashMap<EmitterId, Entity>);

impl EmitterSceneEntities {
    /// The preview entity instancing `emitter`, if currently spawned.
    pub fn get(&self, emitter: EmitterId) -> Option<Entity> {
        self.0.get(&emitter).copied()
    }
}

// ============================================================================
// Resources
// ============================================================================

/// The singleton parent entity whose `Children` are the open documents.
///
/// In user-visible (tab-bar) order.
#[derive(Resource)]
pub struct DocumentRoot(pub Entity);

/// The currently focused document, if any.
#[derive(Resource, Default)]
pub struct ActiveDocument(pub Option<Entity>);

/// One-shot request to focus a document's tab in the outer dock.
///
/// Emitted when a document is opened or created (and when a re-open is
/// redirected to an already-open document); read by the UI, which moves dock
/// focus to the tab.
#[derive(Message, Debug, Clone, Copy)]
pub struct FocusDocument(pub Entity);

/// Allocates render layers (1..=31) to documents.
#[derive(Resource, Default)]
pub struct RenderLayerPool {
    used: u32,
}

impl RenderLayerPool {
    pub fn allocate(&mut self) -> usize {
        for layer in 1..=31 {
            let bit = 1u32 << layer;
            if self.used & bit == 0 {
                self.used |= bit;
                return layer;
            }
        }
        panic!("render layer exhaustion: no more than 31 documents at once");
    }

    pub fn free(&mut self, layer: usize) {
        if layer < 32 {
            self.used &= !(1u32 << layer);
        }
    }
}

/// Cache of `(doc_entity, viewport_idx) → Image handle`, rebuilt every frame.
///
/// Rebuilt by reconciliation. Used by the UI to look up the egui texture for
/// each viewport panel.
#[derive(Resource, Default)]
pub struct DocumentViewports {
    pub by_doc: HashMap<Entity, ViewportSlots>,
}

#[derive(Default)]
pub struct ViewportSlots {
    pub images: HashMap<usize, Handle<Image>>,
    pub cameras: HashMap<usize, Entity>,
}

/// Per-viewport desired pixel size, keyed by `(doc_entity, viewport_index)`.
///
/// Written by the UI on render and consumed by the resize-to-fit system.
#[derive(Resource, Default)]
pub struct ViewportSizeRequests(pub HashMap<(Entity, usize), UVec2>);
