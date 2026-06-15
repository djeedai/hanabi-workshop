//! Document-centric data model — ECS form.
//!
//! Each open document is an **entity** carrying [`DocumentContent`] and
//! [`DocumentUi`] components. Document entities are children of the singleton
//! [`DocumentRoot`] entity so that `Children` provides stable tab ordering.
//!
//! ## Edit boundary
//!
//! `DocumentContent` fields are private and are only mutated via
//! `pub(crate)` setter methods. The convention is that **only the
//! `apply_edits` system in `crate::edits` calls those setters.** Code
//! review enforces this — the setters are intentionally not public.
//! `DocumentUi` is freely mutable: UI state isn't undoable.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use bevy::prelude::*;
use bevy_hanabi::EffectAsset;
use egui_dock::{DockState, NodeIndex};

use crate::effect_graph::model::EffectGraph;
pub use hanabi_effect_graph::ModifierGroup;

/// Snapshot the node-graph panel's [`GraphView`](hanabi_node_graph::GraphView)
/// (pan/zoom and world positions) into a serializable
/// [`GraphLayout`](hanabi_effect_graph::model::GraphLayout) for saving. Entries
/// are sorted by id so saved files are diff-stable.
pub fn graph_view_to_layout(
    view: &hanabi_node_graph::GraphView,
) -> hanabi_effect_graph::model::GraphLayout {
    use hanabi_effect_graph::model::{GraphLayout, NodeId as MNodeId, StackId as MStackId};

    let mut node_pos: Vec<(MNodeId, (f64, f64))> = view
        .positions
        .iter()
        .filter_map(|(id, p)| MNodeId::new(id.get()).map(|m| (m, (p.x, p.y))))
        .collect();
    node_pos.sort_by_key(|(id, _)| id.get());

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
    }
}

/// Rebuild a [`GraphView`](hanabi_node_graph::GraphView) from a persisted
/// [`GraphLayout`](hanabi_effect_graph::model::GraphLayout). Any node/stack not
/// in the layout is left unplaced for the panel's auto-layout to seed.
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
    view
}

/// Source of process-unique [`DocumentContent::preview_tag`] values. Monotonic
/// and never reused, so two open documents — even ones baked from byte-identical
/// graphs — get distinct preview-asset names (and therefore distinct
/// `hanabi/{name}_…` shader paths), letting shader errors be attributed to the
/// right document.
static NEXT_PREVIEW_TAG: AtomicU64 = AtomicU64::new(1);

/// Allocate a fresh, process-unique preview tag for a new document.
pub fn next_preview_tag() -> u64 {
    NEXT_PREVIEW_TAG.fetch_add(1, Ordering::Relaxed)
}

// ============================================================================
// Components
// ============================================================================

/// Content of a document. Fields are private; mutation goes through
/// `pub(crate)` setters used by `apply_edits` only.
#[derive(Component)]
pub struct DocumentContent {
    name: String,
    path: Option<PathBuf>,
    /// The canonical edit model. The `effect` handle below is a *derived* bake
    /// output of this graph (see [`crate::effect_graph::bake`]).
    graph: EffectGraph,
    effect: Handle<EffectAsset>,
    dirty: bool,
    render_layer: usize,
    /// Process-unique tag baked into the preview asset's name to disambiguate
    /// this document's shaders from other open documents'. See
    /// [`next_preview_tag`].
    preview_tag: u64,
}

impl DocumentContent {
    pub fn new(
        name: String,
        path: Option<PathBuf>,
        graph: EffectGraph,
        effect: Handle<EffectAsset>,
        render_layer: usize,
        preview_tag: u64,
    ) -> Self {
        Self {
            name,
            path,
            graph,
            effect,
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
    pub fn graph(&self) -> &EffectGraph {
        &self.graph
    }
    /// Mutable access to the canonical graph. Only callable from
    /// [`crate::edits::apply_edits`] (the single edit writer).
    pub(crate) fn graph_mut(&mut self) -> &mut EffectGraph {
        &mut self.graph
    }
    pub fn effect(&self) -> &Handle<EffectAsset> {
        &self.effect
    }
    pub fn dirty(&self) -> bool {
        self.dirty
    }
    pub fn render_layer(&self) -> usize {
        self.render_layer
    }
    /// Process-unique preview tag; baked into the preview asset name.
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
}

/// Per-document UI state. Freely mutable — not part of the edit channel.
#[derive(Component)]
pub struct DocumentUi {
    pub dock: DockState<PanelKind>,
    /// Persistable view state for the node-graph panel (pan/zoom/positions).
    pub graph_view: hanabi_node_graph::GraphView,
}

impl Default for DocumentUi {
    fn default() -> Self {
        Self {
            dock: default_dock(),
            graph_view: hanabi_node_graph::GraphView::default(),
        }
    }
}

/// Builds the default per-document dock layout, three columns left-to-right:
/// `[(Viewport on top, Properties on bottom) ≈28% | Graph (Shaders tabbed
/// behind) ≈50% | Effect ≈21%]`. The Viewport is sized to be roughly square;
/// the Graph occupies the widest middle column.
pub fn default_dock() -> DockState<PanelKind> {
    // The middle column hosts the Graph (visible) with the Shaders panel tabbed
    // behind it.
    let mut dock = DockState::new(vec![PanelKind::Graph, PanelKind::Shaders]);
    let surface = dock.main_surface_mut();
    // Left column: Viewport on top (≈ square), Properties below.
    let [middle_node, left_node] =
        surface.split_left(NodeIndex::root(), 0.28, vec![PanelKind::Viewport(0)]);
    surface.split_below(left_node, 0.5, vec![PanelKind::Properties]);
    // Right column: Effect outline.
    surface.split_right(middle_node, 0.7, vec![PanelKind::Effect]);
    dock
}

/// Panel kinds that may appear inside a document's dock area.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PanelKind {
    Viewport(usize),
    /// Outline of the effect: particle layout + modifier groups.
    Effect,
    /// User-defined properties on the effect's `Module`.
    Properties,
    /// Generated WGSL shaders (init / update / render) baked by hanabi.
    Shaders,
    /// Node-graph editor canvas (PoC).
    Graph,
}

/// Component on the per-document camera entity. Stores the local viewport
/// index, the render-target image handle, and the orbit-camera state
/// (target/yaw/pitch/distance). `Transform` is derived from the orbit
/// state by [`crate::plugins::camera_control::apply_camera_controls`].
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
}

/// Marker for the (single) scene root of a document. Children of this
/// entity are the visible scene content (light, mesh, ...).
#[derive(Component)]
pub struct DocumentSceneRoot;

// ============================================================================
// Resources
// ============================================================================

/// The singleton parent entity whose `Children` are the open documents,
/// in user-visible (tab-bar) order.
#[derive(Resource)]
pub struct DocumentRoot(pub Entity);

/// The currently focused document, if any.
#[derive(Resource, Default)]
pub struct ActiveDocument(pub Option<Entity>);

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

/// Cache rebuilt every frame by reconciliation: `(doc_entity, viewport_idx)
/// → Image handle`. Used by the UI to look up the egui texture for each
/// viewport panel.
#[derive(Resource, Default)]
pub struct DocumentViewports {
    pub by_doc: HashMap<Entity, ViewportSlots>,
}

#[derive(Default)]
pub struct ViewportSlots {
    pub images: HashMap<usize, Handle<Image>>,
    pub cameras: HashMap<usize, Entity>,
}

/// Per-viewport desired pixel size, keyed by `(doc_entity, viewport_index)`,
/// written by the UI on render and consumed by the resize-to-fit system.
#[derive(Resource, Default)]
pub struct ViewportSizeRequests(pub HashMap<(Entity, usize), UVec2>);
