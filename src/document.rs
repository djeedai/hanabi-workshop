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

use bevy::prelude::*;
use bevy_hanabi::EffectAsset;
use egui_dock::{DockState, NodeIndex};

// ============================================================================
// Components
// ============================================================================

/// Content of a document. Fields are private; mutation goes through
/// `pub(crate)` setters used by `apply_edits` only.
#[derive(Component)]
pub struct DocumentContent {
    name: String,
    path: Option<PathBuf>,
    effect: Handle<EffectAsset>,
    dirty: bool,
    render_layer: usize,
}

impl DocumentContent {
    pub fn new(
        name: String,
        path: Option<PathBuf>,
        effect: Handle<EffectAsset>,
        render_layer: usize,
    ) -> Self {
        Self {
            name,
            path,
            effect,
            dirty: false,
            render_layer,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn path(&self) -> Option<&std::path::Path> {
        self.path.as_deref()
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
    pub selected_modifier: Option<ModifierSelection>,
}

impl Default for DocumentUi {
    fn default() -> Self {
        Self {
            dock: default_dock(),
            selected_modifier: None,
        }
    }
}

/// Which of the three modifier lists a modifier lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModifierGroup {
    Init,
    Update,
    Render,
}

impl ModifierGroup {
    pub fn label(self) -> &'static str {
        match self {
            ModifierGroup::Init => "Init",
            ModifierGroup::Update => "Update",
            ModifierGroup::Render => "Render",
        }
    }

    /// Lowercase tag used in hanabi's baked shader path
    /// (`hanabi/{asset}_{init|update|render}_{hash}.wgsl`) and as a
    /// stable key for per-group UI state.
    pub fn suffix(self) -> &'static str {
        match self {
            ModifierGroup::Init => "init",
            ModifierGroup::Update => "update",
            ModifierGroup::Render => "render",
        }
    }
}

/// A selection inside the modifier outline: which group, and the index
/// within that group's modifier list at the time the user clicked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModifierSelection {
    pub group: ModifierGroup,
    pub idx: usize,
}

/// Builds the default per-document dock layout:
/// `[(Effect on top, Properties on bottom) 20% | Viewport 60% | (Details +
/// Shaders tabs) 20%]` left-to-right. The left column is split vertically so user
/// properties live below the effect outline; they're typically only
/// a couple per effect so the lower pane is short.
pub fn default_dock() -> DockState<PanelKind> {
    let mut dock = DockState::new(vec![PanelKind::Viewport(0)]);
    let surface = dock.main_surface_mut();
    let [viewport_node, outline_node] =
        surface.split_left(NodeIndex::root(), 0.2, vec![PanelKind::Effect]);
    // Below the Effect outline: a short Properties pane (≈ 25% of the
    // left column).
    surface.split_below(outline_node, 0.75, vec![PanelKind::Properties]);
    surface.split_right(
        viewport_node,
        0.75,
        vec![PanelKind::Details, PanelKind::Shaders],
    );
    dock
}

/// Panel kinds that may appear inside a document's dock area.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PanelKind {
    Viewport(usize),
    /// Details of the currently-selected modifier (formerly the
    /// "Properties" tab — renamed to free the name for the new
    /// user-properties panel).
    Details,
    /// Outline of the effect: particle layout + modifier groups.
    Effect,
    /// User-defined properties on the effect's `Module`.
    Properties,
    /// Generated WGSL shaders (init / update / render) baked by hanabi.
    Shaders,
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
