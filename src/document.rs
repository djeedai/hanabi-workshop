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

use bevy::prelude::*;
use bevy_hanabi::EffectAsset;
use egui_dock::{DockState, NodeIndex};
use std::collections::HashMap;
use std::path::PathBuf;

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
    pub selected_modifier: Option<usize>,
}

impl Default for DocumentUi {
    fn default() -> Self {
        Self {
            dock: default_dock(),
            selected_modifier: None,
        }
    }
}

/// Builds the default per-document dock layout.
pub fn default_dock() -> DockState<PanelKind> {
    let mut dock = DockState::new(vec![PanelKind::Viewport(0)]);
    let surface = dock.main_surface_mut();
    let [_left, center] = surface.split_left(NodeIndex::root(), 0.25, vec![PanelKind::Outline]);
    surface.split_right(center, 0.7, vec![PanelKind::Properties]);
    dock
}

/// Panel kinds that may appear inside a document's dock area.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PanelKind {
    Viewport(usize),
    Properties,
    Outline,
}

/// Component on the per-document camera entity. Stores the local viewport
/// index and the render-target image handle.
#[derive(Component)]
pub struct ViewportCamera {
    pub viewport_index: usize,
    pub image: Handle<Image>,
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
