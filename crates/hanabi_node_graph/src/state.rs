//! Persistable per-graph view state.
//!
//! Pan, zoom, grid, node positions and selection. `GraphView` is
//! serde-serializable; transient interaction bookkeeping is `#[serde(skip)]`.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::{
    transform::WorldPos,
    viewer::{FlowLink, Link, NodeId, PortAddr, StackId},
};

/// Max hold time for a secondary press to count as a right-click, not a pan.
///
/// Opens the context menu. Distance can't be used: a trackpad two-finger tap
/// jumps the pointer between the two touch points, which egui — and any
/// distance check — reads as movement, not a click.
pub const RIGHT_CLICK_MAX_SECS: f64 = 0.35;

/// A movable canvas unit: a free node or a whole stack.
///
/// The paint/hit z-order is expressed over these units, so a stack (its frame
/// and every member) rises and falls as one and never interleaves with another
/// unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CanvasItem {
    Node(NodeId),
    Stack(StackId),
}

/// Grid configuration for the canvas background and snapping.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GridConfig {
    /// Draw the background grid.
    pub enabled: bool,
    /// Snap node drags to grid intersections.
    pub snap: bool,
    /// World-space spacing between minor grid lines.
    pub spacing: f64,
    /// Number of minor cells per major (emphasized) line.
    pub major_every: u32,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            snap: true,
            spacing: 24.0,
            major_every: 5,
        }
    }
}

impl GridConfig {
    /// Snap a world position to the nearest grid intersection.
    pub fn snap_pos(&self, p: WorldPos) -> WorldPos {
        let s = self.spacing.max(f64::EPSILON);
        (p / s).round() * s
    }
}

/// An in-progress reordering of a stack member.
///
/// The member node being dragged within its stack, the index it started at, the
/// index it would land at given the current cursor, and the world offset from
/// the node's min corner to the grab point (for the drag ghost).
#[derive(Debug, Clone, Copy)]
pub struct ReorderDrag {
    pub stack: StackId,
    pub node: NodeId,
    pub from_index: usize,
    pub target_index: usize,
    pub grab_offset: WorldPos,
}

/// The grabbed item that anchors a [`CanvasDrag`].
///
/// Used for grid snapping: the primary item snaps and the rest of the selection
/// follows rigidly.
#[derive(Debug, Clone, Copy)]
pub enum DragItem {
    Node(NodeId),
    Stack(StackId),
}

/// The endpoint anchoring an in-progress flow-link drag.
///
/// Either a node's flow-output pin, or a stack's flow-input pin — whichever end
/// the user grabbed first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FlowAnchor {
    Node(NodeId),
    Stack(StackId),
}

/// An in-progress free move of the canvas selection.
///
/// Every selected free node and stack translates together by the same delta.
/// Captures each item's origin at grab time so the move stays rigid regardless
/// of snapping.
#[derive(Debug, Clone)]
pub struct CanvasDrag {
    /// The grabbed item's origin at grab time; its snapped position drives the
    /// group delta.
    pub primary_origin: WorldPos,
    /// World offset from the primary item's origin to the grab point.
    pub grab_offset: WorldPos,
    /// Origins of every dragged free node, captured at grab time.
    pub nodes: Vec<(NodeId, WorldPos)>,
    /// Origins of every dragged stack, captured at grab time.
    pub stacks: Vec<(StackId, WorldPos)>,
}

/// Transient, per-frame interaction bookkeeping.
///
/// Never persisted.
#[derive(Debug, Clone, Default)]
pub struct Interaction {
    /// Free move of the canvas selection (free nodes + stacks) in progress.
    pub canvas_drag: Option<CanvasDrag>,
    /// Stack member currently being dragged to a new position in its stack.
    pub reordering: Option<ReorderDrag>,
    /// Output port a new link is being dragged from.
    pub pending_link_from: Option<PortAddr>,
    /// True when the in-progress link drag was started from an *input* pin
    /// (so it completes by dropping on an output, wiring source → input).
    pub pending_from_input: bool,
    /// Existing link being detached by dragging its input end. When set,
    /// `pending_link_from` carries that link's original output source.
    ///
    /// Never set when the grabbed input accepts multiple links
    /// ([`PortDesc::with_multiple_links`](super::viewer::PortDesc::with_multiple_links)):
    /// grabbing such an input always starts a fresh link instead of
    /// detaching one of its existing edges.
    pub detaching_link: Option<Link>,
    /// Endpoint a new flow link is being dragged from (a node's flow-output
    /// pin, or a stack's flow-input pin).
    pub pending_flow_link_from: Option<FlowAnchor>,
    /// Existing flow link being detached by dragging its stack (flow-input)
    /// end. When set, `pending_flow_link_from` carries that link's original
    /// node (flow-output) source.
    pub detaching_flow_link: Option<FlowLink>,
    /// Anchor of an in-progress box selection (world space).
    pub box_select_start: Option<WorldPos>,
    /// Screen position and time of the last secondary-button press over the
    /// canvas. Used to recognise a brief tap as a right-click (egui clears its
    /// own `press_start_time` on release, so we capture the time ourselves).
    pub secondary_press: Option<(egui::Pos2, f64)>,
}

/// All persistable view state for one graph canvas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphView {
    /// World coordinate shown at the canvas top-left.
    pub pan: WorldPos,
    /// Screen pixels per world unit.
    pub zoom: f64,
    pub grid: GridConfig,
    /// Free-node min-corner positions in world space. Stack members derive
    /// their position from their stack instead.
    pub positions: HashMap<NodeId, WorldPos>,
    /// Stack min-corner positions in world space.
    pub stack_positions: HashMap<StackId, WorldPos>,
    #[serde(skip)]
    pub selection: HashSet<NodeId>,
    /// Currently-selected stacks. Stacks are canvas-movable units like free
    /// nodes; transient, like node selection.
    #[serde(skip)]
    pub selected_stacks: HashSet<StackId>,
    /// Currently-selected edges. Selected by left-click; removable with
    /// Delete. Transient, like node selection.
    #[serde(skip)]
    pub selected_links: HashSet<Link>,
    /// Currently-selected flow links. Selected by left-click; removable with
    /// Delete. Transient, like node selection.
    #[serde(skip)]
    pub selected_flow_links: HashSet<FlowLink>,
    /// Stack members the user has collapsed to a single header-aligned pin.
    ///
    /// Transient, like selection: toggled by the section-header chevron and
    /// not persisted with the file.
    #[serde(skip)]
    pub collapsed: HashSet<NodeId>,
    /// Back-to-front paint order of canvas units. Later entries paint on top
    /// and win overlapping hit-tests; pressing a unit raises it to the end, so
    /// a click brings a node or stack to the front for good. Units absent here
    /// (never raised) sort behind every listed one, in layout order. Persisted
    /// so the arrangement survives save/load.
    #[serde(default)]
    pub z_order: Vec<CanvasItem>,
    #[serde(skip)]
    pub interaction: Interaction,
    /// Last rendered viewport size, used to preserve its center when resized.
    #[serde(skip)]
    last_viewport_size: Option<egui::Vec2>,
}

impl Default for GraphView {
    fn default() -> Self {
        Self {
            pan: WorldPos::ZERO,
            zoom: 1.0,
            grid: GridConfig::default(),
            positions: HashMap::new(),
            stack_positions: HashMap::new(),
            selection: HashSet::new(),
            selected_stacks: HashSet::new(),
            selected_links: HashSet::new(),
            selected_flow_links: HashSet::new(),
            collapsed: HashSet::new(),
            z_order: Vec::new(),
            interaction: Interaction::default(),
            last_viewport_size: None,
        }
    }
}

/// Hard limits on zoom so the transform never degenerates.
pub const MIN_ZOOM: f64 = 0.05;
pub const MAX_ZOOM: f64 = 2.0;

impl GraphView {
    /// Record a viewport size while preserving its world-space center.
    ///
    /// The first size establishes a baseline. Later size changes adjust
    /// [`pan`] so added or removed space is distributed equally across opposite
    /// canvas edges.
    ///
    /// [`pan`]: Self::pan
    pub fn update_viewport_size(&mut self, size: egui::Vec2) {
        if size.x <= 0.0 || size.y <= 0.0 {
            return;
        }
        let Some(previous) = self.last_viewport_size.replace(size) else {
            return;
        };
        let delta = previous - size;
        self.pan += WorldPos::new(delta.x as f64, delta.y as f64) / (2.0 * self.zoom);
    }

    /// Position of a node, defaulting to the origin if unknown.
    pub fn position(&self, id: NodeId) -> WorldPos {
        self.positions.get(&id).copied().unwrap_or(WorldPos::ZERO)
    }

    /// Ensure a node has a stored position, seeding `default` if absent.
    pub fn ensure_position(&mut self, id: NodeId, default: WorldPos) -> WorldPos {
        *self.positions.entry(id).or_insert(default)
    }

    /// Position of a stack, defaulting to the origin if unknown.
    pub fn stack_position(&self, id: StackId) -> WorldPos {
        self.stack_positions
            .get(&id)
            .copied()
            .unwrap_or(WorldPos::ZERO)
    }

    /// Ensure a stack has a stored position, seeding `default` if absent.
    #[allow(dead_code)]
    pub fn ensure_stack_position(&mut self, id: StackId, default: WorldPos) -> WorldPos {
        *self.stack_positions.entry(id).or_insert(default)
    }

    pub fn set_zoom_clamped(&mut self, zoom: f64) {
        self.zoom = zoom.clamp(MIN_ZOOM, MAX_ZOOM);
    }

    /// Whether a stack member is currently collapsed.
    pub fn is_collapsed(&self, id: NodeId) -> bool {
        self.collapsed.contains(&id)
    }

    /// Toggle a stack member between collapsed and expanded.
    pub fn toggle_collapsed(&mut self, id: NodeId) {
        if !self.collapsed.remove(&id) {
            self.collapsed.insert(id);
        }
    }

    /// Raise a canvas unit to the front of the persistent z-order.
    ///
    /// A no-op if the unit is already frontmost. Called when a node or stack is
    /// pressed so it pops in front of everything it overlaps and stays there.
    pub fn raise(&mut self, item: CanvasItem) {
        if self.z_order.last() == Some(&item) {
            return;
        }
        self.z_order.retain(|i| *i != item);
        self.z_order.push(item);
    }

    /// Sort key placing a canvas unit in back-to-front paint order.
    ///
    /// Listed (previously raised) units order by their position, front last;
    /// unlisted units share the lowest key so they sit behind every listed one
    /// and, being a stable-sort tie, keep their relative layout order.
    pub fn z_key(&self, item: CanvasItem) -> (u8, usize) {
        match self.z_order.iter().position(|i| *i == item) {
            Some(pos) => (1, pos),
            None => (0, 0),
        }
    }

    /// Clear node, stack and edge selection.
    ///
    /// Returns true if anything was selected before.
    pub fn clear_selection(&mut self) -> bool {
        let had = !self.selection.is_empty()
            || !self.selected_stacks.is_empty()
            || !self.selected_links.is_empty()
            || !self.selected_flow_links.is_empty();
        self.selection.clear();
        self.selected_stacks.clear();
        self.selected_links.clear();
        self.selected_flow_links.clear();
        had
    }
}
