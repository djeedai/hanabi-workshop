//! Persistable per-graph view state: pan, zoom, grid, node positions and
//! selection. `GraphView` is serde-serializable; transient interaction
//! bookkeeping is `#[serde(skip)]`.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::transform::WorldPos;
use super::viewer::{Link, NodeId, PortAddr, StackId};

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
            snap: false,
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

/// An in-progress reordering of a stack member: the member node being
/// dragged within its stack, the index it started at, the index it would
/// land at given the current cursor, and the world offset from the node's
/// min corner to the grab point (for the drag ghost).
#[derive(Debug, Clone, Copy)]
pub struct ReorderDrag {
    pub stack: StackId,
    pub node: NodeId,
    pub from_index: usize,
    pub target_index: usize,
    pub grab_offset: WorldPos,
}

/// Transient, per-frame interaction bookkeeping. Never persisted.
#[derive(Debug, Clone, Default)]
pub struct Interaction {
    /// Node currently being dragged, plus the world offset from the node's
    /// min corner to the grab point.
    pub dragging_node: Option<(NodeId, WorldPos)>,
    /// Stack currently being dragged by its header, plus the world offset
    /// from the stack's origin to the grab point.
    pub dragging_stack: Option<(StackId, WorldPos)>,
    /// Stack member currently being dragged to a new position in its stack.
    pub reordering: Option<ReorderDrag>,
    /// Output port a new link is being dragged from.
    pub pending_link_from: Option<PortAddr>,
    /// True when the in-progress link drag was started from an *input* pin
    /// (so it completes by dropping on an output, wiring source → input).
    pub pending_from_input: bool,
    /// Existing link being detached by dragging its input end. When set,
    /// `pending_link_from` carries that link's original output source.
    pub detaching_link: Option<Link>,
    /// Anchor of an in-progress box selection (world space).
    pub box_select_start: Option<WorldPos>,
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
    /// Currently-selected edges. Selected by left-click; removable with
    /// Delete. Transient, like node selection.
    #[serde(skip)]
    pub selected_links: HashSet<Link>,
    #[serde(skip)]
    pub interaction: Interaction,
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
            selected_links: HashSet::new(),
            interaction: Interaction::default(),
        }
    }
}

/// Hard limits on zoom so the transform never degenerates.
pub const MIN_ZOOM: f64 = 0.05;
pub const MAX_ZOOM: f64 = 8.0;

impl GraphView {
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

    /// Clear both node and edge selection. Returns true if anything was
    /// selected before.
    pub fn clear_selection(&mut self) -> bool {
        let had = !self.selection.is_empty() || !self.selected_links.is_empty();
        self.selection.clear();
        self.selected_links.clear();
        had
    }
}
