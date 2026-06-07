//! Interactions the widget reports back to its consumer each frame.
//!
//! The widget never mutates graph topology. View-state changes (pan,
//! zoom, node positions, selection) are applied directly to `GraphView`
//! by the widget; anything that would change the *consumer's* data is
//! surfaced here as a [`GraphAction`] for the consumer to apply (e.g. by
//! emitting an `EditRequest`).

use super::transform::WorldPos;
use super::viewer::{Link, NodeId, PortAddr, StackId};

/// A structural change the consumer may choose to apply.
#[derive(Debug, Clone)]
pub enum GraphAction {
    /// A node finished being dragged to a new world position. (Position
    /// is also already written into `GraphView`; this lets the consumer
    /// react, e.g. mark a sidecar dirty.)
    NodeMoved { node: NodeId, to: WorldPos },
    /// A stack finished being dragged (by its header) to a new world
    /// position. The position is already written into `GraphView`.
    StackMoved { stack: StackId, to: WorldPos },
    /// The selection set changed this frame.
    SelectionChanged,
    /// The user dragged a new link from an output to an input port.
    LinkRequested { from: PortAddr, to: PortAddr },
    /// The user requested deletion of an existing link.
    LinkDeleteRequested { link: Link },
    /// The user requested deletion of the given nodes (e.g. Delete key).
    NodesDeleteRequested { nodes: Vec<NodeId> },
    /// The user requested a context menu at a world position (right-click
    /// on empty canvas).
    ContextMenu { at: WorldPos },
}

/// The widget's return value: the underlying egui response plus the list
/// of actions raised this frame.
pub struct GraphResponse {
    /// The canvas-level egui response (hover/focus/etc.) for consumers
    /// that need it.
    #[allow(dead_code)]
    pub response: egui::Response,
    pub actions: Vec<GraphAction>,
}
