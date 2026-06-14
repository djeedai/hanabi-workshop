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
    /// A stack member was dragged to a new slot within its stack. The
    /// member at `from_index` should move so it ends up at `to_index`
    /// (i.e. remove at `from_index`, then insert at `to_index`).
    StackMemberMoved {
        stack: StackId,
        from_index: usize,
        to_index: usize,
    },
    /// The selection set changed this frame.
    SelectionChanged,
    /// The user dragged a new link from an output to an input port.
    LinkRequested { from: PortAddr, to: PortAddr },
    /// The user requested deletion of an existing link.
    LinkDeleteRequested { link: Link },
    /// The user requested deletion of the given nodes (e.g. Delete key).
    NodesDeleteRequested { nodes: Vec<NodeId> },
    /// The user requested deletion of the given stacks (e.g. Delete key while
    /// stacks are selected). The consumer decides what this means for its
    /// domain (e.g. emptying a fixed pipeline stage vs. removing the container).
    StacksDeleteRequested { stacks: Vec<StackId> },
    /// The user requested a context menu at a world position (right-click
    /// on empty canvas).
    ContextMenu { at: WorldPos },
    /// The user released an in-progress link drag over empty canvas. The
    /// consumer may offer to create a node and wire it to `source`: an output
    /// pin (when `source_is_output`) needing a consumer, or an input pin
    /// awaiting a producer.
    LinkDropped {
        source: PortAddr,
        source_is_output: bool,
        at: WorldPos,
    },
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
