//! Interactions the widget reports back to its consumer each frame.
//!
//! The widget never mutates graph topology. View-state changes (pan,
//! zoom, node positions, selection) are applied directly to `GraphView`
//! by the widget; anything that would change the *consumer's* data is
//! surfaced here as a [`GraphAction`] for the consumer to apply (e.g. by
//! emitting an `EditRequest`).

use super::{
    transform::WorldPos,
    viewer::{FlowLink, Link, NodeId, PortAddr, StackId},
};

/// A structural change the consumer may choose to apply.
#[derive(Debug, Clone)]
pub enum GraphAction {
    /// A node finished being dragged to a new world position. `from` is its
    /// position at grab time. (`to` is also already written into `GraphView`;
    /// this lets the consumer react, e.g. record an undoable move.)
    NodeMoved {
        node: NodeId,
        from: WorldPos,
        to: WorldPos,
    },
    /// A stack finished being dragged (by its header) to a new world
    /// position. `from` is its position at grab time; `to` is already written
    /// into `GraphView`.
    StackMoved {
        stack: StackId,
        from: WorldPos,
        to: WorldPos,
    },
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
    /// The user dragged a new flow link from a node's flow-output pin to a
    /// stack's flow-input pin (or the reverse, from the stack's flow-input
    /// pin out to a node's flow-output pin — the widget always reports it in
    /// node → stack order regardless of which end was grabbed).
    FlowLinkRequested { from: NodeId, to: StackId },
    /// The user requested deletion of an existing flow link.
    FlowLinkDeleteRequested { link: FlowLink },
    /// The user requested deletion of the given nodes (e.g. Delete key).
    NodesDeleteRequested { nodes: Vec<NodeId> },
    /// The user requested deletion of the given stacks (e.g. Delete key while
    /// stacks are selected). The consumer decides what this means for its
    /// domain (e.g. emptying a fixed pipeline stage vs. removing the
    /// container).
    StacksDeleteRequested { stacks: Vec<StackId> },
    /// The user clicked the "Add" button at the bottom of a stack, requesting a
    /// new member be appended to that stack (e.g. via a group-specific menu).
    StackAddRequested { stack: StackId },
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

/// A drawn input value chip and its screen rect.
///
/// Reported so the consumer can overlay a real editor (e.g. a `DragValue`)
/// directly on top of it. The widget stays value-type-agnostic: it only reports
/// *where* each chip is and *which* port it belongs to; the consumer resolves
/// the value's type and decides how to edit it.
#[derive(Debug, Clone, Copy)]
pub struct ChipHit {
    pub port: PortAddr,
    pub rect: egui::Rect,
    /// The (zoom-scaled) font size the chip text was drawn at, so an overlaid
    /// editor can match it instead of using egui's default body font.
    pub font_size: f32,
    /// The chip's inner padding, so an overlaid editor sizes to the same box.
    pub pad: f32,
    /// The owning node's body rect (screen space), so an overlaid editor can be
    /// clipped to the node and not spill past its borders.
    pub clip: egui::Rect,
    /// Screen rect of this row's collapse/expand chevron, when it has one
    /// (a collapsible editor row). The consumer senses clicks on it to toggle.
    pub chevron: Option<egui::Rect>,
    /// Whether this row's collapsible editor is currently expanded. `false` for
    /// a collapsed preview row and for non-collapsible chips.
    pub expanded: bool,
}

/// A graph location suitable for an external drop.
///
/// The target carries only graph geometry and identity. Consumers decide
/// whether their own payload can be applied to it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExternalDropTarget {
    /// An input's inline value row or chip.
    Input(PortAddr),
    /// A node body outside an input value target.
    Node(NodeId),
    /// Empty canvas at the pointer's world position.
    Canvas(WorldPos),
}

/// The widget's return value.
///
/// The underlying egui response, hovered node, and actions raised this frame.
pub struct GraphResponse {
    /// The canvas-level egui response (hover/focus/etc.) for consumers
    /// that need it.
    #[allow(dead_code)]
    pub response: egui::Response,
    /// Topmost node body geometrically under the pointer.
    ///
    /// Includes free nodes and stack members, and remains set while the pointer
    /// is over one of the node's ports or inline value controls.
    pub hovered_node: Option<NodeId>,
    /// External-drop target geometrically under the pointer.
    ///
    /// Input value rows take priority over node bodies, which take priority
    /// over empty canvas. `None` means the pointer is outside the canvas.
    pub external_drop_target: Option<ExternalDropTarget>,
    pub actions: Vec<GraphAction>,
}
