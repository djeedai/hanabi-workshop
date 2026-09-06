//! The data the widget reads from its consumer.
//!
//! The widget never owns graph topology; the consumer implements
//! [`GraphViewer`] to expose nodes, ports and links.

use std::{borrow::Cow, num::NonZeroU32};

use serde::{Deserialize, Serialize};

/// Identifier of a node, one-based.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub NonZeroU32);

impl NodeId {
    /// Construct from a one-based index. Returns `None` for `0`.
    pub fn new(one_based: u32) -> Option<Self> {
        NonZeroU32::new(one_based).map(Self)
    }

    /// The one-based id value.
    pub fn get(self) -> u32 {
        self.0.get()
    }

    /// Zero-based index into a node array.
    #[allow(dead_code)] // public API; used by adapters/consumers
    pub fn index(self) -> usize {
        (self.0.get() - 1) as usize
    }
}

/// Identifier of a stack (ordered node container), one-based.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StackId(pub NonZeroU32);

#[allow(dead_code)]
impl StackId {
    /// Construct from a one-based index. Returns `None` for `0`.
    pub fn new(one_based: u32) -> Option<Self> {
        NonZeroU32::new(one_based).map(Self)
    }

    /// The one-based id value.
    pub fn get(self) -> u32 {
        self.0.get()
    }
}

/// Which side of a node a port sits on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PortSide {
    Input,
    Output,
}

/// Identifies a port within a single node.
///
/// Its side and its index in that side's list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PortId {
    pub side: PortSide,
    pub index: u16,
}

impl PortId {
    pub fn input(index: u16) -> Self {
        Self {
            side: PortSide::Input,
            index,
        }
    }

    pub fn output(index: u16) -> Self {
        Self {
            side: PortSide::Output,
            index,
        }
    }
}

/// A fully-qualified port address: a node plus one of its ports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PortAddr {
    pub node: NodeId,
    pub port: PortId,
}

impl PortAddr {
    pub fn new(node: NodeId, port: PortId) -> Self {
        Self { node, port }
    }
}

/// A directed link from an output port to an input port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Link {
    /// The upstream output port the value flows from.
    pub from: PortAddr,
    /// The downstream input port the value flows into.
    pub to: PortAddr,
}

/// A fixed vertical connection between two stacks.
///
/// Drawn from the bottom edge of `from` to the top edge of `to`. Used to depict
/// the ordered init → update → render particle pipeline; non-interactive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StackLink {
    pub from: StackId,
    pub to: StackId,
}

/// An interactive vertical connection from a node's flow-output pin to a
/// stack's flow-input pin.
///
/// Unlike [`StackLink`], a flow link is user-authored: draggable, selectable
/// and deletable, with its own connection policy delegated to
/// [`GraphViewer::validate_flow_link`]. Used e.g. to depict a spawn source
/// context feeding a modifier stack's Init stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FlowLink {
    /// The upstream node exposing a bottom flow-output pin (see
    /// [`NodeDesc::with_flow_output`]).
    pub from: NodeId,
    /// The downstream stack exposing a top flow-input pin (see
    /// [`StackDesc::with_flow_input`]).
    pub to: StackId,
}

/// Outcome of asking the consumer whether a link is valid.
///
/// `Ok(())` if the connection is allowed, or `Err(reason)` with a short
/// human-readable explanation if not. The widget never inspects *why* a link is
/// valid — any implicit-conversion styling falls out of the ports' own colors,
/// not from this result.
pub type LinkVerdict = Result<(), Cow<'static, str>>;

/// Description of a single port, supplied per-frame by the viewer.
#[derive(Debug, Clone, Default)]
pub struct PortDesc {
    pub label: Cow<'static, str>,
    /// Optional accent used for the port marker (e.g. value-type color).
    pub color: Option<egui::Color32>,
    /// Number of small square markers drawn as a vertical column beside the
    /// pin. The widget is type-agnostic and draws exactly this many squares
    /// (`0` draws none); consumers use it to convey a count such as vector
    /// arity.
    pub arity: u8,
    /// Optional inline value shown as a recessed chip next to the port
    /// (e.g. an inlined literal). When set, the port reads as a value
    /// field rather than a connection target.
    pub value: Option<Cow<'static, str>>,
    /// Whether this row participates in linking / hit-testing. `false` for
    /// read-only display rows (e.g. enum / non-expr modifier fields): they show
    /// a label and value chip but draw no pin and are excluded from
    /// hit-testing.
    pub connectable: bool,
    /// Reserved world-space height of a full-width editor box drawn *below* the
    /// label line for a collapsible row in its expanded state. `None` keeps the
    /// row a single line (collapsed, or not collapsible at all); `Some(h)` adds
    /// an `h`-tall host-painted box under the label, growing the node.
    pub expand_height: Option<f64>,
    /// Whether this row is a collapsible editor row. Such a row draws a small
    /// chevron left of its label (reported back for click-toggling) and hosts a
    /// full-width box the consumer paints a preview or editor into.
    pub collapsible: bool,
    /// Whether this input accepts more than one incoming link (e.g. a GPU
    /// event input fed by several emitters).
    ///
    /// Only meaningful on an input port; ignored on outputs, which already
    /// fan out to any number of links. Grabbing a multiple-link input's pin
    /// always starts a fresh link rather than detaching one of its existing
    /// edges — the ambiguity of picking *which* edge to remove is left
    /// unresolved by design. Existing edges stay individually selectable and
    /// deletable.
    pub accepts_multiple_links: bool,
}

impl PortDesc {
    pub fn new(label: impl Into<Cow<'static, str>>) -> Self {
        Self {
            label: label.into(),
            color: None,
            arity: 0,
            value: None,
            connectable: true,
            expand_height: None,
            collapsible: false,
            accepts_multiple_links: false,
        }
    }

    /// Attach an inline value chip (e.g. an inlined literal expression).
    pub fn with_value(mut self, value: impl Into<Cow<'static, str>>) -> Self {
        self.value = Some(value.into());
        self
    }

    /// A read-only display row: an inline value chip with no pin.
    ///
    /// Excluded from linking (e.g. an enum or other non-expr modifier field).
    pub fn display_value(mut self, value: impl Into<Cow<'static, str>>) -> Self {
        self.value = Some(value.into());
        self.connectable = false;
        self
    }

    /// Reserve a full-width editor box on the line(s) below the label.
    ///
    /// Unlike [`PortDesc::collapsible`] the row keeps its pin (stays
    /// connectable) and shows no chevron: the reserved box is always present
    /// for the consumer to paint a host editor into. `height` is the box's
    /// world-space height below the label line.
    pub fn with_editor_box(mut self, height: f64) -> Self {
        self.value = Some(Cow::Borrowed(""));
        self.expand_height = Some(height);
        self
    }

    /// Reserve a host-painted editor box on a pin-less config row.
    ///
    /// Like [`PortDesc::with_editor_box`] but non-connectable, for a modifier
    /// config field that has no expression pin (e.g. a `CpuValue` vector).
    pub fn display_editor_box(mut self, height: f64) -> Self {
        self.value = Some(Cow::Borrowed(""));
        self.connectable = false;
        self.expand_height = Some(height);
        self
    }

    /// A collapsible host-painted editor row with a chevron toggle.
    ///
    /// The row draws a chevron left of its label and a full-width box the
    /// consumer paints into. When collapsed (`expanded_height` is `None`) the
    /// box is a single preview line; when expanded (`Some(h)`) it grows an
    /// `h`-tall editor box below the label line.
    pub fn collapsible(mut self, expanded_height: Option<f64>) -> Self {
        self.value = Some(Cow::Borrowed(""));
        self.connectable = false;
        self.collapsible = true;
        self.expand_height = expanded_height;
        self
    }

    /// Tint the port marker (e.g. by data type).
    pub fn with_color(mut self, color: egui::Color32) -> Self {
        self.color = Some(color);
        self
    }

    /// Set the number of square markers drawn beside the pin.
    ///
    /// The widget draws exactly `arity` squares (`0` draws none); consumers
    /// pass a count such as vector arity.
    pub fn with_arity(mut self, arity: u8) -> Self {
        self.arity = arity;
        self
    }

    /// Mark an input as accepting more than one incoming link.
    ///
    /// No-op on an output (see [`Self::accepts_multiple_links`]).
    pub fn with_multiple_links(mut self, accepts_multiple_links: bool) -> Self {
        self.accepts_multiple_links = accepts_multiple_links;
        self
    }
}

/// Per-frame description of a node, supplied by the viewer.
///
/// A node is a box with a header and input/output ports, whether it floats
/// freely on the canvas or sits inside a [`StackDesc`].
#[derive(Debug, Clone, Default)]
pub struct NodeDesc {
    pub title: Cow<'static, str>,
    pub inputs: Vec<PortDesc>,
    pub outputs: Vec<PortDesc>,
    /// Optional header accent color.
    pub accent: Option<egui::Color32>,
    /// Optional warning shown as an icon right of the title, with this text
    /// as its hover tooltip (e.g. a shadowed modifier whose writes are dead).
    pub warning: Option<Cow<'static, str>>,
    /// Whether to show a close (✕) button in the header that requests this
    /// node's deletion. Off by default.
    pub closable: bool,
    /// Whether this node exposes an interactive vertical flow-output pin at
    /// the bottom-center of its body (e.g. a spawn source context feeding a
    /// modifier stack's Init stage).
    ///
    /// Distinct from the fixed, non-interactive [`StackLink`] pins: a flow
    /// output is user-draggable, and its links are reported through
    /// [`GraphViewer::flow_links`] and validated through
    /// [`GraphViewer::validate_flow_link`].
    pub flow_output: bool,
}

impl NodeDesc {
    pub fn new(title: impl Into<Cow<'static, str>>) -> Self {
        Self {
            title: title.into(),
            ..Default::default()
        }
    }

    pub fn with_inputs(mut self, inputs: Vec<PortDesc>) -> Self {
        self.inputs = inputs;
        self
    }

    pub fn with_outputs(mut self, outputs: Vec<PortDesc>) -> Self {
        self.outputs = outputs;
        self
    }

    pub fn with_accent(mut self, accent: egui::Color32) -> Self {
        self.accent = Some(accent);
        self
    }

    /// Flag the node with a warning icon and hover tooltip.
    pub fn with_warning(mut self, text: impl Into<Cow<'static, str>>) -> Self {
        self.warning = Some(text.into());
        self
    }

    /// Show a header close (✕) button that requests this node's deletion.
    pub fn closable(mut self) -> Self {
        self.closable = true;
        self
    }

    /// Expose an interactive flow-output pin at the bottom-center of the
    /// node's body.
    pub fn with_flow_output(mut self, flow_output: bool) -> Self {
        self.flow_output = flow_output;
        self
    }
}

/// Per-frame description of a stack: an ordered container of member nodes.
///
/// A stack carries no ports of its own (e.g. a modifier list); its only
/// relationship between members is their order. Member nodes keep their own
/// ports, and links attach to those member ports directly.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct StackDesc {
    pub id: StackId,
    pub title: Cow<'static, str>,
    /// Member nodes, top to bottom in execution order.
    pub members: Vec<NodeId>,
    /// Optional frame/header accent color.
    pub accent: Option<egui::Color32>,
    /// Whether this stack exposes an interactive vertical flow-input pin at
    /// the top-center of its frame (e.g. an Init stack accepting a spawn
    /// source context).
    ///
    /// Distinct from the fixed, non-interactive [`StackLink`] pins: a flow
    /// input is user-draggable, and its links are reported through
    /// [`GraphViewer::flow_links`] and validated through
    /// [`GraphViewer::validate_flow_link`].
    pub flow_input: bool,
}

#[allow(dead_code)]
impl StackDesc {
    pub fn new(id: StackId, title: impl Into<Cow<'static, str>>) -> Self {
        Self {
            id,
            title: title.into(),
            members: Vec::new(),
            accent: None,
            flow_input: false,
        }
    }

    pub fn with_members(mut self, members: Vec<NodeId>) -> Self {
        self.members = members;
        self
    }

    pub fn with_accent(mut self, accent: egui::Color32) -> Self {
        self.accent = Some(accent);
        self
    }

    /// Expose an interactive flow-input pin at the top-center of the stack's
    /// frame.
    pub fn with_flow_input(mut self, flow_input: bool) -> Self {
        self.flow_input = flow_input;
        self
    }
}

/// Implemented by the consumer to expose its graph to the widget.
///
/// The widget reads this every frame; the consumer owns all topology and
/// applies any structural change reported via [`super::GraphResponse`].
pub trait GraphViewer {
    /// All node ids to render, in any order.
    ///
    /// Includes both free nodes and stack members.
    fn node_ids(&self) -> Vec<NodeId>;

    /// Describe a node. Called once per visible node per frame.
    fn node(&self, id: NodeId) -> NodeDesc;

    /// All links to render.
    ///
    /// Links always connect node ports, including the ports of stack-member
    /// nodes.
    fn links(&self) -> Vec<Link>;

    /// Stacks (ordered node containers).
    ///
    /// Any node listed as a stack member is laid out by its stack rather than
    /// as a free node. Defaults to no stacks.
    fn stacks(&self) -> Vec<StackDesc> {
        Vec::new()
    }

    /// Fixed vertical connections between stacks.
    ///
    /// E.g. the init → update → render pipeline. Drawn bottom-to-top; not
    /// selectable or editable. Defaults to none.
    fn stack_links(&self) -> Vec<StackLink> {
        Vec::new()
    }

    /// Interactive vertical flow links to render and hit-test.
    ///
    /// Connects a node's flow-output pin ([`NodeDesc::with_flow_output`]) to a
    /// stack's flow-input pin ([`StackDesc::with_flow_input`]), e.g. a spawn
    /// source context feeding a modifier stack's Init stage. Unlike
    /// [`Self::stack_links`], these are user-authored: draggable, selectable
    /// and deletable. Defaults to none.
    fn flow_links(&self) -> Vec<FlowLink> {
        Vec::new()
    }

    /// Decide whether a link from output `from` to input `to` is valid.
    ///
    /// The widget always passes the output port as `from` and the input port as
    /// `to`, regardless of which end the user grabbed.
    ///
    /// Connection policy is owned entirely by the consumer: type/cast rules,
    /// cycle prevention, and any execution-order constraints between stacked
    /// nodes all live here. Return `Ok(())` to allow the link, or
    /// `Err(reason)` with a short explanation to reject it. The default
    /// permits every connection.
    fn validate_link(&self, _from: PortAddr, _to: PortAddr) -> LinkVerdict {
        Ok(())
    }

    /// Decide whether a flow link from node `from`'s flow-output pin to stack
    /// `to`'s flow-input pin is valid.
    ///
    /// Mirrors [`Self::validate_link`] for the flow-link kind: the widget
    /// itself has no notion of source/effect topology, so the consumer owns
    /// every rule (e.g. a source driving at most one effect). Return `Ok(())`
    /// to allow the link, or `Err(reason)` with a short explanation to
    /// reject it. The default permits every connection.
    fn validate_flow_link(&self, _from: NodeId, _to: StackId) -> LinkVerdict {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal viewer with no nodes/links, used to exercise trait defaults
    /// and a custom `validate_flow_link` override.
    struct RejectingViewer;

    impl GraphViewer for RejectingViewer {
        fn node_ids(&self) -> Vec<NodeId> {
            Vec::new()
        }

        fn node(&self, _id: NodeId) -> NodeDesc {
            NodeDesc::new("node")
        }

        fn links(&self) -> Vec<Link> {
            Vec::new()
        }

        fn validate_flow_link(&self, from: NodeId, _to: StackId) -> LinkVerdict {
            if from.get() == 1 {
                Err(Cow::Borrowed("rejected"))
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn validate_flow_link_default_permits_everything() {
        struct PermissiveViewer;
        impl GraphViewer for PermissiveViewer {
            fn node_ids(&self) -> Vec<NodeId> {
                Vec::new()
            }
            fn node(&self, _id: NodeId) -> NodeDesc {
                NodeDesc::new("node")
            }
            fn links(&self) -> Vec<Link> {
                Vec::new()
            }
        }
        let viewer = PermissiveViewer;
        assert!(
            viewer
                .validate_flow_link(NodeId::new(1).unwrap(), StackId::new(1).unwrap())
                .is_ok()
        );
    }

    #[test]
    fn validate_flow_link_routes_to_consumer_override() {
        let viewer = RejectingViewer;
        assert!(
            viewer
                .validate_flow_link(NodeId::new(1).unwrap(), StackId::new(1).unwrap())
                .is_err()
        );
        assert!(
            viewer
                .validate_flow_link(NodeId::new(2).unwrap(), StackId::new(1).unwrap())
                .is_ok()
        );
    }

    #[test]
    fn node_desc_with_flow_output_builder() {
        let desc = NodeDesc::new("n").with_flow_output(true);
        assert!(desc.flow_output);
        let desc = NodeDesc::new("n").with_flow_output(false);
        assert!(!desc.flow_output);
    }

    #[test]
    fn stack_desc_with_flow_input_builder() {
        let desc = StackDesc::new(StackId::new(1).unwrap(), "s").with_flow_input(true);
        assert!(desc.flow_input);
    }

    #[test]
    fn port_desc_with_multiple_links_builder() {
        let port = PortDesc::new("p").with_multiple_links(true);
        assert!(port.accepts_multiple_links);
        let port = PortDesc::new("p");
        assert!(!port.accepts_multiple_links);
    }
}
