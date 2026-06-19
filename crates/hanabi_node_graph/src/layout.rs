//! Per-frame node/port geometry, computed entirely in world space.
//!
//! Layout is independent of pan/zoom — only node/stack positions (from
//! `GraphView`) and the viewer's port counts matter. Screen conversion
//! happens at render/hit-test time.
//!
//! Free nodes take their position straight from `GraphView`. Stack members
//! are positioned by their stack: stacked top-to-bottom in order, inside a
//! container frame whose origin is the stack's stored position.

use std::{borrow::Cow, collections::HashMap};

use super::{
    state::GraphView,
    transform::{Transform, WorldPos, WorldRect},
    viewer::{GraphViewer, NodeDesc, NodeId, PortId, PortSide, StackId},
};

pub const NODE_WIDTH: f64 = 170.0;
pub const HEADER_H: f64 = 26.0;
pub const PORT_ROW_H: f64 = 22.0;
pub const BODY_PAD_TOP: f64 = 6.0;
pub const BODY_PAD_BOTTOM: f64 = 8.0;
pub const PORT_RADIUS: f64 = 5.0;
/// Size (square) of the header close button, in world units.
pub const CLOSE_BTN_SIZE: f64 = 14.0;
/// Margin between the close button and the header's right edge, world units.
pub const CLOSE_BTN_MARGIN: f64 = 6.0;
/// Pick/grab tolerance around a port center, wider than the drawn pin.
///
/// Makes ports easy to grab. Also the radius of the hover highlight.
pub const PORT_GRAB_RADIUS: f64 = PORT_RADIUS * 1.8;
/// Screen-space clamp (px) applied to the grab tolerance.
///
/// Keeps ports easy to hit at any zoom. The hover highlight and the hit-test
/// share this, keeping the visible ring and the clickable area identical.
pub const PORT_GRAB_MIN_PX: f32 = 4.0;
pub const PORT_GRAB_MAX_PX: f32 = 18.0;

/// Grab tolerance in screen pixels at the current zoom.
///
/// The radius of the hover highlight ring.
pub fn port_grab_radius_screen(t: &Transform) -> f32 {
    t.world_len_to_screen(PORT_GRAB_RADIUS)
        .clamp(PORT_GRAB_MIN_PX, PORT_GRAB_MAX_PX)
}

/// The same tolerance expressed in world units.
///
/// Hit-testing in world space then matches the on-screen highlight regardless
/// of zoom.
pub fn port_grab_radius_world(t: &Transform) -> f64 {
    t.screen_len_to_world(port_grab_radius_screen(t))
}

/// Title-bar height of a stack frame.
pub const STACK_HEADER_H: f64 = 24.0;
/// Inner padding between a stack frame and its members.
pub const STACK_PAD: f64 = 8.0;
/// Vertical gap between consecutive stack members.
pub const MEMBER_GAP: f64 = 6.0;
/// Height of the "Add" button row at the bottom of a stack frame.
pub const STACK_FOOTER_H: f64 = 20.0;

/// Geometry of a single port.
#[derive(Debug, Clone)]
pub struct PortLayout {
    pub id: PortId,
    pub center: WorldPos,
    pub label: Cow<'static, str>,
    pub color: Option<egui::Color32>,
    /// Inline value chip text, when this port carries an inlined value.
    pub value: Option<Cow<'static, str>>,
    /// Whether this port participates in linking / hit-testing.
    pub connectable: bool,
}

/// Geometry of a single node and its ports.
#[derive(Debug, Clone)]
pub struct NodeLayout {
    pub id: NodeId,
    pub rect: WorldRect,
    pub title: Cow<'static, str>,
    pub accent: Option<egui::Color32>,
    pub inputs: Vec<PortLayout>,
    pub outputs: Vec<PortLayout>,
    /// `Some` when this node is a member of a stack (laid out by it and not
    /// free-draggable); `None` for a free node.
    pub stack: Option<StackId>,
    /// Optional warning tooltip text, shown via an icon right of the title.
    pub warning: Option<Cow<'static, str>>,
    /// The close (✕) button in the top-right of the header, when the node
    /// opted into one ([`NodeDesc::closable`]).
    pub close_button: Option<WorldRect>,
}

impl NodeLayout {
    /// Look up a port's center by id.
    pub fn port_center(&self, port: PortId) -> Option<WorldPos> {
        let list = match port.side {
            PortSide::Input => &self.inputs,
            PortSide::Output => &self.outputs,
        };
        list.iter().find(|p| p.id == port).map(|p| p.center)
    }

    /// Look up a port's accent color by id (e.g. its data-type color).
    pub fn port_color(&self, port: PortId) -> Option<egui::Color32> {
        let list = match port.side {
            PortSide::Input => &self.inputs,
            PortSide::Output => &self.outputs,
        };
        list.iter().find(|p| p.id == port).and_then(|p| p.color)
    }
}

/// Geometry of a stack frame (an ordered node container).
#[derive(Debug, Clone)]
pub struct StackLayout {
    #[allow(dead_code)]
    pub id: StackId,
    pub rect: WorldRect,
    pub title: Cow<'static, str>,
    pub accent: Option<egui::Color32>,
    /// Member node ids, top to bottom in order.
    pub members: Vec<NodeId>,
    /// The "Add modifier" button row at the bottom of the frame.
    pub add_button: WorldRect,
}

impl StackLayout {
    /// Center of the stack's top edge — the inbound pipeline pin.
    pub fn top_pin(&self) -> WorldPos {
        WorldPos::new(self.rect.center().x, self.rect.min.y)
    }

    /// Center of the stack's bottom edge — the outbound pipeline pin.
    pub fn bottom_pin(&self) -> WorldPos {
        WorldPos::new(self.rect.center().x, self.rect.max().y)
    }
}

/// Everything the widget needs to render and hit-test one frame.
#[derive(Debug, Clone, Default)]
pub struct GraphLayout {
    /// All nodes (free and stacked members), each carrying its membership.
    pub nodes: Vec<NodeLayout>,
    /// Stack container frames.
    pub stacks: Vec<StackLayout>,
}

/// Height of a node body given its row count.
fn node_height(rows: usize) -> f64 {
    HEADER_H + BODY_PAD_TOP + rows as f64 * PORT_ROW_H + BODY_PAD_BOTTOM
}

/// Build the geometry of one node placed with its min corner at `min`.
fn node_layout(desc: &NodeDesc, min: WorldPos, stack: Option<StackId>) -> NodeLayout {
    let rows = desc.inputs.len().max(desc.outputs.len());
    let rect = WorldRect::new(min, NODE_WIDTH, node_height(rows));
    let port_y = |i: usize| min.y + HEADER_H + BODY_PAD_TOP + (i as f64 + 0.5) * PORT_ROW_H;

    let inputs = desc
        .inputs
        .iter()
        .enumerate()
        .map(|(i, p)| PortLayout {
            id: PortId::input(i as u16),
            center: WorldPos::new(min.x, port_y(i)),
            label: p.label.clone(),
            color: p.color,
            value: p.value.clone(),
            connectable: p.connectable,
        })
        .collect();
    let outputs = desc
        .outputs
        .iter()
        .enumerate()
        .map(|(i, p)| PortLayout {
            id: PortId::output(i as u16),
            center: WorldPos::new(min.x + NODE_WIDTH, port_y(i)),
            label: p.label.clone(),
            color: p.color,
            value: p.value.clone(),
            connectable: p.connectable,
        })
        .collect();

    NodeLayout {
        id: NodeId::new(1).unwrap(), // overwritten by caller
        rect,
        title: desc.title.clone(),
        accent: desc.accent,
        inputs,
        outputs,
        stack,
        warning: desc.warning.clone(),
        close_button: desc.closable.then(|| {
            WorldRect::new(
                WorldPos::new(
                    min.x + NODE_WIDTH - CLOSE_BTN_MARGIN - CLOSE_BTN_SIZE,
                    min.y + (HEADER_H - CLOSE_BTN_SIZE) * 0.5,
                ),
                CLOSE_BTN_SIZE,
                CLOSE_BTN_SIZE,
            )
        }),
    }
}

/// Compute geometry for every node and stack the viewer exposes.
pub fn compute(viewer: &dyn GraphViewer, view: &GraphView) -> GraphLayout {
    let stacks_desc = viewer.stacks();

    // Map each member node to its owning stack, so the free-node pass can
    // skip nodes that a stack lays out.
    let mut membership: HashMap<NodeId, StackId> = HashMap::new();
    for s in &stacks_desc {
        for &m in &s.members {
            membership.insert(m, s.id);
        }
    }

    let mut nodes = Vec::new();
    let mut stacks = Vec::new();

    // Stacks: lay members out top-to-bottom inside a container frame.
    for s in &stacks_desc {
        let origin = view.stack_position(s.id);
        let member_x = origin.x + STACK_PAD;
        let mut cursor_y = origin.y + STACK_HEADER_H + STACK_PAD;

        for (i, &member) in s.members.iter().enumerate() {
            if i > 0 {
                cursor_y += MEMBER_GAP;
            }
            let desc = viewer.node(member);
            let mut layout = node_layout(&desc, WorldPos::new(member_x, cursor_y), Some(s.id));
            layout.id = member;
            cursor_y += layout.rect.height;
            nodes.push(layout);
        }

        let content_h = (cursor_y - origin.y).max(STACK_HEADER_H);
        // A full-width "Add modifier" button sits below the members.
        let button_top = origin.y + content_h + STACK_PAD;
        let add_button = WorldRect::new(
            WorldPos::new(member_x, button_top),
            NODE_WIDTH,
            STACK_FOOTER_H,
        );
        let total_h = (button_top + STACK_FOOTER_H + STACK_PAD) - origin.y;
        let rect = WorldRect::new(origin, NODE_WIDTH + STACK_PAD * 2.0, total_h);
        stacks.push(StackLayout {
            id: s.id,
            rect,
            title: s.title.clone(),
            accent: s.accent,
            members: s.members.clone(),
            add_button,
        });
    }

    // Free nodes: everything not claimed by a stack.
    for id in viewer.node_ids() {
        if membership.contains_key(&id) {
            continue;
        }
        let desc = viewer.node(id);
        let mut layout = node_layout(&desc, view.position(id), None);
        layout.id = id;
        nodes.push(layout);
    }

    GraphLayout { nodes, stacks }
}
