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
    state::{CanvasItem, GraphView},
    transform::{Transform, WorldPos, WorldRect},
    viewer::{GraphViewer, NodeDesc, NodeId, PortDesc, PortId, PortSide, StackId},
};

pub const NODE_WIDTH: f64 = 204.0;
pub const HEADER_H: f64 = 26.0;
pub const PORT_ROW_H: f64 = 22.0;
pub const BODY_PAD_TOP: f64 = 6.0;
pub const BODY_PAD_BOTTOM: f64 = 8.0;
pub const PORT_RADIUS: f64 = 5.0;
/// Size (square) of the header close button, in world units.
pub const CLOSE_BTN_SIZE: f64 = 14.0;
/// Margin between the close button and the header's right edge, world units.
pub const CLOSE_BTN_MARGIN: f64 = 6.0;
/// Size (square) of a section's collapse/expand chevron, in world units.
pub const TOGGLE_SIZE: f64 = 12.0;
/// Margin between the collapse chevron and the header's left edge, world units.
pub const TOGGLE_MARGIN: f64 = 6.0;
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
/// Size (square) of the stack header's expand/collapse-all button, world units.
pub const STACK_BTN_SIZE: f64 = 16.0;
/// Margin between the collapse-all button and the header's right edge, world.
pub const STACK_BTN_MARGIN: f64 = 6.0;

/// Geometry of a single port.
#[derive(Debug, Clone)]
pub struct PortLayout {
    pub id: PortId,
    pub center: WorldPos,
    pub label: Cow<'static, str>,
    pub color: Option<egui::Color32>,
    /// Number of square markers drawn beside the pin; `0` draws none.
    pub arity: u8,
    /// Inline value chip text, when this port carries an inlined value.
    pub value: Option<Cow<'static, str>>,
    /// Whether this port participates in linking / hit-testing.
    pub connectable: bool,
    /// Full height of this row in world units (a single line, plus any
    /// reserved inline-editor box).
    pub row_height: f64,
    /// Whether this row is a collapsible editor row (chevron + host box).
    pub collapsible: bool,
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
    /// Whether this member is currently collapsed to its header alone, with
    /// every pin folded onto a single header-aligned pin. Always `false` for
    /// free nodes.
    pub collapsed: bool,
    /// The collapse/expand chevron in the top-left of a stacked member's
    /// header. `None` for free nodes and for members with no body to fold.
    pub collapse_toggle: Option<WorldRect>,
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
    /// The collapse/expand-all button at the right of the stack header.
    pub collapse_all_button: WorldRect,
    /// Whether every collapsible member is currently collapsed, so the
    /// collapse-all button can show the matching (expand) affordance.
    pub all_collapsed: bool,
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

/// Height of a node body given its rows' total height.
fn node_height(body: f64) -> f64 {
    HEADER_H + BODY_PAD_TOP + body + BODY_PAD_BOTTOM
}

/// Cumulative row tops and centers for one column of ports.
///
/// Each row is `PORT_ROW_H` tall on its label line, plus any reserved
/// inline-editor height; the label/pin centers on the first line. Returns the
/// per-row `(center_y, row_height)` and the column's total height.
fn column_rows(ports: &[PortDesc], top: f64) -> (Vec<(f64, f64)>, f64) {
    let mut rows = Vec::with_capacity(ports.len());
    let mut y = top;
    for p in ports {
        let row_h = PORT_ROW_H + p.expand_height.unwrap_or(0.0);
        rows.push((y + PORT_ROW_H * 0.5, row_h));
        y += row_h;
    }
    (rows, y - top)
}

/// Build the geometry of one node placed with its min corner at `min`.
fn node_layout(desc: &NodeDesc, min: WorldPos, stack: Option<StackId>) -> NodeLayout {
    let body_top = min.y + HEADER_H + BODY_PAD_TOP;
    let (in_rows, in_total) = column_rows(&desc.inputs, body_top);
    let (out_rows, out_total) = column_rows(&desc.outputs, body_top);
    let body = in_total.max(out_total);
    let rect = WorldRect::new(min, NODE_WIDTH, node_height(body));

    let inputs = desc
        .inputs
        .iter()
        .enumerate()
        .map(|(i, p)| PortLayout {
            id: PortId::input(i as u16),
            center: WorldPos::new(min.x, in_rows[i].0),
            label: p.label.clone(),
            color: p.color,
            arity: p.arity,
            value: p.value.clone(),
            connectable: p.connectable,
            row_height: in_rows[i].1,
            collapsible: p.collapsible,
        })
        .collect();
    let outputs = desc
        .outputs
        .iter()
        .enumerate()
        .map(|(i, p)| PortLayout {
            id: PortId::output(i as u16),
            center: WorldPos::new(min.x + NODE_WIDTH, out_rows[i].0),
            label: p.label.clone(),
            color: p.color,
            arity: p.arity,
            value: p.value.clone(),
            connectable: p.connectable,
            row_height: out_rows[i].1,
            collapsible: p.collapsible,
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
        collapsed: false,
        collapse_toggle: None,
    }
}

/// Fold a stacked member down to its header alone.
///
/// Every input pin collapses onto a single point on the left edge, every output
/// pin onto a single point on the right edge, both vertically centered on the
/// header — so the section reads as one node with one pin per side. The
/// member's real port ids are preserved (sharing the folded center) so existing
/// links still resolve to the collapsed pin.
fn collapse_member(layout: &mut NodeLayout) {
    let min = layout.rect.min;
    let width = layout.rect.width;
    layout.rect = WorldRect::new(min, width, HEADER_H);
    layout.collapsed = true;
    let mid_y = min.y + HEADER_H * 0.5;
    let in_pin = WorldPos::new(min.x, mid_y);
    for p in &mut layout.inputs {
        p.center = in_pin;
    }
    let out_pin = WorldPos::new(min.x + width, mid_y);
    for p in &mut layout.outputs {
        p.center = out_pin;
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

    // Stacks: lay members out top-to-bottom as flat sections that span the
    // full frame width. Members sit flush with the frame's side edges so each
    // member's input/output pins land on the stack's outer border, making the
    // stack read as a single node whose pins live on its edge.
    for s in &stacks_desc {
        let origin = view.stack_position(s.id);
        let member_x = origin.x;
        let mut cursor_y = origin.y + STACK_HEADER_H + STACK_PAD;

        let mut collapsible = 0usize;
        let mut collapsed_count = 0usize;
        for (i, &member) in s.members.iter().enumerate() {
            if i > 0 {
                cursor_y += MEMBER_GAP;
            }
            let desc = viewer.node(member);
            let mut layout = node_layout(&desc, WorldPos::new(member_x, cursor_y), Some(s.id));
            layout.id = member;
            // Members with a body fold/unfold via a header chevron; an empty
            // member (no ports) is already header-only and needs no toggle.
            let has_body = !desc.inputs.is_empty() || !desc.outputs.is_empty();
            if has_body {
                collapsible += 1;
                layout.collapse_toggle = Some(WorldRect::new(
                    WorldPos::new(
                        member_x + TOGGLE_MARGIN,
                        cursor_y + (HEADER_H - TOGGLE_SIZE) * 0.5,
                    ),
                    TOGGLE_SIZE,
                    TOGGLE_SIZE,
                ));
                if view.is_collapsed(member) {
                    collapsed_count += 1;
                    collapse_member(&mut layout);
                }
            }
            cursor_y += layout.rect.height;
            nodes.push(layout);
        }

        let content_h = (cursor_y - origin.y).max(STACK_HEADER_H);
        // An "Add modifier" button sits below the members, inset from the side
        // edges so it doesn't run into the pin column.
        let button_top = origin.y + content_h + STACK_PAD;
        let add_button = WorldRect::new(
            WorldPos::new(member_x + STACK_PAD, button_top),
            NODE_WIDTH - STACK_PAD * 2.0,
            STACK_FOOTER_H,
        );
        let total_h = (button_top + STACK_FOOTER_H + STACK_PAD) - origin.y;
        let rect = WorldRect::new(origin, NODE_WIDTH, total_h);
        let collapse_all_button = WorldRect::new(
            WorldPos::new(
                origin.x + NODE_WIDTH - STACK_BTN_MARGIN - STACK_BTN_SIZE,
                origin.y + (STACK_HEADER_H - STACK_BTN_SIZE) * 0.5,
            ),
            STACK_BTN_SIZE,
            STACK_BTN_SIZE,
        );
        stacks.push(StackLayout {
            id: s.id,
            rect,
            title: s.title.clone(),
            accent: s.accent,
            members: s.members.clone(),
            add_button,
            collapse_all_button,
            all_collapsed: collapsible > 0 && collapsed_count == collapsible,
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

    // Paint/hit precedence follows the persistent z-order: each unit's rank
    // (`z_key`) places it back-to-front, front last, so it renders on top of —
    // and, since interaction hit-tests with `.rev()`, is grabbed in preference
    // to — anything it overlaps. A member ranks with its owning stack, so a
    // stack's members stay contiguous (a stable-sort tie) and move as one unit.
    nodes.sort_by_key(|n| view.z_key(node_item(n)));
    stacks.sort_by_key(|s| view.z_key(CanvasItem::Stack(s.id)));

    GraphLayout { nodes, stacks }
}

/// The canvas unit a node layout belongs to: its owning stack, or itself.
pub fn node_item(node: &NodeLayout) -> CanvasItem {
    match node.stack {
        Some(sid) => CanvasItem::Stack(sid),
        None => CanvasItem::Node(node.id),
    }
}
