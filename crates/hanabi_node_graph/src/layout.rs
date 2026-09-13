//! Per-frame node/section/container geometry, computed entirely in world space.
//!
//! Layout is independent of pan/zoom — only node/container positions (from
//! `GraphView`), the folded-section set and the viewer's port counts matter.
//! Screen conversion happens at render/hit-test time.
//!
//! Free nodes take their position straight from `GraphView`. Section members
//! are positioned by their container: sections stack top-to-bottom inside one
//! frame whose origin is the container's stored position, and each section's
//! members stack top-to-bottom in order beneath its header.

use std::{borrow::Cow, collections::HashMap};

use super::{
    state::{CanvasItem, GraphView},
    transform::{Transform, WorldPos, WorldRect},
    viewer::{ContainerId, GraphViewer, NodeDesc, NodeId, PortDesc, PortId, PortSide, SectionId},
};

pub const NODE_WIDTH: f64 = 204.0;
/// Width of a container and each of its stacked member nodes.
pub const CONTAINER_WIDTH: f64 = 300.0;
pub const HEADER_H: f64 = 26.0;
pub const PORT_ROW_H: f64 = 22.0;
/// Width reserved for field names in stacked member rows.
pub const MEMBER_FIELD_WIDTH: f64 = 120.0;
pub const BODY_PAD_TOP: f64 = 6.0;
pub const BODY_PAD_BOTTOM: f64 = 8.0;
pub const PORT_RADIUS: f64 = 5.0;
/// Size (square) of the header close button, in world units.
pub const CLOSE_BTN_SIZE: f64 = 14.0;
/// Margin between the close button and the header's right edge, world units.
pub const CLOSE_BTN_MARGIN: f64 = 6.0;
/// Size (square) of a member's collapse/expand chevron, in world units.
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

/// Title-bar height of a container frame.
pub const CONTAINER_HEADER_H: f64 = 28.0;
/// Padding below a container's last section, in world units.
pub const CONTAINER_PAD_BOTTOM: f64 = 6.0;
/// Title-bar height of a section header inside a container.
pub const SECTION_HEADER_H: f64 = 24.0;
/// Inner padding between a section's header/footer and its members.
pub const SECTION_PAD: f64 = 8.0;
/// Vertical gap between consecutive section members.
pub const MEMBER_GAP: f64 = 0.0;
/// Height of the "Add" button row at the bottom of a section.
pub const SECTION_FOOTER_H: f64 = 20.0;
/// Size (square) of a section header's button (chevron, collapse-all).
pub const SECTION_BTN_SIZE: f64 = 16.0;
/// Margin between a section header's buttons and its edges, world units.
pub const SECTION_BTN_MARGIN: f64 = 6.0;

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
    /// Whether the value chip fills the rest of its label line.
    pub fill_value_width: bool,
    /// Whether this input accepts more than one incoming link
    /// ([`PortDesc::with_multiple_links`]).
    pub accepts_multiple_links: bool,
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
    /// `Some` when this node is a member of a container section (laid out by
    /// it and not free-draggable); `None` for a free node.
    pub section: Option<SectionId>,
    /// The container owning this node's section, when it has one.
    pub container: Option<ContainerId>,
    /// Optional warning tooltip text, shown via an icon right of the title.
    pub warning: Option<Cow<'static, str>>,
    /// The close (✕) button in the top-right of the header, when the node
    /// opted into one ([`NodeDesc::closable`]).
    pub close_button: Option<WorldRect>,
    /// Whether this member is currently collapsed to its header alone, with
    /// every pin folded onto a single header-aligned pin. Always `false` for
    /// free nodes.
    pub collapsed: bool,
    /// Whether this member belongs to a folded section and so is not drawn at
    /// all. Its ports still resolve, folded onto the section's edge anchors,
    /// so links into a folded section stay visible.
    pub hidden: bool,
    /// The collapse/expand chevron in the top-left of a member's header.
    /// `None` for free nodes and for members with no body to fold.
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

    /// Whether this node can be grabbed, hovered or hit-tested at all.
    ///
    /// A member of a folded section is geometry-only: its ports anchor links
    /// on the section boundary, but nothing about it is interactive until the
    /// section is expanded again.
    pub fn interactive(&self) -> bool {
        !self.hidden
    }
}

/// Geometry of one collapsible section inside a container frame.
#[derive(Debug, Clone)]
pub struct SectionLayout {
    pub id: SectionId,
    /// The container this section belongs to.
    pub container: ContainerId,
    /// The section's full extent, header included.
    pub rect: WorldRect,
    /// The section's header band.
    pub header: WorldRect,
    pub title: Cow<'static, str>,
    pub accent: Option<egui::Color32>,
    /// Member node ids, top to bottom in order.
    pub members: Vec<NodeId>,
    /// Whether the section is folded to its header alone.
    pub collapsed: bool,
    /// The fold/unfold chevron at the header's left edge.
    pub toggle: WorldRect,
    /// The "Add member" button row at the bottom of an expanded section that
    /// opted into one ([`SectionDesc::can_add_member`]).
    ///
    /// [`SectionDesc::can_add_member`]: super::viewer::SectionDesc::can_add_member
    pub add_button: Option<WorldRect>,
    /// The collapse/expand-all-members button at the header's right edge,
    /// present while the section is expanded and has foldable members.
    pub collapse_all_button: Option<WorldRect>,
    /// Whether every collapsible member is currently collapsed, so the
    /// collapse-all button can show the matching (expand) affordance.
    pub all_collapsed: bool,
    /// Optional warning tooltip text, shown via an icon right of the title.
    pub warning: Option<Cow<'static, str>>,
    /// Whether a folded section hides any member input pin, so an aggregate
    /// anchor is drawn on its left edge.
    pub folds_inputs: bool,
    /// Whether a folded section hides any member output pin, so an aggregate
    /// anchor is drawn on its right edge.
    pub folds_outputs: bool,
}

impl SectionLayout {
    /// Left-edge anchor every hidden member input pin folds onto.
    pub fn fold_input_anchor(&self) -> WorldPos {
        WorldPos::new(self.rect.min.x, self.header.center().y)
    }

    /// Right-edge anchor every hidden member output pin folds onto.
    pub fn fold_output_anchor(&self) -> WorldPos {
        WorldPos::new(self.rect.max().x, self.header.center().y)
    }
}

/// Geometry of a container frame: the movable pipeline unit.
#[derive(Debug, Clone)]
pub struct ContainerLayout {
    pub id: ContainerId,
    pub rect: WorldRect,
    /// The container's own title band, above its first section.
    pub header: WorldRect,
    pub title: Cow<'static, str>,
    pub accent: Option<egui::Color32>,
    /// The close (✕) button in the container header, when it opted into one.
    pub close_button: Option<WorldRect>,
    /// Optional warning tooltip text, shown via an icon right of the title.
    pub warning: Option<Cow<'static, str>>,
    /// Sections, top to bottom in order.
    pub sections: Vec<SectionLayout>,
}

/// Everything the widget needs to render and hit-test one frame.
#[derive(Debug, Clone, Default)]
pub struct GraphLayout {
    /// All nodes (free nodes and section members), each carrying its
    /// membership.
    pub nodes: Vec<NodeLayout>,
    /// Container frames.
    pub containers: Vec<ContainerLayout>,
}

impl GraphLayout {
    /// Every section of every container, in paint order.
    pub fn sections(&self) -> impl Iterator<Item = &SectionLayout> {
        self.containers.iter().flat_map(|c| c.sections.iter())
    }

    /// The section with the given id, if it is laid out this frame.
    #[cfg(test)]
    pub fn section(&self, id: SectionId) -> Option<&SectionLayout> {
        self.sections().find(|s| s.id == id)
    }

    /// The container with the given id, if it is laid out this frame.
    #[cfg(test)]
    pub fn container(&self, id: ContainerId) -> Option<&ContainerLayout> {
        self.containers.iter().find(|c| c.id == id)
    }
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
fn node_layout(
    desc: &NodeDesc,
    min: WorldPos,
    width: f64,
    section: Option<SectionId>,
) -> NodeLayout {
    let body_top = min.y + HEADER_H + BODY_PAD_TOP;
    let (in_rows, in_total) = column_rows(&desc.inputs, body_top);
    let (out_rows, out_total) = column_rows(&desc.outputs, body_top);
    let body = in_total.max(out_total);
    let rect = WorldRect::new(min, width, node_height(body));

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
            fill_value_width: p.fill_value_width,
            accepts_multiple_links: p.accepts_multiple_links,
        })
        .collect();
    let outputs = desc
        .outputs
        .iter()
        .enumerate()
        .map(|(i, p)| PortLayout {
            id: PortId::output(i as u16),
            center: WorldPos::new(min.x + width, out_rows[i].0),
            label: p.label.clone(),
            color: p.color,
            arity: p.arity,
            value: p.value.clone(),
            connectable: p.connectable,
            row_height: out_rows[i].1,
            collapsible: p.collapsible,
            fill_value_width: p.fill_value_width,
            accepts_multiple_links: p.accepts_multiple_links,
        })
        .collect();

    NodeLayout {
        id: NodeId::new(1).unwrap(), // overwritten by caller
        rect,
        title: desc.title.clone(),
        accent: desc.accent,
        inputs,
        outputs,
        section,
        container: None,
        warning: desc.warning.clone(),
        close_button: desc.closable.then(|| {
            WorldRect::new(
                WorldPos::new(
                    min.x + width - CLOSE_BTN_MARGIN - CLOSE_BTN_SIZE,
                    min.y + (HEADER_H - CLOSE_BTN_SIZE) * 0.5,
                ),
                CLOSE_BTN_SIZE,
                CLOSE_BTN_SIZE,
            )
        }),
        collapsed: false,
        hidden: false,
        collapse_toggle: None,
    }
}

/// Fold a section member down to its header alone.
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
    fold_pins(
        layout,
        WorldPos::new(min.x, mid_y),
        WorldPos::new(min.x + width, mid_y),
    );
}

/// Fold a member of a folded section onto that section's edge anchors.
///
/// The member is not drawn and not interactive, but its ports keep resolving —
/// onto the aggregate anchors on the section boundary — so links crossing into
/// a folded section stay visible and selectable.
fn hide_member(layout: &mut NodeLayout, in_anchor: WorldPos, out_anchor: WorldPos) {
    layout.rect = WorldRect::new(in_anchor, 0.0, 0.0);
    layout.collapsed = true;
    layout.hidden = true;
    layout.collapse_toggle = None;
    layout.close_button = None;
    fold_pins(layout, in_anchor, out_anchor);
}

/// Move every input pin onto `in_anchor` and every output pin onto
/// `out_anchor`.
fn fold_pins(layout: &mut NodeLayout, in_anchor: WorldPos, out_anchor: WorldPos) {
    for p in &mut layout.inputs {
        p.center = in_anchor;
    }
    for p in &mut layout.outputs {
        p.center = out_anchor;
    }
}

/// Compute geometry for every node and container the viewer exposes.
pub fn compute(viewer: &dyn GraphViewer, view: &GraphView) -> GraphLayout {
    let containers_desc = viewer.containers();

    // Map each member node to its owning section, so the free-node pass can
    // skip nodes that a container lays out.
    let mut membership: HashMap<NodeId, SectionId> = HashMap::new();
    for c in &containers_desc {
        for s in &c.sections {
            for &m in &s.members {
                membership.insert(m, s.id);
            }
        }
    }

    let mut nodes = Vec::new();
    let mut containers = Vec::new();

    // Containers: one outer frame per pipeline, its sections stacked
    // top-to-bottom under the container header. Members sit flush with the
    // frame's side edges so each member's input/output pins land on the
    // container's outer border, making the whole pipeline read as one node
    // whose pins live on its edge.
    for c in &containers_desc {
        let origin = view.container_position(c.id);
        let member_x = origin.x;
        let mut cursor_y = origin.y + CONTAINER_HEADER_H;
        let mut sections = Vec::with_capacity(c.sections.len());

        for s in &c.sections {
            let section_top = cursor_y;
            let header = WorldRect::new(
                WorldPos::new(origin.x, section_top),
                CONTAINER_WIDTH,
                SECTION_HEADER_H,
            );
            let toggle = WorldRect::new(
                WorldPos::new(
                    origin.x + SECTION_BTN_MARGIN,
                    section_top + (SECTION_HEADER_H - SECTION_BTN_SIZE) * 0.5,
                ),
                SECTION_BTN_SIZE,
                SECTION_BTN_SIZE,
            );
            cursor_y += SECTION_HEADER_H;
            let collapsed = view.is_section_collapsed(s.id);

            let mut collapsible = 0usize;
            let mut collapsed_count = 0usize;
            let mut folds_inputs = false;
            let mut folds_outputs = false;
            let mut member_layouts = Vec::with_capacity(s.members.len());

            if collapsed {
                // Folded: members keep geometry (so links resolve) but fold
                // onto the section's edge anchors and are never drawn.
                let anchor_y = header.center().y;
                let in_anchor = WorldPos::new(origin.x, anchor_y);
                let out_anchor = WorldPos::new(origin.x + CONTAINER_WIDTH, anchor_y);
                for &member in &s.members {
                    let desc = viewer.node(member);
                    let mut layout = node_layout(&desc, in_anchor, CONTAINER_WIDTH, Some(s.id));
                    layout.id = member;
                    hide_member(&mut layout, in_anchor, out_anchor);
                    folds_inputs |= layout.inputs.iter().any(|p| p.connectable);
                    folds_outputs |= layout.outputs.iter().any(|p| p.connectable);
                    member_layouts.push(layout);
                }
            } else {
                for (i, &member) in s.members.iter().enumerate() {
                    if i > 0 {
                        cursor_y += MEMBER_GAP;
                    }
                    let desc = viewer.node(member);
                    let mut layout = node_layout(
                        &desc,
                        WorldPos::new(member_x, cursor_y),
                        CONTAINER_WIDTH,
                        Some(s.id),
                    );
                    layout.id = member;
                    // Members with a body fold/unfold via a header chevron; an
                    // empty member (no ports) is already header-only and needs
                    // no toggle.
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
                    member_layouts.push(layout);
                }
            }

            // An "Add" button sits below the members, inset from the side edges
            // so it doesn't run into the pin column.
            let add_button = (!collapsed && s.can_add_member).then(|| {
                let top = cursor_y + SECTION_PAD;
                cursor_y = top + SECTION_FOOTER_H;
                WorldRect::new(
                    WorldPos::new(member_x + SECTION_PAD, top),
                    CONTAINER_WIDTH - SECTION_PAD * 2.0,
                    SECTION_FOOTER_H,
                )
            });
            if !collapsed {
                cursor_y += SECTION_PAD;
            }

            let collapse_all_button = (!collapsed && collapsible > 0).then(|| {
                WorldRect::new(
                    WorldPos::new(
                        origin.x + CONTAINER_WIDTH - SECTION_BTN_MARGIN - SECTION_BTN_SIZE,
                        section_top + (SECTION_HEADER_H - SECTION_BTN_SIZE) * 0.5,
                    ),
                    SECTION_BTN_SIZE,
                    SECTION_BTN_SIZE,
                )
            });

            sections.push(SectionLayout {
                id: s.id,
                container: c.id,
                rect: WorldRect::new(
                    WorldPos::new(origin.x, section_top),
                    CONTAINER_WIDTH,
                    cursor_y - section_top,
                ),
                header,
                title: s.title.clone(),
                accent: s.accent,
                members: s.members.clone(),
                collapsed,
                toggle,
                add_button,
                collapse_all_button,
                all_collapsed: collapsible > 0 && collapsed_count == collapsible,
                warning: s.warning.clone(),
                folds_inputs,
                folds_outputs,
            });
            for mut layout in member_layouts {
                layout.container = Some(c.id);
                nodes.push(layout);
            }
        }

        let total_h = (cursor_y + CONTAINER_PAD_BOTTOM) - origin.y;
        let rect = WorldRect::new(origin, CONTAINER_WIDTH, total_h);
        containers.push(ContainerLayout {
            id: c.id,
            rect,
            header: WorldRect::new(origin, CONTAINER_WIDTH, CONTAINER_HEADER_H),
            title: c.title.clone(),
            accent: c.accent,
            close_button: c.closable.then(|| {
                WorldRect::new(
                    WorldPos::new(
                        origin.x + CONTAINER_WIDTH - CLOSE_BTN_MARGIN - CLOSE_BTN_SIZE,
                        origin.y + (CONTAINER_HEADER_H - CLOSE_BTN_SIZE) * 0.5,
                    ),
                    CLOSE_BTN_SIZE,
                    CLOSE_BTN_SIZE,
                )
            }),
            warning: c.warning.clone(),
            sections,
        });
    }

    // Free nodes: everything not claimed by a container section.
    for id in viewer.node_ids() {
        if membership.contains_key(&id) {
            continue;
        }
        let desc = viewer.node(id);
        let mut layout = node_layout(&desc, view.position(id), NODE_WIDTH, None);
        layout.id = id;
        nodes.push(layout);
    }

    // Paint/hit precedence follows the persistent z-order: each unit's rank
    // (`z_key`) places it back-to-front, front last, so it renders on top of —
    // and, since interaction hit-tests with `.rev()`, is grabbed in preference
    // to — anything it overlaps. A member ranks with its owning container, so a
    // container's members stay contiguous (a stable-sort tie) and move as one
    // unit.
    nodes.sort_by_key(|n| view.z_key(node_item(n)));
    containers.sort_by_key(|c| view.z_key(CanvasItem::Container(c.id)));

    GraphLayout { nodes, containers }
}

/// The canvas unit a node layout belongs to: its owning container, or itself.
pub fn node_item(node: &NodeLayout) -> CanvasItem {
    match node.container {
        Some(cid) => CanvasItem::Container(cid),
        None => CanvasItem::Node(node.id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viewer::{ContainerDesc, Link, NodeDesc, SectionDesc};

    /// One container with an Emitter section (a single source-like member) and
    /// an Init section holding two members with ports, plus one free node.
    struct PipelineViewer;

    const CONTAINER: u32 = 1;
    const EMITTER_SECTION: u32 = 2;
    const INIT_SECTION: u32 = 3;
    const SOURCE_NODE: u32 = 4;
    const INIT_A: u32 = 5;
    const INIT_B: u32 = 6;
    const FREE_NODE: u32 = 7;

    impl GraphViewer for PipelineViewer {
        fn node_ids(&self) -> Vec<NodeId> {
            [SOURCE_NODE, INIT_A, INIT_B, FREE_NODE]
                .into_iter()
                .map(|id| NodeId::new(id).unwrap())
                .collect()
        }

        fn node(&self, id: NodeId) -> NodeDesc {
            match id.get() {
                SOURCE_NODE => NodeDesc::new("CPU Spawner")
                    .with_inputs(vec![PortDesc::new("count").display_value("32")]),
                FREE_NODE => NodeDesc::new("free").with_outputs(vec![PortDesc::new("out")]),
                _ => NodeDesc::new("modifier")
                    .with_inputs(vec![PortDesc::new("in")])
                    .with_outputs(vec![PortDesc::new("out")]),
            }
        }

        fn links(&self) -> Vec<Link> {
            Vec::new()
        }

        fn containers(&self) -> Vec<ContainerDesc> {
            vec![
                ContainerDesc::new(ContainerId::new(CONTAINER).unwrap(), "Pipeline")
                    .closable()
                    .with_sections(vec![
                        SectionDesc::new(SectionId::new(EMITTER_SECTION).unwrap(), "Emitter")
                            .with_members(vec![NodeId::new(SOURCE_NODE).unwrap()]),
                        SectionDesc::new(SectionId::new(INIT_SECTION).unwrap(), "Init")
                            .with_members(vec![
                                NodeId::new(INIT_A).unwrap(),
                                NodeId::new(INIT_B).unwrap(),
                            ])
                            .with_add_member(true),
                    ]),
            ]
        }
    }

    fn node_of(layout: &GraphLayout, id: u32) -> &NodeLayout {
        layout
            .nodes
            .iter()
            .find(|n| n.id.get() == id)
            .expect("node laid out")
    }

    #[test]
    fn container_frames_all_sections_and_members() {
        let layout = compute(&PipelineViewer, &GraphView::default());
        let container = layout
            .container(ContainerId::new(CONTAINER).unwrap())
            .expect("container laid out");
        assert_eq!(container.sections.len(), 2);
        assert!(container.close_button.is_some());

        // Sections are ordered top to bottom, below the container header and
        // inside the container's bounds.
        let emitter = &container.sections[0];
        let init = &container.sections[1];
        assert!(emitter.rect.min.y >= container.header.max().y);
        assert!(init.rect.min.y >= emitter.rect.max().y);
        assert!(init.rect.max().y <= container.rect.max().y);
        assert_eq!(
            node_of(&layout, SOURCE_NODE).rect.min.y,
            emitter.header.max().y
        );
        assert_eq!(node_of(&layout, INIT_A).rect.min.y, init.header.max().y);

        // Only the Init section opted into an add button.
        assert!(emitter.add_button.is_none());
        assert!(init.add_button.is_some());

        // Members are laid out inside their section, flush with the container
        // edges so their pins land on its border.
        for id in [SOURCE_NODE, INIT_A, INIT_B] {
            let node = node_of(&layout, id);
            assert_eq!(node.container, Some(ContainerId::new(CONTAINER).unwrap()));
            assert!(!node.hidden);
            assert_eq!(node.rect.min.x, container.rect.min.x);
        }
        let a = node_of(&layout, INIT_A);
        let b = node_of(&layout, INIT_B);
        assert_eq!(b.rect.min.y, a.rect.max().y);
        assert_eq!(a.rect.width, CONTAINER_WIDTH);

        // A free node is laid out on its own, outside any container.
        let free = node_of(&layout, FREE_NODE);
        assert_eq!(free.container, None);
        assert_eq!(free.section, None);
        assert_eq!(free.rect.width, NODE_WIDTH);
    }

    #[test]
    fn folded_section_hides_members_and_anchors_their_pins() {
        let mut view = GraphView::default();
        view.toggle_section_collapsed(SectionId::new(INIT_SECTION).unwrap());
        let layout = compute(&PipelineViewer, &view);

        let init = layout
            .section(SectionId::new(INIT_SECTION).unwrap())
            .expect("init section");
        assert!(init.collapsed);
        assert_eq!(init.rect.height, SECTION_HEADER_H);
        assert!(init.add_button.is_none());
        assert!(init.collapse_all_button.is_none());
        assert!(init.folds_inputs && init.folds_outputs);

        for id in [INIT_A, INIT_B] {
            let node = node_of(&layout, id);
            assert!(node.hidden);
            assert!(!node.interactive());
            assert_eq!(node.inputs[0].center, init.fold_input_anchor());
            assert_eq!(node.outputs[0].center, init.fold_output_anchor());
        }

        // Folding shortens the container: everything below the folded section
        // moves up with it.
        let expanded = compute(&PipelineViewer, &GraphView::default());
        let folded_h = layout
            .container(ContainerId::new(CONTAINER).unwrap())
            .unwrap()
            .rect
            .height;
        let expanded_h = expanded
            .container(ContainerId::new(CONTAINER).unwrap())
            .unwrap()
            .rect
            .height;
        assert!(folded_h < expanded_h);
    }

    #[test]
    fn container_position_offsets_the_whole_pipeline() {
        let mut view = GraphView::default();
        view.ensure_container_position(
            ContainerId::new(CONTAINER).unwrap(),
            WorldPos::new(100.0, 50.0),
        );
        let layout = compute(&PipelineViewer, &view);
        let container = layout
            .container(ContainerId::new(CONTAINER).unwrap())
            .unwrap();
        assert_eq!(container.rect.min, WorldPos::new(100.0, 50.0));
        assert_eq!(node_of(&layout, INIT_A).rect.min.x, 100.0);
        assert!(node_of(&layout, INIT_A).rect.min.y > 50.0);
    }

    #[test]
    fn member_collapse_folds_pins_onto_its_own_header() {
        let mut view = GraphView::default();
        view.toggle_collapsed(NodeId::new(INIT_A).unwrap());
        let layout = compute(&PipelineViewer, &view);
        let a = node_of(&layout, INIT_A);
        assert!(a.collapsed);
        assert!(!a.hidden);
        assert_eq!(a.rect.height, HEADER_H);
        assert_eq!(a.inputs[0].center.y, a.rect.min.y + HEADER_H * 0.5);

        let init = layout
            .section(SectionId::new(INIT_SECTION).unwrap())
            .unwrap();
        assert!(!init.all_collapsed);
    }

    #[test]
    fn node_item_maps_members_to_their_container() {
        let layout = compute(&PipelineViewer, &GraphView::default());
        assert_eq!(
            node_item(node_of(&layout, INIT_A)),
            CanvasItem::Container(ContainerId::new(CONTAINER).unwrap())
        );
        assert_eq!(
            node_item(node_of(&layout, FREE_NODE)),
            CanvasItem::Node(NodeId::new(FREE_NODE).unwrap())
        );
    }
}
