//! Input handling: hit-testing, panning, zooming and selection.
//!
//! Covers pan (right/middle drag or two-finger scroll), zoom-to-cursor (pinch
//! or modifier-scroll), node dragging with optional grid snap, link dragging,
//! selection (click + marquee) and delete/context-menu intents. All
//! hit-testing is done in world space.

use egui::PointerButton;

use super::{
    layout::{NodeLayout, STACK_HEADER_H, StackLayout, port_grab_radius_world},
    response::GraphAction,
    state::{
        CanvasDrag, CanvasItem, DragItem, FlowAnchor, GraphView, RIGHT_CLICK_MAX_SECS, ReorderDrag,
    },
    transform::{Transform, WorldPos, WorldRect},
    viewer::{FlowLink, GraphViewer, Link, LinkVerdict, NodeId, PortAddr, PortSide, StackId},
};

/// Topmost node whose body contains `w` (later-drawn nodes win).
fn node_at(layouts: &[NodeLayout], w: WorldPos) -> Option<NodeId> {
    layouts
        .iter()
        .rev()
        .find(|n| n.rect.contains(w))
        .map(|n| n.id)
}

/// Stack whose header band contains `w`, returning its id and origin.
fn stack_header_at(stacks: &[StackLayout], w: WorldPos) -> Option<(StackId, WorldPos)> {
    stacks.iter().rev().find_map(|s| {
        let header = WorldRect::new(s.rect.min, s.rect.width, STACK_HEADER_H);
        header.contains(w).then_some((s.id, s.rect.min))
    })
}

/// A press target resolved between a node body and a stack header.
enum Grab {
    /// A node body (free node or stack member).
    Node(NodeId),
    /// A stack header band, with the stack's origin.
    Header(StackId, WorldPos),
}

/// Frontmost of the node body / stack header under `w`, resolved by z-order.
///
/// A stack header belongs to the whole stack unit; a node belongs to its stack
/// (as a member) or to itself (when free). Whichever unit paints in front owns
/// the press, so grabbing a raised stack's header isn't stolen by a member
/// drawn beneath it, nor a free node's body by a header behind it.
fn node_or_header_at(
    view: &GraphView,
    layouts: &[NodeLayout],
    stacks: &[StackLayout],
    w: WorldPos,
) -> Option<Grab> {
    let node = node_at(layouts, w);
    let header = stack_header_at(stacks, w);
    let node_z = node.map(|n| {
        let item = layouts
            .iter()
            .find(|l| l.id == n)
            .and_then(|l| l.stack)
            .map(CanvasItem::Stack)
            .unwrap_or(CanvasItem::Node(n));
        view.z_key(item)
    });
    let header_z = header.map(|(s, _)| view.z_key(CanvasItem::Stack(s)));
    match (node, header) {
        (Some(n), Some((s, o))) => {
            if header_z > node_z {
                Some(Grab::Header(s, o))
            } else {
                Some(Grab::Node(n))
            }
        }
        (Some(n), None) => Some(Grab::Node(n)),
        (None, Some((s, o))) => Some(Grab::Header(s, o)),
        (None, None) => None,
    }
}

/// Stack whose bottom "Add" button contains `w`.
fn stack_add_button_at(stacks: &[StackLayout], w: WorldPos) -> Option<StackId> {
    stacks
        .iter()
        .rev()
        .find_map(|s| s.add_button.contains(w).then_some(s.id))
}

/// Stack whose header expand/collapse-all button contains `w`.
fn stack_collapse_all_at(stacks: &[StackLayout], w: WorldPos) -> Option<StackId> {
    stacks
        .iter()
        .rev()
        .find_map(|s| s.collapse_all_button.contains(w).then_some(s.id))
}

/// Topmost node whose header close button contains `w` (later-drawn wins).
fn close_button_at(layouts: &[NodeLayout], w: WorldPos) -> Option<NodeId> {
    layouts
        .iter()
        .rev()
        .find_map(|n| n.close_button.filter(|r| r.contains(w)).map(|_| n.id))
}

/// Topmost member whose header collapse chevron contains `w`.
fn collapse_toggle_at(layouts: &[NodeLayout], w: WorldPos) -> Option<NodeId> {
    layouts
        .iter()
        .rev()
        .find_map(|n| n.collapse_toggle.filter(|r| r.contains(w)).map(|_| n.id))
}

/// Index a dragged member would land at within `stack`.
///
/// Given the cursor's world `y`: the count of the stack's *other* members whose
/// vertical center sits above the cursor.
fn reorder_target_index(
    layouts: &[NodeLayout],
    stack: StackId,
    dragged: NodeId,
    cursor_y: f64,
) -> usize {
    layouts
        .iter()
        .filter(|n| n.stack == Some(stack) && n.id != dragged)
        .filter(|n| n.rect.center().y < cursor_y)
        .count()
}

/// Port within grab range of `w`, returning its address and world center.
///
/// The tolerance is the zoom-matched world radius so the clickable area tracks
/// the on-screen hover highlight at any zoom.
fn port_at(
    layouts: &[NodeLayout],
    t: &Transform,
    w: WorldPos,
    side: PortSide,
) -> Option<(PortAddr, WorldPos)> {
    let radius = port_grab_radius_world(t);
    let r2 = radius * radius;
    for node in layouts.iter().rev() {
        if node.collapsed {
            // A collapsed member's pins are folded onto one point and aren't
            // individually connectable; expand it to wire specific fields.
            continue;
        }
        let ports = match side {
            PortSide::Input => &node.inputs,
            PortSide::Output => &node.outputs,
        };
        for p in ports {
            if !p.connectable {
                continue;
            }
            if p.center.distance_squared(w) <= r2 {
                return Some((PortAddr::new(node.id, p.id), p.center));
            }
        }
    }
    None
}

/// Nearest port on either side within grab range of `w`.
///
/// Outputs win ties, matching the drag-start priority.
fn port_at_any(layouts: &[NodeLayout], t: &Transform, w: WorldPos) -> Option<(PortAddr, WorldPos)> {
    port_at(layouts, t, w, PortSide::Output).or_else(|| port_at(layouts, t, w, PortSide::Input))
}

/// Whether `addr` names an input that accepts multiple incoming links.
///
/// `false` for an unresolvable address or an output (only inputs are
/// configurable this way).
fn port_accepts_multiple_links(layouts: &[NodeLayout], addr: PortAddr) -> bool {
    layouts
        .iter()
        .find(|n| n.id == addr.node)
        .and_then(|n| n.inputs.iter().find(|p| p.id == addr.port))
        .is_some_and(|p| p.accepts_multiple_links)
}

/// Node flow-output pin within grab range of `w`.
fn flow_output_at(
    layouts: &[NodeLayout],
    t: &Transform,
    w: WorldPos,
) -> Option<(NodeId, WorldPos)> {
    let radius = port_grab_radius_world(t);
    let r2 = radius * radius;
    layouts.iter().rev().find_map(|n| {
        let center = n.flow_output_pin()?;
        (center.distance_squared(w) <= r2).then_some((n.id, center))
    })
}

/// Node flow-input pin within grab range of `w`.
fn flow_input_at(
    stacks: &[StackLayout],
    t: &Transform,
    w: WorldPos,
) -> Option<(StackId, WorldPos)> {
    let radius = port_grab_radius_world(t);
    let r2 = radius * radius;
    stacks.iter().rev().find_map(|s| {
        let center = s.flow_input_pin()?;
        (center.distance_squared(w) <= r2).then_some((s.id, center))
    })
}

/// Cubic Bézier control points `[from, c1, c2, to]` of a link.
///
/// In world space, or `None` if either endpoint port can't be resolved.
fn link_curve_points(layouts: &[NodeLayout], link: &Link) -> Option<[WorldPos; 4]> {
    let by = |id: NodeId| layouts.iter().find(|n| n.id == id);
    let from = by(link.from.node)?.port_center(link.from.port)?;
    let to = by(link.to.node)?.port_center(link.to.port)?;
    let handle = ((to.x - from.x).abs() * 0.5).clamp(24.0, 160.0);
    Some([
        from,
        from + WorldPos::new(handle, 0.0),
        to - WorldPos::new(handle, 0.0),
        to,
    ])
}

/// Point on a cubic Bézier at parameter `s` in `[0, 1]`.
fn bezier_at(p: &[WorldPos; 4], s: f64) -> WorldPos {
    let mt = 1.0 - s;
    p[0] * (mt * mt * mt)
        + p[1] * (3.0 * mt * mt * s)
        + p[2] * (3.0 * mt * s * s)
        + p[3] * (s * s * s)
}

/// Minimum distance (world units) from `w` to a link's spline, sampled.
fn link_distance(layouts: &[NodeLayout], link: &Link, w: WorldPos) -> Option<f64> {
    let p = link_curve_points(layouts, link)?;
    let mut best = f64::INFINITY;
    let steps = 18;
    for i in 0..=steps {
        let s = i as f64 / steps as f64;
        best = best.min(bezier_at(&p, s).distance_squared(w));
    }
    Some(best.sqrt())
}

/// Whether a link's spline passes through `rect` (any sampled point inside).
fn link_in_rect(layouts: &[NodeLayout], link: &Link, rect: WorldRect) -> bool {
    let Some(p) = link_curve_points(layouts, link) else {
        return false;
    };
    let steps = 18;
    (0..=steps).any(|i| rect.contains(bezier_at(&p, i as f64 / steps as f64)))
}

/// Nearest link to `w` within a small screen-space pick radius, if any.
fn link_at(
    layouts: &[NodeLayout],
    viewer: &dyn GraphViewer,
    t: &Transform,
    w: WorldPos,
) -> Option<Link> {
    let threshold = t.screen_len_to_world(6.0);
    viewer
        .links()
        .into_iter()
        .filter_map(|l| link_distance(layouts, &l, w).map(|d| (l, d)))
        .filter(|(_, d)| *d <= threshold)
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(l, _)| l)
}

/// World-space endpoints of a flow link, or `None` if either pin can't be
/// resolved (e.g. the node or stack no longer opts into one).
fn flow_link_points(
    nodes: &[NodeLayout],
    stacks: &[StackLayout],
    link: &FlowLink,
) -> Option<(WorldPos, WorldPos)> {
    let from = nodes
        .iter()
        .find(|n| n.id == link.from)?
        .flow_output_pin()?;
    let to = stacks.iter().find(|s| s.id == link.to)?.flow_input_pin()?;
    Some((from, to))
}

/// Cubic Bezier control points `[from, c1, c2, to]` of a flow link, with
/// vertical tangents (the flow-link analog of [`link_curve_points`]).
fn flow_link_curve_points(
    nodes: &[NodeLayout],
    stacks: &[StackLayout],
    link: &FlowLink,
) -> Option<[WorldPos; 4]> {
    let (from, to) = flow_link_points(nodes, stacks, link)?;
    let handle = ((to.y - from.y).abs() * 0.5).clamp(24.0, 160.0);
    Some([
        from,
        from + WorldPos::new(0.0, handle),
        to - WorldPos::new(0.0, handle),
        to,
    ])
}

/// Minimum distance (world units) from `w` to a flow link's spline, sampled.
fn flow_link_distance(
    nodes: &[NodeLayout],
    stacks: &[StackLayout],
    link: &FlowLink,
    w: WorldPos,
) -> Option<f64> {
    let p = flow_link_curve_points(nodes, stacks, link)?;
    let mut best = f64::INFINITY;
    let steps = 18;
    for i in 0..=steps {
        let s = i as f64 / steps as f64;
        best = best.min(bezier_at(&p, s).distance_squared(w));
    }
    Some(best.sqrt())
}

/// Whether a flow link's spline passes through `rect` (any sampled point
/// inside).
fn flow_link_in_rect(
    nodes: &[NodeLayout],
    stacks: &[StackLayout],
    link: &FlowLink,
    rect: WorldRect,
) -> bool {
    let Some(p) = flow_link_curve_points(nodes, stacks, link) else {
        return false;
    };
    let steps = 18;
    (0..=steps).any(|i| rect.contains(bezier_at(&p, i as f64 / steps as f64)))
}

/// Nearest flow link to `w` within a small screen-space pick radius, if any.
fn flow_link_at(
    nodes: &[NodeLayout],
    stacks: &[StackLayout],
    viewer: &dyn GraphViewer,
    t: &Transform,
    w: WorldPos,
) -> Option<FlowLink> {
    let threshold = t.screen_len_to_world(6.0);
    viewer
        .flow_links()
        .into_iter()
        .filter_map(|l| flow_link_distance(nodes, stacks, &l, w).map(|d| (l, d)))
        .filter(|(_, d)| *d <= threshold)
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(l, _)| l)
}

/// Resolve the action (if any) for an in-progress flow-link drag ending.
///
/// `accepted` is the consumer-validated candidate under the cursor, when one
/// was found; `detached` is the flow link that grabbing the anchor pin
/// detached, when the anchor was a stack's flow-input pin (see
/// [`FlowAnchor::Stack`]). A node's flow-output anchor never detaches: a
/// drop that isn't accepted just cancels. A stack's flow-input anchor rewires
/// on an accepted drop, or re-requests deletion of the detached link
/// otherwise.
fn flow_link_drop_action(
    anchor: FlowAnchor,
    accepted: Option<FlowTarget>,
    detached: Option<FlowLink>,
) -> Option<GraphAction> {
    match anchor {
        FlowAnchor::Node(from_node) => match accepted {
            Some(FlowTarget::Stack(to_stack)) => Some(GraphAction::FlowLinkRequested {
                from: from_node,
                to: to_stack,
            }),
            _ => None,
        },
        FlowAnchor::Stack(to_stack) => match accepted {
            Some(FlowTarget::Node(from_node)) => Some(GraphAction::FlowLinkRequested {
                from: from_node,
                to: to_stack,
            }),
            _ => detached.map(|link| GraphAction::FlowLinkDeleteRequested { link }),
        },
    }
}

/// The candidate drop target of an in-progress link drag.
///
/// The port under the cursor on the droppable side, plus the consumer's verdict
/// on whether the connection is allowed.
#[derive(Debug, Clone)]
pub struct LinkTarget {
    /// The candidate port under the cursor (an output if the anchor is an
    /// input, otherwise an input).
    pub addr: PortAddr,
    /// Its world center, for snapping the dragged spline's endpoint.
    pub center: WorldPos,
    /// Whether the connection is allowed (`Ok`), or why not (`Err(reason)`).
    pub verdict: LinkVerdict,
}

/// The opposite endpoint of an in-progress flow-link drag: whichever kind of
/// pin the anchor is *not*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FlowTarget {
    Node(NodeId),
    Stack(StackId),
}

/// The candidate drop target of an in-progress flow-link drag.
///
/// Mirrors [`LinkTarget`] for the flow-link kind.
#[derive(Debug, Clone)]
pub struct FlowLinkTarget {
    /// The candidate pin under the cursor (a node's flow output if the anchor
    /// is a stack's flow input, otherwise a stack's flow input).
    pub target: FlowTarget,
    /// Its world center, for snapping the dragged spline's endpoint.
    pub center: WorldPos,
    /// Whether the connection is allowed (`Ok`), or why not (`Err(reason)`).
    pub verdict: LinkVerdict,
}

/// What the pointer is hovering this frame, for render highlighting.
#[derive(Debug, Clone, Default)]
pub struct Hover {
    /// Topmost node body geometrically under the pointer.
    ///
    /// Unlike [`node`], this remains set when a port has visual feedback
    /// priority.
    ///
    /// [`node`]: Self::node
    pub node_body: Option<NodeId>,
    pub node: Option<NodeId>,
    pub stack: Option<StackId>,
    /// Stack whose "Add" button is under the cursor this frame.
    pub add_button: Option<StackId>,
    /// Stack whose header expand/collapse-all button is under the cursor.
    pub collapse_all: Option<StackId>,
    /// Node whose header close button is under the cursor this frame.
    pub close: Option<NodeId>,
    /// World center of a port under the cursor (within grab tolerance), for
    /// drawing a pin-specific hover highlight.
    pub port: Option<WorldPos>,
    /// World center of a flow pin under the cursor (within grab tolerance),
    /// for drawing a pin-specific hover highlight.
    pub flow_port: Option<WorldPos>,
    /// During a link drag, the validated candidate target under the cursor
    /// (if any), used to snap the spline, blend toward the target's color, and
    /// show a rejection reason.
    pub link_target: Option<LinkTarget>,
    /// During a flow-link drag, the validated candidate target under the
    /// cursor (if any). Mirrors [`Self::link_target`] for the flow-link kind.
    pub flow_link_target: Option<FlowLinkTarget>,
    /// Nodes currently under the in-progress marquee rectangle. They render
    /// as hovered to preview what a drag-selection will capture.
    pub marquee: Vec<NodeId>,
    /// Stacks currently under the in-progress marquee rectangle, previewed as
    /// pending selection.
    pub marquee_stacks: Vec<StackId>,
    /// Links currently crossing the in-progress marquee rectangle, previewed
    /// as pending selection.
    pub marquee_links: Vec<Link>,
    /// Flow links currently crossing the in-progress marquee rectangle,
    /// previewed as pending selection.
    pub marquee_flow_links: Vec<FlowLink>,
}

/// Process all input for this frame.
///
/// Mutates `view` (pan/zoom/positions/selection/interaction) and pushes
/// structural intents into `actions`. Returns what the pointer hovers for
/// render highlighting.
pub fn handle(
    ui: &egui::Ui,
    response: &egui::Response,
    t: &Transform,
    layouts: &[NodeLayout],
    stacks: &[StackLayout],
    viewer: &dyn GraphViewer,
    view: &mut GraphView,
    actions: &mut Vec<GraphAction>,
) -> Hover {
    let hover_world = response.hover_pos().map(|p| t.screen_to_world(p));
    let node_under_pointer = ui
        .input(|input| input.pointer.hover_pos())
        .filter(|pointer| response.rect.contains(*pointer))
        .map(|pointer| t.screen_to_world(pointer))
        .and_then(|world| node_at(layouts, world));
    // A port under the cursor takes feedback priority over the node it sits
    // on: show a pin-specific highlight + connect cursor instead of the
    // node's grab/edge-highlight.
    let hovered_port = hover_world.and_then(|w| port_at_any(layouts, t, w));
    // A flow pin (node flow-output or stack flow-input) under the cursor
    // takes the same feedback priority as an ordinary port.
    let hovered_flow_port = hover_world.and_then(|w| {
        flow_output_at(layouts, t, w)
            .map(|(_, c)| c)
            .or_else(|| flow_input_at(stacks, t, w).map(|(_, c)| c))
    });
    let hovered_node = if hovered_port.is_some() || hovered_flow_port.is_some() {
        None
    } else {
        hover_world.and_then(|world| node_at(layouts, world))
    };

    // --- Validate an in-progress link drag against a candidate target ---
    // The anchor is the port the user grabbed; the droppable side is the
    // opposite one. A port on the opposite side is offered to the consumer to
    // validate; a port on the *same* side is rejected by the widget itself,
    // since a link always runs output → input regardless of consumer policy.
    let link_target = view.interaction.pending_link_from.and_then(|anchor| {
        let w = hover_world?;
        let from_input = view.interaction.pending_from_input;
        let (target_side, anchor_side) = if from_input {
            (PortSide::Output, PortSide::Input)
        } else {
            (PortSide::Input, PortSide::Output)
        };
        if let Some((cand, center)) = port_at(layouts, t, w, target_side) {
            let (from, to) = if from_input {
                (cand, anchor)
            } else {
                (anchor, cand)
            };
            return Some(LinkTarget {
                addr: cand,
                center,
                verdict: viewer.validate_link(from, to),
            });
        }
        if let Some((cand, center)) = port_at(layouts, t, w, anchor_side)
            && cand != anchor
        {
            let reason = if from_input {
                "can't connect two inputs"
            } else {
                "can't connect two outputs"
            };
            return Some(LinkTarget {
                addr: cand,
                center,
                verdict: Err(reason.into()),
            });
        }
        None
    });

    // Validate an in-progress flow-link drag against a candidate target.
    // Mirrors the value-link validation above, but the anchor's kind
    // (node vs. stack) already determines the droppable side, so there's no
    // same-side rejection to compute.
    let flow_link_target = view.interaction.pending_flow_link_from.and_then(|anchor| {
        let w = hover_world?;
        match anchor {
            FlowAnchor::Node(from_node) => {
                let (to_stack, center) = flow_input_at(stacks, t, w)?;
                Some(FlowLinkTarget {
                    target: FlowTarget::Stack(to_stack),
                    center,
                    verdict: viewer.validate_flow_link(from_node, to_stack),
                })
            }
            FlowAnchor::Stack(to_stack) => {
                let (from_node, center) = flow_output_at(layouts, t, w)?;
                Some(FlowLinkTarget {
                    target: FlowTarget::Node(from_node),
                    center,
                    verdict: viewer.validate_flow_link(from_node, to_stack),
                })
            }
        }
    });

    // Grab cursor over anything draggable (free nodes move, stack members
    // reorder, stack headers move the whole stack); Grabbing while a drag
    // is active; Crosshair over a port (start/complete a connection).
    let dragging = view.interaction.canvas_drag.is_some() || view.interaction.reordering.is_some();
    // Stacks don't accept drops and never interact with a dragged node, so
    // suppress their hover highlight mid-drag — lighting one up implies a
    // relationship that doesn't exist.
    let hovered_stack = if dragging {
        None
    } else {
        hover_world.and_then(|w| stack_header_at(stacks, w).map(|(id, _)| id))
    };
    let hovered_add_button = if dragging {
        None
    } else {
        hover_world.and_then(|w| stack_add_button_at(stacks, w))
    };
    let hovered_collapse_all = if dragging {
        None
    } else {
        hover_world.and_then(|w| stack_collapse_all_at(stacks, w))
    };
    let hovered_close = if dragging {
        None
    } else {
        hover_world.and_then(|w| close_button_at(layouts, w))
    };
    let hovered_toggle = if dragging {
        None
    } else {
        hover_world.and_then(|w| collapse_toggle_at(layouts, w))
    };

    if dragging {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
    } else if matches!(&link_target, Some(lt) if lt.verdict.is_err())
        || matches!(&flow_link_target, Some(lt) if lt.verdict.is_err())
    {
        // Hovering a target the consumer rejects: show it can't be dropped.
        ui.ctx().set_cursor_icon(egui::CursorIcon::NotAllowed);
    } else if hovered_close.is_some() || hovered_toggle.is_some() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    } else if hovered_port.is_some() || hovered_flow_port.is_some() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
    } else if hovered_add_button.is_some() || hovered_collapse_all.is_some() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    } else if hovered_stack.is_some() || hovered_node.is_some() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
    }

    // --- Scroll-to-pan, pinch / modifier-scroll to zoom ---
    //
    // A two-finger trackpad drag or a plain mouse wheel pans the canvas; a
    // pinch gesture or ⌘/Ctrl + scroll zooms toward the cursor. egui folds
    // zoom-modifier scrolls into `zoom_delta` and out of `smooth_scroll_delta`,
    // so the two gestures never fight. The pan sign matches `ScrollArea`, so
    // two-finger scrolling feels like a normal scroll view.
    //
    // The geometry check keeps navigation active over same-layer inline chip
    // editors, while the top-layer check prevents the canvas from stealing
    // scroll input from menus, popups, and other floating UI.
    let pointer = ui.input(|i| i.pointer.hover_pos());
    if let Some(cursor) = pointer.filter(|p| {
        response.rect.contains(*p) && canvas_accepts_navigation(ui.ctx(), response.layer_id, *p)
    }) {
        let (scroll, zoom) = ui.input(|i| (i.smooth_scroll_delta, i.zoom_delta()));
        if scroll != egui::Vec2::ZERO {
            view.pan -= t.screen_vec_to_world(scroll);
        }
        if zoom != 1.0 {
            let w_before = t.screen_to_world(cursor);
            view.set_zoom_clamped(view.zoom * zoom as f64);
            let t2 = Transform::new(t.origin, view.pan, view.zoom);
            let w_after = t2.screen_to_world(cursor);
            view.pan += w_before - w_after;
        }
    }

    // --- Pan (right- or middle-button drag, stateless) ---
    // egui distinguishes a drag from a click by movement threshold, so a
    // right-button *drag* pans while a right-button *click* (handled below)
    // opens the context menu.
    if response.dragged_by(PointerButton::Secondary) || response.dragged_by(PointerButton::Middle) {
        let d = response.drag_delta();
        view.pan -= t.screen_vec_to_world(d);
    }

    // --- Begin a primary-button drag: classify what was grabbed ---
    if response.drag_started_by(PointerButton::Primary) {
        // Classify against the press origin, not the current pointer: by the
        // time `drag_started` fires the cursor has already moved past egui's
        // drag threshold, which can carry it off a small target like a port.
        if let Some(p) = ui
            .input(|i| i.pointer.press_origin())
            .or_else(|| response.interact_pointer_pos())
        {
            let w = t.screen_to_world(p);
            let shift = ui.input(|i| i.modifiers.shift);
            if stack_add_button_at(stacks, w).is_some() {
                // The bottom "Add" button is a click target; a press there must
                // not begin a marquee or canvas drag. The click is emitted in
                // the click handler below.
            } else if stack_collapse_all_at(stacks, w).is_some() {
                // The header collapse-all button is a click target; suppress
                // drag so a press there can't start moving the stack.
            } else if collapse_toggle_at(layouts, w).is_some() {
                // The header collapse chevron is a click target; suppress drag
                // so a press there can't start reordering. Click handled below.
            } else if close_button_at(layouts, w).is_some() {
                // The header close button is a click target; suppress drag so a
                // press there can't start moving the node. Click handled below.
            } else if let Some((addr, _)) = port_at(layouts, t, w, PortSide::Output) {
                view.interaction.pending_link_from = Some(addr);
            } else if let Some((addr, _)) = port_at(layouts, t, w, PortSide::Input) {
                // Grabbing an input pin anchors a link at that input; the user
                // drags the free (source) end out to an output. Any pre-existing
                // link into the input is detached: rewired if dropped on a new
                // output, removed otherwise.
                //
                // A multiple-link input never detaches on grab: picking one of
                // its existing edges to remove would be ambiguous. Grabbing it
                // just starts a fresh link search instead.
                view.interaction.pending_link_from = Some(addr);
                view.interaction.pending_from_input = true;
                let accepts_multiple_links = port_accepts_multiple_links(layouts, addr);
                view.interaction.detaching_link = (!accepts_multiple_links)
                    .then(|| viewer.links().into_iter().find(|l| l.to == addr))
                    .flatten();
            } else if let Some((node, _)) = flow_output_at(layouts, t, w) {
                // Grabbing a node's flow-output pin starts a fresh flow-link
                // drag; a node driving more than one stack is a consumer
                // policy question decided by `validate_flow_link`, not a
                // widget-level detach.
                view.interaction.pending_flow_link_from = Some(FlowAnchor::Node(node));
            } else if let Some((stack, _)) = flow_input_at(stacks, t, w) {
                // Grabbing a stack's flow-input pin anchors the drag there and
                // detaches any existing incoming flow link, mirroring an
                // ordinary single-source input pin.
                view.interaction.pending_flow_link_from = Some(FlowAnchor::Stack(stack));
                view.interaction.detaching_flow_link =
                    viewer.flow_links().into_iter().find(|l| l.to == stack);
            } else {
                // Node body vs stack header is resolved by z-order so the
                // frontmost unit under the cursor owns the press.
                match node_or_header_at(view, layouts, stacks, w) {
                    Some(Grab::Node(node)) => {
                        let layout = layouts.iter().find(|n| n.id == node);
                        match layout.and_then(|n| n.stack) {
                            // Stack member: drag reorders it within its stack.
                            // Members are not part of the canvas selection —
                            // they move and delete by different rules than free
                            // nodes.
                            Some(sid) => {
                                view.raise(CanvasItem::Stack(sid));
                                if let Some((from_index, grab_offset)) = stacks
                                    .iter()
                                    .find(|s| s.id == sid)
                                    .and_then(|s| s.members.iter().position(|m| *m == node))
                                    .zip(layout.map(|n| w - n.rect.min))
                                {
                                    view.interaction.reordering = Some(ReorderDrag {
                                        stack: sid,
                                        node,
                                        from_index,
                                        target_index: from_index,
                                        grab_offset,
                                    });
                                }
                            }
                            // Free node: select it (unless already selected) and
                            // start a group drag of the whole canvas selection.
                            None => {
                                view.raise(CanvasItem::Node(node));
                                if !view.selection.contains(&node) {
                                    if !shift {
                                        view.clear_selection();
                                    }
                                    view.selection.insert(node);
                                    actions.push(GraphAction::SelectionChanged);
                                }
                                view.interaction.canvas_drag =
                                    Some(begin_canvas_drag(view, DragItem::Node(node), w));
                            }
                        }
                    }
                    Some(Grab::Header(stack, _origin)) => {
                        // A stack is a canvas-movable unit like a free node:
                        // select it (unless already selected) and group-drag the
                        // selection.
                        view.raise(CanvasItem::Stack(stack));
                        if !view.selected_stacks.contains(&stack) {
                            if !shift {
                                view.clear_selection();
                            }
                            view.selected_stacks.insert(stack);
                            actions.push(GraphAction::SelectionChanged);
                        }
                        view.interaction.canvas_drag =
                            Some(begin_canvas_drag(view, DragItem::Stack(stack), w));
                    }
                    None => {
                        if !shift && view.clear_selection() {
                            actions.push(GraphAction::SelectionChanged);
                        }
                        view.interaction.box_select_start = Some(w);
                    }
                }
            }
        }
    }

    // --- Continue a primary drag ---
    if response.dragged_by(PointerButton::Primary) {
        if let (Some(drag), Some(p)) = (
            view.interaction.canvas_drag.clone(),
            response.interact_pointer_pos(),
        ) {
            // Snap the grabbed (primary) item, then translate the whole
            // selection rigidly by the resulting delta.
            let mut new_primary = t.screen_to_world(p) - drag.grab_offset;
            if view.grid.snap {
                new_primary = view.grid.snap_pos(new_primary);
            }
            let delta = new_primary - drag.primary_origin;
            for (id, origin) in &drag.nodes {
                view.positions.insert(*id, *origin + delta);
            }
            for (id, origin) in &drag.stacks {
                view.stack_positions.insert(*id, *origin + delta);
            }
        }
        if let (Some(mut rd), Some(p)) =
            (view.interaction.reordering, response.interact_pointer_pos())
        {
            let cursor_y = t.screen_to_world(p).y;
            rd.target_index = reorder_target_index(layouts, rd.stack, rd.node, cursor_y);
            view.interaction.reordering = Some(rd);
        }
    }

    // --- Cancel any in-progress drag on Escape ---
    // Pressing Escape during a drag abandons it without committing. egui does
    // not report `drag_stopped` for an Escape-cancelled drag, so the transient
    // state is cleared here; otherwise a pending link's rubber-band would leak
    // and follow the cursor forever after the button is released.
    if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
        view.interaction.pending_link_from = None;
        view.interaction.pending_from_input = false;
        view.interaction.detaching_link = None;
        view.interaction.pending_flow_link_from = None;
        view.interaction.detaching_flow_link = None;
        view.interaction.box_select_start = None;
        view.interaction.reordering = None;
        if let Some(drag) = view.interaction.canvas_drag.take() {
            // Snap every dragged item back to where the drag began.
            for (id, origin) in &drag.nodes {
                view.positions.insert(*id, *origin);
            }
            for (id, origin) in &drag.stacks {
                view.stack_positions.insert(*id, *origin);
            }
        }
    }

    // --- End a primary drag ---
    if response.drag_stopped_by(PointerButton::Primary) {
        if let Some(drag) = view.interaction.canvas_drag.take() {
            for (node, origin) in &drag.nodes {
                actions.push(GraphAction::NodeMoved {
                    node: *node,
                    from: *origin,
                    to: view.position(*node),
                });
            }
            for (stack, origin) in &drag.stacks {
                actions.push(GraphAction::StackMoved {
                    stack: *stack,
                    from: *origin,
                    to: view.stack_position(*stack),
                });
            }
        }
        if let Some(rd) = view.interaction.reordering.take()
            && rd.target_index != rd.from_index
        {
            actions.push(GraphAction::StackMemberMoved {
                stack: rd.stack,
                from_index: rd.from_index,
                to_index: rd.target_index,
            });
        }
        if let Some(from) = view.interaction.pending_link_from.take() {
            let detached = view.interaction.detaching_link.take();
            let from_input = std::mem::take(&mut view.interaction.pending_from_input);
            let drop_world = response
                .interact_pointer_pos()
                .map(|p| t.screen_to_world(p));
            // A candidate port sits under the cursor (any verdict).
            let target_present = link_target.is_some();
            // The candidate the consumer accepts.
            let accepted = link_target
                .as_ref()
                .filter(|lt| lt.verdict.is_ok())
                .map(|lt| lt.addr);

            if from_input {
                // Anchor is an input pin. Dropping on an accepted output wires
                // it in (displacing any prior source); dropping elsewhere with a
                // prior link detaches it; dropping on empty canvas with no prior
                // link offers to create a producer node feeding the input.
                if let Some(out) = accepted {
                    actions.push(GraphAction::LinkRequested {
                        from: out,
                        to: from,
                    });
                } else if let Some(old) = detached {
                    actions.push(GraphAction::LinkDeleteRequested { link: old });
                } else if !target_present && let Some(at) = drop_world {
                    actions.push(GraphAction::LinkDropped {
                        source: from,
                        source_is_output: false,
                        at,
                    });
                }
            } else {
                // Anchor is an output pin: a fresh link dragged to an input.
                match (accepted, target_present) {
                    (Some(to), _) => {
                        actions.push(GraphAction::LinkRequested { from, to });
                    }
                    // Dropped on a rejected target: cancel.
                    (None, true) => {}
                    // Dropped on empty canvas: offer to create a consumer node
                    // and feed this output into it.
                    (None, false) => {
                        if let Some(at) = drop_world {
                            actions.push(GraphAction::LinkDropped {
                                source: from,
                                source_is_output: true,
                                at,
                            });
                        }
                    }
                }
            }
        }
        if let Some(anchor) = view.interaction.pending_flow_link_from.take() {
            let detached = view.interaction.detaching_flow_link.take();
            // The candidate the consumer accepts.
            let accepted = flow_link_target
                .as_ref()
                .filter(|lt| lt.verdict.is_ok())
                .map(|lt| lt.target);
            actions.extend(flow_link_drop_action(anchor, accepted, detached));
        }
        if let Some(start) = view.interaction.box_select_start.take()
            && let Some(p) = response.interact_pointer_pos()
        {
            let end = t.screen_to_world(p);
            let size = (end - start).abs();
            let rect = WorldRect::new(start.min(end), size.x, size.y);
            // The marquee captures the canvas's movable units — free nodes
            // and whole stacks — plus links. Stack members are excluded;
            // they reorder within their stack rather than move freely.
            for node in layouts {
                if node.stack.is_none() && rects_intersect(rect, node.rect) {
                    view.selection.insert(node.id);
                }
            }
            for stack in stacks {
                if rects_intersect(rect, stack.rect) {
                    view.selected_stacks.insert(stack.id);
                }
            }
            for link in viewer.links() {
                if link_in_rect(layouts, &link, rect) {
                    view.selected_links.insert(link);
                }
            }
            for link in viewer.flow_links() {
                if flow_link_in_rect(layouts, stacks, &link, rect) {
                    view.selected_flow_links.insert(link);
                }
            }
            actions.push(GraphAction::SelectionChanged);
        }
    }

    // --- Plain click: selection (nodes and edges) ---
    if response.clicked_by(PointerButton::Primary)
        && let Some(p) = response.interact_pointer_pos()
    {
        let w = t.screen_to_world(p);
        let shift = ui.input(|i| i.modifiers.shift);
        if let Some(stack) = stack_add_button_at(stacks, w) {
            actions.push(GraphAction::StackAddRequested { stack });
        } else if let Some(sid) = stack_collapse_all_at(stacks, w) {
            // Fold or unfold every collapsible member of the stack at once.
            // Collapse when any is expanded, otherwise expand them all.
            let members: Vec<NodeId> = layouts
                .iter()
                .filter(|n| n.stack == Some(sid) && n.collapse_toggle.is_some())
                .map(|n| n.id)
                .collect();
            let collapse = members.iter().any(|id| !view.is_collapsed(*id));
            for id in members {
                if collapse {
                    view.collapsed.insert(id);
                } else {
                    view.collapsed.remove(&id);
                }
            }
        } else if let Some(node) = collapse_toggle_at(layouts, w) {
            // Header chevron: fold/unfold this member. Pure view state, so
            // it's applied directly with no action emitted.
            view.toggle_collapsed(node);
        } else if let Some(node) = close_button_at(layouts, w) {
            // Header close button: delete just this node. The consumer maps
            // it to the right edit (remove a free node, or a stack member).
            actions.push(GraphAction::NodesDeleteRequested { nodes: vec![node] });
        } else if port_at(layouts, t, w, PortSide::Output).is_some()
            || port_at(layouts, t, w, PortSide::Input).is_some()
        {
            // Clicking a port is not a selection gesture.
        } else if flow_output_at(layouts, t, w).is_some() || flow_input_at(stacks, t, w).is_some() {
            // Clicking a flow pin is not a selection gesture.
        } else if let Some(grab) = node_or_header_at(view, layouts, stacks, w) {
            // Node body vs stack header is resolved by z-order. Clicking a
            // free node selects the node; clicking a stack member or a stack
            // header selects the parent stack (the stack is the unit).
            match grab {
                Grab::Node(node) => {
                    match layouts.iter().find(|n| n.id == node).and_then(|n| n.stack) {
                        Some(sid) => click_select_stack(view, sid, shift, actions),
                        None => click_select_node(view, node, shift, actions),
                    }
                }
                Grab::Header(stack, _) => click_select_stack(view, stack, shift, actions),
            }
        } else if let Some(link) = link_at(layouts, viewer, t, w) {
            if shift {
                if !view.selected_links.insert(link) {
                    view.selected_links.remove(&link);
                }
            } else {
                view.clear_selection();
                view.selected_links.insert(link);
            }
            actions.push(GraphAction::SelectionChanged);
        } else if let Some(link) = flow_link_at(layouts, stacks, viewer, t, w) {
            if shift {
                if !view.selected_flow_links.insert(link) {
                    view.selected_flow_links.remove(&link);
                }
            } else {
                view.clear_selection();
                view.selected_flow_links.insert(link);
            }
            actions.push(GraphAction::SelectionChanged);
        } else if view.clear_selection() {
            actions.push(GraphAction::SelectionChanged);
        }
    }

    // --- Right-click in place: context menu. We detect it by press/release
    //     *timing*, not movement: a trackpad two-finger tap jumps the pointer
    //     between its two touch points, which egui (and any distance check)
    //     reads as a drag, suppressing `secondary_clicked()`. A brief press is
    //     a right-click; a longer hold is a pan (handled above). ---
    if response.contains_pointer()
        && ui.input(|i| i.pointer.button_pressed(PointerButton::Secondary))
    {
        let pos = ui.input(|i| i.pointer.interact_pos().or_else(|| i.pointer.latest_pos()));
        if let Some(pos) = pos {
            view.interaction.secondary_press = Some((pos, ui.input(|i| i.time)));
        }
    }
    if ui.input(|i| i.pointer.button_released(PointerButton::Secondary))
        && let Some((press_pos, press_time)) = view.interaction.secondary_press.take()
    {
        let release_pos = ui
            .input(|i| i.pointer.interact_pos().or_else(|| i.pointer.latest_pos()))
            .unwrap_or(press_pos);
        if ui.input(|i| i.time) - press_time <= RIGHT_CLICK_MAX_SECS {
            actions.push(GraphAction::ContextMenu {
                at: t.screen_to_world(release_pos),
            });
        }
    }

    // --- Delete key removes the current selection (nodes, stacks and edges) ---
    let has_selection = !view.selection.is_empty()
        || !view.selected_links.is_empty()
        || !view.selected_flow_links.is_empty()
        || !view.selected_stacks.is_empty();
    if (response.hovered() || response.has_focus()) && has_selection {
        let del =
            ui.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace));
        if del {
            if !view.selected_links.is_empty() {
                for link in view.selected_links.drain() {
                    actions.push(GraphAction::LinkDeleteRequested { link });
                }
            }
            if !view.selected_flow_links.is_empty() {
                for link in view.selected_flow_links.drain() {
                    actions.push(GraphAction::FlowLinkDeleteRequested { link });
                }
            }
            if !view.selection.is_empty() {
                actions.push(GraphAction::NodesDeleteRequested {
                    nodes: view.selection.drain().collect(),
                });
            }
            if !view.selected_stacks.is_empty() {
                actions.push(GraphAction::StacksDeleteRequested {
                    stacks: view.selected_stacks.drain().collect(),
                });
            }
        }
    }

    // Free nodes, stacks and links under the in-progress marquee, previewed as
    // pending selection (stack members are not marquee-selectable).
    let (marquee, marquee_stacks, marquee_links, marquee_flow_links) = match (
        view.interaction.box_select_start,
        response.interact_pointer_pos(),
    ) {
        (Some(start), Some(p)) => {
            let end = t.screen_to_world(p);
            let size = (end - start).abs();
            let rect = WorldRect::new(start.min(end), size.x, size.y);
            let nodes = layouts
                .iter()
                .filter(|n| n.stack.is_none() && rects_intersect(rect, n.rect))
                .map(|n| n.id)
                .collect();
            let m_stacks = stacks
                .iter()
                .filter(|s| rects_intersect(rect, s.rect))
                .map(|s| s.id)
                .collect();
            let links = viewer
                .links()
                .into_iter()
                .filter(|l| link_in_rect(layouts, l, rect))
                .collect();
            let flow_links = viewer
                .flow_links()
                .into_iter()
                .filter(|l| flow_link_in_rect(layouts, stacks, l, rect))
                .collect();
            (nodes, m_stacks, links, flow_links)
        }
        _ => (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
    };

    Hover {
        node_body: node_under_pointer,
        node: hovered_node,
        stack: hovered_stack,
        add_button: hovered_add_button,
        collapse_all: hovered_collapse_all,
        close: hovered_close,
        port: hovered_port.map(|(_, c)| c),
        flow_port: hovered_flow_port,
        link_target,
        flow_link_target,
        marquee,
        marquee_stacks,
        marquee_links,
        marquee_flow_links,
    }
}

fn canvas_accepts_navigation(
    context: &egui::Context,
    canvas_layer: egui::LayerId,
    pointer: egui::Pos2,
) -> bool {
    context
        .layer_id_at(pointer)
        .is_none_or(|top_layer| top_layer == canvas_layer)
}

/// Apply a plain/shift click to free-node selection.
fn click_select_node(
    view: &mut GraphView,
    node: NodeId,
    shift: bool,
    actions: &mut Vec<GraphAction>,
) {
    view.raise(CanvasItem::Node(node));
    if shift {
        if !view.selection.insert(node) {
            view.selection.remove(&node);
        }
    } else {
        view.clear_selection();
        view.selection.insert(node);
    }
    actions.push(GraphAction::SelectionChanged);
}

/// Apply a plain/shift click to stack selection.
fn click_select_stack(
    view: &mut GraphView,
    stack: StackId,
    shift: bool,
    actions: &mut Vec<GraphAction>,
) {
    view.raise(CanvasItem::Stack(stack));
    if shift {
        if !view.selected_stacks.insert(stack) {
            view.selected_stacks.remove(&stack);
        }
    } else {
        view.clear_selection();
        view.selected_stacks.insert(stack);
    }
    actions.push(GraphAction::SelectionChanged);
}

/// Capture the current canvas selection as a rigid group drag.
///
/// Captures free nodes + stacks anchored on `primary`, grabbed at world point
/// `grab_world`.
fn begin_canvas_drag(view: &GraphView, primary: DragItem, grab_world: WorldPos) -> CanvasDrag {
    let nodes = view
        .selection
        .iter()
        .map(|&id| (id, view.position(id)))
        .collect();
    let stacks = view
        .selected_stacks
        .iter()
        .map(|&id| (id, view.stack_position(id)))
        .collect();
    let primary_origin = match primary {
        DragItem::Node(id) => view.position(id),
        DragItem::Stack(id) => view.stack_position(id),
    };
    CanvasDrag {
        primary_origin,
        grab_offset: grab_world - primary_origin,
        nodes,
        stacks,
    }
}

fn rects_intersect(a: WorldRect, b: WorldRect) -> bool {
    a.min.cmple(b.max()).all() && a.max().cmpge(b.min).all()
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;
    use crate::{
        layout::PortLayout,
        viewer::{NodeDesc, PortId, StackDesc},
    };

    fn node_layout(id: u32, min: WorldPos, flow_output: bool) -> NodeLayout {
        NodeLayout {
            id: NodeId::new(id).unwrap(),
            rect: WorldRect::new(min, 100.0, 80.0),
            title: Cow::Borrowed("n"),
            accent: None,
            inputs: vec![PortLayout {
                id: PortId::input(0),
                center: WorldPos::new(min.x, min.y + 40.0),
                label: Cow::Borrowed("in"),
                color: None,
                arity: 0,
                value: None,
                connectable: true,
                row_height: 22.0,
                collapsible: false,
                accepts_multiple_links: false,
            }],
            outputs: Vec::new(),
            stack: None,
            warning: None,
            close_button: None,
            collapsed: false,
            collapse_toggle: None,
            flow_output,
        }
    }

    fn stack_layout(id: u32, min: WorldPos, flow_input: bool) -> StackLayout {
        StackLayout {
            id: StackId::new(id).unwrap(),
            rect: WorldRect::new(min, 100.0, 120.0),
            title: Cow::Borrowed("s"),
            accent: None,
            members: Vec::new(),
            add_button: WorldRect::new(min, 10.0, 10.0),
            collapse_all_button: WorldRect::new(min, 10.0, 10.0),
            all_collapsed: false,
            flow_input,
        }
    }

    fn transform() -> Transform {
        Transform::new(egui::pos2(0.0, 0.0), WorldPos::ZERO, 1.0)
    }

    // --- Geometry: flow-pin hit-testing -----------------------------------

    #[test]
    fn flow_output_at_hits_only_within_grab_radius() {
        let n = node_layout(1, WorldPos::new(0.0, 0.0), true);
        let t = transform();
        let pin = n.flow_output_pin().unwrap();
        assert_eq!(
            flow_output_at(std::slice::from_ref(&n), &t, pin),
            Some((n.id, pin))
        );
        let far = WorldPos::new(pin.x + 1000.0, pin.y + 1000.0);
        assert_eq!(flow_output_at(std::slice::from_ref(&n), &t, far), None);
    }

    #[test]
    fn flow_output_at_is_none_without_flow_output() {
        let n = node_layout(1, WorldPos::new(0.0, 0.0), false);
        let t = transform();
        assert_eq!(
            flow_output_at(std::slice::from_ref(&n), &t, n.rect.center()),
            None
        );
    }

    #[test]
    fn flow_input_at_hits_only_within_grab_radius() {
        let s = stack_layout(1, WorldPos::new(0.0, 0.0), true);
        let t = transform();
        let pin = s.flow_input_pin().unwrap();
        assert_eq!(
            flow_input_at(std::slice::from_ref(&s), &t, pin),
            Some((s.id, pin))
        );
        let far = WorldPos::new(pin.x + 1000.0, pin.y - 1000.0);
        assert_eq!(flow_input_at(std::slice::from_ref(&s), &t, far), None);
    }

    // --- Multiple-link input behavior --------------------------------------

    #[test]
    fn port_accepts_multiple_links_reports_input_flag() {
        let mut n = node_layout(1, WorldPos::new(0.0, 0.0), false);
        n.inputs[0].accepts_multiple_links = true;
        let addr = PortAddr::new(n.id, n.inputs[0].id);
        assert!(port_accepts_multiple_links(std::slice::from_ref(&n), addr));
    }

    #[test]
    fn port_accepts_multiple_links_is_false_for_single_link_input() {
        let n = node_layout(1, WorldPos::new(0.0, 0.0), false);
        let addr = PortAddr::new(n.id, n.inputs[0].id);
        assert!(!port_accepts_multiple_links(std::slice::from_ref(&n), addr));
    }

    #[test]
    fn port_accepts_multiple_links_is_false_for_unresolvable_address() {
        let n = node_layout(1, WorldPos::new(0.0, 0.0), false);
        let missing = PortAddr::new(NodeId::new(2).unwrap(), n.inputs[0].id);
        assert!(!port_accepts_multiple_links(
            std::slice::from_ref(&n),
            missing
        ));
    }

    // --- Validation routing: flow_link_at nearest-link selection -----------

    struct OneFlowLinkViewer;
    impl GraphViewer for OneFlowLinkViewer {
        fn node_ids(&self) -> Vec<NodeId> {
            vec![NodeId::new(1).unwrap()]
        }
        fn node(&self, _id: NodeId) -> NodeDesc {
            NodeDesc::new("n").with_flow_output(true)
        }
        fn links(&self) -> Vec<Link> {
            Vec::new()
        }
        fn stacks(&self) -> Vec<StackDesc> {
            vec![StackDesc::new(StackId::new(1).unwrap(), "s").with_flow_input(true)]
        }
        fn flow_links(&self) -> Vec<FlowLink> {
            vec![FlowLink {
                from: NodeId::new(1).unwrap(),
                to: StackId::new(1).unwrap(),
            }]
        }
    }

    #[test]
    fn flow_link_at_finds_link_near_its_spline() {
        let viewer = OneFlowLinkViewer;
        let n = node_layout(1, WorldPos::new(0.0, 0.0), true);
        let s = stack_layout(1, WorldPos::new(0.0, 200.0), true);
        let t = transform();
        let points = flow_link_curve_points(
            std::slice::from_ref(&n),
            std::slice::from_ref(&s),
            &FlowLink {
                from: n.id,
                to: s.id,
            },
        )
        .unwrap();
        let midpoint = bezier_at(&points, 0.5);
        assert_eq!(
            flow_link_at(
                std::slice::from_ref(&n),
                std::slice::from_ref(&s),
                &viewer,
                &t,
                midpoint,
            ),
            Some(FlowLink {
                from: n.id,
                to: s.id,
            })
        );
    }

    #[test]
    fn flow_link_at_is_none_far_from_any_link() {
        let viewer = OneFlowLinkViewer;
        let n = node_layout(1, WorldPos::new(0.0, 0.0), true);
        let s = stack_layout(1, WorldPos::new(0.0, 200.0), true);
        let t = transform();
        let far = WorldPos::new(5000.0, 5000.0);
        assert_eq!(
            flow_link_at(
                std::slice::from_ref(&n),
                std::slice::from_ref(&s),
                &viewer,
                &t,
                far,
            ),
            None
        );
    }

    // --- Action generation: flow-link drop resolution -----------------------

    #[test]
    fn flow_link_drop_requests_link_from_node_anchor() {
        let action = flow_link_drop_action(
            FlowAnchor::Node(NodeId::new(1).unwrap()),
            Some(FlowTarget::Stack(StackId::new(2).unwrap())),
            None,
        );
        assert!(matches!(
            action,
            Some(GraphAction::FlowLinkRequested { from, to })
                if from == NodeId::new(1).unwrap() && to == StackId::new(2).unwrap()
        ));
    }

    #[test]
    fn flow_link_drop_from_node_anchor_cancels_without_target() {
        // An output-side (node) anchor never detaches: dropping with no
        // accepted target just cancels, regardless of any `detached` value.
        let action = flow_link_drop_action(FlowAnchor::Node(NodeId::new(1).unwrap()), None, None);
        assert!(action.is_none());
    }

    #[test]
    fn flow_link_drop_requests_rewire_from_stack_anchor() {
        let action = flow_link_drop_action(
            FlowAnchor::Stack(StackId::new(1).unwrap()),
            Some(FlowTarget::Node(NodeId::new(3).unwrap())),
            Some(FlowLink {
                from: NodeId::new(9).unwrap(),
                to: StackId::new(1).unwrap(),
            }),
        );
        assert!(matches!(
            action,
            Some(GraphAction::FlowLinkRequested { from, to })
                if from == NodeId::new(3).unwrap() && to == StackId::new(1).unwrap()
        ));
    }

    #[test]
    fn flow_link_drop_from_stack_anchor_detaches_without_target() {
        let old = FlowLink {
            from: NodeId::new(9).unwrap(),
            to: StackId::new(1).unwrap(),
        };
        let action =
            flow_link_drop_action(FlowAnchor::Stack(StackId::new(1).unwrap()), None, Some(old));
        assert!(matches!(
            action,
            Some(GraphAction::FlowLinkDeleteRequested { link }) if link == old
        ));
    }

    #[test]
    fn flow_link_drop_from_stack_anchor_is_none_without_target_or_detach() {
        let action = flow_link_drop_action(FlowAnchor::Stack(StackId::new(1).unwrap()), None, None);
        assert!(action.is_none());
    }
}
