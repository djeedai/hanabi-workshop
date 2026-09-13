//! Input handling: hit-testing, panning, zooming and selection.
//!
//! Covers pan (right/middle drag or two-finger scroll), zoom-to-cursor (pinch
//! or modifier-scroll), container and free-node dragging with optional grid
//! snap, link dragging, section folding, member reordering, selection
//! (click + marquee) and delete/context-menu intents. All hit-testing is done
//! in world space.

use egui::PointerButton;

use super::{
    layout::{ContainerLayout, NodeLayout, SectionLayout, port_grab_radius_world},
    response::GraphAction,
    state::{CanvasDrag, CanvasItem, DragItem, GraphView, RIGHT_CLICK_MAX_SECS, ReorderDrag},
    transform::{Transform, WorldPos, WorldRect},
    viewer::{ContainerId, GraphViewer, Link, LinkVerdict, NodeId, PortAddr, PortSide, SectionId},
};

/// Every section of every container, front-to-back.
fn sections(containers: &[ContainerLayout]) -> impl Iterator<Item = &SectionLayout> {
    containers.iter().rev().flat_map(|c| c.sections.iter())
}

/// Topmost interactive node whose body contains `w` (later-drawn nodes win).
///
/// Members of a folded section are skipped: they are link anchors only.
fn node_at(layouts: &[NodeLayout], w: WorldPos) -> Option<NodeId> {
    layouts
        .iter()
        .rev()
        .find(|n| n.interactive() && n.rect.contains(w))
        .map(|n| n.id)
}

/// Topmost container whose frame contains `w`.
fn container_at(containers: &[ContainerLayout], w: WorldPos) -> Option<ContainerId> {
    containers
        .iter()
        .rev()
        .find(|c| c.rect.contains(w))
        .map(|c| c.id)
}

/// Section whose header band contains `w`.
fn section_header_at(containers: &[ContainerLayout], w: WorldPos) -> Option<SectionId> {
    sections(containers).find_map(|s| s.header.contains(w).then_some(s.id))
}

/// A press target resolved between a node body and a container frame.
enum Grab {
    /// A node body (free node or section member).
    Node(NodeId),
    /// Container chrome — its header, a section header, or any frame area not
    /// covered by a member — with the container's origin.
    Container(ContainerId, WorldPos),
}

/// Frontmost of the node body / container frame under `w`, resolved by
/// z-order.
///
/// A container's chrome belongs to the whole pipeline unit; a node belongs to
/// its container (as a member) or to itself (when free). Whichever unit paints
/// in front owns the press, so grabbing a raised container's header isn't
/// stolen by a free node drawn beneath it. Within one container a member body
/// always wins over the frame behind it, since the two share a z key.
fn grab_at(
    view: &GraphView,
    layouts: &[NodeLayout],
    containers: &[ContainerLayout],
    w: WorldPos,
) -> Option<Grab> {
    let node = node_at(layouts, w);
    let container = container_at(containers, w);
    let node_z = node.map(|n| {
        let item = layouts
            .iter()
            .find(|l| l.id == n)
            .and_then(|l| l.container)
            .map(CanvasItem::Container)
            .unwrap_or(CanvasItem::Node(n));
        view.z_key(item)
    });
    let container_z = container.map(|c| view.z_key(CanvasItem::Container(c)));
    let origin_of = |id: ContainerId| {
        containers
            .iter()
            .find(|c| c.id == id)
            .map(|c| c.rect.min)
            .unwrap_or(WorldPos::ZERO)
    };
    match (node, container) {
        (Some(n), Some(c)) => {
            if container_z > node_z {
                Some(Grab::Container(c, origin_of(c)))
            } else {
                Some(Grab::Node(n))
            }
        }
        (Some(n), None) => Some(Grab::Node(n)),
        (None, Some(c)) => Some(Grab::Container(c, origin_of(c))),
        (None, None) => None,
    }
}

/// Section whose bottom "Add" button contains `w`.
fn section_add_button_at(containers: &[ContainerLayout], w: WorldPos) -> Option<SectionId> {
    sections(containers).find_map(|s| s.add_button.filter(|r| r.contains(w)).map(|_| s.id))
}

/// Section whose header expand/collapse-all-members button contains `w`.
fn section_collapse_all_at(containers: &[ContainerLayout], w: WorldPos) -> Option<SectionId> {
    sections(containers).find_map(|s| {
        s.collapse_all_button
            .filter(|r| r.contains(w))
            .map(|_| s.id)
    })
}

/// Section whose header fold chevron contains `w`.
fn section_toggle_at(containers: &[ContainerLayout], w: WorldPos) -> Option<SectionId> {
    sections(containers).find_map(|s| s.toggle.contains(w).then_some(s.id))
}

/// Container whose header close button contains `w`.
fn container_close_at(containers: &[ContainerLayout], w: WorldPos) -> Option<ContainerId> {
    containers
        .iter()
        .rev()
        .find_map(|c| c.close_button.filter(|r| r.contains(w)).map(|_| c.id))
}

/// Topmost node whose header close button contains `w` (later-drawn wins).
fn close_button_at(layouts: &[NodeLayout], w: WorldPos) -> Option<NodeId> {
    layouts.iter().rev().find_map(|n| {
        (n.interactive() && n.close_button.is_some_and(|r| r.contains(w))).then_some(n.id)
    })
}

/// Topmost member whose header collapse chevron contains `w`.
fn collapse_toggle_at(layouts: &[NodeLayout], w: WorldPos) -> Option<NodeId> {
    layouts.iter().rev().find_map(|n| {
        (n.interactive() && n.collapse_toggle.is_some_and(|r| r.contains(w))).then_some(n.id)
    })
}

/// Index a dragged member would land at within `section`.
///
/// Given the cursor's world `y`: the count of the section's *other* members
/// whose vertical center sits above the cursor.
fn reorder_target_index(
    layouts: &[NodeLayout],
    section: SectionId,
    dragged: NodeId,
    cursor_y: f64,
) -> usize {
    layouts
        .iter()
        .filter(|n| n.section == Some(section) && n.id != dragged)
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
            // A collapsed member's pins — and every pin of a folded section's
            // hidden members — are folded onto one point and aren't
            // individually connectable; expand to wire specific fields.
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
    pub container: Option<ContainerId>,
    /// Section whose header band is under the cursor this frame.
    pub section: Option<SectionId>,
    /// Section whose "Add" button is under the cursor this frame.
    pub add_button: Option<SectionId>,
    /// Section whose header expand/collapse-all-members button is under the
    /// cursor.
    pub collapse_all: Option<SectionId>,
    /// Section whose header fold chevron is under the cursor.
    pub section_toggle: Option<SectionId>,
    /// Node whose header close button is under the cursor this frame.
    pub close: Option<NodeId>,
    /// Container whose header close button is under the cursor this frame.
    pub container_close: Option<ContainerId>,
    /// World center of a port under the cursor (within grab tolerance), for
    /// drawing a pin-specific hover highlight.
    pub port: Option<WorldPos>,
    /// During a link drag, the validated candidate target under the cursor
    /// (if any), used to snap the spline, blend toward the target's color, and
    /// show a rejection reason.
    pub link_target: Option<LinkTarget>,
    /// Nodes currently under the in-progress marquee rectangle. They render
    /// as hovered to preview what a drag-selection will capture.
    pub marquee: Vec<NodeId>,
    /// Containers currently under the in-progress marquee rectangle, previewed
    /// as pending selection.
    pub marquee_containers: Vec<ContainerId>,
    /// Links currently crossing the in-progress marquee rectangle, previewed
    /// as pending selection.
    pub marquee_links: Vec<Link>,
}

/// Process all input for this frame.
///
/// Mutates `view` (pan/zoom/positions/folding/selection/interaction) and pushes
/// structural intents into `actions`. Returns what the pointer hovers for
/// render highlighting.
pub fn handle(
    ui: &egui::Ui,
    response: &egui::Response,
    t: &Transform,
    layouts: &[NodeLayout],
    containers: &[ContainerLayout],
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
    let hovered_node = if hovered_port.is_some() {
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

    // Grab cursor over anything draggable (free nodes and containers move,
    // section members reorder); Grabbing while a drag is active; Crosshair
    // over a port (start/complete a connection).
    let dragging = view.interaction.canvas_drag.is_some() || view.interaction.reordering.is_some();
    // Containers don't accept drops and never interact with a dragged node, so
    // suppress their hover highlight mid-drag — lighting one up implies a
    // relationship that doesn't exist. Outside a drag the highlight follows the
    // press target, so hovering a member lifts the member alone and hovering
    // the pipeline's own chrome lifts the whole frame.
    let hovered_container = if dragging {
        None
    } else {
        hover_world.and_then(|w| match grab_at(view, layouts, containers, w) {
            Some(Grab::Container(id, _)) => Some(id),
            _ => None,
        })
    };
    let hovered_section = if dragging {
        None
    } else {
        hover_world.and_then(|w| section_header_at(containers, w))
    };
    let hovered_add_button = if dragging {
        None
    } else {
        hover_world.and_then(|w| section_add_button_at(containers, w))
    };
    let hovered_collapse_all = if dragging {
        None
    } else {
        hover_world.and_then(|w| section_collapse_all_at(containers, w))
    };
    let hovered_section_toggle = if dragging {
        None
    } else {
        hover_world.and_then(|w| section_toggle_at(containers, w))
    };
    let hovered_close = if dragging {
        None
    } else {
        hover_world.and_then(|w| close_button_at(layouts, w))
    };
    let hovered_container_close = if dragging {
        None
    } else {
        hover_world.and_then(|w| container_close_at(containers, w))
    };
    let hovered_toggle = if dragging {
        None
    } else {
        hover_world.and_then(|w| collapse_toggle_at(layouts, w))
    };

    if dragging {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
    } else if matches!(&link_target, Some(lt) if lt.verdict.is_err()) {
        // Hovering a target the consumer rejects: show it can't be dropped.
        ui.ctx().set_cursor_icon(egui::CursorIcon::NotAllowed);
    } else if hovered_close.is_some()
        || hovered_container_close.is_some()
        || hovered_toggle.is_some()
        || hovered_section_toggle.is_some()
    {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    } else if hovered_port.is_some() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
    } else if hovered_add_button.is_some() || hovered_collapse_all.is_some() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    } else if hovered_container.is_some() || hovered_node.is_some() {
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
            if section_add_button_at(containers, w).is_some() {
                // The bottom "Add" button is a click target; a press there must
                // not begin a marquee or canvas drag. The click is emitted in
                // the click handler below.
            } else if section_collapse_all_at(containers, w).is_some()
                || section_toggle_at(containers, w).is_some()
            {
                // Section header buttons are click targets; suppress drag so a
                // press there can't start moving the container.
            } else if container_close_at(containers, w).is_some() {
                // The container close button is a click target; suppress drag.
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
            } else {
                // Node body vs container chrome is resolved by z-order so the
                // frontmost unit under the cursor owns the press.
                match grab_at(view, layouts, containers, w) {
                    Some(Grab::Node(node)) => {
                        let layout = layouts.iter().find(|n| n.id == node);
                        let owning = layout.and_then(|n| n.section).and_then(|sid| {
                            containers
                                .iter()
                                .flat_map(|c| c.sections.iter())
                                .find(|s| s.id == sid)
                        });
                        match owning {
                            // Section member: drag reorders it within its
                            // section. Members are not part of the canvas
                            // selection — they move and delete by different
                            // rules than free nodes.
                            Some(section) => {
                                view.raise(CanvasItem::Container(section.container));
                                if let Some((from_index, grab_offset)) = section
                                    .members
                                    .iter()
                                    .position(|m| *m == node)
                                    .zip(layout.map(|n| w - n.rect.min))
                                {
                                    view.interaction.reordering = Some(ReorderDrag {
                                        section: section.id,
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
                    Some(Grab::Container(container, _origin)) => {
                        // A container is a canvas-movable unit like a free
                        // node: select it (unless already selected) and
                        // group-drag the selection.
                        view.raise(CanvasItem::Container(container));
                        if !view.selected_containers.contains(&container) {
                            if !shift {
                                view.clear_selection();
                            }
                            view.selected_containers.insert(container);
                            actions.push(GraphAction::SelectionChanged);
                        }
                        view.interaction.canvas_drag =
                            Some(begin_canvas_drag(view, DragItem::Container(container), w));
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
            for (id, origin) in &drag.containers {
                view.container_positions.insert(*id, *origin + delta);
            }
        }
        if let (Some(mut rd), Some(p)) =
            (view.interaction.reordering, response.interact_pointer_pos())
        {
            let cursor_y = t.screen_to_world(p).y;
            rd.target_index = reorder_target_index(layouts, rd.section, rd.node, cursor_y);
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
        view.interaction.box_select_start = None;
        view.interaction.reordering = None;
        if let Some(drag) = view.interaction.canvas_drag.take() {
            // Snap every dragged item back to where the drag began.
            for (id, origin) in &drag.nodes {
                view.positions.insert(*id, *origin);
            }
            for (id, origin) in &drag.containers {
                view.container_positions.insert(*id, *origin);
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
            for (container, origin) in &drag.containers {
                actions.push(GraphAction::ContainerMoved {
                    container: *container,
                    from: *origin,
                    to: view.container_position(*container),
                });
            }
        }
        if let Some(rd) = view.interaction.reordering.take()
            && rd.target_index != rd.from_index
        {
            actions.push(GraphAction::SectionMemberMoved {
                section: rd.section,
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
        if let Some(start) = view.interaction.box_select_start.take()
            && let Some(p) = response.interact_pointer_pos()
        {
            let end = t.screen_to_world(p);
            let size = (end - start).abs();
            let rect = WorldRect::new(start.min(end), size.x, size.y);
            // The marquee captures the canvas's movable units — free nodes
            // and whole containers — plus links. Section members are excluded;
            // they reorder within their section rather than move freely.
            for node in layouts {
                if node.container.is_none() && rects_intersect(rect, node.rect) {
                    view.selection.insert(node.id);
                }
            }
            for container in containers {
                if rects_intersect(rect, container.rect) {
                    view.selected_containers.insert(container.id);
                }
            }
            for link in viewer.links() {
                if link_in_rect(layouts, &link, rect) {
                    view.selected_links.insert(link);
                }
            }
            actions.push(GraphAction::SelectionChanged);
        }
    }

    // --- Plain click: selection (nodes, containers and edges) ---
    if response.clicked_by(PointerButton::Primary)
        && let Some(p) = response.interact_pointer_pos()
    {
        let w = t.screen_to_world(p);
        let shift = ui.input(|i| i.modifiers.shift);
        if let Some(section) = section_add_button_at(containers, w) {
            actions.push(GraphAction::SectionAddRequested { section });
        } else if let Some(sid) = section_toggle_at(containers, w) {
            // Section header chevron: fold/unfold the whole phase. Persisted
            // view state, so it's applied directly with no action emitted.
            view.toggle_section_collapsed(sid);
        } else if let Some(sid) = section_collapse_all_at(containers, w) {
            // Fold or unfold every collapsible member of the section at once.
            // Collapse when any is expanded, otherwise expand them all.
            let members: Vec<NodeId> = layouts
                .iter()
                .filter(|n| n.section == Some(sid) && n.collapse_toggle.is_some())
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
        } else if let Some(container) = container_close_at(containers, w) {
            // Container close button: delete the whole pipeline. The consumer
            // decides what that removes in its own model.
            actions.push(GraphAction::ContainersDeleteRequested {
                containers: vec![container],
            });
        } else if let Some(node) = collapse_toggle_at(layouts, w) {
            // Header chevron: fold/unfold this member. Pure view state, so
            // it's applied directly with no action emitted.
            view.toggle_collapsed(node);
        } else if let Some(node) = close_button_at(layouts, w) {
            // Header close button: delete just this node. The consumer maps
            // it to the right edit (remove a free node, or a section member).
            actions.push(GraphAction::NodesDeleteRequested { nodes: vec![node] });
        } else if port_at(layouts, t, w, PortSide::Output).is_some()
            || port_at(layouts, t, w, PortSide::Input).is_some()
        {
            // Clicking a port is not a selection gesture.
        } else if let Some(grab) = grab_at(view, layouts, containers, w) {
            // Node body vs container chrome is resolved by z-order. Clicking a
            // free node selects the node; clicking a section member or any
            // container chrome selects the container (the pipeline is the
            // unit).
            match grab {
                Grab::Node(node) => {
                    match layouts
                        .iter()
                        .find(|n| n.id == node)
                        .and_then(|n| n.container)
                    {
                        Some(cid) => click_select_container(view, cid, shift, actions),
                        None => click_select_node(view, node, shift, actions),
                    }
                }
                Grab::Container(container, _) => {
                    click_select_container(view, container, shift, actions)
                }
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

    // --- Delete key removes the current selection (nodes, containers, edges) ---
    let has_selection = !view.selection.is_empty()
        || !view.selected_links.is_empty()
        || !view.selected_containers.is_empty();
    if (response.hovered() || response.has_focus()) && has_selection {
        let del =
            ui.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace));
        if del {
            if !view.selected_links.is_empty() {
                for link in view.selected_links.drain() {
                    actions.push(GraphAction::LinkDeleteRequested { link });
                }
            }
            if !view.selection.is_empty() {
                actions.push(GraphAction::NodesDeleteRequested {
                    nodes: view.selection.drain().collect(),
                });
            }
            if !view.selected_containers.is_empty() {
                actions.push(GraphAction::ContainersDeleteRequested {
                    containers: view.selected_containers.drain().collect(),
                });
            }
        }
    }

    // Free nodes, containers and links under the in-progress marquee, previewed
    // as pending selection (section members are not marquee-selectable).
    let (marquee, marquee_containers, marquee_links) = match (
        view.interaction.box_select_start,
        response.interact_pointer_pos(),
    ) {
        (Some(start), Some(p)) => {
            let end = t.screen_to_world(p);
            let size = (end - start).abs();
            let rect = WorldRect::new(start.min(end), size.x, size.y);
            let nodes = layouts
                .iter()
                .filter(|n| n.container.is_none() && rects_intersect(rect, n.rect))
                .map(|n| n.id)
                .collect();
            let m_containers = containers
                .iter()
                .filter(|c| rects_intersect(rect, c.rect))
                .map(|c| c.id)
                .collect();
            let links = viewer
                .links()
                .into_iter()
                .filter(|l| link_in_rect(layouts, l, rect))
                .collect();
            (nodes, m_containers, links)
        }
        _ => (Vec::new(), Vec::new(), Vec::new()),
    };

    Hover {
        node_body: node_under_pointer,
        node: hovered_node,
        container: hovered_container,
        section: hovered_section,
        add_button: hovered_add_button,
        collapse_all: hovered_collapse_all,
        section_toggle: hovered_section_toggle,
        close: hovered_close,
        container_close: hovered_container_close,
        port: hovered_port.map(|(_, c)| c),
        link_target,
        marquee,
        marquee_containers,
        marquee_links,
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

/// Apply a plain/shift click to container selection.
fn click_select_container(
    view: &mut GraphView,
    container: ContainerId,
    shift: bool,
    actions: &mut Vec<GraphAction>,
) {
    view.raise(CanvasItem::Container(container));
    if shift {
        if !view.selected_containers.insert(container) {
            view.selected_containers.remove(&container);
        }
    } else {
        view.clear_selection();
        view.selected_containers.insert(container);
    }
    actions.push(GraphAction::SelectionChanged);
}

/// Capture the current canvas selection as a rigid group drag.
///
/// Captures free nodes + containers anchored on `primary`, grabbed at world
/// point `grab_world`.
fn begin_canvas_drag(view: &GraphView, primary: DragItem, grab_world: WorldPos) -> CanvasDrag {
    let nodes = view
        .selection
        .iter()
        .map(|&id| (id, view.position(id)))
        .collect();
    let containers = view
        .selected_containers
        .iter()
        .map(|&id| (id, view.container_position(id)))
        .collect();
    let primary_origin = match primary {
        DragItem::Node(id) => view.position(id),
        DragItem::Container(id) => view.container_position(id),
    };
    CanvasDrag {
        primary_origin,
        grab_offset: grab_world - primary_origin,
        nodes,
        containers,
    }
}

fn rects_intersect(a: WorldRect, b: WorldRect) -> bool {
    a.min.cmple(b.max()).all() && a.max().cmpge(b.min).all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        layout::{self, CONTAINER_HEADER_H, SECTION_HEADER_H},
        viewer::{ContainerDesc, NodeDesc, PortDesc, SectionDesc},
    };

    const CONTAINER: u32 = 1;
    const INIT_SECTION: u32 = 2;
    const RENDER_SECTION: u32 = 3;
    const INIT_A: u32 = 4;
    const INIT_B: u32 = 5;
    const FREE: u32 = 6;

    fn container_id() -> ContainerId {
        ContainerId::new(CONTAINER).unwrap()
    }

    fn node_id(raw: u32) -> NodeId {
        NodeId::new(raw).unwrap()
    }

    /// One closable container with two add-capable sections, plus a free node.
    struct PipelineViewer;

    impl GraphViewer for PipelineViewer {
        fn node_ids(&self) -> Vec<NodeId> {
            [INIT_A, INIT_B, FREE].into_iter().map(node_id).collect()
        }

        fn node(&self, _id: NodeId) -> NodeDesc {
            NodeDesc::new("member")
                .with_inputs(vec![PortDesc::new("in")])
                .with_outputs(vec![PortDesc::new("out")])
                .closable()
        }

        fn links(&self) -> Vec<Link> {
            Vec::new()
        }

        fn containers(&self) -> Vec<ContainerDesc> {
            vec![
                ContainerDesc::new(container_id(), "Pipeline")
                    .closable()
                    .with_sections(vec![
                        SectionDesc::new(SectionId::new(INIT_SECTION).unwrap(), "Init")
                            .with_members(vec![node_id(INIT_A), node_id(INIT_B)])
                            .with_add_member(true),
                        SectionDesc::new(SectionId::new(RENDER_SECTION).unwrap(), "Render")
                            .with_add_member(true),
                    ]),
            ]
        }
    }

    /// A view with the free node parked clear of the container, so hit-tests
    /// exercise container chrome rather than an incidental overlap.
    fn base_view() -> GraphView {
        let mut view = GraphView::default();
        view.ensure_position(node_id(FREE), WorldPos::new(600.0, 0.0));
        view
    }

    fn laid_out(view: &GraphView) -> layout::GraphLayout {
        layout::compute(&PipelineViewer, view)
    }

    fn transform() -> Transform {
        Transform::new(egui::pos2(0.0, 0.0), WorldPos::ZERO, 1.0)
    }

    #[test]
    fn container_chrome_owns_presses_outside_member_bodies() {
        let view = base_view();
        let layout = laid_out(&view);
        let container = layout.container(container_id()).unwrap();

        // The outer header belongs to the container...
        let header_point = WorldPos::new(
            container.rect.center().x,
            container.rect.min.y + CONTAINER_HEADER_H * 0.5,
        );
        assert!(matches!(
            grab_at(&view, &layout.nodes, &layout.containers, header_point),
            Some(Grab::Container(id, _)) if id == container_id()
        ));

        // ...as does a section header, while a member body owns its own press.
        let init = layout
            .section(SectionId::new(INIT_SECTION).unwrap())
            .unwrap();
        assert!(matches!(
            grab_at(&view, &layout.nodes, &layout.containers, init.header.center()),
            Some(Grab::Container(id, _)) if id == container_id()
        ));
        let member = layout
            .nodes
            .iter()
            .find(|n| n.id == node_id(INIT_A))
            .unwrap();
        assert!(matches!(
            grab_at(&view, &layout.nodes, &layout.containers, member.rect.center()),
            Some(Grab::Node(id)) if id == node_id(INIT_A)
        ));
    }

    #[test]
    fn section_chrome_hit_tests_resolve_by_button() {
        let view = base_view();
        let layout = laid_out(&view);
        let init = layout
            .section(SectionId::new(INIT_SECTION).unwrap())
            .unwrap();

        assert_eq!(
            section_toggle_at(&layout.containers, init.toggle.center()),
            Some(init.id)
        );
        assert_eq!(
            section_add_button_at(&layout.containers, init.add_button.unwrap().center()),
            Some(init.id)
        );
        assert_eq!(
            section_collapse_all_at(
                &layout.containers,
                init.collapse_all_button.unwrap().center()
            ),
            Some(init.id)
        );
        assert_eq!(
            section_header_at(&layout.containers, init.header.center()),
            Some(init.id)
        );
        let container = layout.container(container_id()).unwrap();
        assert_eq!(
            container_close_at(&layout.containers, container.close_button.unwrap().center()),
            Some(container_id())
        );
    }

    #[test]
    fn folded_section_members_are_not_interactive() {
        let mut view = base_view();
        view.toggle_section_collapsed(SectionId::new(INIT_SECTION).unwrap());
        let layout = laid_out(&view);
        let init = layout
            .section(SectionId::new(INIT_SECTION).unwrap())
            .unwrap();
        let anchor = init.fold_input_anchor();

        // Hidden members never win a body press, a close-button press or a
        // port grab, even though their pins resolve to the fold anchors.
        assert_eq!(node_at(&layout.nodes, anchor), None);
        assert_eq!(close_button_at(&layout.nodes, anchor), None);
        assert_eq!(collapse_toggle_at(&layout.nodes, anchor), None);
        assert_eq!(
            port_at(&layout.nodes, &transform(), anchor, PortSide::Input),
            None
        );
        assert!(matches!(
            grab_at(&view, &layout.nodes, &layout.containers, anchor),
            Some(Grab::Container(id, _)) if id == container_id()
        ));
        // The section header still fills the folded section's whole height.
        assert_eq!(init.rect.height, SECTION_HEADER_H);
    }

    #[test]
    fn reorder_target_index_counts_members_above_the_cursor() {
        let view = base_view();
        let layout = laid_out(&view);
        let section = SectionId::new(INIT_SECTION).unwrap();
        let b = layout
            .nodes
            .iter()
            .find(|n| n.id == node_id(INIT_B))
            .unwrap();

        // Dragging A down past B's center lands it after B.
        assert_eq!(
            reorder_target_index(
                &layout.nodes,
                section,
                node_id(INIT_A),
                b.rect.center().y + 1.0
            ),
            1
        );
        // Dragging A while still above B's center leaves it first.
        assert_eq!(
            reorder_target_index(&layout.nodes, section, node_id(INIT_A), b.rect.min.y - 1.0),
            0
        );
    }

    #[test]
    fn canvas_drag_moves_every_selected_container_and_node() {
        let mut view = base_view();
        view.ensure_container_position(container_id(), WorldPos::new(10.0, 10.0));
        view.positions
            .insert(node_id(FREE), WorldPos::new(500.0, 40.0));
        view.selected_containers.insert(container_id());
        view.selection.insert(node_id(FREE));

        let drag = begin_canvas_drag(
            &view,
            DragItem::Container(container_id()),
            WorldPos::new(20.0, 20.0),
        );
        assert_eq!(drag.primary_origin, WorldPos::new(10.0, 10.0));
        assert_eq!(drag.grab_offset, WorldPos::new(10.0, 10.0));
        assert_eq!(
            drag.containers,
            vec![(container_id(), WorldPos::new(10.0, 10.0))]
        );
        assert_eq!(
            drag.nodes,
            vec![(node_id(FREE), WorldPos::new(500.0, 40.0))]
        );
    }

    /// [`PipelineViewer`] plus one link from the free node's output into the
    /// first Init member's input.
    struct LinkedViewer;

    fn test_link() -> Link {
        Link {
            from: PortAddr::new(node_id(FREE), crate::viewer::PortId::output(0)),
            to: PortAddr::new(node_id(INIT_A), crate::viewer::PortId::input(0)),
        }
    }

    impl GraphViewer for LinkedViewer {
        fn node_ids(&self) -> Vec<NodeId> {
            PipelineViewer.node_ids()
        }
        fn node(&self, id: NodeId) -> NodeDesc {
            PipelineViewer.node(id)
        }
        fn links(&self) -> Vec<Link> {
            vec![test_link()]
        }
        fn containers(&self) -> Vec<ContainerDesc> {
            PipelineViewer.containers()
        }
    }

    #[test]
    fn a_link_into_a_folded_section_rides_its_boundary_anchor() {
        let mut view = base_view();
        view.toggle_section_collapsed(SectionId::new(INIT_SECTION).unwrap());
        let layout = layout::compute(&LinkedViewer, &view);
        let init = layout
            .section(SectionId::new(INIT_SECTION).unwrap())
            .unwrap();

        // The hidden member's pin resolves to the section's fold anchor, so the
        // edge stays drawn — and stays pickable — on the section boundary.
        let points = link_curve_points(&layout.nodes, &test_link()).expect("link endpoints");
        assert_eq!(points[3], init.fold_input_anchor());
        let midpoint = bezier_at(&points, 0.5);
        assert_eq!(
            link_at(&layout.nodes, &LinkedViewer, &transform(), midpoint),
            Some(test_link())
        );
    }

    #[test]
    fn marquee_captures_containers_but_not_their_members() {
        let view = base_view();
        let layout = laid_out(&view);
        let container = layout.container(container_id()).unwrap();
        let rect = WorldRect::new(
            container.rect.min - WorldPos::new(5.0, 5.0),
            container.rect.width + 10.0,
            container.rect.height + 10.0,
        );
        let members: Vec<NodeId> = layout
            .nodes
            .iter()
            .filter(|n| n.container.is_none() && rects_intersect(rect, n.rect))
            .map(|n| n.id)
            .collect();
        assert!(members.iter().all(|id| *id == node_id(FREE)));
        assert!(rects_intersect(rect, container.rect));
    }
}
