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
    state::{CanvasDrag, DragItem, GraphView, RIGHT_CLICK_MAX_SECS, ReorderDrag},
    transform::{Transform, WorldPos, WorldRect},
    viewer::{GraphViewer, Link, LinkVerdict, NodeId, PortAddr, PortSide, StackId},
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

/// Stack whose bottom "Add" button contains `w`.
fn stack_add_button_at(stacks: &[StackLayout], w: WorldPos) -> Option<StackId> {
    stacks
        .iter()
        .rev()
        .find_map(|s| s.add_button.contains(w).then_some(s.id))
}

/// Topmost node whose header close button contains `w` (later-drawn wins).
fn close_button_at(layouts: &[NodeLayout], w: WorldPos) -> Option<NodeId> {
    layouts
        .iter()
        .rev()
        .find_map(|n| n.close_button.filter(|r| r.contains(w)).map(|_| n.id))
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
    pub node: Option<NodeId>,
    pub stack: Option<StackId>,
    /// Stack whose "Add" button is under the cursor this frame.
    pub add_button: Option<StackId>,
    /// Node whose header close button is under the cursor this frame.
    pub close: Option<NodeId>,
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
    /// Stacks currently under the in-progress marquee rectangle, previewed as
    /// pending selection.
    pub marquee_stacks: Vec<StackId>,
    /// Links currently crossing the in-progress marquee rectangle, previewed
    /// as pending selection.
    pub marquee_links: Vec<Link>,
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
    // A port under the cursor takes feedback priority over the node it sits
    // on: show a pin-specific highlight + connect cursor instead of the
    // node's grab/edge-highlight.
    let hovered_port = hover_world.and_then(|w| port_at_any(layouts, t, w));
    let hovered_node = if hovered_port.is_some() {
        None
    } else {
        hover_world.and_then(|w| node_at(layouts, w))
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
        if let Some((cand, center)) = port_at(layouts, t, w, anchor_side) {
            if cand != anchor {
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
        }
        None
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
    let hovered_close = if dragging {
        None
    } else {
        hover_world.and_then(|w| close_button_at(layouts, w))
    };

    if dragging {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
    } else if matches!(&link_target, Some(lt) if lt.verdict.is_err()) {
        // Hovering a target the consumer rejects: show it can't be dropped.
        ui.ctx().set_cursor_icon(egui::CursorIcon::NotAllowed);
    } else if hovered_close.is_some() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    } else if hovered_port.is_some() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
    } else if hovered_add_button.is_some() {
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
    // The gesture is gated on the pointer being geometrically inside the canvas
    // rather than on `response.hovered()`: the inline chip editors are drawn as
    // `Foreground` overlays, so once a pan slides one under the pointer egui's
    // layer-aware hover test reports the canvas as occluded and the pan would
    // freeze. A plain rect test keeps the gesture going over those overlays
    // (which don't consume scroll themselves).
    let pointer = ui.input(|i| i.pointer.hover_pos());
    if let Some(cursor) = pointer.filter(|p| response.rect.contains(*p)) {
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
            } else if close_button_at(layouts, w).is_some() {
                // The header close button is a click target; suppress drag so a
                // press there can't start moving the node. Click handled below.
            } else if let Some((addr, _)) = port_at(layouts, t, w, PortSide::Output) {
                view.interaction.pending_link_from = Some(addr);
            } else if let Some(existing) = port_at(layouts, t, w, PortSide::Input)
                .and_then(|(addr, _)| viewer.links().into_iter().find(|l| l.to == addr))
            {
                // Grabbing a connected input detaches its link and lets the
                // user rewire its source to another input.
                view.interaction.pending_link_from = Some(existing.from);
                view.interaction.detaching_link = Some(existing);
            } else if let Some((addr, _)) = port_at(layouts, t, w, PortSide::Input) {
                // Grabbing an unconnected input starts a link the user drags
                // out to an output. Inputs whose value is an inlined literal
                // have no link to detach, so this is the only way to wire
                // them up.
                view.interaction.pending_link_from = Some(addr);
                view.interaction.pending_from_input = true;
            } else if let Some(node) = node_at(layouts, w) {
                let layout = layouts.iter().find(|n| n.id == node);
                match layout.and_then(|n| n.stack) {
                    // Stack member: drag reorders it within its stack. Members
                    // are not part of the canvas selection — they move and
                    // delete by different rules than free nodes.
                    Some(sid) => {
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
                    // Free node: select it (unless already selected) and start
                    // a group drag of the whole canvas selection.
                    None => {
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
            } else if let Some((stack, _origin)) = stack_header_at(stacks, w) {
                // A stack is a canvas-movable unit like a free node: select it
                // (unless already selected) and group-drag the selection.
                if !view.selected_stacks.contains(&stack) {
                    if !shift {
                        view.clear_selection();
                    }
                    view.selected_stacks.insert(stack);
                    actions.push(GraphAction::SelectionChanged);
                }
                view.interaction.canvas_drag =
                    Some(begin_canvas_drag(view, DragItem::Stack(stack), w));
            } else {
                if !shift {
                    if view.clear_selection() {
                        actions.push(GraphAction::SelectionChanged);
                    }
                }
                view.interaction.box_select_start = Some(w);
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

    // --- End a primary drag ---
    if response.drag_stopped_by(PointerButton::Primary) {
        if let Some(drag) = view.interaction.canvas_drag.take() {
            for (node, _) in &drag.nodes {
                actions.push(GraphAction::NodeMoved {
                    node: *node,
                    to: view.position(*node),
                });
            }
            for (stack, _) in &drag.stacks {
                actions.push(GraphAction::StackMoved {
                    stack: *stack,
                    to: view.stack_position(*stack),
                });
            }
        }
        if let Some(rd) = view.interaction.reordering.take() {
            if rd.target_index != rd.from_index {
                actions.push(GraphAction::StackMemberMoved {
                    stack: rd.stack,
                    from_index: rd.from_index,
                    to_index: rd.target_index,
                });
            }
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
                // Anchor is an input pin; complete by dropping on an accepted
                // output, wiring that output's value into the input.
                if let Some(out) = accepted {
                    actions.push(GraphAction::LinkRequested {
                        from: out,
                        to: from,
                    });
                } else if !target_present && let Some(at) = drop_world {
                    // Dropped on empty canvas: offer to create a producer node
                    // and feed it into this input.
                    actions.push(GraphAction::LinkDropped {
                        source: from,
                        source_is_output: false,
                        at,
                    });
                }
            } else {
                match (accepted, detached, target_present) {
                    // Rewire a detached link onto a different, accepted input.
                    (Some(to), Some(old), _) if old.to != to => {
                        actions.push(GraphAction::LinkDeleteRequested { link: old });
                        actions.push(GraphAction::LinkRequested { from, to });
                    }
                    // Dropped a detached link back on its own input: no change.
                    (Some(_), Some(_), _) => {}
                    // New link from an output port to an accepted input.
                    (Some(to), None, _) => {
                        actions.push(GraphAction::LinkRequested { from, to });
                    }
                    // A detached link dropped on empty canvas is removed.
                    (None, Some(old), false) => {
                        actions.push(GraphAction::LinkDeleteRequested { link: old });
                    }
                    // Dropped on a rejected target: cancel, leaving any
                    // detached link intact.
                    (None, _, true) => {}
                    // A fresh output link dropped on empty canvas: offer to
                    // create a consumer node and feed this output into it.
                    (None, None, false) => {
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
        if let Some(start) = view.interaction.box_select_start.take() {
            if let Some(p) = response.interact_pointer_pos() {
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
                actions.push(GraphAction::SelectionChanged);
            }
        }
    }

    // --- Plain click: selection (nodes and edges) ---
    if response.clicked_by(PointerButton::Primary) {
        if let Some(p) = response.interact_pointer_pos() {
            let w = t.screen_to_world(p);
            let shift = ui.input(|i| i.modifiers.shift);
            if let Some(stack) = stack_add_button_at(stacks, w) {
                actions.push(GraphAction::StackAddRequested { stack });
            } else if let Some(node) = close_button_at(layouts, w) {
                // Header close button: delete just this node. The consumer maps
                // it to the right edit (remove a free node, or a stack member).
                actions.push(GraphAction::NodesDeleteRequested { nodes: vec![node] });
            } else if port_at(layouts, t, w, PortSide::Output).is_some()
                || port_at(layouts, t, w, PortSide::Input).is_some()
            {
                // Clicking a port is not a selection gesture.
            } else if let Some(node) = node_at(layouts, w) {
                // Clicking a free node selects the node; clicking a stack
                // member selects its parent stack (the stack is the unit).
                match layouts.iter().find(|n| n.id == node).and_then(|n| n.stack) {
                    Some(sid) => click_select_stack(view, sid, shift, actions),
                    None => click_select_node(view, node, shift, actions),
                }
            } else if let Some((stack, _)) = stack_header_at(stacks, w) {
                click_select_stack(view, stack, shift, actions);
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
    if ui.input(|i| i.pointer.button_released(PointerButton::Secondary)) {
        if let Some((press_pos, press_time)) = view.interaction.secondary_press.take() {
            let release_pos = ui
                .input(|i| i.pointer.interact_pos().or_else(|| i.pointer.latest_pos()))
                .unwrap_or(press_pos);
            if ui.input(|i| i.time) - press_time <= RIGHT_CLICK_MAX_SECS {
                actions.push(GraphAction::ContextMenu {
                    at: t.screen_to_world(release_pos),
                });
            }
        }
    }

    // --- Delete key removes the current selection (nodes, stacks and edges) ---
    let has_selection = !view.selection.is_empty()
        || !view.selected_links.is_empty()
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
    let (marquee, marquee_stacks, marquee_links) = match (
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
            (nodes, m_stacks, links)
        }
        _ => (Vec::new(), Vec::new(), Vec::new()),
    };

    Hover {
        node: hovered_node,
        stack: hovered_stack,
        add_button: hovered_add_button,
        close: hovered_close,
        port: hovered_port.map(|(_, c)| c),
        link_target,
        marquee,
        marquee_stacks,
        marquee_links,
    }
}

/// Apply a plain/shift click to free-node selection.
fn click_select_node(
    view: &mut GraphView,
    node: NodeId,
    shift: bool,
    actions: &mut Vec<GraphAction>,
) {
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
