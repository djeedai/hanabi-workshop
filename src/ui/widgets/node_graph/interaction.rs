//! Input handling: hit-testing, pan, zoom-to-cursor, node dragging with
//! optional grid snap, link dragging, selection (click + marquee) and
//! delete/context-menu intents. All hit-testing is done in world space.

use egui::PointerButton;

use super::layout::{NodeLayout, StackLayout, PORT_RADIUS, STACK_HEADER_H};
use super::response::GraphAction;
use super::state::{GraphView, ReorderDrag};
use super::transform::{Transform, WorldPos, WorldRect};
use super::viewer::{GraphViewer, Link, NodeId, PortAddr, PortSide, StackId};

/// Topmost node whose body contains `w` (later-drawn nodes win).
fn node_at(layouts: &[NodeLayout], w: WorldPos) -> Option<NodeId> {
    layouts.iter().rev().find(|n| n.rect.contains(w)).map(|n| n.id)
}

/// Stack whose header band contains `w`, returning its id and origin.
fn stack_header_at(stacks: &[StackLayout], w: WorldPos) -> Option<(StackId, WorldPos)> {
    stacks.iter().rev().find_map(|s| {
        let header = WorldRect::new(s.rect.min, s.rect.width, STACK_HEADER_H);
        header.contains(w).then_some((s.id, s.rect.min))
    })
}

/// Index a dragged member would land at within `stack` given the cursor's
/// world `y`: the count of the stack's *other* members whose vertical
/// center sits above the cursor.
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
fn port_at(layouts: &[NodeLayout], w: WorldPos, side: PortSide) -> Option<(PortAddr, WorldPos)> {
    let r = PORT_RADIUS * 1.8;
    let r2 = r * r;
    for node in layouts.iter().rev() {
        let ports = match side {
            PortSide::Input => &node.inputs,
            PortSide::Output => &node.outputs,
        };
        for p in ports {
            if p.center.distance_squared(w) <= r2 {
                return Some((PortAddr::new(node.id, p.id), p.center));
            }
        }
    }
    None
}

/// Cubic Bézier control points `[from, c1, c2, to]` of a link in world
/// space, or `None` if either endpoint port can't be resolved.
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
    p[0] * (mt * mt * mt) + p[1] * (3.0 * mt * mt * s) + p[2] * (3.0 * mt * s * s) + p[3] * (s * s * s)
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

/// Process all input for this frame. Mutates `view` (pan/zoom/positions/
/// selection/interaction) and pushes structural intents into `actions`.
/// What the pointer is hovering this frame, for render highlighting.
#[derive(Debug, Clone, Default)]
pub struct Hover {
    pub node: Option<NodeId>,
    pub stack: Option<StackId>,
    /// Nodes currently under the in-progress marquee rectangle. They render
    /// as hovered to preview what a drag-selection will capture.
    pub marquee: Vec<NodeId>,
    /// Links currently crossing the in-progress marquee rectangle, previewed
    /// as pending selection.
    pub marquee_links: Vec<Link>,
}

/// Process all input for this frame. Mutates `view` (pan/zoom/positions/
/// selection/interaction) and pushes structural intents into `actions`.
/// Returns what the pointer hovers for render highlighting.
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
    let hovered_node = hover_world.and_then(|w| node_at(layouts, w));
    let hovered_stack = hover_world.and_then(|w| stack_header_at(stacks, w).map(|(id, _)| id));

    // Grab cursor over anything draggable (free nodes move, stack members
    // reorder, stack headers move the whole stack); Grabbing while a drag
    // is active.
    let dragging = view.interaction.dragging_node.is_some()
        || view.interaction.dragging_stack.is_some()
        || view.interaction.reordering.is_some();
    if dragging {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
    } else if hovered_stack.is_some() || hovered_node.is_some() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
    }

    // --- Zoom to cursor (scroll while hovered) ---
    if response.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            if let Some(cursor) = response.hover_pos() {
                let w_before = t.screen_to_world(cursor);
                let factor = 1.1_f64.powf(scroll as f64 / 50.0);
                view.set_zoom_clamped(view.zoom * factor);
                let t2 = Transform::new(t.origin, view.pan, view.zoom);
                let w_after = t2.screen_to_world(cursor);
                view.pan += w_before - w_after;
            }
        }
    }

    // --- Pan (right- or middle-button drag, stateless) ---
    // egui distinguishes a drag from a click by movement threshold, so a
    // right-button *drag* pans while a right-button *click* (handled below)
    // opens the context menu.
    if response.dragged_by(PointerButton::Secondary)
        || response.dragged_by(PointerButton::Middle)
    {
        let d = response.drag_delta();
        view.pan -= t.screen_vec_to_world(d);
    }

    // --- Begin a primary-button drag: classify what was grabbed ---
    if response.drag_started_by(PointerButton::Primary) {
        if let Some(p) = response.interact_pointer_pos() {
            let w = t.screen_to_world(p);
            let shift = ui.input(|i| i.modifiers.shift);
            if let Some((addr, _)) = port_at(layouts, w, PortSide::Output) {
                view.interaction.pending_link_from = Some(addr);
            } else if let Some(existing) = port_at(layouts, w, PortSide::Input)
                .and_then(|(addr, _)| viewer.links().into_iter().find(|l| l.to == addr))
            {
                // Grabbing a connected input detaches its link and lets the
                // user rewire its source to another input.
                view.interaction.pending_link_from = Some(existing.from);
                view.interaction.detaching_link = Some(existing);
            } else if let Some((addr, _)) = port_at(layouts, w, PortSide::Input) {
                // Grabbing an unconnected input starts a link the user drags
                // out to an output. Inputs whose value is an inlined literal
                // have no link to detach, so this is the only way to wire
                // them up.
                view.interaction.pending_link_from = Some(addr);
                view.interaction.pending_from_input = true;
            } else if let Some(node) = node_at(layouts, w) {
                if !view.selection.contains(&node) {
                    if !shift {
                        view.selection.clear();
                        view.selected_links.clear();
                    }
                    view.selection.insert(node);
                    actions.push(GraphAction::SelectionChanged);
                }
                let layout = layouts.iter().find(|n| n.id == node);
                match layout.and_then(|n| n.stack) {
                    // Stack member: drag reorders it within its stack.
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
                    // Free node: drag moves it freely on the canvas.
                    None => {
                        let min = view.position(node);
                        view.interaction.dragging_node = Some((node, w - min));
                    }
                }
            } else if let Some((stack, origin)) = stack_header_at(stacks, w) {
                if !shift && view.clear_selection() {
                    actions.push(GraphAction::SelectionChanged);
                }
                view.interaction.dragging_stack = Some((stack, w - origin));
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
        if let (Some((node, off)), Some(p)) = (
            view.interaction.dragging_node,
            response.interact_pointer_pos(),
        ) {
            let w = t.screen_to_world(p);
            let mut new_min = w - off;
            if view.grid.snap {
                new_min = view.grid.snap_pos(new_min);
            }
            view.positions.insert(node, new_min);
        }
        if let (Some((stack, off)), Some(p)) = (
            view.interaction.dragging_stack,
            response.interact_pointer_pos(),
        ) {
            let w = t.screen_to_world(p);
            let mut new_origin = w - off;
            if view.grid.snap {
                new_origin = view.grid.snap_pos(new_origin);
            }
            view.stack_positions.insert(stack, new_origin);
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
        if let Some((node, _)) = view.interaction.dragging_node.take() {
            actions.push(GraphAction::NodeMoved {
                node,
                to: view.position(node),
            });
        }
        if let Some((stack, _)) = view.interaction.dragging_stack.take() {
            actions.push(GraphAction::StackMoved {
                stack,
                to: view.stack_position(stack),
            });
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
            let drop_w = response
                .interact_pointer_pos()
                .map(|p| t.screen_to_world(p));

            if from_input {
                // Anchor is an input pin; complete by dropping on an output,
                // wiring that output's value into the input.
                if let Some((out, _)) = drop_w.and_then(|w| port_at(layouts, w, PortSide::Output)) {
                    actions.push(GraphAction::LinkRequested { from: out, to: from });
                }
            } else {
                let dropped_on =
                    drop_w.and_then(|w| port_at(layouts, w, PortSide::Input).map(|(to, _)| to));
                match (dropped_on, detached) {
                    // Rewire a detached link onto a different input.
                    (Some(to), Some(old)) if old.to != to => {
                        actions.push(GraphAction::LinkDeleteRequested { link: old });
                        actions.push(GraphAction::LinkRequested { from, to });
                    }
                    // Dropped a detached link back on its own input: no change.
                    (Some(_), Some(_)) => {}
                    // New link from an output port to an input.
                    (Some(to), None) => {
                        actions.push(GraphAction::LinkRequested { from, to });
                    }
                    // A detached link dropped on empty canvas is removed.
                    (None, Some(old)) => {
                        actions.push(GraphAction::LinkDeleteRequested { link: old });
                    }
                    (None, None) => {}
                }
            }
        }
        if let Some(start) = view.interaction.box_select_start.take() {
            if let Some(p) = response.interact_pointer_pos() {
                let end = t.screen_to_world(p);
                let size = (end - start).abs();
                let rect = WorldRect::new(start.min(end), size.x, size.y);
                for node in layouts {
                    if rects_intersect(rect, node.rect) {
                        view.selection.insert(node.id);
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
            if port_at(layouts, w, PortSide::Output).is_some()
                || port_at(layouts, w, PortSide::Input).is_some()
            {
                // Clicking a port is not a selection gesture.
            } else if let Some(node) = node_at(layouts, w) {
                if shift {
                    if !view.selection.insert(node) {
                        view.selection.remove(&node);
                    }
                } else {
                    view.selection.clear();
                    view.selected_links.clear();
                    view.selection.insert(node);
                }
                actions.push(GraphAction::SelectionChanged);
            } else if let Some(link) = link_at(layouts, viewer, t, w) {
                if shift {
                    if !view.selected_links.insert(link) {
                        view.selected_links.remove(&link);
                    }
                } else {
                    view.selection.clear();
                    view.selected_links.clear();
                    view.selected_links.insert(link);
                }
                actions.push(GraphAction::SelectionChanged);
            } else if view.clear_selection() {
                actions.push(GraphAction::SelectionChanged);
            }
        }
    }

    // --- Right-click in place: context menu (a right-button *drag* pans
    //     instead, handled above; egui won't fire `secondary_clicked` for
    //     a drag) ---
    if response.secondary_clicked() {
        if let Some(p) = response.interact_pointer_pos().or_else(|| response.hover_pos()) {
            let w = t.screen_to_world(p);
            actions.push(GraphAction::ContextMenu { at: w });
        }
    }

    // --- Delete key removes the current selection (nodes and edges) ---
    let has_selection = !view.selection.is_empty() || !view.selected_links.is_empty();
    if (response.hovered() || response.has_focus()) && has_selection {
        let del = ui.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace));
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
        }
    }

    // Nodes and links under the in-progress marquee, previewed as pending
    // selection.
    let (marquee, marquee_links) = match (
        view.interaction.box_select_start,
        response.interact_pointer_pos(),
    ) {
        (Some(start), Some(p)) => {
            let end = t.screen_to_world(p);
            let size = (end - start).abs();
            let rect = WorldRect::new(start.min(end), size.x, size.y);
            let nodes = layouts
                .iter()
                .filter(|n| rects_intersect(rect, n.rect))
                .map(|n| n.id)
                .collect();
            let links = viewer
                .links()
                .into_iter()
                .filter(|l| link_in_rect(layouts, l, rect))
                .collect();
            (nodes, links)
        }
        _ => (Vec::new(), Vec::new()),
    };

    Hover {
        node: hovered_node,
        stack: hovered_stack,
        marquee,
        marquee_links,
    }
}

fn rects_intersect(a: WorldRect, b: WorldRect) -> bool {
    a.min.cmple(b.max()).all() && a.max().cmpge(b.min).all()
}
