//! Input handling: hit-testing, pan, zoom-to-cursor, node dragging with
//! optional grid snap, link dragging, selection (click + marquee) and
//! delete/context-menu intents. All hit-testing is done in world space.

use egui::PointerButton;

use super::layout::{NodeLayout, StackLayout, PORT_RADIUS, STACK_HEADER_H};
use super::response::GraphAction;
use super::state::GraphView;
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

/// Minimum distance (world units) from `w` to a link's spline, sampled.
fn link_distance(layouts: &[NodeLayout], link: &Link, w: WorldPos) -> Option<f64> {
    let by = |id: NodeId| layouts.iter().find(|n| n.id == id);
    let from = by(link.from.node)?.port_center(link.from.port)?;
    let to = by(link.to.node)?.port_center(link.to.port)?;
    let handle = ((to.x - from.x).abs() * 0.5).clamp(24.0, 160.0);
    let c1 = from + WorldPos::new(handle, 0.0);
    let c2 = to - WorldPos::new(handle, 0.0);
    let mut best = f64::INFINITY;
    let steps = 18;
    for i in 0..=steps {
        let s = i as f64 / steps as f64;
        let mt = 1.0 - s;
        let point =
            from * (mt * mt * mt) + c1 * (3.0 * mt * mt * s) + c2 * (3.0 * mt * s * s) + to * (s * s * s);
        best = best.min(point.distance_squared(w));
    }
    Some(best.sqrt())
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
#[derive(Debug, Clone, Copy, Default)]
pub struct Hover {
    pub node: Option<NodeId>,
    pub stack: Option<StackId>,
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

    // A free node is move-draggable from anywhere on its body.
    let hovered_free_node = hovered_node.filter(|id| {
        layouts
            .iter()
            .find(|n| n.id == *id)
            .is_some_and(|n| n.stack.is_none())
    });

    // Grab cursor over anything draggable; Grabbing while a drag is active.
    if view.interaction.dragging_stack.is_some() || view.interaction.dragging_node.is_some() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
    } else if hovered_stack.is_some() || hovered_free_node.is_some() {
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
            } else if let Some(node) = node_at(layouts, w) {
                if !view.selection.contains(&node) {
                    if !shift {
                        view.selection.clear();
                        view.selected_links.clear();
                    }
                    view.selection.insert(node);
                    actions.push(GraphAction::SelectionChanged);
                }
                // Only free nodes drag on the canvas; stacked members are
                // positioned by their stack.
                let is_free = layouts
                    .iter()
                    .find(|n| n.id == node)
                    .map_or(true, |n| n.stack.is_none());
                if is_free {
                    let min = view.position(node);
                    view.interaction.dragging_node = Some((node, w - min));
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
        if let Some(from) = view.interaction.pending_link_from.take() {
            if let Some(p) = response.interact_pointer_pos() {
                let w = t.screen_to_world(p);
                if let Some((to, _)) = port_at(layouts, w, PortSide::Input) {
                    actions.push(GraphAction::LinkRequested { from, to });
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
                    nodes: view.selection.iter().copied().collect(),
                });
            }
        }
    }

    Hover {
        node: hovered_node,
        stack: hovered_stack,
    }
}

fn rects_intersect(a: WorldRect, b: WorldRect) -> bool {
    a.min.cmple(b.max()).all() && a.max().cmpge(b.min).all()
}
