//! `NodeGraph`: a reusable egui node-graph canvas.
//!
//! The widget depends only on `egui` and `serde`. The consumer supplies
//! graph topology by implementing [`GraphViewer`], owns persistable view
//! state in a [`GraphView`], and applies any structural change the widget
//! reports via [`GraphResponse`].
//!
//! The canvas is an "infinite", pan/zoomable `f64` world plane with an
//! optional snapping grid. Nodes have input/output ports linked by
//! spline edges.

mod curve;
mod icons;
mod interaction;
mod layout;
mod render;
mod response;
mod spline;
mod state;
mod transform;
mod viewer;

pub use curve::{CurveEditor, CurveResponse, GradientBar};
pub use response::{ChipHit, ExternalDropTarget, GraphAction, GraphResponse};
pub use state::{CanvasItem, GraphView, GridConfig};
pub use transform::{Transform, WorldPos, WorldRect};
pub use viewer::{
    GraphViewer, Link, LinkVerdict, NodeDesc, NodeId, PortAddr, PortDesc, PortId, PortSide,
    StackDesc, StackId, StackLink,
};

/// The node-graph widget.
///
/// Stateless; all persistent state lives in the caller-owned [`GraphView`].
pub struct NodeGraph;

// Resolve domain-neutral external-drop metadata at a screen position.
fn external_drop_target_at(
    pointer: Option<egui::Pos2>,
    canvas: egui::Rect,
    transform: &Transform,
    nodes: &[layout::NodeLayout],
    chips: &[ChipHit],
) -> Option<ExternalDropTarget> {
    let pointer = pointer.filter(|pointer| canvas.contains(*pointer))?;
    if let Some(chip) = chips.iter().rev().find(|chip| chip.rect.contains(pointer)) {
        return Some(ExternalDropTarget::Input(chip.port));
    }

    let world = transform.screen_to_world(pointer);
    nodes
        .iter()
        .rev()
        .find(|node| node.rect.contains(world))
        .map(|node| ExternalDropTarget::Node(node.id))
        .or(Some(ExternalDropTarget::Canvas(world)))
}

impl NodeGraph {
    /// Fit all canvas items into the viewport.
    ///
    /// Computes the current world-space content bounds and updates pan and zoom
    /// to center them with at least `padding` screen points where zoom limits
    /// allow. Returns `false` when the graph has no canvas items.
    pub fn fit_to_content(
        view: &mut GraphView,
        viewer: &dyn GraphViewer,
        viewport_size: egui::Vec2,
        padding: f32,
    ) -> bool {
        let layout = layout::compute(viewer, view);
        let mut rects = layout.stacks.iter().map(|stack| stack.rect).chain(
            layout
                .nodes
                .iter()
                .filter(|node| node.stack.is_none())
                .map(|node| node.rect),
        );
        let Some(first) = rects.next() else {
            return false;
        };

        let (mut min, mut max) = (first.min, first.max());
        for rect in rects {
            min = min.min(rect.min);
            max = max.max(rect.max());
        }

        let padding = padding.max(0.0);
        let available = egui::vec2(
            (viewport_size.x - padding * 2.0).max(1.0),
            (viewport_size.y - padding * 2.0).max(1.0),
        );
        let content_size = max - min;
        let zoom = (available.x as f64 / content_size.x).min(available.y as f64 / content_size.y);
        view.set_zoom_clamped(zoom);

        let viewport_world = WorldPos::new(
            viewport_size.x as f64 / view.zoom,
            viewport_size.y as f64 / view.zoom,
        );
        view.pan = (min + max) * 0.5 - viewport_world * 0.5;
        true
    }

    /// Render the graph and process this frame's input.
    ///
    /// `view` holds pan/zoom/positions/selection; `viewer` supplies the
    /// topology to draw. `paint_chips` is invoked once per node, right after
    /// that node's body is painted, with the canvas rect and the node's input
    /// value chips; the consumer paints an interactive editor over each chip
    /// into the passed `ui`. Painting them inline — on the canvas layer, in
    /// z-order, interleaved with the node bodies — lets a raised node's body
    /// cover the editors of anything beneath it, and lets egui route input to
    /// the frontmost editor under the pointer (within a layer the last-painted
    /// widget wins).
    pub fn show(
        ui: &mut egui::Ui,
        view: &mut GraphView,
        viewer: &dyn GraphViewer,
        mut paint_chips: impl FnMut(&mut egui::Ui, egui::Rect, &[ChipHit]),
    ) -> GraphResponse {
        let (rect, response) =
            ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
        view.update_viewport_size(rect.size());
        let painter = ui.painter_at(rect);

        // Canvas background.
        painter.rect_filled(rect, 0.0, ui.visuals().extreme_bg_color);

        // Layout is pan/zoom-independent; compute against current positions.
        let layout = layout::compute(viewer, view);
        let t = Transform::new(rect.min, view.pan, view.zoom);

        // Process input (may change pan/zoom/positions/selection).
        let mut actions = Vec::new();
        let hovered = interaction::handle(
            ui,
            &response,
            &t,
            &layout.nodes,
            &layout.stacks,
            viewer,
            view,
            &mut actions,
        );

        // Re-derive transform + layout so this frame's drag/zoom is
        // reflected without a one-frame lag.
        let t = Transform::new(rect.min, view.pan, view.zoom);
        let layout = layout::compute(viewer, view);
        let palette = render::Palette::from_visuals(ui.visuals());

        render::draw_grid(&painter, &t, rect, view);
        let mut selected_stacks: std::collections::HashSet<viewer::StackId> =
            view.selected_stacks.clone();
        selected_stacks.extend(hovered.marquee_stacks.iter().copied());
        let mut selected: std::collections::HashSet<viewer::NodeId> = view.selection.clone();
        selected.extend(hovered.marquee.iter().copied());

        // Links ride with their frontmost endpoint (below), so raising a node
        // lifts its edges above whatever it now covers. Selection outlines cover
        // the live selection plus anything under an in-progress marquee.
        let mut selected_links: std::collections::HashSet<viewer::Link> =
            view.selected_links.clone();
        selected_links.extend(hovered.marquee_links.iter().copied());
        if let Some(addr) = view.interaction.pending_link_from
            && let (Some(node), Some(cursor)) = (
                layout.nodes.iter().find(|n| n.id == addr.node),
                response.hover_pos(),
            )
            && let Some(from_world) = node.port_center(addr.port)
        {
            let anchor_is_input = addr.port.side == viewer::PortSide::Input;
            let anchor_color = node.port_color(addr.port).unwrap_or(palette.link);

            // Endpoint and tint follow the validated target under the
            // cursor: an accepted target magnetises the spline and
            // blends toward its type color (a differing color previews
            // an implicit cast); a rejected/absent target leaves the
            // free end at the cursor in the anchor's solid color.
            let (end, target_color) = match &hovered.link_target {
                Some(lt) if lt.verdict.is_ok() => {
                    let col = layout
                        .nodes
                        .iter()
                        .find(|n| n.id == lt.addr.node)
                        .and_then(|n| n.port_color(lt.addr.port))
                        .unwrap_or(anchor_color);
                    (t.world_to_screen(lt.center), col)
                }
                _ => (cursor, anchor_color),
            };
            render::draw_pending_link(
                &painter,
                &t,
                from_world,
                end,
                anchor_is_input,
                anchor_color,
                target_color,
            );
        }

        // Paint canvas units — free nodes and whole stacks — back-to-front in
        // the persistent z-order. Each is one z-unit painted contiguously (a
        // stack's frame then its members, a free node on its own), so a raised
        // unit fully covers, and never intermixes with, anything beneath it.
        // `layout::compute` already ranked both lists, so a stack's members are
        // contiguous and each list is in z-order.
        enum Unit<'a> {
            Node(&'a layout::NodeLayout),
            Stack(&'a layout::StackLayout, Vec<&'a layout::NodeLayout>),
        }
        let mut units: Vec<((u8, usize), Unit)> = Vec::new();
        for s in &layout.stacks {
            let members = layout
                .nodes
                .iter()
                .filter(|n| n.stack == Some(s.id))
                .collect();
            units.push((view.z_key(CanvasItem::Stack(s.id)), Unit::Stack(s, members)));
        }
        for n in layout.nodes.iter().filter(|n| n.stack.is_none()) {
            units.push((view.z_key(CanvasItem::Node(n.id)), Unit::Node(n)));
        }
        units.sort_by_key(|(key, _)| *key);

        // Map every node/stack to its unit's paint rank so each link can ride
        // with its frontmost endpoint — drawn at the later-painted of its two
        // ends, over everything behind that end and under it. Node edges paint
        // just before that unit's body (tucking under the front node); pipeline
        // connectors paint just after (their pins stay above the stack frames).
        let mut node_rank: std::collections::HashMap<viewer::NodeId, usize> =
            std::collections::HashMap::new();
        let mut stack_rank: std::collections::HashMap<viewer::StackId, usize> =
            std::collections::HashMap::new();
        for (i, (_key, unit)) in units.iter().enumerate() {
            match unit {
                Unit::Stack(s, members) => {
                    stack_rank.insert(s.id, i);
                    for m in members {
                        node_rank.insert(m.id, i);
                    }
                }
                Unit::Node(n) => {
                    node_rank.insert(n.id, i);
                }
            }
        }
        let mut link_buckets: Vec<Vec<viewer::Link>> = vec![Vec::new(); units.len()];
        for link in viewer.links() {
            let (Some(&a), Some(&b)) =
                (node_rank.get(&link.from.node), node_rank.get(&link.to.node))
            else {
                continue;
            };
            link_buckets[a.max(b)].push(link);
        }
        let mut stack_link_buckets: Vec<Vec<viewer::StackLink>> = vec![Vec::new(); units.len()];
        for link in viewer.stack_links() {
            let (Some(&a), Some(&b)) = (stack_rank.get(&link.from), stack_rank.get(&link.to))
            else {
                continue;
            };
            stack_link_buckets[a.max(b)].push(link);
        }

        let mut node_paint = render::NodePaint::default();
        let mut drop_chips = Vec::new();
        for (i, (_key, unit)) in units.iter().enumerate() {
            // Node edges landing at this unit as their front end, under its body.
            if !link_buckets[i].is_empty() {
                render::draw_links(
                    &painter,
                    &t,
                    &layout.nodes,
                    &link_buckets[i],
                    &selected_links,
                    &palette,
                );
            }
            match unit {
                Unit::Stack(s, members) => {
                    render::draw_stacks(
                        &painter,
                        &t,
                        std::slice::from_ref(*s),
                        &selected_stacks,
                        hovered.stack,
                        hovered.add_button,
                        hovered.collapse_all,
                        &palette,
                    );
                    for m in members {
                        let np = render::draw_nodes(
                            &painter,
                            &t,
                            std::slice::from_ref(*m),
                            &selected,
                            hovered.node,
                            hovered.port,
                            hovered.close,
                            response.hover_pos(),
                            &palette,
                        );
                        paint_chips(ui, rect, &np.chips);
                        drop_chips.extend_from_slice(&np.chips);
                        node_paint.warning_tooltip =
                            node_paint.warning_tooltip.take().or(np.warning_tooltip);
                    }
                }
                Unit::Node(n) => {
                    let np = render::draw_nodes(
                        &painter,
                        &t,
                        std::slice::from_ref(*n),
                        &selected,
                        hovered.node,
                        hovered.port,
                        hovered.close,
                        response.hover_pos(),
                        &palette,
                    );
                    paint_chips(ui, rect, &np.chips);
                    drop_chips.extend_from_slice(&np.chips);
                    node_paint.warning_tooltip =
                        node_paint.warning_tooltip.take().or(np.warning_tooltip);
                }
            }
            // Pipeline connectors joining this unit as their front stack, above
            // its frame so the connector pins read clearly.
            if !stack_link_buckets[i].is_empty() {
                render::draw_stack_links(
                    &painter,
                    &t,
                    &layout.stacks,
                    &stack_link_buckets[i],
                    &palette,
                );
            }
        }

        // Live stack-member reorder overlay.
        if let Some(rd) = view.interaction.reordering
            && let Some(cursor) = response.hover_pos()
        {
            render::draw_reorder_overlay(
                &painter,
                &t,
                &layout.nodes,
                &layout.stacks,
                &rd,
                cursor,
                &palette,
            );
        }

        // Marquee selection rectangle.
        if let Some(start) = view.interaction.box_select_start
            && let Some(cursor) = response.hover_pos()
        {
            let r = egui::Rect::from_two_pos(t.world_to_screen(start), cursor);
            painter.rect_stroke(
                r,
                0.0,
                egui::Stroke::new(1.0, palette.selected),
                egui::StrokeKind::Inside,
            );
            painter.rect_filled(r, 0.0, palette.selected.gamma_multiply(0.1));
        }

        // Rejection tooltip — drawn last so it sits above every node and edge,
        // and anchored to the rejected target pin (not the cursor) so it stays
        // still and legible while the pointer keeps moving during the drag.
        //
        // Both this and the warning callout below paint on a Tooltip-order
        // overlay (clipped to the canvas) so they sit above the inline value
        // chips, which are placed on `Order::Foreground` by the panel.
        let overlay = ui
            .ctx()
            .layer_painter(egui::LayerId::new(
                egui::Order::Tooltip,
                ui.id().with("graph-overlay"),
            ))
            .with_clip_rect(rect);
        if let Some((center, reason)) = hovered
            .link_target
            .as_ref()
            .and_then(|lt| lt.verdict.as_ref().err().map(|r| (lt.center, r)))
        {
            render::draw_tooltip(&overlay, t.world_to_screen(center), reason.as_ref());
        }

        // Warning tooltip for a hovered node warning icon, anchored to the icon
        // and drawn above everything.
        if let Some((pin, text)) = node_paint.warning_tooltip {
            render::draw_warning(&overlay, pin, text.as_ref());
        }

        let external_drop_target = external_drop_target_at(
            ui.ctx().pointer_hover_pos(),
            rect,
            &t,
            &layout.nodes,
            &drop_chips,
        );
        GraphResponse {
            response,
            hovered_node: hovered.node_body,
            external_drop_target,
            actions,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;

    fn node(id: u32, min: WorldPos) -> layout::NodeLayout {
        layout::NodeLayout {
            id: NodeId::new(id).unwrap(),
            rect: WorldRect::new(min, 100.0, 80.0),
            title: Cow::Borrowed("node"),
            accent: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
            stack: None,
            warning: None,
            close_button: None,
            collapsed: false,
            collapse_toggle: None,
        }
    }

    fn chip(port: PortAddr, rect: egui::Rect) -> ChipHit {
        ChipHit {
            port,
            rect,
            font_size: 11.0,
            pad: 3.0,
            clip: rect,
            chevron: None,
            expanded: false,
        }
    }

    #[test]
    fn external_drop_target_priority_and_transform() {
        let canvas = egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(400.0, 300.0));
        let transform = Transform::new(canvas.min, WorldPos::new(100.0, 200.0), 2.0);
        let nodes = vec![node(1, WorldPos::new(110.0, 210.0))];
        let port = PortAddr::new(NodeId::new(1).unwrap(), PortId::input(2));
        let chip_rect = egui::Rect::from_min_size(egui::pos2(50.0, 60.0), egui::vec2(40.0, 20.0));

        assert_eq!(
            external_drop_target_at(
                Some(chip_rect.center()),
                canvas,
                &transform,
                &nodes,
                &[chip(port, chip_rect)],
            ),
            Some(ExternalDropTarget::Input(port))
        );
        assert_eq!(
            external_drop_target_at(
                Some(egui::pos2(40.0, 50.0)),
                canvas,
                &transform,
                &nodes,
                &[],
            ),
            Some(ExternalDropTarget::Node(NodeId::new(1).unwrap()))
        );
        assert_eq!(
            external_drop_target_at(
                Some(egui::pos2(310.0, 220.0)),
                canvas,
                &transform,
                &nodes,
                &[],
            ),
            Some(ExternalDropTarget::Canvas(WorldPos::new(250.0, 300.0)))
        );
    }

    #[test]
    fn external_drop_target_is_none_outside_canvas() {
        let canvas = egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(400.0, 300.0));
        let transform = Transform::new(canvas.min, WorldPos::ZERO, 1.0);

        assert_eq!(
            external_drop_target_at(Some(egui::pos2(9.0, 20.0)), canvas, &transform, &[], &[],),
            None
        );
        assert_eq!(
            external_drop_target_at(None, canvas, &transform, &[], &[]),
            None
        );
    }
}
