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

mod interaction;
mod layout;
mod render;
mod response;
mod spline;
mod state;
mod transform;
mod viewer;

// Re-exported as the widget's public API; some items have no in-repo
// consumer yet, hence the allow.
#[allow(unused_imports)]
pub use response::{GraphAction, GraphResponse};
#[allow(unused_imports)]
pub use state::{GraphView, GridConfig};
#[allow(unused_imports)]
pub use transform::{Transform, WorldPos, WorldRect};
#[allow(unused_imports)]
pub use viewer::{
    GraphViewer, Link, LinkVerdict, NodeDesc, NodeId, PortAddr, PortDesc, PortId, PortSide,
    StackDesc, StackId, StackLink,
};

/// The node-graph widget. Stateless; all persistent state lives in the
/// caller-owned [`GraphView`].
pub struct NodeGraph;

impl NodeGraph {
    /// Render the graph into the available space and process input for
    /// this frame. `view` holds pan/zoom/positions/selection; `viewer`
    /// supplies the topology to draw.
    pub fn show(
        ui: &mut egui::Ui,
        view: &mut GraphView,
        viewer: &dyn GraphViewer,
    ) -> GraphResponse {
        let (rect, response) =
            ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
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
        render::draw_stacks(
            &painter,
            &t,
            &layout.stacks,
            &selected_stacks,
            hovered.stack,
            hovered.add_button,
            &palette,
        );
        render::draw_stack_links(&painter, &t, &layout.stacks, &viewer.stack_links(), &palette);

        // Selection outlines cover the live selection plus anything under an
        // in-progress marquee (previewed as pending selection).
        let mut selected_links: std::collections::HashSet<viewer::Link> =
            view.selected_links.clone();
        selected_links.extend(hovered.marquee_links.iter().copied());
        render::draw_links(
            &painter,
            &t,
            &layout.nodes,
            &viewer.links(),
            &selected_links,
            &palette,
        );
        // Live rubber-band link — drawn *before* the nodes so the pin marker
        // covers the spline's endpoint (avoids a faint anti-aliased seam on
        // the pin where the curve terminates).
        if let Some(addr) = view.interaction.pending_link_from {
            if let (Some(node), Some(cursor)) = (
                layout.nodes.iter().find(|n| n.id == addr.node),
                response.hover_pos(),
            ) {
                if let Some(from_world) = node.port_center(addr.port) {
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
            }
        }

        let mut selected: std::collections::HashSet<viewer::NodeId> = view.selection.clone();
        selected.extend(hovered.marquee.iter().copied());
        let node_paint = render::draw_nodes(
            &painter,
            &t,
            &layout.nodes,
            &selected,
            hovered.node,
            hovered.port,
            hovered.close,
            response.hover_pos(),
            &palette,
        );

        // A click on an input value chip requests an edit. It's a click (no
        // drag), so the node isn't moved; selection updates as usual. The
        // consumer resolves the value type and presents an editor.
        if let Some(port) = node_paint.hovered_chip {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            if response.clicked() {
                actions.push(GraphAction::PortValueEditRequested { port });
            }
        }

        // Live stack-member reorder overlay.
        if let Some(rd) = view.interaction.reordering {
            if let Some(cursor) = response.hover_pos() {
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
        }

        // Marquee selection rectangle.
        if let Some(start) = view.interaction.box_select_start {
            if let Some(cursor) = response.hover_pos() {
                let r = egui::Rect::from_two_pos(t.world_to_screen(start), cursor);
                painter.rect_stroke(
                    r,
                    0.0,
                    egui::Stroke::new(1.0, palette.selected),
                    egui::StrokeKind::Inside,
                );
                painter.rect_filled(r, 0.0, palette.selected.gamma_multiply(0.1));
            }
        }

        // Rejection tooltip — drawn last so it sits above every node and edge,
        // and anchored to the rejected target pin (not the cursor) so it stays
        // still and legible while the pointer keeps moving during the drag.
        if let Some((center, reason)) = hovered
            .link_target
            .as_ref()
            .and_then(|lt| lt.verdict.as_ref().err().map(|r| (lt.center, r)))
        {
            render::draw_tooltip(&painter, t.world_to_screen(center), reason.as_ref());
        }

        // Warning tooltip for a hovered node warning icon, anchored to the icon
        // and drawn above everything.
        if let Some((pin, text)) = node_paint.warning_tooltip {
            render::draw_warning(&painter, pin, text.as_ref());
        }

        GraphResponse { response, actions }
    }
}
