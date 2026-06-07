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
    GraphViewer, Link, NodeDesc, NodeId, PortAddr, PortDesc, PortId, PortSide, StackDesc, StackId,
    StackLink,
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
        render::draw_stacks(&painter, &t, &layout.stacks, hovered.stack, &palette);
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
        let mut selected: std::collections::HashSet<viewer::NodeId> = view.selection.clone();
        selected.extend(hovered.marquee.iter().copied());
        render::draw_nodes(&painter, &t, &layout.nodes, &selected, hovered.node, &palette);

        // Pin-specific hover highlight (drawn over the pins it covers).
        if let Some(center) = hovered.port {
            render::draw_port_hover(&painter, &t, center);
        }

        // Live rubber-band link.
        if let Some(addr) = view.interaction.pending_link_from {
            if let (Some(node), Some(cursor)) = (
                layout.nodes.iter().find(|n| n.id == addr.node),
                response.hover_pos(),
            ) {
                if let Some(from_world) = node.port_center(addr.port) {
                    let anchor_is_input = addr.port.side == viewer::PortSide::Input;
                    render::draw_pending_link(
                        &painter,
                        &t,
                        from_world,
                        cursor,
                        anchor_is_input,
                        &palette,
                    );
                }
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

        GraphResponse { response, actions }
    }
}
