//! Node-graph editor panel.
//!
//! Renders the [`NodeGraph`] widget against a small hand-built demo graph
//! so the widget mechanics (pan/zoom/grid/spline edges/drag/selection) can
//! be inspected. A small toolbar toggles the grid and snapping.

use bevy_egui::egui;

use bevy::prelude::{debug, Entity};

use crate::ui::widgets::node_graph::{
    GraphAction, GraphView, GraphViewer, Link, NodeDesc, NodeGraph, NodeId, PortAddr, PortDesc,
    PortId, StackDesc, StackId, WorldPos,
};

/// Demo topology: a few free Expr nodes, one "Update" stack of modifier
/// nodes, plus a mutable link set so link-create and delete actions are
/// observable.
struct DemoViewer {
    links: Vec<Link>,
}

fn nid(i: u32) -> NodeId {
    NodeId::new(i).unwrap()
}

fn update_stack() -> StackId {
    StackId::new(1).unwrap()
}

fn default_links() -> Vec<Link> {
    vec![
        Link {
            from: PortAddr::new(nid(1), PortId::output(0)),
            to: PortAddr::new(nid(3), PortId::input(0)),
        },
        Link {
            from: PortAddr::new(nid(2), PortId::output(0)),
            to: PortAddr::new(nid(3), PortId::input(1)),
        },
        Link {
            from: PortAddr::new(nid(3), PortId::output(0)),
            to: PortAddr::new(nid(4), PortId::input(0)),
        },
        // Expr output feeding a value inside a specific stacked modifier.
        Link {
            from: PortAddr::new(nid(3), PortId::output(0)),
            to: PortAddr::new(nid(6), PortId::input(0)),
        },
        Link {
            from: PortAddr::new(nid(1), PortId::output(0)),
            to: PortAddr::new(nid(5), PortId::input(0)),
        },
    ]
}

impl GraphViewer for DemoViewer {
    fn node_ids(&self) -> Vec<NodeId> {
        (1..=7).map(nid).collect()
    }

    fn node(&self, id: NodeId) -> NodeDesc {
        let expr = egui::Color32::from_rgb(70, 110, 160);
        let modifier = egui::Color32::from_rgb(120, 80, 130);
        match id.get() {
            1 => NodeDesc::new("Time")
                .with_outputs(vec![PortDesc::new("t")])
                .with_accent(expr),
            2 => NodeDesc::new("Spawn Rate")
                .with_outputs(vec![PortDesc::new("value")])
                .with_accent(egui::Color32::from_rgb(90, 130, 80)),
            3 => NodeDesc::new("Multiply")
                .with_inputs(vec![PortDesc::new("a"), PortDesc::new("b")])
                .with_outputs(vec![PortDesc::new("out")])
                .with_accent(egui::Color32::from_rgb(150, 110, 60)),
            4 => NodeDesc::new("Set Lifetime")
                .with_inputs(vec![PortDesc::new("value")])
                .with_accent(egui::Color32::from_rgb(140, 80, 120)),
            5 => NodeDesc::new("Linear Drag")
                .with_inputs(vec![PortDesc::new("coeff")])
                .with_accent(modifier),
            6 => NodeDesc::new("Accel Force")
                .with_inputs(vec![PortDesc::new("accel")])
                .with_accent(modifier),
            7 => NodeDesc::new("Set Color")
                .with_inputs(vec![PortDesc::new("color")])
                .with_accent(modifier),
            _ => NodeDesc::new("?"),
        }
    }

    fn links(&self) -> Vec<Link> {
        self.links.clone()
    }

    fn stacks(&self) -> Vec<StackDesc> {
        vec![
            StackDesc::new(update_stack(), "Update")
                .with_members(vec![nid(5), nid(6), nid(7)])
                .with_accent(egui::Color32::from_rgb(80, 60, 90)),
        ]
    }
}

/// Seed initial positions for the demo nodes and stack the first time the
/// panel is shown (so they don't all stack at the origin).
fn seed_positions(view: &mut GraphView) {
    let defaults = [
        (1u32, WorldPos::new(40.0, 60.0)),
        (2, WorldPos::new(40.0, 200.0)),
        (3, WorldPos::new(300.0, 120.0)),
        (4, WorldPos::new(560.0, 140.0)),
    ];
    for (i, pos) in defaults {
        view.ensure_position(nid(i), pos);
    }
    view.ensure_stack_position(update_stack(), WorldPos::new(820.0, 60.0));
}

pub fn show(ui: &mut egui::Ui, doc_entity: Entity, view: &mut GraphView) {
    seed_positions(view);

    // The demo's link set is mutable view-local state, persisted in egui temp
    // memory keyed per document so Delete / link-drag are observable.
    let links_id = egui::Id::new(("graph-demo-links", doc_entity));
    let mut links: Vec<Link> = ui
        .data_mut(|d| d.get_temp::<Vec<Link>>(links_id))
        .unwrap_or_else(default_links);

    egui::TopBottomPanel::top("graph-toolbar")
        .frame(egui::Frame::new().inner_margin(egui::Margin::symmetric(6, 4)))
        .show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.checkbox(&mut view.grid.enabled, "Grid");
                ui.checkbox(&mut view.grid.snap, "Snap");
                ui.separator();
                if ui.button("Reset view").clicked() {
                    view.pan = WorldPos::ZERO;
                    view.zoom = 1.0;
                }
                if ui.button("Reset links").clicked() {
                    links = default_links();
                }
                ui.separator();
                ui.weak(format!("zoom {:.0}%", view.zoom * 100.0));
            });
        });

    let viewer = DemoViewer {
        links: links.clone(),
    };
    let resp = NodeGraph::show(ui, view, &viewer);

    for action in &resp.actions {
        match action {
            GraphAction::NodeMoved { node, to } => {
                debug!("node {} moved to ({:.1}, {:.1})", node.get(), to.x, to.y);
            }
            GraphAction::StackMoved { stack, to } => {
                debug!("stack {} moved to ({:.1}, {:.1})", stack.get(), to.x, to.y);
            }
            GraphAction::LinkRequested { from, to } => {
                let link = Link {
                    from: *from,
                    to: *to,
                };
                if !links.contains(&link) {
                    links.push(link);
                }
                debug!(
                    "link added {}:{:?} -> {}:{:?}",
                    from.node.get(),
                    from.port,
                    to.node.get(),
                    to.port
                );
            }
            GraphAction::LinkDeleteRequested { link } => {
                links.retain(|l| l != link);
                debug!("link deleted {:?}", link);
            }
            GraphAction::NodesDeleteRequested { nodes } => {
                debug!("delete requested for {} node(s)", nodes.len());
            }
            GraphAction::ContextMenu { at } => {
                debug!("context menu at ({:.1}, {:.1})", at.x, at.y);
            }
            GraphAction::SelectionChanged => {}
        }
    }

    ui.data_mut(|d| d.insert_temp(links_id, links));
}
