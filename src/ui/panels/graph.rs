//! Node-graph editor panel.
//!
//! Renders the [`NodeGraph`] widget against the document's real
//! [`EffectAsset`] via [`GraphSnapshot`]: the module's `Expr` DAG plus the
//! three modifier stacks (init/update/render). Read-only for now —
//! structural `GraphAction`s are logged, not applied (editing is a later,
//! upstream-gated phase). A small toolbar toggles the grid and snapping.

use bevy_egui::egui;

use bevy::prelude::{debug, Assets, Handle};
use bevy_hanabi::EffectAsset;

use crate::graph_adapter::GraphSnapshot;
use crate::ui::widgets::node_graph::{GraphAction, GraphView, NodeGraph, WorldPos};

pub fn show(
    ui: &mut egui::Ui,
    effects: &Assets<EffectAsset>,
    effect_handle: &Handle<EffectAsset>,
    view: &mut GraphView,
) {
    // The canonical asset may still be loading; retry next frame.
    let Some(asset) = effects.get(effect_handle) else {
        ui.centered_and_justified(|ui| {
            ui.weak("Loading effect…");
        });
        return;
    };

    let snapshot = GraphSnapshot::build(asset);
    snapshot.seed_positions(view);

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
                ui.separator();
                ui.weak(format!("zoom {:.0}%", view.zoom * 100.0));
            });
        });

    let resp = NodeGraph::show(ui, view, &snapshot);

    // Structural edits are not yet wired to the edit channel; log intent so
    // the read-only adapter can be exercised without mutating the asset.
    for action in &resp.actions {
        match action {
            GraphAction::NodeMoved { node, to } => {
                debug!("node {} moved to ({:.1}, {:.1})", node.get(), to.x, to.y);
            }
            GraphAction::StackMoved { stack, to } => {
                debug!("stack {} moved to ({:.1}, {:.1})", stack.get(), to.x, to.y);
            }
            GraphAction::StackMemberMoved {
                stack,
                from_index,
                to_index,
            } => {
                debug!(
                    "stack {} member {} -> {} (not applied)",
                    stack.get(),
                    from_index,
                    to_index
                );
            }
            GraphAction::LinkRequested { from, to } => {
                debug!(
                    "link requested {}:{:?} -> {}:{:?} (not applied)",
                    from.node.get(),
                    from.port,
                    to.node.get(),
                    to.port
                );
            }
            GraphAction::LinkDeleteRequested { link } => {
                debug!("link delete requested {:?} (not applied)", link);
            }
            GraphAction::NodesDeleteRequested { nodes } => {
                debug!("delete requested for {} node(s) (not applied)", nodes.len());
            }
            GraphAction::ContextMenu { at } => {
                debug!("context menu at ({:.1}, {:.1})", at.x, at.y);
            }
            GraphAction::SelectionChanged => {}
        }
    }
}
