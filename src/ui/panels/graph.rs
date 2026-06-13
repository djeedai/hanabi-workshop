//! Node-graph editor panel.
//!
//! Renders the [`NodeGraph`] widget directly against the document's canonical
//! [`EffectGraph`] via [`GraphReader`]: its expression nodes, ordered modifier
//! stacks (init/update/render), links, and inline-default value chips. Modifier
//! reordering and link create/delete are wired to the edit channel; the
//! remaining structural `GraphAction`s (node move/delete) are still logged, not
//! applied (graph-level editing is a later phase). A small toolbar toggles the
//! grid and snapping.

use bevy_egui::egui;

use bevy::ecs::message::MessageWriter;
use bevy::ecs::reflect::AppTypeRegistry;
use bevy::prelude::{Entity, debug};

use crate::edits::{EditKind, EditRequest};
use crate::effect_graph::model::EffectGraph;
use crate::effect_graph::view::{GraphReader, group_of_widget_stack};
use crate::ui::widgets::node_graph::{GraphAction, GraphView, NodeGraph, WorldPos};

pub fn show(
    ui: &mut egui::Ui,
    doc_entity: Entity,
    graph: &EffectGraph,
    type_registry: &AppTypeRegistry,
    edits: &mut MessageWriter<EditRequest>,
    view: &mut GraphView,
) {
    let registry = type_registry.read();
    let reader = GraphReader::new(graph, &registry);
    reader.seed_positions(view);

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

    let resp = NodeGraph::show(ui, view, &reader);

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
                // Reorder a modifier within its list via the edit channel —
                // the same MoveModifier edit the Effect panel emits. `to_index`
                // is already the post-removal target, matching MoveModifier.
                if let Some(group) = group_of_widget_stack(graph, *stack) {
                    edits.write(EditRequest::new(
                        doc_entity,
                        EditKind::MoveModifier {
                            group,
                            from: *from_index,
                            to: *to_index,
                        },
                    ));
                }
            }
            GraphAction::LinkRequested { from, to } => {
                // The widget only emits accepted (validated) targets, so we map
                // the port addresses straight back to a model link and add it.
                if let Some(link) = reader.resolve_link(*from, *to) {
                    edits.write(EditRequest::new(doc_entity, EditKind::AddLink { link }));
                } else {
                    debug!(
                        "link requested {}:{:?} -> {}:{:?} could not be resolved",
                        from.node.get(),
                        from.port,
                        to.node.get(),
                        to.port
                    );
                }
            }
            GraphAction::LinkDeleteRequested { link } => {
                if let Some(resolved) = reader.resolve_link(link.from, link.to) {
                    edits.write(EditRequest::new(
                        doc_entity,
                        EditKind::RemoveLink { link: resolved },
                    ));
                } else {
                    debug!("link delete requested {:?} could not be resolved", link);
                }
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
