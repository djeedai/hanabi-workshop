//! Node-graph editor panel.
//!
//! Renders the [`NodeGraph`] widget directly against the document's canonical
//! [`EffectGraph`] via [`GraphReader`]: its expression nodes, ordered modifier
//! stacks (init/update/render), links, and inline-default value chips. Modifier
//! reordering, link create/delete, node create (right-click menu, or dragging a
//! pin into empty space to spawn a connected node) and node / stack deletion
//! (Delete key) are all wired to the edit channel. A small toolbar toggles the
//! grid and snapping.

use bevy_egui::egui;

use bevy::ecs::message::MessageWriter;
use bevy::ecs::reflect::AppTypeRegistry;
use bevy::prelude::{Entity, debug};
use bevy::reflect::TypeRegistry;
use bevy_hanabi::graph::expr::{BinaryOperator, TernaryOperator, UnaryOperator};
use bevy_hanabi::{Attribute, BuiltInOperator, Value};

use crate::document::ModifierGroup;
use crate::edits::{EditKind, EditRequest};
use crate::effect_graph::model::{
    EffectGraph, ExprNode, GraphLink, InputSlot, NodeId, PortRef, SharedStr,
};
use crate::effect_graph::schema::{OUTPUT_PORT, expr_input_ports};
use crate::effect_graph::view::{GraphReader, group_of_widget_stack};
use crate::modifier_registry;
use crate::ui::widgets::node_graph::{
    GraphAction, GraphView, NodeGraph, NodeId as WNodeId, PortAddr, PortId, WorldPos,
};

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
                // Only free nodes (expression nodes) are deletable here; stack
                // members are removed via stack emptying or the Effect panel.
                for wid in nodes {
                    let Some(id) = NodeId::new(wid.get()) else {
                        continue;
                    };
                    if is_stack_member(graph, id) {
                        continue;
                    }
                    edits.write(EditRequest::new(doc_entity, EditKind::RemoveNode { id }));
                }
            }
            GraphAction::StacksDeleteRequested { stacks } => {
                // Deleting a selected stack empties it: the init/update/render
                // stages are fixed, so we drop every member rather than the
                // container. Remove back-to-front so earlier indices stay valid
                // as the sequential edits apply.
                for wstack in stacks {
                    let Some(group) = group_of_widget_stack(graph, *wstack) else {
                        continue;
                    };
                    let Some(count) = graph.stack(group).map(|s| s.members.len()) else {
                        continue;
                    };
                    for idx in (0..count).rev() {
                        edits.write(EditRequest::new(
                            doc_entity,
                            EditKind::RemoveModifier { group, idx },
                        ));
                    }
                }
            }
            GraphAction::ContextMenu { at } => {
                if let Some(screen) = ui
                    .ctx()
                    .pointer_interact_pos()
                    .or_else(|| ui.ctx().pointer_latest_pos())
                {
                    let opened_at = ui.ctx().cumulative_pass_nr();
                    ui.ctx().data_mut(|d| {
                        d.insert_temp(
                            menu_id(doc_entity),
                            PendingMenu {
                                screen,
                                at: *at,
                                link: None,
                                opened_at,
                            },
                        )
                    });
                }
            }
            GraphAction::LinkDropped {
                source,
                source_is_output,
                at,
            } => {
                // Releasing a link in empty space opens the create menu filtered
                // to nodes the dangling pin can connect to; the chosen node is
                // wired to that pin automatically.
                if let Some(screen) = ui
                    .ctx()
                    .pointer_interact_pos()
                    .or_else(|| ui.ctx().pointer_latest_pos())
                {
                    let opened_at = ui.ctx().cumulative_pass_nr();
                    ui.ctx().data_mut(|d| {
                        d.insert_temp(
                            menu_id(doc_entity),
                            PendingMenu {
                                screen,
                                at: *at,
                                link: Some(LinkSource {
                                    source: *source,
                                    source_is_output: *source_is_output,
                                }),
                                opened_at,
                            },
                        )
                    });
                }
            }
            GraphAction::SelectionChanged => {}
        }
    }

    context_menu(ui, doc_entity, &reader, graph, &registry, edits, view);
}

/// The dangling pin that opened a create menu via a dropped link.
#[derive(Clone, Copy)]
struct LinkSource {
    /// The pin the link drag started from (widget address).
    source: PortAddr,
    /// Whether `source` is an output pin (needs a consumer) rather than an
    /// input pin (needs a producer).
    source_is_output: bool,
}

/// A pending create-node menu: where to draw it, the world position to place a
/// new node at, the dangling pin to auto-wire (if opened by a dropped link), and
/// the egui pass it opened on (so the very click that opened it can't be
/// mistaken for a click-outside that dismisses it the same frame).
#[derive(Clone, Copy)]
struct PendingMenu {
    screen: egui::Pos2,
    at: WorldPos,
    link: Option<LinkSource>,
    opened_at: u64,
}

/// Which node categories a create menu offers, depending on how it was opened.
#[derive(Clone, Copy, PartialEq)]
enum MenuFilter {
    /// Plain right-click: every producer node plus modifiers.
    Full,
    /// Dropped from an output pin: only nodes that accept an input.
    Consumer,
    /// Dropped from an input pin: only nodes that yield a value.
    Producer,
}

/// egui memory key for the pending create-node context menu of one document.
fn menu_id(doc: Entity) -> egui::Id {
    egui::Id::new(("graph-context-menu", doc))
}

/// Whether `id` is a member of any modifier stack (vs. a free expression node).
fn is_stack_member(graph: &EffectGraph, id: NodeId) -> bool {
    graph.stacks.iter().any(|s| s.members.contains(&id))
}

/// Render the create-node context menu if one is pending, applying the chosen
/// creation edit. When the menu was opened by a dropped link, the chosen node
/// is also auto-wired to the dangling pin. Dismissed on a selection, an outside
/// click, or `Escape`.
fn context_menu(
    ui: &mut egui::Ui,
    doc: Entity,
    reader: &GraphReader,
    graph: &EffectGraph,
    registry: &TypeRegistry,
    edits: &mut MessageWriter<EditRequest>,
    view: &mut GraphView,
) {
    let id = menu_id(doc);
    let Some(menu) = ui.ctx().data(|d| d.get_temp::<PendingMenu>(id)) else {
        return;
    };

    let filter = match menu.link {
        None => MenuFilter::Full,
        Some(LinkSource {
            source_is_output: true,
            ..
        }) => MenuFilter::Consumer,
        Some(LinkSource {
            source_is_output: false,
            ..
        }) => MenuFilter::Producer,
    };

    let mut close = false;
    let mut chosen: Option<EditKind> = None;
    let area = egui::Area::new(id.with("area"))
        .order(egui::Order::Foreground)
        .fixed_pos(menu.screen)
        .show(ui.ctx(), |ui| {
            egui::Frame::menu(ui.style()).show(ui, |ui| {
                chosen = create_menu(ui, graph, registry, filter);
            });
        });

    if let Some(kind) = chosen {
        // A standalone expression node is placed at the cursor; its id is the
        // next one the allocator will mint, so we can pre-seed the layout
        // position and build any auto-link before the edit applies. Modifier
        // nodes are positioned by their stack and need no seed.
        if let EditKind::AddExprNode { inputs, .. } = &kind {
            if let Some(wid) = WNodeId::new(graph.next_id) {
                view.ensure_position(wid, menu.at);
            }
            if let (Some(new_id), Some(LinkSource { source, source_is_output })) =
                (NodeId::new(graph.next_id), menu.link)
            {
                if let Some(link) =
                    auto_link(reader, new_id, inputs, source, source_is_output)
                {
                    // The node edit must land before its link references it.
                    edits.write(EditRequest::new(doc, kind.clone()));
                    edits.write(EditRequest::new(doc, EditKind::AddLink { link }));
                    close = true;
                }
            }
        }
        if !close {
            edits.write(EditRequest::new(doc, kind));
            close = true;
        }
    }

    // A click outside the menu dismisses it — but not on the pass it opened on,
    // where the very secondary click that spawned the menu would otherwise be
    // read as a click-outside and close it the same frame.
    let opened_this_pass = ui.ctx().cumulative_pass_nr() == menu.opened_at;
    if (!opened_this_pass && area.response.clicked_elsewhere())
        || ui.ctx().input(|i| i.key_pressed(egui::Key::Escape))
    {
        close = true;
    }
    if close {
        ui.ctx().data_mut(|d| d.remove::<PendingMenu>(id));
    }
}

/// Build the link wiring a freshly-created expression node (`new_id`, with the
/// given operand `inputs`) to the dangling pin that opened the menu. An output
/// `source` feeds the new node's first input; an input `source` is fed by the
/// new node's output.
fn auto_link(
    reader: &GraphReader,
    new_id: NodeId,
    inputs: &[InputSlot],
    source: PortAddr,
    source_is_output: bool,
) -> Option<GraphLink> {
    let source_node = NodeId::new(source.node.get())?;
    if source_is_output {
        // Output → the new consumer's first input port.
        let port = inputs.first()?.name.clone();
        Some(GraphLink {
            from: PortRef {
                node: source_node,
                port: OUTPUT_PORT.into(),
            },
            to: PortRef { node: new_id, port },
        })
    } else {
        // The new producer's output → the source input port. Resolve the input
        // port name through the reader, which exists for the source node.
        let new_out = PortAddr::new(WNodeId::new(new_id.get())?, PortId::output(0));
        reader.resolve_link(new_out, source)
    }
}
/// The create-node menu contents, restricted to the categories allowed by
/// `filter`. Returns the chosen creation [`EditKind`], or `None` if nothing was
/// picked this frame.
fn create_menu(
    ui: &mut egui::Ui,
    graph: &EffectGraph,
    registry: &TypeRegistry,
    filter: MenuFilter,
) -> Option<EditKind> {
    let mut chosen: Option<EditKind> = None;
    ui.set_min_width(150.0);

    ui.menu_button("Binary op", |ui| {
        let ops = [
            ("Add (+)", BinaryOperator::Add),
            ("Subtract (-)", BinaryOperator::Sub),
            ("Multiply (*)", BinaryOperator::Mul),
            ("Divide (/)", BinaryOperator::Div),
            ("Min", BinaryOperator::Min),
            ("Max", BinaryOperator::Max),
            ("Remainder (%)", BinaryOperator::Remainder),
            ("Step", BinaryOperator::Step),
        ];
        for (label, op) in ops {
            if ui.button(label).clicked() {
                chosen = Some(add_expr(ExprNode::Binary(op)));
                ui.close();
            }
        }
    });

    ui.menu_button("Unary op", |ui| {
        let ops = [
            ("Absolute", UnaryOperator::Abs),
            ("Floor", UnaryOperator::Floor),
            ("Ceil", UnaryOperator::Ceil),
            ("Fract", UnaryOperator::Fract),
            ("Round", UnaryOperator::Round),
            ("Sign", UnaryOperator::Sign),
            ("Sqrt", UnaryOperator::Sqrt),
            ("Normalize", UnaryOperator::Normalize),
            ("Saturate", UnaryOperator::Saturate),
            ("Sin", UnaryOperator::Sin),
            ("Cos", UnaryOperator::Cos),
        ];
        for (label, op) in ops {
            if ui.button(label).clicked() {
                chosen = Some(add_expr(ExprNode::Unary(op)));
                ui.close();
            }
        }
    });

    ui.menu_button("Ternary op", |ui| {
        let ops = [
            ("Mix", TernaryOperator::Mix),
            ("Clamp", TernaryOperator::Clamp),
            ("SmoothStep", TernaryOperator::SmoothStep),
        ];
        for (label, op) in ops {
            if ui.button(label).clicked() {
                chosen = Some(add_expr(ExprNode::Ternary(op)));
                ui.close();
            }
        }
    });

    // Source nodes yield a value but take no input, so they're hidden when the
    // menu must offer a *consumer* for a dropped output.
    if filter != MenuFilter::Consumer {
        ui.menu_button("Built-in", |ui| {
            let ops = [
                ("Time", BuiltInOperator::Time),
                ("Delta time", BuiltInOperator::DeltaTime),
                ("Virtual time", BuiltInOperator::VirtualTime),
                ("Real time", BuiltInOperator::RealTime),
                ("Alpha cutoff", BuiltInOperator::AlphaCutoff),
            ];
            for (label, op) in ops {
                if ui.button(label).clicked() {
                    chosen = Some(add_expr(ExprNode::BuiltIn(op)));
                    ui.close();
                }
            }
        });

        ui.menu_button("Attribute", |ui| {
            for &attr in Attribute::all() {
                if attr == Attribute::ID || attr == Attribute::PARTICLE_COUNTER {
                    continue;
                }
                if ui.button(attr.name()).clicked() {
                    chosen = Some(add_expr(ExprNode::Attribute(attr)));
                    ui.close();
                }
            }
        });

        if !graph.properties.is_empty() {
            ui.menu_button("Property", |ui| {
                for prop in &graph.properties {
                    if ui.button(&*prop.name).clicked() {
                        chosen = Some(add_expr(ExprNode::Property(prop.id)));
                        ui.close();
                    }
                }
            });
        }
    }

    // Modifiers consume expressions but don't produce a value, so they're only
    // offered for a plain create (not as an auto-wire target).
    if filter == MenuFilter::Full {
        ui.separator();

        ui.menu_button("Modifier", |ui| {
            for group in [
                ModifierGroup::Init,
                ModifierGroup::Update,
                ModifierGroup::Render,
            ] {
                ui.menu_button(group.label(), |ui| {
                    for kind in modifier_registry::iter_modifier_kinds_for(registry, group) {
                        if ui.button(kind.display_name()).clicked() {
                            let at = graph.stack(group).map(|s| s.members.len()).unwrap_or(0);
                            chosen = Some(EditKind::AddModifierFromTemplate {
                                group,
                                type_id: kind.type_id,
                                at,
                            });
                            ui.close();
                        }
                    }
                });
            }
        });
    }

    chosen
}

/// Build an [`EditKind::AddExprNode`] for `expr`, seeding each operand input
/// port with a neutral scalar default so the node bakes once connected.
fn add_expr(expr: ExprNode) -> EditKind {
    let inputs = expr_input_ports(&expr)
        .iter()
        .map(|name| InputSlot {
            name: SharedStr::from(*name),
            default: Value::from(0.0f32),
        })
        .collect();
    EditKind::AddExprNode { expr, inputs }
}
