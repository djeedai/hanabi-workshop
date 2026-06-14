//! Node-graph editor panel.
//!
//! Renders the [`NodeGraph`] widget directly against the document's canonical
//! [`EffectGraph`] via [`GraphReader`]: its expression nodes, ordered modifier
//! stacks (init/update/render), links, and inline-default value chips. Modifier
//! reordering, link create/delete, node create (a searchable, categorized
//! picker opened by right-click or by dragging a pin into empty space — the
//! latter type-filters candidates and auto-wires the chosen node), modifier
//! create (the "Add" button at the bottom of each stack opens a group-specific
//! modifier menu) and node / stack deletion (Delete key) are all wired to the
//! edit channel. A small toolbar toggles the grid and snapping.

use bevy_egui::egui;

use bevy::ecs::message::MessageWriter;
use bevy::ecs::reflect::AppTypeRegistry;
use bevy::prelude::{Entity, debug};
use bevy::reflect::TypeRegistry;
use bevy_hanabi::graph::expr::{BinaryOperator, TernaryOperator, UnaryOperator};
use bevy_hanabi::{Attribute, BuiltInOperator, ScalarType, Value, ValueType, VectorType};

use crate::document::ModifierGroup;
use crate::edits::{EditKind, EditRequest};
use crate::effect_graph::model::{
    EffectGraph, ExprNode, GraphLink, InputSlot, NodeId, PortRef, SharedStr,
};
use crate::effect_graph::schema::{OUTPUT_PORT, expr_input_ports};
use crate::effect_graph::view::{GraphReader, can_cast, group_of_widget_stack};
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
            GraphAction::StackAddRequested { stack } => {
                // The "Add" button on a stack opens a group-specific modifier
                // menu (init/update/render modifiers for that stage only).
                if let Some(group) = group_of_widget_stack(graph, *stack) {
                    if let Some(screen) = ui
                        .ctx()
                        .pointer_interact_pos()
                        .or_else(|| ui.ctx().pointer_latest_pos())
                    {
                        let opened_at = ui.ctx().cumulative_pass_nr();
                        ui.ctx().data_mut(|d| {
                            d.insert_temp(
                                stack_menu_id(doc_entity),
                                PendingStackMenu {
                                    screen,
                                    group,
                                    opened_at,
                                },
                            )
                        });
                    }
                }
            }
            GraphAction::SelectionChanged => {}
        }
    }

    context_menu(ui, doc_entity, &reader, graph, edits, view);
    stack_menu(ui, doc_entity, graph, &registry, edits);
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

/// A pending per-stack modifier menu: where to draw it, which modifier group it
/// targets, and the pass it opened on (for the same self-close guard as the
/// create-node menu).
#[derive(Clone, Copy)]
struct PendingStackMenu {
    screen: egui::Pos2,
    group: ModifierGroup,
    opened_at: u64,
}

/// egui memory key for the pending per-stack modifier menu of one document.
fn stack_menu_id(doc: Entity) -> egui::Id {
    egui::Id::new(("graph-stack-menu", doc))
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

    // The value type carried by the dangling pin (if opened from a link), used
    // to type-filter producer candidates.
    let pin_type = menu
        .link
        .and_then(|ls| reader.port_type(ls.source, ls.source_is_output));

    let state_id = id.with("picker-state");
    let mut state = ui
        .ctx()
        .data(|d| d.get_temp::<PickerState>(state_id))
        .unwrap_or_default();
    let catalog = picker_catalog(graph);

    let mut close = false;
    let mut chosen: Option<EditKind> = None;
    let area = egui::Area::new(id.with("area"))
        .order(egui::Order::Foreground)
        .fixed_pos(menu.screen)
        .show(ui.ctx(), |ui| {
            egui::Frame::menu(ui.style()).show(ui, |ui| {
                chosen = picker_body(
                    ui,
                    &catalog,
                    menu.link,
                    pin_type,
                    filter,
                    &mut state,
                    menu.opened_at,
                );
            });
        });
    ui.ctx()
        .data_mut(|d| d.insert_temp(state_id, state.clone()));

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
        ui.ctx().data_mut(|d| {
            d.remove::<PendingMenu>(id);
            d.remove::<PickerState>(state_id);
        });
    }
}

/// Render the per-stack modifier menu if one is pending, emitting an
/// [`EditKind::AddModifierFromTemplate`] for the chosen modifier (appended to
/// the end of that group's stack). Dismissed on a selection, an outside click,
/// or `Escape`, with the same opening-pass self-close guard as [`context_menu`].
fn stack_menu(
    ui: &mut egui::Ui,
    doc: Entity,
    graph: &EffectGraph,
    registry: &TypeRegistry,
    edits: &mut MessageWriter<EditRequest>,
) {
    let id = stack_menu_id(doc);
    let Some(menu) = ui.ctx().data(|d| d.get_temp::<PendingStackMenu>(id)) else {
        return;
    };

    let mut close = false;
    let mut chosen: Option<EditKind> = None;
    let area = egui::Area::new(id.with("area"))
        .order(egui::Order::Foreground)
        .fixed_pos(menu.screen)
        .show(ui.ctx(), |ui| {
            egui::Frame::menu(ui.style()).show(ui, |ui| {
                ui.set_min_width(170.0);
                ui.label(format!("Add {} modifier", menu.group.label()));
                ui.separator();
                egui::ScrollArea::vertical()
                    .max_height(360.0)
                    .show(ui, |ui| {
                        for kind in modifier_registry::iter_modifier_kinds_for(registry, menu.group)
                        {
                            if ui.button(kind.display_name()).clicked() {
                                let at =
                                    graph.stack(menu.group).map(|s| s.members.len()).unwrap_or(0);
                                chosen = Some(EditKind::AddModifierFromTemplate {
                                    group: menu.group,
                                    type_id: kind.type_id,
                                    at,
                                });
                            }
                        }
                    });
            });
        });

    if let Some(kind) = chosen {
        edits.write(EditRequest::new(doc, kind));
        close = true;
    }

    let opened_this_pass = ui.ctx().cumulative_pass_nr() == menu.opened_at;
    if (!opened_this_pass && area.response.clicked_elsewhere())
        || ui.ctx().input(|i| i.key_pressed(egui::Key::Escape))
    {
        close = true;
    }
    if close {
        ui.ctx().data_mut(|d| d.remove::<PendingStackMenu>(id));
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
/// User-facing grouping of create-node entries. Categories read by intent
/// (Math, Trigonometry, …) rather than by operator arity.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PickerCategory {
    Math,
    Trig,
    Vector,
    Interp,
    Comparison,
    Logic,
    Random,
    Bitwise,
    BuiltIn,
    Attribute,
    Property,
}

impl PickerCategory {
    /// Categories in display order.
    const ALL: [PickerCategory; 11] = [
        PickerCategory::Math,
        PickerCategory::Trig,
        PickerCategory::Vector,
        PickerCategory::Interp,
        PickerCategory::Comparison,
        PickerCategory::Logic,
        PickerCategory::Random,
        PickerCategory::Bitwise,
        PickerCategory::BuiltIn,
        PickerCategory::Attribute,
        PickerCategory::Property,
    ];

    fn label(self) -> &'static str {
        match self {
            PickerCategory::Math => "Math",
            PickerCategory::Trig => "Trigonometry",
            PickerCategory::Vector => "Vector",
            PickerCategory::Interp => "Interpolation",
            PickerCategory::Comparison => "Comparison",
            PickerCategory::Logic => "Logic",
            PickerCategory::Random => "Random",
            PickerCategory::Bitwise => "Bit manipulation",
            PickerCategory::BuiltIn => "Built-in",
            PickerCategory::Attribute => "Attribute",
            PickerCategory::Property => "Property",
        }
    }
}

/// One selectable create-node entry in the rich picker.
struct PickerNode {
    category: PickerCategory,
    label: std::borrow::Cow<'static, str>,
    /// Lowercased haystack (label + synonyms) for token search.
    search: String,
    kind: EditKind,
    /// Whether the node accepts at least one input (an operator vs. a source).
    accepts_input: bool,
    /// Natural output value type, when statically known (`None` = operand
    /// dependent / unknown, so never type-filtered out).
    output_type: Option<ValueType>,
}

/// Build a [`PickerNode`] for an expression with a `'static` label and a
/// space-separated list of search synonyms.
fn picker_entry(
    category: PickerCategory,
    label: &'static str,
    synonyms: &'static str,
    expr: ExprNode,
    output_type: Option<ValueType>,
) -> PickerNode {
    let accepts_input = !expr_input_ports(&expr).is_empty();
    PickerNode {
        category,
        search: format!("{label} {synonyms}").to_lowercase(),
        label: std::borrow::Cow::Borrowed(label),
        kind: add_expr(expr),
        accepts_input,
        output_type,
    }
}

/// The full catalog of create-node entries, grouped by user-facing category.
/// Attributes and properties are sourced from the current graph.
fn picker_catalog(graph: &EffectGraph) -> Vec<PickerNode> {
    use PickerCategory as C;
    let f32t = ValueType::Scalar(ScalarType::Float);
    let u32t = ValueType::Scalar(ScalarType::Uint);
    let boolt = ValueType::Scalar(ScalarType::Bool);
    let vec2t = ValueType::Vector(VectorType::VEC2F);
    let vec3t = ValueType::Vector(VectorType::VEC3F);
    let vec4t = ValueType::Vector(VectorType::VEC4F);

    let mut v = vec![
        // Math.
        picker_entry(C::Math, "Add", "+ plus sum", ExprNode::Binary(BinaryOperator::Add), Some(f32t)),
        picker_entry(C::Math, "Subtract", "- minus difference", ExprNode::Binary(BinaryOperator::Sub), Some(f32t)),
        picker_entry(C::Math, "Multiply", "* times product", ExprNode::Binary(BinaryOperator::Mul), Some(f32t)),
        picker_entry(C::Math, "Divide", "/ quotient ratio", ExprNode::Binary(BinaryOperator::Div), Some(f32t)),
        picker_entry(C::Math, "Remainder", "% mod modulo", ExprNode::Binary(BinaryOperator::Remainder), Some(f32t)),
        picker_entry(C::Math, "Minimum", "min", ExprNode::Binary(BinaryOperator::Min), Some(f32t)),
        picker_entry(C::Math, "Maximum", "max", ExprNode::Binary(BinaryOperator::Max), Some(f32t)),
        picker_entry(C::Math, "Step", "threshold", ExprNode::Binary(BinaryOperator::Step), Some(f32t)),
        picker_entry(C::Math, "Absolute", "abs magnitude", ExprNode::Unary(UnaryOperator::Abs), Some(f32t)),
        picker_entry(C::Math, "Floor", "round down", ExprNode::Unary(UnaryOperator::Floor), Some(f32t)),
        picker_entry(C::Math, "Ceil", "ceiling round up", ExprNode::Unary(UnaryOperator::Ceil), Some(f32t)),
        picker_entry(C::Math, "Fract", "fractional frac", ExprNode::Unary(UnaryOperator::Fract), Some(f32t)),
        picker_entry(C::Math, "Round", "nearest", ExprNode::Unary(UnaryOperator::Round), Some(f32t)),
        picker_entry(C::Math, "Sign", "signum", ExprNode::Unary(UnaryOperator::Sign), Some(f32t)),
        picker_entry(C::Math, "Square root", "sqrt", ExprNode::Unary(UnaryOperator::Sqrt), Some(f32t)),
        picker_entry(C::Math, "Inverse square root", "rsqrt invsqrt", ExprNode::Unary(UnaryOperator::InvSqrt), Some(f32t)),
        picker_entry(C::Math, "Exp", "exponential e", ExprNode::Unary(UnaryOperator::Exp), Some(f32t)),
        picker_entry(C::Math, "Exp2", "exponential base 2", ExprNode::Unary(UnaryOperator::Exp2), Some(f32t)),
        picker_entry(C::Math, "Log", "logarithm natural ln", ExprNode::Unary(UnaryOperator::Log), Some(f32t)),
        picker_entry(C::Math, "Log2", "logarithm base 2", ExprNode::Unary(UnaryOperator::Log2), Some(f32t)),
        picker_entry(C::Math, "Saturate", "clamp01 clamp", ExprNode::Unary(UnaryOperator::Saturate), Some(f32t)),
        // Trigonometry.
        picker_entry(C::Trig, "Sine", "sin", ExprNode::Unary(UnaryOperator::Sin), Some(f32t)),
        picker_entry(C::Trig, "Cosine", "cos", ExprNode::Unary(UnaryOperator::Cos), Some(f32t)),
        picker_entry(C::Trig, "Tangent", "tan", ExprNode::Unary(UnaryOperator::Tan), Some(f32t)),
        picker_entry(C::Trig, "Arcsine", "asin", ExprNode::Unary(UnaryOperator::Asin), Some(f32t)),
        picker_entry(C::Trig, "Arccosine", "acos", ExprNode::Unary(UnaryOperator::Acos), Some(f32t)),
        picker_entry(C::Trig, "Arctangent", "atan", ExprNode::Unary(UnaryOperator::Atan), Some(f32t)),
        picker_entry(C::Trig, "Atan2", "arctangent2 atan2", ExprNode::Binary(BinaryOperator::Atan2), Some(f32t)),
        // Vector.
        picker_entry(C::Vector, "Vec2", "vector2 compose xy", ExprNode::Binary(BinaryOperator::Vec2), Some(vec2t)),
        picker_entry(C::Vector, "Vec3", "vector3 compose xyz", ExprNode::Ternary(TernaryOperator::Vec3), Some(vec3t)),
        picker_entry(C::Vector, "Vec4 (xyz, w)", "vector4 compose xyzw", ExprNode::Binary(BinaryOperator::Vec4XyzW), Some(vec4t)),
        picker_entry(C::Vector, "Cross product", "cross", ExprNode::Binary(BinaryOperator::Cross), Some(vec3t)),
        picker_entry(C::Vector, "Dot product", "dot", ExprNode::Binary(BinaryOperator::Dot), Some(f32t)),
        picker_entry(C::Vector, "Distance", "dist", ExprNode::Binary(BinaryOperator::Distance), Some(f32t)),
        picker_entry(C::Vector, "Length", "magnitude norm", ExprNode::Unary(UnaryOperator::Length), Some(f32t)),
        picker_entry(C::Vector, "Normalize", "unit direction", ExprNode::Unary(UnaryOperator::Normalize), None),
        // Interpolation.
        picker_entry(C::Interp, "Mix", "lerp linear interpolate blend", ExprNode::Ternary(TernaryOperator::Mix), None),
        picker_entry(C::Interp, "Clamp", "limit bound", ExprNode::Ternary(TernaryOperator::Clamp), None),
        picker_entry(C::Interp, "Smoothstep", "smooth interpolate ease", ExprNode::Ternary(TernaryOperator::SmoothStep), None),
        // Comparison.
        picker_entry(C::Comparison, "Greater than", "> gt greater", ExprNode::Binary(BinaryOperator::GreaterThan), Some(boolt)),
        picker_entry(C::Comparison, "Greater or equal", ">= gte", ExprNode::Binary(BinaryOperator::GreaterThanOrEqual), Some(boolt)),
        picker_entry(C::Comparison, "Less than", "< lt less", ExprNode::Binary(BinaryOperator::LessThan), Some(boolt)),
        picker_entry(C::Comparison, "Less or equal", "<= lte", ExprNode::Binary(BinaryOperator::LessThanOrEqual), Some(boolt)),
        // Logic.
        picker_entry(C::Logic, "All", "and reduce true", ExprNode::Unary(UnaryOperator::All), Some(boolt)),
        picker_entry(C::Logic, "Any", "or reduce true", ExprNode::Unary(UnaryOperator::Any), Some(boolt)),
        // Random.
        picker_entry(C::Random, "Uniform random", "random rand uniform range", ExprNode::Binary(BinaryOperator::UniformRand), Some(f32t)),
        picker_entry(C::Random, "Normal random", "random rand gaussian normal", ExprNode::Binary(BinaryOperator::NormalRand), Some(f32t)),
        // Bit manipulation.
        picker_entry(C::Bitwise, "Pack4x8 snorm", "pack snorm", ExprNode::Unary(UnaryOperator::Pack4x8snorm), Some(u32t)),
        picker_entry(C::Bitwise, "Pack4x8 unorm", "pack unorm", ExprNode::Unary(UnaryOperator::Pack4x8unorm), Some(u32t)),
        picker_entry(C::Bitwise, "Unpack4x8 snorm", "unpack snorm", ExprNode::Unary(UnaryOperator::Unpack4x8snorm), Some(vec4t)),
        picker_entry(C::Bitwise, "Unpack4x8 unorm", "unpack unorm", ExprNode::Unary(UnaryOperator::Unpack4x8unorm), Some(vec4t)),
    ];

    // Built-in source values.
    let builtins = [
        ("Time", "time elapsed", BuiltInOperator::Time),
        ("Delta time", "dt frame", BuiltInOperator::DeltaTime),
        ("Virtual time", "virtual", BuiltInOperator::VirtualTime),
        ("Real time", "real wall", BuiltInOperator::RealTime),
        ("Alpha cutoff", "alpha cutoff threshold", BuiltInOperator::AlphaCutoff),
    ];
    for (label, syn, op) in builtins {
        v.push(PickerNode {
            category: C::BuiltIn,
            search: format!("{label} {syn}").to_lowercase(),
            label: std::borrow::Cow::Borrowed(label),
            kind: add_expr(ExprNode::BuiltIn(op)),
            accepts_input: false,
            output_type: Some(op.value_type()),
        });
    }

    // Particle attributes (sources).
    for &attr in Attribute::all() {
        if attr == Attribute::ID || attr == Attribute::PARTICLE_COUNTER {
            continue;
        }
        let name = attr.name();
        v.push(PickerNode {
            category: C::Attribute,
            search: name.to_lowercase(),
            label: std::borrow::Cow::Owned(name.to_string()),
            kind: add_expr(ExprNode::Attribute(attr)),
            accepts_input: false,
            output_type: Some(attr.value_type()),
        });
    }

    // User properties (sources).
    for prop in &graph.properties {
        v.push(PickerNode {
            category: C::Property,
            search: prop.name.to_lowercase(),
            label: std::borrow::Cow::Owned(prop.name.to_string()),
            kind: add_expr(ExprNode::Property(prop.id)),
            accepts_input: false,
            output_type: Some(prop.default.value_type()),
        });
    }

    v
}

/// Persistent UI state of the rich create-node picker while it is open.
#[derive(Clone, Default)]
struct PickerState {
    search: String,
    /// Offer producers whose output implicitly casts to the pin's type.
    match_through_cast: bool,
    /// Ignore the pin's type entirely.
    show_all_types: bool,
}

/// How strictly a producer's output type must match the dangling pin's type.
#[derive(Clone, Copy, PartialEq)]
enum TypeMatch {
    Exact,
    Cast,
    All,
}

/// Render the rich create-node picker: a search box, optional type-relaxation
/// toggles (when opened from a link), and the filtered catalog laid out in
/// category columns that wrap sideways. Returns the chosen creation [`EditKind`].
fn picker_body(
    ui: &mut egui::Ui,
    catalog: &[PickerNode],
    link: Option<LinkSource>,
    pin_type: Option<ValueType>,
    filter: MenuFilter,
    state: &mut PickerState,
    opened_at: u64,
) -> Option<EditKind> {
    // The type-relaxation toggles only make sense when filtering producers by a
    // dangling input pin's type; an output drop offers polymorphic operators
    // whose input type can't be predicted, so there is nothing to relax.
    let producer_link = matches!(
        link,
        Some(LinkSource {
            source_is_output: false,
            ..
        })
    );
    let mode = if !producer_link || state.show_all_types {
        TypeMatch::All
    } else if state.match_through_cast {
        TypeMatch::Cast
    } else {
        TypeMatch::Exact
    };

    // Filter, group, and pack into columns *before* drawing anything: the menu's
    // width must be derived from the actual content. A floating Area otherwise
    // latches to the widest width it has ever shown (a full unfiltered catalog),
    // because `ui.separator()` and the scroll area expand to the available width.
    let query = state.search.to_lowercase();
    let tokens: Vec<&str> = query.split_whitespace().collect();
    let visible: Vec<&PickerNode> = catalog
        .iter()
        .filter(|n| {
            // Structural: an output drop needs a consumer (a node with an input).
            if filter == MenuFilter::Consumer && !n.accepts_input {
                return false;
            }
            // Type: only meaningful for a producer feeding a typed input pin.
            if producer_link && mode != TypeMatch::All {
                if let (Some(tt), Some(ot)) = (pin_type, n.output_type) {
                    let ok = match mode {
                        TypeMatch::Exact => ot == tt,
                        TypeMatch::Cast => can_cast(ot, tt),
                        TypeMatch::All => true,
                    };
                    if !ok {
                        return false;
                    }
                }
            }
            // Search: every token must appear in the node's haystack.
            tokens.iter().all(|tok| n.search.contains(tok))
        })
        .collect();

    let mut groups: Vec<(PickerCategory, Vec<&PickerNode>)> = Vec::new();
    for cat in PickerCategory::ALL {
        let items: Vec<&PickerNode> =
            visible.iter().copied().filter(|n| n.category == cat).collect();
        if !items.is_empty() {
            groups.push((cat, items));
        }
    }

    const COLUMN_BUDGET: usize = 13;
    let mut columns: Vec<Vec<(PickerCategory, Vec<&PickerNode>)>> = Vec::new();
    let mut current: Vec<(PickerCategory, Vec<&PickerNode>)> = Vec::new();
    let mut current_rows = 0usize;
    for (cat, items) in groups {
        let rows = items.len() + 1; // category header + its entries
        if current_rows > 0 && current_rows + rows > COLUMN_BUDGET {
            columns.push(std::mem::take(&mut current));
            current_rows = 0;
        }
        current_rows += rows;
        current.push((cat, items));
    }
    if !current.is_empty() {
        columns.push(current);
    }

    // Pin the menu width to the column count so a narrow (filtered) result can't
    // inherit a previous frame's wider layout. `COLUMN_W` comfortably fits the
    // longest label; `COLUMN_GAP` covers the inter-column separator and spacing.
    const COLUMN_W: f32 = 160.0;
    const COLUMN_GAP: f32 = 20.0;
    const SCROLLBAR_W: f32 = 16.0;
    let n = columns.len().max(1) as f32;
    let content_w = n * COLUMN_W + (n - 1.0) * COLUMN_GAP;
    let menu_width = content_w.max(232.0) + SCROLLBAR_W;
    ui.set_max_width(menu_width);

    let search = ui.add(
        egui::TextEdit::singleline(&mut state.search)
            .hint_text("Search nodes…")
            .desired_width(f32::INFINITY),
    );
    // Focus the search box on the pass the menu opened, so the user can type
    // immediately without first clicking it.
    if ui.ctx().cumulative_pass_nr() == opened_at {
        search.request_focus();
    }

    if producer_link {
        ui.horizontal(|ui| {
            ui.checkbox(&mut state.match_through_cast, "Match via cast")
                .on_hover_text(
                    "Also offer producers whose type implicitly casts to the pin \
                     (e.g. scalar splat into a vector).",
                );
            ui.checkbox(&mut state.show_all_types, "Show all")
                .on_hover_text("Ignore the pin's type and offer every compatible node.");
        });
    }
    ui.separator();

    if columns.is_empty() {
        ui.weak("No matching nodes");
        return None;
    }

    let mut chosen: Option<EditKind> = None;
    egui::ScrollArea::vertical()
        .max_height(440.0)
        .auto_shrink([false, true])
        .show(ui, |ui| {
            ui.horizontal_top(|ui| {
                for (ci, column) in columns.iter().enumerate() {
                    ui.vertical(|ui| {
                        ui.set_min_width(COLUMN_W);
                        for (cat, items) in column {
                            ui.strong(cat.label());
                            for n in items {
                                if ui.button(n.label.as_ref()).clicked() {
                                    chosen = Some(n.kind.clone());
                                }
                            }
                            ui.add_space(4.0);
                        }
                    });
                    if ci + 1 < columns.len() {
                        ui.separator();
                    }
                }
            });
        });

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
