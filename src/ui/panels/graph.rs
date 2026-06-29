//! Node-graph editor panel.
//!
//! Renders the [`NodeGraph`] widget directly against the document's canonical
//! [`EffectGraph`] via [`GraphReader`]: its expression nodes, ordered modifier
//! stacks (init/update/render), links, and inline-default value chips. Modifier
//! reordering, link create/delete, node create (a searchable, categorized
//! picker opened by right-click or by dragging a pin into empty space — the
//! latter type-filters candidates and auto-wires the chosen node), modifier
//! create (the "Add" button at the bottom of each stack opens a group-specific
//! modifier menu), node / stack deletion (the Delete key, or a per-modifier
//! header close button) and the shadowed-modifier warning badge are all wired
//! to the edit channel. A small toolbar toggles the grid and snapping.

use std::collections::HashSet;

use bevy::{
    ecs::{message::MessageWriter, reflect::AppTypeRegistry},
    prelude::{Entity, debug},
    reflect::TypeRegistry,
};
use bevy_egui::egui;
use bevy_hanabi::{
    Attribute, BuiltInOperator, ScalarType, ScalarValue, Value, ValueType, VectorType,
    graph::expr::{BinaryOperator, TernaryOperator, UnaryOperator},
};
use hanabi_node_graph::{
    ChipHit, CurveEditor, GradientBar, GraphAction, GraphView, NodeGraph, NodeId as WNodeId,
    PortAddr, PortId, WorldPos,
};

use super::value_edit;
use crate::{
    app_commands::{DialogKind, PendingFileDialogs},
    document::ModifierGroup,
    edits::{EditKind, EditRequest},
    effect_graph::{
        model::{
            EditValue, EffectGraph, ExprNode, GraphLink, ImageBinding, InputDefault, InputSlot,
            NodeId, PortRef, SharedStr,
        },
        schema::{
            FlagDef, OUTPUT_PORT, expr_has_image_input, expr_input_ports, expr_port_is_image,
            is_select_image_input,
        },
        view::{
            EditableChip, GraphReader, PortType, can_cast, group_of_widget_stack,
            keys_to_gradient3, keys_to_gradient4,
        },
    },
    modifier_registry,
};

pub fn show(
    ui: &mut egui::Ui,
    doc_entity: Entity,
    graph: &EffectGraph,
    effects: &bevy::asset::Assets<bevy_hanabi::EffectAsset>,
    effect_handle: &bevy::asset::Handle<bevy_hanabi::EffectAsset>,
    type_registry: &AppTypeRegistry,
    edits: &mut MessageWriter<EditRequest>,
    pending: &mut PendingFileDialogs,
    view: &mut GraphView,
) {
    let registry = type_registry.read();
    // Shadowed-modifier analysis runs against the baked preview asset (whose
    // modifier order matches the graph's stack members), feeding the per-node
    // warning badge. Absent while the asset is still loading.
    let shadowed = effects
        .get(effect_handle)
        .map(|asset| crate::effect_graph::validation::shadowed_modifiers(asset, &registry))
        .unwrap_or_default();
    let reader = GraphReader::new(graph, &registry)
        .with_shadows(shadowed)
        .with_expanded(read_expanded(ui, doc_entity));
    reader.seed_positions(view);

    egui::Panel::top("graph-toolbar")
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
                // The header close button (and Delete key) routes here. Free
                // expression nodes are removed directly; stack members map to a
                // RemoveModifier on their group. Members are dropped back-to-
                // front per group so earlier indices stay valid as edits apply.
                let mut members: Vec<(ModifierGroup, usize)> = Vec::new();
                for wid in nodes {
                    let Some(id) = NodeId::new(wid.get()) else {
                        continue;
                    };
                    match member_index(graph, id) {
                        Some(gi) => members.push(gi),
                        None => {
                            edits.write(EditRequest::new(doc_entity, EditKind::RemoveNode { id }));
                        }
                    }
                }
                members.sort_by(|a, b| b.1.cmp(&a.1));
                for (group, idx) in members {
                    edits.write(EditRequest::new(
                        doc_entity,
                        EditKind::RemoveModifier { group, idx },
                    ));
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
    chip_overlays(
        ui,
        doc_entity,
        &reader,
        resp.response.rect,
        &resp.chips,
        edits,
        pending,
    );
    chip_editor(ui, doc_entity, &reader, edits);
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

/// A pending create-node menu.
///
/// Where to draw it, the world position to place a new node at, the dangling
/// pin to auto-wire (if opened by a dropped link), and the egui pass it opened
/// on (so the very click that opened it can't be mistaken for a click-outside
/// that dismisses it the same frame).
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

/// A pending per-stack modifier menu.
///
/// Where to draw it, which modifier group it targets, and the pass it opened on
/// (for the same self-close guard as the create-node menu).
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

/// A pending inline value-chip editor.
///
/// Where to draw it, which widget port's chip was clicked, and the pass it
/// opened on (same self-close guard as the menus).
#[derive(Clone, Copy)]
struct PendingChipEdit {
    screen: egui::Pos2,
    port: PortAddr,
    opened_at: u64,
}

/// egui memory key for the pending value-chip editor of one document.
fn chip_edit_id(doc: Entity) -> egui::Id {
    egui::Id::new(("graph-chip-edit", doc))
}

/// The `(group, index)` of `id` within its modifier stack.
///
/// `None` if it's a free expression node.
fn member_index(graph: &EffectGraph, id: NodeId) -> Option<(ModifierGroup, usize)> {
    graph.stacks.iter().find_map(|s| {
        s.members
            .iter()
            .position(|&m| m == id)
            .map(|idx| (s.group, idx))
    })
}

/// Render the create-node context menu if one is pending.
///
/// Applies the chosen creation edit. When the menu was opened by a dropped
/// link, the chosen node is also auto-wired to the dangling pin. Dismissed on a
/// selection, an outside click, or `Escape`.
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

    // Whether the dangling input pin (producer drop) feeds the render stage. If
    // so, exposed-property producers are hidden — hanabi can't bind properties
    // in the render shader, the same reason a dragged such link is refused by
    // `validate_link`.
    let target_reaches_render = menu
        .link
        .filter(|ls| !ls.source_is_output)
        .and_then(|ls| NodeId::new(ls.source.node.get()))
        .is_some_and(|n| crate::ui::graph_validation::node_reaches_render(graph, n));

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
                    target_reaches_render,
                    filter,
                    &mut state,
                    menu.opened_at,
                );
            });
        });
    ui.ctx()
        .data_mut(|d| d.insert_temp(state_id, state.clone()));

    if let Some(kind) = chosen {
        // A standalone expression or image node is placed at the cursor; its id
        // is the next one the allocator will mint, so we can pre-seed the layout
        // position and build any auto-link before the edit applies. Modifier
        // nodes are positioned by their stack and need no seed.
        let standalone_inputs: Option<&[InputSlot]> = match &kind {
            EditKind::AddExprNode { inputs, .. } => Some(inputs),
            EditKind::AddImageNode => Some(&[]),
            _ => None,
        };
        if let Some(_inputs) = standalone_inputs {
            if let Some(wid) = WNodeId::new(graph.next_id) {
                view.ensure_position(wid, menu.at);
            }
            if let (
                Some(new_id),
                Some(LinkSource {
                    source,
                    source_is_output,
                }),
            ) = (NodeId::new(graph.next_id), menu.link)
            {
                if let Some(link) = auto_link(reader, new_id, &kind, source, source_is_output) {
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

/// Render the per-stack modifier menu if one is pending.
///
/// Emits an [`EditKind::AddModifierFromTemplate`] for the chosen modifier
/// (appended to the end of that group's stack). Dismissed on a selection, an
/// outside click, or `Escape`, with the same opening-pass self-close guard as
/// [`context_menu`].
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
                                let at = graph
                                    .stack(menu.group)
                                    .map(|s| s.members.len())
                                    .unwrap_or(0);
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

/// Overlay a real editor on every editable input value chip drawn this frame.
///
/// So the value can be edited directly on the node (no extra click).
///
/// Each control lives in a `Foreground` `Area` over the chip's screen rect.
/// egui routes a press on that rect to the overlay (using the previous frame's
/// layer order) before the canvas can claim it, so the node is not moved or
/// panned while editing. Scalars and bools get an inline editor; richer values
/// (vectors, enum attributes) fall back to a click target that opens the
/// `chip_editor` popup.
fn chip_overlays(
    ui: &mut egui::Ui,
    doc: Entity,
    reader: &GraphReader,
    canvas: egui::Rect,
    chips: &[ChipHit],
    edits: &mut MessageWriter<EditRequest>,
    pending: &mut PendingFileDialogs,
) {
    for hit in chips {
        // Skip chips scrolled off the canvas; visible ones are clipped to it.
        if !canvas.intersects(hit.rect) {
            continue;
        }
        let Some(chip) = reader.editable_chip(hit.port) else {
            continue;
        };
        // Clip every overlaid editor to the owning node so a wide control (a
        // long DragValue, a combo button) never spills past the node's border.
        let clip = canvas.intersect(hit.clip);
        match chip {
            EditableChip::Literal { node, port, value } => match value {
                Value::Scalar(
                    ScalarValue::Float(_)
                    | ScalarValue::Int(_)
                    | ScalarValue::Uint(_)
                    | ScalarValue::Bool(_),
                ) => {
                    if let Some(new) =
                        inline_chip_control(ui, ("chip-lit", doc, node, &port), clip, hit, value)
                    {
                        edits.write(EditRequest::new(
                            doc,
                            EditKind::SetInputDefault { node, port, new },
                        ));
                    }
                }
                // Vectors (and anything else) are too wide to scrub inline; a
                // click opens the popup editor instead.
                _ => {
                    if chip_click_target(ui, ("chip-vec", doc, node, &port), clip, hit.rect) {
                        open_chip_popup(ui, doc, hit.port, hit.rect);
                    }
                }
            },
            EditableChip::Bool { node, field, value } => {
                if let Some(new) =
                    inline_checkbox(ui, ("chip-bool", doc, node, &field), clip, hit, value)
                {
                    edits.write(EditRequest::new(
                        doc,
                        EditKind::SetModifierConfig {
                            node,
                            field,
                            new: EditValue::Bool(new),
                        },
                    ));
                }
            }
            EditableChip::Attribute {
                group,
                idx,
                current,
            } => {
                let names: Vec<&str> = Attribute::all().iter().map(|a| a.name()).collect();
                if let Some(sel) = inline_combo(
                    ui,
                    ("chip-attr", doc, hit.port),
                    clip,
                    hit,
                    current.name(),
                    &names,
                ) {
                    if let Some(new) = Attribute::from_name(names[sel]) {
                        edits.write(EditRequest::new(
                            doc,
                            EditKind::SetModifierAttribute {
                                group,
                                idx,
                                new,
                                reset_value: None,
                            },
                        ));
                    }
                }
            }
            EditableChip::Enum {
                node,
                field,
                type_path,
                current,
                variants,
            } => {
                let names: Vec<&str> = variants.iter().map(|v| v.as_ref()).collect();
                if let Some(sel) = inline_combo(
                    ui,
                    ("chip-enum", doc, node, &field),
                    clip,
                    hit,
                    &current,
                    &names,
                ) {
                    edits.write(EditRequest::new(
                        doc,
                        EditKind::SetModifierConfig {
                            node,
                            field,
                            new: EditValue::Enum {
                                type_path,
                                variant: variants[sel].clone(),
                            },
                        },
                    ));
                }
            }
            EditableChip::Flags {
                node,
                field,
                type_path,
                bits,
                defs,
            } => {
                if let Some(new_bits) = inline_flags(
                    ui,
                    ("chip-flags", doc, node, &field),
                    clip,
                    hit,
                    bits,
                    &defs,
                ) {
                    edits.write(EditRequest::new(
                        doc,
                        EditKind::SetModifierConfig {
                            node,
                            field,
                            new: EditValue::Flags {
                                type_path,
                                bits: new_bits,
                            },
                        },
                    ));
                }
            }
            EditableChip::ImageBinding {
                node,
                port,
                current,
                slots,
            } => {
                // Options: unbound, pick-an-asset, then each texture slot.
                let cur = match &current {
                    ImageBinding::Unbound => "(unbound)".to_string(),
                    ImageBinding::Asset(path) => {
                        let s = path.to_string();
                        s.rsplit(['/', '\\']).next().unwrap_or(&s).to_string()
                    }
                    ImageBinding::Slot(id) => slots
                        .iter()
                        .find(|(sid, _)| sid == id)
                        .map(|(_, n)| format!("[{n}]"))
                        .unwrap_or_else(|| "[missing]".to_string()),
                };
                let mut options: Vec<String> = vec!["(unbound)".into(), "Asset…".into()];
                options.extend(slots.iter().map(|(_, n)| format!("[{n}]")));
                let labels: Vec<&str> = options.iter().map(String::as_str).collect();
                // Either the inline binding of a consumer port, or an Image node.
                let make_edit = |binding: ImageBinding| match &port {
                    Some(p) => EditKind::SetInputImageBinding {
                        node,
                        port: p.clone(),
                        binding,
                    },
                    None => EditKind::SetImageNodeBinding { node, binding },
                };
                let combo_id = ("chip-imgbind", doc, node);
                if let Some(sel) = inline_combo(ui, combo_id, clip, hit, &cur, &labels) {
                    match sel {
                        0 => {
                            edits.write(EditRequest::new(doc, make_edit(ImageBinding::Unbound)));
                        }
                        1 => pending.spawn(DialogKind::BindImageNode {
                            doc,
                            node,
                            port: port.clone(),
                        }),
                        i => {
                            edits.write(EditRequest::new(
                                doc,
                                make_edit(ImageBinding::Slot(slots[i - 2].0)),
                            ));
                        }
                    }
                }
            }
            EditableChip::Gradient3 { node, field, keys } => {
                if chevron_toggle(ui, doc, node, &field, hit) {
                    continue;
                }
                if hit.expanded {
                    if let Some(new) =
                        curve_inline_editor(ui, ("grad3", doc, node, &field), clip, hit.rect, keys)
                    {
                        edits.write(EditRequest::new(
                            doc,
                            EditKind::SetModifierConfig { node, field, new },
                        ));
                    }
                } else {
                    curve_preview(ui, ("grad3-prev", doc, node, &field), clip, hit.rect, &keys);
                }
            }
            EditableChip::Gradient4 { node, field, keys } => {
                if chevron_toggle(ui, doc, node, &field, hit) {
                    continue;
                }
                if hit.expanded {
                    if let Some(new) = gradient_inline_editor(
                        ui,
                        ("grad4", doc, node, &field),
                        clip,
                        hit.rect,
                        keys,
                    ) {
                        edits.write(EditRequest::new(
                            doc,
                            EditKind::SetModifierConfig { node, field, new },
                        ));
                    }
                } else {
                    gradient_preview(ui, ("grad4-prev", doc, node, &field), clip, hit.rect, &keys);
                }
            }
        }
    }
}
/// Overlay an inline value editor covering the widget-drawn chip.
///
/// Matches its zoom-scaled font and box so it reads as part of the node.
/// Returns `Some(new)` on the frame the gesture commits. Painting is clipped to
/// `clip` (the owning node's rect) so a wide editor never spills past it.
fn inline_chip_control(
    ui: &mut egui::Ui,
    id_base: impl std::hash::Hash + Copy,
    clip: egui::Rect,
    hit: &ChipHit,
    value: Value,
) -> Option<Value> {
    let rect = hit.rect;
    let mut out = None;
    egui::Area::new(egui::Id::new(("chip-area", id_base)))
        .order(egui::Order::Foreground)
        .fixed_pos(rect.min)
        .show(ui.ctx(), |ui| {
            ui.set_clip_rect(clip);
            // Match the chip's font and padding, and drop egui's default minimum
            // interact size, so the control is exactly the chip's size at any
            // zoom (otherwise it renders oversized and spills over the port name).
            // DragValue resolves its font via the `Button` text style, so both
            // that entry and the global override are set.
            let font = egui::FontId::monospace(hit.font_size);
            ui.spacing_mut().interact_size = egui::Vec2::ZERO;
            ui.spacing_mut().button_padding = egui::vec2(hit.pad, hit.pad * 0.5);
            ui.style_mut().override_font_id = Some(font.clone());
            ui.style_mut()
                .text_styles
                .insert(egui::TextStyle::Button, font);
            // Cover the chip the widget painted so only this control shows.
            let rr = rect.height() * 0.25;
            ui.painter()
                .rect_filled(rect, rr, ui.visuals().extreme_bg_color);
            out = value_edit::inline_value_editor(ui, id_base, value, rect.size());
        });
    out
}

/// A transparent click target over `rect` (for chips edited via the popup).
///
/// Returns whether it was clicked this frame. Clipped to `clip`.
fn chip_click_target(
    ui: &mut egui::Ui,
    id_base: impl std::hash::Hash + Copy,
    clip: egui::Rect,
    rect: egui::Rect,
) -> bool {
    let mut clicked = false;
    egui::Area::new(egui::Id::new(("chip-click", id_base)))
        .order(egui::Order::Foreground)
        .fixed_pos(rect.min)
        .show(ui.ctx(), |ui| {
            ui.set_clip_rect(clip);
            let resp = ui.allocate_rect(rect, egui::Sense::click());
            if resp.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            }
            clicked = resp.clicked();
        });
    clicked
}

/// Overlay a curve editor inline in a `Vec3` gradient's reserved chip box.
///
/// Draft keys live in egui memory keyed by `id_base` so a drag survives across
/// frames; returns the new [`EditValue`] only on the frame a gesture commits.
/// Painting and interaction are clipped to `clip` (the owning node body).
fn curve_inline_editor(
    ui: &mut egui::Ui,
    id_base: impl std::hash::Hash + Copy,
    clip: egui::Rect,
    rect: egui::Rect,
    seed: Vec<(f32, f32)>,
) -> Option<EditValue> {
    let mut out = None;
    egui::Area::new(egui::Id::new(("grad-edit", id_base)))
        .order(egui::Order::Foreground)
        .fixed_pos(rect.min)
        .show(ui.ctx(), |ui| {
            ui.set_clip_rect(clip.intersect(rect));
            let key = egui::Id::new(("grad3-draft", id_base));
            let mut keys: Vec<(f32, f32)> = ui.ctx().data_mut(|d| d.get_temp(key)).unwrap_or(seed);
            ui.set_width(rect.width());
            let resp = CurveEditor::new(&mut keys)
                .y_range(0.0, 2.0)
                .height(rect.height())
                .show(ui);
            ui.ctx().data_mut(|d| d.insert_temp(key, keys.clone()));
            if resp.committed {
                out = Some(keys_to_gradient3(&keys));
            }
        });
    out
}

/// Overlay a gradient-bar editor inline in a `Vec4` gradient's reserved box.
///
/// See [`curve_inline_editor`]; commits on a stop drag, add, remove, or color
/// pick.
fn gradient_inline_editor(
    ui: &mut egui::Ui,
    id_base: impl std::hash::Hash + Copy,
    clip: egui::Rect,
    rect: egui::Rect,
    seed: Vec<(f32, [f32; 4])>,
) -> Option<EditValue> {
    let mut out = None;
    egui::Area::new(egui::Id::new(("grad-edit", id_base)))
        .order(egui::Order::Foreground)
        .fixed_pos(rect.min)
        .show(ui.ctx(), |ui| {
            ui.set_clip_rect(clip.intersect(rect));
            let key = egui::Id::new(("grad4-draft", id_base));
            let mut keys: Vec<(f32, [f32; 4])> =
                ui.ctx().data_mut(|d| d.get_temp(key)).unwrap_or(seed);
            ui.set_width(rect.width());
            let resp = GradientBar::new(&mut keys)
                .height((rect.height() - 24.0).clamp(10.0, 40.0))
                .show(ui);
            ui.ctx().data_mut(|d| d.insert_temp(key, keys.clone()));
            if resp.committed {
                out = Some(keys_to_gradient4(&keys));
            }
        });
    out
}

/// egui-memory id holding the set of expanded gradient editors for a document.
fn grad_expanded_id(doc: Entity) -> egui::Id {
    egui::Id::new(("grad-expanded", doc))
}

/// The `(node, field)` pairs whose gradient editor is expanded for `doc`.
fn read_expanded(ui: &egui::Ui, doc: Entity) -> HashSet<(u32, String)> {
    ui.ctx()
        .data_mut(|d| d.get_temp(grad_expanded_id(doc)))
        .unwrap_or_default()
}

/// Sense a click on a collapsible row's chevron and toggle its expanded state.
///
/// Returns whether it was toggled this frame, so the caller can skip drawing
/// the now-stale collapsed/expanded body until the next layout pass.
fn chevron_toggle(
    ui: &mut egui::Ui,
    doc: Entity,
    node: NodeId,
    field: &str,
    hit: &ChipHit,
) -> bool {
    let Some(ch) = hit.chevron else {
        return false;
    };
    let mut clicked = false;
    egui::Area::new(egui::Id::new(("grad-chevron", doc, node, field)))
        .order(egui::Order::Foreground)
        .fixed_pos(ch.min)
        .show(ui.ctx(), |ui| {
            ui.set_clip_rect(hit.clip);
            let resp = ui.allocate_rect(ch, egui::Sense::click());
            if resp.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            }
            clicked = resp.clicked();
        });
    if clicked {
        let entry = (node.get(), field.to_string());
        ui.ctx().data_mut(|d| {
            let mut set: HashSet<(u32, String)> =
                d.get_temp(grad_expanded_id(doc)).unwrap_or_default();
            if !set.remove(&entry) {
                set.insert(entry);
            }
            d.insert_temp(grad_expanded_id(doc), set);
        });
        ui.ctx().request_repaint();
    }
    clicked
}

/// Paint a collapsed `Vec3` gradient as a small line preview over its chip box.
fn curve_preview(
    ui: &mut egui::Ui,
    id_base: impl std::hash::Hash + Copy,
    clip: egui::Rect,
    rect: egui::Rect,
    keys: &[(f32, f32)],
) {
    let mut sorted = keys.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    egui::Area::new(egui::Id::new(("grad-prev", id_base)))
        .order(egui::Order::Foreground)
        .fixed_pos(rect.min)
        .show(ui.ctx(), |ui| {
            ui.set_clip_rect(clip.intersect(rect));
            let inner = rect.shrink(2.0);
            let pts: Vec<egui::Pos2> = sorted
                .iter()
                .map(|&(r, v)| {
                    let x = inner.left() + r.clamp(0.0, 1.0) * inner.width();
                    let y = inner.bottom() - (v / 2.0).clamp(0.0, 1.0) * inner.height();
                    egui::Pos2::new(x, y)
                })
                .collect();
            let color = egui::Color32::from_rgb(120, 200, 255);
            if pts.len() >= 2 {
                ui.painter()
                    .add(egui::Shape::line(pts, egui::Stroke::new(1.5, color)));
            } else if let Some(&p) = pts.first() {
                ui.painter().circle_filled(p, 1.5, color);
            }
        });
}

/// Paint a collapsed `Vec4` gradient as a color strip over its chip box.
fn gradient_preview(
    ui: &mut egui::Ui,
    id_base: impl std::hash::Hash + Copy,
    clip: egui::Rect,
    rect: egui::Rect,
    keys: &[(f32, [f32; 4])],
) {
    let mut sorted = keys.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    egui::Area::new(egui::Id::new(("grad-prev", id_base)))
        .order(egui::Order::Foreground)
        .fixed_pos(rect.min)
        .show(ui.ctx(), |ui| {
            ui.set_clip_rect(clip.intersect(rect));
            let inner = rect.shrink(1.0);
            let painter = ui.painter();
            let steps = (inner.width() as usize / 2).max(1);
            for i in 0..steps {
                let t0 = i as f32 / steps as f32;
                let t1 = (i + 1) as f32 / steps as f32;
                let x0 = inner.left() + t0 * inner.width();
                let x1 = inner.left() + t1 * inner.width();
                let c = sample_gradient(&sorted, (t0 + t1) * 0.5);
                painter.rect_filled(
                    egui::Rect::from_min_max(
                        egui::Pos2::new(x0, inner.top()),
                        egui::Pos2::new(x1, inner.bottom()),
                    ),
                    0.0,
                    c,
                );
            }
        });
}

/// Linearly sample a sorted `(ratio, rgba)` color gradient at `t ∈ [0, 1]`.
fn sample_gradient(sorted: &[(f32, [f32; 4])], t: f32) -> egui::Color32 {
    let to_c = |c: [f32; 4]| {
        egui::Color32::from(egui::Rgba::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]))
    };
    match sorted {
        [] => egui::Color32::TRANSPARENT,
        [single] => to_c(single.1),
        _ => {
            if t <= sorted[0].0 {
                return to_c(sorted[0].1);
            }
            if t >= sorted[sorted.len() - 1].0 {
                return to_c(sorted[sorted.len() - 1].1);
            }
            let hi = sorted
                .iter()
                .position(|k| k.0 >= t)
                .unwrap_or(sorted.len() - 1);
            let (r0, a) = sorted[hi - 1];
            let (r1, b) = sorted[hi];
            let f = if r1 > r0 { (t - r0) / (r1 - r0) } else { 0.0 };
            let mix = std::array::from_fn(|i| a[i] + (b[i] - a[i]) * f);
            to_c(mix)
        }
    }
}

/// Overlay a checkbox on a modifier's `bool` config chip.
///
/// Draws a single compact square toggle (a recessed box with a checkmark when
/// set) at the chip's left, scaled to the chip's zoom level, and senses clicks
/// on it. Returns the new value on the frame it is toggled. The chip itself
/// carries no text, so the box reads as a self-contained checkbox.
fn inline_checkbox(
    ui: &mut egui::Ui,
    id_base: impl std::hash::Hash + Copy,
    clip: egui::Rect,
    hit: &ChipHit,
    value: bool,
) -> Option<bool> {
    // A square box the height of the chip row, anchored at its left.
    let side = hit.rect.height();
    let box_rect = egui::Rect::from_min_size(hit.rect.min, egui::Vec2::splat(side));
    let mut out = None;
    egui::Area::new(egui::Id::new(("chip-bool", id_base)))
        .order(egui::Order::Foreground)
        .fixed_pos(box_rect.min)
        .show(ui.ctx(), |ui| {
            ui.set_clip_rect(clip);
            let resp = ui.allocate_rect(box_rect, egui::Sense::click());
            if resp.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            }
            let visuals = ui.style().interact(&resp);
            let rr = side * 0.2;
            ui.painter().rect(
                box_rect,
                rr,
                ui.visuals().extreme_bg_color,
                egui::Stroke::new(1.0, visuals.bg_stroke.color),
                egui::StrokeKind::Inside,
            );
            if value {
                // A checkmark inscribed in the box.
                let inner = box_rect.shrink(side * 0.28);
                let pts = vec![
                    egui::Pos2::new(inner.left(), inner.center().y),
                    egui::Pos2::new(inner.left() + inner.width() * 0.35, inner.bottom()),
                    egui::Pos2::new(inner.right(), inner.top()),
                ];
                let w = (side * 0.12).clamp(1.0, 3.0);
                ui.painter().add(egui::Shape::line(
                    pts,
                    egui::Stroke::new(w, visuals.fg_stroke.color),
                ));
            }
            if resp.clicked() {
                out = Some(!value);
            }
        });
    out
}

/// Overlay an `egui::ComboBox` on the chip for a data-less enum / attribute.
///
/// Matches the chip's zoom-scaled font. Returns the index of the option the
/// user just selected (if any). The dropdown list itself renders at normal size
/// in its own popup, unaffected by the chip's tiny font.
fn inline_combo(
    ui: &mut egui::Ui,
    id_base: impl std::hash::Hash + Copy,
    clip: egui::Rect,
    hit: &ChipHit,
    current: &str,
    options: &[&str],
) -> Option<usize> {
    let rect = hit.rect;
    let mut chosen = None;
    egui::Area::new(egui::Id::new(("chip-combo", id_base)))
        .order(egui::Order::Foreground)
        .fixed_pos(rect.min)
        .show(ui.ctx(), |ui| {
            ui.set_clip_rect(clip);
            let font = egui::FontId::proportional(hit.font_size);
            ui.spacing_mut().interact_size = egui::Vec2::ZERO;
            ui.spacing_mut().button_padding = egui::vec2(hit.pad, hit.pad * 0.5);
            ui.style_mut().override_font_id = Some(font.clone());
            ui.style_mut()
                .text_styles
                .insert(egui::TextStyle::Button, font);
            // Ellipsize the selected text rather than letting it spill or get
            // hard-clipped at the node border.
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
            // Fill from the chip's left to the inset node margin: bound the ui so
            // the selected text truncates at the margin (not the far screen edge)
            // and the closed combo fills the free space instead of hugging left.
            let avail = (clip.max.x - rect.min.x).max(rect.width());
            ui.set_max_width(avail);
            let rr = rect.height() * 0.25;
            ui.painter()
                .rect_filled(rect, rr, ui.visuals().extreme_bg_color);
            egui::ComboBox::from_id_salt(egui::Id::new(("chip-combo-box", id_base)))
                .selected_text(current)
                .width(avail)
                .show_ui(ui, |ui| {
                    // The dropdown list shows at the normal theme size, not the
                    // chip's tiny font, and sizes to its content (one line per
                    // entry) up to a reasonable maximum.
                    *ui.style_mut() = (*ui.ctx().global_style()).clone();
                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                    let body = egui::TextStyle::Body.resolve(ui.style());
                    let widest = options
                        .iter()
                        .map(|o| {
                            ui.painter()
                                .layout_no_wrap((*o).to_owned(), body.clone(), egui::Color32::WHITE)
                                .size()
                                .x
                        })
                        .fold(0.0_f32, f32::max);
                    let pad = ui.spacing().button_padding.x * 2.0 + ui.spacing().item_spacing.x;
                    ui.set_min_width((widest + pad + 8.0).min(360.0));
                    for (i, opt) in options.iter().enumerate() {
                        if ui.selectable_label(*opt == current, *opt).clicked() {
                            chosen = Some(i);
                        }
                    }
                });
        });
    chosen
}

/// Overlay a bitflags editor on the chip: a combo button showing the active
/// flag names that opens a checklist of independently-toggleable bits.
///
/// Returns the new mask on the frame a bit is toggled. The dropdown renders at
/// normal size; only the chip button matches the zoom-scaled font.
fn inline_flags(
    ui: &mut egui::Ui,
    id_base: impl std::hash::Hash + Copy,
    clip: egui::Rect,
    hit: &ChipHit,
    bits: u64,
    defs: &[FlagDef],
) -> Option<u64> {
    let rect = hit.rect;
    let mut new_bits = None;
    let summary = {
        let active: Vec<&str> = defs
            .iter()
            .filter(|d| bits & d.bits != 0)
            .map(|d| d.name)
            .collect();
        if active.is_empty() {
            "none".to_string()
        } else {
            active.join("|")
        }
    };
    egui::Area::new(egui::Id::new(("chip-flags", id_base)))
        .order(egui::Order::Foreground)
        .fixed_pos(rect.min)
        .show(ui.ctx(), |ui| {
            ui.set_clip_rect(clip);
            let font = egui::FontId::proportional(hit.font_size);
            ui.spacing_mut().interact_size = egui::Vec2::ZERO;
            ui.spacing_mut().button_padding = egui::vec2(hit.pad, hit.pad * 0.5);
            ui.style_mut().override_font_id = Some(font.clone());
            ui.style_mut()
                .text_styles
                .insert(egui::TextStyle::Button, font);
            // Ellipsize the active-flags summary rather than letting it spill or
            // get hard-clipped at the node border.
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
            // Fill from the chip's left to the inset node margin (see inline_combo).
            let avail = (clip.max.x - rect.min.x).max(rect.width());
            ui.set_max_width(avail);
            let rr = rect.height() * 0.25;
            ui.painter()
                .rect_filled(rect, rr, ui.visuals().extreme_bg_color);
            egui::ComboBox::from_id_salt(egui::Id::new(("chip-flags-box", id_base)))
                .selected_text(summary)
                .width(avail)
                .show_ui(ui, |ui| {
                    // Reset to the theme size so the checklist is legible
                    // regardless of the chip's tiny zoom-scaled font.
                    *ui.style_mut() = (*ui.ctx().global_style()).clone();
                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                    for def in defs {
                        let mut on = bits & def.bits != 0;
                        if ui.checkbox(&mut on, def.name).changed() {
                            new_bits = Some(if on {
                                bits | def.bits
                            } else {
                                bits & !def.bits
                            });
                        }
                    }
                });
        });
    new_bits
}

/// Record a pending chip popup just below the chip, for `chip_editor` to draw.
fn open_chip_popup(ui: &mut egui::Ui, doc: Entity, port: PortAddr, rect: egui::Rect) {
    let opened_at = ui.ctx().cumulative_pass_nr();
    ui.ctx().data_mut(|d| {
        d.insert_temp(
            chip_edit_id(doc),
            PendingChipEdit {
                screen: rect.left_bottom(),
                port,
                opened_at,
            },
        )
    });
}

/// Render the inline value-chip editor popup if one is pending.
///
/// Resolves the clicked widget port back to its model target via
/// [`GraphReader::editable_chip`] and presents a type-appropriate editor,
/// emitting the matching edit on commit. Dismissed on an outside click or
/// `Escape`, with the same opening-pass guard as the menus.
fn chip_editor(
    ui: &mut egui::Ui,
    doc: Entity,
    reader: &GraphReader,
    edits: &mut MessageWriter<EditRequest>,
) {
    let id = chip_edit_id(doc);
    let Some(pending) = ui.ctx().data(|d| d.get_temp::<PendingChipEdit>(id)) else {
        return;
    };

    // Re-resolve the chip each frame; the target may have vanished (e.g. the
    // node was deleted) — close if so.
    let Some(chip) = reader.editable_chip(pending.port) else {
        ui.ctx().data_mut(|d| d.remove::<PendingChipEdit>(id));
        return;
    };

    let mut close = false;
    let area = egui::Area::new(id.with("area"))
        .order(egui::Order::Foreground)
        .fixed_pos(pending.screen)
        .show(ui.ctx(), |ui| {
            egui::Frame::menu(ui.style()).show(ui, |ui| {
                match chip {
                    EditableChip::Literal { node, port, value } => {
                        if let Some(new) =
                            value_edit::value_editor(ui, ("chip", doc, node, &port), value)
                        {
                            edits.write(EditRequest::new(
                                doc,
                                EditKind::SetInputDefault { node, port, new },
                            ));
                        }
                    }
                    // Attribute and enum chips are edited inline via a combo box,
                    // never through this popup; gradients edit inline in-node.
                    _ => {}
                }
            });
        });

    let opened_this_pass = ui.ctx().cumulative_pass_nr() == pending.opened_at;
    if (!opened_this_pass && area.response.clicked_elsewhere())
        || ui.ctx().input(|i| i.key_pressed(egui::Key::Escape))
    {
        close = true;
    }
    if close {
        ui.ctx().data_mut(|d| d.remove::<PendingChipEdit>(id));
    }
}

/// Build the link wiring a freshly-created expression node to the dangling pin.
/// Wires `new_id` (with the given operand `inputs`) to the pin that opened the
/// menu. An output `source` feeds the new node's first input; an input `source`
/// is fed by the new node's output.
fn auto_link(
    reader: &GraphReader,
    new_id: NodeId,
    kind: &EditKind,
    source: PortAddr,
    source_is_output: bool,
) -> Option<GraphLink> {
    let source_node = NodeId::new(source.node.get())?;
    if source_is_output {
        // Output → the new consumer's first type-compatible input port, so an
        // image output lands on an image port and a value output on a value
        // port regardless of the schema's port order.
        let EditKind::AddExprNode { expr, .. } = kind else {
            return None;
        };
        let ports = expr_input_ports(expr);
        let source_is_image = reader.port_type(source, true) == Some(PortType::Image);
        let port = ports
            .iter()
            .find(|p| expr_port_is_image(expr, p) == source_is_image)
            .or_else(|| ports.first())
            .copied()?;
        Some(GraphLink {
            from: PortRef {
                node: source_node,
                port: OUTPUT_PORT.into(),
            },
            to: PortRef {
                node: new_id,
                port: port.into(),
            },
        })
    } else {
        // The new producer's output → the source input port. Resolve the input
        // port name through the reader, which exists for the source node.
        let new_out = PortAddr::new(WNodeId::new(new_id.get())?, PortId::output(0));
        reader.resolve_link(new_out, source)
    }
}
/// User-facing grouping of create-node entries.
///
/// Categories read by intent (Math, Trigonometry, …) rather than by operator
/// arity.
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
    Texture,
}

impl PickerCategory {
    /// Categories in display order.
    const ALL: [PickerCategory; 12] = [
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
        PickerCategory::Texture,
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
            PickerCategory::Texture => "Texture",
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
    /// Whether the node has an image-typed input port (only the texture
    /// sampler). Distinguishes the image pseudo-type from value operands so an
    /// image output offers only image-accepting consumers.
    has_image_input: bool,
    /// Natural output type, when statically known (`None` = operand
    /// dependent / unknown, so never type-filtered out).
    output_type: Option<PortType>,
    /// True for a reference to an *exposed* user property. Such a value can't
    /// enter the render context (hanabi has no render-shader property binding),
    /// so the menu hides it when the dangling input pin reaches the render
    /// stage.
    is_exposed_property: bool,
}

/// Build a [`PickerNode`] for an expression.
///
/// Takes a `'static` label and a space-separated list of search synonyms.
fn picker_entry(
    category: PickerCategory,
    label: &'static str,
    synonyms: &'static str,
    expr: ExprNode,
    output_type: Option<ValueType>,
) -> PickerNode {
    let ports = expr_input_ports(&expr);
    PickerNode {
        category,
        search: format!("{label} {synonyms}").to_lowercase(),
        label: std::borrow::Cow::Borrowed(label),
        accepts_input: !ports.is_empty(),
        has_image_input: expr_has_image_input(&expr),
        kind: add_expr(expr),
        output_type: output_type.map(PortType::Value),
        is_exposed_property: false,
    }
}

/// The full catalog of create-node entries, grouped by category.
///
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
        picker_entry(
            C::Math,
            "Add",
            "+ plus sum",
            ExprNode::Binary(BinaryOperator::Add),
            Some(f32t),
        ),
        picker_entry(
            C::Math,
            "Subtract",
            "- minus difference",
            ExprNode::Binary(BinaryOperator::Sub),
            Some(f32t),
        ),
        picker_entry(
            C::Math,
            "Multiply",
            "* times product",
            ExprNode::Binary(BinaryOperator::Mul),
            Some(f32t),
        ),
        picker_entry(
            C::Math,
            "Divide",
            "/ quotient ratio",
            ExprNode::Binary(BinaryOperator::Div),
            Some(f32t),
        ),
        picker_entry(
            C::Math,
            "Remainder",
            "% mod modulo",
            ExprNode::Binary(BinaryOperator::Remainder),
            Some(f32t),
        ),
        picker_entry(
            C::Math,
            "Minimum",
            "min",
            ExprNode::Binary(BinaryOperator::Min),
            Some(f32t),
        ),
        picker_entry(
            C::Math,
            "Maximum",
            "max",
            ExprNode::Binary(BinaryOperator::Max),
            Some(f32t),
        ),
        picker_entry(
            C::Math,
            "Step",
            "threshold",
            ExprNode::Binary(BinaryOperator::Step),
            Some(f32t),
        ),
        picker_entry(
            C::Math,
            "Absolute",
            "abs magnitude",
            ExprNode::Unary(UnaryOperator::Abs),
            Some(f32t),
        ),
        picker_entry(
            C::Math,
            "Floor",
            "round down",
            ExprNode::Unary(UnaryOperator::Floor),
            Some(f32t),
        ),
        picker_entry(
            C::Math,
            "Ceil",
            "ceiling round up",
            ExprNode::Unary(UnaryOperator::Ceil),
            Some(f32t),
        ),
        picker_entry(
            C::Math,
            "Fract",
            "fractional frac",
            ExprNode::Unary(UnaryOperator::Fract),
            Some(f32t),
        ),
        picker_entry(
            C::Math,
            "Round",
            "nearest",
            ExprNode::Unary(UnaryOperator::Round),
            Some(f32t),
        ),
        picker_entry(
            C::Math,
            "Sign",
            "signum",
            ExprNode::Unary(UnaryOperator::Sign),
            Some(f32t),
        ),
        picker_entry(
            C::Math,
            "Square root",
            "sqrt",
            ExprNode::Unary(UnaryOperator::Sqrt),
            Some(f32t),
        ),
        picker_entry(
            C::Math,
            "Inverse square root",
            "rsqrt invsqrt",
            ExprNode::Unary(UnaryOperator::InvSqrt),
            Some(f32t),
        ),
        picker_entry(
            C::Math,
            "Exp",
            "exponential e",
            ExprNode::Unary(UnaryOperator::Exp),
            Some(f32t),
        ),
        picker_entry(
            C::Math,
            "Exp2",
            "exponential base 2",
            ExprNode::Unary(UnaryOperator::Exp2),
            Some(f32t),
        ),
        picker_entry(
            C::Math,
            "Log",
            "logarithm natural ln",
            ExprNode::Unary(UnaryOperator::Log),
            Some(f32t),
        ),
        picker_entry(
            C::Math,
            "Log2",
            "logarithm base 2",
            ExprNode::Unary(UnaryOperator::Log2),
            Some(f32t),
        ),
        picker_entry(
            C::Math,
            "Saturate",
            "clamp01 clamp",
            ExprNode::Unary(UnaryOperator::Saturate),
            Some(f32t),
        ),
        // Trigonometry.
        picker_entry(
            C::Trig,
            "Sine",
            "sin",
            ExprNode::Unary(UnaryOperator::Sin),
            Some(f32t),
        ),
        picker_entry(
            C::Trig,
            "Cosine",
            "cos",
            ExprNode::Unary(UnaryOperator::Cos),
            Some(f32t),
        ),
        picker_entry(
            C::Trig,
            "Tangent",
            "tan",
            ExprNode::Unary(UnaryOperator::Tan),
            Some(f32t),
        ),
        picker_entry(
            C::Trig,
            "Arcsine",
            "asin",
            ExprNode::Unary(UnaryOperator::Asin),
            Some(f32t),
        ),
        picker_entry(
            C::Trig,
            "Arccosine",
            "acos",
            ExprNode::Unary(UnaryOperator::Acos),
            Some(f32t),
        ),
        picker_entry(
            C::Trig,
            "Arctangent",
            "atan",
            ExprNode::Unary(UnaryOperator::Atan),
            Some(f32t),
        ),
        picker_entry(
            C::Trig,
            "Atan2",
            "arctangent2 atan2",
            ExprNode::Binary(BinaryOperator::Atan2),
            Some(f32t),
        ),
        // Vector.
        picker_entry(
            C::Vector,
            "Vec2",
            "vector2 compose xy",
            ExprNode::Binary(BinaryOperator::Vec2),
            Some(vec2t),
        ),
        picker_entry(
            C::Vector,
            "Vec3",
            "vector3 compose xyz",
            ExprNode::Ternary(TernaryOperator::Vec3),
            Some(vec3t),
        ),
        picker_entry(
            C::Vector,
            "Vec4 (xyz, w)",
            "vector4 compose xyzw",
            ExprNode::Binary(BinaryOperator::Vec4XyzW),
            Some(vec4t),
        ),
        picker_entry(
            C::Vector,
            "Cross product",
            "cross",
            ExprNode::Binary(BinaryOperator::Cross),
            Some(vec3t),
        ),
        picker_entry(
            C::Vector,
            "Dot product",
            "dot",
            ExprNode::Binary(BinaryOperator::Dot),
            Some(f32t),
        ),
        picker_entry(
            C::Vector,
            "Distance",
            "dist",
            ExprNode::Binary(BinaryOperator::Distance),
            Some(f32t),
        ),
        picker_entry(
            C::Vector,
            "Length",
            "magnitude norm",
            ExprNode::Unary(UnaryOperator::Length),
            Some(f32t),
        ),
        picker_entry(
            C::Vector,
            "Normalize",
            "unit direction",
            ExprNode::Unary(UnaryOperator::Normalize),
            None,
        ),
        // Interpolation.
        picker_entry(
            C::Interp,
            "Mix",
            "lerp linear interpolate blend",
            ExprNode::Ternary(TernaryOperator::Mix),
            None,
        ),
        picker_entry(
            C::Interp,
            "Clamp",
            "limit bound",
            ExprNode::Ternary(TernaryOperator::Clamp),
            None,
        ),
        picker_entry(
            C::Interp,
            "Smoothstep",
            "smooth interpolate ease",
            ExprNode::Ternary(TernaryOperator::SmoothStep),
            None,
        ),
        // Comparison.
        picker_entry(
            C::Comparison,
            "Greater than",
            "> gt greater",
            ExprNode::Binary(BinaryOperator::GreaterThan),
            Some(boolt),
        ),
        picker_entry(
            C::Comparison,
            "Greater or equal",
            ">= gte",
            ExprNode::Binary(BinaryOperator::GreaterThanOrEqual),
            Some(boolt),
        ),
        picker_entry(
            C::Comparison,
            "Less than",
            "< lt less",
            ExprNode::Binary(BinaryOperator::LessThan),
            Some(boolt),
        ),
        picker_entry(
            C::Comparison,
            "Less or equal",
            "<= lte",
            ExprNode::Binary(BinaryOperator::LessThanOrEqual),
            Some(boolt),
        ),
        // Logic.
        picker_entry(
            C::Logic,
            "All",
            "and reduce true",
            ExprNode::Unary(UnaryOperator::All),
            Some(boolt),
        ),
        picker_entry(
            C::Logic,
            "Any",
            "or reduce true",
            ExprNode::Unary(UnaryOperator::Any),
            Some(boolt),
        ),
        // Random.
        picker_entry(
            C::Random,
            "Uniform random",
            "random rand uniform range",
            ExprNode::Binary(BinaryOperator::UniformRand),
            Some(f32t),
        ),
        picker_entry(
            C::Random,
            "Normal random",
            "random rand gaussian normal",
            ExprNode::Binary(BinaryOperator::NormalRand),
            Some(f32t),
        ),
        // Bit manipulation.
        picker_entry(
            C::Bitwise,
            "Pack4x8 snorm",
            "pack snorm",
            ExprNode::Unary(UnaryOperator::Pack4x8snorm),
            Some(u32t),
        ),
        picker_entry(
            C::Bitwise,
            "Pack4x8 unorm",
            "pack unorm",
            ExprNode::Unary(UnaryOperator::Pack4x8unorm),
            Some(u32t),
        ),
        picker_entry(
            C::Bitwise,
            "Unpack4x8 snorm",
            "unpack snorm",
            ExprNode::Unary(UnaryOperator::Unpack4x8snorm),
            Some(vec4t),
        ),
        picker_entry(
            C::Bitwise,
            "Unpack4x8 unorm",
            "unpack unorm",
            ExprNode::Unary(UnaryOperator::Unpack4x8unorm),
            Some(vec4t),
        ),
    ];

    // Built-in source values.
    let builtins = [
        ("Time", "time elapsed", BuiltInOperator::Time),
        ("Delta time", "dt frame", BuiltInOperator::DeltaTime),
        ("Virtual time", "virtual", BuiltInOperator::VirtualTime),
        ("Real time", "real wall", BuiltInOperator::RealTime),
        (
            "Alpha cutoff",
            "alpha cutoff threshold",
            BuiltInOperator::AlphaCutoff,
        ),
    ];
    for (label, syn, op) in builtins {
        v.push(PickerNode {
            category: C::BuiltIn,
            search: format!("{label} {syn}").to_lowercase(),
            label: std::borrow::Cow::Borrowed(label),
            kind: add_expr(ExprNode::BuiltIn(op)),
            accepts_input: false,
            has_image_input: false,
            output_type: Some(PortType::Value(op.value_type())),
            is_exposed_property: false,
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
            has_image_input: false,
            output_type: Some(PortType::Value(attr.value_type())),
            is_exposed_property: false,
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
            has_image_input: false,
            output_type: Some(PortType::Value(prop.default.value_type())),
            is_exposed_property: prop.exposed,
        });
    }

    // Textures. The image node sources the `Image` pseudo-type; the sampler
    // reads a color from it.
    v.push(PickerNode {
        category: C::Texture,
        search: "image texture slot".to_string(),
        label: std::borrow::Cow::Borrowed("Image"),
        kind: EditKind::AddImageNode,
        accepts_input: false,
        has_image_input: false,
        output_type: Some(PortType::Image),
        is_exposed_property: false,
    });
    v.push(picker_entry(
        C::Texture,
        "Sample Texture",
        "texture sample read color",
        ExprNode::TextureSample,
        Some(vec4t),
    ));
    v.push(PickerNode {
        category: C::Texture,
        search: "select image index switch choose pick".to_string(),
        label: std::borrow::Cow::Borrowed("Select Image"),
        kind: add_expr(ExprNode::SelectImage { count: 1 }),
        accepts_input: true,
        has_image_input: true,
        output_type: Some(PortType::Image),
        is_exposed_property: false,
    });

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

/// Render the rich create-node picker.
///
/// A search box, optional type-relaxation toggles (when opened from a link),
/// and the filtered catalog laid out in category columns that wrap sideways.
/// Returns the chosen creation [`EditKind`].
fn picker_body(
    ui: &mut egui::Ui,
    catalog: &[PickerNode],
    link: Option<LinkSource>,
    pin_type: Option<PortType>,
    target_reaches_render: bool,
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
            // An exposed property can't feed the render stage; hide it when the
            // dangling input pin reaches render.
            if target_reaches_render && n.is_exposed_property {
                return false;
            }
            // The image pseudo-type connects only to image ports. Gate both
            // directions on it before any value-type relaxation: an operator
            // never produces or consumes an image, so an image/value mismatch
            // is refused regardless of the relaxation toggles.
            match (pin_type, filter) {
                (Some(PortType::Image), MenuFilter::Consumer) => {
                    if !n.has_image_input {
                        return false;
                    }
                }
                (Some(PortType::Image), MenuFilter::Producer) => {
                    if n.output_type != Some(PortType::Image) {
                        return false;
                    }
                }
                (Some(PortType::Value(_)), MenuFilter::Producer) => {
                    if n.output_type == Some(PortType::Image) {
                        return false;
                    }
                }
                _ => {}
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
        let items: Vec<&PickerNode> = visible
            .iter()
            .copied()
            .filter(|n| n.category == cat)
            .collect();
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

/// Build an [`EditKind::AddExprNode`] for `expr`.
///
/// Seeds each operand input port with a neutral default so the node bakes once
/// connected: a value port gets a scalar, the sampler's image port an unbound
/// image binding. A `SelectImage` node's image inputs are link-only and carry
/// no default, so only its `index` selector is seeded.
fn add_expr(expr: ExprNode) -> EditKind {
    let inputs = expr_input_ports(&expr)
        .iter()
        .filter(|name| {
            !matches!(expr, ExprNode::SelectImage { .. }) || !is_select_image_input(name)
        })
        .map(|name| {
            // The sampler's `image` port carries an image binding; its
            // `coordinates` port a `vec2`; the image selector's `index` a `u32`;
            // everything else a scalar.
            let default: InputDefault = match (&expr, *name) {
                (ExprNode::TextureSample, "image") => ImageBinding::Unbound.into(),
                (ExprNode::TextureSample, "coordinates") => {
                    Value::from(bevy::math::Vec2::ZERO).into()
                }
                (ExprNode::SelectImage { .. }, "index") => Value::from(0u32).into(),
                _ => Value::from(0.0f32).into(),
            };
            InputSlot {
                name: SharedStr::from(*name),
                default,
            }
        })
        .collect();
    EditKind::AddExprNode { expr, inputs }
}
