//! Editor-level validity rules for emitter-graph edits.
//!
//! These encode `bevy_hanabi` runtime constraints that the [`EmitterGraph`]
//! model itself can legally represent but that produce a broken emitter, so the
//! editor rejects them at edit time rather than waiting for the bake. They live
//! in the UI layer because they exist to gate *interactions* (a dragged link, a
//! create-node menu entry), not to describe the graph data.
//!
//! ## Properties in the render context
//!
//! `bevy_hanabi` 0.18 binds module properties only in the init/update compute
//! shaders; the render shader has none. An
//! *exposed* property that reaches a render modifier therefore bakes to an
//! `Expr::Property` the render shader can't resolve, and the emitter silently
//! stops rendering. (Edit-only properties are inlined to literals at bake, so
//! they're render-safe.) [`link_routes_property_to_render`] rejects a dragged
//! link that would create this, and [`node_reaches_render`] lets the
//! create-node menu hide exposed property producers when the dangling input pin
//! feeds render.
//!
//! ## Document topology: source links and event links
//!
//! Whether connecting a source to an emitter ([`source_link_is_valid`]) or a
//! spawn-event node to a GPU source ([`event_link_is_valid`]) is legal depends
//! on the *whole* [`EffectGraph`] — same-parent fan-in, no self-parent, no
//! cycles, GPU source only, spawn-event nodes placed in Update — exactly the
//! rules [`crate::effect_graph::validation::validate_topology`] already checks
//! independent of baking. Rather than duplicate that logic, both functions
//! speculatively apply the candidate link to a clone of the document and reuse
//! `validate_topology` to decide: a new topology error blocks the drag, one
//! already present is pre-existing and not this link's fault. The one
//! exception is the single-child-per-parent restriction, a *temporary* bake
//! limitation the model itself is meant to outlive (see
//! `validate_topology`'s doc comment) — the editor still allows full
//! model-level fan-out, so that specific error is never treated as a rejection
//! here.

use std::collections::HashSet;

#[cfg(test)]
use bevy::reflect::TypePath;

use crate::{
    document::ModifierGroup,
    effect_graph::{
        model::{
            EffectGraph, EmitterGraph, EmitterId, EventLink, ExprNode, NodeId, NodePayload,
            SourceId, SourceLink,
        },
        validation::validate_topology,
    },
};

/// Substring of the temporary single-child-per-parent [`TopologyError`]
/// message (see this module's doc comment): never treated as a drag-time
/// rejection.
///
/// [`TopologyError`]: crate::effect_graph::validation::TopologyError
const SINGLE_CHILD_RESTRICTION_MARKER: &str =
    "only a single child per parent is currently supported";

/// Incomplete GPU-source wiring is expected while either link is being drawn.
const UNCONNECTED_GPU_SOURCE_MARKER: &str = "does not drive an emitter via a source link";

/// Whether connecting `source` to drive `emitter` is legal, speculatively.
///
/// `Err` carries a short human-readable reason (e.g. a cycle, or an emitter
/// receiving its own spawn output). Connecting *displaces* whichever links
/// already used either endpoint (see
/// [`crate::effect_graph::edit::set_source_link`]) rather than being refused
/// for that alone, so this only rejects a link that would introduce a new
/// topology problem beyond the ones displacement resolves.
pub fn source_link_is_valid(
    effect_graph: &EffectGraph,
    source: SourceId,
    emitter: EmitterId,
) -> Result<(), String> {
    validate_speculative(effect_graph, |v| {
        v.source_links
            .retain(|l| l.source != source && l.emitter != emitter);
        v.source_links.push(SourceLink { source, emitter });
    })
}

/// Whether connecting spawn-event `node` to GPU source `target` is legal.
///
/// `Err` carries a short human-readable reason (e.g. `target` isn't a GPU
/// source, `node` isn't an Update-stack `EmitSpawnEventModifier`, mixed
/// parents, or a cycle). Unlike a source link, an event link never displaces
/// anything (a GPU source's event input accepts multiple links), so any
/// rejection here is this link's own fault.
pub fn event_link_is_valid(
    effect_graph: &EffectGraph,
    node: NodeId,
    target: SourceId,
) -> Result<(), String> {
    validate_speculative(effect_graph, |v| {
        v.event_links.push(EventLink { node, target });
    })
}

/// Apply `mutate` to a clone of `effect_graph` and report the first *new*
/// topology error it introduces, ignoring errors that represent an expected
/// intermediate wiring state.
fn validate_speculative(
    effect_graph: &EffectGraph,
    mutate: impl FnOnce(&mut EffectGraph),
) -> Result<(), String> {
    let before = validate_topology(effect_graph);
    let mut hypothetical = effect_graph.clone();
    mutate(&mut hypothetical);
    validate_topology(&hypothetical)
        .into_iter()
        .find(|error| {
            !before.contains(error)
                && !error.message.contains(SINGLE_CHILD_RESTRICTION_MARKER)
                && !error.message.contains(UNCONNECTED_GPU_SOURCE_MARKER)
        })
        .map_or(Ok(()), |error| Err(error.message))
}

/// Whether linking `from → to` would carry an exposed property into render.
///
/// Hanabi can't bind an *exposed* property in the render context, so the editor
/// refuses such a link the same way it refuses an incompatible port type.
///
/// Evaluated against the *current* graph (the proposed link is not yet
/// present): the link feeds `from → to`, so it changes neither `from`'s
/// upstream cone nor `to`'s downstream cone. It is `true` exactly when `from`
/// already carries an exposed-property value *and* `to` already feeds the
/// render stage.
pub fn link_routes_property_to_render(graph: &EmitterGraph, from: NodeId, to: NodeId) -> bool {
    carries_exposed_property(graph, from) && node_reaches_render(graph, to)
}

/// True if `node` is a render-stack modifier or transitively feeds one.
///
/// Reached through its (existing) output links.
pub fn node_reaches_render(graph: &EmitterGraph, node: NodeId) -> bool {
    let render_members: HashSet<NodeId> = graph
        .stack(ModifierGroup::Render)
        .map(|s| s.members.iter().copied().collect())
        .unwrap_or_default();
    let mut stack = vec![node];
    let mut seen = HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if render_members.contains(&n) {
            return true;
        }
        for link in &graph.links {
            if link.from.node == n {
                stack.push(link.to.node);
            }
        }
    }
    false
}

/// True if `node` is or transitively depends on an exposed property.
///
/// Reached through its (existing) input links. Such a value cannot legally
/// enter the render context.
fn carries_exposed_property(graph: &EmitterGraph, node: NodeId) -> bool {
    let mut stack = vec![node];
    let mut seen = HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if is_exposed_property(graph, n) {
            return true;
        }
        for link in &graph.links {
            if link.to.node == n {
                stack.push(link.from.node);
            }
        }
    }
    false
}

/// True for an `ExprNode::Property` node referencing an exposed property.
///
/// Edit-only property references are inlined to literals at bake time, so they
/// are render-safe and excluded.
fn is_exposed_property(graph: &EmitterGraph, node: NodeId) -> bool {
    match graph.node(node).map(|n| &n.payload) {
        Some(NodePayload::Expr(ExprNode::Property(id))) => {
            graph.property(*id).is_some_and(|p| p.exposed)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use bevy_hanabi::{Value, graph::expr::BinaryOperator};

    use super::*;
    use crate::effect_graph::model::{
        GraphLink, GraphNode, GraphStack, InputSlot, ModifierNodeData, PortRef, PropertyDef,
        SourceContext, SourceKind, StackId,
    };

    /// Test-only id counter mirroring [`EffectGraph::next_id`], since these
    /// fixtures build a bare [`EmitterGraph`] by hand rather than going
    /// through the effect-level allocator.
    fn alloc(counter: &mut u32) -> u32 {
        *counter += 1;
        *counter
    }

    fn modifier_node(graph: &mut EmitterGraph, counter: &mut u32, group: ModifierGroup) -> NodeId {
        let id = NodeId::new(alloc(counter)).unwrap();
        graph.nodes.push(GraphNode {
            id,
            payload: NodePayload::Modifier(ModifierNodeData::Known {
                type_path: "test::Modifier".into(),
                config: Default::default(),
            }),
            inputs: vec![InputSlot {
                name: "in".into(),
                default: Value::from(0.0f32).into(),
            }],
        });
        let stack = StackId::new(alloc(counter)).unwrap();
        graph.stacks.push(GraphStack {
            id: stack,
            group,
            members: vec![id],
        });
        id
    }

    fn property_node(graph: &mut EmitterGraph, counter: &mut u32, exposed: bool) -> NodeId {
        let pid = crate::effect_graph::model::PropertyId::new(alloc(counter)).unwrap();
        graph.properties.push(PropertyDef {
            id: pid,
            name: "p".into(),
            default: Value::from(1.0f32),
            exposed,
        });
        let id = NodeId::new(alloc(counter)).unwrap();
        graph.nodes.push(GraphNode {
            id,
            payload: NodePayload::Expr(ExprNode::Property(pid)),
            inputs: vec![],
        });
        id
    }

    fn link(graph: &mut EmitterGraph, from: NodeId, to: NodeId, to_port: &str) {
        graph.links.push(GraphLink {
            from: PortRef {
                node: from,
                port: "out".into(),
            },
            to: PortRef {
                node: to,
                port: to_port.into(),
            },
        });
    }

    #[test]
    fn exposed_property_directly_into_render_is_rejected() {
        let mut counter = 0;
        let mut g = EmitterGraph::empty(EmitterId::new(alloc(&mut counter)).unwrap());
        let prop = property_node(&mut g, &mut counter, true);
        let render = modifier_node(&mut g, &mut counter, ModifierGroup::Render);
        assert!(link_routes_property_to_render(&g, prop, render));
    }

    #[test]
    fn edit_only_property_into_render_is_allowed() {
        let mut counter = 0;
        let mut g = EmitterGraph::empty(EmitterId::new(alloc(&mut counter)).unwrap());
        let prop = property_node(&mut g, &mut counter, false);
        let render = modifier_node(&mut g, &mut counter, ModifierGroup::Render);
        // Edit-only properties inline to literals at bake, so they're render-safe.
        assert!(!link_routes_property_to_render(&g, prop, render));
    }

    #[test]
    fn exposed_property_into_init_is_allowed() {
        let mut counter = 0;
        let mut g = EmitterGraph::empty(EmitterId::new(alloc(&mut counter)).unwrap());
        let prop = property_node(&mut g, &mut counter, true);
        // A render stack exists but isn't on the path.
        modifier_node(&mut g, &mut counter, ModifierGroup::Render);
        let init = modifier_node(&mut g, &mut counter, ModifierGroup::Init);
        assert!(!link_routes_property_to_render(&g, prop, init));
    }

    #[test]
    fn exposed_property_reaches_render_transitively() {
        let mut counter = 0;
        let mut g = EmitterGraph::empty(EmitterId::new(alloc(&mut counter)).unwrap());
        let prop = property_node(&mut g, &mut counter, true);
        let render = modifier_node(&mut g, &mut counter, ModifierGroup::Render);

        // An intermediate expression node fed by the exposed property.
        let mid = NodeId::new(alloc(&mut counter)).unwrap();
        g.nodes.push(GraphNode {
            id: mid,
            payload: NodePayload::Expr(ExprNode::Binary(BinaryOperator::Add)),
            inputs: vec![
                InputSlot {
                    name: "lhs".into(),
                    default: Value::from(0.0f32).into(),
                },
                InputSlot {
                    name: "rhs".into(),
                    default: Value::from(0.0f32).into(),
                },
            ],
        });
        link(&mut g, prop, mid, "lhs");

        // Proposed link: the intermediate node into the render modifier. The
        // property taint reaches render through `mid`.
        assert!(link_routes_property_to_render(&g, mid, render));

        // Equivalently, proposing the property into `mid` (which already feeds
        // render) is also rejected.
        link(&mut g, mid, render, "in");
        assert!(link_routes_property_to_render(&g, prop, mid));
    }

    /// Minimal two-emitter [`EffectGraph`] fixture for the document-topology
    /// tests: `driver` has an Update-stack `EmitSpawnEventModifier`-shaped
    /// node (by type path, since these fixtures don't depend on
    /// `bevy_hanabi`'s modifier types) and `driven` is otherwise unlinked.
    /// Neither emitter has a source link yet.
    fn two_emitter_effect() -> (EffectGraph, u32, EmitterId, NodeId, EmitterId) {
        let mut counter = 0;
        let driver_id = EmitterId::new(alloc(&mut counter)).unwrap();
        let mut driver = EmitterGraph::empty(driver_id);
        let event_node = NodeId::new(alloc(&mut counter)).unwrap();
        driver.nodes.push(GraphNode {
            id: event_node,
            payload: NodePayload::Modifier(ModifierNodeData::Known {
                type_path: bevy_hanabi::EmitSpawnEventModifier::type_path().into(),
                config: Default::default(),
            }),
            inputs: vec![],
        });
        let update_stack = StackId::new(alloc(&mut counter)).unwrap();
        driver.stacks.push(GraphStack {
            id: update_stack,
            group: ModifierGroup::Update,
            members: vec![event_node],
        });

        let driven_id = EmitterId::new(alloc(&mut counter)).unwrap();
        let driven = EmitterGraph::empty(driven_id);

        let mut effect_graph = EffectGraph::empty();
        effect_graph.emitters.push(driver);
        effect_graph.emitters.push(driven);
        effect_graph.next_id = counter + 1;
        (effect_graph, counter, driver_id, event_node, driven_id)
    }

    #[test]
    fn event_link_from_update_spawn_event_to_gpu_source_is_valid() {
        let (mut effect_graph, mut counter, .., event_node, _driven) = two_emitter_effect();
        let source = SourceId::new(alloc(&mut counter)).unwrap();
        effect_graph.sources.push(SourceContext {
            id: source,
            kind: SourceKind::GpuEvent,
        });
        assert!(event_link_is_valid(&effect_graph, event_node, source).is_ok());
    }

    #[test]
    fn event_link_to_cpu_spawner_is_rejected() {
        let (mut effect_graph, mut counter, .., event_node, _driven) = two_emitter_effect();
        let source = SourceId::new(alloc(&mut counter)).unwrap();
        effect_graph.sources.push(SourceContext {
            id: source,
            kind: SourceKind::CpuSpawner {
                settings: Default::default(),
            },
        });
        assert!(event_link_is_valid(&effect_graph, event_node, source).is_err());
    }

    #[test]
    fn event_link_from_non_spawn_event_node_is_rejected() {
        let (mut effect_graph, mut counter, driver_id, ..) = two_emitter_effect();
        // A node that isn't the fixture's known `EmitSpawnEventModifier`.
        let other_node = NodeId::new(alloc(&mut counter)).unwrap();
        effect_graph
            .emitter_mut(driver_id)
            .unwrap()
            .nodes
            .push(GraphNode {
                id: other_node,
                payload: NodePayload::Modifier(ModifierNodeData::Known {
                    type_path: "test::NotASpawnEvent".into(),
                    config: Default::default(),
                }),
                inputs: vec![],
            });
        let source = SourceId::new(alloc(&mut counter)).unwrap();
        effect_graph.sources.push(SourceContext {
            id: source,
            kind: SourceKind::GpuEvent,
        });
        assert!(event_link_is_valid(&effect_graph, other_node, source).is_err());
    }

    #[test]
    fn event_link_mixing_parents_is_rejected() {
        let (mut effect_graph, mut counter, .., event_node, driven_id) = two_emitter_effect();
        let source = SourceId::new(alloc(&mut counter)).unwrap();
        effect_graph.sources.push(SourceContext {
            id: source,
            kind: SourceKind::GpuEvent,
        });
        effect_graph.event_links.push(EventLink {
            node: event_node,
            target: source,
        });

        // A second spawn-event node, from the *other* emitter, targeting the same
        // source: mixes parents.
        let other_event_node = NodeId::new(alloc(&mut counter)).unwrap();
        let driven = effect_graph.emitter_mut(driven_id).unwrap();
        driven.nodes.push(GraphNode {
            id: other_event_node,
            payload: NodePayload::Modifier(ModifierNodeData::Known {
                type_path: bevy_hanabi::EmitSpawnEventModifier::type_path().into(),
                config: Default::default(),
            }),
            inputs: vec![],
        });
        let update_stack = StackId::new(alloc(&mut counter)).unwrap();
        driven.stacks.push(GraphStack {
            id: update_stack,
            group: ModifierGroup::Update,
            members: vec![other_event_node],
        });

        assert!(event_link_is_valid(&effect_graph, other_event_node, source).is_err());
    }

    #[test]
    fn event_link_creating_a_parent_cycle_is_rejected() {
        let (mut effect_graph, mut counter, driver_id, event_node, driven_id) =
            two_emitter_effect();
        let source = SourceId::new(alloc(&mut counter)).unwrap();
        effect_graph.sources.push(SourceContext {
            id: source,
            kind: SourceKind::GpuEvent,
        });
        // `driver` already spawns `driven` via `source`.
        effect_graph.event_links.push(EventLink {
            node: event_node,
            target: source,
        });
        effect_graph.source_links.push(SourceLink {
            source,
            emitter: driven_id,
        });

        // `driven` also has a spawn-event node and a GPU source of its own;
        // linking it back to a source that (transitively) drives `driver`
        // would close the loop.
        let back_event_node = NodeId::new(alloc(&mut counter)).unwrap();
        let driven = effect_graph.emitter_mut(driven_id).unwrap();
        driven.nodes.push(GraphNode {
            id: back_event_node,
            payload: NodePayload::Modifier(ModifierNodeData::Known {
                type_path: bevy_hanabi::EmitSpawnEventModifier::type_path().into(),
                config: Default::default(),
            }),
            inputs: vec![],
        });
        let update_stack = StackId::new(alloc(&mut counter)).unwrap();
        driven.stacks.push(GraphStack {
            id: update_stack,
            group: ModifierGroup::Update,
            members: vec![back_event_node],
        });
        let back_source = SourceId::new(alloc(&mut counter)).unwrap();
        effect_graph.sources.push(SourceContext {
            id: back_source,
            kind: SourceKind::GpuEvent,
        });
        effect_graph.source_links.push(SourceLink {
            source: back_source,
            emitter: driver_id,
        });

        assert!(event_link_is_valid(&effect_graph, back_event_node, back_source).is_err());
    }

    #[test]
    fn second_child_source_from_same_parent_is_allowed_model_level() {
        // Full model-level fan-out is allowed; only the bake enforces a
        // single connected child per parent (see this module's doc comment).
        let (mut effect_graph, mut counter, .., event_node, driven) = two_emitter_effect();
        let first = SourceId::new(alloc(&mut counter)).unwrap();
        effect_graph.sources.push(SourceContext {
            id: first,
            kind: SourceKind::GpuEvent,
        });
        effect_graph.event_links.push(EventLink {
            node: event_node,
            target: first,
        });
        effect_graph.source_links.push(SourceLink {
            source: first,
            emitter: driven,
        });

        let second = SourceId::new(alloc(&mut counter)).unwrap();
        effect_graph.sources.push(SourceContext {
            id: second,
            kind: SourceKind::GpuEvent,
        });
        assert!(event_link_is_valid(&effect_graph, event_node, second).is_ok());
    }

    #[test]
    fn source_link_displacing_an_existing_link_is_allowed() {
        let (mut effect_graph, mut counter, _driver_id, _event_node, driven_id) =
            two_emitter_effect();
        let source = SourceId::new(alloc(&mut counter)).unwrap();
        effect_graph.sources.push(SourceContext {
            id: source,
            kind: SourceKind::CpuSpawner {
                settings: Default::default(),
            },
        });
        effect_graph.source_links.push(SourceLink {
            source,
            emitter: driven_id,
        });

        // Re-pointing the same source at the same emitter is a no-op
        // displacement, not a new topology problem.
        assert!(source_link_is_valid(&effect_graph, source, driven_id).is_ok());
    }
}
