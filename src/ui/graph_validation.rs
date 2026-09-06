//! Editor-level validity rules for emitter-graph edits.
//!
//! These encode `bevy_hanabi` runtime constraints that the [`EmitterGraph`]
//! model itself can legally represent but that produce a broken emitter, so the
//! editor rejects them at edit time rather than waiting for the bake. They live
//! in the UI layer because they exist to gate *interactions* (a dragged link, a
//! create-node menu entry), not to describe the graph data.
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

#[cfg(test)]
use bevy::reflect::TypePath;

use crate::effect_graph::{
    model::{EffectGraph, EmitterId, EventLink, NodeId, SourceId, SourceLink},
    validation::validate_topology,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        document::ModifierGroup,
        effect_graph::model::{
            EmitterGraph, GraphNode, GraphStack, ModifierNodeData, NodePayload, SourceContext,
            SourceKind, StackId,
        },
    };

    /// Test-only id counter mirroring [`EffectGraph::next_id`], since these
    /// fixtures build a bare [`EmitterGraph`] by hand rather than going
    /// through the effect-level allocator.
    fn alloc(counter: &mut u32) -> u32 {
        *counter += 1;
        *counter
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
