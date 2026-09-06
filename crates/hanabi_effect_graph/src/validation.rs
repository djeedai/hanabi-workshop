//! Effect-graph validation checks.
//!
//! Home for analyses that flag suspicious-but-legal emitters in the UI (e.g.
//! the per-node Graph warning badge). Currently a single check:
//! shadowed-modifier detection. More checks (orphaned nodes, unused
//! properties, missing required attributes, …) belong here as they land.
//!
//! ## Shadowed modifiers
//!
//! A modifier is *shadowed* when every particle attribute it fully overwrites
//! is also overwritten by some later modifier in the same group, making its
//! writes dead. Built on the [`ModifierOverwrites`] type-data callback, which
//! gives the per-instance set of attributes a modifier *fully assigns to*
//! (distinct from `Modifier::attributes()`, which mixes reads and writes in
//! upstream bevy_hanabi).
//!
//! Only meaningful within a single group (Init / Update): each runs
//! strictly in order, with subsequent overwrites discarding any previous
//! per-particle value. The Render group is skipped — render modifiers
//! write vertex-shader variables rather than particle attributes.
//!
//! ## Effect topology validation
//!
//! [`validate_topology`] checks the effect-level wiring of an [`EffectGraph`]:
//! which spawn source drives which emitter, and which `EmitSpawnEventModifier`
//! nodes feed which GPU source context. This is structural, edit-time
//! validation distinct from [`shadowed_modifiers`] above — it never touches a
//! baked [`EffectAsset`] — and is meant to run *before* [`bake_effect`], since
//! a topology error (an ambiguous parent, a cycle, a GPU source with no
//! emitter, …) would make the derived per-emitter spawner/channel baking
//! inputs meaningless.
//!
//! [`EffectGraph`]: crate::model::EffectGraph
//! [`bake_effect`]: crate::bake::bake_effect

use std::collections::HashMap;

use bevy::reflect::{TypePath, TypeRegistry};
use bevy_hanabi::{Attribute, EffectAsset, EmitSpawnEventModifier, InheritAttributeModifier};

use crate::{
    ModifierGroup,
    model::{
        EffectGraph, EmitterId, ExprNode, ModifierNodeData, NodeId, NodePayload, SourceId,
        SourceKind,
    },
    modifier_registry::ModifierOverwrites,
};

/// Find every fully-shadowed modifier and what shadows it.
///
/// Returns, for each shadowed modifier, the `(attribute, shadower_idx)` pairs
/// that explain why it has no observable effect, keyed by `(group, idx)` where
/// `idx` is the modifier's position within its group's stack. Only Init and
/// Update groups are analysed; Render is never included.
pub fn shadowed_modifiers(
    asset: &EffectAsset,
    registry: &TypeRegistry,
) -> HashMap<(ModifierGroup, usize), Vec<(Attribute, usize)>> {
    let mut out = HashMap::new();
    analyze_group(asset, ModifierGroup::Init, registry, &mut out);
    analyze_group(asset, ModifierGroup::Update, registry, &mut out);
    out
}

fn analyze_group(
    asset: &EffectAsset,
    group: ModifierGroup,
    registry: &TypeRegistry,
    out: &mut HashMap<(ModifierGroup, usize), Vec<(Attribute, usize)>>,
) {
    // Per-modifier set of attributes the modifier fully overwrites.
    let overwrites: Vec<Vec<Attribute>> = match group {
        ModifierGroup::Init => asset
            .init_modifiers()
            .map(|m| overwrites_for(m.as_reflect(), registry))
            .collect(),
        ModifierGroup::Update => asset
            .update_modifiers()
            .map(|m| overwrites_for(m.as_reflect(), registry))
            .collect(),
        ModifierGroup::Render => return,
    };

    // For each modifier i, look forward for the *earliest* later j that
    // overwrites each attribute in i's set. If every attribute is covered, i
    // is fully shadowed.
    for (i, w_i) in overwrites.iter().enumerate() {
        if w_i.is_empty() {
            continue;
        }
        let mut hits: Vec<(Attribute, usize)> = Vec::with_capacity(w_i.len());
        for &attr in w_i {
            let earliest = overwrites
                .iter()
                .enumerate()
                .skip(i + 1)
                .find(|(_, w_j)| w_j.contains(&attr))
                .map(|(j, _)| j);
            match earliest {
                Some(j) => hits.push((attr, j)),
                None => {
                    // At least one produced attribute survives — not shadowed.
                    hits.clear();
                    break;
                }
            }
        }
        if !hits.is_empty() {
            out.insert((group, i), hits);
        }
    }
}

/// Per-instance overwrite set lookup.
///
/// Falls back to empty if the modifier's type isn't registered or carries no
/// [`ModifierOverwrites`] data (e.g. a read-modify-write or third-party
/// modifier) — we conservatively skip shadow analysis rather than risk a false
/// positive.
fn overwrites_for(m: &dyn bevy::reflect::Reflect, registry: &TypeRegistry) -> Vec<Attribute> {
    let Some(reg) = registry.get(std::any::Any::type_id(m)) else {
        return Vec::new();
    };
    let Some(rm) = reg.data::<ModifierOverwrites>() else {
        return Vec::new();
    };
    (rm.overwrites)(m)
}

// ── Effect topology validation
// ─────────────────────────────────────────────────

/// What a [`TopologyError`] is attributed to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologySubject {
    /// A specific emitter pipeline (e.g. a missing or duplicate source link).
    Emitter(EmitterId),
    /// A specific spawn source context (e.g. a GPU source with no emitter).
    Source(SourceId),
    /// A specific graph node (e.g. an emitter misplaced outside Update).
    Node(NodeId),
    /// The effect graph as a whole, with no single element to blame.
    Graph,
}

/// A structural topology problem, attributed to the element to blame.
///
/// [`validate_topology`] collects every error it can rather than stopping at
/// the first, so the caller (typically the editor) can surface all of them at
/// once.
#[derive(Debug, Clone, PartialEq)]
pub struct TopologyError {
    pub subject: TopologySubject,
    pub message: String,
}

impl TopologyError {
    fn emitter(id: EmitterId, message: impl Into<String>) -> Self {
        Self {
            subject: TopologySubject::Emitter(id),
            message: message.into(),
        }
    }

    fn source(id: SourceId, message: impl Into<String>) -> Self {
        Self {
            subject: TopologySubject::Source(id),
            message: message.into(),
        }
    }

    fn node(id: NodeId, message: impl Into<String>) -> Self {
        Self {
            subject: TopologySubject::Node(id),
            message: message.into(),
        }
    }
}

/// Validate the inter-emitter topology of an [`EffectGraph`].
///
/// Checks, independent of baking:
///
/// - every emitter has exactly one linked spawn source, and every source drives
///   at most one emitter (no reuse);
/// - every GPU source context has at least one linked event node and drives an
///   emitter via a source link, and every event link targets a GPU (not CPU)
///   source;
/// - every event link's node exists, is a known `EmitSpawnEventModifier`, and
///   sits in its owning emitter's Update stack;
/// - every GPU source's event nodes share a single parent emitter (no mixed
///   parents), and no emitter is its own ancestor (no cycles);
/// - a CPU-rooted emitter (no parent) never reads `ParentAttribute` or carries
///   an `InheritAttributeModifier`, since neither has meaning without a parent;
/// - value links stay within their owning emitter;
/// - (temporary) no parent emitter drives more than one connected GPU child —
///   the runtime supports a single child emitter per parent for now.
///
/// Every check runs independently and contributes to the same error list;
/// this function never panics, even on a badly malformed, mid-edit graph.
pub fn validate_topology(effect_graph: &EffectGraph) -> Vec<TopologyError> {
    let mut errors = Vec::new();
    check_orphan_references(effect_graph, &mut errors);
    check_source_link_cardinality(effect_graph, &mut errors);
    check_gpu_sources_and_event_links(effect_graph, &mut errors);
    check_emitter_placement(effect_graph, &mut errors);
    check_mixed_parents_and_cycles(effect_graph, &mut errors);
    check_cpu_root_parent_usage(effect_graph, &mut errors);
    check_cross_emitter_value_links(effect_graph, &mut errors);
    check_single_child_restriction(effect_graph, &mut errors);
    errors
}

/// A link or reference naming an id that does not exist in this document.
fn check_orphan_references(effect_graph: &EffectGraph, errors: &mut Vec<TopologyError>) {
    for link in &effect_graph.source_links {
        if effect_graph.source(link.source).is_none() {
            errors.push(TopologyError::source(
                link.source,
                "source link references a source context that does not exist",
            ));
        }
        if effect_graph.emitter(link.emitter).is_none() {
            errors.push(TopologyError::emitter(
                link.emitter,
                "source link references an emitter that does not exist",
            ));
        }
    }
    for link in &effect_graph.event_links {
        if effect_graph.source(link.target).is_none() {
            errors.push(TopologyError::node(
                link.node,
                "event link targets a source context that does not exist",
            ));
        }
        if effect_graph.emitter_owning_node(link.node).is_none() {
            errors.push(TopologyError::node(
                link.node,
                "event link references a node that does not exist in any emitter",
            ));
        }
    }
}

/// Every emitter must have exactly one source link; every source drives at
/// most one emitter.
fn check_source_link_cardinality(effect_graph: &EffectGraph, errors: &mut Vec<TopologyError>) {
    for emitter in &effect_graph.emitters {
        let count = effect_graph
            .source_links
            .iter()
            .filter(|l| l.emitter == emitter.id)
            .count();
        match count {
            0 => errors.push(TopologyError::emitter(
                emitter.id,
                "emitter has no linked spawn source",
            )),
            1 => {}
            n => errors.push(TopologyError::emitter(
                emitter.id,
                format!("emitter has {n} linked spawn sources; expected exactly one"),
            )),
        }
    }

    let mut seen: HashMap<SourceId, u32> = HashMap::new();
    for link in &effect_graph.source_links {
        *seen.entry(link.source).or_insert(0) += 1;
    }
    for (source, count) in seen {
        if count > 1 {
            errors.push(TopologyError::source(
                source,
                format!("source drives {count} emitters; a source may drive only one"),
            ));
        }
    }
}

/// Every GPU source needs at least one emitter; every emitted-into GPU source
/// must be connected to an emitter; every event link must target a GPU (not
/// CPU) source.
fn check_gpu_sources_and_event_links(effect_graph: &EffectGraph, errors: &mut Vec<TopologyError>) {
    for source in &effect_graph.sources {
        if !matches!(source.kind, SourceKind::GpuEvent) {
            continue;
        }
        let has_emitters = effect_graph.events_for_source(source.id).next().is_some();
        if !has_emitters {
            errors.push(TopologyError::source(
                source.id,
                "GPU source context has no linked spawn-event node; nothing will spawn into it",
            ));
        } else if effect_graph.emitter_for_source(source.id).is_none() {
            // A spawn-event node feeds this source, but no emitter consumes it: the
            // channel indices `bake::derive_emitter_child_indices` would derive don't
            // correspond to any real child, so a connected sibling could be
            // silently shifted onto the wrong channel. Reject rather than
            // bake a mismatched topology.
            errors.push(TopologyError::source(
                source.id,
                "GPU source context has linked spawn-event nodes but does not drive an emitter \
                 via a source link",
            ));
        }
    }

    for link in &effect_graph.event_links {
        let Some(target) = effect_graph.source(link.target) else {
            continue; // already reported as an orphan reference
        };
        if !matches!(target.kind, SourceKind::GpuEvent) {
            errors.push(TopologyError::node(
                link.node,
                "event link targets a CPU spawner source, not a GPU event source",
            ));
        }
    }
}

/// Every emitter must be a known `EmitSpawnEventModifier` node placed in its
/// owning emitter's Update stack.
fn check_emitter_placement(effect_graph: &EffectGraph, errors: &mut Vec<TopologyError>) {
    for link in &effect_graph.event_links {
        let Some(emitter_id) = effect_graph.emitter_owning_node(link.node) else {
            continue; // already reported as an orphan reference
        };
        let emitter = effect_graph
            .emitter(emitter_id)
            .expect("resolved from this emitter");
        let Some(node) = emitter.node(link.node) else {
            continue;
        };
        let is_known_emitter = matches!(
            &node.payload,
            NodePayload::Modifier(ModifierNodeData::Known { type_path, .. })
                if type_path.as_ref() == EmitSpawnEventModifier::type_path()
        );
        if !is_known_emitter {
            errors.push(TopologyError::node(
                link.node,
                "event link node is not a known EmitSpawnEventModifier",
            ));
        }
    }

    for emitter in &effect_graph.emitters {
        for node in &emitter.nodes {
            let is_known_emitter = matches!(
                &node.payload,
                NodePayload::Modifier(ModifierNodeData::Known { type_path, .. })
                    if type_path.as_ref() == EmitSpawnEventModifier::type_path()
            );
            if !is_known_emitter {
                continue;
            }
            let in_update_stack = emitter
                .stacks
                .iter()
                .any(|s| s.group == ModifierGroup::Update && s.members.contains(&node.id));
            if !in_update_stack {
                errors.push(TopologyError::node(
                    node.id,
                    "EmitSpawnEventModifier must be placed in its emitter's Update stack",
                ));
            }
        }
    }
}

/// A GPU source's emitters must share a single parent emitter, and no emitter
/// may be its own ancestor.
fn check_mixed_parents_and_cycles(effect_graph: &EffectGraph, errors: &mut Vec<TopologyError>) {
    for source in &effect_graph.sources {
        if !matches!(source.kind, SourceKind::GpuEvent) {
            continue;
        }
        let mut parents = effect_graph
            .events_for_source(source.id)
            .filter_map(|n| effect_graph.emitter_owning_node(n))
            .collect::<Vec<_>>();
        parents.sort_by_key(|e| e.get());
        parents.dedup();
        if parents.len() > 1 {
            errors.push(TopologyError::source(
                source.id,
                "GPU source receives event links from more than one parent emitter",
            ));
        }
    }

    for emitter in &effect_graph.emitters {
        let mut seen = vec![emitter.id];
        let mut current = emitter.id;
        // Bounded by the emitter count: a well-formed forest can never chain
        // deeper than that, so overrunning it means we looped back around.
        for _ in 0..=effect_graph.emitters.len() {
            let Some(parent) = effect_graph.parent_emitter(current) else {
                break;
            };
            if seen.contains(&parent) {
                errors.push(TopologyError::emitter(
                    emitter.id,
                    "emitter participates in a parent-child cycle",
                ));
                break;
            }
            seen.push(parent);
            current = parent;
        }
    }
}

/// A CPU-rooted emitter (no parent) never reads a parent particle's state.
fn check_cpu_root_parent_usage(effect_graph: &EffectGraph, errors: &mut Vec<TopologyError>) {
    for emitter in &effect_graph.emitters {
        if effect_graph.parent_emitter(emitter.id).is_some() {
            continue;
        }
        for node in &emitter.nodes {
            match &node.payload {
                NodePayload::Expr(ExprNode::ParentAttribute(_)) => {
                    errors.push(TopologyError::node(
                        node.id,
                        "ParentAttribute has no meaning on a CPU-rooted emitter (no parent)",
                    ));
                }
                NodePayload::Modifier(ModifierNodeData::Known { type_path, .. })
                    if type_path.as_ref() == InheritAttributeModifier::type_path() =>
                {
                    errors.push(TopologyError::node(
                        node.id,
                        "InheritAttributeModifier has no meaning on a CPU-rooted emitter (no \
                         parent)",
                    ));
                }
                _ => {}
            }
        }
    }
}

/// A value link's endpoints must belong to the emitter that owns the link.
fn check_cross_emitter_value_links(effect_graph: &EffectGraph, errors: &mut Vec<TopologyError>) {
    for emitter in &effect_graph.emitters {
        for link in &emitter.links {
            for &node in &[link.from.node, link.to.node] {
                if emitter.node(node).is_some() {
                    continue;
                }
                match effect_graph.emitter_owning_node(node) {
                    Some(other) => errors.push(TopologyError::node(
                        node,
                        format!(
                            "value link in emitter {} crosses into emitter {}",
                            emitter.id.get(),
                            other.get()
                        ),
                    )),
                    None => errors.push(TopologyError::node(
                        node,
                        "value link references a node that does not exist in any emitter",
                    )),
                }
            }
        }
    }
}

/// Temporary restriction: a parent emitter may drive at most one *connected*
/// child GPU source context.
///
/// The runtime supports a single child emitter per parent today (see
/// `EmitSpawnEventModifier::child_index` upstream); lifting this once
/// multi-child parents are supported only requires removing this check, since
/// [`crate::bake::derive_emitter_child_indices`] already assigns distinct
/// channels per target source.
fn check_single_child_restriction(effect_graph: &EffectGraph, errors: &mut Vec<TopologyError>) {
    for emitter in &effect_graph.emitters {
        let mut children: Vec<SourceId> = effect_graph
            .event_links
            .iter()
            .filter(|l| effect_graph.emitter_owning_node(l.node) == Some(emitter.id))
            .map(|l| l.target)
            .filter(|target| effect_graph.emitter_for_source(*target).is_some())
            .collect();
        children.sort_by_key(|s| s.get());
        children.dedup();
        if children.len() > 1 {
            errors.push(TopologyError::emitter(
                emitter.id,
                "emitter has more than one connected child emitter via distinct GPU sources; \
                 only a single child per parent is currently supported",
            ));
        }
    }
}

#[cfg(test)]
mod topology_tests {
    use std::collections::BTreeMap;

    use bevy_hanabi::SpawnerSettings;

    use super::*;
    use crate::model::{
        EmitterGraph, EmitterId, EventLink, GraphLink, GraphNode, GraphStack, PortRef,
        SourceContext, SourceLink, StackId,
    };

    fn cpu_source(id: u32) -> SourceContext {
        SourceContext {
            id: SourceId::new(id).unwrap(),
            kind: SourceKind::CpuSpawner {
                settings: SpawnerSettings::default(),
            },
        }
    }

    fn gpu_source(id: u32) -> SourceContext {
        SourceContext {
            id: SourceId::new(id).unwrap(),
            kind: SourceKind::GpuEvent,
        }
    }

    fn spawn_event_node(id: u32) -> GraphNode {
        GraphNode {
            id: NodeId::new(id).unwrap(),
            payload: NodePayload::Modifier(ModifierNodeData::Known {
                type_path: EmitSpawnEventModifier::type_path().into(),
                config: BTreeMap::new(),
            }),
            inputs: vec![],
        }
    }

    /// A minimal, valid CPU-rooted single emitter with no children.
    fn single_emitter_effect() -> EffectGraph {
        let emitter_id = EmitterId::new(1).unwrap();
        EffectGraph {
            emitters: vec![EmitterGraph::empty(emitter_id)],
            sources: vec![cpu_source(2)],
            source_links: vec![SourceLink {
                source: SourceId::new(2).unwrap(),
                emitter: emitter_id,
            }],
            event_links: vec![],
            next_id: 3,
        }
    }

    #[test]
    fn valid_single_emitter_has_no_errors() {
        assert_eq!(validate_topology(&single_emitter_effect()), vec![]);
    }

    #[test]
    fn missing_source_link_is_reported() {
        let mut effect_graph = single_emitter_effect();
        effect_graph.source_links.clear();
        let errors = validate_topology(&effect_graph);
        assert!(errors.iter().any(|e| {
            e.subject == TopologySubject::Emitter(EmitterId::new(1).unwrap())
                && e.message.contains("no linked spawn source")
        }));
    }

    #[test]
    fn duplicate_source_link_is_reported() {
        let mut effect_graph = single_emitter_effect();
        let extra_source = SourceId::new(10).unwrap();
        effect_graph.sources.push(SourceContext {
            id: extra_source,
            kind: SourceKind::CpuSpawner {
                settings: SpawnerSettings::default(),
            },
        });
        effect_graph.source_links.push(SourceLink {
            source: extra_source,
            emitter: EmitterId::new(1).unwrap(),
        });
        let errors = validate_topology(&effect_graph);
        assert!(errors.iter().any(|e| {
            e.subject == TopologySubject::Emitter(EmitterId::new(1).unwrap())
                && e.message.contains("expected exactly one")
        }));
    }

    #[test]
    fn source_reuse_is_reported() {
        let mut effect_graph = single_emitter_effect();
        let emitter2 = EmitterId::new(20).unwrap();
        effect_graph.emitters.push(EmitterGraph::empty(emitter2));
        effect_graph.source_links.push(SourceLink {
            source: SourceId::new(2).unwrap(),
            emitter: emitter2,
        });
        let errors = validate_topology(&effect_graph);
        assert!(errors.iter().any(|e| {
            e.subject == TopologySubject::Source(SourceId::new(2).unwrap())
                && e.message.contains("drives 2 emitters")
        }));
    }

    /// A supported CPU-root → single GPU-child topology: two spawn-event nodes
    /// in the parent feed the same GPU source that drives the child.
    fn cpu_to_gpu_child_effect() -> (EffectGraph, EmitterId, EmitterId) {
        let parent_id = EmitterId::new(1).unwrap();
        let child_id = EmitterId::new(2).unwrap();
        let event_a = NodeId::new(3).unwrap();
        let event_b = NodeId::new(4).unwrap();
        let update_stack = StackId::new(5).unwrap();
        let cpu_src = SourceId::new(6).unwrap();
        let gpu_src = SourceId::new(7).unwrap();

        let mut parent = EmitterGraph::empty(parent_id);
        parent.nodes.push(spawn_event_node(3));
        parent.nodes.push(spawn_event_node(4));
        parent.stacks.push(GraphStack {
            id: update_stack,
            group: ModifierGroup::Update,
            members: vec![event_a, event_b],
        });

        let effect_graph = EffectGraph {
            emitters: vec![parent, EmitterGraph::empty(child_id)],
            sources: vec![cpu_source(6), gpu_source(7)],
            source_links: vec![
                SourceLink {
                    source: cpu_src,
                    emitter: parent_id,
                },
                SourceLink {
                    source: gpu_src,
                    emitter: child_id,
                },
            ],
            event_links: vec![
                EventLink {
                    node: event_a,
                    target: gpu_src,
                },
                EventLink {
                    node: event_b,
                    target: gpu_src,
                },
            ],
            next_id: 8,
        };
        (effect_graph, parent_id, child_id)
    }

    #[test]
    fn supported_cpu_to_gpu_child_with_multiple_emitters_has_no_errors() {
        let (effect_graph, ..) = cpu_to_gpu_child_effect();
        assert_eq!(validate_topology(&effect_graph), vec![]);
    }

    #[test]
    fn parent_resolves_through_multiple_event_nodes() {
        let (effect_graph, parent_id, child_id) = cpu_to_gpu_child_effect();
        assert_eq!(effect_graph.parent_emitter(child_id), Some(parent_id));
        assert_eq!(effect_graph.parent_emitter(parent_id), None);
    }

    #[test]
    fn gpu_source_without_emitter_is_reported() {
        let (mut effect_graph, ..) = cpu_to_gpu_child_effect();
        effect_graph.event_links.clear();
        let errors = validate_topology(&effect_graph);
        assert!(errors.iter().any(|e| {
            e.subject == TopologySubject::Source(SourceId::new(7).unwrap())
                && e.message.contains("no linked spawn-event node")
        }));
    }

    #[test]
    fn emitter_without_event_link_is_allowed() {
        let mut effect_graph = single_emitter_effect();
        let event_node = NodeId::new(3).unwrap();
        let update_stack = StackId::new(4).unwrap();
        let emitter_graph = &mut effect_graph.emitters[0];
        emitter_graph.nodes.push(spawn_event_node(3));
        emitter_graph.stacks.push(GraphStack {
            id: update_stack,
            group: ModifierGroup::Update,
            members: vec![event_node],
        });
        effect_graph.next_id = 5;

        assert_eq!(validate_topology(&effect_graph), vec![]);
    }

    /// A GPU source with a linked spawn-event node but no [`SourceLink`] to any
    /// emitter must be rejected, not silently tolerated.
    ///
    /// Regression test: an incompletely-authored GPU source in this shape
    /// used to pass validation (the fan-out check already filters to
    /// connected children, so it never flagged an orphan), yet
    /// `bake::derive_emitter_child_indices` counted it as a distinct target
    /// anyway — shifting a genuinely connected sibling's channel index.
    /// Rejecting here ensures this state can never reach `bake_effect`.
    #[test]
    fn orphan_gpu_source_with_spawn_event_is_reported() {
        let (mut effect_graph, parent_id, ..) = cpu_to_gpu_child_effect();
        let orphan_gpu = SourceId::new(200).unwrap();
        let orphan_event = NodeId::new(201).unwrap();
        effect_graph.sources.push(SourceContext {
            id: orphan_gpu,
            kind: SourceKind::GpuEvent,
        });
        // No `SourceLink` for `orphan_gpu`: it has an event node but drives no
        // emitter.
        effect_graph
            .emitter_mut(parent_id)
            .unwrap()
            .nodes
            .push(spawn_event_node(201));
        effect_graph
            .emitter_mut(parent_id)
            .unwrap()
            .stacks
            .iter_mut()
            .find(|s| s.group == ModifierGroup::Update)
            .unwrap()
            .members
            .push(orphan_event);
        effect_graph.event_links.push(EventLink {
            node: orphan_event,
            target: orphan_gpu,
        });

        let errors = validate_topology(&effect_graph);
        assert!(errors.iter().any(|e| {
            e.subject == TopologySubject::Source(orphan_gpu)
                && e.message.contains("does not drive an emitter")
        }));
    }

    #[test]
    fn event_link_to_cpu_source_is_reported() {
        let (mut effect_graph, ..) = cpu_to_gpu_child_effect();
        let cpu_src = SourceId::new(6).unwrap();
        for link in &mut effect_graph.event_links {
            link.target = cpu_src;
        }
        let errors = validate_topology(&effect_graph);
        assert!(errors.iter().any(|e| {
            e.subject == TopologySubject::Node(NodeId::new(3).unwrap())
                && e.message.contains("not a GPU event source")
        }));
    }

    #[test]
    fn spawn_event_node_not_a_known_modifier_is_reported() {
        let (mut effect_graph, parent_id, ..) = cpu_to_gpu_child_effect();
        effect_graph.emitter_mut(parent_id).unwrap().nodes[0] = GraphNode {
            id: NodeId::new(3).unwrap(),
            payload: NodePayload::Modifier(ModifierNodeData::Unknown {
                type_path: "some::Other".into(),
                raw: "()".into(),
            }),
            inputs: vec![],
        };
        let errors = validate_topology(&effect_graph);
        assert!(errors.iter().any(|e| {
            e.subject == TopologySubject::Node(NodeId::new(3).unwrap())
                && e.message.contains("not a known EmitSpawnEventModifier")
        }));
    }

    #[test]
    fn spawn_event_node_outside_update_stack_is_reported() {
        let (mut effect_graph, parent_id, ..) = cpu_to_gpu_child_effect();
        effect_graph.emitter_mut(parent_id).unwrap().stacks.clear();
        let errors = validate_topology(&effect_graph);
        assert!(errors.iter().any(|e| {
            e.subject == TopologySubject::Node(NodeId::new(3).unwrap())
                && e.message
                    .contains("must be placed in its emitter's Update stack")
        }));
    }

    #[test]
    fn mixed_parent_spawn_event_nodes_are_reported() {
        let (mut effect_graph, .., child_id) = cpu_to_gpu_child_effect();
        // Move the second spawn-event node to a brand-new, unrelated emitter.
        let other_parent = EmitterId::new(30).unwrap();
        let mut other = EmitterGraph::empty(other_parent);
        let other_event = NodeId::new(31).unwrap();
        other.nodes.push(GraphNode {
            id: other_event,
            payload: NodePayload::Modifier(ModifierNodeData::Known {
                type_path: EmitSpawnEventModifier::type_path().into(),
                config: BTreeMap::new(),
            }),
            inputs: vec![],
        });
        other.stacks.push(GraphStack {
            id: StackId::new(32).unwrap(),
            group: ModifierGroup::Update,
            members: vec![other_event],
        });
        effect_graph.emitters.push(other);
        effect_graph.event_links[1].node = other_event;

        let errors = validate_topology(&effect_graph);
        let gpu_src = effect_graph.source_for_emitter(child_id).unwrap();
        assert!(errors.iter().any(|e| {
            e.subject == TopologySubject::Source(gpu_src)
                && e.message.contains("more than one parent emitter")
        }));
    }

    #[test]
    fn self_parent_cycle_is_reported() {
        let emitter_id = EmitterId::new(1).unwrap();
        let event_node = NodeId::new(2).unwrap();
        let stack_id = StackId::new(3).unwrap();
        let source_id = SourceId::new(4).unwrap();

        let mut emitter_graph = EmitterGraph::empty(emitter_id);
        emitter_graph.nodes.push(spawn_event_node(2));
        emitter_graph.stacks.push(GraphStack {
            id: stack_id,
            group: ModifierGroup::Update,
            members: vec![event_node],
        });

        let effect_graph = EffectGraph {
            emitters: vec![emitter_graph],
            sources: vec![gpu_source(4)],
            source_links: vec![SourceLink {
                source: source_id,
                emitter: emitter_id,
            }],
            event_links: vec![EventLink {
                node: event_node,
                target: source_id,
            }],
            next_id: 5,
        };

        let errors = validate_topology(&effect_graph);
        assert!(errors.iter().any(|e| {
            e.subject == TopologySubject::Emitter(emitter_id) && e.message.contains("cycle")
        }));
    }

    #[test]
    fn cpu_root_using_inherit_attribute_is_reported() {
        let mut effect_graph = single_emitter_effect();
        let node_id = NodeId::new(10).unwrap();
        effect_graph.emitters[0].nodes.push(GraphNode {
            id: node_id,
            payload: NodePayload::Modifier(ModifierNodeData::Known {
                type_path: InheritAttributeModifier::type_path().into(),
                config: BTreeMap::new(),
            }),
            inputs: vec![],
        });
        let errors = validate_topology(&effect_graph);
        assert!(errors.iter().any(|e| {
            e.subject == TopologySubject::Node(node_id)
                && e.message.contains("InheritAttributeModifier")
        }));
    }

    #[test]
    fn cpu_root_using_parent_attribute_expr_is_reported() {
        let mut effect_graph = single_emitter_effect();
        let node_id = NodeId::new(10).unwrap();
        effect_graph.emitters[0].nodes.push(GraphNode {
            id: node_id,
            payload: NodePayload::Expr(ExprNode::ParentAttribute(Attribute::POSITION)),
            inputs: vec![],
        });
        let errors = validate_topology(&effect_graph);
        assert!(errors.iter().any(|e| {
            e.subject == TopologySubject::Node(node_id) && e.message.contains("ParentAttribute")
        }));
    }

    #[test]
    fn cross_emitter_value_link_is_reported() {
        let (mut effect_graph, parent_id, child_id) = cpu_to_gpu_child_effect();
        let foreign_node = NodeId::new(50).unwrap();
        effect_graph
            .emitter_mut(child_id)
            .unwrap()
            .nodes
            .push(GraphNode {
                id: foreign_node,
                payload: NodePayload::Expr(ExprNode::Literal(bevy_hanabi::Value::from(1.0_f32))),
                inputs: vec![],
            });
        // Wire a link inside the parent emitter that reaches into the child.
        effect_graph
            .emitter_mut(parent_id)
            .unwrap()
            .links
            .push(GraphLink {
                from: PortRef {
                    node: foreign_node,
                    port: "out".into(),
                },
                to: PortRef {
                    node: NodeId::new(3).unwrap(),
                    port: "count".into(),
                },
            });
        let errors = validate_topology(&effect_graph);
        assert!(errors.iter().any(|e| {
            e.subject == TopologySubject::Node(foreign_node) && e.message.contains("crosses into")
        }));
    }

    #[test]
    fn sibling_fanout_from_one_parent_is_rejected() {
        let (mut effect_graph, parent_id, ..) = cpu_to_gpu_child_effect();
        // A second, distinct GPU source + child fed by the same parent.
        let second_gpu = SourceId::new(100).unwrap();
        let second_child = EmitterId::new(101).unwrap();
        let second_event = NodeId::new(102).unwrap();
        effect_graph.sources.push(SourceContext {
            id: second_gpu,
            kind: SourceKind::GpuEvent,
        });
        effect_graph
            .emitters
            .push(EmitterGraph::empty(second_child));
        effect_graph.source_links.push(SourceLink {
            source: second_gpu,
            emitter: second_child,
        });
        effect_graph
            .emitter_mut(parent_id)
            .unwrap()
            .nodes
            .push(spawn_event_node(102));
        effect_graph
            .emitter_mut(parent_id)
            .unwrap()
            .stacks
            .iter_mut()
            .find(|s| s.group == ModifierGroup::Update)
            .unwrap()
            .members
            .push(second_event);
        effect_graph.event_links.push(EventLink {
            node: second_event,
            target: second_gpu,
        });

        let errors = validate_topology(&effect_graph);
        assert!(errors.iter().any(|e| {
            e.subject == TopologySubject::Emitter(parent_id)
                && e.message.contains("only a single child per parent")
        }));
    }

    #[test]
    fn orphan_source_link_reference_is_reported() {
        let mut effect_graph = single_emitter_effect();
        let ghost = SourceId::new(999).unwrap();
        effect_graph.source_links.push(SourceLink {
            source: ghost,
            emitter: EmitterId::new(1).unwrap(),
        });
        let errors = validate_topology(&effect_graph);
        assert!(errors.iter().any(|e| {
            e.subject == TopologySubject::Source(ghost) && e.message.contains("does not exist")
        }));
    }

    #[test]
    fn orphan_event_link_node_is_reported() {
        let (mut effect_graph, ..) = cpu_to_gpu_child_effect();
        let ghost = NodeId::new(999).unwrap();
        effect_graph.event_links.push(EventLink {
            node: ghost,
            target: SourceId::new(7).unwrap(),
        });
        let errors = validate_topology(&effect_graph);
        assert!(errors.iter().any(|e| {
            e.subject == TopologySubject::Node(ghost) && e.message.contains("does not exist")
        }));
    }
}
