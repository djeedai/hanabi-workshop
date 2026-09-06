//! Read-only bridge from the effect-level [`EffectGraph`] to the standalone
//! [`node_graph`] widget.
//!
//! Implements [`GraphViewer`] directly over the canonical [`EffectGraph`], so
//! the widget renders the document's real graph — every emitter's nodes,
//! ordered modifier stacks, and links, plus the document's spawn source
//! contexts and the topology links between them — with no intermediate
//! projection. (This replaces the old `graph_adapter`, which reconstructed
//! graph topology from the *baked* `EffectAsset` because the asset is not a
//! graph.)
//!
//! Node, stack, and source ids all draw from the same document-wide allocator
//! (all `NonZeroU32`), so they map 1:1 onto the widget's id types with no
//! collision regardless of which emitter — or no emitter at all, for a source
//! context — they belong to. Every emitter's Init/Update/Render stacks and
//! every free expression/modifier node render on one shared canvas; only
//! value/expression links stay emitter-local (an invariant the model itself
//! upholds — see [`super::model`] — not something this bridge enforces).
//! Inline defaults — already modeled as unlinked [`InputSlot`]s — render as
//! value chips without any literal-hiding pass.
//!
//! ## Sources and cross-emitter links
//!
//! A [`SourceContext`] renders as an ordinary node addressed by its
//! [`SourceId`] (through [`wsource`]), with an interactive flow-output pin
//! ([`NodeDesc::with_flow_output`]) feeding an emitter's Init stack's
//! flow-input pin ([`StackDesc::with_flow_input`]) via a [`FlowLink`] built
//! from [`EffectGraph::source_links`]. A GPU event source additionally exposes
//! one multiple-link input port, fed by ordinary [`Link`]s from every
//! Update-stack `EmitSpawnEventModifier` that targets it (built from
//! [`EffectGraph::event_links`]); such an emitter is the only kind of modifier
//! node with an output port at all. Since a node id and a source id can never
//! collide, [`Self::validate_link`] tells the two kinds of link apart just by
//! checking whether the `to` port's node id resolves to a
//! [`EffectGraph::source`] — an event link — or an ordinary graph node — a
//! value link — and dispatches accordingly.
//!
//! The widget stays free of any `bevy_hanabi` import; this module is the
//! consumer that bridges the two.
//!
//! [`node_graph`]: hanabi_node_graph
//! [`InputSlot`]: super::model::InputSlot
//! [`SourceContext`]: super::model::SourceContext

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
};

use bevy::{
    math::{Vec3, Vec4},
    reflect::{TypePath, TypeRegistry},
};
use bevy_egui::egui::Color32;
use bevy_hanabi::{
    Attribute, CpuValue, EmitSpawnEventModifier, Gradient, ScalarType, SpawnerSettings,
    ToWgslString, Value, ValueType, VectorType, VectorValue,
};
use hanabi_node_graph::{
    FlowLink, GraphView, GraphViewer, Link, LinkVerdict, NodeDesc, NodeId as WNodeId, PortAddr,
    PortDesc, PortId, PortSide, StackDesc, StackId as WStackId, StackLink, WorldPos,
};

use super::{
    model::{
        EditValue, EffectGraph, EmitterId, ExprNode, GradientVec3, GradientVec4, GraphLink,
        GraphNode, ImageBinding, ModifierNodeData, NodeId, NodePayload, PortRef, PropertyDef,
        SharedStr, SlotId, SourceContext, SourceId, SourceKind, TextureSlotDef, TextureValue,
        is_select_image_input,
    },
    schema::{FieldRole, FlagDef, OUTPUT_PORT, flag_defs, modifier_schema},
};
use crate::{
    document::ModifierGroup,
    ui::{graph_validation, modifier_names::display_name_for_type},
};

/// Horizontal spacing between auto-layout columns (world units).
const COL_W: f64 = 220.0;
/// Vertical spacing between auto-layout rows (world units).
const ROW_H: f64 = 90.0;
/// Vertical gap left between consecutive seeded stacks (world units).
const STACK_GAP: f64 = 48.0;
// Rough geometry constants mirroring the widget's layout, used only to estimate
// stack heights when seeding so taller stacks don't pile on shorter ones.
const EST_NODE_HEADER: f64 = 26.0;
const EST_ROW_H: f64 = 22.0;
const EST_NODE_BODY_PAD: f64 = 14.0;
const EST_STACK_HEADER: f64 = 24.0;
const EST_STACK_PAD: f64 = 8.0;
const EST_MEMBER_GAP: f64 = 6.0;

/// Max displayed length of an inlined value chip; longer values are truncated.
const CHIP_MAX: usize = 18;

/// Reserved port name for a spawn-event node's output and a GPU source's input.
const EVENT_PORT: &str = "event";

/// A read-only view of a effect-level [`EffectGraph`] as graph topology.
///
/// Borrows the graph and the type registry (needed for modifier schemas and
/// display names); builds no precomputed snapshot.
pub struct GraphReader<'a> {
    effect_graph: &'a EffectGraph,
    registry: &'a TypeRegistry,
    /// node id → `(group, index)` for stack members, across every emitter;
    /// drives accents, execution order, and which nodes float vs. live in a
    /// stack.
    member_of: HashMap<NodeId, (ModifierGroup, usize)>,
    /// `(group, index)` → the attributes that make a modifier shadowed, paired
    /// with the index of the later modifier that overwrites each. Drives the
    /// per-node warning badge. Empty unless seeded via [`Self::with_shadows`].
    shadowed: HashMap<(ModifierGroup, usize), Vec<(Attribute, usize)>>,
    /// `(node id, row key)` pairs whose collapsible host editor is expanded.
    ///
    /// Modifier gradients use their field name; Image nodes use the reserved
    /// `image` row key.
    expanded: HashSet<(u32, String)>,
    /// GPU source id → current strict-bake failure for that event chain.
    source_warnings: HashMap<SourceId, String>,
}

/// An editable inline value the user clicked, resolved to its model target.
///
/// The widget is value-type-agnostic, so this is how the panel learns what
/// editor to present and which edit to emit.
pub enum EditableChip {
    /// An inlined literal on an expression operand port.
    Literal {
        node: NodeId,
        port: SharedStr,
        value: Value,
    },
    /// A modifier's `attribute` config field (e.g. `SetAttributeModifier`).
    Attribute {
        group: ModifierGroup,
        idx: usize,
        current: Attribute,
    },
    /// The attribute selector of a Get Attribute expression node.
    ExpressionAttribute { node: NodeId, current: Attribute },
    /// A modifier's `bool` config field (e.g.
    /// `SizeOverLifetimeModifier::screen_space_size`), edited as an inline
    /// checkbox. `value` is the current state.
    Bool {
        node: NodeId,
        field: SharedStr,
        value: bool,
    },
    /// An Age node's normalized-age or clamp toggle.
    AgeOption {
        node: NodeId,
        normalized: bool,
        clamped: bool,
        is_clamped: bool,
    },
    /// A modifier's data-less enum config field (e.g. `ShapeDimension`,
    /// `OrientMode`). `variants` are the selectable unit-variant names.
    Enum {
        node: NodeId,
        field: SharedStr,
        type_path: SharedStr,
        current: SharedStr,
        variants: Vec<SharedStr>,
    },
    /// A modifier's bitflags config field (e.g. `ColorBlendMask`). `defs` are
    /// the independently-toggleable named bits; `bits` is the current mask.
    Flags {
        node: NodeId,
        field: SharedStr,
        type_path: SharedStr,
        bits: u64,
        defs: Vec<FlagDef>,
    },
    /// An image-binding selector. With `port` set it targets a consumer's
    /// inline image input (sampler `image` / modifier `texture_slot`);
    /// without, an Image source node. `current` is the present binding;
    /// `slots` are the selectable texture-slot `(id, name)` pairs in slot
    /// order, offered alongside asset/unbound.
    ImageBinding {
        node: NodeId,
        port: Option<SharedStr>,
        current: ImageBinding,
        /// Required texture view dimension for a texture-read input. Image
        /// source nodes have no intrinsic dimension until a consumer uses them.
        dimension: Option<bevy_hanabi::SlotDimension>,
        slots: Vec<(SlotId, SharedStr)>,
    },
    /// A `Vec3` analytical gradient config field (e.g. size over lifetime),
    /// edited as a uniform-scalar curve. `keys` are `(ratio, value)` pairs.
    Gradient3 {
        node: NodeId,
        field: SharedStr,
        keys: Vec<(f32, f32)>,
    },
    /// A `Vec4` analytical gradient config field (e.g. color over lifetime),
    /// edited as a color stop strip. `keys` are `(ratio, rgba)` pairs.
    Gradient4 {
        node: NodeId,
        field: SharedStr,
        keys: Vec<(f32, [f32; 4])>,
    },
    /// A single-valued `CpuValue<Vec3>`/`CpuValue<Vec4>` config field (e.g.
    /// `SetSizeModifier::size`), edited as an inline multi-component scrubber.
    /// `value` carries the current vector and its component count.
    VectorConfig {
        node: NodeId,
        field: SharedStr,
        value: VectorValue,
    },
    /// A CPU spawner's particle emission count (`SpawnerSettings::count`).
    /// Editable only when authored as a single scalar (not a random range),
    /// matching a source node's rows built by `source_node_desc`.
    CpuSpawnerCount { source: SourceId, value: f32 },
    /// A CPU spawner's emission duration in seconds
    /// (`SpawnerSettings::spawn_duration`).
    CpuSpawnerSpawnDuration { source: SourceId, value: f32 },
    /// A CPU spawner's emission period in seconds (`SpawnerSettings::period`).
    CpuSpawnerPeriod { source: SourceId, value: f32 },
    /// A CPU spawner's repeat count (`0` = infinite;
    /// `SpawnerSettings::cycle_count`).
    CpuSpawnerCycleCount { source: SourceId, value: u32 },
    /// Whether a CPU spawner begins its emission cycle automatically
    /// (`SpawnerSettings::starts_active`).
    CpuSpawnerStartsActive { source: SourceId, value: bool },
    /// Whether a CPU spawner emits as soon as it starts
    /// (`SpawnerSettings::emits_on_start`).
    CpuSpawnerEmitsOnStart { source: SourceId, value: bool },
}

impl<'a> GraphReader<'a> {
    pub fn new(effect_graph: &'a EffectGraph, registry: &'a TypeRegistry) -> Self {
        let mut member_of = HashMap::new();
        for emitter in &effect_graph.emitters {
            for stack in &emitter.stacks {
                for (idx, &member) in stack.members.iter().enumerate() {
                    member_of.insert(member, (stack.group, idx));
                }
            }
        }
        Self {
            effect_graph,
            registry,
            member_of,
            shadowed: HashMap::new(),
            expanded: HashSet::new(),
            source_warnings: HashMap::new(),
        }
    }

    /// The node with the given id, searched across every emitter.
    ///
    /// Node ids are unique across the whole document (see
    /// [`EffectGraph::next_id`]), so this is unambiguous regardless of which
    /// emitter actually owns `id`.
    fn model_node(&self, id: NodeId) -> Option<&'a GraphNode> {
        self.effect_graph.emitters.iter().find_map(|e| e.node(id))
    }

    /// The property with the given id, searched across every emitter.
    fn model_property(&self, id: super::model::PropertyId) -> Option<&'a PropertyDef> {
        self.effect_graph
            .emitters
            .iter()
            .find_map(|e| e.property(id))
    }

    /// The texture slot with the given id, searched across every emitter.
    fn model_texture_slot(&self, id: SlotId) -> Option<&'a TextureSlotDef> {
        self.effect_graph
            .emitters
            .iter()
            .find_map(|e| e.texture_slot(id))
    }

    /// Every value/expression link across every emitter.
    ///
    /// Safe to flatten: a [`GraphLink`]'s endpoints always resolve within its
    /// own emitter (an invariant the model upholds, checked independently by
    /// [`crate::effect_graph::validation::validate_topology`]), so searching
    /// across all of them finds the same, unique answer a per-emitter search
    /// would.
    fn model_links(&self) -> impl Iterator<Item = &'a GraphLink> {
        self.effect_graph
            .emitters
            .iter()
            .flat_map(|e| e.links.iter())
    }

    /// Every node across every emitter (not including spawn source contexts).
    fn model_nodes(&self) -> impl Iterator<Item = &'a GraphNode> {
        self.effect_graph
            .emitters
            .iter()
            .flat_map(|e| e.nodes.iter())
    }

    /// Attach shadowed-modifier analysis, keyed by `(group, index)`.
    ///
    /// See [`crate::effect_graph::validation`]; shadowed members render a
    /// warning badge.
    pub fn with_shadows(
        mut self,
        shadowed: HashMap<(ModifierGroup, usize), Vec<(Attribute, usize)>>,
    ) -> Self {
        self.shadowed = shadowed;
        self
    }

    /// Attach strict-bake failures to their GPU Event source contexts.
    pub fn with_bake_errors(
        mut self,
        errors: &[crate::effect_graph::bake::EffectBakeError],
    ) -> Self {
        for error in errors {
            let Some(source) = error.source else {
                continue;
            };
            if !self
                .effect_graph
                .source(source)
                .is_some_and(|source| matches!(source.kind, SourceKind::GpuEvent))
            {
                continue;
            }
            let warning = self.source_warnings.entry(source).or_insert_with(|| {
                "This GPU event chain is currently ignored by the preview and blocks baking.\n\
                 The graph can still be saved."
                    .to_string()
            });
            warning.push_str("\n\n");
            warning.push_str(&error.error.message);
        }
        self
    }

    /// Mark which collapsible host editor rows are expanded.
    ///
    /// Keyed by `(node id, config field)`; absent rows render collapsed.
    pub fn with_expanded(mut self, expanded: HashSet<(u32, String)>) -> Self {
        self.expanded = expanded;
        self
    }

    /// Apply seed positions for any node/stack/source the view hasn't placed
    /// yet.
    ///
    /// A freshly opened graph lays itself out instead of piling at the origin.
    /// User drags persist (only unset positions are seeded).
    pub fn seed_positions(&self, view: &mut GraphView) {
        let (expr_seed, stack_seed) = self.seed_layout();
        for (id, pos) in expr_seed {
            view.ensure_position(id, pos);
        }
        for (id, pos) in stack_seed {
            view.ensure_stack_position(id, pos);
        }
    }

    /// The spawn source context for `id`, if it exists.
    ///
    /// Used by chip-overlay editors (e.g. the CPU spawner's inline fields;
    /// see [`crate::ui::panels::graph`]) that need to read the source's
    /// current settings to reconstruct the whole value on commit.
    pub fn source(&self, id: SourceId) -> Option<&SourceContext> {
        self.effect_graph.source(id)
    }

    /// The emitter owning a node, if any (`None` for a source id).
    pub fn emitter_of_node(&self, node: NodeId) -> Option<EmitterId> {
        self.effect_graph.emitter_owning_node(node)
    }

    /// The emitter a spawn source currently drives, if it is connected.
    ///
    /// `None` for an unconnected source — the caller (active-emitter tracking;
    /// see [`crate::ui::panels::graph`]) treats that as "leave the active
    /// emitter untouched" rather than as a resolution failure.
    pub fn emitter_of_source(&self, source: SourceId) -> Option<EmitterId> {
        self.effect_graph.emitter_for_source(source)
    }

    /// The emitter a canvas item (widget node id) drives, for active-emitter
    /// tracking.
    ///
    /// A source id resolves through [`Self::emitter_of_source`] (so an
    /// unconnected source yields `None`, leaving the active emitter alone); an
    /// ordinary node id resolves through [`Self::emitter_of_node`].
    pub fn emitter_of_canvas_node(&self, id: WNodeId) -> Option<EmitterId> {
        if let Some(source) = SourceId::new(id.get())
            && self.effect_graph.source(source).is_some()
        {
            return self.emitter_of_source(source);
        }
        self.emitter_of_node(NodeId::new(id.get())?)
    }

    /// The emitter owning a widget stack id, if any.
    pub fn emitter_of_stack(&self, stack: WStackId) -> Option<EmitterId> {
        let id = super::model::StackId::new(stack.get())?;
        self.effect_graph.emitter_owning_stack(id)
    }

    /// The next id [`crate::edits`] should hand to a fresh document-topology
    /// mutation, mirroring [`EffectGraph::next_id`].
    pub fn next_id(&self) -> u32 {
        self.effect_graph.next_id
    }

    /// The `(group, index)` of a stack member, if `node` is one.
    ///
    /// Public accessor for [`Self::member_of`], needed by the Graph panel to
    /// resolve e.g. a config-field edit target without duplicating the
    /// stack-scan this reader already did.
    pub fn member_of(&self, node: NodeId) -> Option<(ModifierGroup, usize)> {
        self.member_of.get(&node).copied()
    }

    /// Whether `node`'s sole output is the event pin (an Update-stack
    /// `EmitSpawnEventModifier`), as opposed to an ordinary value/image output
    /// or none at all.
    ///
    /// Lets the Graph panel recognize a dangling drag released from an event
    /// output (its [`PortType::Value`]/[`PortType::Image`]-oriented
    /// [`Self::port_type`] doesn't cover it) so it can offer "New GPU Event
    /// Source" instead of the ordinary producer/consumer picker.
    pub fn is_event_source(&self, node: NodeId) -> bool {
        let Some(n) = self.model_node(node) else {
            return false;
        };
        let NodePayload::Modifier(ModifierNodeData::Known { type_path, .. }) = &n.payload else {
            return false;
        };
        is_emit_spawn_event_modifier(type_path)
            && self
                .member_of(node)
                .is_some_and(|(g, _)| g == ModifierGroup::Update)
    }

    /// The connectable input port names of a node, in order.
    ///
    /// Operand ports for an expression, expression-field ports for a modifier.
    /// These come first in the node's input list, so their indices double as
    /// the widget port index.
    fn connectable_inputs(&self, node: &GraphNode) -> Vec<Cow<'static, str>> {
        match &node.payload {
            NodePayload::Expr(e) => e.input_ports().iter().map(|s| Cow::Borrowed(*s)).collect(),
            NodePayload::Modifier(ModifierNodeData::Known { type_path, .. }) => self
                .schema_ports(type_path)
                .into_iter()
                .map(Cow::Owned)
                .collect(),
            NodePayload::Modifier(ModifierNodeData::Unknown { .. }) => Vec::new(),
        }
    }

    /// Map a widget link back to a model [`GraphLink`].
    ///
    /// Returns `None` if either endpoint no longer resolves. The inverse of the
    /// index↔name mapping this reader builds for the widget: outputs are a
    /// node's single `out` port; inputs are looked up by their position in
    /// [`connectable_inputs`].
    ///
    /// [`connectable_inputs`]: Self::connectable_inputs
    pub fn resolve_link(&self, from: PortAddr, to: PortAddr) -> Option<GraphLink> {
        let from_node = NodeId::new(from.node.get())?;
        let to_node = NodeId::new(to.node.get())?;
        let target = self.model_node(to_node)?;
        let to_port = self
            .connectable_inputs(target)
            .get(to.port.index as usize)?
            .as_ref()
            .to_owned();
        Some(GraphLink {
            from: PortRef {
                node: from_node,
                port: OUTPUT_PORT.into(),
            },
            to: PortRef {
                node: to_node,
                port: SharedStr::from(to_port),
            },
        })
    }

    /// Map a widget link back to a model event link `(emitter, GPU source)`.
    ///
    /// `None` if `to` doesn't address a spawn source at all (an ordinary value
    /// link — see [`Self::resolve_link`]) or if `from` no longer resolves.
    /// Doesn't itself require `from` to be a valid emitter or `to` a GPU (as
    /// opposed to CPU) source — that's [`Self::validate_link`]'s job, run
    /// before the widget ever reports the drag as accepted.
    pub fn resolve_event_link(&self, from: PortAddr, to: PortAddr) -> Option<(NodeId, SourceId)> {
        let from_node = NodeId::new(from.node.get())?;
        let target = SourceId::new(to.node.get())?;
        self.effect_graph.source(target)?;
        Some((from_node, target))
    }

    /// Map a widget flow link back to a model source link `(source, emitter)`.
    ///
    /// `None` if `from` doesn't address a spawn source or `to` doesn't resolve
    /// to any emitter's stack (not necessarily its Init stack — see
    /// [`Self::validate_flow_link`]).
    pub fn resolve_flow_link(&self, from: WNodeId, to: WStackId) -> Option<(SourceId, EmitterId)> {
        let source = SourceId::new(from.get())?;
        self.effect_graph.source(source)?;
        let emitter = self.emitter_of_stack(to)?;
        Some((source, emitter))
    }

    /// Expression-port field names of a modifier type, in declaration order.
    fn schema_ports(&self, type_path: &str) -> Vec<String> {
        self.registry
            .get_with_type_path(type_path)
            .and_then(|reg| modifier_schema(reg.type_info()))
            .map(|s| s.ports().map(|f| f.name.to_string()).collect())
            .unwrap_or_default()
    }

    /// Whether `port` on `node` is a modifier texture-slot field (image-typed).
    fn is_modifier_texture_port(&self, node: NodeId, port: &str) -> bool {
        let Some(NodePayload::Modifier(ModifierNodeData::Known { type_path, .. })) =
            self.model_node(node).map(|n| &n.payload)
        else {
            return false;
        };
        self.registry
            .get_with_type_path(type_path)
            .and_then(|reg| modifier_schema(reg.type_info()))
            .is_some_and(|s| {
                s.fields
                    .iter()
                    .any(|f| &*f.name == port && matches!(f.role, FieldRole::Texture))
            })
    }

    /// The source node feeding `node`'s input `port`, if a link targets it.
    fn linked_source(&self, node: NodeId, port: &str) -> Option<NodeId> {
        self.model_links()
            .find(|l| l.to.node == node && &*l.to.port == port)
            .map(|l| l.from.node)
    }

    /// The inline-default literal for `node`'s input `port`, if it carries a
    /// value default.
    fn inline_default(&self, node: NodeId, port: &str) -> Option<Value> {
        self.model_node(node)?
            .inputs
            .iter()
            .find(|s| &*s.name == port)
            .and_then(|s| s.default.as_value())
    }

    /// The inline image binding for `node`'s input `port`, if it carries one.
    fn inline_image(&self, node: NodeId, port: &str) -> Option<ImageBinding> {
        self.model_node(node)?
            .inputs
            .iter()
            .find(|s| &*s.name == port)
            .and_then(|s| s.default.as_image().cloned())
    }

    /// Output type of an expression node, if it can be inferred.
    ///
    /// `None` for modifier nodes or when the type can't be inferred. Operators
    /// infer from their first operand; a `visited` set guards against malformed
    /// cyclic graphs.
    fn output_type(&self, node: NodeId) -> Option<PortType> {
        self.output_type_rec(node, &mut Vec::new())
    }

    /// Output [`ValueType`] of expression `node`, if it resolves to a value.
    ///
    /// `None` for modifier or image-typed nodes and for types that cannot be
    /// inferred. Used to retype an operator's sibling operand defaults when a
    /// link connects a value into it.
    pub fn node_output_value_type(&self, node: NodeId) -> Option<ValueType> {
        match self.output_type(node)? {
            PortType::Value(vt) => Some(vt),
            PortType::Image => None,
        }
    }

    fn output_type_rec(&self, node: NodeId, visited: &mut Vec<NodeId>) -> Option<PortType> {
        if visited.contains(&node) {
            return None;
        }
        visited.push(node);
        let result = match &self.model_node(node)?.payload {
            NodePayload::Expr(e) => match e {
                ExprNode::Literal(v) => Some(PortType::Value(v.value_type())),
                ExprNode::Property(pid) => self
                    .model_property(*pid)
                    .map(|p| PortType::Value(p.default.value_type())),
                ExprNode::Attribute(a) | ExprNode::ParentAttribute(a) => {
                    Some(PortType::Value(a.value_type()))
                }
                ExprNode::Age { .. } => Some(PortType::Value(ValueType::Scalar(ScalarType::Float))),
                ExprNode::BuiltIn(op) => Some(PortType::Value(op.value_type())),
                ExprNode::Cast(vt) => Some(PortType::Value(*vt)),
                ExprNode::Image(_) => Some(PortType::Image),
                ExprNode::TextureSample1d
                | ExprNode::TextureSample2d
                | ExprNode::TextureSample3d
                | ExprNode::TextureLoad1d
                | ExprNode::TextureLoad2d
                | ExprNode::TextureLoad3d => {
                    Some(PortType::Value(ValueType::Vector(VectorType::VEC4F)))
                }
                ExprNode::SelectImage { .. } => Some(PortType::Image),
                ExprNode::Unary(_) | ExprNode::Binary(_) | ExprNode::Ternary(_) => {
                    // Operator-aware output type: reductions, comparisons,
                    // swizzles, and constructors need more than the naive
                    // "first operand's type" rule.
                    e.output_value_type(|port| match self.operand_type_rec(node, port, visited) {
                        Some(PortType::Value(vt)) => Some(vt),
                        _ => None,
                    })
                    .map(PortType::Value)
                }
            },
            NodePayload::Modifier(_) => None,
        };
        visited.pop();
        result
    }

    /// Type expected at `node`'s input `port`.
    ///
    /// The texture-sampling `image` input expects the [`Image`] pseudo-type;
    /// every other input takes the linked source's output type, or the inline
    /// default's type.
    ///
    /// [`Image`]: PortType::Image
    fn operand_type(&self, node: NodeId, port: &str) -> Option<PortType> {
        self.operand_type_rec(node, port, &mut Vec::new())
    }

    fn operand_type_rec(
        &self,
        node: NodeId,
        port: &str,
        visited: &mut Vec<NodeId>,
    ) -> Option<PortType> {
        // The sampler's image input is image-typed regardless of what feeds it,
        // so it colors as an image port and only accepts an image.
        if port == "image"
            && matches!(
                self.model_node(node).map(|n| &n.payload),
                Some(NodePayload::Expr(expr)) if expr.texture_dimension().is_some()
            )
        {
            return Some(PortType::Image);
        }
        // A modifier's texture-slot field is likewise image-typed: it accepts an
        // image source, no matter what currently feeds it.
        if self.is_modifier_texture_port(node, port) {
            return Some(PortType::Image);
        }
        // A `SelectImage` node's image inputs are image-typed; only its `index`
        // selector takes a value.
        if matches!(
            self.model_node(node).map(|n| &n.payload),
            Some(NodePayload::Expr(ExprNode::SelectImage { .. }))
        ) && is_select_image_input(port)
        {
            return Some(PortType::Image);
        }
        if let Some(src) = self.linked_source(node, port) {
            self.output_type_rec(src, visited)
        } else {
            self.inline_default(node, port)
                .map(|v| PortType::Value(v.value_type()))
        }
    }

    /// Type carried by a widget port.
    ///
    /// An output reports the node's output type; an input reports the type it
    /// expects (image pseudo-type for a sampler's image input, else the linked
    /// source or inline default). Used to filter create-menu candidates against
    /// the type of the dangling pin that opened the menu.
    pub fn port_type(&self, addr: PortAddr, is_output: bool) -> Option<PortType> {
        let node = NodeId::new(addr.node.get())?;
        if is_output {
            self.output_type(node)
        } else {
            let name = self
                .connectable_inputs(self.model_node(node)?)
                .get(addr.port.index as usize)
                .cloned()?;
            self.operand_type(node, &name)
        }
    }

    /// Resolve a widget input-port chip to the model value it edits.
    ///
    /// Returns `None` when the chip isn't editable (an output port, a linked
    /// input, or a config field with no editor yet). The widget reports only
    /// *which* port was clicked; this maps it back to the model so the panel
    /// can present a type-appropriate editor.
    pub fn editable_chip(&self, addr: PortAddr) -> Option<EditableChip> {
        if addr.port.side != PortSide::Input {
            return None;
        }
        if let Some(source_id) = SourceId::new(addr.node.get())
            && let Some(SourceContext {
                kind: SourceKind::CpuSpawner { settings },
                ..
            }) = self.effect_graph.source(source_id)
        {
            return cpu_spawner_chip(source_id, settings, addr.port.index as usize);
        }
        let node_id = NodeId::new(addr.node.get())?;
        let node = self.model_node(node_id)?;
        let conn = self.connectable_inputs(node);
        let idx = addr.port.index as usize;

        if idx < conn.len() {
            // A connectable operand port: editable only when nothing is linked
            // into it. An image port offers a binding selector; every other port
            // edits its inline literal default.
            let name = conn[idx].as_ref();
            if self.linked_source(node_id, name).is_some() {
                return None;
            }
            if self.operand_type(node_id, name) == Some(PortType::Image) {
                // A `SelectImage` image input is fed by a link only: it carries
                // no inline binding, so it offers no selector chip.
                if matches!(
                    &node.payload,
                    NodePayload::Expr(ExprNode::SelectImage { .. })
                ) {
                    return None;
                }
                return Some(EditableChip::ImageBinding {
                    node: node_id,
                    port: Some(SharedStr::from(name)),
                    current: self.inline_image(node_id, name).unwrap_or_default(),
                    dimension: self
                        .model_node(node_id)
                        .and_then(|node| match &node.payload {
                            NodePayload::Expr(expr) => expr.texture_dimension(),
                            NodePayload::Modifier(_) => Some(bevy_hanabi::SlotDimension::D2),
                        }),
                    slots: self.texture_slot_pairs(node_id),
                });
            }
            let value = self.inline_default(node_id, name)?;
            return Some(EditableChip::Literal {
                node: node_id,
                port: SharedStr::from(name),
                value,
            });
        }

        // An image node's binding row sits just past its (empty) operand ports.
        if let NodePayload::Expr(ExprNode::Image(current)) = &node.payload
            && idx == conn.len()
        {
            return Some(EditableChip::ImageBinding {
                node: node_id,
                port: None,
                current: current.clone(),
                dimension: None,
                slots: self.texture_slot_pairs(node_id),
            });
        }

        if let Some(attribute) = node_expression_attribute(&node.payload) {
            if idx == conn.len() {
                return Some(EditableChip::ExpressionAttribute {
                    node: node_id,
                    current: attribute,
                });
            }
        }

        if let Some((normalized, clamped)) = node_age_options(&node.payload) {
            return match idx - conn.len() - 1 {
                0 => Some(EditableChip::AgeOption {
                    node: node_id,
                    normalized,
                    clamped,
                    is_clamped: false,
                }),
                1 if normalized => Some(EditableChip::AgeOption {
                    node: node_id,
                    normalized,
                    clamped,
                    is_clamped: true,
                }),
                _ => None,
            };
        }

        // Otherwise it's a modifier config display row.
        let NodePayload::Modifier(ModifierNodeData::Known { type_path, config }) = &node.payload
        else {
            return None;
        };
        let field = self
            .config_fields(type_path)
            .into_iter()
            .nth(idx - conn.len())?;
        match config.get(field.as_str())? {
            EditValue::Bool(b) => Some(EditableChip::Bool {
                node: node_id,
                field: SharedStr::from(field.as_str()),
                value: *b,
            }),
            EditValue::Attribute(attr) => {
                let (group, midx) = self.member_of.get(&node_id).copied()?;
                Some(EditableChip::Attribute {
                    group,
                    idx: midx,
                    current: *attr,
                })
            }
            EditValue::Enum {
                type_path: enum_path,
                variant,
            } => {
                let variants = self.enum_variants(enum_path);
                if variants.is_empty() {
                    return None;
                }
                Some(EditableChip::Enum {
                    node: node_id,
                    field: SharedStr::from(field.as_str()),
                    type_path: enum_path.clone(),
                    current: variant.clone(),
                    variants,
                })
            }
            EditValue::Flags {
                type_path: flags_path,
                bits,
            } => {
                let defs = flag_defs(flags_path);
                if defs.is_empty() {
                    return None;
                }
                Some(EditableChip::Flags {
                    node: node_id,
                    field: SharedStr::from(field.as_str()),
                    type_path: flags_path.clone(),
                    bits: *bits,
                    defs,
                })
            }
            EditValue::Gradient3(GradientVec3::Analytical(grad)) => Some(EditableChip::Gradient3 {
                node: node_id,
                field: SharedStr::from(field.as_str()),
                keys: grad.keys().iter().map(|k| (k.ratio(), k.value.x)).collect(),
            }),
            EditValue::Gradient4(GradientVec4::Analytical(grad)) => Some(EditableChip::Gradient4 {
                node: node_id,
                field: SharedStr::from(field.as_str()),
                keys: grad
                    .keys()
                    .iter()
                    .map(|k| (k.ratio(), k.value.to_array()))
                    .collect(),
            }),
            EditValue::CpuVec3(CpuValue::Single(v)) => Some(EditableChip::VectorConfig {
                node: node_id,
                field: SharedStr::from(field.as_str()),
                value: VectorValue::new_vec3(*v),
            }),
            EditValue::CpuVec4(CpuValue::Single(v)) => Some(EditableChip::VectorConfig {
                node: node_id,
                field: SharedStr::from(field.as_str()),
                value: VectorValue::new_vec4(*v),
            }),
            _ => None,
        }
    }

    /// The selectable unit-variant names of a data-less enum type.
    ///
    /// In declaration order. Empty if the type isn't a registered enum.
    fn enum_variants(&self, type_path: &str) -> Vec<SharedStr> {
        use bevy::reflect::{TypeInfo, enums::VariantInfo};
        let Some(reg) = self.registry.get_with_type_path(type_path) else {
            return Vec::new();
        };
        let TypeInfo::Enum(info) = reg.type_info() else {
            return Vec::new();
        };
        info.iter()
            .filter(|v| matches!(v, VariantInfo::Unit(_)))
            .map(|v| SharedStr::from(v.name()))
            .collect()
    }

    /// Build a node's input ports.
    ///
    /// Connectable expr ports first, then read-only config display rows for a
    /// modifier.
    fn input_ports(&self, node: &GraphNode) -> Vec<PortDesc> {
        let mut ports = Vec::new();
        for name in self.connectable_inputs(node) {
            let mut port = PortDesc::new(prettify_label(&name));
            let ty = self.operand_type(node.id, &name);
            if let Some(t) = ty {
                port = port.with_color(port_type_color(t));
            }
            // Pips reflect the port's own declared type (its inline default)
            // rather than the linked source, so they stay stable — and keep
            // conveying the slot's arity — whether or not something is wired in.
            let declared = self
                .inline_default(node.id, &name)
                .map(|v| PortType::Value(v.value_type()))
                .or(ty);
            if let Some(t) = declared {
                port = port.with_arity(port_type_arity(t));
            }
            if self.linked_source(node.id, &name).is_some() {
                // Linked: a connection target; the link is emitted by `links()`.
                ports.push(port);
            } else if ty == Some(PortType::Image) {
                if matches!(
                    &node.payload,
                    NodePayload::Expr(ExprNode::SelectImage { .. })
                ) {
                    // A `SelectImage` image input is a link-only target; it shows
                    // as a bare pin with no inline binding selector.
                    ports.push(port);
                } else {
                    // An unconnected image port shows its inline binding as a
                    // clickable selector, like the Image source node's row.
                    let binding = self.inline_image(node.id, &name).unwrap_or_default();
                    ports.push(port.with_value(self.image_binding_label(&binding)));
                }
            } else if let Some(def) = self.inline_default(node.id, &name) {
                // Vec3/Vec4 defaults get a multi-component editor box on the
                // line(s) below the (potentially long) label; everything else
                // shows a single-line value chip.
                if let Some(height) = vector_editor_height(&def) {
                    ports.push(port.with_editor_box(height));
                } else {
                    ports.push(port.with_value(short_literal(&def.to_wgsl_string())));
                }
            } else {
                // Optional, unconnected port with no default.
                ports.push(port);
            }
        }
        // Read-only display rows for a modifier's non-expr configuration.
        if let NodePayload::Modifier(ModifierNodeData::Known { type_path, config }) = &node.payload
        {
            for field in self.config_fields(type_path) {
                if let Some(value) = config.get(field.as_str()) {
                    let exp = self.expanded.contains(&(node.id.get(), field.clone()));
                    let port = match value {
                        EditValue::Gradient3(GradientVec3::Analytical(_)) => {
                            PortDesc::new(prettify_label(&field)).collapsible(exp.then_some(96.0))
                        }
                        EditValue::Gradient4(GradientVec4::Analytical(_)) => {
                            PortDesc::new(prettify_label(&field)).collapsible(exp.then_some(54.0))
                        }
                        // A bool renders as a compact checkbox overlaid by the
                        // panel; the chip itself carries no text.
                        EditValue::Bool(_) => {
                            PortDesc::new(prettify_label(&field)).display_value("")
                        }
                        // A single-valued vector gets the inline per-component
                        // editor box below its label, like an operand vec3/vec4.
                        EditValue::CpuVec3(CpuValue::Single(_))
                        | EditValue::CpuVec4(CpuValue::Single(_)) => {
                            PortDesc::new(prettify_label(&field))
                                .display_editor_box(VECTOR_EDITOR_ROW_H)
                        }
                        _ => PortDesc::new(prettify_label(&field))
                            .display_value(format_config(value)),
                    };
                    ports.push(port);
                }
            }
        }
        if let Some(attribute) = node_expression_attribute(&node.payload) {
            ports.push(PortDesc::new("Attribute").display_value(attribute.name().to_string()));
        }
        if node_age_options(&node.payload).is_some() {
            ports.push(PortDesc::new("Normalize").display_value(""));
            if matches!(
                &node.payload,
                NodePayload::Expr(ExprNode::Age {
                    normalized: true,
                    ..
                })
            ) {
                ports.push(PortDesc::new("Clamped").display_value(""));
            }
        }
        // An image node shows a collapsible preview and binding selector.
        if let NodePayload::Expr(ExprNode::Image(_)) = &node.payload {
            let expanded = self
                .expanded
                .contains(&(node.id.get(), "image".to_string()));
            ports.push(
                PortDesc::new(prettify_label("image")).collapsible(expanded.then_some(112.0)),
            );
        }
        ports
    }

    /// The selectable texture-slot `(id, name)` pairs of the emitter owning
    /// `node`, in slot order.
    ///
    /// Scoped to that one emitter: a texture-slot binding, like every other
    /// value/expression reference, stays emitter-local.
    fn texture_slot_pairs(&self, node: NodeId) -> Vec<(SlotId, SharedStr)> {
        let Some(emitter_id) = self.effect_graph.emitter_owning_node(node) else {
            return Vec::new();
        };
        self.effect_graph
            .emitter(emitter_id)
            .map(|e| {
                e.texture_slots
                    .iter()
                    .map(|s| (s.id, s.name.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// A short label for an image node's binding, for its display row.
    ///
    /// An asset shows its file name, a texture slot its bracketed name, and an
    /// unbound source a placeholder.
    fn image_binding_label(&self, binding: &ImageBinding) -> String {
        match binding {
            ImageBinding::Unbound => "(unbound)".to_string(),
            ImageBinding::Asset(path) => {
                let s = path.to_string();
                s.rsplit(['/', '\\']).next().unwrap_or(&s).to_string()
            }
            ImageBinding::TypedAsset { path, .. } => {
                let s = path.to_string();
                s.rsplit(['/', '\\']).next().unwrap_or(&s).to_string()
            }
            ImageBinding::Slot(id) => self
                .model_texture_slot(*id)
                .map(|s| format!("[{}]", s.name))
                .unwrap_or_else(|| "[missing]".to_string()),
        }
    }

    /// Config field names of a modifier type, in declaration order.
    fn config_fields(&self, type_path: &str) -> Vec<String> {
        self.registry
            .get_with_type_path(type_path)
            .and_then(|reg| modifier_schema(reg.type_info()))
            .map(|s| s.config().map(|f| f.name.to_string()).collect())
            .unwrap_or_default()
    }

    /// Whether linking `from → to` would close a cycle.
    ///
    /// I.e. `from` already depends transitively on `to`.
    fn would_cycle(&self, from: NodeId, to: NodeId) -> bool {
        let mut stack = vec![from];
        let mut seen: HashSet<NodeId> = HashSet::new();
        while let Some(n) = stack.pop() {
            if n == to {
                return true;
            }
            if !seen.insert(n) {
                continue;
            }
            for l in self.model_links() {
                if l.to.node == n {
                    stack.push(l.from.node);
                }
            }
        }
        false
    }

    /// Execution rank `(group_order, index)` of a stacked modifier member.
    ///
    /// `None` for a free expression node. Lower ranks run earlier.
    fn exec_rank(&self, node: NodeId) -> Option<(u32, usize)> {
        self.member_of
            .get(&node)
            .map(|(group, idx)| (group_order(*group), *idx))
    }

    /// Longest chain of *linked* operands below a node (leaves are 0). Inline
    /// defaults are not nodes and don't add depth.
    fn node_depth(
        &self,
        node: NodeId,
        memo: &mut HashMap<NodeId, u32>,
        visited: &mut Vec<NodeId>,
    ) -> u32 {
        if let Some(d) = memo.get(&node) {
            return *d;
        }
        if visited.contains(&node) {
            return 0;
        }
        visited.push(node);
        let depth = self
            .model_node(node)
            .map(|n| {
                self.connectable_inputs(n)
                    .iter()
                    .filter_map(|p| self.linked_source(node, p))
                    .map(|src| self.node_depth(src, memo, visited))
                    .max()
                    .map(|d| d + 1)
                    .unwrap_or(0)
            })
            .unwrap_or(0);
        visited.pop();
        memo.insert(node, depth);
        depth
    }

    /// Compute seed positions: free expr nodes laid left→right by dependency
    /// depth; stacks parked in a right-hand column, stacked vertically; spawn
    /// source contexts get their own leftmost column, since they feed an
    /// Init stack's flow input rather than participating in the expression
    /// graph.
    fn seed_layout(&self) -> (Vec<(WNodeId, WorldPos)>, Vec<(WStackId, WorldPos)>) {
        let mut memo = HashMap::new();
        let mut by_depth: HashMap<u32, Vec<NodeId>> = HashMap::new();
        let mut max_depth = 0u32;
        for node in self.model_nodes() {
            // Modifier members are laid out by their stack, not as free nodes.
            if self.member_of.contains_key(&node.id) {
                continue;
            }
            let d = self.node_depth(node.id, &mut memo, &mut Vec::new());
            max_depth = max_depth.max(d);
            by_depth.entry(d).or_default().push(node.id);
        }

        let mut expr_seed = Vec::new();
        for (depth, ids) in &by_depth {
            for (row, id) in ids.iter().enumerate() {
                let pos = WorldPos::new(*depth as f64 * COL_W + 40.0, row as f64 * ROW_H + 60.0);
                expr_seed.push((wnode(*id), pos));
            }
        }
        for (row, source) in self.effect_graph.sources.iter().enumerate() {
            let pos = WorldPos::new(-COL_W + 40.0, row as f64 * ROW_H + 60.0);
            expr_seed.push((wsource(source.id), pos));
        }

        let stack_x = (max_depth as f64 + 1.0) * COL_W + 120.0;
        let mut stack_seed = Vec::new();
        let mut cursor_y = 60.0;
        for emitter in &self.effect_graph.emitters {
            for stack in &emitter.stacks {
                stack_seed.push((wstack(stack.id.0), WorldPos::new(stack_x, cursor_y)));
                cursor_y += self.estimated_stack_height(stack) + STACK_GAP;
            }
        }
        (expr_seed, stack_seed)
    }

    /// Estimate a stack's rendered height from its members' port counts.
    fn estimated_stack_height(&self, stack: &super::model::GraphStack) -> f64 {
        let mut h = EST_STACK_HEADER + EST_STACK_PAD * 2.0;
        for (i, member) in stack.members.iter().enumerate() {
            if i > 0 {
                h += EST_MEMBER_GAP;
            }
            // Sum each input row plus any editor box it reserves below the label
            // (e.g. an inline vec3/vec4 default), so the estimate tracks the real
            // rendered height and stacks don't seed on top of one another.
            let body = self.model_node(*member).map(|n| {
                let ports = self.input_ports(n);
                let rows = ports.len().max(1) as f64 * EST_ROW_H;
                let boxes: f64 = ports.iter().filter_map(|p| p.expand_height).sum();
                rows + boxes
            });
            h += EST_NODE_HEADER + EST_NODE_BODY_PAD + body.unwrap_or(EST_ROW_H);
        }
        h
    }
}

impl GraphViewer for GraphReader<'_> {
    fn node_ids(&self) -> Vec<WNodeId> {
        self.model_nodes()
            .map(|n| wnode(n.id))
            .chain(self.effect_graph.sources.iter().map(|s| wsource(s.id)))
            .collect()
    }

    fn node(&self, id: WNodeId) -> NodeDesc {
        if let Some(source_id) = SourceId::new(id.get())
            && let Some(source) = self.effect_graph.source(source_id)
        {
            return self.source_node_desc(source);
        }
        let Some(model_id) = NodeId::new(id.get()) else {
            return NodeDesc::new("?");
        };
        let Some(node) = self.model_node(model_id) else {
            return NodeDesc::new("?");
        };
        match &node.payload {
            NodePayload::Expr(e) => {
                let mut out = PortDesc::new(prettify_label("out"));
                if let Some(t) = self.output_type(model_id) {
                    out = out
                        .with_color(port_type_color(t))
                        .with_arity(port_type_arity(t));
                }
                let mut inputs = self.input_ports(node);
                // A property reference shows its current value as a read-only chip
                // so the wired-in value is visible without opening the panel.
                if let ExprNode::Property(pid) = e
                    && let Some(prop) = self.model_property(*pid)
                {
                    inputs.push(
                        PortDesc::new("")
                            .display_value(short_literal(&prop.default.to_wgsl_string())),
                    );
                }
                NodeDesc::new(self.expr_title(e))
                    .with_inputs(inputs)
                    .with_outputs(vec![out])
                    .with_accent(expr_accent(e))
            }
            NodePayload::Modifier(data) => {
                let (title, type_path) = match data {
                    ModifierNodeData::Known { type_path, .. } => (
                        display_name_for_type(base_name(type_path)).into_owned(),
                        type_path,
                    ),
                    ModifierNodeData::Unknown { type_path, .. } => {
                        (format!("{} (?)", base_name(type_path)), type_path)
                    }
                };
                let member = self.member_of.get(&model_id);
                // Stacked members render as grey sections inside their stack's
                // single node frame, so they carry no per-group header accent.
                let mut desc = NodeDesc::new(title)
                    .with_inputs(self.input_ports(node))
                    .closable();
                // An `EmitSpawnEventModifier` in an Update stack gets one event
                // output, the only kind of output any modifier node has; every
                // other modifier (and an Emit modifier outside Update, an
                // invalid mid-edit state) keeps none.
                if is_emit_spawn_event_modifier(type_path)
                    && member.is_some_and(|&(g, _)| g == ModifierGroup::Update)
                {
                    desc = desc.with_outputs(vec![
                        PortDesc::new(prettify_label(EVENT_PORT)).with_color(EVENT_COLOR),
                    ]);
                }
                if let Some(text) = member.and_then(|&(g, i)| self.shadow_warning(g, i)) {
                    desc = desc.with_warning(text);
                }
                desc
            }
        }
    }

    fn links(&self) -> Vec<Link> {
        let mut out = Vec::new();
        for link in self.model_links() {
            let Some(target) = self.model_node(link.to.node) else {
                continue;
            };
            let names = self.connectable_inputs(target);
            let Some(index) = names.iter().position(|n| n.as_ref() == &*link.to.port) else {
                continue;
            };
            out.push(Link {
                from: PortAddr::new(wnode(link.from.node), PortId::output(0)),
                to: PortAddr::new(wnode(link.to.node), PortId::input(index as u16)),
            });
        }
        // Event links render as ordinary links too: an emitter's single event
        // output (index 0) feeding a GPU source's single multiple-link event input
        // (also index 0).
        for link in &self.effect_graph.event_links {
            out.push(Link {
                from: PortAddr::new(wnode(link.node), PortId::output(0)),
                to: PortAddr::new(wsource(link.target), PortId::input(0)),
            });
        }
        out
    }

    fn stacks(&self) -> Vec<StackDesc> {
        self.effect_graph
            .emitters
            .iter()
            .flat_map(|emitter| {
                emitter.stacks.iter().map(|stack| {
                    let members = stack.members.iter().map(|m| wnode(*m)).collect();
                    let mut desc = StackDesc::new(wstack(stack.id.0), stack.group.label())
                        .with_members(members)
                        .with_accent(stack_accent(group_order(stack.group)));
                    // Only an Init stack accepts a spawn source's flow output.
                    if stack.group == ModifierGroup::Init {
                        desc = desc.with_flow_input(true);
                    }
                    desc
                })
            })
            .collect()
    }

    fn stack_links(&self) -> Vec<StackLink> {
        let mut links = Vec::new();
        for emitter in &self.effect_graph.emitters {
            let id_of = |group: ModifierGroup| emitter.stack(group).map(|s| wstack(s.id.0));
            if let (Some(init), Some(update)) =
                (id_of(ModifierGroup::Init), id_of(ModifierGroup::Update))
            {
                links.push(StackLink {
                    from: init,
                    to: update,
                });
            }
            if let (Some(update), Some(render)) =
                (id_of(ModifierGroup::Update), id_of(ModifierGroup::Render))
            {
                links.push(StackLink {
                    from: update,
                    to: render,
                });
            }
        }
        links
    }

    fn flow_links(&self) -> Vec<FlowLink> {
        self.effect_graph
            .source_links
            .iter()
            .filter_map(|link| {
                let init = self
                    .effect_graph
                    .emitter(link.emitter)?
                    .stack(ModifierGroup::Init)?;
                Some(FlowLink {
                    from: wsource(link.source),
                    to: wstack(init.id.0),
                })
            })
            .collect()
    }

    fn validate_link(&self, from: PortAddr, to: PortAddr) -> LinkVerdict {
        if from.node == to.node {
            return Err("a node can't feed its own input".into());
        }
        let Some(from_id) = NodeId::new(from.node.get()) else {
            return Ok(());
        };

        // An event link: `to` addresses a spawn source's event input rather
        // than an ordinary node's operand port.
        if let Some(target) = SourceId::new(to.node.get())
            && self.effect_graph.source(target).is_some()
        {
            return graph_validation::event_link_is_valid(self.effect_graph, from_id, target)
                .map_err(Cow::Owned);
        }

        let Some(to_id) = NodeId::new(to.node.get()) else {
            return Ok(());
        };
        if self.would_cycle(from_id, to_id) {
            return Err("would create a cycle".into());
        }
        // Stacked modifiers run in a fixed order; values only flow forward.
        if let (Some(a), Some(b)) = (self.exec_rank(from_id), self.exec_rank(to_id))
            && a > b
        {
            return Err("a later stage can't feed an earlier one".into());
        }
        // Type compatibility, with a few implicit casts.
        let from_ty = self.output_type(from_id);
        let to_ty = self
            .model_node(to_id)
            .and_then(|n| {
                self.connectable_inputs(n)
                    .get(to.port.index as usize)
                    .cloned()
            })
            .and_then(|name| self.operand_type(to_id, &name));
        match (from_ty, to_ty) {
            (Some(ft), Some(tt)) => cast_verdict(ft, tt),
            _ => Ok(()),
        }
    }

    fn validate_flow_link(&self, from: WNodeId, to: WStackId) -> LinkVerdict {
        let Some(source_id) =
            SourceId::new(from.get()).filter(|&id| self.effect_graph.source(id).is_some())
        else {
            return Err("not a spawn source".into());
        };
        let Some(emitter_id) = self.emitter_of_stack(to) else {
            return Err("unknown stack".into());
        };
        let Some(init) = self
            .effect_graph
            .emitter(emitter_id)
            .and_then(|e| e.stack(ModifierGroup::Init))
        else {
            return Err("emitter has no Init stack".into());
        };
        if wstack(init.id.0) != to {
            return Err("a spawn source can only feed an emitter's Init stack".into());
        }
        graph_validation::source_link_is_valid(self.effect_graph, source_id, emitter_id)
            .map_err(Cow::Owned)
    }
}

impl GraphReader<'_> {
    /// Tooltip text for a shadowed modifier at `(group, idx)`, or `None` when
    /// it isn't shadowed. Mirrors the Emitter panel's wording.
    fn shadow_warning(&self, group: ModifierGroup, idx: usize) -> Option<String> {
        let hits = self.shadowed.get(&(group, idx))?;
        let mut tip = String::from(
            "This modifier has no emitter: every attribute it writes is \
             overwritten by a later modifier in the same group.\n",
        );
        for (attr, j) in hits {
            tip.push_str(&format!("  • {} → overwritten by #{}\n", attr.name(), j));
        }
        tip.truncate(tip.trim_end().len());
        Some(tip)
    }

    /// Describe a spawn source context as a widget node.
    ///
    /// Both kinds expose a flow-output pin feeding an emitter's Init stack; a
    /// GPU event source additionally exposes one event input accepting links
    /// from by every Update-stack `EmitSpawnEventModifier` that targets it
    /// (see this module's doc comment). A CPU spawner exposes all of its
    /// settings as editable display rows (see [`cpu_spawner_ports`] /
    /// [`cpu_spawner_chip`]), committed via
    /// [`crate::edits::EditKind::SetCpuSpawnerSettings`]. A `count`,
    /// `spawn_duration`, or `period` authored as a random range (not
    /// [`bevy_hanabi::CpuValue::Single`]) shows its range but has no inline
    /// editor, so scrubbing never silently collapses it to a scalar.
    fn source_node_desc(&self, source: &SourceContext) -> NodeDesc {
        match &source.kind {
            SourceKind::CpuSpawner { settings } => NodeDesc::new("CPU Spawner")
                .with_flow_output(true)
                .with_inputs(cpu_spawner_ports(settings))
                .closable(),
            SourceKind::GpuEvent => {
                let mut desc = NodeDesc::new("GPU Event")
                    .with_flow_output(true)
                    .with_inputs(vec![
                        PortDesc::new(prettify_label(EVENT_PORT))
                            .with_color(EVENT_COLOR)
                            .with_multiple_links(true),
                    ])
                    .closable();
                if let Some(warning) = self.source_warnings.get(&source.id) {
                    desc = desc.with_warning(warning.clone());
                }
                desc
            }
        }
    }

    /// Short, human-readable title for an expression node.
    fn expr_title(&self, expr: &ExprNode) -> String {
        match expr {
            ExprNode::Literal(v) => v.to_wgsl_string(),
            ExprNode::Attribute(_) | ExprNode::Age { .. } => "Get Attribute".to_string(),
            ExprNode::ParentAttribute(a) => {
                format!("parent.{}", a.name())
            }
            ExprNode::Property(pid) => self
                .model_property(*pid)
                .map(|p| format!("${}", p.name))
                .unwrap_or_else(|| "$prop".to_string()),
            ExprNode::BuiltIn(op) => op.to_wgsl_string(),
            ExprNode::Unary(op) => format!("{op:?}"),
            ExprNode::Binary(op) => format!("{op:?}"),
            ExprNode::Ternary(op) => format!("{op:?}"),
            ExprNode::Cast(_) => "Cast".to_string(),
            ExprNode::Image(_) => "Image".to_string(),
            ExprNode::TextureSample1d => "Sample Texture 1D".to_string(),
            ExprNode::TextureSample2d => "Sample Texture 2D".to_string(),
            ExprNode::TextureSample3d => "Sample Texture 3D".to_string(),
            ExprNode::TextureLoad1d => "Load Texture 1D".to_string(),
            ExprNode::TextureLoad2d => "Load Texture 2D".to_string(),
            ExprNode::TextureLoad3d => "Load Texture 3D".to_string(),
            ExprNode::SelectImage { .. } => "Select Image".to_string(),
        }
    }
}

/// Map a model node id to the widget's node id (both one-based `NonZeroU32`).
fn wnode(id: NodeId) -> WNodeId {
    WNodeId::new(id.get()).expect("node ids are non-zero")
}

/// Map a model source id to the widget's node id: a spawn source context
/// renders as an ordinary widget node (see this module's doc comment).
fn wsource(id: SourceId) -> WNodeId {
    WNodeId::new(id.get()).expect("source ids are non-zero")
}

/// Map a model stack id to the widget's stack id.
fn wstack(id: std::num::NonZeroU32) -> WStackId {
    WStackId::new(id.get()).expect("stack ids are non-zero")
}

/// Resolve the [`ModifierGroup`] for a widget stack id by matching it against
/// the document's stacks, across every emitter. Returns `None` if the id has no
/// corresponding stack (e.g. a stale widget id after a structural change).
pub fn group_of_widget_stack(effect_graph: &EffectGraph, stack: WStackId) -> Option<ModifierGroup> {
    effect_graph
        .emitters
        .iter()
        .flat_map(|e| &e.stacks)
        .find(|s| s.id.0.get() == stack.get())
        .map(|s| s.group)
}

/// Whether a modifier's type path names the known `EmitSpawnEventModifier`,
/// the only modifier kind that gets an event output port (and only when it
/// sits in an Update stack — see [`GraphReader::node`]).
fn is_emit_spawn_event_modifier(type_path: &str) -> bool {
    type_path == EmitSpawnEventModifier::type_path()
}

/// Execution order of a modifier group: Init < Update < Render.
fn group_order(group: ModifierGroup) -> u32 {
    match group {
        ModifierGroup::Init => 0,
        ModifierGroup::Update => 1,
        ModifierGroup::Render => 2,
    }
}

/// The last path segment of a type path, ignoring generics.
fn base_name(path: &str) -> &str {
    let head = path.split('<').next().unwrap_or(path);
    head.rsplit("::").next().unwrap_or(head)
}

/// Accent color for an expression node, by variant family.
fn expr_accent(expr: &ExprNode) -> Color32 {
    match expr {
        ExprNode::Literal(_) => Color32::from_rgb(90, 130, 80),
        ExprNode::Property(_) => Color32::from_rgb(150, 120, 60),
        ExprNode::Attribute(_) | ExprNode::Age { .. } | ExprNode::ParentAttribute(_) => {
            Color32::from_rgb(70, 110, 160)
        }

        ExprNode::BuiltIn(_) => Color32::from_rgb(60, 130, 140),
        ExprNode::Unary(_) | ExprNode::Binary(_) | ExprNode::Ternary(_) => {
            Color32::from_rgb(150, 110, 60)
        }
        ExprNode::Cast(_) => Color32::from_rgb(120, 90, 150),
        ExprNode::Image(_)
        | ExprNode::TextureSample1d
        | ExprNode::TextureSample2d
        | ExprNode::TextureSample3d
        | ExprNode::TextureLoad1d
        | ExprNode::TextureLoad2d
        | ExprNode::TextureLoad3d
        | ExprNode::SelectImage { .. } => Color32::from_rgb(150, 80, 110),
    }
}

/// The convenience options of an Age source node, if it is one.
///
/// The legacy raw `Attribute(AGE)` representation behaves as both options
/// disabled, preserving compatible graph files while exposing the same UI.
fn node_age_options(payload: &NodePayload) -> Option<(bool, bool)> {
    match payload {
        NodePayload::Expr(ExprNode::Attribute(attribute)) if *attribute == Attribute::AGE => {
            Some((false, false))
        }

        NodePayload::Expr(ExprNode::Age {
            normalized,
            clamped,
        }) => Some((*normalized, *clamped)),
        _ => None,
    }
}

/// The selected attribute of a Get Attribute expression node, if it is one.
fn node_expression_attribute(payload: &NodePayload) -> Option<Attribute> {
    match payload {
        NodePayload::Expr(ExprNode::Attribute(attribute)) => Some(*attribute),
        NodePayload::Expr(ExprNode::Age { .. }) => Some(Attribute::AGE),
        _ => None,
    }
}

/// Frame accent for a modifier stack.
fn stack_accent(group: u32) -> Color32 {
    match group {
        0 => Color32::from_rgb(80, 60, 90),
        1 => Color32::from_rgb(68, 68, 90),
        _ => Color32::from_rgb(55, 75, 90),
    }
}

/// The type carried by a graph port: an ordinary value type, or the editor-only
/// "image" pseudo-type produced by a texture reference.
///
/// [`Image`] exists only in the editor's type system; at bake time it lowers to
/// a `u32` slot index. In the editor it is opaque: image ports connect only to
/// other image ports and never cast to or from a value type.
///
/// [`Image`]: PortType::Image
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortType {
    Value(ValueType),
    Image,
}

/// Pin color for a value type, so compatible ports share a hue.
fn value_type_color(vt: ValueType) -> Color32 {
    const FLOAT: Color32 = Color32::from_rgb(0x5A, 0xB0, 0xE6);
    const INT: Color32 = Color32::from_rgb(0x8C, 0xCB, 0x5E);
    const UINT: Color32 = Color32::from_rgb(0xB9, 0x8C, 0xE6);
    const BOOL: Color32 = Color32::from_rgb(0xE0, 0x6C, 0x6C);
    match vt {
        ValueType::Scalar(ScalarType::Float) => FLOAT,
        ValueType::Scalar(ScalarType::Int) => INT,
        ValueType::Scalar(ScalarType::Uint) => UINT,
        ValueType::Scalar(ScalarType::Bool) => BOOL,
        ValueType::Vector(v) => match v {
            VectorType::VEC2F | VectorType::VEC3F | VectorType::VEC4F => FLOAT,
            VectorType::VEC2I | VectorType::VEC3I | VectorType::VEC4I => INT,
            VectorType::VEC2U | VectorType::VEC3U | VectorType::VEC4U => UINT,
            _ => Color32::GRAY,
        },
        ValueType::Matrix(_) => Color32::from_rgb(0xE0, 0xB0, 0x6C),
        _ => Color32::GRAY,
    }
}

/// Pin color for a port type, so compatible ports share a hue.
fn port_type_color(ty: PortType) -> Color32 {
    const IMAGE: Color32 = Color32::from_rgb(0xE6, 0x8C, 0xB9);
    match ty {
        PortType::Value(vt) => value_type_color(vt),
        PortType::Image => IMAGE,
    }
}

/// Pin color for an event port (an emitter's output, a GPU source's input):
/// a distinct hue from every value/image port type, since an event link never
/// mixes with either.
const EVENT_COLOR: Color32 = Color32::from_rgb(0xE6, 0xC8, 0x5A);

/// Number of pin pips to draw for a port type.
///
/// The node-graph widget draws exactly this many squares. A single pip on
/// every scalar port would be noise, so we apply an `arity < 2` rule here:
/// only multi-component (vector) ports are marked, reporting their component
/// count; everything else reports `0`.
fn port_type_arity(ty: PortType) -> u8 {
    let components = match ty {
        PortType::Value(ValueType::Scalar(_)) => 1,
        PortType::Value(ValueType::Vector(v)) => v.count() as u8,
        _ => 0,
    };
    if components < 2 { 0 } else { components }
}

/// Whether an output of type `from` may feed an input of type `to`.
///
/// Identical value types connect directly and a scalar splats into a vector of
/// the same scalar (see [`value_cast_verdict`]). The [`Image`] pseudo-type is
/// opaque: it connects only to another image port, and never casts to or from a
/// value type.
///
/// [`Image`]: PortType::Image
fn cast_verdict(from: PortType, to: PortType) -> LinkVerdict {
    match (from, to) {
        (PortType::Value(f), PortType::Value(t)) => value_cast_verdict(f, t),
        (PortType::Image, PortType::Image) => Ok(()),
        (PortType::Image, PortType::Value(_)) => {
            Err("a texture image can only feed an image input".into())
        }
        (PortType::Value(_), PortType::Image) => Err("an image input takes only an image".into()),
    }
}

/// Whether an output of value type `from` may feed an input of value type `to`:
/// identical types connect directly, a scalar splats into a vector of the same
/// scalar.
fn value_cast_verdict(from: ValueType, to: ValueType) -> LinkVerdict {
    if from == to {
        return Ok(());
    }
    if let (ValueType::Scalar(s), ValueType::Vector(v)) = (from, to)
        && v.elem_type() == s
    {
        return Ok(());
    }
    Err(format!("no implicit cast {} → {}", type_short(from), type_short(to)).into())
}

/// Whether a value of type `from` connects to an input of type `to`, directly
/// or through an implicit cast (see [`cast_verdict`]).
pub fn can_cast(from: PortType, to: PortType) -> bool {
    cast_verdict(from, to).is_ok()
}

/// WGSL-flavoured short name for a value type, for rejection messages.
fn type_short(vt: ValueType) -> &'static str {
    match vt {
        ValueType::Scalar(ScalarType::Float) => "f32",
        ValueType::Scalar(ScalarType::Int) => "i32",
        ValueType::Scalar(ScalarType::Uint) => "u32",
        ValueType::Scalar(ScalarType::Bool) => "bool",
        ValueType::Vector(v) => match v {
            VectorType::VEC2F => "vec2<f32>",
            VectorType::VEC3F => "vec3<f32>",
            VectorType::VEC4F => "vec4<f32>",
            VectorType::VEC2I => "vec2<i32>",
            VectorType::VEC3I => "vec3<i32>",
            VectorType::VEC4I => "vec4<i32>",
            VectorType::VEC2U => "vec2<u32>",
            VectorType::VEC3U => "vec3<u32>",
            VectorType::VEC4U => "vec4<u32>",
            _ => "vecN",
        },
        ValueType::Matrix(_) => "matrix",
        _ => "?",
    }
}

/// Reserved world-space height of the single-line inline vector editor.
const VECTOR_EDITOR_ROW_H: f64 = 22.0;

/// Box height for an inline vector editor, or `None` for non-vec3/vec4 values.
///
/// Vec3 and Vec4 literals are edited as a row of per-component scrubbers below
/// the label; other value kinds keep their single-line chip.
fn vector_editor_height(value: &Value) -> Option<f64> {
    let Value::Vector(vv) = value else {
        return None;
    };
    match vv.vector_type() {
        VectorType::VEC3F | VectorType::VEC4F => Some(VECTOR_EDITOR_ROW_H),
        _ => None,
    }
}

/// Convert a snake_case port name into a Title Case display label.
///
/// Splits on underscores and capitalizes each word's first character, so
/// `some_value` reads as `Some Value`. An empty name yields an empty label.
fn prettify_label(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for word in name.split('_').filter(|w| !w.is_empty()) {
        if !out.is_empty() {
            out.push(' ');
        }
        let mut chars = word.chars();
        if let Some(first) = chars.next() {
            out.extend(first.to_uppercase());
            out.push_str(chars.as_str());
        }
    }
    out
}

/// Tidy a wgsl literal for compact inline display and truncate.
fn short_literal(value: &str) -> String {
    let trimmed = match value.find('(') {
        Some(open) if value[..open].contains('<') || value[..open].starts_with("vec") => {
            &value[open..]
        }
        _ => value,
    };
    let cleaned = trimmed
        .replace(".,", ",")
        .replace(".)", ")")
        .trim_end_matches('.')
        .to_string();
    if cleaned.chars().count() > CHIP_MAX {
        let head: String = cleaned.chars().take(CHIP_MAX - 1).collect();
        format!("{head}…")
    } else {
        cleaned
    }
}

/// Compact display string for a modifier's non-expr config value.
fn format_config(value: &EditValue) -> String {
    match value {
        EditValue::Bool(b) => b.to_string(),
        EditValue::U32(n) => n.to_string(),
        EditValue::Scalar(v) => short_literal(&v.to_wgsl_string()),
        EditValue::UVec2(u) => format!("({}, {})", u.x, u.y),
        EditValue::Color(c) => short_literal(&Value::from(*c).to_wgsl_string()),
        EditValue::Attribute(a) => a.name().to_string(),
        EditValue::CpuVec3(cv) => format_cpu_vec3(cv),
        EditValue::CpuVec4(cv) => format_cpu_vec4(cv),
        EditValue::Gradient3(g) => match g {
            GradientVec3::Analytical(_) => "gradient".to_string(),
            GradientVec3::Lut(t) => format_texture(t),
        },
        EditValue::Gradient4(g) => match g {
            GradientVec4::Analytical(_) => "gradient".to_string(),
            GradientVec4::Lut(t) => format_texture(t),
        },
        EditValue::Texture(t) => format_texture(t),
        EditValue::Enum { variant, .. } => variant.to_string(),
        EditValue::Flags { type_path, bits } => format_flags(type_path, *bits),
        EditValue::Raw(_) => "…".to_string(),
    }
}

/// Build a `Vec3` analytical gradient from uniform-scalar curve keys.
///
/// Each key value is splatted to all three components, matching how the
/// curve editor edits size over lifetime as a single scalar track.
pub fn keys_to_gradient3(keys: &[(f32, f32)]) -> EditValue {
    let mut g = Gradient::new();
    for (ratio, v) in keys {
        g.add_key(*ratio, Vec3::splat(*v));
    }
    EditValue::Gradient3(GradientVec3::Analytical(g))
}

/// Build a `Vec4` analytical gradient from color-stop keys.
pub fn keys_to_gradient4(keys: &[(f32, [f32; 4])]) -> EditValue {
    let mut g = Gradient::new();
    for (ratio, c) in keys {
        g.add_key(*ratio, Vec4::from_array(*c));
    }
    EditValue::Gradient4(GradientVec4::Analytical(g))
}

/// Compact display of a bitflags mask as its active flag names (e.g. `R|G|B`).
///
/// Falls back to hex for an unknown flags type, and shows `none` when no bit is
/// set.
fn format_flags(type_path: &str, bits: u64) -> String {
    let defs = flag_defs(type_path);
    if defs.is_empty() {
        return format!("0x{bits:X}");
    }
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
}

fn format_cpu_vec3(cv: &bevy_hanabi::CpuValue<bevy::math::Vec3>) -> String {
    match cv {
        bevy_hanabi::CpuValue::Single(v) => short_literal(&Value::from(*v).to_wgsl_string()),
        bevy_hanabi::CpuValue::Uniform((a, b)) => format!(
            "{} … {}",
            short_literal(&Value::from(*a).to_wgsl_string()),
            short_literal(&Value::from(*b).to_wgsl_string())
        ),
        _ => "?".to_string(),
    }
}

fn format_cpu_vec4(cv: &bevy_hanabi::CpuValue<bevy::math::Vec4>) -> String {
    match cv {
        bevy_hanabi::CpuValue::Single(v) => short_literal(&Value::from(*v).to_wgsl_string()),
        bevy_hanabi::CpuValue::Uniform((a, b)) => format!(
            "{} … {}",
            short_literal(&Value::from(*a).to_wgsl_string()),
            short_literal(&Value::from(*b).to_wgsl_string())
        ),
        _ => "?".to_string(),
    }
}

fn format_cpu_f32(cv: &bevy_hanabi::CpuValue<f32>) -> String {
    match cv {
        bevy_hanabi::CpuValue::Single(v) => short_literal(&Value::from(*v).to_wgsl_string()),
        bevy_hanabi::CpuValue::Uniform((a, b)) => format!(
            "{} … {}",
            short_literal(&Value::from(*a).to_wgsl_string()),
            short_literal(&Value::from(*b).to_wgsl_string())
        ),
        _ => "?".to_string(),
    }
}

/// Input rows for all CPU spawner settings, in a fixed order.
fn cpu_spawner_ports(settings: &SpawnerSettings) -> Vec<PortDesc> {
    vec![
        PortDesc::new("Count").display_value(format_cpu_f32(&settings.count())),
        PortDesc::new("Spawn duration (s)")
            .display_value(format_cpu_f32(&settings.spawn_duration())),
        PortDesc::new("Period (s)").display_value(format_cpu_f32(&settings.period())),
        PortDesc::new("Cycle count").display_value(settings.cycle_count().to_string()),
        // A bool renders as a compact checkbox overlaid by the panel, like a
        // modifier's bool config field; the chip itself carries no text.
        PortDesc::new("Starts active").display_value(""),
        PortDesc::new("Emit on start").display_value(""),
    ]
}

/// Resolve a CPU spawner source node's clicked row index to its editable
/// chip, by the fixed order [`cpu_spawner_ports`] builds.
///
/// Scalar-or-range values are editable only when authored as a single scalar
/// ([`bevy_hanabi::CpuValue::Single`]); a random range has no editor here.
fn cpu_spawner_chip(
    source: SourceId,
    settings: &SpawnerSettings,
    idx: usize,
) -> Option<EditableChip> {
    match idx {
        0 => match settings.count() {
            CpuValue::Single(value) => Some(EditableChip::CpuSpawnerCount { source, value }),
            _ => None,
        },
        1 => match settings.spawn_duration() {
            CpuValue::Single(value) => {
                Some(EditableChip::CpuSpawnerSpawnDuration { source, value })
            }
            _ => None,
        },
        2 => match settings.period() {
            CpuValue::Single(value) => Some(EditableChip::CpuSpawnerPeriod { source, value }),
            _ => None,
        },
        3 => Some(EditableChip::CpuSpawnerCycleCount {
            source,
            value: settings.cycle_count(),
        }),
        4 => Some(EditableChip::CpuSpawnerStartsActive {
            source,
            value: settings.starts_active(),
        }),
        5 => Some(EditableChip::CpuSpawnerEmitsOnStart {
            source,
            value: settings.emits_on_start(),
        }),
        _ => None,
    }
}

fn format_texture(t: &TextureValue) -> String {
    match t {
        TextureValue::Asset(path) => path.to_string(),
        TextureValue::Slot { name } => format!("[{name}]"),
    }
}
