//! Baking an [`EffectGraph`] into runtime [`bevy_hanabi::EffectAsset`]s.
//!
//! The edit model expresses wiring with explicit [`GraphLink`]s and per-input
//! inline defaults; the runtime model wires expressions through `ExprHandle`
//! arena indices inside a [`Module`]. Baking rebuilds that arena: it walks the
//! expression nodes in dependency order, materializing each as a `Module`
//! expression and recording the resulting handle, then resolves every operand
//! to either a linked source node's handle or its inline-default literal.
//!
//! Properties bake per [`PropertyDef::exposed`]: an exposed property becomes a
//! real `Module` property (settable per instance at runtime), while an
//! edit-only property is inlined to a literal constant at each reference, so it
//! has no runtime cost.
//!
//! Each emitter's [`SpawnerSettings`] belongs to its linked CPU spawn-source
//! node, while each `EmitSpawnEventModifier::child_index` is derived from
//! inter-emitter GPU-event links. Neither value lives inside the individual
//! [`EmitterGraph`], so the per-emitter baking functions ([`bake_emitter`] and
//! friends) take them as explicit parameters. [`bake_effect`] resolves those
//! inputs from the complete [`EffectGraph`] and bakes every emitter; [`bake`]
//! is a strict convenience for a single CPU-spawned emitter.
//!
//! This module covers expression and property baking. Modifier instantiation
//! and final [`EffectAsset`] assembly build on the `NodeId → ExprHandle` map it
//! produces.
//!
//! [`GraphLink`]: crate::model::GraphLink
//! [`SpawnerSettings`]: bevy_hanabi::SpawnerSettings

use std::collections::HashMap;

use bevy::{
    asset::AssetPath,
    math::{UVec2, Vec2, Vec3, Vec4},
    reflect::{
        PartialReflect, Reflect, ReflectMut, TypePath, TypeRegistry,
        enums::{DynamicEnum, DynamicVariant},
    },
};
use bevy_hanabi::{
    BoxedModifier, EffectAsset, EffectShaderSources, EmitSpawnEventModifier, Expr, ExprHandle,
    ModifierContext, Module, ReflectModifier, SetPositionCircleModifier, SetVelocityCircleModifier,
    SetVelocityTangentModifier, SlotDimension, SpawnerSettings, TangentAccelModifier, Value,
    graph::expr::{PropertyHandle, TextureSampleExpr},
};

use super::{
    model::{
        EditValue, EffectGraph, EmitterGraph, EmitterId, ExprNode, GradientVec3, GradientVec4,
        ImageBinding, ModifierNodeData, NodeId, NodePayload, PortRef, PropertyDef, PropertyId,
        SharedStr, SlotId, SourceId, SourceKind,
    },
    schema::{FieldRole, modifier_schema},
    validation,
};
use crate::ModifierGroup;

/// What a [`BakeError`] is attributed to.
///
/// Lets the UI surface the error in context — e.g. highlight the offending node
/// or property, or show a graph-level banner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BakeSubject {
    /// A specific graph node (e.g. an expression with a missing operand).
    Node(NodeId),
    /// A specific user property (e.g. an exposed-name conflict).
    Property(PropertyId),
    /// A specific emitter within an [`EffectGraph`].
    Emitter(EmitterId),
    /// A specific effect-level spawn source context.
    Source(SourceId),
    /// The graph as a whole, with no single element to blame.
    Graph,
}

/// A problem encountered while baking, attributed to the element to blame.
///
/// Baking collects every error it can rather than stopping at the first.
#[derive(Debug, Clone, PartialEq)]
pub struct BakeError {
    pub subject: BakeSubject,
    pub message: String,
}

impl BakeError {
    fn node(node: NodeId, message: impl Into<String>) -> Self {
        Self {
            subject: BakeSubject::Node(node),
            message: message.into(),
        }
    }

    fn property(id: PropertyId, message: impl Into<String>) -> Self {
        Self {
            subject: BakeSubject::Property(id),
            message: message.into(),
        }
    }

    fn emitter(id: EmitterId, message: impl Into<String>) -> Self {
        Self {
            subject: BakeSubject::Emitter(id),
            message: message.into(),
        }
    }

    fn source(id: SourceId, message: impl Into<String>) -> Self {
        Self {
            subject: BakeSubject::Source(id),
            message: message.into(),
        }
    }

    #[allow(dead_code)]
    fn graph(message: impl Into<String>) -> Self {
        Self {
            subject: BakeSubject::Graph,
            message: message.into(),
        }
    }
}

/// Resolved property bindings produced by [`bake_properties`].
///
/// The runtime handle of each exposed property, plus every property's
/// definition indexed by stable id (used to resolve [`ExprNode::Property`]
/// references).
struct PropertyBindings<'a> {
    handles: HashMap<PropertyId, PropertyHandle>,
    defs: HashMap<PropertyId, &'a PropertyDef>,
}

/// Register exposed properties and index every property by its stable id.
///
/// Properties are referenced by id, not name, so display names are free to
/// collide. The one name constraint is on **exposed** properties: each becomes
/// a runtime `Module` property keyed by name, so a name shared by two exposed
/// properties is an inconsistency that blocks baking. It is reported as a
/// [`BakeError`] (never a panic — `Module::add_property` would panic on a
/// duplicate name, so the second add is skipped) so the author can fix it.
fn bake_properties<'a>(
    graph: &'a EmitterGraph,
    module: &mut Module,
    errors: &mut Vec<BakeError>,
) -> PropertyBindings<'a> {
    let mut handles = HashMap::new();
    let mut defs = HashMap::with_capacity(graph.properties.len());
    let mut exposed_names: HashMap<&str, PropertyId> = HashMap::new();
    for prop in &graph.properties {
        if defs.insert(prop.id, prop).is_some() {
            // Two properties sharing an id is a structural inconsistency (ids are
            // unique by construction); references to it would be ambiguous.
            errors.push(BakeError::property(
                prop.id,
                format!("duplicate property id {}", prop.id.get()),
            ));
            continue;
        }
        if prop.exposed {
            let name: &str = &prop.name;
            if exposed_names.contains_key(name) {
                errors.push(BakeError::property(
                    prop.id,
                    format!("two exposed properties share the name '{name}'; rename one to bake"),
                ));
                continue;
            }
            exposed_names.insert(name, prop.id);
            let handle = module.add_property(name, prop.default);
            handles.insert(prop.id, handle);
        }
    }
    PropertyBindings { handles, defs }
}

/// The graph origin of a baked `Expr::Literal`.
///
/// Used to map a value tweak to the promotable module expression it produced.
/// Lets the live-tweak fast path upload a new value through the proxy property
/// bound to that expression instead of re-baking the whole graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LiteralSite {
    /// A literal expression node, identified by its node id.
    Node(NodeId),
    /// An inline default on an input port (modifier or operator), identified by
    /// the owning node and the port name.
    Input { node: NodeId, port: SharedStr },
}

/// Provenance from a bake: every baked literal mapped to its graph site.
///
/// Keyed into the baked module's expression arena.
pub type LiteralSites = HashMap<LiteralSite, ExprHandle>;

/// One resolved texture slot produced by a bake.
///
/// Identifies what the renderer should bind to a baked [`Module`] texture slot.
/// Slots are ordered by sampling index, matching the module's texture layout.
#[derive(Debug, Clone, PartialEq)]
pub enum PlannedImage {
    /// A pinned asset, loaded by path and bound to every instance.
    Asset(AssetPath<'static>),
    /// A host-supplied named slot, filled per instance through
    /// [`bevy_hanabi::EffectMaterial`]; shown with a placeholder in preview.
    Runtime(SharedStr),
    /// No image chosen yet; shown with a placeholder until one is bound.
    Unbound,
}

/// The ordered texture bindings of a bake.
///
/// Slot `i` of the baked [`Module`] (and entry `i` of the host's
/// [`bevy_hanabi::EffectMaterial`] images) binds to `slots[i]`.
pub type TexturePlan = Vec<PlannedImage>;

/// Side outputs of a bake beyond the [`EffectAsset`] itself.
///
/// Bundles the literal provenance (driving the live value-tweak fast path) and
/// the texture plan (driving material wiring), so callers thread one value.
#[derive(Debug, Clone, Default)]
pub struct BakeProvenance {
    pub literal_sites: LiteralSites,
    pub texture_plan: TexturePlan,
}

/// Expression-node baking context.
///
/// Holds the graph, the property bindings, the `Module` under construction, and
/// the running `NodeId → ExprHandle` cache.
struct ExprBaker<'a, 'm> {
    graph: &'a EmitterGraph,
    props: &'a PropertyBindings<'a>,
    module: &'m mut Module,
    handles: HashMap<NodeId, ExprHandle>,
    /// Maps each baked literal back to the graph site it came from, so a value
    /// tweak can target the corresponding module expression without re-baking.
    literal_sites: HashMap<LiteralSite, ExprHandle>,
    /// Nodes on the current DFS stack, for cycle detection.
    visiting: Vec<NodeId>,
    /// Resolved texture slots, ordered by sampling index — mirrors the order of
    /// `Module::add_texture_slot` calls so it lines up with the baked layout.
    texture_plan: TexturePlan,
    /// Slot names already handed to the module, used to keep them unique.
    used_slot_names: std::collections::HashSet<String>,
    /// Sampling index reserved for each host-supplied [`SlotId`].
    registry_slots: HashMap<SlotId, usize>,
    /// Sampling index allocated for each pinned asset path.
    pinned_asset_slots: HashMap<AssetPath<'static>, usize>,
    /// Slot allocated for each [`ExprNode::Image`] node, so fan-out from one
    /// image source shares a single slot.
    image_node_slots: HashMap<NodeId, usize>,
}

impl<'a, 'm> ExprBaker<'a, 'm> {
    /// Build a baker, reserving a texture slot for every host-supplied slot.
    ///
    /// Host-supplied (named) slots occupy the leading sampling indices in their
    /// authored order, the stable ABI a host game targets; asset-bound and
    /// inline images auto-allocate after them as they are encountered.
    fn new(
        graph: &'a EmitterGraph,
        props: &'a PropertyBindings<'a>,
        module: &'m mut Module,
    ) -> Self {
        let mut baker = Self {
            graph,
            props,
            module,
            handles: HashMap::new(),
            literal_sites: HashMap::new(),
            visiting: Vec::new(),
            texture_plan: Vec::new(),
            used_slot_names: std::collections::HashSet::new(),
            registry_slots: HashMap::new(),
            pinned_asset_slots: HashMap::new(),
            image_node_slots: HashMap::new(),
        };
        for slot in &graph.texture_slots {
            let index = baker.alloc_slot(&slot.name, PlannedImage::Runtime(slot.name.clone()));
            baker.registry_slots.insert(slot.id, index);
        }
        baker
    }

    /// Resolve a node to its `ExprHandle`, baking it on first visit.
    ///
    /// Bakes the node and its operands on first visit, caching the result;
    /// returns `None` once an error has been recorded for this subtree.
    fn resolve(&mut self, node_id: NodeId, errors: &mut Vec<BakeError>) -> Option<ExprHandle> {
        if let Some(h) = self.handles.get(&node_id) {
            return Some(*h);
        }
        if self.visiting.contains(&node_id) {
            errors.push(BakeError::node(node_id, "expression cycle"));
            return None;
        }

        let node = self.graph.node(node_id).or_else(|| {
            errors.push(BakeError::node(
                node_id,
                format!("link references missing node {}", node_id.get()),
            ));
            None
        })?;
        let NodePayload::Expr(expr) = &node.payload else {
            errors.push(BakeError::node(
                node_id,
                "expected an expression node as a link source",
            ));
            return None;
        };

        self.visiting.push(node_id);
        let handle = self.bake_expr(node_id, expr, errors);
        self.visiting.pop();

        if let Some(h) = handle {
            self.handles.insert(node_id, h);
        }
        handle
    }

    /// Bake one expression node, resolving its operand ports first.
    fn bake_expr(
        &mut self,
        node_id: NodeId,
        expr: &ExprNode,
        errors: &mut Vec<BakeError>,
    ) -> Option<ExprHandle> {
        let handle = match expr {
            ExprNode::Literal(v) => {
                let h = self.module.lit(*v);
                self.literal_sites.insert(LiteralSite::Node(node_id), h);
                h
            }
            ExprNode::Property(id) => self.bake_property_ref(node_id, *id, errors)?,
            ExprNode::Attribute(a) => self.module.attr(*a),
            ExprNode::ParentAttribute(a) => self.module.parent_attr(*a),
            ExprNode::BuiltIn(op) => self.module.builtin(*op),
            ExprNode::Unary(op) => {
                let inner = self.operand(node_id, "in", errors)?;
                self.module.unary(*op, inner)
            }
            ExprNode::Binary(op) => {
                let lhs = self.operand(node_id, "lhs", errors)?;
                let rhs = self.operand(node_id, "rhs", errors)?;
                self.module.binary(*op, lhs, rhs)
            }
            ExprNode::Ternary(op) => {
                let a = self.operand(node_id, "a", errors)?;
                let b = self.operand(node_id, "b", errors)?;
                let c = self.operand(node_id, "c", errors)?;
                self.module.ternary(*op, a, b, c)
            }
            ExprNode::Cast(ty) => {
                let inner = self.operand(node_id, "in", errors)?;
                self.module.cast(inner, *ty)
            }
            // An image source has no standalone runtime expression: every
            // consumer resolves it to a constant slot index directly (see
            // `resolve_image_slot`). A handle is still returned so the node can
            // participate in the arena, but nothing references it.
            ExprNode::Image(_) | ExprNode::SelectImage { .. } => self.module.lit(0u32),
            ExprNode::TextureSample => {
                let slot = self.resolve_image_slot(node_id, "image", errors)?;
                let coordinates = self.operand(node_id, "coordinates", errors)?;
                let Ok(slot_index) = u32::try_from(slot) else {
                    errors.push(BakeError::node(
                        node_id,
                        "texture slot index exceeds bevy_hanabi's u32 limit",
                    ));
                    return None;
                };
                let sample = match TextureSampleExpr::new(
                    slot_index,
                    SlotDimension::D2,
                    coordinates,
                    None,
                ) {
                    Ok(sample) => sample,
                    Err(error) => {
                        errors.push(BakeError::node(
                            node_id,
                            format!("could not create texture sample: {error}"),
                        ));
                        return None;
                    }
                };
                self.module.add_expr(Expr::TextureSample(sample))
            }
        };
        Some(handle)
    }

    /// Bake a property reference, by stable id.
    ///
    /// Yields the property's runtime handle if exposed, otherwise its default
    /// value inlined as a literal. A reference to a missing or duplicate-named
    /// exposed property is reported, not fatal.
    fn bake_property_ref(
        &mut self,
        node_id: NodeId,
        id: PropertyId,
        errors: &mut Vec<BakeError>,
    ) -> Option<ExprHandle> {
        let Some(def) = self.props.defs.get(&id) else {
            errors.push(BakeError::node(
                node_id,
                format!("reference to unknown property id {}", id.get()),
            ));
            return None;
        };
        if def.exposed {
            // An exposed property with no handle was dropped as a duplicate-name
            // conflict during registration; that error is already recorded.
            let handle = self.props.handles.get(&id)?;
            Some(self.module.prop(*handle))
        } else {
            let h = self.module.lit(def.default);
            // Record provenance so a live `SetPropertyDefault` on this unexposed
            // property can push its new value through the proxy property this
            // inlined literal is promoted to, instead of re-baking.
            self.literal_sites.insert(LiteralSite::Node(node_id), h);
            Some(h)
        }
    }

    /// Resolve the value feeding input port `port` of `node_id`.
    ///
    /// Uses the source of a link into that port if one exists, else the port's
    /// inline default literal. Errors if neither is available.
    fn operand(
        &mut self,
        node_id: NodeId,
        port: &str,
        errors: &mut Vec<BakeError>,
    ) -> Option<ExprHandle> {
        if let Some(source) = self.linked_source(node_id, port) {
            return self.resolve(source, errors);
        }
        if let Some(default) = self.inline_default(node_id, port) {
            return Some(self.record_inline_literal(node_id, port, default));
        }
        errors.push(BakeError::node(
            node_id,
            format!("input port '{port}' is neither linked nor given a default"),
        ));
        None
    }

    /// Like [`operand`] but for an optional input port.
    ///
    /// A missing link *and* missing inline default is not an error — the port
    /// is simply left unconnected (the field stays at its factory default /
    /// `None`).
    ///
    /// [`operand`]: Self::operand
    fn operand_optional(
        &mut self,
        node_id: NodeId,
        port: &str,
        errors: &mut Vec<BakeError>,
    ) -> Option<ExprHandle> {
        if let Some(source) = self.linked_source(node_id, port) {
            return self.resolve(source, errors);
        }
        let default = self.inline_default(node_id, port)?;
        Some(self.record_inline_literal(node_id, port, default))
    }

    /// Bake an inline-default `value` into a module literal.
    ///
    /// Records its graph site so a later value tweak can find it.
    fn record_inline_literal(
        &mut self,
        node_id: NodeId,
        port: &str,
        value: bevy_hanabi::Value,
    ) -> ExprHandle {
        let h = self.module.lit(value);
        self.literal_sites.insert(
            LiteralSite::Input {
                node: node_id,
                port: port.into(),
            },
            h,
        );
        h
    }

    /// The source node of the (single) link targeting `node_id`'s `port`.
    fn linked_source(&self, node_id: NodeId, port: &str) -> Option<NodeId> {
        let target = PortRef {
            node: node_id,
            port: port.into(),
        };
        self.graph
            .links
            .iter()
            .find(|l| l.to == target)
            .map(|l| l.from.node)
    }

    /// The inline default literal for `node_id`'s input `port`, if it carries a
    /// value default. An image default yields `None` (image ports bake through
    /// their own path).
    fn inline_default(&self, node_id: NodeId, port: &str) -> Option<bevy_hanabi::Value> {
        let node = self.graph.node(node_id)?;
        node.inputs
            .iter()
            .find(|s| &*s.name == port)
            .and_then(|s| s.default.as_value())
    }

    /// The inline image binding on `node_id`'s input `port`, if it carries one.
    fn inline_image(&self, node_id: NodeId, port: &str) -> Option<ImageBinding> {
        let node = self.graph.node(node_id)?;
        node.inputs
            .iter()
            .find(|s| &*s.name == port)
            .and_then(|s| s.default.as_image())
            .cloned()
    }

    /// Allocate a texture slot with a unique name, recording its planned image.
    ///
    /// Returns the slot's sampling index (its position in the module's texture
    /// layout, which is also its index into the host's material image list).
    fn alloc_slot(&mut self, desired_name: &str, image: PlannedImage) -> usize {
        let mut name = desired_name.to_string();
        let mut n = 2;
        while self.used_slot_names.contains(&name) {
            name = format!("{desired_name}_{n}");
            n += 1;
        }
        self.used_slot_names.insert(name.clone());
        let index = self.texture_plan.len();
        self.module.add_texture_slot(name, SlotDimension::D2);
        self.texture_plan.push(image);
        index
    }

    /// Resolve an image binding to a constant slot index, allocating if needed.
    ///
    /// A host-supplied slot reuses its reserved index, pinned assets reuse an
    /// index by path, and each unbound binding allocates a fresh slot.
    fn binding_slot(
        &mut self,
        binding: &ImageBinding,
        node_id: NodeId,
        errors: &mut Vec<BakeError>,
    ) -> Option<usize> {
        match binding {
            ImageBinding::Unbound => Some(self.alloc_slot("image", PlannedImage::Unbound)),
            ImageBinding::Asset(path) => {
                if let Some(index) = self.pinned_asset_slots.get(path) {
                    return Some(*index);
                }
                let name = asset_slot_name(path);
                let index = self.alloc_slot(&name, PlannedImage::Asset(path.clone()));
                self.pinned_asset_slots.insert(path.clone(), index);
                Some(index)
            }
            ImageBinding::Slot(id) => match self.registry_slots.get(id) {
                Some(index) => Some(*index),
                None => {
                    errors.push(BakeError::node(
                        node_id,
                        format!("image references unknown texture slot {}", id.get()),
                    ));
                    None
                }
            },
        }
    }

    /// Resolve the slot for an [`ExprNode::Image`] node, sharing it on fan-out.
    fn image_node_slot(&mut self, node_id: NodeId, errors: &mut Vec<BakeError>) -> Option<usize> {
        if let Some(index) = self.image_node_slots.get(&node_id) {
            return Some(*index);
        }
        let binding = match &self.graph.node(node_id)?.payload {
            NodePayload::Expr(ExprNode::Image(binding)) => binding.clone(),
            _ => {
                errors.push(BakeError::node(
                    node_id,
                    "expected an image source feeding an image port",
                ));
                return None;
            }
        };
        let index = self.binding_slot(&binding, node_id, errors)?;
        self.image_node_slots.insert(node_id, index);
        Some(index)
    }

    /// Resolve the constant slot index feeding image port `port` of `node_id`.
    ///
    /// Prefers a linked image source, falling back to the port's inline image
    /// binding. An image port that is neither linked nor bound is an error.
    fn resolve_image_slot(
        &mut self,
        node_id: NodeId,
        port: &str,
        errors: &mut Vec<BakeError>,
    ) -> Option<usize> {
        if let Some(source) = self.linked_source(node_id, port) {
            return self.image_source_slot(source, errors);
        }
        if let Some(binding) = self.inline_image(node_id, port) {
            return self.binding_slot(&binding, node_id, errors);
        }
        errors.push(BakeError::node(
            node_id,
            format!("image port '{port}' is neither linked nor bound to an image"),
        ));
        None
    }

    /// Resolve an image-typed source node to its constant slot index.
    fn image_source_slot(&mut self, source: NodeId, errors: &mut Vec<BakeError>) -> Option<usize> {
        match &self.graph.node(source)?.payload {
            NodePayload::Expr(ExprNode::Image(_)) => self.image_node_slot(source, errors),
            NodePayload::Expr(ExprNode::SelectImage { count }) => {
                let count = *count;
                self.select_image_slot(source, count, errors)
            }
            _ => {
                errors.push(BakeError::node(
                    source,
                    "expected an image source feeding an image port",
                ));
                None
            }
        }
    }

    /// Resolve a [`ExprNode::SelectImage`] to the slot it selects at bake time.
    ///
    /// This bevy_hanabi version interpolates a texture index into a static WGSL
    /// identifier, so only a compile-time constant `index` can bake: the
    /// selected image input is resolved to its own constant slot. A runtime
    /// `index` is reported as unbakeable.
    fn select_image_slot(
        &mut self,
        node_id: NodeId,
        count: u32,
        errors: &mut Vec<BakeError>,
    ) -> Option<usize> {
        let Some(index) = self.const_u32(node_id, "index") else {
            errors.push(BakeError::node(
                node_id,
                "Select Image needs a compile-time constant 'index' to bake; runtime texture \
                 selection is unsupported by this bevy_hanabi version",
            ));
            return None;
        };
        let index = index.min(count.saturating_sub(1));
        let port = format!("image{index}");
        let Some(source) = self.linked_source(node_id, &port) else {
            errors.push(BakeError::node(
                node_id,
                format!("Select Image input '{port}' is empty"),
            ));
            return None;
        };
        self.image_source_slot(source, errors)
    }

    /// The compile-time constant `u32` value feeding `node_id`'s input `port`.
    ///
    /// Recognises an inline literal default and a link to a literal expression
    /// node; any other source (property, attribute, computed) is not constant
    /// and yields `None`.
    fn const_u32(&self, node_id: NodeId, port: &str) -> Option<u32> {
        if let Some(source) = self.linked_source(node_id, port) {
            match &self.graph.node(source)?.payload {
                NodePayload::Expr(ExprNode::Literal(v)) => value_as_u32(v),
                _ => None,
            }
        } else {
            self.inline_default(node_id, port)
                .and_then(|v| value_as_u32(&v))
        }
    }

    /// Bake one modifier node into a runtime [`BoxedModifier`].
    ///
    /// The modifier instance is created by the registered
    /// [`ReflectModifier::factory`] (which allocates sensible default literals
    /// into the module), then its fields are overwritten by reflection: each
    /// expression-port field is set to the [`operand`] feeding
    /// it, and each configuration field to the matching [`EditValue`] from the
    /// node's config bag. Every failure (unregistered type, type mismatch,
    /// unbakeable value) is collected as a [`BakeError`] rather than panicking.
    ///
    /// [`operand`]: Self::operand
    fn bake_modifier(
        &mut self,
        node_id: NodeId,
        registry: &TypeRegistry,
        child_indices: &HashMap<NodeId, u32>,
        errors: &mut Vec<BakeError>,
    ) -> Option<BoxedModifier> {
        let node = self.graph.node(node_id).or_else(|| {
            errors.push(BakeError::node(
                node_id,
                format!("stack references missing node {}", node_id.get()),
            ));
            None
        })?;
        let NodePayload::Modifier(data) = &node.payload else {
            errors.push(BakeError::node(
                node_id,
                "expected a modifier node in a stack",
            ));
            return None;
        };
        let (type_path, config) = match data {
            ModifierNodeData::Known { type_path, config } => (type_path, config),
            ModifierNodeData::Unknown { type_path, .. } => {
                errors.push(BakeError::node(
                    node_id,
                    format!("modifier type '{type_path}' is not registered; cannot bake"),
                ));
                return None;
            }
        };

        let Some(registration) = registry.get_with_type_path(type_path) else {
            errors.push(BakeError::node(
                node_id,
                format!("modifier type '{type_path}' is not in the type registry"),
            ));
            return None;
        };
        let Some(reflect_modifier) = registration.data::<ReflectModifier>() else {
            errors.push(BakeError::node(
                node_id,
                format!("type '{type_path}' is registered but is not a modifier"),
            ));
            return None;
        };
        let Some(schema) = modifier_schema(registration.type_info()) else {
            errors.push(BakeError::node(
                node_id,
                format!("modifier type '{type_path}' does not reflect as a struct"),
            ));
            return None;
        };

        let mut boxed = (reflect_modifier.factory)(self.module);

        // Expression ports: overwrite each field with the resolved operand
        // (linked source handle or inline-default literal). Optional ports left
        // unconnected keep the factory default.
        for field in schema.ports() {
            // A texture port resolves its image source (linked node or inline
            // binding) to the static slot index used by the modifier's WGSL.
            if matches!(field.role, FieldRole::Texture) {
                let Some(slot) = self.resolve_image_slot(node_id, &field.name, errors) else {
                    continue;
                };
                let Ok(slot_index) = u32::try_from(slot) else {
                    errors.push(BakeError::node(
                        node_id,
                        "texture slot index exceeds bevy_hanabi's u32 limit",
                    ));
                    continue;
                };
                if !set_texture_slot_field(boxed.as_reflect_mut(), &field.name, slot_index) {
                    errors.push(BakeError::node(
                        node_id,
                        format!("could not set texture field '{}'", field.name),
                    ));
                }
                continue;
            }
            let optional = matches!(field.role, FieldRole::ExprPort { optional: true });
            let handle = if optional {
                self.operand_optional(node_id, &field.name, errors)
            } else {
                self.operand(node_id, &field.name, errors)
            };
            let handle = handle.map(|handle| {
                if normalizes_modifier_input(type_path, &field.name) {
                    self.module.normalize(handle)
                } else {
                    handle
                }
            });
            if let Some(handle) = handle
                && !set_expr_field(boxed.as_reflect_mut(), &field.name, handle, optional)
            {
                errors.push(BakeError::node(
                    node_id,
                    format!("could not set expression field '{}'", field.name),
                ));
            }
        }

        // Configuration fields: apply each value present in the config bag;
        // absent fields keep their factory default.
        for field in schema.config() {
            let Some(value) = config.get(field.name.as_ref()) else {
                continue;
            };
            if let Err(message) = apply_config_field(boxed.as_reflect_mut(), &field.name, value) {
                errors.push(BakeError::node(node_id, message));
            }
        }

        // Topology-owned fields: hidden from the schema (see
        // `schema::is_topology_owned_field`), so they are never touched by the
        // config-bag loop above and are overwritten unconditionally here from
        // derived topology. Under the currently supported single-child-per-parent
        // topology every emitter channels into index 0; `child_indices` only
        // disagrees once multi-child parents are supported.
        if type_path.as_ref() == EmitSpawnEventModifier::type_path() {
            let Some(child_index) = child_indices.get(&node_id).copied() else {
                // An unlinked event output is inert. Hanabi has no disabled
                // emitter representation: baking its default child index 0
                // would emit an `append_spawn_events_0` call without the child
                // buffer that defines that helper, producing invalid WGSL.
                return None;
            };
            if let Some(emit) = boxed
                .as_reflect_mut()
                .downcast_mut::<EmitSpawnEventModifier>()
            {
                emit.child_index = child_index;
            }
        }

        Some(boxed)
    }
}

fn normalizes_modifier_input(type_path: &str, field_name: &str) -> bool {
    field_name == "axis"
        && matches!(
            type_path,
            path if path == SetPositionCircleModifier::type_path()
                || path == SetVelocityCircleModifier::type_path()
                || path == SetVelocityTangentModifier::type_path()
                || path == TangentAccelModifier::type_path()
        )
}

/// A texture-slot name derived from an asset path's file stem.
///
/// Purely cosmetic — slot binding is by index, not name — but a readable name
/// helps when inspecting the baked module. Falls back to `"image"` for a path
/// with no usable stem; [`ExprBaker::alloc_slot`] then ensures uniqueness.
fn asset_slot_name(path: &AssetPath) -> String {
    path.path()
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("image")
        .to_string()
}

/// The constant `u32` value of a scalar [`Value`], if it is a scalar.
pub(crate) fn value_as_u32(value: &Value) -> Option<u32> {
    match value {
        Value::Scalar(s) => Some(s.as_u32()),
        _ => None,
    }
}

/// Set an `ExprHandle` (or `Option<ExprHandle>`) field by name.
///
/// Returns `false` if the field is absent or not of the expected handle type.
fn set_expr_field(
    reflect: &mut dyn Reflect,
    name: &str,
    handle: ExprHandle,
    optional: bool,
) -> bool {
    let ReflectMut::Struct(s) = reflect.reflect_mut() else {
        return false;
    };
    let Some(field) = s.field_mut(name) else {
        return false;
    };
    if optional && let Some(slot) = field.try_downcast_mut::<Option<ExprHandle>>() {
        *slot = Some(handle);
        return true;
    }
    if let Some(slot) = field.try_downcast_mut::<ExprHandle>() {
        *slot = handle;
        return true;
    }
    false
}

/// Set a static texture-slot index field by name.
fn set_texture_slot_field(reflect: &mut dyn Reflect, name: &str, slot_index: u32) -> bool {
    let ReflectMut::Struct(s) = reflect.reflect_mut() else {
        return false;
    };
    let Some(field) = s.field_mut(name) else {
        return false;
    };
    let Some(slot) = field.try_downcast_mut::<u32>() else {
        return false;
    };
    *slot = slot_index;
    true
}

/// Apply an [`EditValue`] to the named configuration field of a modifier.
fn apply_config_field(
    reflect: &mut dyn Reflect,
    name: &str,
    value: &EditValue,
) -> Result<(), String> {
    let ReflectMut::Struct(s) = reflect.reflect_mut() else {
        return Err("modifier does not reflect as a struct".to_string());
    };
    let field = s
        .field_mut(name)
        .ok_or_else(|| format!("modifier has no field '{name}'"))?;
    apply_edit_value(field, value, name)
}

/// Write one [`EditValue`] into a reflected field.
///
/// Most variants wrap the field's exact runtime type and are assigned directly;
/// enums and bitflags are built from their stored identity. Values that have no
/// faithful `bevy_hanabi` 0.18 representation (texture-LUT gradients, pinned
/// texture assets) report an error.
fn apply_edit_value(
    field: &mut dyn PartialReflect,
    value: &EditValue,
    name: &str,
) -> Result<(), String> {
    match value {
        EditValue::Bool(b) => assign(field, *b, name),
        EditValue::U32(u) => assign(field, *u, name),
        EditValue::UVec2(v) => assign(field, *v, name),
        EditValue::Color(c) => assign(field, *c, name),
        EditValue::Attribute(a) => assign(field, *a, name),
        EditValue::CpuVec3(v) => assign(field, *v, name),
        EditValue::CpuVec4(v) => assign(field, *v, name),
        EditValue::Scalar(v) => assign_scalar(field, v, name),
        EditValue::Gradient3(g) => match g {
            GradientVec3::Analytical(grad) => assign(field, grad.clone(), name),
            GradientVec3::Lut(_) => Err(format!(
                "field '{name}': texture-LUT gradient has no bevy_hanabi 0.18 representation"
            )),
        },
        EditValue::Gradient4(g) => match g {
            GradientVec4::Analytical(grad) => assign(field, grad.clone(), name),
            GradientVec4::Lut(_) => Err(format!(
                "field '{name}': texture-LUT gradient has no bevy_hanabi 0.18 representation"
            )),
        },
        EditValue::Enum { variant, .. } => assign_enum(field, variant, name),
        EditValue::Flags { bits, .. } => assign_flags(field, *bits, name),
        EditValue::Texture(_) => Err(format!(
            "field '{name}': texture baking is not yet supported"
        )),
        EditValue::Raw(_) => Err(format!("field '{name}': raw config values cannot be baked")),
    }
}

/// Assign a concrete value to a field, failing if the field is of another type.
fn assign<T: Reflect>(field: &mut dyn PartialReflect, value: T, name: &str) -> Result<(), String> {
    match field.try_downcast_mut::<T>() {
        Some(slot) => {
            *slot = value;
            Ok(())
        }
        None => Err(format!(
            "field '{name}': expected {}, found {}",
            std::any::type_name::<T>(),
            field.reflect_type_path()
        )),
    }
}

/// Assign a `bevy_hanabi` [`Value`] to a concrete scalar/vector field.
fn assign_scalar(field: &mut dyn PartialReflect, value: &Value, name: &str) -> Result<(), String> {
    match value {
        Value::Scalar(s) => {
            if let Some(slot) = field.try_downcast_mut::<f32>() {
                *slot = s.as_f32();
            } else if let Some(slot) = field.try_downcast_mut::<i32>() {
                *slot = s.as_i32();
            } else if let Some(slot) = field.try_downcast_mut::<u32>() {
                *slot = s.as_u32();
            } else if let Some(slot) = field.try_downcast_mut::<bool>() {
                *slot = s.as_bool();
            } else {
                return Err(scalar_mismatch(name, field));
            }
        }
        Value::Vector(v) => {
            if let Some(slot) = field.try_downcast_mut::<Vec2>() {
                *slot = v.as_vec2();
            } else if let Some(slot) = field.try_downcast_mut::<Vec3>() {
                *slot = v.as_vec3();
            } else if let Some(slot) = field.try_downcast_mut::<Vec4>() {
                *slot = v.as_vec4();
            } else if let Some(slot) = field.try_downcast_mut::<UVec2>() {
                *slot = v.as_uvec2();
            } else {
                return Err(scalar_mismatch(name, field));
            }
        }
        Value::Matrix(_) => return Err(scalar_mismatch(name, field)),
        _ => return Err(scalar_mismatch(name, field)),
    }
    Ok(())
}

fn scalar_mismatch(name: &str, field: &dyn PartialReflect) -> String {
    format!(
        "field '{name}': scalar value does not match field type {}",
        field.reflect_type_path()
    )
}

/// Set a data-less enum field to the variant of the given name.
///
/// Applies by reflect, which matches the active variant by name.
fn assign_enum(field: &mut dyn PartialReflect, variant: &str, name: &str) -> Result<(), String> {
    let dynamic = DynamicEnum::new(variant.to_string(), DynamicVariant::Unit);
    field
        .try_apply(&dynamic)
        .map_err(|e| format!("field '{name}': cannot select enum variant '{variant}': {e:?}"))
}

/// Set a bitflags newtype field to `bits`.
///
/// The field is a tuple struct wrapping one integer; `bits` is narrowed to the
/// field's actual integer width.
fn assign_flags(field: &mut dyn PartialReflect, bits: u64, name: &str) -> Result<(), String> {
    let ReflectMut::TupleStruct(ts) = field.reflect_mut() else {
        return Err(format!("field '{name}': flags field is not a tuple struct"));
    };
    let inner = ts
        .field_mut(0)
        .ok_or_else(|| format!("field '{name}': flags newtype has no inner value"))?;
    if let Some(slot) = inner.try_downcast_mut::<u8>() {
        *slot = bits as u8;
    } else if let Some(slot) = inner.try_downcast_mut::<u16>() {
        *slot = bits as u16;
    } else if let Some(slot) = inner.try_downcast_mut::<u32>() {
        *slot = bits as u32;
    } else if let Some(slot) = inner.try_downcast_mut::<u64>() {
        *slot = bits;
    } else {
        return Err(format!(
            "field '{name}': unsupported flags integer type {}",
            inner.reflect_type_path()
        ));
    }
    Ok(())
}

/// Build a [`Module`] from `graph`'s expression nodes and properties.
///
/// Returns the module and the `NodeId → ExprHandle` map for every expression
/// node that is reachable from a modifier or another expression.
///
/// Only expression nodes reachable as operands or modifier inputs are
/// materialized; a dangling expression node with no consumer contributes
/// nothing to the arena. Errors (cycles, unknown properties, missing inputs,
/// wrong node kinds) are collected rather than fatal, so the caller can surface
/// all of them at once.
pub fn bake_module(
    graph: &EmitterGraph,
) -> Result<(Module, HashMap<NodeId, ExprHandle>), Vec<BakeError>> {
    let mut module = Module::default();
    let mut errors = Vec::new();

    let props = bake_properties(graph, &mut module, &mut errors);

    let mut baker = ExprBaker::new(graph, &props, &mut module);

    // Resolve every expression node that participates in the graph. A node is a
    // participant if it is the source or target of a link, or an operand-bearing
    // expression; resolving each pulls in its operand subtree transitively.
    let participants = expr_participants(graph);
    for node_id in participants {
        baker.resolve(node_id, &mut errors);
    }

    let handles = std::mem::take(&mut baker.handles);
    drop(baker);

    if errors.is_empty() {
        Ok((module, handles))
    } else {
        Err(errors)
    }
}

/// Bake a whole [`EmitterGraph`] into a runtime [`EffectAsset`].
///
/// Builds the expression [`Module`] (properties + every expression reachable
/// from a stack modifier), instantiates each stack's modifiers in execution
/// order (`Init → Update → Render`), and assembles them with the emitter
/// settings into an
/// `EffectAsset`. Every problem is collected as a [`BakeError`] (attributed to
/// the node, property, or graph at fault) rather than panicking; the asset is
/// returned only when the graph bakes cleanly.
///
/// `spawner` comes from this emitter's linked spawn-source node, while
/// `child_indices` comes from its outgoing GPU-event topology. See
/// [`derive_emitter_spawner_settings`] and [`derive_emitter_child_indices`].
/// `child_indices` contains an entry for each linked
/// `EmitSpawnEventModifier`; unlinked spawn-event nodes are omitted from the
/// baked modifier list and therefore act as no-ops.
pub fn bake_emitter(
    graph: &EmitterGraph,
    registry: &TypeRegistry,
    spawner: SpawnerSettings,
    child_indices: &HashMap<NodeId, u32>,
) -> Result<EffectAsset, Vec<BakeError>> {
    bake_emitter_with_provenance(graph, registry, spawner, child_indices)
        .map(|(asset, _provenance)| asset)
}

/// Like [`bake_emitter`], but also returns the [`BakeProvenance`].
///
/// Maps every baked literal to its graph origin (driving the live-tweak path)
/// and lists the resolved texture slots (driving material wiring).
pub fn bake_emitter_with_provenance(
    graph: &EmitterGraph,
    registry: &TypeRegistry,
    spawner: SpawnerSettings,
    child_indices: &HashMap<NodeId, u32>,
) -> Result<(EffectAsset, BakeProvenance), Vec<BakeError>> {
    let mut module = Module::default();
    let mut errors = Vec::new();
    let props = bake_properties(graph, &mut module, &mut errors);

    let mut baker = ExprBaker::new(graph, &props, &mut module);

    // Bake each stack's members, routing every modifier to its execution stage.
    // A modifier whose kind contradicts its stack (e.g. a render modifier in an
    // Init stack) is reported rather than silently misplaced.
    let mut init: Vec<bevy_hanabi::BoxedModifier> = Vec::new();
    let mut update: Vec<bevy_hanabi::BoxedModifier> = Vec::new();
    let mut render: Vec<Box<dyn bevy_hanabi::RenderModifier>> = Vec::new();
    for stack in &graph.stacks {
        for &member in &stack.members {
            let Some(boxed) = baker.bake_modifier(member, registry, child_indices, &mut errors)
            else {
                continue;
            };
            match (stack.group, boxed.as_render().is_some()) {
                (ModifierGroup::Init, false) => init.push(boxed),
                (ModifierGroup::Update, false) => update.push(boxed),
                (ModifierGroup::Render, true) => {
                    render.push(boxed.as_render().unwrap().boxed_render_clone())
                }
                (ModifierGroup::Render, false) => errors.push(BakeError::node(
                    member,
                    "non-render modifier placed in a Render stack",
                )),
                (group, true) => errors.push(BakeError::node(
                    member,
                    format!("render modifier placed in a {group:?} stack"),
                )),
            }
        }
    }

    let literal_sites = std::mem::take(&mut baker.literal_sites);
    let texture_plan = std::mem::take(&mut baker.texture_plan);
    drop(baker);

    if !errors.is_empty() {
        return Err(errors);
    }

    let mut asset = EffectAsset::new(graph.capacity, spawner, module);
    asset.name = graph.name.to_string();
    asset.simulation_space = graph.simulation_space;
    asset.simulation_condition = graph.simulation_condition;
    asset.z_layer_2d = graph.z_layer_2d;
    for m in init {
        asset = asset.add_modifier(ModifierContext::Init, m);
    }
    for m in update {
        asset = asset.add_modifier(ModifierContext::Update, m);
    }
    for m in render {
        asset = asset.add_render_modifier(m);
    }
    Ok((
        asset,
        BakeProvenance {
            literal_sites,
            texture_plan,
        },
    ))
}

/// Bake a graph for live preview with a document-unique asset name.
///
/// Tags the asset name so its compiled shaders get a document-unique
/// `hanabi/{name}_…` path; `preview_tag` is the owning document's preview tag.
///
/// The tag lives only on the throwaway preview asset (and the proxy cloned from
/// it); the saved graph keeps its plain `name`.
pub fn bake_emitter_preview(
    graph: &EmitterGraph,
    registry: &TypeRegistry,
    preview_tag: u64,
    spawner: SpawnerSettings,
    child_indices: &HashMap<NodeId, u32>,
) -> EffectAsset {
    bake_emitter_preview_with_provenance(graph, registry, preview_tag, spawner, child_indices).0
}

/// Like [`bake_emitter_preview`], but also returns the [`BakeProvenance`].
///
/// Lets the live-tweak path bind value edits to proxy properties and the
/// renderer wire textures. On bake failure the provenance is empty.
pub fn bake_emitter_preview_with_provenance(
    graph: &EmitterGraph,
    registry: &TypeRegistry,
    preview_tag: u64,
    spawner: SpawnerSettings,
    child_indices: &HashMap<NodeId, u32>,
) -> (EffectAsset, BakeProvenance) {
    let (mut asset, provenance) =
        bake_emitter_or_empty_with_provenance(graph, registry, spawner, child_indices);
    asset.name = preview_asset_name(&graph.name, preview_tag);
    (asset, provenance)
}

/// Document-unique preview asset name: `{base}~{tag}`.
///
/// The `~` separator avoids the `_` that hanabi uses to delimit
/// `{name}_{phase}_{hash}` shader paths.
pub fn preview_asset_name(base: &str, preview_tag: u64) -> String {
    format!("{base}~{preview_tag}")
}

/// Bake a graph, falling back to an empty asset when it fails.
///
/// The fallback is inert but renderable. Used by the seeding/reconcile path
/// where the viewport must always have *some* asset to instantiate; the bake
/// errors are logged for the UI to surface separately rather than aborting
/// document creation.
pub fn bake_emitter_or_empty(
    graph: &EmitterGraph,
    registry: &TypeRegistry,
    spawner: SpawnerSettings,
    child_indices: &HashMap<NodeId, u32>,
) -> EffectAsset {
    bake_emitter_or_empty_with_provenance(graph, registry, spawner, child_indices).0
}

/// Like [`bake_emitter_or_empty`], but also returns the [`BakeProvenance`].
///
/// Empty when the bake fails and the inert fallback is used.
pub fn bake_emitter_or_empty_with_provenance(
    graph: &EmitterGraph,
    registry: &TypeRegistry,
    spawner: SpawnerSettings,
    child_indices: &HashMap<NodeId, u32>,
) -> (EffectAsset, BakeProvenance) {
    bake_emitter_with_provenance(graph, registry, spawner, child_indices).unwrap_or_else(|errors| {
        bevy::log::error!(
            "emitter graph failed to bake ({} error(s)): {errors:?}",
            errors.len()
        );
        let mut asset = EffectAsset::new(graph.capacity, spawner, Module::default());
        asset.name = graph.name.to_string();
        (asset, BakeProvenance::default())
    })
}

/// The channel each of `emitter`'s `EmitSpawnEventModifier` nodes derives from
/// the effect graph's inter-emitter topology.
///
/// Keyed by node id, one entry per spawn-event node linked to a GPU source that
/// both `emitter` owns (i.e. every [`EventLink`] whose node
/// [`EffectGraph::emitter_owning_node`] resolves to `emitter`) *and* that is
/// itself drives an emitter via a [`SourceLink`] — a spawn-event node targeting
/// an unconnected GPU source contributes no channel and is omitted from the map
/// entirely, so it can never shift a real, connected sibling's channel index.
/// [`validation::validate_topology`] already rejects that unconnected-source
/// shape outright (see `check_gpu_sources_and_event_links`); this filter is
/// only a defensive backstop for callers that bake without validating first.
/// The channel is the target source's position among `emitter`'s *distinct*
/// connected GPU sources, in ascending [`SourceId`] order — deterministic and
/// independent of [`EffectGraph::event_links`] authoring order. Under the
/// currently supported single-child-per-parent topology (see
/// `validation::check_single_child_restriction`) this is always `0`; the
/// general form is kept so lifting that restriction only requires removing the
/// validation check, not touching this derivation.
///
/// [`EventLink`]: crate::model::EventLink
/// [`SourceLink`]: crate::model::SourceLink
pub fn derive_emitter_child_indices(
    effect_graph: &EffectGraph,
    emitter: EmitterId,
) -> HashMap<NodeId, u32> {
    let mut targets: Vec<SourceId> = effect_graph
        .event_links
        .iter()
        .filter(|l| effect_graph.emitter_owning_node(l.node) == Some(emitter))
        .map(|l| l.target)
        .filter(|target| effect_graph.emitter_for_source(*target).is_some())
        .collect();
    targets.sort_by_key(|s| s.get());
    targets.dedup();
    let channel_of: HashMap<SourceId, u32> = targets
        .into_iter()
        .enumerate()
        .map(|(i, s)| (s, i as u32))
        .collect();

    effect_graph
        .event_links
        .iter()
        .filter(|l| effect_graph.emitter_owning_node(l.node) == Some(emitter))
        .filter_map(|l| channel_of.get(&l.target).map(|&c| (l.node, c)))
        .collect()
}

/// The inert placeholder spawner used to bake a GPU-driven emitter.
///
/// A [`SourceKind::GpuEvent`] context carries no `SpawnerSettings` of its own —
/// all of a GPU-driven emitter's particles come from spawn events written by
/// its parent — but [`bevy_hanabi::EffectAsset::new`] always requires one.
/// `once` with `starts_active(false)` never fires on its own, so every particle
/// the baked asset produces is attributable to an event.
fn inert_spawner() -> SpawnerSettings {
    SpawnerSettings::once(0.0f32.into()).with_starts_active(false)
}

/// The [`SpawnerSettings`] `emitter` should be baked with, derived from its
/// linked [`SourceContext`].
///
/// `Some` with the linked `CpuSpawner`'s settings for a CPU-rooted emitter;
/// `Some(`[`inert_spawner`]`())` for a GPU-driven emitter (no settings of its
/// own — see [`SourceKind::GpuEvent`]); `None` if `emitter` has no linked
/// source at all (a topology error caught by
/// [`validation::validate_topology`] before this is ever called in practice).
///
/// [`SourceContext`]: crate::model::SourceContext
pub fn derive_emitter_spawner_settings(
    effect_graph: &EffectGraph,
    emitter: EmitterId,
) -> Option<SpawnerSettings> {
    let source = effect_graph.source_for_emitter(emitter)?;
    match &effect_graph.source(source)?.kind {
        SourceKind::CpuSpawner { settings } => Some(*settings),
        SourceKind::GpuEvent => Some(inert_spawner()),
    }
}

/// One emitter's baked output within an effect bake.
#[derive(Clone)]
pub struct BakedEmitter {
    pub emitter: EmitterId,
    pub asset: EffectAsset,
    pub provenance: BakeProvenance,
    /// The parent emitter that spawns particles into this one via a GPU source
    /// context, if any (see [`EffectGraph::parent_emitter`]).
    pub parent: Option<EmitterId>,
}

/// The result of [`bake_effect`], in effect-graph order.
#[derive(Clone, Default)]
pub struct BakedEffect {
    pub emitters: Vec<BakedEmitter>,
}

impl BakedEffect {
    /// The baked outcome of a specific emitter, if it was baked.
    pub fn emitter(&self, id: EmitterId) -> Option<&BakedEmitter> {
        self.emitters.iter().find(|e| e.emitter == id)
    }
}

/// One emitter's error within a [`bake_effect`] call, attributed
/// to the emitter (and its source, if linked) in addition to whatever node or
/// property the wrapped [`BakeError`] itself blames.
#[derive(Debug, Clone, PartialEq)]
pub struct EffectBakeError {
    pub emitter: EmitterId,
    pub source: Option<SourceId>,
    pub error: BakeError,
}

/// Bake every emitter of an [`EffectGraph`] into its own [`EffectAsset`].
///
/// Validates the effect's inter-emitter topology first (see
/// [`validation::validate_topology`]); a topology error aborts the whole bake
/// with no emitter touched, since it would make the derived per-emitter
/// spawner/channel inputs below meaningless. Each topology error is reported
/// as a [`EffectBakeError`] attributed to whichever emitter, source, or node
/// [`validation::TopologyError`] blamed.
///
/// Otherwise, every emitter in [`EffectGraph::emitters`] order is baked
/// sequentially and independently, with its [`SpawnerSettings`] and
/// `EmitSpawnEventModifier` channels derived from topology (see
/// [`derive_emitter_spawner_settings`] and [`derive_emitter_child_indices`]).
/// Every error from every emitter is collected — baking one emitter never stops
/// another from being attempted — but the whole call fails if *any* emitter
/// fails: there is no silent per-emitter empty-asset fallback (unlike
/// [`bake_emitter_or_empty`], which exists for a different, single-emitter
/// preview use case). Before returning, the complete hierarchy is passed
/// through Hanabi's shader generator with each emitter's resolved parent layout
/// and child-event count, so runtime-invalid assets never escape as successful
/// bake results.
pub fn bake_effect(
    effect_graph: &EffectGraph,
    registry: &TypeRegistry,
) -> Result<BakedEffect, Vec<EffectBakeError>> {
    let topology_errors = validation::validate_topology(effect_graph);
    if !topology_errors.is_empty() {
        return Err(topology_errors
            .into_iter()
            .map(|e| topology_error_to_effect_bake_error(effect_graph, e))
            .collect());
    }

    let mut baked = Vec::with_capacity(effect_graph.emitters.len());
    let mut errors = Vec::new();
    for emitter in &effect_graph.emitters {
        let source = effect_graph.source_for_emitter(emitter.id);
        // Topology validation above guarantees every emitter has exactly one
        // linked source of a recognised kind, so this only fails on a bug.
        let Some(spawner) = derive_emitter_spawner_settings(effect_graph, emitter.id) else {
            errors.push(EffectBakeError {
                emitter: emitter.id,
                source,
                error: BakeError::emitter(
                    emitter.id,
                    "emitter has no linked spawn source to derive settings from",
                ),
            });
            continue;
        };
        let child_indices = derive_emitter_child_indices(effect_graph, emitter.id);
        match bake_emitter_with_provenance(emitter, registry, spawner, &child_indices) {
            Ok((asset, provenance)) => baked.push(BakedEmitter {
                emitter: emitter.id,
                asset,
                provenance,
                parent: effect_graph.parent_emitter(emitter.id),
            }),
            Err(emitter_errors) => {
                errors.extend(emitter_errors.into_iter().map(|error| EffectBakeError {
                    emitter: emitter.id,
                    source,
                    error,
                }))
            }
        }
    }

    if errors.is_empty() {
        for emitter in &baked {
            let parent_layout = emitter.parent.and_then(|parent| {
                baked
                    .iter()
                    .find(|candidate| candidate.emitter == parent)
                    .map(|parent| parent.asset.particle_layout())
            });
            let child_count = baked
                .iter()
                .filter(|candidate| candidate.parent == Some(emitter.emitter))
                .count();
            if let Err(error) = EffectShaderSources::generate(
                &emitter.asset,
                parent_layout.as_ref(),
                child_count as u32,
            ) {
                errors.push(EffectBakeError {
                    emitter: emitter.emitter,
                    source: effect_graph.source_for_emitter(emitter.emitter),
                    error: BakeError::emitter(
                        emitter.emitter,
                        format!("shader generation failed: {error}"),
                    ),
                });
            }
        }
    }

    if errors.is_empty() {
        Ok(BakedEffect { emitters: baked })
    } else {
        Err(errors)
    }
}

/// Wrap a [`validation::TopologyError`] into a [`EffectBakeError`] for
/// [`bake_effect`]'s aggregate error list.
fn topology_error_to_effect_bake_error(
    effect_graph: &EffectGraph,
    e: validation::TopologyError,
) -> EffectBakeError {
    use validation::TopologySubject;
    let (subject, message) = (e.subject, e.message);
    match subject {
        TopologySubject::Emitter(id) => EffectBakeError {
            emitter: id,
            source: effect_graph.source_for_emitter(id),
            error: BakeError::emitter(id, message),
        },
        TopologySubject::Source(id) => EffectBakeError {
            emitter: effect_graph
                .emitter_for_source(id)
                .unwrap_or(id_placeholder_emitter()),
            source: Some(id),
            error: BakeError::source(id, message),
        },
        TopologySubject::Node(id) => {
            let emitter = effect_graph
                .emitter_owning_node(id)
                .unwrap_or(id_placeholder_emitter());
            EffectBakeError {
                emitter,
                source: effect_graph.source_for_emitter(emitter),
                error: BakeError::node(id, message),
            }
        }
        TopologySubject::Graph => EffectBakeError {
            emitter: id_placeholder_emitter(),
            source: None,
            error: BakeError {
                subject: BakeSubject::Graph,
                message,
            },
        },
    }
}

/// A placeholder [`EmitterId`] for a [`EffectBakeError`] whose topology error
/// has no emitter to attribute to (an orphan reference, or a graph-wide
/// problem).
///
/// `1` is always a valid, allocatable [`NonZeroU32`], and is never
/// dereferenced as a real lookup key by `bake_effect` callers — it is only ever
/// compared or displayed alongside the error's own descriptive message.
fn id_placeholder_emitter() -> EmitterId {
    EmitterId::new(1).unwrap()
}

/// Bake a document's single, connected, CPU-driven emitter into an
/// [`EffectAsset`].
///
/// A strict convenience for callers that only ever handle one emitter (e.g.
/// the offline `.hnb` → `EffectAsset` processor): requires `effect_graph` to
/// contain exactly one emitter, driven by exactly one
/// [`SourceKind::CpuSpawner`] source. Rejects a GPU-driven emitter, a
/// multi-emitter document, or a disconnected emitter with a [`BakeError`]
/// directing the caller to [`bake_effect`] instead of attempting a partial or
/// best-guess bake.
///
/// [`SourceKind::CpuSpawner`]: crate::model::SourceKind::CpuSpawner
pub fn bake(
    effect_graph: &EffectGraph,
    registry: &TypeRegistry,
) -> Result<EffectAsset, Vec<BakeError>> {
    if effect_graph.emitters.len() != 1 || effect_graph.sources.len() != 1 {
        return Err(vec![BakeError {
            subject: BakeSubject::Graph,
            message: format!(
                "expected exactly one emitter and one spawn source for a single-emitter bake, \
                 found {} emitter(s) and {} source(s); use bake_effect for multi-emitter documents",
                effect_graph.emitters.len(),
                effect_graph.sources.len()
            ),
        }]);
    }
    let emitter = &effect_graph.emitters[0];
    let source = &effect_graph.sources[0];
    let SourceKind::CpuSpawner { settings } = &source.kind else {
        return Err(vec![BakeError::source(
            source.id,
            "GPU-driven emitters are not supported by the single-emitter bake; use bake_effect",
        )]);
    };
    if effect_graph.source_for_emitter(emitter.id) != Some(source.id) {
        return Err(vec![BakeError::emitter(
            emitter.id,
            "emitter is not linked to the document's only spawn source; use bake_effect",
        )]);
    }
    bake_emitter(emitter, registry, *settings, &HashMap::new())
}

/// Expression nodes that participate in the baked module.
///
/// Every link endpoint plus every operand-bearing expression node, so an
/// inline-defaulted operator with no incoming links is still built.
fn expr_participants(graph: &EmitterGraph) -> Vec<NodeId> {
    let mut seen = Vec::new();
    let push = |id: NodeId, seen: &mut Vec<NodeId>| {
        if !seen.contains(&id) {
            seen.push(id);
        }
    };
    for link in &graph.links {
        push(link.from.node, &mut seen);
        push(link.to.node, &mut seen);
    }
    for node in &graph.nodes {
        if let NodePayload::Expr(expr) = &node.payload
            && !expr.input_ports().is_empty()
        {
            push(node.id, &mut seen);
        }
    }
    seen
}

#[cfg(test)]
mod tests {
    use bevy_hanabi::{
        Attribute, Expr, SimulationCondition, SimulationSpace, SpawnerSettings, Value,
        graph::expr::{BinaryOperator, UnaryOperator},
    };

    use super::*;
    use crate::model::{GraphLink, GraphNode, InputSlot, PortRef};

    /// A fixed emitter id for tests that build a bare [`EmitterGraph`] outside
    /// any [`EffectGraph`] document.
    fn test_emitter_id() -> EmitterId {
        EmitterId::new(1).expect("1 is a valid NonZeroU32")
    }

    fn graph_with(
        nodes: Vec<GraphNode>,
        links: Vec<GraphLink>,
        props: Vec<PropertyDef>,
    ) -> EmitterGraph {
        EmitterGraph {
            id: test_emitter_id(),
            name: "t".into(),
            capacity: 32,
            simulation_space: SimulationSpace::Global,
            simulation_condition: SimulationCondition::Always,
            z_layer_2d: 0.0,
            properties: props,
            texture_slots: vec![],
            nodes,
            stacks: vec![],
            links,
        }
    }

    fn expr_node(id: u32, expr: ExprNode, inputs: Vec<InputSlot>) -> GraphNode {
        GraphNode {
            id: NodeId::new(id).unwrap(),
            payload: NodePayload::Expr(expr),
            inputs,
        }
    }

    fn pid(n: u32) -> PropertyId {
        PropertyId::new(n).unwrap()
    }

    fn prop_def(id: u32, name: &str, default: Value, exposed: bool) -> PropertyDef {
        PropertyDef {
            id: pid(id),
            name: name.into(),
            default,
            exposed,
        }
    }

    #[test]
    fn bakes_binary_with_link_and_inline_default() {
        // n1 = attr(position); n2 = n1 + lit(2.0 via inline default on rhs)
        let n1 = expr_node(1, ExprNode::Attribute(Attribute::POSITION), vec![]);
        let n2 = expr_node(
            2,
            ExprNode::Binary(BinaryOperator::Add),
            vec![InputSlot {
                name: "rhs".into(),
                default: Value::from(2.0f32).into(),
            }],
        );
        let link = GraphLink {
            from: PortRef {
                node: NodeId::new(1).unwrap(),
                port: "out".into(),
            },
            to: PortRef {
                node: NodeId::new(2).unwrap(),
                port: "lhs".into(),
            },
        };
        let graph = graph_with(vec![n1, n2], vec![link], vec![]);

        let (module, handles) = bake_module(&graph).expect("bake");
        assert_eq!(handles.len(), 2);
        let top = handles[&NodeId::new(2).unwrap()];
        assert!(matches!(module.get(top), Some(Expr::Binary { .. })));
    }

    #[test]
    fn exposed_property_becomes_module_property() {
        // A property reference consumed by a unary so it participates in baking.
        let prop = expr_node(1, ExprNode::Property(pid(10)), vec![]);
        let unary = expr_node(
            2,
            ExprNode::Unary(bevy_hanabi::graph::expr::UnaryOperator::Abs),
            vec![],
        );
        let link = GraphLink {
            from: PortRef {
                node: NodeId::new(1).unwrap(),
                port: "out".into(),
            },
            to: PortRef {
                node: NodeId::new(2).unwrap(),
                port: "in".into(),
            },
        };
        let graph = graph_with(
            vec![prop, unary],
            vec![link],
            vec![prop_def(10, "speed", Value::from(4.0f32), true)],
        );

        let (module, _) = bake_module(&graph).expect("bake");
        assert_eq!(module.properties().len(), 1);
        assert_eq!(module.properties()[0].name(), "speed");
    }

    #[test]
    fn edit_only_property_is_inlined() {
        let n1 = expr_node(1, ExprNode::Property(pid(10)), vec![]);
        let unary = expr_node(
            2,
            ExprNode::Unary(bevy_hanabi::graph::expr::UnaryOperator::Abs),
            vec![],
        );
        let link = GraphLink {
            from: PortRef {
                node: NodeId::new(1).unwrap(),
                port: "out".into(),
            },
            to: PortRef {
                node: NodeId::new(2).unwrap(),
                port: "in".into(),
            },
        };
        let graph = graph_with(
            vec![n1, unary],
            vec![link],
            vec![prop_def(10, "tweak", Value::from(7.0f32), false)],
        );

        let (module, handles) = bake_module(&graph).expect("bake");
        assert!(module.properties().is_empty());
        let lit = handles[&NodeId::new(1).unwrap()];
        assert!(matches!(module.get(lit), Some(Expr::Literal(_))));
    }

    #[test]
    fn unexposed_property_ref_records_node_site() {
        let n1 = expr_node(1, ExprNode::Property(pid(10)), vec![]);
        let graph = graph_with(
            vec![n1],
            vec![],
            vec![prop_def(10, "tweak", Value::from(7.0f32), false)],
        );

        let mut module = Module::default();
        let mut errors = Vec::new();
        let props = bake_properties(&graph, &mut module, &mut errors);
        let mut baker = ExprBaker::new(&graph, &props, &mut module);
        let node = NodeId::new(1).unwrap();
        let h = baker.resolve(node, &mut errors).expect("resolve");
        assert!(errors.is_empty());
        assert_eq!(
            baker.literal_sites.get(&LiteralSite::Node(node)).copied(),
            Some(h),
            "an unexposed property reference records a Node site for its inlined literal"
        );
        assert!(matches!(baker.module.get(h), Some(Expr::Literal(_))));
    }

    #[test]
    fn detects_cycle() {
        // n1(unary) -> n2(unary) -> n1 : a cycle.
        let n1 = expr_node(
            1,
            ExprNode::Unary(bevy_hanabi::graph::expr::UnaryOperator::Abs),
            vec![],
        );
        let n2 = expr_node(
            2,
            ExprNode::Unary(bevy_hanabi::graph::expr::UnaryOperator::Abs),
            vec![],
        );
        let links = vec![
            GraphLink {
                from: PortRef {
                    node: NodeId::new(1).unwrap(),
                    port: "out".into(),
                },
                to: PortRef {
                    node: NodeId::new(2).unwrap(),
                    port: "in".into(),
                },
            },
            GraphLink {
                from: PortRef {
                    node: NodeId::new(2).unwrap(),
                    port: "out".into(),
                },
                to: PortRef {
                    node: NodeId::new(1).unwrap(),
                    port: "in".into(),
                },
            },
        ];
        let graph = graph_with(vec![n1, n2], links, vec![]);

        let errors = bake_module(&graph).unwrap_err();
        assert!(errors.iter().any(|e| e.message.contains("cycle")));
    }

    #[test]
    fn unknown_property_errors() {
        let n1 = expr_node(1, ExprNode::Property(pid(99)), vec![]);
        let unary = expr_node(
            2,
            ExprNode::Unary(bevy_hanabi::graph::expr::UnaryOperator::Abs),
            vec![],
        );
        let link = GraphLink {
            from: PortRef {
                node: NodeId::new(1).unwrap(),
                port: "out".into(),
            },
            to: PortRef {
                node: NodeId::new(2).unwrap(),
                port: "in".into(),
            },
        };
        let graph = graph_with(vec![n1, unary], vec![link], vec![]);

        let errors = bake_module(&graph).unwrap_err();
        assert!(errors.iter().any(|e| {
            e.subject == BakeSubject::Node(NodeId::new(1).unwrap())
                && e.message.contains("unknown property")
        }));
    }

    #[test]
    fn duplicate_exposed_property_name_errors() {
        // Exposed properties become runtime Module properties keyed by name; a
        // name collision is an inconsistency that blocks baking (but never panics).
        let graph = graph_with(
            vec![],
            vec![],
            vec![
                prop_def(10, "dup", Value::from(1.0f32), true),
                prop_def(11, "dup", Value::from(2.0f32), true),
            ],
        );

        let errors = bake_module(&graph).unwrap_err();
        // The error is attributed to the conflicting (second) property so the UI
        // can link straight to it.
        assert!(errors.iter().any(|e| {
            e.subject == BakeSubject::Property(pid(11))
                && e.message.contains("share the name 'dup'")
        }));
    }

    #[test]
    fn duplicate_edit_only_property_name_is_tolerated() {
        // Non-exposed properties are baked to literals and referenced by id, so a
        // shared display name is harmless and must not fail the bake.
        let graph = graph_with(
            vec![],
            vec![],
            vec![
                prop_def(10, "tweak", Value::from(1.0f32), false),
                prop_def(11, "tweak", Value::from(2.0f32), false),
            ],
        );

        let (module, _) = bake_module(&graph).expect("edit-only duplicates are harmless");
        assert!(module.properties().is_empty());
    }

    #[test]
    fn distinct_ids_resolve_independently() {
        // Two edit-only properties share a name but have distinct ids; each
        // reference resolves to its own value via id, not the shared name.
        let r1 = expr_node(1, ExprNode::Property(pid(10)), vec![]);
        let r2 = expr_node(2, ExprNode::Property(pid(11)), vec![]);
        let add = expr_node(3, ExprNode::Binary(BinaryOperator::Add), vec![]);
        let links = vec![
            GraphLink {
                from: PortRef {
                    node: NodeId::new(1).unwrap(),
                    port: "out".into(),
                },
                to: PortRef {
                    node: NodeId::new(3).unwrap(),
                    port: "lhs".into(),
                },
            },
            GraphLink {
                from: PortRef {
                    node: NodeId::new(2).unwrap(),
                    port: "out".into(),
                },
                to: PortRef {
                    node: NodeId::new(3).unwrap(),
                    port: "rhs".into(),
                },
            },
        ];
        let graph = graph_with(
            vec![r1, r2, add],
            links,
            vec![
                prop_def(10, "same", Value::from(1.0f32), false),
                prop_def(11, "same", Value::from(2.0f32), false),
            ],
        );

        let (module, handles) = bake_module(&graph).expect("bake");
        // Both references baked to distinct literal expressions.
        assert!(matches!(
            module.get(handles[&NodeId::new(1).unwrap()]),
            Some(Expr::Literal(_))
        ));
        assert!(matches!(
            module.get(handles[&NodeId::new(2).unwrap()]),
            Some(Expr::Literal(_))
        ));
    }

    // --- Modifier baking (B2) ---

    use std::collections::BTreeMap;

    use bevy::{asset::AssetPlugin, ecs::reflect::AppTypeRegistry, prelude::*, reflect::TypePath};
    use bevy_hanabi::{
        ColorBlendMask, ColorBlendMode, CpuValue, ParticleTextureModifier, SetColorModifier,
        SetPositionSphereModifier,
    };

    use crate::model::ModifierNodeData;

    /// A type registry populated with all built-in modifiers.
    ///
    /// Includes their [`ReflectModifier`] factories, via `bevy_hanabi`'s own
    /// registration.
    fn test_registry() -> AppTypeRegistry {
        let registry = AppTypeRegistry::default();
        bevy_hanabi::register_modifiers(&registry);
        registry
    }

    fn modifier_node(
        id: u32,
        type_path: &str,
        config: BTreeMap<crate::model::SharedStr, EditValue>,
        inputs: Vec<InputSlot>,
    ) -> GraphNode {
        GraphNode {
            id: NodeId::new(id).unwrap(),
            payload: NodePayload::Modifier(ModifierNodeData::Known {
                type_path: type_path.into(),
                config,
            }),
            inputs,
        }
    }

    /// Drive [`ExprBaker::bake_modifier`] for a single node.
    ///
    /// Resolves operands on demand against `graph`.
    fn bake_one(
        graph: &EmitterGraph,
        registry: &TypeRegistry,
        node_id: NodeId,
    ) -> (Option<BoxedModifier>, Vec<BakeError>) {
        let mut module = Module::default();
        let mut errors = Vec::new();
        let props = bake_properties(graph, &mut module, &mut errors);
        let mut baker = ExprBaker::new(graph, &props, &mut module);
        let baked = baker.bake_modifier(node_id, registry, &HashMap::new(), &mut errors);
        (baked, errors)
    }

    #[test]
    fn unlinked_emit_spawn_event_is_omitted() {
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            AssetPlugin::default(),
            crate::modifier_registry::ModifierRegistryPlugin,
        ));
        let registry = app.world().resource::<AppTypeRegistry>().read();
        let emitter = modifier_node(
            1,
            EmitSpawnEventModifier::type_path(),
            BTreeMap::new(),
            vec![InputSlot {
                name: "count".into(),
                default: Value::from(1_u32).into(),
            }],
        );
        let graph = graph_with(vec![emitter], vec![], vec![]);

        let (baked, errors) = bake_one(&graph, &registry, NodeId::new(1).unwrap());
        assert!(baked.is_none());
        assert!(errors.is_empty());
    }

    #[test]
    fn bakes_modifier_enum_and_flags_config() {
        let mut config = BTreeMap::new();
        config.insert(
            "blend".into(),
            EditValue::Enum {
                type_path: ColorBlendMode::type_path().into(),
                variant: "Add".into(),
            },
        );
        config.insert(
            "mask".into(),
            EditValue::Flags {
                type_path: ColorBlendMask::type_path().into(),
                bits: ColorBlendMask::RGB.bits() as u64,
            },
        );
        config.insert(
            "color".into(),
            EditValue::CpuVec4(CpuValue::Single(Vec4::new(0.2, 0.4, 0.6, 1.0))),
        );
        let node = modifier_node(1, SetColorModifier::type_path(), config, vec![]);
        let graph = graph_with(vec![node], vec![], vec![]);

        let (baked, errors) = bake_one(&graph, &test_registry().read(), NodeId::new(1).unwrap());
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        let m = baked.expect("baked");
        assert!(m.as_render().is_some(), "expected a render modifier");
        let scm = m
            .as_reflect()
            .downcast_ref::<SetColorModifier>()
            .expect("SetColorModifier");
        assert_eq!(scm.blend, ColorBlendMode::Add);
        assert_eq!(scm.mask, ColorBlendMask::RGB);
        assert_eq!(scm.color, CpuValue::Single(Vec4::new(0.2, 0.4, 0.6, 1.0)));
    }

    #[test]
    fn bakes_modifier_ports_from_inline_defaults() {
        // No links: each required port is fed by its inline-default literal.
        let node = modifier_node(
            1,
            SetPositionSphereModifier::type_path(),
            BTreeMap::new(),
            vec![
                InputSlot {
                    name: "center".into(),
                    default: Value::from(Vec3::new(1.0, 2.0, 3.0)).into(),
                },
                InputSlot {
                    name: "radius".into(),
                    default: Value::from(5.0_f32).into(),
                },
            ],
        );
        let graph = graph_with(vec![node], vec![], vec![]);

        let mut module = Module::default();
        let mut errors = Vec::new();
        let props = bake_properties(&graph, &mut module, &mut errors);
        let mut baker = ExprBaker::new(&graph, &props, &mut module);
        let baked = baker
            .bake_modifier(
                NodeId::new(1).unwrap(),
                &test_registry().read(),
                &HashMap::new(),
                &mut errors,
            )
            .expect("baked");
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");

        let m = baked;
        assert!(m.as_render().is_none(), "expected a plain modifier");
        let spm = m
            .as_reflect()
            .downcast_ref::<SetPositionSphereModifier>()
            .expect("SetPositionSphereModifier");
        // The port fields point at the inline-default literals in the module.
        assert_eq!(
            module.get(spm.radius),
            Some(&Expr::Literal(bevy_hanabi::graph::expr::LiteralExpr::new(
                5.0_f32
            )))
        );
        assert_eq!(
            module.get(spm.center),
            Some(&Expr::Literal(bevy_hanabi::graph::expr::LiteralExpr::new(
                Vec3::new(1.0, 2.0, 3.0)
            )))
        );
    }

    #[test]
    fn normalizes_modifier_direction_axes() {
        let axis = Vec3::new(0.0, 1.94, 1.95);
        let node = modifier_node(
            1,
            SetPositionCircleModifier::type_path(),
            BTreeMap::new(),
            vec![
                InputSlot {
                    name: "center".into(),
                    default: Value::from(Vec3::ZERO).into(),
                },
                InputSlot {
                    name: "axis".into(),
                    default: Value::from(axis).into(),
                },
                InputSlot {
                    name: "radius".into(),
                    default: Value::from(1.0_f32).into(),
                },
            ],
        );
        let graph = graph_with(vec![node], vec![], vec![]);

        let mut module = Module::default();
        let mut errors = Vec::new();
        let props = bake_properties(&graph, &mut module, &mut errors);
        let mut baker = ExprBaker::new(&graph, &props, &mut module);
        let baked = baker
            .bake_modifier(
                NodeId::new(1).unwrap(),
                &test_registry().read(),
                &HashMap::new(),
                &mut errors,
            )
            .expect("baked");
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");

        let modifier = baked
            .as_reflect()
            .downcast_ref::<SetPositionCircleModifier>()
            .expect("SetPositionCircleModifier");
        let Some(Expr::Unary { op, expr }) = module.get(modifier.axis) else {
            panic!("expected the axis to bake as a unary expression");
        };
        assert_eq!(*op, UnaryOperator::Normalize);
        assert_eq!(
            module.get(*expr),
            Some(&Expr::Literal(bevy_hanabi::graph::expr::LiteralExpr::new(
                axis
            )))
        );
    }

    // --- Texture slot baking (Phase D) ---

    use crate::model::{ImageBinding, SlotId, TextureSlotDef};

    fn slot_def(id: u32, name: &str) -> TextureSlotDef {
        TextureSlotDef {
            id: SlotId::new(id).unwrap(),
            name: name.into(),
        }
    }

    fn graph_with_textures(
        nodes: Vec<GraphNode>,
        links: Vec<GraphLink>,
        texture_slots: Vec<TextureSlotDef>,
    ) -> EmitterGraph {
        let mut graph = graph_with(nodes, links, vec![]);
        graph.texture_slots = texture_slots;
        graph
    }

    fn sampler_node(id: u32, image: Option<ImageBinding>) -> GraphNode {
        let mut inputs = vec![InputSlot {
            name: "coordinates".into(),
            default: Value::from(Vec2::ZERO).into(),
        }];
        if let Some(binding) = image {
            inputs.insert(
                0,
                InputSlot {
                    name: "image".into(),
                    default: binding.into(),
                },
            );
        }
        expr_node(id, ExprNode::TextureSample, inputs)
    }

    fn image_link(from: u32, to: u32, to_port: &str) -> GraphLink {
        GraphLink {
            from: PortRef {
                node: NodeId::new(from).unwrap(),
                port: "out".into(),
            },
            to: PortRef {
                node: NodeId::new(to).unwrap(),
                port: to_port.into(),
            },
        }
    }

    #[test]
    fn modifier_texture_slot_bakes_asset_to_static_u32_index() {
        // An inline asset binding on the modifier's texture port allocates a
        // slot; the field bakes to its static `u32` index.
        let node = modifier_node(
            1,
            ParticleTextureModifier::type_path(),
            BTreeMap::new(),
            vec![InputSlot {
                name: "texture_slot".into(),
                default: ImageBinding::Asset("ramps/fire.png".into()).into(),
            }],
        );
        let graph = graph_with(vec![node], vec![], vec![]);

        let mut module = Module::default();
        let mut errors = Vec::new();
        let props = bake_properties(&graph, &mut module, &mut errors);
        let mut baker = ExprBaker::new(&graph, &props, &mut module);
        let baked = baker
            .bake_modifier(
                NodeId::new(1).unwrap(),
                &test_registry().read(),
                &HashMap::new(),
                &mut errors,
            )
            .expect("baked");
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        assert_eq!(baker.texture_plan.len(), 1);
        assert!(matches!(baker.texture_plan[0], PlannedImage::Asset(_)));
        drop(baker);

        let ptm = baked
            .as_reflect()
            .downcast_ref::<ParticleTextureModifier>()
            .expect("ParticleTextureModifier");
        assert_eq!(ptm.texture_slot, 0);
        assert_eq!(module.texture_layout().layout.len(), 1);
    }

    #[test]
    fn texture_sample_bakes_slot_metadata() {
        // A sampler's inline image binding resolves to a static 2D texture
        // slot. The upstream expression owns that slot metadata directly.
        let node = sampler_node(1, Some(ImageBinding::Asset("fire.png".into())));
        let graph = graph_with(vec![node], vec![], vec![]);

        let mut module = Module::default();
        let mut errors = Vec::new();
        let props = bake_properties(&graph, &mut module, &mut errors);
        let mut baker = ExprBaker::new(&graph, &props, &mut module);
        let handle = baker
            .resolve(NodeId::new(1).unwrap(), &mut errors)
            .expect("baked");
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        assert_eq!(baker.texture_plan.len(), 1);
        drop(baker);

        let Some(Expr::TextureSample(tse)) = module.get(handle) else {
            panic!("expected a TextureSample expression");
        };
        assert_eq!(tse.slot_index, 0);
        assert_eq!(tse.slot_dimension, SlotDimension::D2);
        assert_eq!(tse.array_index, None);
        assert_eq!(
            module.texture_layout().layout[0].dimension,
            SlotDimension::D2
        );
        assert_eq!(
            module.get(tse.coordinates),
            Some(&Expr::Literal(bevy_hanabi::graph::expr::LiteralExpr::new(
                Vec2::ZERO
            )))
        );
    }

    #[test]
    fn image_node_fan_out_shares_one_slot() {
        // Two samplers fed by the same Image node share its single slot.
        let image = expr_node(
            1,
            ExprNode::Image(ImageBinding::Asset("a.png".into())),
            vec![],
        );
        let graph = graph_with(
            vec![image, sampler_node(2, None), sampler_node(3, None)],
            vec![image_link(1, 2, "image"), image_link(1, 3, "image")],
            vec![],
        );

        let mut module = Module::default();
        let mut errors = Vec::new();
        let props = bake_properties(&graph, &mut module, &mut errors);
        let mut baker = ExprBaker::new(&graph, &props, &mut module);
        baker
            .resolve(NodeId::new(2).unwrap(), &mut errors)
            .expect("baked s2");
        baker
            .resolve(NodeId::new(3).unwrap(), &mut errors)
            .expect("baked s3");
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        assert_eq!(baker.texture_plan.len(), 1, "fan-out must share one slot");
    }

    #[test]
    fn identical_assets_across_image_nodes_and_inline_bindings_share_one_slot() {
        let first_image = expr_node(
            1,
            ExprNode::Image(ImageBinding::Asset("shared.png".into())),
            vec![],
        );
        let second_image = expr_node(
            2,
            ExprNode::Image(ImageBinding::Asset("shared.png".into())),
            vec![],
        );
        let graph = graph_with(
            vec![
                first_image,
                second_image,
                sampler_node(3, None),
                sampler_node(4, None),
                sampler_node(5, Some(ImageBinding::Asset("shared.png".into()))),
            ],
            vec![image_link(1, 3, "image"), image_link(2, 4, "image")],
            vec![],
        );

        let mut module = Module::default();
        let mut errors = Vec::new();
        let props = bake_properties(&graph, &mut module, &mut errors);
        let mut baker = ExprBaker::new(&graph, &props, &mut module);
        let handles = [3, 4, 5].map(|id| {
            baker
                .resolve(NodeId::new(id).unwrap(), &mut errors)
                .expect("baked")
        });
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        assert_eq!(baker.texture_plan.len(), 1);
        assert_eq!(
            baker.texture_plan[0],
            PlannedImage::Asset("shared.png".into())
        );
        drop(baker);

        assert_eq!(module.texture_layout().layout.len(), 1);
        for handle in handles {
            let Some(Expr::TextureSample(sample)) = module.get(handle) else {
                panic!("expected a TextureSample expression");
            };
            assert_eq!(sample.slot_index, 0);
            assert_eq!(sample.slot_dimension, SlotDimension::D2);
            assert_eq!(sample.array_index, None);
        }
    }

    #[test]
    fn runtime_slots_stay_leading_and_unbound_bindings_stay_distinct() {
        let graph = graph_with_textures(
            vec![
                sampler_node(1, Some(ImageBinding::Slot(SlotId::new(20).unwrap()))),
                sampler_node(2, Some(ImageBinding::Slot(SlotId::new(10).unwrap()))),
                sampler_node(3, Some(ImageBinding::Unbound)),
                sampler_node(4, Some(ImageBinding::Unbound)),
            ],
            vec![],
            vec![slot_def(10, "first"), slot_def(20, "second")],
        );

        let mut module = Module::default();
        let mut errors = Vec::new();
        let props = bake_properties(&graph, &mut module, &mut errors);
        let mut baker = ExprBaker::new(&graph, &props, &mut module);
        let handles = [1, 2, 3, 4].map(|id| {
            baker
                .resolve(NodeId::new(id).unwrap(), &mut errors)
                .expect("baked")
        });
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        assert_eq!(
            baker.texture_plan,
            vec![
                PlannedImage::Runtime("first".into()),
                PlannedImage::Runtime("second".into()),
                PlannedImage::Unbound,
                PlannedImage::Unbound,
            ]
        );
        drop(baker);

        assert_eq!(module.texture_layout().layout.len(), 4);
        for (handle, slot) in handles.into_iter().zip([1, 0, 2, 3]) {
            let Some(Expr::TextureSample(sample)) = module.get(handle) else {
                panic!("expected a TextureSample expression");
            };
            assert_eq!(sample.slot_index, slot);
            assert_eq!(sample.slot_dimension, SlotDimension::D2);
            assert_eq!(sample.array_index, None);
        }
    }

    #[test]
    fn host_slot_binding_reuses_reserved_index() {
        // A registry texture slot reserves index 0; an `ImageBinding::Slot`
        // referencing it reuses that index without allocating a new one.
        let node = sampler_node(1, Some(ImageBinding::Slot(SlotId::new(7).unwrap())));
        let graph = graph_with_textures(vec![node], vec![], vec![slot_def(7, "noise")]);

        let mut module = Module::default();
        let mut errors = Vec::new();
        let props = bake_properties(&graph, &mut module, &mut errors);
        let mut baker = ExprBaker::new(&graph, &props, &mut module);
        let handle = baker
            .resolve(NodeId::new(1).unwrap(), &mut errors)
            .expect("baked");
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        assert_eq!(baker.texture_plan.len(), 1);
        assert!(matches!(baker.texture_plan[0], PlannedImage::Runtime(_)));
        drop(baker);

        let Some(Expr::TextureSample(tse)) = module.get(handle) else {
            panic!("expected a TextureSample expression");
        };
        assert_eq!(tse.slot_index, 0);
        assert_eq!(tse.slot_dimension, SlotDimension::D2);
        assert_eq!(tse.array_index, None);
    }

    #[test]
    fn select_image_with_constant_index_bakes_selected_slot() {
        // A constant `index` selects one input; only the selected image gets a
        // slot, recorded in the sampler's static slot metadata.
        let a = expr_node(
            1,
            ExprNode::Image(ImageBinding::Asset("a.png".into())),
            vec![],
        );
        let b = expr_node(
            2,
            ExprNode::Image(ImageBinding::Asset("b.png".into())),
            vec![],
        );
        let select = expr_node(
            3,
            ExprNode::SelectImage { count: 2 },
            vec![InputSlot {
                name: "index".into(),
                default: Value::from(1u32).into(),
            }],
        );
        let graph = graph_with(
            vec![a, b, select, sampler_node(4, None)],
            vec![
                image_link(1, 3, "image0"),
                image_link(2, 3, "image1"),
                image_link(3, 4, "image"),
            ],
            vec![],
        );

        let mut module = Module::default();
        let mut errors = Vec::new();
        let props = bake_properties(&graph, &mut module, &mut errors);
        let mut baker = ExprBaker::new(&graph, &props, &mut module);
        let handle = baker
            .resolve(NodeId::new(4).unwrap(), &mut errors)
            .expect("baked");
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        assert_eq!(baker.texture_plan.len(), 1);
        match &baker.texture_plan[0] {
            PlannedImage::Asset(path) => {
                assert!(path.path().to_str().unwrap().contains("b.png"))
            }
            other => panic!("expected b.png asset slot, got {other:?}"),
        }
        drop(baker);

        let Some(Expr::TextureSample(tse)) = module.get(handle) else {
            panic!("expected a TextureSample expression");
        };
        assert_eq!(tse.slot_index, 0);
        assert_eq!(tse.slot_dimension, SlotDimension::D2);
        assert_eq!(tse.array_index, None);
    }

    #[test]
    fn select_image_with_runtime_index_errors() {
        // A non-constant `index` cannot bake in this bevy_hanabi revision.
        let select = expr_node(1, ExprNode::SelectImage { count: 2 }, vec![]);
        let graph = graph_with(
            vec![select, sampler_node(2, None)],
            vec![image_link(1, 2, "image")],
            vec![],
        );

        let mut module = Module::default();
        let mut errors = Vec::new();
        let props = bake_properties(&graph, &mut module, &mut errors);
        let mut baker = ExprBaker::new(&graph, &props, &mut module);
        let result = baker.resolve(NodeId::new(2).unwrap(), &mut errors);
        assert!(result.is_none());
        assert!(
            errors
                .iter()
                .any(|e| e.message.contains("compile-time constant")),
            "expected a runtime-selection error, got: {errors:?}"
        );
    }

    #[test]
    fn unregistered_modifier_type_errors() {
        let node = modifier_node(1, "not::a::real::Modifier", BTreeMap::new(), vec![]);
        let graph = graph_with(vec![node], vec![], vec![]);

        let (baked, errors) = bake_one(&graph, &test_registry().read(), NodeId::new(1).unwrap());
        assert!(baked.is_none());
        assert!(errors.iter().any(|e| {
            e.subject == BakeSubject::Node(NodeId::new(1).unwrap())
                && e.message.contains("not in the type registry")
        }));
    }

    // --- Whole-graph assembly (B3) ---

    use crate::model::{GraphStack, StackId};

    fn graph_with_stacks(nodes: Vec<GraphNode>, stacks: Vec<GraphStack>) -> EmitterGraph {
        EmitterGraph {
            id: test_emitter_id(),
            name: "t".into(),
            capacity: 32,
            simulation_space: SimulationSpace::Global,
            simulation_condition: SimulationCondition::Always,
            z_layer_2d: 0.0,
            properties: vec![],
            texture_slots: vec![],
            nodes,
            stacks,
            links: vec![],
        }
    }

    fn stack(id: u32, group: ModifierGroup, members: Vec<u32>) -> GraphStack {
        GraphStack {
            id: StackId::new(id).unwrap(),
            group,
            members: members
                .into_iter()
                .map(|m| NodeId::new(m).unwrap())
                .collect(),
        }
    }

    #[test]
    fn bakes_whole_graph_into_effect_asset() {
        // Init: position-sphere (ports from inline defaults). Render: set-color.
        let pos = modifier_node(
            1,
            SetPositionSphereModifier::type_path(),
            BTreeMap::new(),
            vec![
                InputSlot {
                    name: "center".into(),
                    default: Value::from(Vec3::ZERO).into(),
                },
                InputSlot {
                    name: "radius".into(),
                    default: Value::from(2.0_f32).into(),
                },
            ],
        );
        let color = modifier_node(2, SetColorModifier::type_path(), BTreeMap::new(), vec![]);
        let graph = graph_with_stacks(
            vec![pos, color],
            vec![
                stack(1, ModifierGroup::Init, vec![1]),
                stack(2, ModifierGroup::Render, vec![2]),
            ],
        );

        let asset = bake_emitter(
            &graph,
            &test_registry().read(),
            SpawnerSettings::default(),
            &HashMap::new(),
        )
        .expect("bake");
        assert_eq!(asset.name, "t");
        assert_eq!(asset.capacity(), 32);
        assert_eq!(asset.init_modifiers().count(), 1);
        assert_eq!(asset.update_modifiers().count(), 0);
        assert_eq!(asset.render_modifiers().count(), 1);
        assert!(
            asset
                .init_modifiers()
                .next()
                .unwrap()
                .as_reflect()
                .downcast_ref::<SetPositionSphereModifier>()
                .is_some()
        );
        assert!(
            asset
                .render_modifiers()
                .next()
                .unwrap()
                .as_modifier()
                .as_reflect()
                .downcast_ref::<SetColorModifier>()
                .is_some()
        );
    }

    #[test]
    fn provenance_records_inline_default_sites() {
        let pos = modifier_node(
            1,
            SetPositionSphereModifier::type_path(),
            BTreeMap::new(),
            vec![
                InputSlot {
                    name: "center".into(),
                    default: Value::from(Vec3::ZERO).into(),
                },
                InputSlot {
                    name: "radius".into(),
                    default: Value::from(2.0_f32).into(),
                },
            ],
        );
        let graph = graph_with_stacks(vec![pos], vec![stack(1, ModifierGroup::Init, vec![1])]);

        let (asset, provenance) = bake_emitter_with_provenance(
            &graph,
            &test_registry().read(),
            SpawnerSettings::default(),
            &HashMap::new(),
        )
        .expect("bake");
        let sites = &provenance.literal_sites;

        let node = NodeId::new(1).unwrap();
        let radius = sites
            .get(&LiteralSite::Input {
                node,
                port: "radius".into(),
            })
            .copied()
            .expect("radius inline-default site recorded");
        assert!(matches!(asset.module().get(radius), Some(Expr::Literal(_))));
        assert!(sites.contains_key(&LiteralSite::Input {
            node,
            port: "center".into(),
        }));
    }

    #[test]
    fn render_modifier_in_init_stack_errors() {
        let color = modifier_node(1, SetColorModifier::type_path(), BTreeMap::new(), vec![]);
        let graph = graph_with_stacks(vec![color], vec![stack(1, ModifierGroup::Init, vec![1])]);

        let errors = match bake_emitter(
            &graph,
            &test_registry().read(),
            SpawnerSettings::default(),
            &HashMap::new(),
        ) {
            Err(errors) => errors,
            Ok(_) => panic!("expected a bake error"),
        };
        assert!(errors.iter().any(|e| {
            e.subject == BakeSubject::Node(NodeId::new(1).unwrap())
                && e.message.contains("render modifier placed in a Init stack")
        }));
    }

    /// A spawn-event node targeting a GPU source with no
    /// [`crate::model::SourceLink`] (unconnected) must not shift a real,
    /// connected sibling's derived channel — even when the orphan source's
    /// id sorts before the connected one.
    ///
    /// Regression test: `derive_emitter_child_indices` used to count *every*
    /// distinct event target in ascending `SourceId` order, with no regard
    /// for whether each target was actually connected to an emitter. An
    /// orphan GPU source with a smaller id than the real child's source
    /// shifted the real child's spawn-event nodes onto channel 1 instead of 0,
    /// even though `validation::validate_topology` considered the document
    /// valid (its fan-out check already filters to connected children
    /// only).
    #[test]
    fn derive_child_indices_ignores_unconnected_gpu_sources() {
        let parent_id = test_emitter_id();
        let orphan_event = NodeId::new(10).unwrap();
        let real_event = NodeId::new(11).unwrap();
        // Deliberately smaller than `connected_gpu` so a buggy ascending-sort
        // derivation would put it in channel 0 and shift the real child to 1.
        let orphan_gpu = SourceId::new(1).unwrap();
        let connected_gpu = SourceId::new(20).unwrap();
        let child_emitter = EmitterId::new(2).unwrap();

        let parent = graph_with(
            vec![
                modifier_node(
                    10,
                    EmitSpawnEventModifier::type_path(),
                    BTreeMap::new(),
                    vec![],
                ),
                modifier_node(
                    11,
                    EmitSpawnEventModifier::type_path(),
                    BTreeMap::new(),
                    vec![],
                ),
            ],
            vec![],
            vec![],
        );

        let effect_graph = EffectGraph {
            emitters: vec![parent],
            sources: vec![],
            // Only `connected_gpu` has a source link; `orphan_gpu` has none.
            source_links: vec![crate::model::SourceLink {
                source: connected_gpu,
                emitter: child_emitter,
            }],
            event_links: vec![
                crate::model::EventLink {
                    node: orphan_event,
                    target: orphan_gpu,
                },
                crate::model::EventLink {
                    node: real_event,
                    target: connected_gpu,
                },
            ],
            next_id: 100,
        };

        let channels = derive_emitter_child_indices(&effect_graph, parent_id);
        assert_eq!(channels.get(&real_event), Some(&0));
        assert_eq!(
            channels.get(&orphan_event),
            None,
            "a spawn-event node targeting an unconnected GPU source must not be assigned a channel"
        );
    }
}
