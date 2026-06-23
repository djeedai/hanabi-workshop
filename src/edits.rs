//! Edit-message scaffolding.
//!
//! See `crate::document` for the architectural commitment. The rule:
//!
//! * UI code emits [`EditRequest`] messages; it never mutates the document
//!   directly.
//! * [`apply_edits`] is the **only** caller of `DocumentContent::graph_mut` and
//!   the only system holding `Query<&mut DocumentContent>` and
//!   `ResMut<Assets<EffectAsset>>` for write access. Every edit mutates the
//!   canonical [`EffectGraph`] and re-bakes it into the preview
//!   [`EffectAsset`].
//! * [`crate::history::record_history`] maintains the per-document undo stack
//!   from [`EditApplied`] events.
//!
//! [`EffectAsset`]: bevy_hanabi::EffectAsset
//! [`EffectGraph`]: crate::effect_graph::model::EffectGraph

use std::any::TypeId;

use bevy::prelude::*;
use bevy_hanabi::{
    Attribute, EffectAsset, EffectProperties, ParticleEffect, SimulationCondition, SimulationSpace,
    SpawnerSettings, Value,
};

use crate::{
    document::{DocumentContent, DocumentSceneRoot, ModifierGroup},
    effect_graph::{
        bake::{LiteralSite, bake_preview_with_provenance},
        edit::{self as graph_edit, RemovedModifier, RemovedNode, RemovedTextureSlot},
        model::{
            EditValue, ExprNode, GraphLink, ImageBinding, InputSlot, NodeId, NodePayload,
            PropertyDef, PropertyId, SharedStr, SlotId,
        },
    },
    history::EditDirection,
    playback::PlaybackCommand,
    proxy::ProxyEffect,
};

/// A pending mutation to a document, addressed to one document entity.
#[derive(Message, Debug, Clone)]
pub struct EditRequest {
    pub doc: Entity,
    /// Where the request comes from. UI code always emits `Fresh`;
    /// `history_dispatch` rewrites Undo/Redo replays.
    pub direction: EditDirection,
    pub kind: EditKind,
}

impl EditRequest {
    pub fn new(doc: Entity, kind: EditKind) -> Self {
        Self {
            doc,
            direction: EditDirection::Fresh,
            kind,
        }
    }

    /// Flip `direction` to `Undo` (for replays popped from `History.past`).
    pub fn with_undo(mut self) -> Self {
        self.direction = EditDirection::Undo;
        self
    }

    /// Flip `direction` to `Redo` (for replays popped from `History.future`).
    pub fn with_redo(mut self) -> Self {
        self.direction = EditDirection::Redo;
        self
    }
}

/// The actual edit payload.
///
/// Each variant carries the *new* value and is applied to the document's
/// canonical [`EffectGraph`]; `apply_edits` reads the current value to build
/// the inverse, then re-bakes the graph into the preview asset.
///
/// [`EffectGraph`]: crate::effect_graph::model::EffectGraph
#[derive(Debug, Clone)]
pub enum EditKind {
    /// Rename the document (shown in the tab title). Mutates
    /// `DocumentContent.name`, not the graph. Not yet bound in the UI.
    #[allow(dead_code)]
    RenameDocument { new: String },

    // --- Effect header ---
    /// Set the effect's name (`EffectGraph.header.name`).
    SetEffectName { new: String },
    /// Set `EffectGraph.header.simulation_space`.
    SetSimulationSpace { new: SimulationSpace },
    /// Set `EffectGraph.header.simulation_condition`.
    SetSimulationCondition { new: SimulationCondition },
    /// Replace `EffectGraph.header.spawner`.
    SetSpawnerSettings { new: SpawnerSettings },
    /// Set `EffectGraph.header.capacity` (max live particle count).
    SetCapacity { new: u32 },
    /// Set `EffectGraph.header.z_layer_2d`.
    SetZLayer2d { new: f32 },

    // --- Modifier stacks ---
    /// Add a fresh modifier of `type_id` (a registered Hanabi modifier struct)
    /// into `group` at position `at`. The node's config and required input
    /// defaults are read from the registry factory's instance.
    AddModifierFromTemplate {
        group: ModifierGroup,
        /// `TypeId` of the Hanabi modifier struct. In-process only — never
        /// serialized.
        type_id: TypeId,
        at: usize,
    },
    /// Re-insert a previously-removed modifier node with its links. The inverse
    /// of [`EditKind::RemoveModifier`]; not emitted by the UI.
    InsertModifierNode { removed: RemovedModifier },
    /// Remove the modifier at `idx` in `group` (node + incident links).
    RemoveModifier { group: ModifierGroup, idx: usize },
    /// Move the modifier from `from` to `to` within `group`. `to` is the target
    /// index *after* removal of the source slot.
    MoveModifier {
        group: ModifierGroup,
        from: usize,
        to: usize,
    },
    /// Retarget a `SetAttributeModifier` node at `idx` in `group`. When the new
    /// attribute's value type differs from the node's inline `value` literal,
    /// that literal is reset so the baked modifier stays type-correct.
    ///
    /// - `reset_value: None`: forward path; apply computes the reset from
    ///   `new.default_value()` if needed. UI emits with `None`.
    /// - `reset_value: Some(v)`: undo path; force the literal to `v`.
    SetModifierAttribute {
        group: ModifierGroup,
        idx: usize,
        new: Attribute,
        reset_value: Option<Value>,
    },
    /// Set a non-expression configuration field of a modifier node to `new`
    /// (e.g. a data-less enum like `ShapeDimension`, or a flags field).
    /// Inverse: the same edit carrying the field's previous [`EditValue`].
    SetModifierConfig {
        node: NodeId,
        field: SharedStr,
        new: EditValue,
    },

    // --- Expression input defaults ---
    /// Set the inline default literal of an expression input port (an unlinked
    /// modifier or operator port). The "live tweak" path for slider drags.
    SetInputDefault {
        node: NodeId,
        port: SharedStr,
        new: Value,
    },
    /// Set the inline image binding of an image input port (an unlinked sampler
    /// `image` or modifier `texture_slot`). Structural: re-bakes. Inverse: the
    /// same edit carrying the previous binding.
    SetInputImageBinding {
        node: NodeId,
        port: SharedStr,
        binding: ImageBinding,
    },
    /// Set the value of a standalone `ExprNode::Literal` node (one whose value
    /// is the node itself, not an input-port default). Applied via
    /// `graph_edit::set_literal_value`; not yet emitted by any UI affordance.
    #[allow(dead_code)]
    SetLiteralValue { node: NodeId, new: Value },

    // --- Standalone expression nodes ---
    /// Add a standalone expression node (literal / operator / attribute /
    /// property / built-in) with its operand input defaults. Inverse:
    /// [`EditKind::RemoveNode`] with the freshly-allocated id.
    AddExprNode {
        expr: ExprNode,
        inputs: Vec<InputSlot>,
    },
    /// Remove a node with its incident links and any stack membership. Inverse:
    /// [`EditKind::InsertNode`].
    RemoveNode { id: NodeId },
    /// Re-insert a removed node with its links and membership. Used only as the
    /// inverse of [`EditKind::RemoveNode`]; not emitted by the UI.
    InsertNode { removed: RemovedNode },

    // --- Image source nodes and texture slots ---
    /// Add an image source node, initially unbound. Inverse:
    /// [`EditKind::RemoveNode`].
    AddImageNode,
    /// Set the binding of an image node (asset, texture slot, or unbound).
    /// Inverse: the same edit carrying the previous binding.
    SetImageNodeBinding { node: NodeId, binding: ImageBinding },
    /// Add a texture slot. Inverse: [`EditKind::RemoveTextureSlot`].
    AddTextureSlot,
    /// Remove a texture slot. Inverse: [`EditKind::InsertTextureSlot`].
    RemoveTextureSlot { id: SlotId },
    /// Re-insert a removed texture slot at its original index. Used only as the
    /// inverse of [`EditKind::RemoveTextureSlot`]; not emitted by the UI.
    InsertTextureSlot { removed: RemovedTextureSlot },
    /// Rename a texture slot. Inverse: the same edit carrying the old name.
    RenameTextureSlot { id: SlotId, new: SharedStr },
    /// Move a texture slot to a new index (reassigning sampling indices).
    /// Inverse: the same edit carrying the old index.
    ReorderTextureSlot { id: SlotId, to: usize },

    // --- Links ---
    /// Connect an output port to an input port. The graph view validates the
    /// connection (type, cycles, stage order) before emitting this. Inverse:
    /// [`EditKind::AddLink`] restoring any displaced link, else
    /// [`EditKind::RemoveLink`].
    AddLink { link: GraphLink },
    /// Disconnect the link targeting an input port. Inverse:
    /// [`EditKind::AddLink`].
    RemoveLink { link: GraphLink },

    // --- User properties (addressed by stable id) ---
    /// Add a brand-new property. Inverse: [`EditKind::RemoveProperty`] with the
    /// freshly-allocated id.
    AddProperty {
        name: String,
        value: Value,
        exposed: bool,
    },
    /// Remove a property by id. Each `Property` reference is demoted to a
    /// `Literal` of the property's default. Inverse:
    /// [`EditKind::RestoreProperty`].
    RemoveProperty { id: PropertyId },
    /// Re-add a removed property and re-promote its former references. Used
    /// only as the inverse of [`EditKind::RemoveProperty`].
    RestoreProperty {
        def: PropertyDef,
        repromote: Vec<NodeId>,
    },
    /// Rename a property by id.
    RenameProperty { id: PropertyId, new: String },
    /// Replace a property's default (initial) value.
    SetPropertyDefault { id: PropertyId, new: Value },
    /// Toggle whether a property is exposed as a runtime parameter (`true`) or
    /// inlined to literals at bake time (`false`).
    SetPropertyExposed { id: PropertyId, exposed: bool },
}

/// Emitted by [`apply_edits`] after a mutation.
///
/// Carries the inverse edit and the direction flag the history recorder uses.
#[derive(Message, Debug, Clone)]
pub struct EditApplied {
    pub doc: Entity,
    pub inverse: EditRequest,
    pub direction: EditDirection,
    /// True when the edit was applied as a live GPU value upload (a promoted
    /// literal tweak or an exposed property's default) and needs no proxy
    /// rebuild. False for everything else (proxy must be re-built from
    /// canonical to mirror the change).
    pub is_literal_edit: bool,
}

/// User-driven history navigation.
///
/// Consumed by `crate::history`.
#[derive(Message, Debug, Clone, Copy)]
pub enum HistoryRequest {
    Undo(Entity),
    Redo(Entity),
}

pub struct EditPlugin;

impl Plugin for EditPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<EditRequest>()
            .add_message::<EditApplied>()
            .add_message::<HistoryRequest>()
            .add_systems(
                Update,
                (
                    crate::history::history_dispatch,
                    apply_edits,
                    crate::history::record_history,
                )
                    .chain()
                    .in_set(EditSystems),
            );
    }
}

/// Systems needing freshly-applied edits should run `.after(EditSystems)`.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct EditSystems;

/// The single writer of `DocumentContent` and the preview `EffectAsset`.
///
/// Writes via `graph_mut`. Every edit mutates the canonical [`EffectGraph`],
/// re-bakes it into the document's preview asset, then forces a `bevy_hanabi`
/// recompile and a `Respawn` so the new particle layout binds cleanly (see the
/// `CachedPipelines` ordering note in `crate::plugins::reconcile`).
///
/// [`EffectGraph`]: crate::effect_graph::model::EffectGraph
pub fn apply_edits(
    mut requests: MessageReader<EditRequest>,
    mut applied: MessageWriter<EditApplied>,
    mut playback: MessageWriter<PlaybackCommand>,
    mut contents: Query<&mut DocumentContent>,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut children_q: Query<&Children>,
    scene_roots: Query<(), With<DocumentSceneRoot>>,
    mut particle_effects: Query<&mut ParticleEffect>,
    proxies: Query<&ProxyEffect>,
    mut effect_props: Query<&mut EffectProperties>,
    type_registry: Res<AppTypeRegistry>,
) {
    for req in requests.read() {
        let Ok(mut content) = contents.get_mut(req.doc) else {
            warn!("edit request for missing document: {:?}", req);
            continue;
        };

        // `RenameDocument` is the only edit that touches `DocumentContent`
        // metadata rather than the graph; it needs no re-bake or respawn.
        if let EditKind::RenameDocument { new } = &req.kind {
            let old = content.set_name(new.clone());
            applied.write(EditApplied {
                doc: req.doc,
                inverse: EditRequest {
                    doc: req.doc,
                    direction: req.direction,
                    kind: EditKind::RenameDocument { new: old },
                },
                direction: req.direction,
                is_literal_edit: false,
            });
            continue;
        }

        let registry = type_registry.read();

        // Mutate the canonical graph and capture the inverse edit.
        let inverse_kind = match apply_to_graph(content.graph_mut(), &registry, &req.kind) {
            Ok(inverse) => inverse,
            Err(err) => {
                warn!("edit refused ({err}): {:?}", req.kind);
                continue;
            }
        };
        content.mark_dirty(true);

        // Live value-upload fast path: an edit that only changes a value already
        // backed by a GPU property — a promoted literal tweak, or an exposed
        // user property's default — can be pushed straight to the GPU via
        // `EffectProperties`, skipping the re-bake / shader recompile / respawn.
        // Edits with no such binding (render-reachable or non-promotable
        // literals, unexposed properties) fall through to the full rebake path.
        if let Some(uploads) = fast_upload_target(&req.kind, &content, proxies.get(req.doc).ok())
            && let Some(pe) = proxy_props_entity(req.doc, &children_q, &scene_roots, &effect_props)
        {
            for (name, value) in &uploads {
                if let Ok(props) = effect_props.get_mut(pe) {
                    EffectProperties::set_if_changed(props, name, *value);
                }
            }
            drop(registry);
            applied.write(EditApplied {
                doc: req.doc,
                inverse: EditRequest {
                    doc: req.doc,
                    direction: req.direction,
                    kind: inverse_kind,
                },
                direction: req.direction,
                is_literal_edit: true,
            });
            continue;
        }

        // Re-bake the mutated graph into the live preview asset, refreshing the
        // literal provenance the fast path above depends on.
        let (new_asset, new_sites) =
            bake_preview_with_provenance(content.graph(), &registry, content.preview_tag());
        drop(registry);
        content.set_literal_sites(new_sites);
        if let Some(asset) = effects.get_mut(content.effect()) {
            *asset = new_asset;
        } else {
            warn!("apply_edits: missing preview asset for {:?}", req.doc);
        }

        // Force hanabi to re-process the effect, then despawn/respawn the
        // instance so the rebuilt particle layout cannot collide with cached
        // pipelines from the previous layout.
        touch_particle_effect(
            req.doc,
            children_q.reborrow(),
            &scene_roots,
            particle_effects.reborrow(),
        );
        playback.write(PlaybackCommand::Respawn(req.doc));

        applied.write(EditApplied {
            doc: req.doc,
            inverse: EditRequest {
                doc: req.doc,
                direction: req.direction, // unused on inverse, kept for symmetry
                kind: inverse_kind,
            },
            direction: req.direction,
            is_literal_edit: false,
        });
    }
}

/// The live GPU value uploads that realise `kind`, if it's fully GPU-bound.
///
/// If `kind` only changes values already backed by live GPU properties, returns
/// the `(property name, new value)` uploads that realise it — driving the
/// value-upload fast path in [`apply_edits`]. Returns `None` (forcing a rebake)
/// for edits that change shader structure or whose value isn't fully GPU-bound:
///
/// * `SetInputDefault` / `SetLiteralValue` — bound only if the literal was
///   promoted to a proxy tweak property (init/update-reachable, promotable
///   type).
/// * `SetPropertyDefault` for an **exposed** property — a runtime `Module`
///   property settable by its own name.
/// * `SetPropertyDefault` for an **unexposed** property — inlined to a literal
///   at each reference; bound only if *every* reference was promoted (else
///   rebake, so render-reachable references aren't left stale).
fn fast_upload_target(
    kind: &EditKind,
    content: &DocumentContent,
    proxy: Option<&ProxyEffect>,
) -> Option<Vec<(String, Value)>> {
    match kind {
        EditKind::SetInputDefault { node, port, new } => {
            let site = LiteralSite::Input {
                node: *node,
                port: port.clone(),
            };
            Some(vec![(proxy?.tweak_props.get(&site)?.clone(), *new)])
        }
        EditKind::SetLiteralValue { node, new } => {
            let site = LiteralSite::Node(*node);
            Some(vec![(proxy?.tweak_props.get(&site)?.clone(), *new)])
        }
        EditKind::SetPropertyDefault { id, new } => {
            let def = content.graph().properties.iter().find(|p| p.id == *id)?;
            if def.exposed {
                return Some(vec![(def.name.to_string(), *new)]);
            }
            let proxy = proxy?;
            let mut uploads = Vec::new();
            for n in &content.graph().nodes {
                if let NodePayload::Expr(ExprNode::Property(pid)) = &n.payload
                    && pid == id
                {
                    let name = proxy.tweak_props.get(&LiteralSite::Node(n.id))?;
                    uploads.push((name.clone(), *new));
                }
            }
            Some(uploads)
        }
        _ => None,
    }
}

/// Locate the proxy `ParticleEffect` entity for `doc`.
///
/// It carries [`EffectProperties`] and is a grandchild of the document via its
/// [`DocumentSceneRoot`]. Mirrors [`touch_particle_effect`]'s navigation.
fn proxy_props_entity(
    doc: Entity,
    children_q: &Query<&Children>,
    scene_roots: &Query<(), With<DocumentSceneRoot>>,
    effect_props: &Query<&mut EffectProperties>,
) -> Option<Entity> {
    let doc_children = children_q.get(doc).ok()?;
    for &child in doc_children {
        if scene_roots.get(child).is_err() {
            continue;
        }
        let Ok(scene_children) = children_q.get(child) else {
            continue;
        };
        for &grandchild in scene_children {
            if effect_props.get(grandchild).is_ok() {
                return Some(grandchild);
            }
        }
    }
    None
}

/// Apply one [`EditKind`] to the canonical graph and return the inverse edit.
///
/// Resilience principle: a refused edit (missing node/property, unregistered
/// modifier, out-of-range index) returns `Err` and is skipped by the caller —
/// never a panic. `RenameDocument` is handled by the caller and is unreachable
/// here.
fn apply_to_graph(
    graph: &mut crate::effect_graph::model::EffectGraph,
    registry: &bevy::reflect::TypeRegistry,
    kind: &EditKind,
) -> Result<EditKind, String> {
    Ok(match kind {
        EditKind::RenameDocument { .. } => {
            unreachable!("RenameDocument is handled before re-baking")
        }

        // --- Effect header ---
        EditKind::SetEffectName { new } => {
            let old = graph_edit::set_effect_name(graph, SharedStr::from(new.as_str()));
            EditKind::SetEffectName {
                new: old.to_string(),
            }
        }
        EditKind::SetSimulationSpace { new } => {
            let old = graph_edit::set_simulation_space(graph, *new);
            EditKind::SetSimulationSpace { new: old }
        }
        EditKind::SetSimulationCondition { new } => {
            let old = graph_edit::set_simulation_condition(graph, *new);
            EditKind::SetSimulationCondition { new: old }
        }
        EditKind::SetSpawnerSettings { new } => {
            let old = graph_edit::set_spawner(graph, *new);
            EditKind::SetSpawnerSettings { new: old }
        }
        EditKind::SetCapacity { new } => {
            let old = graph_edit::set_capacity(graph, *new);
            EditKind::SetCapacity { new: old }
        }
        EditKind::SetZLayer2d { new } => {
            let old = graph_edit::set_z_layer_2d(graph, *new);
            EditKind::SetZLayer2d { new: old }
        }

        // --- Modifier stacks ---
        EditKind::AddModifierFromTemplate { group, type_id, at } => {
            let id = graph_edit::add_modifier_from_template(graph, registry, *group, *type_id, *at)
                .ok_or("modifier type is not registered")?;
            let idx = graph
                .stack(*group)
                .and_then(|s| s.members.iter().position(|m| *m == id))
                .ok_or("added modifier not found in its stack")?;
            EditKind::RemoveModifier { group: *group, idx }
        }
        EditKind::InsertModifierNode { removed } => {
            let group = removed.group;
            let node_id = removed.node.id;
            if !graph_edit::insert_modifier(graph, removed.clone()) {
                return Err("target stack is missing".to_string());
            }
            let idx = graph
                .stack(group)
                .and_then(|s| s.members.iter().position(|m| *m == node_id))
                .ok_or("inserted modifier not found in its stack")?;
            EditKind::RemoveModifier { group, idx }
        }
        EditKind::RemoveModifier { group, idx } => {
            let removed = graph_edit::remove_modifier(graph, *group, *idx)
                .ok_or("no modifier at the given index")?;
            EditKind::InsertModifierNode { removed }
        }
        EditKind::MoveModifier { group, from, to } => {
            if !graph_edit::move_stack_member(graph, *group, *from, *to) {
                return Err("move index out of range".to_string());
            }
            EditKind::MoveModifier {
                group: *group,
                from: *to,
                to: *from,
            }
        }
        EditKind::SetModifierAttribute {
            group,
            idx,
            new,
            reset_value,
        } => {
            let (old_attr, rewrote_old) =
                graph_edit::set_modifier_attribute(graph, *group, *idx, *new, *reset_value)?;
            EditKind::SetModifierAttribute {
                group: *group,
                idx: *idx,
                new: old_attr,
                reset_value: rewrote_old,
            }
        }
        EditKind::SetModifierConfig { node, field, new } => {
            let old = graph_edit::set_modifier_config(graph, *node, field, new.clone())
                .ok_or("modifier node has no such config field")?;
            EditKind::SetModifierConfig {
                node: *node,
                field: field.clone(),
                new: old,
            }
        }

        // --- Expression input defaults ---
        EditKind::SetInputDefault { node, port, new } => {
            let old = graph_edit::set_input_default(graph, *node, port, *new);
            EditKind::SetInputDefault {
                node: *node,
                port: port.clone(),
                new: old.unwrap_or(*new),
            }
        }
        EditKind::SetInputImageBinding {
            node,
            port,
            binding,
        } => {
            let old = graph_edit::set_input_image_binding(graph, *node, port, binding.clone())
                .ok_or("node not found")?;
            EditKind::SetInputImageBinding {
                node: *node,
                port: port.clone(),
                binding: old,
            }
        }
        EditKind::SetLiteralValue { node, new } => {
            let old = graph_edit::set_literal_node(graph, *node, *new)
                .ok_or("node is not a literal expression")?;
            EditKind::SetLiteralValue {
                node: *node,
                new: old,
            }
        }

        // --- Standalone expression nodes ---
        EditKind::AddExprNode { expr, inputs } => {
            let id = graph_edit::add_expr_node(graph, expr.clone(), inputs.clone());
            EditKind::RemoveNode { id }
        }
        EditKind::RemoveNode { id } => {
            let removed = graph_edit::remove_node(graph, *id).ok_or("node not found")?;
            EditKind::InsertNode { removed }
        }
        EditKind::InsertNode { removed } => {
            let id = removed.node.id;
            graph_edit::insert_node(graph, removed.clone());
            EditKind::RemoveNode { id }
        }

        // --- Image source nodes and texture slots ---
        EditKind::AddImageNode => {
            let id = graph_edit::add_image_node(graph);
            EditKind::RemoveNode { id }
        }
        EditKind::SetImageNodeBinding { node, binding } => {
            let old = graph_edit::set_image_node_binding(graph, *node, binding.clone())
                .ok_or("not an image node")?;
            EditKind::SetImageNodeBinding {
                node: *node,
                binding: old,
            }
        }
        EditKind::AddTextureSlot => {
            let id = graph_edit::add_texture_slot(graph);
            EditKind::RemoveTextureSlot { id }
        }
        EditKind::RemoveTextureSlot { id } => {
            let removed =
                graph_edit::remove_texture_slot(graph, *id).ok_or("texture slot not found")?;
            EditKind::InsertTextureSlot { removed }
        }
        EditKind::InsertTextureSlot { removed } => {
            let id = removed.slot.id;
            graph_edit::insert_texture_slot(graph, removed.clone());
            EditKind::RemoveTextureSlot { id }
        }
        EditKind::RenameTextureSlot { id, new } => {
            let old = graph_edit::rename_texture_slot(graph, *id, new.clone())
                .ok_or("texture slot not found")?;
            EditKind::RenameTextureSlot { id: *id, new: old }
        }
        EditKind::ReorderTextureSlot { id, to } => {
            let from = graph_edit::reorder_texture_slot(graph, *id, *to)
                .ok_or("texture slot not found")?;
            EditKind::ReorderTextureSlot { id: *id, to: from }
        }

        // --- Links ---
        EditKind::AddLink { link } => {
            let to_node = link.to.node;
            let inverse = match graph_edit::add_link(graph, link.clone()) {
                Some(displaced) => EditKind::AddLink { link: displaced },
                None => EditKind::RemoveLink { link: link.clone() },
            };
            graph_edit::normalize_select_image(graph, to_node);
            inverse
        }
        EditKind::RemoveLink { link } => {
            let removed = graph_edit::remove_link_to(graph, &link.to)
                .ok_or("no link targets that input port")?;
            graph_edit::normalize_select_image(graph, link.to.node);
            EditKind::AddLink { link: removed }
        }

        // --- User properties ---
        EditKind::AddProperty {
            name,
            value,
            exposed,
        } => {
            if crate::proxy::is_tweak_prop_name(name) {
                return Err(format!("property name {name:?} uses the reserved prefix"));
            }
            let id =
                graph_edit::add_property(graph, SharedStr::from(name.as_str()), *value, *exposed);
            EditKind::RemoveProperty { id }
        }
        EditKind::RemoveProperty { id } => {
            let (def, repromote) =
                graph_edit::remove_property(graph, *id).ok_or("property not found")?;
            EditKind::RestoreProperty { def, repromote }
        }
        EditKind::RestoreProperty { def, repromote } => {
            let id = def.id;
            graph_edit::restore_property(graph, def.clone(), repromote);
            EditKind::RemoveProperty { id }
        }
        EditKind::RenameProperty { id, new } => {
            if crate::proxy::is_tweak_prop_name(new) {
                return Err(format!("property name {new:?} uses the reserved prefix"));
            }
            let old = graph_edit::rename_property(graph, *id, SharedStr::from(new.as_str()))
                .ok_or("property not found")?;
            EditKind::RenameProperty {
                id: *id,
                new: old.to_string(),
            }
        }
        EditKind::SetPropertyDefault { id, new } => {
            let old =
                graph_edit::set_property_default(graph, *id, *new).ok_or("property not found")?;
            EditKind::SetPropertyDefault { id: *id, new: old }
        }
        EditKind::SetPropertyExposed { id, exposed } => {
            let old = graph_edit::set_property_exposed(graph, *id, *exposed)
                .ok_or("property not found")?;
            EditKind::SetPropertyExposed {
                id: *id,
                exposed: old,
            }
        }
    })
}

/// Force `bevy_hanabi`'s `compile_effects` to re-process the doc's
/// `ParticleEffect`.
///
/// We do this after every `EffectAsset` mutation because hanabi reacts to
/// `Ref<ParticleEffect>::is_changed()`, not to
/// `AssetEvent<EffectAsset>::Modified`. The cost is one shader rebuild per
/// commit, which is acceptable at our edit-once-per-drag cadence.
fn touch_particle_effect(
    doc: Entity,
    children_q: Query<&Children>,
    scene_roots: &Query<(), With<DocumentSceneRoot>>,
    mut particle_effects: Query<&mut ParticleEffect>,
) {
    let Ok(doc_children) = children_q.get(doc) else {
        return;
    };
    for &child in doc_children {
        if scene_roots.get(child).is_err() {
            continue;
        }
        let Ok(scene_children) = children_q.get(child) else {
            continue;
        };
        for &grandchild in scene_children {
            if let Ok(mut effect) = particle_effects.get_mut(grandchild) {
                effect.set_changed();
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use bevy::reflect::TypeRegistry;

    use super::*;
    use crate::{
        effect_graph::{
            demo::demo_graph,
            model::{EffectGraph, ModifierNodeData, NodePayload, PortRef},
            schema::OUTPUT_PORT,
        },
        modifier_registry::ModifierRegistryPlugin,
    };

    /// Build an `App` carrying a populated modifier registry.
    ///
    /// Mirrors the setup `add_modifier_from_template_bakes` uses.
    fn registry_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()));
        app.add_plugins(ModifierRegistryPlugin);
        app
    }

    /// A canonical copy of `g` for structural comparison.
    ///
    /// The `nodes`, `links` and `properties` collections are sorted (their Vec
    /// order carries no semantics — references are by id, and layout lives in
    /// `GraphLayout`), and the monotonic id allocator is zeroed (undo never
    /// rewinds `next_id`, since ids are never recycled). Stack member order is
    /// left untouched — it *is* semantic (the pipeline executes modifiers in
    /// that order).
    fn canonical(g: &EffectGraph) -> EffectGraph {
        let mut g = g.clone();
        g.next_id = 0;
        g.nodes.sort_by_key(|n| n.id);
        g.properties.sort_by_key(|p| p.id);
        g.links.sort_by(|a, b| {
            (a.from.node, &a.from.port, a.to.node, &a.to.port).cmp(&(
                b.from.node,
                &b.from.port,
                b.to.node,
                &b.to.port,
            ))
        });
        g.stacks.sort_by_key(|s| match s.group {
            ModifierGroup::Init => 0u8,
            ModifierGroup::Update => 1,
            ModifierGroup::Render => 2,
        });
        g
    }

    /// Drive one `EditKind` through the full undo/redo cycle.
    ///
    /// Asserts the inverse is correct:
    ///
    /// 1. Apply `edit` (it must change the graph).
    /// 2. Apply the returned inverse (undo): the graph returns to its original
    ///    structure (modulo the monotonic id allocator).
    /// 3. Apply that inverse's own inverse (redo): the graph returns *exactly*
    ///    to the post-edit state, matching how `history` replays a redo by
    ///    re-applying the captured inverse rather than the original edit.
    fn assert_round_trip(registry: &TypeRegistry, edit: EditKind) {
        assert_round_trip_on(registry, demo_graph(), edit);
    }

    /// Like [`assert_round_trip`], but from an explicit base graph.
    ///
    /// E.g. the demo plus a synthetic standalone literal node.
    fn assert_round_trip_on(registry: &TypeRegistry, original: EffectGraph, edit: EditKind) {
        let mut g = original.clone();
        let inverse = apply_to_graph(&mut g, registry, &edit)
            .unwrap_or_else(|e| panic!("forward edit refused ({e}): {edit:?}"));
        let post_edit = g.clone();
        assert_ne!(
            canonical(&g),
            canonical(&original),
            "edit must change the graph: {edit:?}"
        );

        let redo = apply_to_graph(&mut g, registry, &inverse)
            .unwrap_or_else(|e| panic!("inverse refused ({e}): {inverse:?}"));
        assert_eq!(
            canonical(&g),
            canonical(&original),
            "undo must restore the original graph: {edit:?}"
        );

        apply_to_graph(&mut g, registry, &redo)
            .unwrap_or_else(|e| panic!("redo refused ({e}): {redo:?}"));
        assert_eq!(
            g, post_edit,
            "redo must restore the post-edit state: {edit:?}"
        );
    }

    fn property_id(g: &EffectGraph, name: &str) -> PropertyId {
        g.properties
            .iter()
            .find(|p| &*p.name == name)
            .unwrap_or_else(|| panic!("demo property {name:?} missing"))
            .id
    }

    /// A demo graph with one synthetic standalone literal node appended.
    ///
    /// The demo itself carries none. Returns the graph and the new node's id.
    fn demo_with_standalone_literal(value: Value) -> (EffectGraph, NodeId) {
        let mut g = demo_graph();
        let id = graph_edit::add_expr_node(&mut g, ExprNode::Literal(value), Vec::new());
        (g, id)
    }

    #[test]
    fn round_trip_header_edits() {
        let app = registry_app();
        let registry = app.world().resource::<AppTypeRegistry>().read();

        assert_round_trip(
            &registry,
            EditKind::SetEffectName {
                new: "renamed".to_string(),
            },
        );
        assert_round_trip(
            &registry,
            EditKind::SetSimulationSpace {
                new: SimulationSpace::Local,
            },
        );
        assert_round_trip(
            &registry,
            EditKind::SetSimulationCondition {
                new: SimulationCondition::Always,
            },
        );
        assert_round_trip(
            &registry,
            EditKind::SetSpawnerSettings {
                new: SpawnerSettings::rate(50.0.into()),
            },
        );
        assert_round_trip(&registry, EditKind::SetCapacity { new: 16384 });
        assert_round_trip(&registry, EditKind::SetZLayer2d { new: 3.0 });
    }

    #[test]
    fn round_trip_modifier_stack_edits() {
        let app = registry_app();
        let registry = app.world().resource::<AppTypeRegistry>().read();
        let g = demo_graph();

        let update_len = g.stack(ModifierGroup::Update).unwrap().members.len();
        assert_round_trip(
            &registry,
            EditKind::AddModifierFromTemplate {
                group: ModifierGroup::Update,
                type_id: TypeId::of::<bevy_hanabi::AccelModifier>(),
                at: update_len,
            },
        );

        assert_round_trip(
            &registry,
            EditKind::RemoveModifier {
                group: ModifierGroup::Init,
                idx: 0,
            },
        );

        assert_round_trip(
            &registry,
            EditKind::MoveModifier {
                group: ModifierGroup::Render,
                from: 0,
                to: 1,
            },
        );
    }

    #[test]
    fn round_trip_modifier_attribute_and_config() {
        let app = registry_app();
        let registry = app.world().resource::<AppTypeRegistry>().read();
        let g = demo_graph();

        // The demo's lifetime node (Init #2) is a `SetAttributeModifier`.
        // Same-typed retarget (LIFETIME -> AGE, both f32): no value reset.
        assert_round_trip(
            &registry,
            EditKind::SetModifierAttribute {
                group: ModifierGroup::Init,
                idx: 2,
                new: Attribute::AGE,
                reset_value: None,
            },
        );
        // Differently-typed retarget (LIFETIME -> POSITION, f32 -> vec3): the
        // inline `value` literal is reset, and the inverse must restore it.
        assert_round_trip(
            &registry,
            EditKind::SetModifierAttribute {
                group: ModifierGroup::Init,
                idx: 2,
                new: Attribute::POSITION,
                reset_value: None,
            },
        );

        // Flip the position-sphere node's `dimension` enum config.
        let pos = g.stack(ModifierGroup::Init).unwrap().members[0];
        let dimension = match &g.node(pos).unwrap().payload {
            NodePayload::Modifier(ModifierNodeData::Known { config, .. }) => {
                config.get("dimension").cloned().expect("dimension config")
            }
            _ => unreachable!("position-sphere node is a known modifier"),
        };
        let flipped = match dimension {
            EditValue::Enum { type_path, .. } => EditValue::Enum {
                type_path,
                variant: "Volume".into(),
            },
            other => panic!("expected an enum dimension, got {other:?}"),
        };
        assert_round_trip(
            &registry,
            EditKind::SetModifierConfig {
                node: pos,
                field: "dimension".into(),
                new: flipped,
            },
        );
    }

    #[test]
    fn round_trip_input_and_literal() {
        let app = registry_app();
        let registry = app.world().resource::<AppTypeRegistry>().read();
        let g = demo_graph();

        // The position-sphere node has an unlinked `center` input slot.
        let pos = g.stack(ModifierGroup::Init).unwrap().members[0];
        assert_round_trip(
            &registry,
            EditKind::SetInputDefault {
                node: pos,
                port: "center".into(),
                new: Vec3::new(9.0, 9.0, 9.0).into(),
            },
        );

        let (g, literal) = demo_with_standalone_literal(Value::from(0.0f32));
        assert_round_trip_on(
            &registry,
            g,
            EditKind::SetLiteralValue {
                node: literal,
                new: Value::from(999.0f32),
            },
        );
    }

    #[test]
    fn round_trip_expr_nodes() {
        let app = registry_app();
        let registry = app.world().resource::<AppTypeRegistry>().read();

        assert_round_trip(
            &registry,
            EditKind::AddExprNode {
                expr: ExprNode::Literal(Value::from(3.0f32)),
                inputs: Vec::new(),
            },
        );

        // Removing a standalone literal (no links, no stack membership) and
        // re-inserting it must restore the node exactly.
        let (g, literal) = demo_with_standalone_literal(Value::from(7.0f32));
        assert_round_trip_on(&registry, g, EditKind::RemoveNode { id: literal });
    }

    #[test]
    fn round_trip_links() {
        let app = registry_app();
        let registry = app.world().resource::<AppTypeRegistry>().read();
        let g = demo_graph();

        // Removing an existing demo link round-trips through AddLink.
        let existing = g.links.first().cloned().expect("demo has links");
        assert_round_trip(
            &registry,
            EditKind::RemoveLink {
                link: existing.clone(),
            },
        );

        // Adding a link onto an empty input port displaces nothing; the inverse
        // is a RemoveLink.
        let (node, port) = g
            .nodes
            .iter()
            .filter(|n| matches!(n.payload, NodePayload::Modifier(_)))
            .flat_map(|n| n.inputs.iter().map(move |s| (n.id, s.name.clone())))
            .find(|(node, port)| {
                !g.links
                    .iter()
                    .any(|l| l.to.node == *node && l.to.port == *port)
            })
            .expect("an unlinked modifier input");
        let source = g
            .nodes
            .iter()
            .map(|n| n.id)
            .find(|&id| id != node)
            .expect("another node");
        assert_round_trip(
            &registry,
            EditKind::AddLink {
                link: GraphLink {
                    from: PortRef {
                        node: source,
                        port: OUTPUT_PORT.into(),
                    },
                    to: PortRef { node, port },
                },
            },
        );

        // Adding a link onto an already-linked input displaces the old link;
        // the inverse is an AddLink restoring it.
        let other_source = g
            .nodes
            .iter()
            .map(|n| n.id)
            .find(|&id| id != existing.from.node && id != existing.to.node)
            .expect("a third node");
        assert_round_trip(
            &registry,
            EditKind::AddLink {
                link: GraphLink {
                    from: PortRef {
                        node: other_source,
                        port: OUTPUT_PORT.into(),
                    },
                    to: existing.to.clone(),
                },
            },
        );
    }

    #[test]
    fn round_trip_properties() {
        let app = registry_app();
        let registry = app.world().resource::<AppTypeRegistry>().read();
        let g = demo_graph();

        assert_round_trip(
            &registry,
            EditKind::AddProperty {
                name: "extra".to_string(),
                value: Value::from(1.0f32),
                exposed: false,
            },
        );

        // `gravity` is referenced by a Property node; removal demotes the
        // reference and the inverse must re-promote it.
        let gravity = property_id(&g, "gravity");
        assert_round_trip(&registry, EditKind::RemoveProperty { id: gravity });
        assert_round_trip(
            &registry,
            EditKind::RenameProperty {
                id: gravity,
                new: "g".to_string(),
            },
        );
        assert_round_trip(
            &registry,
            EditKind::SetPropertyDefault {
                id: gravity,
                new: Vec3::new(1.0, 2.0, 3.0).into(),
            },
        );
        assert_round_trip(
            &registry,
            EditKind::SetPropertyExposed {
                id: gravity,
                exposed: false,
            },
        );
    }
}
