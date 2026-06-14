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
//!   [`EffectAsset`](bevy_hanabi::EffectAsset).
//! * [`crate::history::record_history`] maintains the per-document undo stack
//!   from [`EditApplied`] events.

use std::any::TypeId;

use bevy::prelude::*;
use bevy_hanabi::{
    Attribute, EffectAsset, ParticleEffect, SimulationCondition, SimulationSpace, SpawnerSettings,
    Value,
};

use crate::document::{DocumentContent, DocumentSceneRoot, ModifierGroup};
use crate::effect_graph::bake::bake_preview;
use crate::effect_graph::edit::{self as graph_edit, RemovedModifier, RemovedNode};
use crate::effect_graph::model::{
    EditValue, ExprNode, GraphLink, InputSlot, NodeId, PropertyDef, PropertyId, SharedStr,
};
use crate::history::EditDirection;
use crate::playback::PlaybackCommand;

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

/// The actual edit payload. Each variant carries the *new* value and is applied
/// to the document's canonical [`EffectGraph`]; `apply_edits` reads the current
/// value to build the inverse, then re-bakes the graph into the preview asset.
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
    /// (e.g. a data-less enum like `ShapeDimension`, or a flags field). Inverse:
    /// the same edit carrying the field's previous [`EditValue`].
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
    /// Set the value of a standalone `ExprNode::Literal` node (one whose value
    /// is the node itself, not an input-port default).
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
    /// Re-add a removed property and re-promote its former references. Used only
    /// as the inverse of [`EditKind::RemoveProperty`].
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

/// Emitted by [`apply_edits`] after a mutation. Carries the inverse edit
/// and the direction flag the history recorder uses.
#[derive(Message, Debug, Clone)]
pub struct EditApplied {
    pub doc: Entity,
    pub inverse: EditRequest,
    pub direction: EditDirection,
    /// True for `SetLiteralValue` (no proxy rebuild needed; value
    /// already uploaded as a property). False for everything else
    /// (proxy must be re-built from canonical to mirror the change).
    pub is_literal_edit: bool,
}

/// User-driven history navigation. Consumed by `crate::history`.
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

/// Systems that depend on freshly-applied edits should be ordered
/// `.after(EditSystems)`.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct EditSystems;

/// The single writer of `DocumentContent` (via `graph_mut`) and of the preview
/// `EffectAsset`. Every edit mutates the canonical [`EffectGraph`], re-bakes it
/// into the document's preview asset, then forces a `bevy_hanabi` recompile and
/// a `Respawn` so the new particle layout binds cleanly (see the
/// `CachedPipelines` ordering note in `crate::plugins::reconcile`).
pub fn apply_edits(
    mut requests: MessageReader<EditRequest>,
    mut applied: MessageWriter<EditApplied>,
    mut playback: MessageWriter<PlaybackCommand>,
    mut contents: Query<&mut DocumentContent>,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut children_q: Query<&Children>,
    scene_roots: Query<(), With<DocumentSceneRoot>>,
    mut particle_effects: Query<&mut ParticleEffect>,
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

        // Re-bake the mutated graph into the live preview asset.
        let new_asset = bake_preview(content.graph(), &registry, content.preview_tag());
        drop(registry);
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
        EditKind::SetZLayer2d { new } => {
            let old = graph_edit::set_z_layer_2d(graph, *new);
            EditKind::SetZLayer2d { new: old }
        }

        // --- Modifier stacks ---
        EditKind::AddModifierFromTemplate {
            group,
            type_id,
            at,
        } => {
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

        // --- Links ---
        EditKind::AddLink { link } => match graph_edit::add_link(graph, link.clone()) {
            Some(displaced) => EditKind::AddLink { link: displaced },
            None => EditKind::RemoveLink { link: link.clone() },
        },
        EditKind::RemoveLink { link } => {
            let removed = graph_edit::remove_link_to(graph, &link.to)
                .ok_or("no link targets that input port")?;
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
            let id = graph_edit::add_property(graph, SharedStr::from(name.as_str()), *value, *exposed);
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
/// `ParticleEffect`. We do this after every `EffectAsset` mutation
/// because hanabi reacts to `Ref<ParticleEffect>::is_changed()`, not to
/// `AssetEvent<EffectAsset>::Modified`. The cost is one shader rebuild
/// per commit, which is acceptable at our edit-once-per-drag cadence.
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
