//! Edit-message scaffolding.
//!
//! See `crate::document` for the architectural commitment. The rule:
//!
//! * UI code emits [`EditRequest`] messages; it never mutates the document
//!   directly.
//! * [`apply_edits`] is the **only** caller of `DocumentContent::graph_mut` and
//!   the only system holding `Query<&mut DocumentContent>` and
//!   `ResMut<Assets<EffectAsset>>` for write access. Every edit mutates the
//!   canonical [`EffectGraph`] and re-bakes the affected emitter into its
//!   preview [`EffectAsset`].
//! * [`crate::history::record_history`] maintains the per-document undo stack
//!   from [`EditApplied`] events.
//!
//! Every emitter-scoped [`EditKind`] variant carries an explicit
//! [`EmitterId`] naming which pipeline it mutates — value/expression links stay
//! emitter-local (see [`crate::effect_graph::edit`]), but the containing
//! document is an [`EffectGraph`] forest of emitters plus the spawn source
//! contexts and topology links that drive them, so every request must say
//! which emitter it targets. A handful of variants are effect-level rather
//! than emitter-scoped (creating/deleting emitter pipelines and spawn sources,
//! source links, event links) and carry the ids they need directly instead.
//!
//! [`EffectAsset`]: bevy_hanabi::EffectAsset
//! [`EffectGraph`]: crate::effect_graph::model::EffectGraph

use std::{any::TypeId, collections::HashSet};

use bevy::prelude::*;
use bevy_hanabi::{
    Attribute, EffectAsset, EffectProperties, SimulationCondition, SimulationSpace,
    SpawnerSettings, Value,
};
use hanabi_node_graph::{NodeId as WidgetNodeId, StackId as WidgetStackId, WorldPos};

use crate::{
    document::{
        DocumentContent, DocumentSceneRoot, DocumentUi, EmitterSceneEntities, ModifierGroup,
        bake_effect_records,
    },
    effect_graph::{
        bake::LiteralSite,
        edit::{
            self as graph_edit, RemovedEmitter, RemovedModifier, RemovedNode, RemovedProperty,
            RemovedSource, RemovedTextureSlot,
        },
        model::{
            EditValue, EffectGraph, EmitterId, ExprNode, GraphLink, ImageBinding, InputSlot,
            NodeId, NodePayload, PropertyId, SharedStr, SlotId, SourceId, SourceKind,
        },
        schema::value_type_zero,
        view::GraphReader,
    },
    history::EditDirection,
    playback::PlaybackCommand,
    proxy::{ProxyEmitters, ProxyInstance, proxy_props_entity},
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

/// One item's canvas position change, from [`EditKind::MoveLayout`].
///
/// `Id` is a node or stack identifier. The inverse of a move swaps `from` and
/// `to`.
#[derive(Debug, Clone, Copy)]
pub struct PositionChange<Id> {
    pub id: Id,
    pub from: WorldPos,
    pub to: WorldPos,
}

impl<Id: Copy> PositionChange<Id> {
    /// The reverse move (swap `from`/`to`), used to build a `MoveLayout`'s
    /// inverse.
    fn inverted(&self) -> Self {
        Self {
            id: self.id,
            from: self.to,
            to: self.from,
        }
    }
}

/// The actual edit payload.
///
/// Each emitter-scoped variant carries the *new* value plus the [`EmitterId`]
/// it targets, and is applied to that emitter within the document's canonical
/// [`EffectGraph`]; `apply_edits` reads the current value to build the inverse,
/// then re-bakes the affected emitter into its preview asset. A handful of
/// variants are inter-emitter topology (creating/deleting emitter pipelines
/// and spawn sources, source links, event links) rather than emitter-scoped,
/// and carry whichever ids they need directly.
///
/// [`EffectGraph`]: crate::effect_graph::model::EffectGraph
#[derive(Debug, Clone)]
pub enum EditKind {
    /// Rename the document (shown in the tab title). Mutates
    /// `DocumentContent.name`, not the graph. Not yet bound in the UI.
    #[allow(dead_code)]
    RenameDocument { new: String },

    /// Move dragged nodes and/or stacks to new canvas positions.
    ///
    /// A view-only edit: it mutates the per-document `GraphView` layout (saved
    /// with the file) rather than the graph, so no re-bake or respawn. One
    /// drag — including a multi-selection — produces a single `MoveLayout`, so
    /// it undoes as a unit. Inverse: the same edit with every `from`/`to`
    /// swapped. Node and stack ids are unique across the whole document, so no
    /// `EmitterId` is needed to route this.
    MoveLayout {
        nodes: Vec<PositionChange<WidgetNodeId>>,
        stacks: Vec<PositionChange<WidgetStackId>>,
    },

    // --- Document topology: emitter pipelines and spawn sources ---
    /// Create a new, empty emitter pipeline (a fresh [`EmitterId`] plus its
    /// fixed Init/Update/Render stacks, no spawn source). Inverse:
    /// [`EditKind::DeleteEmitter`] with the freshly-allocated id.
    CreateEmitter { name: String },
    /// Delete an emitter pipeline, its source link, and every event link its
    /// emitters owned. Inverse: [`EditKind::InsertEmitter`].
    DeleteEmitter { emitter: EmitterId },
    /// Re-insert a previously-deleted emitter pipeline with its source link and
    /// event links. Used only as the inverse of
    /// [`EditKind::DeleteEmitter`]; not emitted by the UI.
    InsertEmitter { removed: RemovedEmitter },
    /// Create a new CPU spawner, unconnected to any emitter.
    ///
    /// Inverse: [`EditKind::DeleteSource`] with the freshly-allocated id.
    CreateCpuSource { settings: SpawnerSettings },
    /// Create a GPU event source and its driven emitter pipeline atomically.
    ///
    /// The new source is linked to a fresh Init/Update/Render pipeline. When
    /// `event_node` is present, its event output is linked to the source too.
    /// Inverse: a batch deleting the source and emitter.
    CreateGpuEmitter { event_node: Option<NodeId> },
    /// Delete a spawn source context, its source link, and every event link
    /// that targeted it. Inverse: [`EditKind::InsertSource`].
    DeleteSource { source: SourceId },
    /// Re-insert a previously-deleted spawn source context with its source
    /// link and event links. Used only as the inverse of
    /// [`EditKind::DeleteSource`]; not emitted by the UI.
    InsertSource { removed: RemovedSource },
    /// Connect a spawn source to drive an emitter, displacing whichever links
    /// already used either endpoint (a source and an emitter each accept at
    /// most one link). Inverse: a [`EditKind::Batch`] of
    /// [`EditKind::SetSourceLink`] restoring every displaced link, or
    /// [`EditKind::RemoveSourceLink`] if none were displaced.
    SetSourceLink {
        source: SourceId,
        emitter: EmitterId,
    },
    /// Disconnect a source link. Inverse: [`EditKind::SetSourceLink`].
    RemoveSourceLink {
        source: SourceId,
        emitter: EmitterId,
    },
    /// Connect an `EmitSpawnEventModifier` node's event output to a GPU
    /// source's multiple-link input. Inverse: [`EditKind::RemoveEventLink`].
    AddEventLink { node: NodeId, target: SourceId },
    /// Disconnect an event link. Inverse: [`EditKind::AddEventLink`].
    RemoveEventLink { node: NodeId, target: SourceId },
    /// Replace a `SourceKind::CpuSpawner` context's `SpawnerSettings`.
    /// Inverse: the same edit carrying the previous settings.
    SetCpuSpawnerSettings {
        source: SourceId,
        new: SpawnerSettings,
    },

    // --- Emitter settings ---
    /// Set the emitter's name (`EmitterGraph.name`).
    SetEmitterName { emitter: EmitterId, new: String },
    /// Set `EmitterGraph.simulation_space`.
    SetSimulationSpace {
        emitter: EmitterId,
        new: SimulationSpace,
    },
    /// Set `EmitterGraph.simulation_condition`.
    SetSimulationCondition {
        emitter: EmitterId,
        new: SimulationCondition,
    },
    /// Set `EmitterGraph.capacity` (max live particle count).
    SetCapacity { emitter: EmitterId, new: u32 },
    /// Set `EmitterGraph.z_layer_2d`.
    SetZLayer2d { emitter: EmitterId, new: f32 },

    // --- Modifier stacks ---
    /// Add a fresh modifier of `type_id` (a registered Hanabi modifier struct)
    /// into `group` at position `at`. The node's config and required input
    /// defaults are read from the registry factory's instance.
    AddModifierFromTemplate {
        emitter: EmitterId,
        group: ModifierGroup,
        /// `TypeId` of the Hanabi modifier struct. In-process only — never
        /// serialized.
        type_id: TypeId,
        at: usize,
    },
    /// Re-insert a previously-removed modifier node with its links. The inverse
    /// of [`EditKind::RemoveModifier`]; not emitted by the UI.
    InsertModifierNode {
        emitter: EmitterId,
        removed: RemovedModifier,
    },
    /// Remove the modifier at `idx` in `group` (node + incident links).
    RemoveModifier {
        emitter: EmitterId,
        group: ModifierGroup,
        idx: usize,
    },
    /// Move the modifier from `from` to `to` within `group`. `to` is the target
    /// index *after* removal of the source slot.
    MoveModifier {
        emitter: EmitterId,
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
        emitter: EmitterId,
        group: ModifierGroup,
        idx: usize,
        new: Attribute,
        reset_value: Option<Value>,
    },
    /// Set a non-expression configuration field of a modifier node to `new`
    /// (e.g. a data-less enum like `ShapeDimension`, or a flags field).
    /// Inverse: the same edit carrying the field's previous [`EditValue`].
    SetModifierConfig {
        emitter: EmitterId,
        node: NodeId,
        field: SharedStr,
        new: EditValue,
    },

    // --- Expression input defaults ---
    /// Set the inline default literal of an expression input port (an unlinked
    /// modifier or operator port). The "live tweak" path for slider drags.
    SetInputDefault {
        emitter: EmitterId,
        node: NodeId,
        port: SharedStr,
        new: Value,
    },
    /// Set the inline image binding of an image input port (an unlinked sampler
    /// `image` or modifier `texture_slot`). Structural: re-bakes. Inverse: the
    /// same edit carrying the previous binding.
    SetInputImageBinding {
        emitter: EmitterId,
        node: NodeId,
        port: SharedStr,
        binding: ImageBinding,
    },
    /// Set the value of a standalone `ExprNode::Literal` node (one whose value
    /// is the node itself, not an input-port default). Applied via
    /// `graph_edit::set_literal_value`; not yet emitted by any UI affordance.
    #[allow(dead_code)]
    SetLiteralValue {
        emitter: EmitterId,
        node: NodeId,
        new: Value,
    },

    // --- Standalone expression nodes ---
    /// Add a standalone expression node (literal / operator / attribute /
    /// property / built-in) with its operand input defaults. Inverse:
    /// [`EditKind::RemoveNode`] with the freshly-allocated id.
    AddExprNode {
        emitter: EmitterId,
        expr: ExprNode,
        inputs: Vec<InputSlot>,
    },
    /// Remove a node with its incident links and any stack membership. Inverse:
    /// [`EditKind::InsertNode`].
    RemoveNode { emitter: EmitterId, id: NodeId },
    /// Re-insert a removed node with its links and membership. Used only as the
    /// inverse of [`EditKind::RemoveNode`]; not emitted by the UI.
    InsertNode {
        emitter: EmitterId,
        removed: RemovedNode,
    },

    // --- Image source nodes and texture slots ---
    /// Add an image source node with its initial binding. Inverse:
    /// [`EditKind::RemoveNode`].
    AddImageNode {
        emitter: EmitterId,
        binding: ImageBinding,
    },
    /// Set the binding of an image node (asset, texture slot, or unbound).
    /// Inverse: the same edit carrying the previous binding.
    SetImageNodeBinding {
        emitter: EmitterId,
        node: NodeId,
        binding: ImageBinding,
    },
    /// Add a texture slot. Inverse: [`EditKind::RemoveTextureSlot`].
    AddTextureSlot { emitter: EmitterId },
    /// Remove a texture slot. Inverse: [`EditKind::InsertTextureSlot`].
    RemoveTextureSlot { emitter: EmitterId, id: SlotId },
    /// Re-insert a removed texture slot at its original index. Used only as the
    /// inverse of [`EditKind::RemoveTextureSlot`]; not emitted by the UI.
    InsertTextureSlot {
        emitter: EmitterId,
        removed: RemovedTextureSlot,
    },
    /// Rename a texture slot. Inverse: the same edit carrying the old name.
    RenameTextureSlot {
        emitter: EmitterId,
        id: SlotId,
        new: SharedStr,
    },
    /// Move a texture slot to a new index (reassigning sampling indices).
    /// Inverse: the same edit carrying the old index.
    ReorderTextureSlot {
        emitter: EmitterId,
        id: SlotId,
        to: usize,
    },

    // --- Links ---
    /// Connect an output port to an input port. The graph view validates the
    /// connection (type, cycles, stage order) before emitting this. Inverse:
    /// [`EditKind::AddLink`] restoring any displaced link, else
    /// [`EditKind::RemoveLink`].
    AddLink { emitter: EmitterId, link: GraphLink },
    /// Disconnect the link targeting an input port. Inverse:
    /// [`EditKind::AddLink`].
    RemoveLink { emitter: EmitterId, link: GraphLink },

    // --- User properties (addressed by stable id) ---
    /// Add a brand-new property. Inverse: [`EditKind::RemoveProperty`] with the
    /// freshly-allocated id.
    AddProperty {
        emitter: EmitterId,
        name: String,
        value: Value,
        exposed: bool,
    },
    /// Remove a property by id. Each `Property` reference node is deleted and
    /// the property's default is inlined into the ports it fed. Inverse:
    /// [`EditKind::RestoreProperty`].
    RemoveProperty { emitter: EmitterId, id: PropertyId },
    /// Re-add a removed property, restore the consumer ports it fed, and
    /// re-insert its reference nodes. Used only as the inverse of
    /// [`EditKind::RemoveProperty`].
    RestoreProperty {
        emitter: EmitterId,
        removed: RemovedProperty,
    },
    /// Rename a property by id.
    RenameProperty {
        emitter: EmitterId,
        id: PropertyId,
        new: String,
    },
    /// Replace a property's default (initial) value.
    SetPropertyDefault {
        emitter: EmitterId,
        id: PropertyId,
        new: Value,
    },
    /// Toggle whether a property is exposed as a runtime parameter (`true`) or
    /// inlined to literals at bake time (`false`).
    SetPropertyExposed {
        emitter: EmitterId,
        id: PropertyId,
        exposed: bool,
    },

    /// Apply several edits as one undoable unit, in order.
    ///
    /// Produced when a single user action must make more than one graph change
    /// atomically — connecting a link that also retypes an operator's sibling
    /// operand defaults, or displacing more than one existing source link.
    /// Inverse: a `Batch` of the sub-edit inverses in reverse order.
    Batch(Vec<EditKind>),
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

/// The single writer of `DocumentContent` and every emitter's preview
/// `EffectAsset`.
///
/// Every edit mutates the canonical [`EffectGraph`] via [`apply_to_graph`]. A
/// structural or topology edit — anything that isn't recognised as a pure
/// GPU-bound value tweak by [`fast_upload_target`] — then re-validates and
/// re-bakes the *whole* document transactionally through
/// [`bake_effect_records`]: on success the fresh records are published and every
/// emitter's scene entity is despawned/respawned (see the `CachedPipelines`
/// ordering note in `crate::plugins::reconcile`); on failure (e.g. a cyclic
/// GPU-event topology) nothing is published, the last-known-good records and
/// running preview are left untouched, and the failure is logged — the
/// authored graph stays dirty so the user isn't silently left believing the
/// edit landed. Either way undo history is still recorded, so a bad edit can
/// be undone like any other.
///
/// [`EffectGraph`]: crate::effect_graph::model::EffectGraph
pub fn apply_edits(
    mut requests: MessageReader<EditRequest>,
    mut applied: MessageWriter<EditApplied>,
    mut playback: MessageWriter<PlaybackCommand>,
    mut contents: Query<&mut DocumentContent>,
    mut doc_uis: Query<&mut DocumentUi>,
    mut emitters: ResMut<Assets<EffectAsset>>,
    children_q: Query<&Children>,
    scene_roots: Query<&EmitterSceneEntities, With<DocumentSceneRoot>>,
    mut proxies: Query<&mut ProxyEmitters>,
    mut emitter_props: Query<&mut EffectProperties>,
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

        // `MoveLayout` is view-only: it updates saved canvas positions in the
        // document's `GraphView`, not the graph. The widget has already written
        // the live positions during the drag, so applying `to` is idempotent;
        // it's replayed here so undo/redo can drive it. No re-bake or respawn,
        // and `is_literal_edit` keeps the proxy untouched.
        if let EditKind::MoveLayout { nodes, stacks } = &req.kind {
            if let Ok(mut ui) = doc_uis.get_mut(req.doc) {
                for m in nodes {
                    ui.graph_view.positions.insert(m.id, m.to);
                }
                for m in stacks {
                    ui.graph_view.stack_positions.insert(m.id, m.to);
                }
            }
            content.mark_dirty(true);
            let inverse_kind = EditKind::MoveLayout {
                nodes: nodes.iter().map(PositionChange::inverted).collect(),
                stacks: stacks.iter().map(PositionChange::inverted).collect(),
            };
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

        let registry = type_registry.read();

        // Mutate the canonical document and capture the inverse edit.
        let inverse_kind = match apply_to_graph(
            content.effect_graph_mut(),
            &registry,
            &req.kind,
            req.direction,
        ) {
            Ok(inverse) => inverse,
            Err(err) => {
                warn!("edit refused ({err}): {:?}", req.kind);
                continue;
            }
        };
        content.mark_dirty(true);

        // Which emitter this edit's live preview belongs to, resolved from
        // whichever side of the edit names one (see `emitter_of`), falling back
        // to a `EffectGraph` lookup for the handful of topology edits that only
        // carry a bare node/source id.
        let changed_emitter = emitter_of(&req.kind)
            .or_else(|| emitter_of(&inverse_kind))
            .or_else(|| match &req.kind {
                EditKind::CreateGpuEmitter {
                    event_node: Some(node),
                }
                | EditKind::AddEventLink { node, .. }
                | EditKind::RemoveEventLink { node, .. } => {
                    content.effect_graph().emitter_owning_node(*node)
                }
                EditKind::SetCpuSpawnerSettings { source, .. } => {
                    content.effect_graph().emitter_for_source(*source)
                }
                _ => None,
            });

        // A property's `exposed` flag is purely a bake-time concern: it only
        // selects how the property is lowered when baking (a runtime `Module`
        // property vs. literals inlined at each reference). The live proxy
        // already promotes every value to a GPU property, so the running preview
        // is identical either way for the emitter the property belongs to.
        // Persist the graph change (for save) and record undo history, but skip
        // the re-bake / recompile / respawn that would needlessly reset the
        // simulation.
        if matches!(req.kind, EditKind::SetPropertyExposed { .. }) {
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

        // Live value-upload fast path: an edit that only changes a value already
        // backed by a GPU property — a promoted literal tweak, or an exposed
        // user property's default — can be pushed straight to that emitter's GPU
        // instance via `EffectProperties`, skipping the re-bake / shader
        // recompile / respawn. Edits with no such binding (render-reachable or
        // non-promotable literals, unexposed properties, or a topology edit with
        // no single resolvable emitter) fall through to the full transactional
        // rebake below.
        let fast_upload = changed_emitter.and_then(|emitter| {
            let proxy_instance = proxies.get(req.doc).ok().and_then(|p| p.get(emitter));
            let uploads = fast_upload_target(&req.kind, emitter, &content, proxy_instance)?;
            Some((emitter, uploads))
        });
        if let Some((emitter, uploads)) = fast_upload
            && let Some(pe) = proxy_props_entity(req.doc, emitter, &children_q, &scene_roots)
        {
            for (name, value) in &uploads {
                if let Ok(props) = emitter_props.get_mut(pe) {
                    EffectProperties::set_if_changed(props, name, *value);
                }
            }
            // Remember the tweaked values so a later `Respawn` re-seeds them:
            // the proxy asset's property defaults stay stale until the next
            // structural rebake, so a respawned instance would otherwise revert.
            if let Ok(mut proxy_emitters) = proxies.get_mut(req.doc)
                && let Some(instance) = proxy_emitters.get_mut(emitter)
            {
                for (name, value) in uploads {
                    instance.current_values.insert(name, value);
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

        // Full transactional rebake: validate and bake every emitter in the
        // document together (see `bake_effect_records`), so a structural change to
        // one emitter's topology (e.g. a GPU event link) that breaks another's
        // bake is never published half-applied. On success, publish every
        // fresh record together and respawn the whole document hierarchy (every
        // emitter's scene entity) so no stale `CachedPipelines` survives the new
        // layout. On failure, keep the last-known-good records and running
        // preview exactly as they were: the authored graph stays dirty (already
        // marked above) and the failure is logged, but nothing about the live
        // scene changes.
        let bake_result = bake_effect_records(
            content.effect_graph(),
            &registry,
            content.preview_tag(),
            &mut emitters,
        );
        let is_literal_edit = match bake_result {
            Ok(records) => {
                content.set_emitter_records(records);
                playback.write(PlaybackCommand::Respawn(req.doc));
                false
            }
            Err(bake_errors) => {
                error!(
                    "apply_edits: whole-document rebake failed for {:?}, keeping last valid preview: {bake_errors:?}",
                    req.doc
                );
                if let Some(records) = bake_without_empty_gpu_branches(
                    content.effect_graph(),
                    &bake_errors,
                    &registry,
                    content.preview_tag(),
                    &mut emitters,
                ) {
                    content.set_emitter_records(records);
                    content.set_bake_errors(bake_errors);
                    playback.write(PlaybackCommand::Respawn(req.doc));
                    false
                } else {
                    content.set_bake_errors(bake_errors);
                    true
                }
            }
        };
        drop(registry);

        applied.write(EditApplied {
            doc: req.doc,
            inverse: EditRequest {
                doc: req.doc,
                direction: req.direction, // unused on inverse, kept for symmetry
                kind: inverse_kind,
            },
            direction: req.direction,
            is_literal_edit,
        });
    }
}

/// The [`EmitterId`] `kind` targets, if it names one directly.
///
/// `None` for document-scoped edits (`RenameDocument`, `MoveLayout`) and for
/// the handful of topology edits whose only affected emitter must be resolved
/// dynamically against a live `EffectGraph` (`AddEventLink`, `RemoveEventLink`,
/// `SetCpuSpawnerSettings`, and a `CreateEmitter`/`CreateCpuSource` /
/// `DeleteSource` not yet paired with its inverse) — callers needing those
/// fall back to a `EffectGraph` lookup.
fn emitter_of(kind: &EditKind) -> Option<EmitterId> {
    match kind {
        EditKind::RenameDocument { .. }
        | EditKind::MoveLayout { .. }
        | EditKind::CreateEmitter { .. }
        | EditKind::CreateCpuSource { .. }
        | EditKind::DeleteSource { .. }
        | EditKind::AddEventLink { .. }
        | EditKind::RemoveEventLink { .. }
        | EditKind::SetCpuSpawnerSettings { .. } => None,
        EditKind::CreateGpuEmitter { .. } => None,
        EditKind::DeleteEmitter { emitter } => Some(*emitter),
        EditKind::InsertEmitter { removed } => Some(removed.emitter.id),
        EditKind::InsertSource { removed } => removed.source_link.map(|l| l.emitter),
        EditKind::SetSourceLink { emitter, .. } | EditKind::RemoveSourceLink { emitter, .. } => {
            Some(*emitter)
        }
        EditKind::SetEmitterName { emitter, .. }
        | EditKind::SetSimulationSpace { emitter, .. }
        | EditKind::SetSimulationCondition { emitter, .. }
        | EditKind::SetCapacity { emitter, .. }
        | EditKind::SetZLayer2d { emitter, .. }
        | EditKind::AddModifierFromTemplate { emitter, .. }
        | EditKind::InsertModifierNode { emitter, .. }
        | EditKind::RemoveModifier { emitter, .. }
        | EditKind::MoveModifier { emitter, .. }
        | EditKind::SetModifierAttribute { emitter, .. }
        | EditKind::SetModifierConfig { emitter, .. }
        | EditKind::SetInputDefault { emitter, .. }
        | EditKind::SetInputImageBinding { emitter, .. }
        | EditKind::SetLiteralValue { emitter, .. }
        | EditKind::AddExprNode { emitter, .. }
        | EditKind::RemoveNode { emitter, .. }
        | EditKind::InsertNode { emitter, .. }
        | EditKind::AddImageNode { emitter, .. }
        | EditKind::SetImageNodeBinding { emitter, .. }
        | EditKind::AddTextureSlot { emitter }
        | EditKind::RemoveTextureSlot { emitter, .. }
        | EditKind::InsertTextureSlot { emitter, .. }
        | EditKind::RenameTextureSlot { emitter, .. }
        | EditKind::ReorderTextureSlot { emitter, .. }
        | EditKind::AddLink { emitter, .. }
        | EditKind::RemoveLink { emitter, .. }
        | EditKind::AddProperty { emitter, .. }
        | EditKind::RemoveProperty { emitter, .. }
        | EditKind::RestoreProperty { emitter, .. }
        | EditKind::RenameProperty { emitter, .. }
        | EditKind::SetPropertyDefault { emitter, .. }
        | EditKind::SetPropertyExposed { emitter, .. } => Some(*emitter),
        EditKind::Batch(kinds) => kinds.first().and_then(emitter_of),
    }
}

fn create_gpu_driven_emitter(
    effect_graph: &mut EffectGraph,
    source: SourceId,
) -> Result<EmitterId, String> {
    let emitter = graph_edit::create_emitter(effect_graph, SharedStr::from("New GPU Emitter"));
    let displaced = graph_edit::set_source_link(effect_graph, source, emitter);
    if !displaced.is_empty() {
        let _ = graph_edit::delete_emitter(effect_graph, emitter);
        for link in displaced {
            let _ = graph_edit::set_source_link(effect_graph, link.source, link.emitter);
        }
        return Err("GPU event source is already connected".to_string());
    }
    Ok(emitter)
}

/// Bake a preview projection with invalid GPU-event branches removed.
///
/// The canonical graph is untouched and remains saveable. This fallback only
/// handles empty particle layouts on GPU-driven emitters; all other bake
/// failures retain the previous preview.
fn bake_without_empty_gpu_branches(
    effect_graph: &EffectGraph,
    errors: &[crate::effect_graph::bake::EffectBakeError],
    registry: &bevy::reflect::TypeRegistry,
    preview_tag: u64,
    assets: &mut Assets<EffectAsset>,
) -> Option<std::collections::HashMap<EmitterId, crate::document::EmitterRecord>> {
    let mut omitted: HashSet<EmitterId> = errors
        .iter()
        .filter(|error| error.error.message.starts_with("shader generation failed:"))
        .filter(|error| {
            error
                .source
                .and_then(|source| effect_graph.source(source))
                .is_some_and(|source| matches!(source.kind, SourceKind::GpuEvent))
        })
        .map(|error| error.emitter)
        .collect();
    if omitted.is_empty() {
        return None;
    }

    loop {
        let descendants: Vec<EmitterId> = effect_graph
            .emitters
            .iter()
            .map(|emitter| emitter.id)
            .filter(|emitter| {
                effect_graph
                    .parent_emitter(*emitter)
                    .is_some_and(|parent| omitted.contains(&parent))
            })
            .collect();
        let old_len = omitted.len();
        omitted.extend(descendants);
        if omitted.len() == old_len {
            break;
        }
    }

    let mut preview = effect_graph.clone();
    for emitter in omitted {
        let source = preview.source_for_emitter(emitter);
        let _ = graph_edit::delete_emitter(&mut preview, emitter);
        if let Some(source) = source {
            let _ = graph_edit::delete_source(&mut preview, source);
        }
    }
    bake_effect_records(&preview, registry, preview_tag, assets).ok()
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
    emitter: EmitterId,
    content: &DocumentContent,
    proxy: Option<&ProxyInstance>,
) -> Option<Vec<(String, Value)>> {
    let graph = content.effect_graph().emitter(emitter)?;
    match kind {
        EditKind::SetInputDefault {
            node, port, new, ..
        } => {
            let site = LiteralSite::Input {
                node: *node,
                port: port.clone(),
            };
            Some(vec![(proxy?.tweak_props.get(&site)?.clone(), *new)])
        }
        EditKind::SetLiteralValue { node, new, .. } => {
            let site = LiteralSite::Node(*node);
            Some(vec![(proxy?.tweak_props.get(&site)?.clone(), *new)])
        }
        EditKind::SetPropertyDefault { id, new, .. } => {
            let def = graph.properties.iter().find(|p| p.id == *id)?;
            if def.exposed {
                return Some(vec![(def.name.to_string(), *new)]);
            }
            let proxy = proxy?;
            let mut uploads = Vec::new();
            for n in &graph.nodes {
                if let NodePayload::Expr(ExprNode::Property(pid)) = &n.payload
                    && *pid == *id
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

/// Sibling operand defaults to retype when `link` connects into an operator.
///
/// Returns the `(node, port, new default)` retypes for each unlinked sibling
/// operand of the link's target that must adopt the connected value's type.
/// Empty unless the target is an operator whose operands must share a type (see
/// [`ExprNode::operands_share_type`]), the source resolves to a value, and some
/// sibling currently carries a differently typed default. `emitter` is the
/// pipeline `link` belongs to; the connected source is looked up globally
/// (via `GraphReader`) since [`AddLink`](EditKind::AddLink) never crosses
/// emitters but the reader itself is document-wide.
///
/// [`ExprNode::operands_share_type`]: crate::effect_graph::model::ExprNode::operands_share_type
fn sibling_operand_retypes(
    effect_graph: &EffectGraph,
    emitter: EmitterId,
    registry: &bevy::reflect::TypeRegistry,
    link: &GraphLink,
) -> Vec<(NodeId, SharedStr, Value)> {
    let Some(graph) = effect_graph.emitter(emitter) else {
        return Vec::new();
    };
    let target = link.to.node;
    let Some(node) = graph.node(target) else {
        return Vec::new();
    };
    let NodePayload::Expr(expr) = &node.payload else {
        return Vec::new();
    };
    if !expr.operands_share_type() {
        return Vec::new();
    }

    let reader = GraphReader::new(effect_graph, registry);
    let Some(connected) = reader.node_output_value_type(link.from.node) else {
        return Vec::new();
    };
    let Some(zero) = value_type_zero(connected) else {
        return Vec::new();
    };

    expr.input_ports()
        .iter()
        .copied()
        .filter(|&port| port != &*link.to.port)
        .filter(|&port| {
            let linked = graph
                .links
                .iter()
                .any(|l| l.to.node == target && &*l.to.port == port);
            let mismatched = node
                .inputs
                .iter()
                .find(|s| &*s.name == port)
                .and_then(|s| s.default.as_value())
                .is_some_and(|v| v.value_type() != connected);
            !linked && mismatched
        })
        .map(|port| (target, SharedStr::from(port), zero))
        .collect()
}

/// Apply one [`EditKind`] to the canonical graph and return the inverse edit.
///
/// Resilience principle: a refused edit (missing node/property, unregistered
/// modifier, out-of-range index) returns `Err` and is skipped by the caller —
/// never a panic. `RenameDocument` and `MoveLayout` are handled by the caller
/// and are unreachable here.
fn apply_to_graph(
    effect_graph: &mut EffectGraph,
    registry: &bevy::reflect::TypeRegistry,
    kind: &EditKind,
    direction: EditDirection,
) -> Result<EditKind, String> {
    Ok(match kind {
        EditKind::RenameDocument { .. } => {
            unreachable!("RenameDocument is handled before re-baking")
        }
        EditKind::MoveLayout { .. } => {
            unreachable!("MoveLayout is handled before re-baking")
        }

        // Apply each sub-edit in order; the inverse replays the sub-inverses in
        // reverse so the batch undoes exactly.
        EditKind::Batch(kinds) => {
            let mut inverses = Vec::with_capacity(kinds.len());
            for sub in kinds {
                inverses.push(apply_to_graph(effect_graph, registry, sub, direction)?);
            }
            inverses.reverse();
            EditKind::Batch(inverses)
        }

        // --- Document topology: emitter pipelines and spawn sources ---
        EditKind::CreateEmitter { name } => {
            let emitter = graph_edit::create_emitter(effect_graph, SharedStr::from(name.as_str()));
            EditKind::DeleteEmitter { emitter }
        }
        EditKind::DeleteEmitter { emitter } => {
            if effect_graph.emitters.len() <= 1 {
                return Err("cannot delete a document's last emitter pipeline".to_string());
            }
            let removed = graph_edit::delete_emitter(effect_graph, *emitter)
                .ok_or("emitter pipeline not found")?;
            EditKind::InsertEmitter { removed }
        }
        EditKind::InsertEmitter { removed } => {
            let emitter = removed.emitter.id;
            graph_edit::insert_emitter(effect_graph, removed.clone());
            EditKind::DeleteEmitter { emitter }
        }
        EditKind::CreateCpuSource { settings } => {
            let source = graph_edit::create_source(
                effect_graph,
                SourceKind::CpuSpawner {
                    settings: *settings,
                },
            );
            EditKind::DeleteSource { source }
        }
        EditKind::CreateGpuEmitter { event_node } => {
            let source = graph_edit::create_source(effect_graph, SourceKind::GpuEvent);
            let emitter = match create_gpu_driven_emitter(effect_graph, source) {
                Ok(emitter) => emitter,
                Err(error) => {
                    let _ = graph_edit::delete_source(effect_graph, source);
                    return Err(error);
                }
            };
            if let Some(node) = event_node
                && !graph_edit::add_event_link(effect_graph, *node, source)
            {
                let _ = graph_edit::delete_source(effect_graph, source);
                let _ = graph_edit::delete_emitter(effect_graph, emitter);
                return Err("failed to connect emitter to new GPU source".to_string());
            }
            EditKind::Batch(vec![
                EditKind::DeleteSource { source },
                EditKind::DeleteEmitter { emitter },
            ])
        }
        EditKind::DeleteSource { source } => {
            let removed =
                graph_edit::delete_source(effect_graph, *source).ok_or("source not found")?;
            EditKind::InsertSource { removed }
        }
        EditKind::InsertSource { removed } => {
            let source = removed.source.id;
            graph_edit::insert_source(effect_graph, removed.clone());
            EditKind::DeleteSource { source }
        }
        EditKind::SetSourceLink { source, emitter } => {
            let displaced = graph_edit::set_source_link(effect_graph, *source, *emitter);
            match displaced.as_slice() {
                [] => EditKind::RemoveSourceLink {
                    source: *source,
                    emitter: *emitter,
                },
                _ => EditKind::Batch(
                    displaced
                        .into_iter()
                        .map(|l| EditKind::SetSourceLink {
                            source: l.source,
                            emitter: l.emitter,
                        })
                        .collect(),
                ),
            }
        }
        EditKind::RemoveSourceLink { source, emitter } => {
            if !graph_edit::remove_source_link(effect_graph, *source, *emitter) {
                return Err("no such source link".to_string());
            }
            EditKind::SetSourceLink {
                source: *source,
                emitter: *emitter,
            }
        }
        EditKind::AddEventLink { node, target } => {
            if !graph_edit::add_event_link(effect_graph, *node, *target) {
                return Err("event link already exists".to_string());
            }
            if effect_graph.emitter_for_source(*target).is_none()
                && effect_graph
                    .source(*target)
                    .is_some_and(|source| matches!(source.kind, SourceKind::GpuEvent))
            {
                let emitter = match create_gpu_driven_emitter(effect_graph, *target) {
                    Ok(emitter) => emitter,
                    Err(error) => {
                        let _ = graph_edit::remove_event_link(effect_graph, *node, *target);
                        return Err(error);
                    }
                };
                EditKind::Batch(vec![
                    EditKind::RemoveEventLink {
                        node: *node,
                        target: *target,
                    },
                    EditKind::DeleteEmitter { emitter },
                ])
            } else {
                EditKind::RemoveEventLink {
                    node: *node,
                    target: *target,
                }
            }
        }
        EditKind::RemoveEventLink { node, target } => {
            if !graph_edit::remove_event_link(effect_graph, *node, *target) {
                return Err("no such event link".to_string());
            }
            EditKind::AddEventLink {
                node: *node,
                target: *target,
            }
        }
        EditKind::SetCpuSpawnerSettings { source, new } => {
            let old = graph_edit::set_cpu_spawner_settings(effect_graph, *source, *new)
                .ok_or("source is not a CPU spawner")?;
            EditKind::SetCpuSpawnerSettings {
                source: *source,
                new: old,
            }
        }

        // --- Emitter settings ---
        EditKind::SetEmitterName { emitter, new } => {
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            let old = graph_edit::set_emitter_name(graph, SharedStr::from(new.as_str()));
            EditKind::SetEmitterName {
                emitter: *emitter,
                new: old.to_string(),
            }
        }
        EditKind::SetSimulationSpace { emitter, new } => {
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            let old = graph_edit::set_simulation_space(graph, *new);
            EditKind::SetSimulationSpace {
                emitter: *emitter,
                new: old,
            }
        }
        EditKind::SetSimulationCondition { emitter, new } => {
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            let old = graph_edit::set_simulation_condition(graph, *new);
            EditKind::SetSimulationCondition {
                emitter: *emitter,
                new: old,
            }
        }
        EditKind::SetCapacity { emitter, new } => {
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            let old = graph_edit::set_capacity(graph, *new);
            EditKind::SetCapacity {
                emitter: *emitter,
                new: old,
            }
        }
        EditKind::SetZLayer2d { emitter, new } => {
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            let old = graph_edit::set_z_layer_2d(graph, *new);
            EditKind::SetZLayer2d {
                emitter: *emitter,
                new: old,
            }
        }

        // --- Modifier stacks ---
        EditKind::AddModifierFromTemplate {
            emitter,
            group,
            type_id,
            at,
        } => {
            let id = graph_edit::add_modifier_from_template(
                effect_graph,
                *emitter,
                registry,
                *group,
                *type_id,
                *at,
            )
            .ok_or("modifier type is not registered")?;
            let graph = effect_graph.emitter(*emitter).ok_or("emitter not found")?;
            let idx = graph
                .stack(*group)
                .and_then(|s| s.members.iter().position(|m| *m == id))
                .ok_or("added modifier not found in its stack")?;
            EditKind::RemoveModifier {
                emitter: *emitter,
                group: *group,
                idx,
            }
        }
        EditKind::InsertModifierNode { emitter, removed } => {
            let group = removed.group;
            let node_id = removed.node.id;
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            if !graph_edit::insert_modifier(graph, removed.clone()) {
                return Err("target stack is missing".to_string());
            }
            let idx = graph
                .stack(group)
                .and_then(|s| s.members.iter().position(|m| *m == node_id))
                .ok_or("inserted modifier not found in its stack")?;
            EditKind::RemoveModifier {
                emitter: *emitter,
                group,
                idx,
            }
        }
        EditKind::RemoveModifier {
            emitter,
            group,
            idx,
        } => {
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            let removed = graph_edit::remove_modifier(graph, *group, *idx)
                .ok_or("no modifier at the given index")?;
            EditKind::InsertModifierNode {
                emitter: *emitter,
                removed,
            }
        }
        EditKind::MoveModifier {
            emitter,
            group,
            from,
            to,
        } => {
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            if !graph_edit::move_stack_member(graph, *group, *from, *to) {
                return Err("move index out of range".to_string());
            }
            EditKind::MoveModifier {
                emitter: *emitter,
                group: *group,
                from: *to,
                to: *from,
            }
        }
        EditKind::SetModifierAttribute {
            emitter,
            group,
            idx,
            new,
            reset_value,
        } => {
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            let (old_attr, rewrote_old) =
                graph_edit::set_modifier_attribute(graph, *group, *idx, *new, *reset_value)?;
            EditKind::SetModifierAttribute {
                emitter: *emitter,
                group: *group,
                idx: *idx,
                new: old_attr,
                reset_value: rewrote_old,
            }
        }
        EditKind::SetModifierConfig {
            emitter,
            node,
            field,
            new,
        } => {
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            let old = graph_edit::set_modifier_config(graph, *node, field, new.clone())
                .ok_or("modifier node has no such config field")?;
            EditKind::SetModifierConfig {
                emitter: *emitter,
                node: *node,
                field: field.clone(),
                new: old,
            }
        }

        // --- Expression input defaults ---
        EditKind::SetInputDefault {
            emitter,
            node,
            port,
            new,
        } => {
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            let old = graph_edit::set_input_default(graph, *node, port, *new);
            EditKind::SetInputDefault {
                emitter: *emitter,
                node: *node,
                port: port.clone(),
                new: old.unwrap_or(*new),
            }
        }
        EditKind::SetInputImageBinding {
            emitter,
            node,
            port,
            binding,
        } => {
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            let old = graph_edit::set_input_image_binding(graph, *node, port, binding.clone())
                .ok_or("node not found")?;
            EditKind::SetInputImageBinding {
                emitter: *emitter,
                node: *node,
                port: port.clone(),
                binding: old,
            }
        }
        EditKind::SetLiteralValue { emitter, node, new } => {
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            let old = graph_edit::set_literal_node(graph, *node, *new)
                .ok_or("node is not a literal expression")?;
            EditKind::SetLiteralValue {
                emitter: *emitter,
                node: *node,
                new: old,
            }
        }

        // --- Standalone expression nodes ---
        EditKind::AddExprNode {
            emitter,
            expr,
            inputs,
        } => {
            let id =
                graph_edit::add_expr_node(effect_graph, *emitter, expr.clone(), inputs.clone())?;
            EditKind::RemoveNode {
                emitter: *emitter,
                id,
            }
        }
        EditKind::RemoveNode { emitter, id } => {
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            let removed = graph_edit::remove_node(graph, *id).ok_or("node not found")?;
            EditKind::InsertNode {
                emitter: *emitter,
                removed,
            }
        }
        EditKind::InsertNode { emitter, removed } => {
            let id = removed.node.id;
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            graph_edit::insert_node(graph, removed.clone());
            EditKind::RemoveNode {
                emitter: *emitter,
                id,
            }
        }

        // --- Image source nodes and texture slots ---
        EditKind::AddImageNode { emitter, binding } => {
            let id = graph_edit::add_image_node(effect_graph, *emitter, binding.clone())?;
            EditKind::RemoveNode {
                emitter: *emitter,
                id,
            }
        }
        EditKind::SetImageNodeBinding {
            emitter,
            node,
            binding,
        } => {
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            let old = graph_edit::set_image_node_binding(graph, *node, binding.clone())
                .ok_or("not an image node")?;
            EditKind::SetImageNodeBinding {
                emitter: *emitter,
                node: *node,
                binding: old,
            }
        }
        EditKind::AddTextureSlot { emitter } => {
            let id = graph_edit::add_texture_slot(effect_graph, *emitter)?;
            EditKind::RemoveTextureSlot {
                emitter: *emitter,
                id,
            }
        }
        EditKind::RemoveTextureSlot { emitter, id } => {
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            let removed =
                graph_edit::remove_texture_slot(graph, *id).ok_or("texture slot not found")?;
            EditKind::InsertTextureSlot {
                emitter: *emitter,
                removed,
            }
        }
        EditKind::InsertTextureSlot { emitter, removed } => {
            let id = removed.slot.id;
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            graph_edit::insert_texture_slot(graph, removed.clone());
            EditKind::RemoveTextureSlot {
                emitter: *emitter,
                id,
            }
        }
        EditKind::RenameTextureSlot { emitter, id, new } => {
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            let old = graph_edit::rename_texture_slot(graph, *id, new.clone())
                .ok_or("texture slot not found")?;
            EditKind::RenameTextureSlot {
                emitter: *emitter,
                id: *id,
                new: old,
            }
        }
        EditKind::ReorderTextureSlot { emitter, id, to } => {
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            let from = graph_edit::reorder_texture_slot(graph, *id, *to)
                .ok_or("texture slot not found")?;
            EditKind::ReorderTextureSlot {
                emitter: *emitter,
                id: *id,
                to: from,
            }
        }

        // --- Links ---
        EditKind::AddLink { emitter, link } => {
            let to_node = link.to.node;
            // Retype the target operator's sibling operand defaults to the type
            // being connected, so element-wise operators whose WGSL requires
            // matching operands don't bake to invalid code (e.g. `vec3 + f32`).
            // Only derived for a fresh edit: an undo/redo replay already carries
            // the retypes in its batch, so re-deriving here would nest them.
            let retypes = if direction == EditDirection::Fresh {
                sibling_operand_retypes(effect_graph, *emitter, registry, link)
            } else {
                Vec::new()
            };
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            let link_inverse = match graph_edit::add_link(graph, link.clone()) {
                Some(displaced) => EditKind::AddLink {
                    emitter: *emitter,
                    link: displaced,
                },
                None => EditKind::RemoveLink {
                    emitter: *emitter,
                    link: link.clone(),
                },
            };
            graph_edit::normalize_select_image(graph, to_node);

            if retypes.is_empty() {
                link_inverse
            } else {
                let mut inverses = Vec::with_capacity(retypes.len() + 1);
                for (node, port, new) in retypes {
                    let old = graph_edit::set_input_default(graph, node, &port, new);
                    inverses.push(EditKind::SetInputDefault {
                        emitter: *emitter,
                        node,
                        port,
                        new: old.unwrap_or(new),
                    });
                }
                inverses.reverse();
                inverses.push(link_inverse);
                EditKind::Batch(inverses)
            }
        }
        EditKind::RemoveLink { emitter, link } => {
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            let removed = graph_edit::remove_link_to(graph, &link.to)
                .ok_or("no link targets that input port")?;
            graph_edit::normalize_select_image(graph, link.to.node);
            EditKind::AddLink {
                emitter: *emitter,
                link: removed,
            }
        }

        // --- User properties ---
        EditKind::AddProperty {
            emitter,
            name,
            value,
            exposed,
        } => {
            if crate::proxy::is_tweak_prop_name(name) {
                return Err(format!("property name {name:?} uses the reserved prefix"));
            }
            let id = graph_edit::add_property(
                effect_graph,
                *emitter,
                SharedStr::from(name.as_str()),
                *value,
                *exposed,
            )?;
            EditKind::RemoveProperty {
                emitter: *emitter,
                id,
            }
        }
        EditKind::RemoveProperty { emitter, id } => {
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            let removed = graph_edit::remove_property(graph, *id).ok_or("property not found")?;
            EditKind::RestoreProperty {
                emitter: *emitter,
                removed,
            }
        }
        EditKind::RestoreProperty { emitter, removed } => {
            let id = removed.def.id;
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            graph_edit::restore_property(graph, removed.clone());
            EditKind::RemoveProperty {
                emitter: *emitter,
                id,
            }
        }
        EditKind::RenameProperty { emitter, id, new } => {
            if crate::proxy::is_tweak_prop_name(new) {
                return Err(format!("property name {new:?} uses the reserved prefix"));
            }
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            let old = graph_edit::rename_property(graph, *id, SharedStr::from(new.as_str()))
                .ok_or("property not found")?;
            EditKind::RenameProperty {
                emitter: *emitter,
                id: *id,
                new: old.to_string(),
            }
        }
        EditKind::SetPropertyDefault { emitter, id, new } => {
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            let old =
                graph_edit::set_property_default(graph, *id, *new).ok_or("property not found")?;
            EditKind::SetPropertyDefault {
                emitter: *emitter,
                id: *id,
                new: old,
            }
        }
        EditKind::SetPropertyExposed {
            emitter,
            id,
            exposed,
        } => {
            let graph = effect_graph
                .emitter_mut(*emitter)
                .ok_or("emitter not found")?;
            let old = graph_edit::set_property_exposed(graph, *id, *exposed)
                .ok_or("property not found")?;
            EditKind::SetPropertyExposed {
                emitter: *emitter,
                id: *id,
                exposed: old,
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use bevy::reflect::TypeRegistry;

    use super::*;
    use crate::{
        effect_graph::{
            demo::demo_emitter,
            model::{EmitterGraph, EventLink, ModifierNodeData, NodePayload, PortRef, SourceLink},
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

    /// Wrap [`demo_emitter`] in a minimal single-emitter [`EffectGraph`], with
    /// `next_id` correctly seeded past every id the demo already uses.
    ///
    /// Mirrors the convention `crate::effect_graph::edit`'s own tests use:
    /// most of this module's edits are emitter-scoped and exercised against a
    /// single plain emitter, unconnected to any spawn source.
    fn demo_effect_single() -> (EffectGraph, EmitterId) {
        let graph = demo_emitter();
        let emitter_id = graph.id;
        let max_id = graph
            .nodes
            .iter()
            .map(|n| n.id.get())
            .chain(graph.stacks.iter().map(|s| s.id.get()))
            .chain(graph.properties.iter().map(|p| p.id.get()))
            .chain(graph.texture_slots.iter().map(|s| s.id.get()))
            .chain(std::iter::once(emitter_id.get()))
            .max()
            .unwrap();
        let mut effect_graph = EffectGraph::empty();
        effect_graph.next_id = max_id + 1;
        effect_graph.emitters.push(graph);
        (effect_graph, emitter_id)
    }

    /// A canonical copy of `g` for structural comparison.
    ///
    /// The `nodes`, `links` and `properties` collections are sorted (their Vec
    /// order carries no semantics — references are by id, and layout lives in
    /// `GraphLayout`). Stack member order is left untouched — it *is* semantic
    /// (the pipeline executes modifiers in that order).
    fn canonical(g: &EmitterGraph) -> EmitterGraph {
        let mut g = g.clone();
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

    /// Drive one emitter-scoped [`EditKind`] through the full undo/redo cycle
    /// against the [`demo_effect_single`] fixture.
    ///
    /// Asserts the inverse is correct:
    ///
    /// 1. Apply `edit` (it must change the emitter).
    /// 2. Apply the returned inverse (undo): the emitter returns to its
    ///    original structure (modulo the monotonic id allocator).
    /// 3. Apply that inverse's own inverse (redo): the emitter returns
    ///    *exactly* to the post-edit state, matching how `history` replays a
    ///    redo by re-applying the captured inverse rather than the original
    ///    edit.
    fn assert_round_trip(registry: &TypeRegistry, edit: EditKind) {
        let (effect_graph, emitter) = demo_effect_single();
        assert_round_trip_on(registry, emitter, effect_graph, edit);
    }

    /// Like [`assert_round_trip`], but from an explicit base document.
    ///
    /// E.g. the single-emitter fixture plus a synthetic standalone literal
    /// node.
    fn assert_round_trip_on(
        registry: &TypeRegistry,
        emitter: EmitterId,
        original: EffectGraph,
        edit: EditKind,
    ) {
        let mut effect_graph = original.clone();
        let inverse = apply_to_graph(&mut effect_graph, registry, &edit, EditDirection::Fresh)
            .unwrap_or_else(|e| panic!("forward edit refused ({e}): {edit:?}"));
        let post_edit = effect_graph
            .emitter(emitter)
            .expect("emitter still present")
            .clone();
        assert_ne!(
            canonical(effect_graph.emitter(emitter).unwrap()),
            canonical(original.emitter(emitter).unwrap()),
            "edit must change the graph: {edit:?}"
        );

        let redo = apply_to_graph(&mut effect_graph, registry, &inverse, EditDirection::Undo)
            .unwrap_or_else(|e| panic!("inverse refused ({e}): {inverse:?}"));
        assert_eq!(
            canonical(effect_graph.emitter(emitter).unwrap()),
            canonical(original.emitter(emitter).unwrap()),
            "undo must restore the original graph: {edit:?}"
        );

        apply_to_graph(&mut effect_graph, registry, &redo, EditDirection::Redo)
            .unwrap_or_else(|e| panic!("redo refused ({e}): {redo:?}"));
        assert_eq!(
            effect_graph.emitter(emitter).unwrap(),
            &post_edit,
            "redo must restore the post-edit state: {edit:?}"
        );
    }

    fn property_id(g: &EmitterGraph, name: &str) -> PropertyId {
        g.properties
            .iter()
            .find(|p| &*p.name == name)
            .unwrap_or_else(|| panic!("demo property {name:?} missing"))
            .id
    }

    /// A single-emitter fixture with one synthetic standalone literal node
    /// appended.
    ///
    /// The demo itself carries none. Returns the document, the emitter id, and
    /// the new node's id.
    fn demo_with_standalone_literal(value: Value) -> (EffectGraph, EmitterId, NodeId) {
        let (mut effect_graph, emitter) = demo_effect_single();
        let id = graph_edit::add_expr_node(
            &mut effect_graph,
            emitter,
            ExprNode::Literal(value),
            Vec::new(),
        )
        .expect("emitter exists");
        (effect_graph, emitter, id)
    }

    #[test]
    fn round_trip_emitter_setting_edits() {
        let app = registry_app();
        let registry = app.world().resource::<AppTypeRegistry>().read();
        let (effect_graph, emitter) = demo_effect_single();

        assert_round_trip(
            &registry,
            EditKind::SetEmitterName {
                emitter,
                new: "renamed".to_string(),
            },
        );
        assert_round_trip(
            &registry,
            EditKind::SetSimulationSpace {
                emitter,
                new: SimulationSpace::Local,
            },
        );
        assert_round_trip(
            &registry,
            EditKind::SetSimulationCondition {
                emitter,
                new: SimulationCondition::Always,
            },
        );
        assert_round_trip(
            &registry,
            EditKind::SetCapacity {
                emitter,
                new: 16384,
            },
        );
        assert_round_trip(&registry, EditKind::SetZLayer2d { emitter, new: 3.0 });
        drop(effect_graph);
    }

    #[test]
    fn round_trip_modifier_stack_edits() {
        let app = registry_app();
        let registry = app.world().resource::<AppTypeRegistry>().read();
        let (effect_graph, emitter) = demo_effect_single();
        let g = effect_graph.emitter(emitter).unwrap();

        let update_len = g.stack(ModifierGroup::Update).unwrap().members.len();
        assert_round_trip(
            &registry,
            EditKind::AddModifierFromTemplate {
                emitter,
                group: ModifierGroup::Update,
                type_id: TypeId::of::<bevy_hanabi::AccelModifier>(),
                at: update_len,
            },
        );

        assert_round_trip(
            &registry,
            EditKind::RemoveModifier {
                emitter,
                group: ModifierGroup::Init,
                idx: 0,
            },
        );

        assert_round_trip(
            &registry,
            EditKind::MoveModifier {
                emitter,
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
        let (effect_graph, emitter) = demo_effect_single();
        let g = effect_graph.emitter(emitter).unwrap();

        // The demo's lifetime node (Init #2) is a `SetAttributeModifier`.
        // Same-typed retarget (LIFETIME -> AGE, both f32): no value reset.
        assert_round_trip(
            &registry,
            EditKind::SetModifierAttribute {
                emitter,
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
                emitter,
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
                emitter,
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
        let (effect_graph, emitter) = demo_effect_single();
        let g = effect_graph.emitter(emitter).unwrap();

        // The position-sphere node has an unlinked `center` input slot.
        let pos = g.stack(ModifierGroup::Init).unwrap().members[0];
        assert_round_trip(
            &registry,
            EditKind::SetInputDefault {
                emitter,
                node: pos,
                port: "center".into(),
                new: Vec3::new(9.0, 9.0, 9.0).into(),
            },
        );

        let (graph2, emitter2, literal) = demo_with_standalone_literal(Value::from(0.0f32));
        assert_round_trip_on(
            &registry,
            emitter2,
            graph2,
            EditKind::SetLiteralValue {
                emitter: emitter2,
                node: literal,
                new: Value::from(999.0f32),
            },
        );
    }

    #[test]
    fn round_trip_expr_nodes() {
        let app = registry_app();
        let registry = app.world().resource::<AppTypeRegistry>().read();
        let (effect_graph, emitter) = demo_effect_single();
        drop(effect_graph);

        assert_round_trip(
            &registry,
            EditKind::AddExprNode {
                emitter,
                expr: ExprNode::Literal(Value::from(3.0f32)),
                inputs: Vec::new(),
            },
        );

        // Removing a standalone literal (no links, no stack membership) and
        // re-inserting it must restore the node exactly.
        let (graph2, emitter2, literal) = demo_with_standalone_literal(Value::from(7.0f32));
        assert_round_trip_on(
            &registry,
            emitter2,
            graph2,
            EditKind::RemoveNode {
                emitter: emitter2,
                id: literal,
            },
        );
    }

    #[test]
    fn round_trip_links() {
        let app = registry_app();
        let registry = app.world().resource::<AppTypeRegistry>().read();
        let (effect_graph, emitter) = demo_effect_single();
        let g = effect_graph.emitter(emitter).unwrap();

        // Removing an existing demo link round-trips through AddLink.
        let existing = g.links.first().cloned().expect("demo has links");
        assert_round_trip(
            &registry,
            EditKind::RemoveLink {
                emitter,
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
                emitter,
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
                emitter,
                link: GraphLink {
                    from: PortRef {
                        node: other_source,
                        port: OUTPUT_PORT.into(),
                    },
                    to: existing.to,
                },
            },
        );
    }

    #[test]
    fn round_trip_properties() {
        let app = registry_app();
        let registry = app.world().resource::<AppTypeRegistry>().read();
        let (effect_graph, emitter) = demo_effect_single();
        let g = effect_graph.emitter(emitter).unwrap();

        assert_round_trip(
            &registry,
            EditKind::AddProperty {
                emitter,
                name: "extra".to_string(),
                value: Value::from(1.0f32),
                exposed: false,
            },
        );

        // `gravity` is referenced by a Property node linked into a consumer;
        // removal deletes the node and inlines the value, and the inverse must
        // restore both.
        let gravity = property_id(g, "gravity");
        assert_round_trip(
            &registry,
            EditKind::RemoveProperty {
                emitter,
                id: gravity,
            },
        );
        assert_round_trip(
            &registry,
            EditKind::RenameProperty {
                emitter,
                id: gravity,
                new: "g".to_string(),
            },
        );
        assert_round_trip(
            &registry,
            EditKind::SetPropertyDefault {
                emitter,
                id: gravity,
                new: Vec3::new(1.0, 2.0, 3.0).into(),
            },
        );
        assert_round_trip(
            &registry,
            EditKind::SetPropertyExposed {
                emitter,
                id: gravity,
                exposed: false,
            },
        );
    }

    /// `CreateEmitter`/`DeleteEmitter` dispatch through
    /// `apply_to_graph` and round-trip via the effect-level (not
    /// per-emitter) comparison, since they add/remove a whole emitter.
    #[test]
    fn round_trip_create_delete_emitter() {
        let app = registry_app();
        let registry = app.world().resource::<AppTypeRegistry>().read();
        let (mut effect_graph, _existing) = demo_effect_single();
        let original = effect_graph.clone();

        let inverse = apply_to_graph(
            &mut effect_graph,
            &registry,
            &EditKind::CreateEmitter {
                name: "new_emitter".to_string(),
            },
            EditDirection::Fresh,
        )
        .expect("create emitter pipeline");
        assert_eq!(effect_graph.emitters.len(), original.emitters.len() + 1);
        let EditKind::DeleteEmitter {
            emitter: new_emitter,
        } = inverse
        else {
            panic!("expected DeleteEmitter inverse");
        };

        let redo = apply_to_graph(&mut effect_graph, &registry, &inverse, EditDirection::Undo)
            .expect("delete emitter pipeline (undo)");
        assert_eq!(effect_graph.emitters.len(), original.emitters.len());

        apply_to_graph(&mut effect_graph, &registry, &redo, EditDirection::Redo)
            .expect("insert emitter pipeline (redo)");
        assert_eq!(effect_graph.emitters.len(), original.emitters.len() + 1);
        assert!(effect_graph.emitter(new_emitter).is_some());
    }

    /// `CreateCpuSource`/`DeleteSource` and `SetSourceLink`/`RemoveSourceLink`
    /// dispatch through `apply_to_graph`, including source-link displacement
    /// producing a `Batch` inverse.
    #[test]
    fn round_trip_source_topology() {
        let app = registry_app();
        let registry = app.world().resource::<AppTypeRegistry>().read();
        let (mut effect_graph, effect_a) = demo_effect_single();
        let effect_b = graph_edit::create_emitter(&mut effect_graph, SharedStr::from("b"));

        let inverse = apply_to_graph(
            &mut effect_graph,
            &registry,
            &EditKind::CreateCpuSource {
                settings: SpawnerSettings::rate(30.0.into()),
            },
            EditDirection::Fresh,
        )
        .expect("create source");
        let EditKind::DeleteSource { source } = inverse else {
            panic!("expected DeleteSource inverse");
        };

        // Link the source to effect_a, then to effect_b: the second link must
        // displace the first, producing a `Batch` inverse that restores it.
        apply_to_graph(
            &mut effect_graph,
            &registry,
            &EditKind::SetSourceLink {
                source,
                emitter: effect_a,
            },
            EditDirection::Fresh,
        )
        .expect("link source to effect_a");
        let relink_inverse = apply_to_graph(
            &mut effect_graph,
            &registry,
            &EditKind::SetSourceLink {
                source,
                emitter: effect_b,
            },
            EditDirection::Fresh,
        )
        .expect("relink source to effect_b");
        assert!(
            matches!(relink_inverse, EditKind::Batch(_)),
            "displacing an existing source link must produce a Batch inverse, got {relink_inverse:?}"
        );
        assert_eq!(
            effect_graph.source_links,
            vec![SourceLink {
                source,
                emitter: effect_b
            }]
        );

        apply_to_graph(
            &mut effect_graph,
            &registry,
            &relink_inverse,
            EditDirection::Undo,
        )
        .expect("undo relink");
        assert_eq!(
            effect_graph.source_links,
            vec![SourceLink {
                source,
                emitter: effect_a
            }]
        );

        // Deleting the source must also drop its remaining source link.
        apply_to_graph(
            &mut effect_graph,
            &registry,
            &EditKind::DeleteSource { source },
            EditDirection::Fresh,
        )
        .expect("delete source");
        assert!(effect_graph.sources.is_empty());
        assert!(effect_graph.source_links.is_empty());
    }

    /// `AddEventLink`/`RemoveEventLink` and `SetCpuSpawnerSettings` dispatch
    /// through `apply_to_graph` and round-trip exactly.
    #[test]
    fn round_trip_event_link_and_cpu_spawner_settings() {
        let app = registry_app();
        let registry = app.world().resource::<AppTypeRegistry>().read();
        let (mut effect_graph, emitter) = demo_effect_single();
        let emitter = graph_edit::add_expr_node(
            &mut effect_graph,
            emitter,
            ExprNode::Literal(Value::from(1u32)),
            Vec::new(),
        )
        .expect("emitter exists");
        let gpu_source = graph_edit::create_source(&mut effect_graph, SourceKind::GpuEvent);

        let inverse = apply_to_graph(
            &mut effect_graph,
            &registry,
            &EditKind::AddEventLink {
                node: emitter,
                target: gpu_source,
            },
            EditDirection::Fresh,
        )
        .expect("add event link");
        assert_eq!(effect_graph.event_links.len(), 1);
        assert_eq!(
            effect_graph.emitter_for_source(gpu_source),
            effect_graph.emitters.last().map(|emitter| emitter.id)
        );
        apply_to_graph(&mut effect_graph, &registry, &inverse, EditDirection::Undo)
            .expect("remove event link");
        assert!(effect_graph.event_links.is_empty());
        assert!(effect_graph.emitter_for_source(gpu_source).is_none());

        let cpu_source = graph_edit::create_source(
            &mut effect_graph,
            SourceKind::CpuSpawner {
                settings: SpawnerSettings::rate(10.0.into()),
            },
        );
        let inverse = apply_to_graph(
            &mut effect_graph,
            &registry,
            &EditKind::SetCpuSpawnerSettings {
                source: cpu_source,
                new: SpawnerSettings::rate(99.0.into()),
            },
            EditDirection::Fresh,
        )
        .expect("set cpu spawner settings");
        apply_to_graph(&mut effect_graph, &registry, &inverse, EditDirection::Undo)
            .expect("restore cpu spawner settings");
        let SourceKind::CpuSpawner { settings } = &effect_graph.source(cpu_source).unwrap().kind
        else {
            panic!("expected a CPU spawner");
        };
        assert_eq!(*settings, SpawnerSettings::rate(10.0.into()));
    }

    #[test]
    fn create_gpu_emitter_links_the_allocated_source_and_emitter() {
        let app = registry_app();
        let registry = app.world().resource::<AppTypeRegistry>().read();
        let (mut effect_graph, parent_emitter) = demo_effect_single();
        let update_len = effect_graph
            .emitter(parent_emitter)
            .unwrap()
            .stack(ModifierGroup::Update)
            .unwrap()
            .members
            .len();
        let event_node = graph_edit::add_modifier_from_template(
            &mut effect_graph,
            parent_emitter,
            &registry,
            ModifierGroup::Update,
            TypeId::of::<bevy_hanabi::EmitSpawnEventModifier>(),
            update_len,
        )
        .expect("emitter exists");

        // Simulate another queued edit consuming the id the UI observed before
        // this compound edit is applied.
        let parent_source = graph_edit::create_source(
            &mut effect_graph,
            SourceKind::CpuSpawner {
                settings: SpawnerSettings::default(),
            },
        );
        assert!(
            graph_edit::set_source_link(&mut effect_graph, parent_source, parent_emitter)
                .is_empty()
        );

        let inverse = apply_to_graph(
            &mut effect_graph,
            &registry,
            &EditKind::CreateGpuEmitter {
                event_node: Some(event_node),
            },
            EditDirection::Fresh,
        )
        .expect("create GPU source, emitter, and links");
        let EditKind::Batch(ref inverses) = inverse else {
            panic!("expected compound inverse");
        };
        let [
            EditKind::DeleteSource { source },
            EditKind::DeleteEmitter { emitter },
        ] = inverses.as_slice()
        else {
            panic!("expected source/emitter deletion inverse");
        };
        let (source, child) = (*source, *emitter);
        assert!(effect_graph.source(source).is_some());
        let child_graph = effect_graph.emitter(child).expect("child emitter created");
        assert_eq!(
            child_graph
                .stack(ModifierGroup::Init)
                .expect("child Init stack")
                .members
                .len(),
            0,
            "the editor must not make an authored modifier structurally mandatory"
        );
        assert_eq!(effect_graph.emitter_for_source(source), Some(child));
        assert_eq!(
            effect_graph.event_links,
            vec![EventLink {
                node: event_node,
                target: source,
            }]
        );

        let errors = match crate::effect_graph::bake::bake_effect(&effect_graph, &registry) {
            Ok(_) => panic!("an empty child must not reach Hanabi shader generation"),
            Err(errors) => errors,
        };
        assert!(
            errors.iter().any(|error| {
                error.emitter == child && error.error.message.contains("empty particle layout")
            }),
            "{errors:?}"
        );

        graph_edit::add_modifier_from_template(
            &mut effect_graph,
            child,
            &registry,
            ModifierGroup::Init,
            TypeId::of::<bevy_hanabi::SetPositionSphereModifier>(),
            0,
        )
        .expect("add first authored child attribute");
        let baked = crate::effect_graph::bake::bake_effect(&effect_graph, &registry)
            .expect("bake complete authored hierarchy");
        let parent = baked.emitter(*emitter).expect("baked parent");
        let child_bake = baked.emitter(child).expect("baked child");
        let parent_shaders = bevy_hanabi::EffectShaderSources::generate(&parent.asset, None, 1)
            .expect("generate parent shaders");
        assert!(
            parent_shaders
                .update_shader_source
                .contains("fn append_spawn_events_0")
        );
        bevy_hanabi::EffectShaderSources::generate(
            &child_bake.asset,
            Some(&parent.asset.particle_layout()),
            0,
        )
        .expect("generate child shaders");

        apply_to_graph(&mut effect_graph, &registry, &inverse, EditDirection::Undo)
            .expect("undo compound creation");
        assert!(effect_graph.source(source).is_none());
        assert!(effect_graph.emitter(child).is_none());
    }
}
