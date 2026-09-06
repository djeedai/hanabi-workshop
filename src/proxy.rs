//! Canonical / proxy `EffectAsset` split for live editing.
//!
//! ## Architecture
//!
//! A document's canonical assets — one per emitter pipeline in its
//! [`hanabi_effect_graph::model::EffectGraph`], held in
//! [`crate::document::EmitterRecord::asset`] — are the source-of-truth and are
//! what we save to disk. The asset actually instantiated as a
//! [`bevy_hanabi::ParticleEffect`] in the viewport, for each emitter, is a
//! derived "proxy" asset, held per-[`EmitterId`] in [`ProxyEmitters`].
//!
//! Each proxy is identical to its canonical counterpart *except* that every
//! reachable [`bevy_hanabi::Expr::Literal`] of a CPU-uploadable type is
//! replaced with a [`bevy_hanabi::Expr::Property`] referencing a synthetic
//! property named `__hwk_tweak__<N>`. This lets the editor upload value tweaks
//! via [`bevy_hanabi::EffectProperties::set_if_changed`] without forcing a
//! shader recompile — at the cost of one recompile per *structural* change
//! (add/remove/reorder modifier, add/remove real user-property, document load).
//!
//! The mutation trick: `Module::get_mut(handle)` lets us overwrite an existing
//! arena slot, so the proxy's `ExprHandle` ids stay identical to the
//! canonical's. Modifier fields holding `ExprHandle` need no rewriting; they
//! automatically resolve to the new `Expr::Property`.
//!
//! Because `EffectAsset` doesn't expose a `module_mut()` accessor in
//! bevy_hanabi 0.18, we reach `&mut Module` via bevy_reflect on the private
//! `module` field (see [`module_mut`]).

use std::collections::HashMap as StdHashMap;

use bevy::{
    platform::collections::HashSet,
    prelude::*,
    reflect::{PartialReflect, ReflectMut, ReflectRef},
};
use bevy_egui::EguiPrimaryContextPass;
use bevy_hanabi::{
    EffectAsset, EffectProperties, Expr, ExprHandle, LiteralExpr, Module, Value,
    graph::expr::{PropertyExpr, PropertyHandle},
};
use hanabi_effect_graph::{
    bake::{LiteralSite, LiteralSites},
    model::{EmitterGraph, EmitterId, ExprNode, NodePayload, PropertyId},
};

use crate::{
    document::{DocumentContent, DocumentSceneRoot, EmitterSceneEntities},
    edits::{EditApplied, EditSystems},
    effect_graph::model::{NodeId, SharedStr},
    ui::draw_editor_ui,
};

/// Reserved name prefix for synthetic literal-tweaker properties.
///
/// User `Property` names must not start with this; we validate on load and on
/// user-driven property add.
pub const TWEAK_PROP_PREFIX: &str = "hwk_tweak_";

/// Stable identity of a canonical literal promoted to a proxy `Property`.
///
/// Keyed against the canonical asset's `Module` so the same source literal
/// resolves across proxy rebuilds.
#[derive(Debug, Clone)]
pub struct LiteralBinding {
    /// Handle into the canonical `Module`'s expression arena. Points
    /// at the `Expr::Literal(_)` that has been promoted in the proxy.
    pub canonical_expr: ExprHandle,
    /// Name of the synthetic `Property` in the proxy module. A literal
    /// promoted from an unexposed user-property reference takes that
    /// property's name (so the proxy shader matches the exposed bake);
    /// every other promoted literal takes a [`TWEAK_PROP_PREFIX`] name.
    pub proxy_prop_name: String,
}

/// Live-editing state for one emitter pipeline's proxy `EffectAsset`.
///
/// One entry per [`EmitterId`] lives in a document's [`ProxyEmitters`] map,
/// built by [`ensure_proxy`] once that emitter's canonical asset has loaded.
///
/// `handle` is what the viewport's `ParticleEffect` for this emitter
/// references — the canonical handle stays in the document's
/// [`crate::document::EmitterRecord`] and is never instantiated. See module
/// docs.
#[derive(Debug, Clone)]
pub struct ProxyInstance {
    /// Handle to the proxy `EffectAsset`.
    pub handle: Handle<EffectAsset>,
    /// Bindings produced by [`build_proxy`]: every promoted canonical
    /// literal mapped to its synthetic proxy property.
    pub bindings: Vec<LiteralBinding>,
    /// Live-tweak routing table: maps each graph
    /// [`LiteralSite`] to the proxy property name that drives it on the
    /// GPU. Composed from this emitter's bake provenance crossed with
    /// `bindings`. A `SetInputDefault`/`SetLiteralValue` edit whose site
    /// is present here can upload via `EffectProperties` without a
    /// rebake/recompile; sites absent here (e.g. render-reachable
    /// literals) fall back to the rebake path.
    pub tweak_props: StdHashMap<LiteralSite, String>,
    /// Latest value uploaded for each proxy property since the last
    /// structural rebake, keyed by property name.
    ///
    /// A live tweak (the fast path) updates the canonical graph and the
    /// running [`bevy_hanabi::EffectProperties`], but *not* the proxy
    /// asset's property defaults — those only refresh on a structural
    /// rebake. A `Respawn` recreates the `ParticleEffect` with fresh
    /// `EffectProperties` seeded from those (now stale) defaults, which
    /// would drop the tweak. Re-seeding the respawned component from this
    /// map preserves it. Cleared on rebake, when the defaults become
    /// authoritative again.
    pub current_values: StdHashMap<String, Value>,
}

/// Per-document, per-emitter proxy data.
///
/// One [`ProxyInstance`] per emitter pipeline in the document's `EffectGraph`
/// that has finished its first proxy build. Attached to the document entity
/// alongside its [`DocumentContent`]; built incrementally by [`ensure_proxy`]
/// (each emitter can finish loading independently) and kept in sync with the
/// canonical bakes by [`sync_proxy_on_edit_applied`].
#[derive(Component, Debug, Clone, Default)]
pub struct ProxyEmitters(StdHashMap<EmitterId, ProxyInstance>);

impl ProxyEmitters {
    /// The proxy instance for `emitter`, if built.
    pub fn get(&self, emitter: EmitterId) -> Option<&ProxyInstance> {
        self.0.get(&emitter)
    }
    pub(crate) fn get_mut(&mut self, emitter: EmitterId) -> Option<&mut ProxyInstance> {
        self.0.get_mut(&emitter)
    }
    /// Whether `emitter` already has a built proxy instance.
    pub fn contains(&self, emitter: EmitterId) -> bool {
        self.0.contains_key(&emitter)
    }
    pub(crate) fn insert(&mut self, emitter: EmitterId, instance: ProxyInstance) {
        self.0.insert(emitter, instance);
    }
    /// Drop every entry whose emitter id is not in `live`.
    pub(crate) fn retain_emitters(&mut self, live: &HashSet<EmitterId>) {
        self.0.retain(|id, _| live.contains(id));
    }
}

/// A transient value shown while a continuous editor gesture is active.
///
/// Preview edits update the running proxy and contextual gizmos without
/// mutating the canonical graph or producing undo history.
#[derive(Message, Debug, Clone)]
pub struct LiveValueEdit {
    pub doc: Entity,
    /// Emitter pipeline `site` belongs to. Node ids are unique across the
    /// whole document, so this is technically re-derivable from `site` via
    /// `EffectGraph::emitter_owning_node`, but carrying it explicitly keeps
    /// every downstream lookup (proxy routing, scene-entity resolution) an
    /// O(1) map lookup instead of a graph walk.
    pub emitter: EmitterId,
    pub site: LiteralSite,
    pub value: Value,
}

impl LiveValueEdit {
    pub fn input(
        doc: Entity,
        emitter: EmitterId,
        node: NodeId,
        port: SharedStr,
        value: Value,
    ) -> Self {
        Self {
            doc,
            emitter,
            site: LiteralSite::Input { node, port },
            value,
        }
    }
}

#[derive(Resource, Default)]
pub(crate) struct LiveValuePreviews(StdHashMap<(Entity, EmitterId, LiteralSite), Value>);

impl LiveValuePreviews {
    pub(crate) fn for_document(
        &self,
        document: Entity,
        emitter: EmitterId,
    ) -> impl Iterator<Item = (&LiteralSite, &Value)> {
        self.0.iter().filter_map(move |((doc, eff, site), value)| {
            (*doc == document && *eff == emitter).then_some((site, value))
        })
    }
}

pub struct ProxyPlugin;

impl Plugin for ProxyPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<LiveValueEdit>()
            .init_resource::<LiveValuePreviews>()
            .add_systems(
                Update,
                (ensure_proxy, sync_proxy_on_edit_applied.after(EditSystems)),
            )
            .add_systems(
                EguiPrimaryContextPass,
                apply_live_value_edits.after(draw_editor_ui),
            );
    }
}

pub(crate) fn apply_live_value_edits(
    mut edits: MessageReader<LiveValueEdit>,
    mut previews: ResMut<LiveValuePreviews>,
    documents: Query<&DocumentContent>,
    proxies: Query<&ProxyEmitters>,
    children: Query<&Children>,
    scene_roots: Query<&EmitterSceneEntities, With<DocumentSceneRoot>>,
    mut emitter_props: Query<&mut EffectProperties>,
) {
    let mut next = StdHashMap::new();
    for edit in edits.read() {
        next.insert((edit.doc, edit.emitter, edit.site.clone()), edit.value);
    }

    for ((doc, emitter, site), _) in previews
        .0
        .iter()
        .filter(|(key, _)| !next.contains_key(*key))
    {
        let Some(value) = documents
            .get(*doc)
            .ok()
            .and_then(|content| literal_site_value(content, *emitter, site))
        else {
            continue;
        };
        upload_live_value(
            *doc,
            *emitter,
            site,
            value,
            &proxies,
            &children,
            &scene_roots,
            &mut emitter_props,
        );
    }

    for ((doc, emitter, site), value) in &next {
        upload_live_value(
            *doc,
            *emitter,
            site,
            *value,
            &proxies,
            &children,
            &scene_roots,
            &mut emitter_props,
        );
    }
    previews.0 = next;
}

fn literal_site_value(
    content: &DocumentContent,
    emitter: EmitterId,
    site: &LiteralSite,
) -> Option<Value> {
    let graph = content.effect_graph().emitter(emitter)?;
    match site {
        LiteralSite::Input { node, port } => graph
            .node(*node)?
            .inputs
            .iter()
            .find(|input| input.name == *port)?
            .default
            .as_value(),
        LiteralSite::Node(node) => {
            let NodePayload::Expr(ExprNode::Literal(value)) = &graph.node(*node)?.payload else {
                return None;
            };
            Some(*value)
        }
    }
}

fn upload_live_value(
    doc: Entity,
    emitter: EmitterId,
    site: &LiteralSite,
    value: Value,
    proxies: &Query<&ProxyEmitters>,
    children: &Query<&Children>,
    scene_roots: &Query<&EmitterSceneEntities, With<DocumentSceneRoot>>,
    emitter_props: &mut Query<&mut EffectProperties>,
) {
    let Ok(doc_proxies) = proxies.get(doc) else {
        return;
    };
    let Some(instance) = doc_proxies.get(emitter) else {
        return;
    };
    let Some(name) = instance.tweak_props.get(site) else {
        return;
    };
    let Some(entity) = proxy_props_entity(doc, emitter, children, scene_roots) else {
        return;
    };
    if let Ok(props) = emitter_props.get_mut(entity) {
        EffectProperties::set_if_changed(props, name, value);
    }
}

/// Locate the preview particle entity carrying one emitter's live properties.
pub(crate) fn proxy_props_entity(
    doc: Entity,
    emitter: EmitterId,
    children_q: &Query<&Children>,
    scene_roots: &Query<&EmitterSceneEntities, With<DocumentSceneRoot>>,
) -> Option<Entity> {
    let doc_children = children_q.get(doc).ok()?;
    for child in doc_children.iter() {
        if let Ok(entities) = scene_roots.get(child) {
            return entities.get(emitter);
        }
    }
    None
}

/// Build a [`ProxyInstance`] for every emitter, in every document, that lacks
/// one yet.
///
/// Skips an emitter whose canonical asset isn't loaded yet (we re-try every
/// frame until it is) — this happens per-emitter rather than per-document, so
/// one slow-loading emitter doesn't hold back the rest of its document's
/// proxies. Idempotent.
pub fn ensure_proxy(
    mut commands: Commands,
    mut docs: Query<(Entity, &DocumentContent, Option<&mut ProxyEmitters>)>,
    mut assets: ResMut<Assets<EffectAsset>>,
) {
    for (entity, content, mut existing) in &mut docs {
        for emitter in content.preview_emitter_ids() {
            if existing.as_deref().is_some_and(|p| p.contains(emitter)) {
                continue;
            }
            let Some(handle) = content.emitter_asset(emitter) else {
                continue;
            };
            let Some(sites) = content.literal_sites(emitter) else {
                continue;
            };
            let Some(graph) = content.effect_graph().emitter(emitter) else {
                continue;
            };
            let Some(canonical) = assets.get(handle) else {
                continue; // still loading
            };
            let origins = property_origins(graph, sites);
            let (proxy_asset, bindings) = build_proxy(canonical, &origins);
            let tweak_props = compose_tweak_props(sites, &bindings);
            let proxy_handle = assets.add(proxy_asset);
            let instance = ProxyInstance {
                handle: proxy_handle,
                bindings,
                tweak_props,
                current_values: StdHashMap::new(),
            };
            match existing.as_deref_mut() {
                Some(proxies) => proxies.insert(emitter, instance),
                None => {
                    let mut proxies = ProxyEmitters::default();
                    proxies.insert(emitter, instance);
                    commands.entity(entity).insert(proxies);
                    // The freshly-inserted component isn't visible through
                    // `existing` until next frame's query; further emitters
                    // needing a proxy this frame are picked up next frame's
                    // `ensure_proxy` pass instead of chasing the pending
                    // command here.
                    break;
                }
            }
        }
    }
}

/// Re-sync canonical → proxy for every emitter of every document touched this
/// frame.
///
/// Runs after [`crate::edits::apply_edits`]. Dedup'd: one sync per document
/// even if multiple edits landed in the same frame, and — since
/// `EditApplied` doesn't identify which single emitter a structural edit
/// touched — every emitter in that document is rebuilt, not just one. This is
/// simpler than threading emitter attribution through every `EditKind`'s
/// apply arm, and matches the transactional, whole-document spirit of a
/// structural edit (topology edits especially can touch more than one
/// emitter's shape).
///
/// Live value-upload edits (`is_literal_edit`) don't land here: they bypass
/// proxy-rebuild entirely by uploading via
/// [`bevy_hanabi::EffectProperties::set_if_changed`] inside
/// [`crate::edits::apply_edits`]. Every other edit re-clones each affected
/// emitter's canonical asset and re-runs the promotion pass so the bindings
/// and tweak-prop routing stay correct. Also prunes proxy entries for
/// emitters no longer present in the document (removed by a topology edit).
pub fn sync_proxy_on_edit_applied(
    mut applied: MessageReader<EditApplied>,
    mut docs: Query<(&DocumentContent, &mut ProxyEmitters)>,
    mut assets: ResMut<Assets<EffectAsset>>,
) {
    let mut seen: HashSet<Entity> = HashSet::default();
    for ev in applied.read() {
        if ev.is_literal_edit {
            // Pure value tweak — proxies unchanged in shape, value
            // already uploaded via EffectProperties. No-op here.
            continue;
        }
        if !seen.insert(ev.doc) {
            continue;
        }
        let Ok((content, mut proxies)) = docs.get_mut(ev.doc) else {
            continue;
        };

        let live: HashSet<EmitterId> = content.preview_emitter_ids().collect();
        proxies.retain_emitters(&live);

        for emitter in content.preview_emitter_ids() {
            let Some(handle) = content.emitter_asset(emitter) else {
                continue;
            };
            let Some(sites) = content.literal_sites(emitter) else {
                continue;
            };
            let Some(graph) = content.effect_graph().emitter(emitter) else {
                continue;
            };
            let Some(canonical) = assets.get(handle) else {
                continue;
            };
            let origins = property_origins(graph, sites);
            let (new_proxy_asset, new_bindings) = build_proxy(canonical, &origins);
            let new_tweak_props = compose_tweak_props(sites, &new_bindings);
            let Some(proxy_handle) = proxies.get(emitter).map(|i| i.handle.clone()) else {
                // Not built yet (still loading) — `ensure_proxy` will pick it
                // up once its canonical asset is ready.
                continue;
            };
            if let Some(mut proxy_asset) = assets.get_mut(&proxy_handle) {
                *proxy_asset = new_proxy_asset;
            }
            if let Some(instance) = proxies.get_mut(emitter) {
                instance.bindings = new_bindings;
                instance.tweak_props = new_tweak_props;
                // The rebaked asset's property defaults now mirror the
                // canonical literals, so prior live tweaks are baked in;
                // drop the overrides.
                instance.current_values.clear();
            }
        }
    }
}

/// Build the live-tweak routing table (`site → property name`).
///
/// Crosses the document's bake provenance (`site → canonical ExprHandle`) with
/// the proxy `bindings` (`canonical ExprHandle → property name`). Sites whose
/// literal wasn't promoted (e.g. render-reachable, or a non-promotable type)
/// are absent.
fn compose_tweak_props(
    sites: &LiteralSites,
    bindings: &[LiteralBinding],
) -> StdHashMap<LiteralSite, String> {
    let mut out = StdHashMap::with_capacity(sites.len());
    for (site, handle) in sites {
        if let Some(binding) = find_binding(bindings, *handle) {
            out.insert(site.clone(), binding.proxy_prop_name.clone());
        }
    }
    out
}

/// Origin of a baked literal that came from an unexposed property reference.
///
/// Lets [`build_proxy`] name the promoted proxy property after the user
/// property (deduplicating multiple references to one shared property), so the
/// live preview shader is identical whether the property is exposed or not.
#[derive(Debug, Clone)]
pub struct PropertyOrigin {
    /// Stable id of the source property; the dedup key.
    pub id: PropertyId,
    /// Display name of the source property; used verbatim as the proxy
    /// property name (matching how an exposed property bakes).
    pub name: String,
}

/// Map one emitter's canonical literals baked from an unexposed property to
/// their origin.
///
/// Crosses that emitter's literal provenance (`site → canonical ExprHandle`)
/// with its graph: a [`LiteralSite::Node`] whose graph node is an unexposed
/// [`ExprNode::Property`] reference yields its property's id and name. Exposed
/// properties (already real `Module` properties) and non-property literals are
/// absent.
///
/// [`LiteralSite::Node`]: hanabi_effect_graph::bake::LiteralSite::Node
/// [`ExprNode::Property`]: hanabi_effect_graph::model::ExprNode::Property
fn property_origins(
    graph: &EmitterGraph,
    sites: &LiteralSites,
) -> StdHashMap<ExprHandle, PropertyOrigin> {
    let mut out = StdHashMap::new();
    for (site, handle) in sites {
        let LiteralSite::Node(node_id) = site else {
            continue;
        };
        let Some(node) = graph.node(*node_id) else {
            continue;
        };
        let NodePayload::Expr(ExprNode::Property(pid)) = &node.payload else {
            continue;
        };
        let Some(def) = graph.properties.iter().find(|p| p.id == *pid) else {
            continue;
        };
        if def.exposed {
            continue;
        }
        out.insert(
            *handle,
            PropertyOrigin {
                id: def.id,
                name: def.name.to_string(),
            },
        );
    }
    out
}

/// Build a proxy `EffectAsset` from the canonical one.
///
/// Promotes every reachable `Expr::Literal` of CPU-uploadable type to a
/// synthetic `Property`. Bindings let callers map a canonical `ExprHandle` to
/// the proxy's property name for live uploads.
///
/// Algorithm:
/// 1. Deep-clone the canonical asset (preserves handle ids).
/// 2. Reflect-walk every modifier, collecting every `ExprHandle` they reference
///    directly.
/// 3. Transitively expand by walking each referenced expression with Reflect —
///    picks up operands of `Unary` / `Binary` / `Ternary` / `Cast` /
///    `TextureSample`.
/// 4. For every reachable handle whose `Expr` is `Literal(_)` of a promotable
///    value type, add a synthetic property to the proxy module and overwrite
///    the arena slot with `Expr::Property(...)`.
///
/// A literal carrying a [`PropertyOrigin`] (an unexposed property reference) is
/// promoted to a single property named after that user property: all references
/// to the same property id share one proxy property, mirroring the exposed bake
/// so toggling `exposed` leaves the live shader unchanged. Every other literal
/// gets a synthetic [`TWEAK_PROP_PREFIX`] name.
pub fn build_proxy(
    canonical: &EffectAsset,
    property_origins: &StdHashMap<ExprHandle, PropertyOrigin>,
) -> (EffectAsset, Vec<LiteralBinding>) {
    use bevy::platform::collections::HashMap;

    let mut proxy = canonical.clone();

    // (2) Walk every modifier and remember the *first* labelled path we found
    // to each ExprHandle. Keyed by handle so later visits to the same shared
    // sub-expression don't clobber the original label.
    let mut labels: HashMap<ExprHandle, String> = HashMap::default();
    for (phase, m) in iter_modifiers_labeled(&proxy) {
        let short = m.as_partial_reflect().reflect_short_type_path().to_string();
        let base = format!("{phase} / {short}");
        collect_handles_labeled(m.as_partial_reflect(), &base, &mut labels);
    }
    // (3) Transitively expand through operand expressions.
    expand_via_module_labeled(&mut labels, proxy.module());

    // Snapshot (handle, value) for promotion — stable order by handle index.
    // The `labels` map's values are unused now; only its key set (the
    // init/update-reachable handles) drives promotion.
    let mut to_promote: Vec<(ExprHandle, Value)> = labels
        .iter()
        .filter_map(|(h, _)| {
            let Some(Expr::Literal(lit)) = proxy.module().get(*h) else {
                return None;
            };
            literal_value(lit).map(|v| (*h, v))
        })
        .collect();
    to_promote.sort_by_key(|(h, _)| *h);

    let mut bindings: Vec<LiteralBinding> = Vec::with_capacity(to_promote.len());
    {
        let Some(module) = module_mut(&mut proxy) else {
            warn!(
                "build_proxy: could not reach &mut Module via reflect; \
                   live tweaking disabled for this asset"
            );
            return (proxy, Vec::new());
        };
        // Property names already live in the module (exposed user properties);
        // promoted names must not collide with them or each other.
        let mut used_names: HashSet<String> = module
            .properties()
            .iter()
            .map(|p| p.name().to_string())
            .collect();
        // One promoted property per source property id, so multiple references
        // to an unexposed property share a single proxy property.
        let mut promoted: HashMap<PropertyId, (PropertyHandle, String)> = HashMap::default();
        for (h, value) in to_promote {
            // Reuse the shared property for a repeat reference to the same
            // unexposed source property.
            if let Some(origin) = property_origins.get(&h)
                && let Some((handle, name)) = promoted.get(&origin.id)
            {
                if let Some(slot) = module.get_mut(h) {
                    *slot = Expr::Property(PropertyExpr::new(*handle));
                }
                bindings.push(LiteralBinding {
                    canonical_expr: h,
                    proxy_prop_name: name.clone(),
                });
                continue;
            }

            let prop_name = match property_origins.get(&h) {
                Some(origin) => unique_name(&origin.name, &mut used_names),
                None => {
                    let name = format!("{TWEAK_PROP_PREFIX}{}", bindings.len());
                    used_names.insert(name.clone());
                    name
                }
            };
            let prop_handle = module.add_property(prop_name.clone(), value);
            if let Some(origin) = property_origins.get(&h) {
                promoted.insert(origin.id, (prop_handle, prop_name.clone()));
            }
            if let Some(slot) = module.get_mut(h) {
                *slot = Expr::Property(PropertyExpr::new(prop_handle));
            }
            bindings.push(LiteralBinding {
                canonical_expr: h,
                proxy_prop_name: prop_name,
            });
        }
    }

    (proxy, bindings)
}

/// Pick `base` if free, else the first `base_N` (N≥2) not yet in `used`.
///
/// Records the chosen name in `used` so subsequent calls stay distinct.
fn unique_name(base: &str, used: &mut HashSet<String>) -> String {
    if used.insert(base.to_string()) {
        return base.to_string();
    }
    let mut n = 2;
    loop {
        let candidate = format!("{base}_{n}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
        n += 1;
    }
}

/// Iterator over `(phase_label, &dyn Modifier)` in init/update/render order.
///
/// Render modifiers are upcast via `as_modifier()`.
fn iter_modifiers_labeled(
    asset: &EffectAsset,
) -> impl Iterator<Item = (&'static str, &dyn bevy_hanabi::Modifier)> {
    asset
        .init_modifiers()
        .map(|m| ("init", m))
        .chain(asset.update_modifiers().map(|m| ("update", m)))
        .chain(
            asset
                .render_modifiers()
                .map(|m| ("render", m.as_modifier())),
        )
}

/// Locate the `LiteralBinding` for a given canonical `ExprHandle`.
pub fn find_binding(
    bindings: &[LiteralBinding],
    canonical_expr: ExprHandle,
) -> Option<&LiteralBinding> {
    bindings.iter().find(|b| b.canonical_expr == canonical_expr)
}

/// Read the inner `Value` of a `LiteralExpr` via reflection.
///
/// The field is private in bevy_hanabi 0.18 but exposed by its `Reflect`
/// derive.
pub fn literal_value(lit: &LiteralExpr) -> Option<Value> {
    match lit.reflect_ref() {
        ReflectRef::Struct(s) => s
            .field("value")
            .and_then(|f| f.try_downcast_ref::<Value>())
            .copied(),
        _ => None,
    }
}

/// Reach `&mut Module` on an `EffectAsset` via reflection.
///
/// The `module` field is private in bevy_hanabi 0.18, with no public
/// `module_mut()` accessor; the `Reflect` derive exposes it.
pub fn module_mut(asset: &mut EffectAsset) -> Option<&mut Module> {
    match asset.reflect_mut() {
        ReflectMut::Struct(s) => s
            .field_mut("module")
            .and_then(|f| f.try_downcast_mut::<Module>()),
        _ => None,
    }
}

/// Reflect-walk: record `(ExprHandle → label)` for every handle encountered.
///
/// `label` is built by appending struct field names / tuple indices to
/// `base_path`. We keep the *first* label found for any handle so that operand
/// expansion (which uses a longer derived path) doesn't overwrite a direct
/// modifier-field path.
fn collect_handles_labeled(
    value: &dyn PartialReflect,
    base_path: &str,
    out: &mut bevy::platform::collections::HashMap<ExprHandle, String>,
) {
    if let Some(handle) = value.try_downcast_ref::<ExprHandle>() {
        out.entry(*handle).or_insert_with(|| base_path.to_string());
        return;
    }
    match value.reflect_ref() {
        ReflectRef::Struct(s) => {
            for i in 0..s.field_len() {
                let name = s.name_at(i).unwrap_or("?");
                if let Some(f) = s.field_at(i) {
                    let sub = if base_path.is_empty() {
                        name.to_string()
                    } else {
                        format!("{base_path}.{name}")
                    };
                    collect_handles_labeled(f, &sub, out);
                }
            }
        }
        ReflectRef::TupleStruct(s) => {
            for i in 0..s.field_len() {
                if let Some(f) = s.field(i) {
                    let sub = format!("{base_path}.{i}");
                    collect_handles_labeled(f, &sub, out);
                }
            }
        }
        ReflectRef::Tuple(t) => {
            for i in 0..t.field_len() {
                if let Some(f) = t.field(i) {
                    let sub = format!("{base_path}.{i}");
                    collect_handles_labeled(f, &sub, out);
                }
            }
        }
        ReflectRef::Enum(e) => {
            for i in 0..e.field_len() {
                if let Some(f) = e.field_at(i) {
                    let name = e
                        .name_at(i)
                        .map(|n| n.to_string())
                        .unwrap_or_else(|| i.to_string());
                    let sub = format!("{base_path}.{name}");
                    collect_handles_labeled(f, &sub, out);
                }
            }
        }
        ReflectRef::List(l) => {
            for i in 0..l.len() {
                if let Some(f) = l.get(i) {
                    let sub = format!("{base_path}[{i}]");
                    collect_handles_labeled(f, &sub, out);
                }
            }
        }
        ReflectRef::Array(a) => {
            for i in 0..a.len() {
                if let Some(f) = a.get(i) {
                    let sub = format!("{base_path}[{i}]");
                    collect_handles_labeled(f, &sub, out);
                }
            }
        }
        _ => {}
    }
}

/// Transitively expand by reflecting through each referenced expression.
///
/// Covers `Unary` / `Binary` / `Ternary` / `Cast` operands. New handles found
/// this way get a derived label like `"<parent> ← op"`.
fn expand_via_module_labeled(
    labels: &mut bevy::platform::collections::HashMap<ExprHandle, String>,
    module: &Module,
) {
    let mut work: Vec<ExprHandle> = labels.keys().copied().collect();
    while let Some(h) = work.pop() {
        let Some(expr) = module.get(h) else { continue };
        let parent_label = labels.get(&h).cloned().unwrap_or_else(|| "?".to_string());
        let child_base = format!("{parent_label} ← op");
        let before: Vec<ExprHandle> = labels.keys().copied().collect();
        collect_handles_labeled(expr.as_partial_reflect(), &child_base, labels);
        for h2 in labels.keys().copied().collect::<Vec<_>>() {
            if !before.contains(&h2) {
                work.push(h2);
            }
        }
    }
}

/// True for synthetic literal-tweak property names.
///
/// Created by [`build_proxy`]. The reserved prefix is rejected when validating
/// user-supplied property names.
pub fn is_tweak_prop_name(name: &str) -> bool {
    name.starts_with(TWEAK_PROP_PREFIX)
}

#[cfg(test)]
mod tests {
    use bevy_hanabi::{
        Attribute, EffectAsset, ModifierContext, Module, OrientMode, OrientModifier,
        SetAttributeModifier, SpawnerSettings,
    };

    use super::*;

    /// A literal reachable through a render modifier supports live tweaking.
    ///
    /// Hanabi binds properties in the render shader as well as the simulation
    /// shaders, so proxy promotion covers all modifier contexts.
    #[test]
    fn render_reachable_literal_is_promoted() {
        let mut module = Module::default();
        let init_lit = module.lit(7.0_f32);
        let render_lit = module.lit(1.5_f32);

        let mut asset = EffectAsset::new(256, SpawnerSettings::rate(1.0.into()), module);
        asset = asset.add_modifier(
            ModifierContext::Init,
            Box::new(SetAttributeModifier::new(Attribute::LIFETIME, init_lit)),
        );
        asset = asset.add_render_modifier(Box::new(OrientModifier {
            mode: OrientMode::AlongVelocity,
            rotation: Some(render_lit),
        }));

        let (proxy, bindings) = build_proxy(&asset, &StdHashMap::new());

        // The init literal is promoted to a synthetic property.
        assert!(
            matches!(proxy.module().get(init_lit), Some(Expr::Property(_))),
            "init-reachable literal should be promoted"
        );
        assert!(
            bindings.iter().any(|b| b.canonical_expr == init_lit),
            "init literal should have a binding"
        );

        // Render properties are valid on upstream Hanabi HEAD.
        assert!(
            matches!(proxy.module().get(render_lit), Some(Expr::Property(_))),
            "render-reachable literal should be promoted"
        );
        assert!(
            bindings.iter().any(|b| b.canonical_expr == render_lit),
            "render literal should have a binding"
        );
    }

    /// Two references to one unexposed property share a single proxy property.
    ///
    /// The promoted property is named after the user property (not a synthetic
    /// tweak name), and both reference handles resolve to it — so the live
    /// shader matches what an exposed bake would produce.
    #[test]
    fn unexposed_property_refs_share_named_proxy_property() {
        use std::num::NonZeroU32;

        let mut module = Module::default();
        let ref_a = module.lit(2.0_f32);
        let ref_b = module.lit(2.0_f32);

        let mut asset = EffectAsset::new(256, SpawnerSettings::rate(1.0.into()), module);
        asset = asset.add_modifier(
            ModifierContext::Init,
            Box::new(SetAttributeModifier::new(Attribute::LIFETIME, ref_a)),
        );
        asset = asset.add_modifier(
            ModifierContext::Init,
            Box::new(SetAttributeModifier::new(Attribute::AGE, ref_b)),
        );

        let pid = PropertyId(NonZeroU32::new(7).unwrap());
        let origin = PropertyOrigin {
            id: pid,
            name: "spawn_age".to_string(),
        };
        let mut origins = StdHashMap::new();
        origins.insert(ref_a, origin.clone());
        origins.insert(ref_b, origin);

        let (proxy, bindings) = build_proxy(&asset, &origins);

        // Exactly one property added, named after the user property.
        let props: Vec<&str> = proxy
            .module()
            .properties()
            .iter()
            .map(|p| p.name())
            .collect();
        assert_eq!(
            props,
            vec!["spawn_age"],
            "one property, named after the source"
        );

        // Both references resolve to the same property handle.
        let (Some(Expr::Property(pa)), Some(Expr::Property(pb))) =
            (proxy.module().get(ref_a), proxy.module().get(ref_b))
        else {
            panic!("both references should be promoted to properties");
        };
        assert_eq!(pa.property, pb.property, "shared property handle");

        // Both bindings route to the property name, no synthetic tweak name.
        assert_eq!(bindings.len(), 2);
        assert!(
            bindings.iter().all(|b| b.proxy_prop_name == "spawn_age"),
            "both bindings use the property name"
        );
    }
}
