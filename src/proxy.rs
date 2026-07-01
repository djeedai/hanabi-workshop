//! Canonical / proxy `EffectAsset` split for live editing.
//!
//! ## Architecture
//!
//! The user's `EffectAsset` (the "canonical" asset, stored in
//! [`DocumentContent::effect`]) is the source-of-truth and is what we save to
//! disk. The asset actually instantiated as a [`bevy_hanabi::ParticleEffect`]
//! in the viewport is a derived "proxy" asset, held in [`ProxyEffect::handle`].
//!
//! The proxy is identical to the canonical *except* that every reachable
//! [`bevy_hanabi::Expr::Literal`] of a CPU-uploadable type is replaced with a
//! [`bevy_hanabi::Expr::Property`] referencing a synthetic property named
//! `__hwk_tweak__<N>`. Literals reachable from a render modifier are left
//! alone, because hanabi 0.18's render shader has no property binding and would
//! fail to compile. This lets the editor upload
//! value tweaks via [`bevy_hanabi::EffectProperties::set_if_changed`] without
//! forcing a shader recompile — at the cost of one recompile per *structural*
//! change (add/remove/reorder modifier, add/remove real user-property, document
//! load).
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
use bevy_hanabi::{
    EffectAsset, Expr, ExprHandle, LiteralExpr, Module, Value,
    graph::expr::{PropertyExpr, PropertyHandle},
};
use hanabi_effect_graph::{
    bake::{LiteralSite, LiteralSites},
    model::{ExprNode, NodePayload, PropertyId},
};

use crate::{
    document::DocumentContent,
    edits::{EditApplied, EditSystems},
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

/// Per-document proxy data.
///
/// Inserted by [`ensure_proxy`] once the canonical asset has loaded.
///
/// `handle` is what the viewport's `ParticleEffect` references — the canonical
/// handle stays in [`DocumentContent::effect`] and is never instantiated. See
/// module docs.
#[derive(Component, Debug, Clone)]
pub struct ProxyEffect {
    /// Handle to the proxy `EffectAsset`.
    pub handle: Handle<EffectAsset>,
    /// Bindings produced by [`build_proxy`]: every promoted canonical
    /// literal mapped to its synthetic proxy property.
    pub bindings: Vec<LiteralBinding>,
    /// Live-tweak routing table: maps each graph
    /// [`LiteralSite`] to the proxy property name that drives it on the
    /// GPU. Composed from the document's bake provenance crossed with
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

pub struct ProxyPlugin;

impl Plugin for ProxyPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (ensure_proxy, sync_proxy_on_edit_applied.after(EditSystems)),
        );
    }
}

/// Build a [`ProxyEffect`] for every document that lacks one.
///
/// Skips documents whose canonical asset isn't loaded yet (we re-try every
/// frame until it is). Idempotent.
pub fn ensure_proxy(
    mut commands: Commands,
    docs: Query<(Entity, &DocumentContent), Without<ProxyEffect>>,
    mut assets: ResMut<Assets<EffectAsset>>,
) {
    // Collect first to avoid a borrow conflict when we later call
    // `assets.add(...)` while still iterating the query (the query
    // doesn't touch assets, but borrowck still treats `assets` as
    // re-borrowed inside the loop).
    let pending: Vec<(Entity, AssetId<EffectAsset>)> =
        docs.iter().map(|(e, c)| (e, c.effect().id())).collect();

    for (entity, canonical_id) in pending {
        let Some(canonical) = assets.get(canonical_id) else {
            continue; // still loading
        };
        let Ok((_, content)) = docs.get(entity) else {
            continue;
        };
        let origins = property_origins(content);
        let (proxy_asset, bindings) = build_proxy(canonical, &origins);
        let tweak_props = compose_tweak_props(content.literal_sites(), &bindings);
        let handle = assets.add(proxy_asset);
        commands.entity(entity).insert(ProxyEffect {
            handle,
            bindings,
            tweak_props,
            current_values: StdHashMap::new(),
        });
    }
}

/// Re-sync canonical → proxy for every document touched this frame.
///
/// Runs after [`crate::edits::apply_edits`]. Dedup'd: one sync per document
/// even if multiple edits landed in the same frame.
///
/// Live value-upload edits (`is_literal_edit`) don't land here: they bypass
/// proxy-rebuild entirely by uploading via
/// [`bevy_hanabi::EffectProperties::set_if_changed`] inside
/// [`crate::edits::apply_edits`]. Every other edit re-clones the canonical and
/// re-runs the promotion pass so the bindings and tweak-prop routing stay
/// correct.
pub fn sync_proxy_on_edit_applied(
    mut applied: MessageReader<EditApplied>,
    mut docs: Query<(&DocumentContent, &mut ProxyEffect)>,
    mut assets: ResMut<Assets<EffectAsset>>,
) {
    let mut seen: HashSet<Entity> = HashSet::default();
    for ev in applied.read() {
        if ev.is_literal_edit {
            // Pure value tweak — proxy unchanged in shape, value
            // already uploaded via EffectProperties. No-op here.
            continue;
        }
        if !seen.insert(ev.doc) {
            continue;
        }
        let Ok((content, mut proxy)) = docs.get_mut(ev.doc) else {
            continue;
        };
        let Some(canonical) = assets.get(content.effect()) else {
            continue;
        };
        let origins = property_origins(content);
        let (new_proxy_asset, new_bindings) = build_proxy(canonical, &origins);
        let new_tweak_props = compose_tweak_props(content.literal_sites(), &new_bindings);
        if let Some(mut proxy_asset) = assets.get_mut(&proxy.handle) {
            *proxy_asset = new_proxy_asset;
            proxy.bindings = new_bindings;
            proxy.tweak_props = new_tweak_props;
            // The rebaked asset's property defaults now mirror the canonical
            // literals, so prior live tweaks are baked in; drop the overrides.
            proxy.current_values.clear();
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

/// Map every canonical literal baked from an unexposed property to its origin.
///
/// Crosses the document's literal provenance (`site → canonical ExprHandle`)
/// with the graph: a [`LiteralSite::Node`] whose graph node is an unexposed
/// [`ExprNode::Property`] reference yields its property's id and name. Exposed
/// properties (already real `Module` properties) and non-property literals are
/// absent.
///
/// [`LiteralSite::Node`]: hanabi_effect_graph::bake::LiteralSite::Node
/// [`ExprNode::Property`]: hanabi_effect_graph::model::ExprNode::Property
fn property_origins(content: &DocumentContent) -> StdHashMap<ExprHandle, PropertyOrigin> {
    let graph = content.graph();
    let mut out = StdHashMap::new();
    for (site, handle) in content.literal_sites() {
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
///    the arena slot with `Expr::Property(...)`. Handles reachable from a
///    render modifier are skipped — the render shader has no property binding,
///    so a `Property` there would emit invalid WGSL and stop the effect
///    rendering.
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

    // (2) Walk every init/update modifier and remember the *first* labelled
    // path we found to each ExprHandle. Keyed by handle so later visits to
    // the same shared sub-expression don't clobber the original label. Render
    // modifiers are deliberately excluded: hanabi 0.18's render shader has no
    // property binding, so a literal promoted to a property there generates
    // broken WGSL (see step (4)).
    let mut labels: HashMap<ExprHandle, String> = HashMap::default();
    for (phase, m) in iter_modifiers_labeled(&proxy) {
        if phase == "render" {
            continue;
        }
        let short = m.as_partial_reflect().reflect_short_type_path().to_string();
        let base = format!("{phase} / {short}");
        collect_handles_labeled(m.as_partial_reflect(), &base, &mut labels);
    }
    // (3) Transitively expand through operand expressions.
    expand_via_module_labeled(&mut labels, proxy.module());

    // (3b) Every handle reachable from a render modifier — including ones also
    // reachable from init/update — must stay a literal. The render shader can't
    // bind properties, so promoting any of these would emit a reference to a
    // non-existent `properties.*` symbol and fail to compile.
    let render_reachable = render_reachable_handles(&proxy);

    // (3c) A `TextureSample`'s image index is interpolated into a static WGSL
    // binding name (`material_texture_{i}`), so it must stay a constant literal
    // regardless of which context reaches it — promoting it to a property would
    // emit an invalid identifier. Collect those handles to exclude them.
    let texture_index_handles: HashSet<ExprHandle> = labels
        .keys()
        .chain(render_reachable.iter())
        .filter_map(|h| match proxy.module().get(*h) {
            Some(Expr::TextureSample(ts)) => Some(ts.image),
            _ => None,
        })
        .collect();

    // Snapshot (handle, value) for promotion — stable order by handle index.
    // The `labels` map's values are unused now; only its key set (the
    // init/update-reachable handles) drives promotion.
    let mut to_promote: Vec<(ExprHandle, Value)> = labels
        .iter()
        .filter(|(h, _)| !render_reachable.contains(*h))
        .filter(|(h, _)| !texture_index_handles.contains(*h))
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

/// Every `ExprHandle` reachable from any render modifier.
///
/// Directly or transitively through operand expressions. These must never be
/// promoted to properties: hanabi 0.18's render shader (`vfx_render.wgsl`)
/// carries no `{{PROPERTIES}}` binding, so a `Expr::Property` reached from the
/// render context compiles to a reference to an undefined `properties.*` symbol
/// and the effect fails to render.
fn render_reachable_handles(asset: &EffectAsset) -> HashSet<ExprHandle> {
    use bevy::platform::collections::HashMap;

    let mut reachable: HashMap<ExprHandle, String> = HashMap::default();
    for m in asset.render_modifiers() {
        collect_handles_labeled(
            m.as_modifier().as_partial_reflect(),
            "render",
            &mut reachable,
        );
    }
    expand_via_module_labeled(&mut reachable, asset.module());
    reachable.into_keys().collect()
}

/// Locate the `LiteralBinding` for a given canonical `ExprHandle`.
pub fn find_binding<'a>(
    bindings: &'a [LiteralBinding],
    canonical_expr: ExprHandle,
) -> Option<&'a LiteralBinding> {
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

    /// A literal reachable only through a render modifier must stay a literal.
    ///
    /// Hanabi's render shader has no property binding, so promoting it would
    /// emit invalid WGSL and stop the effect rendering. A literal reachable
    /// from an init/update modifier is still promoted for live tweaking.
    #[test]
    fn render_reachable_literal_is_not_promoted() {
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

        // The render-only literal stays a literal — no property reference can
        // reach the render shader.
        assert!(
            matches!(proxy.module().get(render_lit), Some(Expr::Literal(_))),
            "render-reachable literal must NOT be promoted"
        );
        assert!(
            bindings.iter().all(|b| b.canonical_expr != render_lit),
            "render literal must not have a binding"
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
