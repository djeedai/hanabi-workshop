//! Canonical / proxy `EffectAsset` split for Phase 5b live editing.
//!
//! ## Architecture (see plan.md §9.3)
//!
//! The user's `EffectAsset` (the "canonical" asset, stored in
//! [`DocumentContent::effect`]) is the source-of-truth and is what we
//! save to disk. The asset actually instantiated as a
//! [`bevy_hanabi::ParticleEffect`] in the viewport is a derived
//! "proxy" asset, held in [`ProxyEffect::handle`].
//!
//! The proxy is identical to the canonical *except* that every
//! reachable [`bevy_hanabi::Expr::Literal`] of a CPU-uploadable type
//! is replaced with a [`bevy_hanabi::Expr::Property`] referencing a
//! synthetic property named `__hwk_tweak__<N>`. This lets the editor
//! upload value tweaks via [`bevy_hanabi::EffectProperties::
//! set_if_changed`] without forcing a shader recompile — at the cost
//! of one recompile per *structural* change (add/remove/reorder
//! modifier, add/remove real user-property, document load).
//!
//! The mutation trick: `Module::get_mut(handle)` lets us overwrite an
//! existing arena slot, so the proxy's `ExprHandle` ids stay identical
//! to the canonical's. Modifier fields holding `ExprHandle` need no
//! rewriting; they automatically resolve to the new `Expr::Property`.
//!
//! Because `EffectAsset` doesn't expose a `module_mut()` accessor in
//! bevy_hanabi 0.18, we reach `&mut Module` via bevy_reflect on the
//! private `module` field (see [`module_mut`]).

use bevy::platform::collections::HashSet;
use bevy::prelude::*;
use bevy::reflect::{PartialReflect, ReflectMut, ReflectRef};
use bevy_hanabi::graph::expr::PropertyExpr;
use bevy_hanabi::{EffectAsset, Expr, ExprHandle, LiteralExpr, Module, Value};

use crate::document::DocumentContent;
use crate::edits::{EditApplied, EditSystems};

/// Reserved name prefix for synthetic literal-tweaker properties. User
/// `Property` names must not start with this; we validate on load and
/// on user-driven property add (in `5b-user-properties`).
pub const TWEAK_PROP_PREFIX: &str = "hwk_tweak_";

/// Stable identity of a canonical literal that has been promoted to a
/// synthetic `Property` in the proxy module. Keyed against the
/// canonical asset's `Module` so the same source literal resolves
/// across proxy rebuilds.
///
/// Always empty in the current implementation; the data shape is in
/// place so the upcoming `5b-cat3-literal-edit` work can wire against
/// the final API without churning this file again.
#[derive(Debug, Clone)]
pub struct LiteralBinding {
    /// Handle into the canonical `Module`'s expression arena. Points
    /// at the `Expr::Literal(_)` that has been promoted in the proxy.
    pub canonical_expr: ExprHandle,
    /// Name of the synthetic `Property` in the proxy module. Always
    /// begins with [`TWEAK_PROP_PREFIX`].
    pub proxy_prop_name: String,
    /// Human-readable provenance label for the UI, e.g.
    /// `"init / SetPositionSphereModifier.radius"`. May be `"???"`
    /// if reflection didn't yield a clean path (e.g. the literal is
    /// only reachable through a tuple/array operand).
    /// Kept for diagnostics / future "promoted-literals" listing.
    #[allow(dead_code)]
    pub label: String,
    /// Cached last value uploaded to this property — used to demote
    /// the proxy property back to a canonical literal if the user
    /// later removes the binding, and as a fallback during proxy
    /// rebuilds. Read by future `5b-user-properties` work.
    #[allow(dead_code)]
    pub last_value: Value,
}

/// Per-document proxy data. Inserted by [`ensure_proxy`] once the
/// canonical asset has loaded.
///
/// `handle` is what the viewport's `ParticleEffect` references — the
/// canonical handle stays in [`DocumentContent::effect`] and is never
/// instantiated. See module docs.
#[derive(Component, Debug, Clone)]
pub struct ProxyEffect {
    /// Handle to the proxy `EffectAsset`.
    pub handle: Handle<EffectAsset>,
    /// Bindings populated by [`build_proxy`]. Empty in the current
    /// stub; populated in `5b-cat3-literal-edit`.
    pub bindings: Vec<LiteralBinding>,
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

/// For every document missing a [`ProxyEffect`], try to build one from
/// its canonical asset. Skips documents whose canonical asset isn't
/// loaded yet (we re-try every frame until it is). Idempotent.
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
        let (proxy_asset, bindings) = build_proxy(canonical);
        let handle = assets.add(proxy_asset);
        commands
            .entity(entity)
            .insert(ProxyEffect { handle, bindings });
    }
}

/// After [`crate::edits::apply_edits`] runs, re-sync canonical → proxy
/// for every document touched this frame. Dedup'd: one sync per
/// document even if multiple edits landed in the same frame.
///
/// `SetLiteralValue` edits are *not* meant to land here — they bypass
/// proxy-rebuild entirely by uploading via [`bevy_hanabi::
/// EffectProperties::set_if_changed`] inside the edit's apply arm.
/// But we still re-clone after any non-literal edit (effect name,
/// spawner, simulation space, etc.) and re-run the promotion pass so
/// the bindings stay correct.
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
        let (new_proxy_asset, new_bindings) = build_proxy(canonical);
        if let Some(proxy_asset) = assets.get_mut(&proxy.handle) {
            *proxy_asset = new_proxy_asset;
            proxy.bindings = new_bindings;
        }
    }
}

/// Build a proxy `EffectAsset` from the canonical one, promoting every
/// reachable `Expr::Literal` of CPU-uploadable type to a synthetic
/// `Property`. Bindings let callers map a canonical `ExprHandle` to
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
pub fn build_proxy(canonical: &EffectAsset) -> (EffectAsset, Vec<LiteralBinding>) {
    use bevy::platform::collections::HashMap;

    let mut proxy = canonical.clone();

    // (2) Walk every modifier and remember the *first* labelled path
    // we found to each ExprHandle. Keyed by handle so later visits to
    // the same shared sub-expression don't clobber the original label.
    let mut labels: HashMap<ExprHandle, String> = HashMap::default();
    for (phase, m) in iter_modifiers_labeled(&proxy) {
        let short = m.as_partial_reflect().reflect_short_type_path().to_string();
        let base = format!("{phase} / {short}");
        collect_handles_labeled(m.as_partial_reflect(), &base, &mut labels);
    }
    // (3) Transitively expand through operand expressions.
    expand_via_module_labeled(&mut labels, proxy.module());

    // Snapshot (handle, value, label) — stable order by handle index.
    let mut to_promote: Vec<(ExprHandle, Value, String)> = labels
        .iter()
        .filter_map(|(h, label)| {
            let Some(Expr::Literal(lit)) = proxy.module().get(*h) else {
                return None;
            };
            literal_value(lit).map(|v| (*h, v, label.clone()))
        })
        .collect();
    to_promote.sort_by_key(|(h, _, _)| *h);

    let mut bindings: Vec<LiteralBinding> = Vec::with_capacity(to_promote.len());
    {
        let Some(module) = module_mut(&mut proxy) else {
            warn!(
                "build_proxy: could not reach &mut Module via reflect; \
                   live tweaking disabled for this asset"
            );
            return (proxy, Vec::new());
        };
        for (h, value, label) in to_promote {
            let prop_name = format!("{TWEAK_PROP_PREFIX}{}", bindings.len());
            let prop_handle = module.add_property(prop_name.clone(), value);
            if let Some(slot) = module.get_mut(h) {
                *slot = Expr::Property(PropertyExpr::new(prop_handle));
            }
            bindings.push(LiteralBinding {
                canonical_expr: h,
                proxy_prop_name: prop_name,
                label,
                last_value: value,
            });
        }
    }

    (proxy, bindings)
}

/// Iterator yielding `(phase_label, &dyn Modifier)` for every modifier
/// in init / update / render order. Render modifiers are upcast via
/// `as_modifier()`.
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
pub fn find_binding<'a>(
    bindings: &'a [LiteralBinding],
    canonical_expr: ExprHandle,
) -> Option<&'a LiteralBinding> {
    bindings.iter().find(|b| b.canonical_expr == canonical_expr)
}

/// Read the inner `Value` of a `LiteralExpr` via reflection (the field
/// is private in bevy_hanabi 0.18 but exposed by its `Reflect` derive).
pub fn literal_value(lit: &LiteralExpr) -> Option<Value> {
    match lit.reflect_ref() {
        ReflectRef::Struct(s) => s
            .field("value")
            .and_then(|f| f.try_downcast_ref::<Value>())
            .copied(),
        _ => None,
    }
}

/// Reach `&mut Module` on an `EffectAsset` via reflection. The
/// `module` field is private in bevy_hanabi 0.18, with no public
/// `module_mut()` accessor; the `Reflect` derive exposes it.
pub fn module_mut(asset: &mut EffectAsset) -> Option<&mut Module> {
    match asset.reflect_mut() {
        ReflectMut::Struct(s) => s
            .field_mut("module")
            .and_then(|f| f.try_downcast_mut::<Module>()),
        _ => None,
    }
}

/// Reflect-walk: record `(ExprHandle → label)` for every handle we
/// encounter, with `label` built by appending struct field names /
/// tuple indices to `base_path`. We keep the *first* label found for
/// any handle so that operand expansion (which uses a longer derived
/// path) doesn't overwrite a direct modifier-field path.
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

/// Transitively expand by reflecting through each referenced expression
/// (covers `Unary` / `Binary` / `Ternary` / `Cast` operands). New
/// handles found this way get a derived label like `"<parent> ← op"`.
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

/// List of `(field_name, ExprHandle)` for every direct struct field
/// of `modifier` whose type is `ExprHandle`. Field order matches the
/// modifier struct's declaration order (Reflect preserves it).
///
/// Used by the Properties panel to render per-field editors for each
/// modifier's tweakable expression slots.
pub fn modifier_expr_fields(modifier: &dyn bevy::reflect::Reflect) -> Vec<(String, ExprHandle)> {
    let mut out = Vec::new();
    if let ReflectRef::Struct(s) = modifier.reflect_ref() {
        for i in 0..s.field_len() {
            let Some(field) = s.field_at(i) else { continue };
            if let Some(handle) = field.try_downcast_ref::<ExprHandle>() {
                let name = s.name_at(i).unwrap_or("?").to_string();
                out.push((name, *handle));
            }
        }
    }
    out
}

pub fn property_handle_of(pe: &PropertyExpr) -> Option<bevy_hanabi::graph::expr::PropertyHandle> {
    match pe.reflect_ref() {
        ReflectRef::Struct(s) => s
            .field("property")
            .and_then(|f| f.try_downcast_ref::<bevy_hanabi::graph::expr::PropertyHandle>())
            .copied(),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// User-property mutation helpers.
//
// `Module::properties: Vec<Property>` and `Property::{name, default_value}`
// are all private in bevy_hanabi 0.18 but exposed by their `Reflect`
// derives. These helpers reach in via `bevy_reflect` so the editor can
// rename properties, change their initial value, and remove them with
// auto-demotion of binding sites — operations the public API simply
// doesn't provide. See `hanabi_gaps.md` §3.
// ---------------------------------------------------------------------------

/// True for synthetic literal-tweak property names (created by
/// [`build_proxy`]). These never appear on the canonical module — but
/// listing helpers skip them defensively to keep the "user properties"
/// view clean.
pub fn is_tweak_prop_name(name: &str) -> bool {
    name.starts_with(TWEAK_PROP_PREFIX)
}

/// Iterate user-defined property names (skipping `hwk_tweak_*`).
/// Returns `(name, default_value)` in declaration order.
pub fn user_properties(module: &Module) -> Vec<(String, Value)> {
    module
        .properties()
        .iter()
        .filter(|p| !is_tweak_prop_name(p.name()))
        .map(|p| (p.name().to_string(), *p.default_value()))
        .collect()
}

/// Look up a property's default value by name. None if not present.
#[allow(dead_code)]
pub fn property_default(module: &Module, name: &str) -> Option<Value> {
    module
        .properties()
        .iter()
        .find(|p| p.name() == name)
        .map(|p| *p.default_value())
}

/// True if a property with this name exists.
pub fn property_exists(module: &Module, name: &str) -> bool {
    module.properties().iter().any(|p| p.name() == name)
}

/// Reach `&mut [Property]` on a `Module` via reflection. Both the
/// `properties` field on `Module` and the `name`/`default_value` fields
/// on `Property` are private with `Reflect` derive.
fn properties_list_mut(module: &mut Module) -> Option<&mut dyn bevy::reflect::List> {
    match module.reflect_mut() {
        ReflectMut::Struct(s) => match s.field_mut("properties")?.reflect_mut() {
            ReflectMut::List(l) => Some(l),
            _ => None,
        },
        _ => None,
    }
}

/// Apply a mutation closure to the `Property` reflected at `idx` in
/// the module's properties list.
fn with_property_mut<R>(
    module: &mut Module,
    idx: usize,
    f: impl FnOnce(&mut dyn bevy::reflect::Struct) -> R,
) -> Option<R> {
    let list = properties_list_mut(module)?;
    let item = list.get_mut(idx)?;
    match item.reflect_mut() {
        ReflectMut::Struct(s) => Some(f(s)),
        _ => None,
    }
}

/// Find the index of a property by name within `Module::properties`.
fn property_index(module: &Module, name: &str) -> Option<usize> {
    module.properties().iter().position(|p| p.name() == name)
}

/// Rename a property in place. Returns `true` on success. Fails if no
/// property has `old_name`, or if `new_name` is already taken by a
/// different property (the no-op rename old==new succeeds).
pub fn rename_property(module: &mut Module, old_name: &str, new_name: &str) -> bool {
    if old_name == new_name {
        return property_exists(module, old_name);
    }
    if property_exists(module, new_name) {
        return false;
    }
    let Some(idx) = property_index(module, old_name) else {
        return false;
    };
    let new_owned = new_name.to_string();
    with_property_mut(module, idx, |s| {
        if let Some(name_field) = s
            .field_mut("name")
            .and_then(|f| f.try_downcast_mut::<String>())
        {
            *name_field = new_owned;
        }
    })
    .is_some()
}

/// Set the default value of an existing property. Returns the previous
/// value on success; `None` if the property doesn't exist.
pub fn set_property_default(module: &mut Module, name: &str, new_value: Value) -> Option<Value> {
    let idx = property_index(module, name)?;
    let old = *module.properties()[idx].default_value();
    with_property_mut(module, idx, |s| {
        if let Some(v) = s
            .field_mut("default_value")
            .and_then(|f| f.try_downcast_mut::<Value>())
        {
            *v = new_value;
        }
    })?;
    Some(old)
}

/// Append a brand-new property to the module. Returns `true` on
/// success; `false` if `name` is already taken. Does NOT promote any
/// expressions to use it (callers wanting to restore demoted bindings
/// should use [`restore_property_with_promotions`]).
pub fn add_user_property(module: &mut Module, name: &str, value: Value) -> bool {
    if property_exists(module, name) {
        return false;
    }
    module.add_property(name, value);
    true
}

/// Remove a user property by name.
///
/// Before deletion, every `Expr::Property` in the module's expression
/// arena that references this property is **demoted** to
/// `Expr::Literal(default_value)`. The list of canonical `ExprHandle`s
/// demoted this way is returned so the inverse edit can re-promote
/// them.
///
/// After deletion, every remaining `Expr::Property` whose handle index
/// was greater than the removed property's index is decremented by one
/// (since `PropertyHandle` is index+1 into the properties `Vec`).
///
/// Returns `(removed_default_value, demoted_expr_handles)`. None if
/// the property doesn't exist.
pub fn remove_user_property(module: &mut Module, name: &str) -> Option<(Value, Vec<ExprHandle>)> {
    let idx = property_index(module, name)?;
    let default_value = *module.properties()[idx].default_value();
    let target_handle = module.get_property_by_name(name)?;

    // (1) Find every Expr::Property(target_handle) and demote it.
    let mut demoted: Vec<ExprHandle> = Vec::new();
    let total = expression_count(module);
    for i in 0..total {
        let h = match expr_handle_at(i) {
            Some(h) => h,
            None => continue,
        };
        let Some(expr) = module.get(h) else { continue };
        if let Expr::Property(pe) = expr {
            if let Some(ph) = property_handle_of(pe) {
                if ph == target_handle {
                    demoted.push(h);
                }
            }
        }
    }
    for h in &demoted {
        if let Some(slot) = module.get_mut(*h) {
            *slot = Expr::Literal(LiteralExpr::new(default_value));
        }
    }

    // (2) Decrement higher-indexed PropertyHandles in remaining
    // Expr::Property nodes. PropertyHandle.id is `NonZeroU32` = index+1.
    let removed_id = (idx + 1) as u32;
    for i in 0..total {
        let h = match expr_handle_at(i) {
            Some(h) => h,
            None => continue,
        };
        // Capture replacement plan without holding a borrow over get_mut.
        let new_expr = match module.get(h) {
            Some(Expr::Property(pe)) => {
                let cur = property_handle_of(pe)?;
                let cur_id = property_handle_id(cur)?;
                if cur_id > removed_id {
                    Some(Expr::Property(PropertyExpr::new(make_property_handle(
                        cur_id - 1,
                    )?)))
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some(new) = new_expr {
            if let Some(slot) = module.get_mut(h) {
                *slot = new;
            }
        }
    }

    // (3) Drop the property from the list.
    let list = properties_list_mut(module)?;
    list.remove(idx);

    Some((default_value, demoted))
}

/// Re-add a previously-removed user property at the *end* of the
/// properties list and re-promote each `repromote_exprs` slot from
/// `Expr::Literal` back to `Expr::Property(new_handle)`.
///
/// Used as the inverse of [`remove_user_property`].
///
/// Note: handle indices of *existing* user properties shift only when
/// a property is removed; appending never invalidates older handles.
/// Therefore appending here is safe (we don't need to "re-insert at
/// the original index").
pub fn restore_property_with_promotions(
    module: &mut Module,
    name: &str,
    default_value: Value,
    repromote_exprs: &[ExprHandle],
) -> bool {
    if property_exists(module, name) {
        return false;
    }
    let new_handle = module.add_property(name, default_value);
    for h in repromote_exprs {
        if let Some(slot) = module.get_mut(*h) {
            *slot = Expr::Property(PropertyExpr::new(new_handle));
        }
    }
    true
}

/// Total number of expressions in the module's arena, via reflection
/// on the private `expressions: Vec<Expr>` field. `ExprHandle` is
/// `NonZeroU32` = index+1.
pub fn expression_count(module: &Module) -> usize {
    match module.reflect_ref() {
        ReflectRef::Struct(s) => match s.field("expressions").map(|f| f.reflect_ref()) {
            Some(ReflectRef::List(l)) => l.len(),
            _ => 0,
        },
        _ => 0,
    }
}

/// Build an [`ExprHandle`] for arena slot `i` (0-based). Returns `None`
/// for `i == usize::MAX` (overflow guard) — never happens in practice.
pub fn expr_handle_at(i: usize) -> Option<ExprHandle> {
    let id = u32::try_from(i.checked_add(1)?).ok()?;
    let nz = std::num::NonZeroU32::new(id)?;
    // ExprHandle is `#[derive(Reflect)] pub struct ExprHandle { id: NonZeroU32 }`.
    // We can't call the private `new`, but we can construct via reflection
    // round-trip through the `Default`-derived `Reflect` machinery.
    construct_handle_via_reflect::<ExprHandle>(nz)
}


/// Extract the inner `NonZeroU32` of a `PropertyHandle` via reflection.
fn property_handle_id(h: bevy_hanabi::graph::expr::PropertyHandle) -> Option<u32> {
    match h.reflect_ref() {
        ReflectRef::Struct(s) => s
            .field("id")
            .and_then(|f| f.try_downcast_ref::<std::num::NonZeroU32>())
            .map(|n| n.get()),
        _ => None,
    }
}

/// Build a `PropertyHandle` from a raw 1-based id via reflection.
fn make_property_handle(id: u32) -> Option<bevy_hanabi::graph::expr::PropertyHandle> {
    let nz = std::num::NonZeroU32::new(id)?;
    construct_handle_via_reflect::<bevy_hanabi::graph::expr::PropertyHandle>(nz)
}

/// Generic helper: build a `#[derive(Reflect)] struct { id: NonZeroU32 }`
/// value from its inner id. Uses `DynamicStruct` + `FromReflect`.
fn construct_handle_via_reflect<T: bevy::reflect::FromReflect>(
    id: std::num::NonZeroU32,
) -> Option<T> {
    let mut dyn_struct = bevy::reflect::DynamicStruct::default();
    dyn_struct.insert("id", id);
    T::from_reflect(&dyn_struct)
}
