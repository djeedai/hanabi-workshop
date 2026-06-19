//! Effect-graph validation checks.
//!
//! Home for analyses that flag suspicious-but-legal effects in the UI (e.g.
//! the per-node Graph warning badge). Currently a single check:
//! shadowed-modifier detection. More checks (orphaned nodes, unused
//! properties, missing required attributes, …) belong here as they land.
//!
//! ## Shadowed modifiers
//!
//! A modifier is *shadowed* when every particle attribute it fully
//! overwrites is also overwritten by some later modifier in the same
//! group, making its writes dead. Built on the
//! [`ModifierOverwrites`]
//! type-data callback, which gives the per-instance set of attributes a
//! modifier *fully assigns to* (distinct from `Modifier::attributes()`,
//! which mixes reads and writes in upstream bevy_hanabi).
//!
//! Only meaningful within a single group (Init / Update): each runs
//! strictly in order, with subsequent overwrites discarding any previous
//! per-particle value. The Render group is skipped — render modifiers
//! write vertex-shader variables rather than particle attributes.
//!
//! [`ModifierOverwrites`]: crate::modifier_registry::ModifierOverwrites

use std::collections::HashMap;

use bevy::reflect::TypeRegistry;
use bevy_hanabi::{Attribute, EffectAsset};

use crate::ModifierGroup;
use crate::modifier_registry::ModifierOverwrites;

/// For each shadowed modifier, the `(attribute, shadower_idx)` pairs that
/// explain why it has no effect, keyed by `(group, idx)` where `idx` is the
/// modifier's position within its group's stack. Only Init and Update groups
/// are analysed; Render is never included.
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

/// Per-instance overwrite set lookup. Falls back to empty if the modifier's
/// type isn't registered or carries no [`ModifierOverwrites`] data (e.g. a
/// read-modify-write or third-party modifier) — we conservatively skip shadow
/// analysis rather than risk a false positive.
fn overwrites_for(m: &dyn bevy::reflect::Reflect, registry: &TypeRegistry) -> Vec<Attribute> {
    let Some(reg) = registry.get(std::any::Any::type_id(m)) else {
        return Vec::new();
    };
    let Some(rm) = reg.data::<ModifierOverwrites>() else {
        return Vec::new();
    };
    (rm.overwrites)(m)
}
