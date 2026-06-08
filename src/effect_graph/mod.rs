//! Edit-only effect graph — the canonical model the editor edits and saves.
//!
//! `EffectAsset` is a *baked runtime* container, not an editing structure: its
//! `Module` is an arena of `ExprHandle` indices that shift on edit, almost all
//! of it is private behind reflection, and it can only ever hold a *valid*
//! effect. The [`EffectGraph`] defined here is the opposite: a stable-identity,
//! serializable, partially-valid graph of nodes, ordered modifier stacks, and
//! links that the editor mutates directly. `EffectAsset` becomes a *derived*
//! bake output (see the [`bake`] module) used only for live preview and
//! runtime.
//!
//! This module is a *consumer* of the [`node_graph`](crate::ui::widgets::node_graph)
//! widget, never the reverse — the widget stays free of any `bevy_hanabi`
//! import.
//!
//! ## Identity
//!
//! Node and stack ids are one-based [`NonZeroU32`](std::num::NonZeroU32),
//! minted from a monotonic [`EffectGraph::next_id`] counter and **never
//! reused**. Links and the on-disk layout key on these ids, so they stay valid
//! across inserts, removals, undo, and reload — unlike `ExprHandle`, whose
//! arena index is positional.
//!
//! ## Wiring
//!
//! A node's input ports are *not* stored explicitly; they are derived from its
//! payload (a modifier's reflected `ExprHandle` fields, or an expression node's
//! operands). Each input carries an inline **default value** used when nothing
//! is linked to it; a [`GraphLink`] simply overrides that default. This unifies
//! "inlined literal" with "unconnected pin" and is what lets the graph hold
//! partial, mid-edit states that an `EffectAsset` cannot represent.

#![allow(dead_code)]

pub mod model;
pub mod schema;
pub mod bake;

#[allow(unused_imports)]
pub use model::*;
#[allow(unused_imports)]
pub use schema::*;
