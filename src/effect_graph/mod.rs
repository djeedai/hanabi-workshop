//! Edit-only effect graph — the canonical model the editor edits and saves.
//!
//! `EffectAsset` is a *baked runtime* container, not an editing structure: its
//! `Module` is an arena of `ExprHandle` indices that shift on edit, almost all
//! of it is private behind reflection, and it can only ever hold a *valid*
//! emitter. The [`EffectGraph`] defined here is a stable-identity,
//! serializable, partially-valid authored effect containing one or more
//! [`EmitterGraph`] pipelines and their topology. Each emitter's `EffectAsset`
//! is a derived bake output (see the [`bake`] module) used only for live
//! preview and runtime.
//!
//! This module is a *consumer* of the [`node_graph`]
//! widget, never the reverse — the widget stays free of any `bevy_hanabi`
//! import.
//!
//! ## Identity
//!
//! Node and stack ids are one-based [`NonZeroU32`], minted from the monotonic
//! [`EffectGraph::next_id`] counter and **never reused**. Links and the on-disk
//! layout key on these ids, so they stay valid across inserts, removals, undo,
//! and reload — unlike `ExprHandle`, whose arena index is positional.
//!
//! ## Wiring
//!
//! A node's input ports are *not* stored explicitly; they are derived from its
//! payload (a modifier's reflected `ExprHandle` fields, or an expression node's
//! operands). Each input carries an inline **default value** used when nothing
//! is linked to it; a [`GraphLink`] simply overrides that default. This unifies
//! "inlined literal" with "unconnected pin" and is what lets the graph hold
//! partial, mid-edit states that an `EffectAsset` cannot represent.
//!
//! [`node_graph`]: hanabi_node_graph
//! [`NonZeroU32`]: std::num::NonZeroU32

#![allow(dead_code)]

#[cfg(test)]
pub use hanabi_effect_graph::demo;
pub use hanabi_effect_graph::{bake, model, schema, validation};

pub mod edit;
pub mod view;

#[allow(unused_imports)]
pub use model::*;
#[allow(unused_imports)]
pub use schema::*;
