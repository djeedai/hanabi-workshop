//! Editable effect graphs and their bake to runtime [`EffectAsset`]s.
//!
//! [`EffectGraphAsset`] is the stable-identity, serializable effect an artist
//! edits and saves. Its [`EffectGraph`] contains one or more self-contained
//! [`EmitterGraph`] pipelines plus the spawn sources and inter-emitter topology
//! that connect them. Each emitter bakes to a derived `EffectAsset` used for
//! preview and runtime.
//!
//! This crate can be used in two ways:
//!
//! - **Offline baking**: a build tool consumes [`EffectGraphAsset`] and
//!   produces an [`EffectAsset`] per emitter, e.g. through an
//!   [`AssetProcessor`]. See the `bake` example.
//! - **Runtime loading**: a game loads unbaked [`EffectGraphAsset`] files
//!   during development via [`EffectGraphLoader`] and bakes them in-process.
//!
//! [`EffectAsset`]: bevy_hanabi::EffectAsset
//! [`EffectGraph`]: model::EffectGraph
//! [`EffectGraphAsset`]: model::EffectGraphAsset
//! [`EmitterGraph`]: model::EmitterGraph
//! [`AssetProcessor`]: bevy::asset::processor::AssetProcessor

pub mod bake;
pub mod demo;
pub mod import;
pub mod model;
pub mod modifier_names;
pub mod modifier_ops;
pub mod modifier_registry;
pub mod processor;
pub mod schema;
pub mod validation;

mod loader;
mod modifier_group;

pub use loader::{
    EffectGraphLoader, EffectGraphLoaderError, EffectGraphPlugin, MAGIC_HEADER, from_ron_bytes,
    to_ron_string,
};
pub use modifier_group::ModifierGroup;
