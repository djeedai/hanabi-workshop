//! Edit-time effect graph asset and its bake to a `bevy_hanabi` runtime
//! [`EffectAsset`](bevy_hanabi::EffectAsset).
//!
//! [`EffectGraphAsset`](model::EffectGraphAsset) is a stable-identity,
//! serializable graph the editor mutates directly and saves to disk. An
//! `EffectAsset` is a *derived* bake output of it (see [`bake`]), used for
//! preview and runtime.
//!
//! This crate is `egui`- and editor-agnostic so it can be used in two ways:
//!
//! - **Offline baking**: a build tool consumes [`EffectGraphAsset`] and
//!   produces an [`EffectAsset`](bevy_hanabi::EffectAsset), e.g. through an
//!   [`AssetProcessor`](bevy::asset::processor::AssetProcessor). See the
//!   `bake` example.
//! - **Runtime loading**: a game loads unbaked `EffectGraphAsset` files during
//!   development via [`EffectGraphLoader`] and bakes them in-process.

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
    EffectGraphLoader, EffectGraphLoaderError, EffectGraphPlugin, from_ron_bytes, to_ron_string,
};
pub use modifier_group::ModifierGroup;
