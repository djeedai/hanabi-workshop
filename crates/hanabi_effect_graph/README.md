# 🎆 Hanabi Effect Graph

The `hanabi_effect_graph` crate provides a serializable authoring effect graph and runtime bake pipeline for
[`bevy_hanabi`](https://crates.io/crates/bevy_hanabi) particle effects.

`EffectGraphAsset` is a stable-identity edit model that can be saved as `.hnb`.
It's used as a higher-level editing primitive to Hanabi's own shipping-focused `EffectAsset`.
The crate validates and bakes that graph into a runtime `EffectAsset`,
can import a baked asset back into a graph,
and integrates with Bevy's asset loader and processor APIs.

## Usage

Register the `EffectGraphPlugin` to load `.hnb` files through Bevy's `AssetServer`.
For offline conversion and runtime-loading examples, see:

- [`examples/bake.rs`](examples/bake.rs)
- [`examples/runtime_load.rs`](examples/runtime_load.rs)

The public `from_ron_bytes()` and `to_ron_string()` helpers are the canonical synchronous read/write path.
`bake::bake()` converts the loaded graph after the Hanabi modifier types have been registered.

## Features

Both rendering contexts are enabled by default:

- `2d` forwards to `bevy_hanabi/2d`.
- `3d` forwards to `bevy_hanabi/3d`.

Users needing only one context may disable default features and select it explicitly.

## Compatibility

| `hanabi_effect_graph` | `bevy` | `bevy_hanabi` | `.hnb` | Minimum Rust |
| --- | --- | --- | --- | --- |
| 0.1 | 0.19 | 0.19 | reads 1; writes 1 | 1.95 |

Every release reads every earlier released `.hnb` schema version, unless specified otherwise.
Saving upgrades an older file to the current schema.
The format version is independent from the crate's SemVer.

Versions below 1.0 follow SemVer's pre-1.0 rules: breaking public API changes increment the minor version.

This crate is developed as part of
[Hanabi Workshop](https://github.com/djeedai/hanabi-workshop).

## License

Licensed under either Apache-2.0 or MIT, at your option.

`SPDX-License-Identifier: MIT OR Apache-2.0`
