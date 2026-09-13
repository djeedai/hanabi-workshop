# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-13

### Added

- Added support for effect hierarchies, with CPU spawner and GPU event sources,
  explicit source/event topology, topology validation, and aggregate baking
  through `bake_effect()`.
- Added 1D and 3D texture sampling, and unfiltered texture loads (`textureLoad()`).
- The `ExprNode::Age` authoring node, with optional lifetime normalization and
  clamping. The node maps to the regular `SetAttribute(AGE)`.
- Added compatibility fixtures and migration tests for released `.hnb` schemas.

### Changed

- **Breaking:** `EffectGraph` is now the complete hierarchical effect.
  The former single-emitter fields moved to `EmitterGraph`, `EffectHeader` was
  removed, spawner settings moved to `SourceKind::CpuSpawner`, and all graph
  identities now share an effect-level allocator.
- **Breaking:** `import()` was replaced by `import_emitter()` and `import_effect()`.
  Single-emitter preview and fallback bake helpers were renamed to `bake_emitter_*`,
  and now accept derived spawner and child-channel inputs.
  `bake()` remains a strict convenience for one connected, CPU-driven emitter;
  multi-emitter callers should use `bake_effect()`.
- **Breaking:** Some public enums were extended for multi-emitter topology and texture support.
  `ExprNode::TextureSample` was renamed to `TextureSample2d`, and `TexturePlan` entries now use
  `PlannedTexture` rather than `PlannedImage`.
- `.hnb` output now uses schema version 2. Version 1 files remain readable and
  are automatically migrated to a single CPU-driven emitter when loaded.
- Updated to Bevy 0.19.1, and `bevy_hanabi` 0.20.0-dev (3848132f0048f8a4eafba4d685c00804dc7cdbc4).

## [0.1.0] - 2026-07-12

### Added

- Initial serializable emitter graph, `.hnb` version 1 loader, validation,
  baking, asset processing, and baked asset import.

[Unreleased]: https://github.com/djeedai/hanabi-workshop/compare/effect/v0.2.0...HEAD
[0.2.0]: https://github.com/djeedai/hanabi-workshop/compare/effect/v0.1.0...effect/v0.2.0
[0.1.0]: https://github.com/djeedai/hanabi-workshop/tree/effect%2Fv0.1.0
