# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-13

### Added

- Added movable, selectable `ContainerDesc` frames composed of ordered collapsible `SectionDesc` sections.
- Added container and section warnings, container close actions, section add actions, member reordering,
  collapse-all controls, and folded link anchors.
- Inputs can now accept multiple incoming links through `PortDesc::with_multiple_links()`.
- Added persistent canvas item z-order and section folding state to preserve user view.
- Added full-width inline editor rows for porst, through `PortDesc::with_inline_editor()` and
  `PortDesc::display_inline_editor()`.

### Changed

- **Breaking:** The stack API was replaced by containers and sections.
  `StackId` and `StackDesc` became `SectionId` and `SectionDesc`, while
  `ContainerId` and `ContainerDesc` identify the movable unit.
  `GraphViewer::stacks()` and `stack_links()` were replaced by `containers()`.
  Fixed links between standalone stacks, and the `StackLink` type, were removed.
- **Breaking:** Stack-oriented `GraphAction` variants were replaced by `ContainerMoved`,
  `SectionMemberMoved`, `ContainersDeleteRequested`, and `SectionAddRequested`.
- Container members now use a wider, compact edge-to-edge layout while. This gives more room to inline items,
  and reduces the number of rows, to compensate for collapsing separate nodes into sections in a same node.
  Free nodes retain their original layout (including width).

## [0.1.0] - 2026-07-12

### Added

- Initial reusable egui node graph canvas.

[Unreleased]: https://github.com/djeedai/hanabi-workshop/compare/node/v0.2.0...HEAD
[0.2.0]: https://github.com/djeedai/hanabi-workshop/compare/node/v0.1.0...node/v0.2.0
[0.1.0]: https://github.com/djeedai/hanabi-workshop/tree/node%2Fv0.1.0
