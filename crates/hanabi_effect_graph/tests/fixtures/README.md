# `.hnb` fixture archive

Each subdirectory represents one on-disk schema version and contains frozen
snapshots that the format-compatibility integration tests load.

```
hnb/
  v1/          FORMAT_VERSION 1 — first public format
    demo.hnb     broad: exercises properties, operator sub-graphs, and every
                 modifier group (3 init, 1 update, 6 render modifiers)
    minimal.hnb  compact: empty emitter, no modifiers — covers the base case
```

## Immutability contract

**Once a version directory appears in a tagged release its fixtures are
frozen.** Never edit or delete existing fixture files; they are the ground
truth for the migration ladder. Every `v<N>` file must survive
`from_ron_bytes` without modification, indefinitely.

## How to add fixtures for a future schema version

When `FORMAT_VERSION` is bumped for a breaking schema change:

1. Create `v<NEW>/` alongside the existing directories.
2. Before changing the writer, copy representative files produced by the
   previous released writer into the new directory. Temporary local generation
   code may be used, but remove it before committing so no test can overwrite
   frozen fixtures.
3. Commit the files unchanged; they are the frozen snapshot for
   `FORMAT_VERSION <NEW>`.
4. Update `expected_modifier_counts` in `tests/hnb_compat.rs` for any new
   fixture stems.

The `all_fixtures_load_migrate_and_bake` test auto-discovers every `v*/`
directory, so no other changes to the test file are required.
