//! Format-compatibility integration tests for the `.hnb` on-disk format.
//!
//! Enumerates every frozen fixture under `tests/fixtures/hnb/v*/`, loads each
//! via [`from_ron_bytes`], asserts that it migrates to the current
//! [`FORMAT_VERSION`], bakes it with the crate's modifier registry, and
//! round-trips through [`to_ron_string`] + [`from_ron_bytes`] asserting that
//! the deserialized graph is identical.
//!
//! Adding a fixture directory for a new [`FORMAT_VERSION`] requires no changes
//! here — the test auto-discovers every `v<N>/` subdirectory. Per-file
//! semantic invariants are encoded in [`expected_modifier_counts`] and
//! extended when new files are added.
//!
//! See `tests/fixtures/README.md` for the immutability contract.
//!
//! [`from_ron_bytes`]: hanabi_effect_graph::from_ron_bytes
//! [`to_ron_string`]: hanabi_effect_graph::to_ron_string
//! [`FORMAT_VERSION`]: hanabi_effect_graph::model::FORMAT_VERSION

use std::path::PathBuf;

use bevy::{asset::AssetPlugin, prelude::*};
use hanabi_effect_graph::{
    bake, from_ron_bytes, model::FORMAT_VERSION, modifier_registry::ModifierRegistryPlugin,
    to_ron_string,
};

// ── helpers ──────────────────────────────────────────────────────────────────

fn fixtures_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hnb")
}

/// Expected post-bake `(init, update, render)` modifier counts, by file stem.
///
/// `None` means "only assert the bake succeeds; skip modifier count checks."
/// Extend this table whenever a new fixture stem is added.
fn expected_modifier_counts(stem: &str) -> Option<(usize, usize, usize)> {
    match stem {
        "demo" => Some((3, 1, 6)),
        "minimal" => Some((0, 0, 0)),
        _ => None,
    }
}

/// A discovered `.hnb` fixture file.
struct Fixture {
    path: PathBuf,
    /// The version subdirectory name, e.g. `"v1"`.
    version_dir: String,
    /// The file stem, e.g. `"demo"`.
    stem: String,
}

/// Collect every `*.hnb` file from every `v*/` subdirectory of the fixture
/// root.
///
/// Directories and files within each directory are returned in sorted order so
/// the test output is deterministic.
fn collect_fixtures() -> Vec<Fixture> {
    let base = fixtures_root();
    let mut out = Vec::new();

    let entries = match std::fs::read_dir(&base) {
        Ok(rd) => rd,
        Err(e) => panic!("cannot read fixtures dir {}: {e}", base.display()),
    };

    let mut version_dirs: Vec<_> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().ok().map(|t| t.is_dir()).unwrap_or(false))
        .filter(|e| {
            e.file_name()
                .to_str()
                .map(|s| s.starts_with('v') && s[1..].parse::<u32>().is_ok())
                .unwrap_or(false)
        })
        .collect();
    version_dirs.sort_by_key(|e| e.file_name());

    for vdir in version_dirs {
        let vname = vdir.file_name().to_str().unwrap().to_string();

        let mut hnb_files: Vec<_> = std::fs::read_dir(vdir.path())
            .expect("read version dir")
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("hnb"))
            .collect();
        hnb_files.sort_by_key(|e| e.file_name());

        for file in hnb_files {
            let path = file.path();
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            out.push(Fixture {
                path,
                version_dir: vname.clone(),
                stem,
            });
        }
    }
    out
}

/// Build a Bevy `App` with the modifier registry initialised.
fn build_app() -> App {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        AssetPlugin::default(),
        ModifierRegistryPlugin,
    ));
    app
}

// ── compatibility test
// ────────────────────────────────────────────────────────

/// Load, migrate, validate, bake, and round-trip every historical `.hnb`
/// fixture.
///
/// For each fixture under `tests/fixtures/hnb/v*/`:
///
/// 1. [`from_ron_bytes`] must succeed and return an asset at
///    [`FORMAT_VERSION`].
/// 2. [`bake::bake`] must succeed with the crate's full modifier registry.
/// 3. Selected semantic invariants (modifier counts) are checked for known
///    fixture stems via [`expected_modifier_counts`].
/// 4. [`to_ron_string`] + [`from_ron_bytes`] round-trip must preserve the
///    graph.
#[test]
fn all_fixtures_load_migrate_and_bake() {
    let app = build_app();
    let registry = app.world().resource::<AppTypeRegistry>().read();

    let fixtures = collect_fixtures();
    assert!(
        !fixtures.is_empty(),
        "no .hnb fixtures found under tests/fixtures/hnb/"
    );

    for fx in &fixtures {
        let label = format!("{}/{}", fx.version_dir, fx.stem);

        // 1. Deserialize and check the migrated version stamp.
        let bytes = std::fs::read(&fx.path).unwrap_or_else(|e| panic!("{label}: read failed: {e}"));
        let asset = from_ron_bytes(&bytes)
            .unwrap_or_else(|e| panic!("{label}: from_ron_bytes failed: {e}"));

        assert_eq!(
            asset.version, FORMAT_VERSION,
            "{label}: expected version {FORMAT_VERSION} after migration, got {}",
            asset.version
        );

        // 2. Bake the migrated document with the crate's actual modifier registry.
        //    Every fixture here is a frozen v1 file, so migration always produces
        //    exactly one emitter connected to one CPU spawn source — the shape the
        //    strict single-emitter `bake` accepts.
        let baked = bake::bake(&asset.graph, &registry)
            .unwrap_or_else(|errs| panic!("{label}: bake failed: {errs:?}"));

        // 3. Semantic invariants for known fixture stems.
        if let Some((init, update, render)) = expected_modifier_counts(&fx.stem) {
            assert_eq!(
                baked.init_modifiers().count(),
                init,
                "{label}: expected {init} init modifier(s)"
            );
            assert_eq!(
                baked.update_modifiers().count(),
                update,
                "{label}: expected {update} update modifier(s)"
            );
            assert_eq!(
                baked.render_modifiers().count(),
                render,
                "{label}: expected {render} render modifier(s)"
            );
        }

        // 4. Round-trip: re-serialize with the current writer and reload; the graph
        //    must survive byte-for-byte.
        let reserialized =
            to_ron_string(&asset).unwrap_or_else(|e| panic!("{label}: to_ron_string failed: {e}"));
        let reloaded = from_ron_bytes(reserialized.as_bytes())
            .unwrap_or_else(|e| panic!("{label}: round-trip from_ron_bytes failed: {e}"));

        assert_eq!(
            reloaded.version, FORMAT_VERSION,
            "{label}: round-tripped version mismatch"
        );
        assert_eq!(
            reloaded.graph, asset.graph,
            "{label}: graph not preserved through to_ron_string + from_ron_bytes"
        );
    }
}
