//! Bevy [`AssetLoader`] and plugin for [`EffectGraphAsset`].
//!
//! Loads `.hnb` RON files into a [`EffectGraphAsset`] held by handle. Each
//! contained emitter can then be baked into a runtime [`EffectAsset`] (see
//! [`crate::bake`]) — in-process during development, or offline through an
//! [`AssetProcessor`].
//!
//! # On-disk format
//!
//! A `.hnb` file is a RON-serialized [`EffectGraphAsset`] whose first field is
//! a [`FORMAT_VERSION`] stamp, prefixed by the [`MAGIC_HEADER`] comment line
//! for content detection. [`from_ron_bytes`] and [`to_ron_string`] are the
//! single read/write funnel, shared by the [`EffectGraphLoader`] and the
//! editor's synchronous saves/loads.
//!
//! # Evolving the schema
//!
//! - **Additive, backward-compatible change** (a new optional field, a widened
//!   default): give the field `#[serde(default)]` (or `#[serde(alias = "...")]`
//!   for a rename) and *do not* bump [`FORMAT_VERSION`]. Old files parse
//!   unchanged.
//! - **Breaking change** (a removed/retyped field, changed semantics): bump
//!   [`FORMAT_VERSION`], add one guarded step to `apply_migrations` upgrading
//!   the previous version's shape to the next one, and freeze a fixture of the
//!   old format so the migration stays covered.
//!
//! The writer always stamps the current [`FORMAT_VERSION`]; the reader rejects
//! anything newer and walks the migration ladder for anything older.
//!
//! [`EffectAsset`]: bevy_hanabi::EffectAsset
//! [`AssetProcessor`]: bevy::asset::processor::AssetProcessor
//! [`FORMAT_VERSION`]: crate::model::FORMAT_VERSION

use bevy::{
    app::{App, Plugin},
    asset::{AssetApp, AssetLoader, LoadContext, io::Reader},
};
use serde::Deserialize;
use thiserror::Error;

use crate::{
    ModifierGroup,
    model::{
        EffectGraph, EffectGraphAsset, EmitterGraph, EmitterId, FORMAT_VERSION, GraphLayout,
        SourceContext, SourceId, SourceKind, SourceLink,
    },
};

/// Leading marker line stamped on every `.hnb` file for content detection.
///
/// A RON line comment, so deserialization ignores it and files written by
/// older builds without it still load. It lets `file`-style tools and humans
/// recognize a `.hnb` graph by its first line without parsing the whole
/// document.
pub const MAGIC_HEADER: &str =
    "// Hanabi effect graph - https://github.com/djeedai/hanabi-workshop";

/// Registers [`EffectGraphAsset`] and its [`EffectGraphLoader`].
pub struct EffectGraphPlugin;

impl Plugin for EffectGraphPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<EffectGraphAsset>()
            .init_asset_loader::<EffectGraphLoader>();
    }
}

/// Parse a [`EffectGraphAsset`] from RON bytes.
///
/// Peeks the schema version, rejects versions newer than [`FORMAT_VERSION`],
/// and upgrades older ones through the migration ladder before deserializing.
/// The single source of truth for the `.hnb` on-disk format, shared by
/// [`EffectGraphLoader`] and synchronous editor saves/loads.
pub fn from_ron_bytes(bytes: &[u8]) -> Result<EffectGraphAsset, EffectGraphLoaderError> {
    // Read only the version first; RON ignores the leading magic-header comment
    // and any fields beyond `version`.
    let AssetVersion { version } = ron::de::from_bytes(bytes)?;
    if version > FORMAT_VERSION {
        return Err(EffectGraphLoaderError::UnsupportedVersion {
            found: version,
            supported: FORMAT_VERSION,
        });
    }
    if version == FORMAT_VERSION {
        return Ok(ron::de::from_bytes(bytes)?);
    }
    migrate_from(bytes, version)
}

/// Serialize a [`EffectGraphAsset`] to pretty RON for writing a `.hnb` file.
///
/// Prepends [`MAGIC_HEADER`] so every written file is content-detectable.
pub fn to_ron_string(asset: &EffectGraphAsset) -> Result<String, ron::Error> {
    let body = ron::ser::to_string_pretty(asset, ron::ser::PrettyConfig::default())?;
    Ok(format!("{MAGIC_HEADER}\n{body}"))
}

/// The subset of a [`EffectGraphAsset`] needed to branch on schema version.
#[derive(Deserialize)]
struct AssetVersion {
    version: u32,
}

/// Deserialize and upgrade an asset saved under an older `version`.
///
/// Walks the migration ladder up to [`FORMAT_VERSION`], one guarded step per
/// breaking bump, then stamps the result with [`FORMAT_VERSION`]. Reached
/// only for `version < FORMAT_VERSION`.
fn migrate_from(bytes: &[u8], version: u32) -> Result<EffectGraphAsset, EffectGraphLoaderError> {
    let mut asset = apply_migrations(bytes, version)?;
    asset.version = FORMAT_VERSION;
    Ok(asset)
}

/// Apply ordered `v→v+1` transforms, starting from `from`, until
/// [`FORMAT_VERSION`] is reached.
///
/// Each breaking [`FORMAT_VERSION`] bump appends one guarded step here. Only
/// `1 → 2` exists so far (see [`migrate_v1_to_v2`]); `from_ron_bytes` never
/// calls this for `from >= FORMAT_VERSION`, so a `from` reaching the `else`
/// branch below would mean a future bump forgot to add its ladder step.
fn apply_migrations(bytes: &[u8], from: u32) -> Result<EffectGraphAsset, EffectGraphLoaderError> {
    if from < 2 {
        return migrate_v1_to_v2(bytes);
    }
    unreachable!(
        "from_ron_bytes only reaches apply_migrations for from < FORMAT_VERSION ({FORMAT_VERSION}); \
         version {from} has no migration step defined"
    );
}

/// Upgrade a `version: 1` document to `version: 2`.
///
/// Deserializes the frozen [`legacy_v1`] shape, wraps its single
/// [`EmitterGraph`] into one emitter of a new [`EffectGraph`], and creates a
/// connected [`SourceKind::CpuSpawner`] from the old header's
/// `SpawnerSettings`.
///
/// - The old graph's node/stack/property/slot ids, and its
///   [`GraphLayout::node_pos`]/[`GraphLayout::stack_pos`] entries, are
///   preserved unchanged (the legacy `nodes`/`stacks`/`links`/`layout` values
///   are moved into the new document verbatim).
/// - The new emitter and CPU Spawner ids are minted just above the old graph's
///   allocator (`old_next_id` and `old_next_id + 1`), which is always safe
///   because every id the old allocator ever handed out is strictly less than
///   `old_next_id`.
/// - The document's allocator is left at `old_next_id + 2`.
/// - The new CPU Spawner's layout position (when the file has a layout at all)
///   is placed to the left of the migrated Init stack (or the leftmost node, or
///   a fixed fallback), purely so the migrated document doesn't open with the
///   source context stacked on top of the emitter; the editor's layout tooling
///   is free to move it.
fn migrate_v1_to_v2(bytes: &[u8]) -> Result<EffectGraphAsset, EffectGraphLoaderError> {
    let legacy_v1::EffectGraphAssetV1 {
        version: _,
        graph,
        layout,
    } = ron::de::from_bytes(bytes)?;
    let legacy_v1::EffectGraphV1 {
        header,
        properties,
        texture_slots,
        nodes,
        stacks,
        links,
        next_id: old_next_id,
    } = graph;
    let legacy_v1::EffectHeaderV1 {
        name,
        capacity,
        spawner,
        simulation_space,
        simulation_condition,
        z_layer_2d,
    } = header;

    let emitter_id_num = old_next_id;
    let source_id_num = checked_successor(emitter_id_num)?;
    let new_next_id = checked_successor(source_id_num)?;

    let emitter_id =
        EmitterId::new(emitter_id_num).ok_or_else(|| EffectGraphLoaderError::Migration {
            from: 1,
            source: ron::Error::Message(format!(
                "v1 graph has next_id={emitter_id_num}, which cannot name a nonzero emitter id"
            )),
        })?;
    // `source_id_num` is `emitter_id_num + 1`, so it is nonzero whenever
    // `emitter_id_num` is (the only way `EmitterId::new` above could have failed).
    let source_id = SourceId::new(source_id_num).expect("nonzero by construction above");

    let emitter = EmitterGraph {
        id: emitter_id,
        name,
        capacity,
        simulation_space,
        simulation_condition,
        z_layer_2d,
        properties,
        texture_slots,
        nodes,
        stacks,
        links,
    };

    let layout = layout.map(|mut l| {
        let pos = source_layout_position(&emitter, &l);
        l.source_pos = vec![(source_id, pos)];
        l
    });

    Ok(EffectGraphAsset {
        version: FORMAT_VERSION,
        graph: EffectGraph {
            source_links: vec![SourceLink {
                source: source_id,
                emitter: emitter_id,
            }],
            sources: vec![SourceContext {
                id: source_id,
                kind: SourceKind::CpuSpawner { settings: spawner },
            }],
            emitters: vec![emitter],
            event_links: Vec::new(),
            next_id: new_next_id,
        },
        layout,
    })
}

/// `id + 1`, reported as a migration error instead of panicking on overflow.
fn checked_successor(id: u32) -> Result<u32, EffectGraphLoaderError> {
    id.checked_add(1)
        .ok_or_else(|| EffectGraphLoaderError::Migration {
            from: 1,
            source: ron::Error::Message(format!(
                "id allocator overflow migrating a v1 graph (id {id} has no successor)"
            )),
        })
}

/// A reasonable canvas position for the CPU Spawner minted by
/// [`migrate_v1_to_v2`], given the migrated emitter's own (unmoved) layout.
///
/// Prefers sitting to the left of the migrated Init stack; falls back to the
/// leftmost placed node, then to a fixed offset, so a migrated document never
/// opens with the new source context exactly overlapping existing content.
fn source_layout_position(emitter: &EmitterGraph, layout: &GraphLayout) -> (f64, f64) {
    const LEFT_OFFSET: f64 = 260.0;

    if let Some(stack) = emitter.stack(ModifierGroup::Init) {
        if let Some((_, pos)) = layout.stack_pos.iter().find(|(id, _)| *id == stack.id) {
            return (pos.0 - LEFT_OFFSET, pos.1);
        }
    }
    if let Some((_, pos)) = layout
        .node_pos
        .iter()
        .min_by(|a, b| a.1.0.total_cmp(&b.1.0))
    {
        return (pos.0 - LEFT_OFFSET, pos.1);
    }
    (-LEFT_OFFSET, 0.0)
}

/// Frozen shapes of on-disk `version: 1` documents, used only by
/// [`migrate_v1_to_v2`].
///
/// Each type here mirrors exactly the persisted shape of
/// [`FORMAT_VERSION`] `1` and must never change once that version has
/// shipped — see `tests/fixtures/README.md`'s immutability contract.
/// Unchanged model types ([`GraphNode`], [`GraphStack`], [`GraphLink`],
/// [`PropertyDef`], [`TextureSlotDef`]) are reused directly so this module
/// only shadows the fields that actually moved.
mod legacy_v1 {
    use bevy_hanabi::{SimulationCondition, SimulationSpace, SpawnerSettings};
    use serde::Deserialize;

    use crate::model::{
        GraphLayout, GraphLink, GraphNode, GraphStack, PropertyDef, SharedStr, TextureSlotDef,
    };

    /// The `version: 1` asset shape: a bare `EmitterGraph`, not yet a
    /// `EffectGraph`.
    #[derive(Deserialize)]
    pub struct EffectGraphAssetV1 {
        /// Always `1`; the caller already peeked it via [`super::AssetVersion`]
        /// before choosing this migration step, so it is never read again —
        /// kept only so this struct mirrors the on-disk shape exactly.
        #[allow(dead_code)]
        pub version: u32,
        pub graph: EffectGraphV1,
        pub layout: Option<GraphLayout>,
    }

    /// The `version: 1` `EmitterGraph` shape: header inline, no `id`, and its
    /// own `next_id` allocator (moved to `EffectGraph` in version 2).
    #[derive(Deserialize)]
    pub struct EffectGraphV1 {
        pub header: EffectHeaderV1,
        pub properties: Vec<PropertyDef>,
        #[serde(default)]
        pub texture_slots: Vec<TextureSlotDef>,
        pub nodes: Vec<GraphNode>,
        pub stacks: Vec<GraphStack>,
        pub links: Vec<GraphLink>,
        pub next_id: u32,
    }

    /// The version-1 nested header, which still carries `spawner`.
    #[derive(Deserialize)]
    pub struct EffectHeaderV1 {
        pub name: SharedStr,
        pub capacity: u32,
        pub spawner: SpawnerSettings,
        pub simulation_space: SimulationSpace,
        pub simulation_condition: SimulationCondition,
        pub z_layer_2d: f32,
    }
}

/// Loads `.hnb` RON files into a [`EffectGraphAsset`].
#[derive(Default, bevy::reflect::TypePath)]
pub struct EffectGraphLoader;

/// Errors produced while loading a [`EffectGraphAsset`].
#[derive(Debug, Error)]
pub enum EffectGraphLoaderError {
    #[error("failed to read asset bytes: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to deserialize EffectGraphAsset RON: {0}")]
    Ron(#[from] ron::error::SpannedError),
    #[error("unsupported EffectGraphAsset version {found}; this build supports up to {supported}")]
    UnsupportedVersion { found: u32, supported: u32 },
    #[error("failed to migrate EffectGraphAsset from version {from}: {source}")]
    Migration { from: u32, source: ron::Error },
}

impl AssetLoader for EffectGraphLoader {
    type Asset = EffectGraphAsset;
    type Settings = ();
    type Error = EffectGraphLoaderError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &Self::Settings,
        _load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        from_ron_bytes(&bytes)
    }

    fn extensions(&self) -> &[&str] {
        &["hnb"]
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use bevy_hanabi::{CpuValue, SpawnerSettings};

    use super::*;
    use crate::model::{
        EffectGraph, EmitterId, SourceContext, SourceId, SourceKind, SourceLink, StackId,
    };

    fn sample_asset() -> EffectGraphAsset {
        let emitter_id = EmitterId::new(1).unwrap();
        let source_id = SourceId::new(2).unwrap();
        EffectGraphAsset {
            version: FORMAT_VERSION,
            graph: EffectGraph {
                emitters: vec![EmitterGraph::empty(emitter_id)],
                sources: vec![SourceContext {
                    id: source_id,
                    kind: SourceKind::CpuSpawner {
                        settings: SpawnerSettings::default(),
                    },
                }],
                source_links: vec![SourceLink {
                    source: source_id,
                    emitter: emitter_id,
                }],
                event_links: Vec::new(),
                next_id: 3,
            },
            layout: Some(GraphLayout {
                pan: (1.0, -2.0),
                zoom: 1.5,
                node_pos: Vec::new(),
                stack_pos: Vec::new(),
                source_pos: vec![(source_id, (-260.0, 0.0))],
            }),
        }
    }

    #[test]
    fn ron_round_trips_through_helpers() {
        let asset = sample_asset();
        let text = to_ron_string(&asset).expect("serialize");
        let back = from_ron_bytes(text.as_bytes()).expect("deserialize");
        assert_eq!(asset, back);
    }

    #[test]
    fn writes_magic_header() {
        let text = to_ron_string(&sample_asset()).expect("serialize");
        assert!(text.starts_with(MAGIC_HEADER));
    }

    #[test]
    fn loads_legacy_file_without_header() {
        // A file written before the header existed: plain RON, no leading
        // comment. It must still deserialize.
        let asset = sample_asset();
        let body = ron::ser::to_string_pretty(&asset, ron::ser::PrettyConfig::default())
            .expect("serialize");
        assert!(!body.starts_with(MAGIC_HEADER));
        let back = from_ron_bytes(body.as_bytes()).expect("deserialize");
        assert_eq!(asset, back);
    }

    #[test]
    fn rejects_future_version() {
        let mut asset = sample_asset();
        asset.version = FORMAT_VERSION + 1;
        let text = to_ron_string(&asset).expect("serialize");
        assert!(matches!(
            from_ron_bytes(text.as_bytes()),
            Err(EffectGraphLoaderError::UnsupportedVersion { .. })
        ));
    }

    // ── v1 → v2 migration ────────────────────────────────────────────────────

    fn read_v1_fixture(stem: &str) -> Vec<u8> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/hnb/v1")
            .join(format!("{stem}.hnb"));
        std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
    }

    /// A frozen v1 file (no node/stack graph, `next_id: 1`) migrates to a v2
    /// document with one emitter and one connected `CpuSpawner`, and the
    /// minted ids/allocator land exactly where the migration promises.
    #[test]
    fn migrates_minimal_v1_fixture() {
        let bytes = read_v1_fixture("minimal");
        let asset = from_ron_bytes(&bytes).expect("migrate minimal.hnb");

        assert_eq!(asset.version, FORMAT_VERSION);
        assert_eq!(asset.graph.emitters.len(), 1);
        assert_eq!(asset.graph.sources.len(), 1);
        assert_eq!(asset.graph.source_links.len(), 1);
        assert!(asset.graph.event_links.is_empty());

        let emitter = &asset.graph.emitters[0];
        assert_eq!(emitter.id, EmitterId::new(1).unwrap());
        assert_eq!(emitter.name.as_ref(), "minimal");
        assert_eq!(emitter.capacity, 256);
        assert!(emitter.nodes.is_empty());
        assert!(emitter.stacks.is_empty());

        let source = &asset.graph.sources[0];
        assert_eq!(source.id, SourceId::new(2).unwrap());
        match &source.kind {
            SourceKind::CpuSpawner { settings } => {
                assert_eq!(settings.count(), CpuValue::Single(1.0));
            }
            SourceKind::GpuEvent => panic!("expected a CpuSpawner"),
        }

        assert_eq!(asset.graph.source_links[0].source, source.id);
        assert_eq!(asset.graph.source_links[0].emitter, emitter.id);
        assert_eq!(asset.graph.next_id, 3);
        assert_eq!(asset.layout, None);
    }

    /// A larger frozen v1 file (`next_id: 19`, a populated graph) preserves
    /// every existing node/stack/property/slot id unchanged, and mints the
    /// new emitter/source ids strictly above the old allocator.
    #[test]
    fn migrates_demo_v1_fixture_preserving_ids() {
        let bytes = read_v1_fixture("demo");
        let asset = from_ron_bytes(&bytes).expect("migrate demo.hnb");

        assert_eq!(asset.version, FORMAT_VERSION);
        assert_eq!(asset.graph.emitters.len(), 1);
        let emitter = &asset.graph.emitters[0];

        // The old graph's populated content survived unchanged.
        assert!(!emitter.nodes.is_empty());
        assert!(!emitter.stacks.is_empty());
        assert!(emitter.nodes.iter().all(|n| n.id.get() < 19));
        assert!(emitter.stacks.iter().all(|s| s.id.get() < 19));
        assert!(emitter.properties.iter().all(|p| p.id.get() < 19));

        // New ids were minted strictly above the old allocator, and the
        // allocator now sits two past them.
        assert_eq!(emitter.id, EmitterId::new(19).unwrap());
        assert_eq!(asset.graph.sources[0].id, SourceId::new(20).unwrap());
        assert_eq!(asset.graph.next_id, 21);
    }

    /// When the v1 file carries a layout, the migrated document keeps every
    /// existing node/stack position unchanged and adds one `source_pos` entry
    /// for the newly created CPU Spawner.
    #[test]
    fn migration_preserves_layout_and_places_new_source() {
        let v1 = r#"(
            version: 1,
            graph: (
                header: (
                    name: "with_layout",
                    capacity: 512,
                    spawner: (
                        count: Single(2.0),
                        spawn_duration: Single(0.0),
                        period: Single(0.0),
                        cycle_count: 1,
                        starts_active: true,
                        emit_on_start: true,
                    ),
                    simulation_space: Global,
                    simulation_condition: WhenVisible,
                    z_layer_2d: 0.0,
                ),
                properties: [],
                texture_slots: [],
                nodes: [],
                stacks: [
                    (id: (1), group: Init, members: []),
                ],
                links: [],
                next_id: 2,
            ),
            layout: Some((
                pan: (0.0, 0.0),
                zoom: 1.0,
                node_pos: [],
                stack_pos: [((1), (400.0, 100.0))],
            )),
        )"#;

        let asset = from_ron_bytes(v1.as_bytes()).expect("migrate hand-authored v1 with layout");
        assert_eq!(asset.graph.emitters[0].id, EmitterId::new(2).unwrap());
        let source_id = SourceId::new(3).unwrap();
        assert_eq!(asset.graph.sources[0].id, source_id);
        assert_eq!(asset.graph.next_id, 4);

        let layout = asset.layout.expect("layout preserved");
        assert_eq!(
            layout.stack_pos,
            vec![(StackId::new(1).unwrap(), (400.0, 100.0))]
        );
        // Placed to the left of the migrated Init stack.
        assert_eq!(layout.source_pos, vec![(source_id, (140.0, 100.0))]);
    }

    /// A v1 file whose allocator already sits at `u32::MAX` cannot be
    /// migrated without overflowing the shared id space; the migration must
    /// fail loudly instead of silently colliding or wrapping.
    #[test]
    fn migration_reports_allocator_overflow_instead_of_panicking() {
        let v1 = format!(
            r#"(
                version: 1,
                graph: (
                    header: (
                        name: "overflow",
                        capacity: 256,
                        spawner: (
                            count: Single(1.0),
                            spawn_duration: Single(0.0),
                            period: Single(0.0),
                            cycle_count: 1,
                            starts_active: true,
                            emit_on_start: true,
                        ),
                        simulation_space: Global,
                        simulation_condition: WhenVisible,
                        z_layer_2d: 0.0,
                    ),
                    properties: [],
                    texture_slots: [],
                    nodes: [],
                    stacks: [],
                    links: [],
                    next_id: {},
                ),
                layout: None,
            )"#,
            u32::MAX
        );

        assert!(matches!(
            from_ron_bytes(v1.as_bytes()),
            Err(EffectGraphLoaderError::Migration { from: 1, .. })
        ));
    }
}
