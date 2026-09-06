//! Discovery of browsable emitters for the Home tab.
//!
//! Two sources feed the Home browser: a bundled `examples/` directory shipped
//! with the app, and a persisted list of recently opened/saved user files. Both
//! are surfaced as [`EffectEntry`] rows the browser can preview and open.

use std::path::{Path, PathBuf};

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// A single browsable emitter: a display name and its `.hnb` path on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectEntry {
    pub name: String,
    pub path: PathBuf,
}

impl EffectEntry {
    /// Build an entry from a path, deriving the name from the file stem.
    pub fn from_path(path: PathBuf) -> Self {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("emitter")
            .to_string();
        Self { name, path }
    }
}

// ============================================================================
// Bundled examples
// ============================================================================

/// The bundled example emitters, discovered once at startup.
#[derive(Resource, Default)]
pub struct ExampleLibrary(pub Vec<EffectEntry>);

/// List the `.hnb` entries directly inside `dir`, sorted by name.
fn hnb_entries_in(dir: &Path) -> Vec<EffectEntry> {
    let mut entries: Vec<EffectEntry> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "hnb"))
        .map(EffectEntry::from_path)
        .collect();
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    entries
}

/// Enumerate the bundled example `.hnb` files, sorted by name.
///
/// Uses [`crate::resource_paths::resolve_bundled_root`] to locate the
/// `examples/` directory so discovery is independent of the launch working
/// directory. Returns an empty list when no bundled root with an `assets/`
/// subdirectory can be found.
pub fn discover_examples() -> Vec<EffectEntry> {
    let Some(root) = crate::resource_paths::resolve_bundled_root() else {
        return Vec::new();
    };
    hnb_entries_in(&root.join("examples"))
}

// ============================================================================
// Recent files
// ============================================================================

/// Persisted most-recently-used list of user emitter files.
///
/// Most-recent first, de-duplicated, and capped at [`RecentFiles::CAP`].
#[derive(Resource, Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecentFiles {
    pub paths: Vec<PathBuf>,
}

impl RecentFiles {
    /// Maximum number of remembered files.
    const CAP: usize = 20;

    /// Record a file as most-recently-used.
    ///
    /// Moves an existing entry to the front rather than duplicating it, and
    /// trims the list to [`RecentFiles::CAP`]. Paths are canonicalized when
    /// they resolve on disk so the same file under different spellings
    /// collapses to one entry.
    pub fn record(&mut self, path: &Path) {
        let path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        self.paths.retain(|p| p != &path);
        self.paths.insert(0, path);
        self.paths.truncate(Self::CAP);
    }

    /// The recent files that still exist on disk, most-recent first.
    pub fn entries(&self) -> Vec<EffectEntry> {
        self.paths
            .iter()
            .filter(|p| p.exists())
            .cloned()
            .map(EffectEntry::from_path)
            .collect()
    }
}

/// Path of the persisted recents file under the OS config dir.
fn recents_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("hanabi-workshop").join("recents.ron"))
}

/// Load the recent-files list from disk, or an empty list if unavailable.
pub fn load_recent_files() -> RecentFiles {
    let Some(path) = recents_path() else {
        return RecentFiles::default();
    };
    match std::fs::read_to_string(&path) {
        Ok(text) => ron::from_str(&text).unwrap_or_default(),
        Err(_) => RecentFiles::default(),
    }
}

/// Persist the recent-files list to disk (best-effort).
pub fn save_recent_files(recents: &RecentFiles) {
    let Some(path) = recents_path() else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match ron::ser::to_string(recents) {
        Ok(text) => {
            if let Err(e) = std::fs::write(&path, text) {
                warn!("failed to save recent files to {}: {e}", path.display());
            }
        }
        Err(e) => warn!("failed to serialize recent files: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A path that won't canonicalize, so `record` keeps it verbatim.
    fn p(s: &str) -> PathBuf {
        PathBuf::from(format!("/nonexistent/{s}.hnb"))
    }

    #[test]
    fn record_moves_existing_to_front_without_duplicating() {
        let mut r = RecentFiles::default();
        r.record(&p("a"));
        r.record(&p("b"));
        r.record(&p("a"));
        assert_eq!(r.paths, vec![p("a"), p("b")]);
    }

    #[test]
    fn record_caps_length() {
        let mut r = RecentFiles::default();
        for i in 0..(RecentFiles::CAP + 5) {
            r.record(&p(&i.to_string()));
        }
        assert_eq!(r.paths.len(), RecentFiles::CAP);
        // Most recent first.
        assert_eq!(r.paths[0], p(&(RecentFiles::CAP + 4).to_string()));
    }

    #[test]
    fn round_trips_through_ron() {
        let mut r = RecentFiles::default();
        r.record(&p("a"));
        r.record(&p("b"));
        let text = ron::ser::to_string(&r).unwrap();
        let back: RecentFiles = ron::from_str(&text).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn discovers_bundled_demo_example() {
        let examples = discover_examples();
        assert!(
            examples.iter().any(|e| e.name == "demo"),
            "expected a bundled `demo` example, found: {:?}",
            examples.iter().map(|e| &e.name).collect::<Vec<_>>()
        );
    }

    /// Every shipped `.hnb` in `examples/` must parse and bake, so the browser
    /// never lists an example that fails to open.
    #[test]
    fn bundled_examples_load_and_bake() {
        use bevy::prelude::*;

        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()));
        app.add_plugins(hanabi_effect_graph::modifier_registry::ModifierRegistryPlugin);
        let registry = app.world().resource::<AppTypeRegistry>().read();

        let examples = discover_examples();
        assert!(
            examples.len() >= 2,
            "expected multiple bundled examples, found: {:?}",
            examples.iter().map(|e| &e.name).collect::<Vec<_>>()
        );
        for entry in examples {
            let bytes = std::fs::read(&entry.path).expect("read example");
            let graph = hanabi_effect_graph::from_ron_bytes(&bytes)
                .expect("parse example")
                .graph;
            if let Err(errors) = hanabi_effect_graph::bake::bake_effect(&graph, &registry) {
                panic!("example `{}` failed to bake: {errors:?}", entry.name);
            }
        }
    }
}
