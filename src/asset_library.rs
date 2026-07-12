//! Texture catalog discovery and persisted browser settings.
//!
//! The catalog combines bundled presets, the process asset root, and
//! user-selected external folders without depending on Bevy's asset server.
//! Scans are synchronous, self-contained values so callers can move them onto
//! an asynchronous task without granting that task access to the Bevy world.

use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet, hash_map::Entry},
    ffi::OsStr,
    fmt, fs, io,
    path::{Component, Path, PathBuf},
};

use bevy::{
    asset::AssetPath,
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task, futures_lite::future},
};
use serde::{Deserialize, Serialize};

/// The `assets/` subdirectory name, relative to the bundled resource root.
pub const PROJECT_ASSET_ROOT: &str = "assets";

/// The bundled preset folder relative to `PROJECT_ASSET_ROOT`.
pub const PRESET_TEXTURE_ROOT: &str = "textures/patterns";

/// The settings file name below the application configuration directory.
pub const TEXTURE_LIBRARY_SETTINGS_FILE: &str = "texture-library.ron";

const TEXTURE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "ktx2", "basis", "exr", "hdr"];

/// The catalog section from which a texture was discovered.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TextureSource {
    /// A texture bundled in `assets/textures/patterns`.
    Preset,
    /// A non-preset texture below the process asset root.
    Project,
    /// A texture below the configured root.
    External(PathBuf),
}

impl TextureSource {
    fn priority(&self) -> u8 {
        match self {
            Self::Preset => 0,
            Self::Project => 1,
            Self::External(_) => 2,
        }
    }

    fn sort_path(&self) -> &Path {
        match self {
            Self::External(root) => root,
            Self::Preset | Self::Project => Path::new(""),
        }
    }
}

/// A texture available to the Workshop texture browser.
///
/// `canonical_path` is the filesystem identity used for deduplication.
/// `asset_path` is the value suitable for persisting into an effect graph:
/// paths below the process asset root are relative to that root, while all
/// other paths are absolute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureEntry {
    pub canonical_path: PathBuf,
    pub asset_path: AssetPath<'static>,
    pub display_name: String,
    pub source: TextureSource,
    pub relative_display_path: PathBuf,
}

/// The global set of textures displayed by the Workshop texture browser.
#[derive(Resource, Debug, Clone, Default, PartialEq, Eq)]
pub struct TextureCatalog {
    pub entries: Vec<TextureEntry>,
}

/// Installs texture discovery and persisted browser settings.
pub struct TextureLibraryPlugin;

impl Plugin for TextureLibraryPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TextureCatalog>()
            .insert_resource(load_texture_library_settings())
            .init_resource::<TextureScanWork>()
            .add_message::<TextureLibraryCommand>()
            .add_systems(Startup, request_initial_scan)
            .add_systems(
                Update,
                (handle_texture_library_commands, drive_texture_scan).chain(),
            );
    }
}

/// A global texture-library action.
#[derive(Message, Debug, Clone)]
pub enum TextureLibraryCommand {
    /// Add a recursively scanned external folder.
    AddExternalRoot(PathBuf),
    /// Stop scanning an external folder.
    RemoveExternalRoot(PathBuf),
    /// Persist the asset browser's selected density.
    SetViewMode(TextureViewMode),
    /// Rebuild the catalog from the configured roots.
    Rescan,
}

#[derive(Resource, Default)]
struct TextureScanWork {
    task: Option<Task<TextureScanResult>>,
    rerun: bool,
}

fn request_initial_scan(mut commands: MessageWriter<TextureLibraryCommand>) {
    commands.write(TextureLibraryCommand::Rescan);
}

fn handle_texture_library_commands(
    mut commands: MessageReader<TextureLibraryCommand>,
    mut settings: ResMut<TextureLibrarySettings>,
    mut work: ResMut<TextureScanWork>,
) {
    let process_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut settings_changed = false;
    let mut rescan = false;

    for command in commands.read() {
        match command {
            TextureLibraryCommand::AddExternalRoot(root) => {
                let roots = normalize_external_roots(&process_dir, [root.clone()]);
                if let Some(root) = roots.into_iter().next()
                    && !settings.external_roots.contains(&root)
                {
                    settings.external_roots.push(root);
                    settings_changed = true;
                    rescan = true;
                }
            }
            TextureLibraryCommand::RemoveExternalRoot(root) => {
                let before = settings.external_roots.len();
                settings
                    .external_roots
                    .retain(|candidate| candidate != root);
                settings_changed |= settings.external_roots.len() != before;
                rescan |= settings.external_roots.len() != before;
            }
            TextureLibraryCommand::SetViewMode(mode) => {
                if settings.view_mode != *mode {
                    settings.view_mode = *mode;
                    settings_changed = true;
                }
            }
            TextureLibraryCommand::Rescan => rescan = true,
        }
    }

    if settings_changed {
        save_texture_library_settings(&settings);
    }
    if rescan {
        if work.task.is_some() {
            work.rerun = true;
        } else {
            work.task = Some(spawn_texture_scan(settings.scan_request()));
        }
    }
}

fn drive_texture_scan(
    settings: Res<TextureLibrarySettings>,
    mut catalog: ResMut<TextureCatalog>,
    mut work: ResMut<TextureScanWork>,
) {
    let Some(task) = work.task.as_mut() else {
        return;
    };
    let Some(result) = future::block_on(future::poll_once(task)) else {
        return;
    };

    work.task = None;
    if work.rerun {
        work.rerun = false;
        work.task = Some(spawn_texture_scan(settings.scan_request()));
        return;
    }

    for issue in &result.issues {
        warn!(
            "texture scan skipped {}: {}",
            issue.path.display(),
            issue.message
        );
    }
    info!(
        "texture scan found {} image assets",
        result.catalog.entries.len()
    );
    *catalog = result.catalog;
}

fn spawn_texture_scan(request: TextureScanRequest) -> Task<TextureScanResult> {
    AsyncComputeTaskPool::get().spawn(async move { scan_texture_catalog(request) })
}

/// Normalize a selected file into its persisted graph asset path.
///
/// Files below the bundled `assets/` root become asset-root-relative so they
/// survive the binary being moved to a different working directory; every other
/// file remains absolute.
pub fn persisted_texture_asset_path(path: &Path) -> AssetPath<'static> {
    let asset_root = crate::resource_paths::resolve_bundled_root()
        .map(|r| canonical_or_normalized(&r.join(PROJECT_ASSET_ROOT)))
        .unwrap_or_else(|| {
            let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            canonical_or_normalized(&cwd.join(PROJECT_ASSET_ROOT))
        });
    let canonical = canonical_or_normalized(path);
    let persisted = canonical.strip_prefix(&asset_root).unwrap_or(&canonical);
    AssetPath::from_path_buf(persisted.to_path_buf())
}

/// Inputs for a filesystem-only texture catalog scan.
///
/// The owned request and result are `Send + 'static`, allowing a caller to move
/// the complete operation onto Bevy's task pool. Use
/// `TextureScanRequest::workshop` to apply the Workshop directory conventions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureScanRequest {
    pub asset_root: PathBuf,
    pub preset_root: PathBuf,
    pub external_roots: Vec<PathBuf>,
}

impl TextureScanRequest {
    /// Build a scan request using the Workshop's resolved bundled asset root.
    ///
    /// The asset root is determined by [`crate::resource_paths::resolve_bundled_root`]
    /// so the scan is independent of the launch working directory. Falls back to
    /// a CWD-relative `assets/` path if no bundled root can be resolved (e.g.
    /// incomplete installation), preserving scan behaviour in unusual
    /// environments.
    pub fn workshop(external_roots: impl IntoIterator<Item = PathBuf>) -> Self {
        let root = crate::resource_paths::resolve_bundled_root()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        let asset_root = normalize_path(&root.join(PROJECT_ASSET_ROOT));
        let preset_root = asset_root.join(PRESET_TEXTURE_ROOT);
        let external_roots = normalize_external_roots(&root, external_roots);
        Self {
            asset_root,
            preset_root,
            external_roots,
        }
    }
}

/// A filesystem problem encountered during an otherwise best-effort scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureScanIssue {
    pub path: PathBuf,
    pub message: String,
}

/// The complete output of a texture catalog scan.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TextureScanResult {
    pub catalog: TextureCatalog,
    pub issues: Vec<TextureScanIssue>,
}

#[derive(Debug)]
struct Candidate {
    canonical_path: PathBuf,
    source: TextureSource,
    relative_display_path: PathBuf,
}

/// Recursively discover textures described by an owned scan request.
///
/// Directories reached through symlinks are not traversed. Files are
/// deduplicated by canonical path with `Preset > Project > External`
/// precedence, then sorted deterministically by source and case-insensitive
/// relative display path. Errors are collected in the result and do not abort
/// other roots.
pub fn scan_texture_catalog(request: TextureScanRequest) -> TextureScanResult {
    let mut issues = Vec::new();
    let asset_root = canonical_or_normalized(&request.asset_root);
    let preset_root = canonical_or_normalized(&request.preset_root);
    let process_dir = asset_root.parent().unwrap_or(Path::new(""));
    let external_roots = normalize_external_roots(process_dir, request.external_roots).into_iter();

    let mut candidates = Vec::new();
    scan_root(
        &preset_root,
        TextureSource::Preset,
        &mut candidates,
        &mut issues,
    );
    scan_root(
        &asset_root,
        TextureSource::Project,
        &mut candidates,
        &mut issues,
    );
    for root in external_roots {
        let root = canonical_or_normalized(&root);
        scan_root(
            &root,
            TextureSource::External(root.clone()),
            &mut candidates,
            &mut issues,
        );
    }

    let mut by_identity = HashMap::<PathBuf, TextureEntry>::new();
    for candidate in candidates {
        let entry = entry_from_candidate(candidate, &asset_root);
        match by_identity.entry(entry.canonical_path.clone()) {
            Entry::Vacant(slot) => {
                slot.insert(entry);
            }
            Entry::Occupied(mut slot) => {
                if preferred_entry(&entry, slot.get()) == Ordering::Less {
                    slot.insert(entry);
                }
            }
        }
    }

    let mut entries: Vec<_> = by_identity.into_values().collect();
    entries.sort_by(catalog_order);
    TextureScanResult {
        catalog: TextureCatalog { entries },
        issues,
    }
}

fn scan_root(
    root: &Path,
    source: TextureSource,
    candidates: &mut Vec<Candidate>,
    issues: &mut Vec<TextureScanIssue>,
) {
    let metadata = match fs::metadata(root) {
        Ok(metadata) if metadata.is_dir() => metadata,
        Ok(_) => {
            issues.push(TextureScanIssue {
                path: root.to_path_buf(),
                message: "scan root is not a directory".to_string(),
            });
            return;
        }
        Err(error) => {
            issues.push(TextureScanIssue {
                path: root.to_path_buf(),
                message: error.to_string(),
            });
            return;
        }
    };
    debug_assert!(metadata.is_dir());

    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let read_dir = match fs::read_dir(&directory) {
            Ok(read_dir) => read_dir,
            Err(error) => {
                issues.push(TextureScanIssue {
                    path: directory,
                    message: error.to_string(),
                });
                continue;
            }
        };

        for item in read_dir {
            let item = match item {
                Ok(item) => item,
                Err(error) => {
                    issues.push(TextureScanIssue {
                        path: directory.clone(),
                        message: error.to_string(),
                    });
                    continue;
                }
            };
            let path = item.path();
            let file_type = match item.file_type() {
                Ok(file_type) => file_type,
                Err(error) => {
                    issues.push(TextureScanIssue {
                        path,
                        message: error.to_string(),
                    });
                    continue;
                }
            };

            if file_type.is_dir() {
                pending.push(path);
                continue;
            }
            if file_type.is_symlink() {
                match fs::metadata(&path) {
                    Ok(target) if target.is_dir() => continue,
                    Ok(target) if !target.is_file() => continue,
                    Ok(_) => {}
                    Err(error) => {
                        issues.push(TextureScanIssue {
                            path,
                            message: error.to_string(),
                        });
                        continue;
                    }
                }
            } else if !file_type.is_file() {
                continue;
            }
            if !is_supported_texture_path(&path) {
                continue;
            }

            match fs::canonicalize(&path) {
                Ok(canonical_path) => {
                    let relative_display_path = path
                        .strip_prefix(root)
                        .map(Path::to_path_buf)
                        .unwrap_or_else(|_| {
                            path.file_name().map(PathBuf::from).unwrap_or_default()
                        });
                    candidates.push(Candidate {
                        canonical_path,
                        source: source.clone(),
                        relative_display_path,
                    });
                }
                Err(error) => issues.push(TextureScanIssue {
                    path,
                    message: error.to_string(),
                }),
            }
        }
    }
}

fn entry_from_candidate(candidate: Candidate, asset_root: &Path) -> TextureEntry {
    let persisted_path = candidate
        .canonical_path
        .strip_prefix(asset_root)
        .unwrap_or(&candidate.canonical_path)
        .to_path_buf();
    let display_name = candidate
        .relative_display_path
        .file_stem()
        .or_else(|| candidate.canonical_path.file_stem())
        .and_then(OsStr::to_str)
        .unwrap_or("texture")
        .to_string();
    TextureEntry {
        canonical_path: candidate.canonical_path,
        asset_path: AssetPath::from_path_buf(persisted_path),
        display_name,
        source: candidate.source,
        relative_display_path: candidate.relative_display_path,
    }
}

fn preferred_entry(left: &TextureEntry, right: &TextureEntry) -> Ordering {
    left.source
        .priority()
        .cmp(&right.source.priority())
        .then_with(|| {
            case_insensitive_path(&left.relative_display_path)
                .cmp(&case_insensitive_path(&right.relative_display_path))
        })
        .then_with(|| left.relative_display_path.cmp(&right.relative_display_path))
}

fn catalog_order(left: &TextureEntry, right: &TextureEntry) -> Ordering {
    left.source
        .priority()
        .cmp(&right.source.priority())
        .then_with(|| left.source.sort_path().cmp(right.source.sort_path()))
        .then_with(|| {
            case_insensitive_path(&left.relative_display_path)
                .cmp(&case_insensitive_path(&right.relative_display_path))
        })
        .then_with(|| left.relative_display_path.cmp(&right.relative_display_path))
        .then_with(|| left.canonical_path.cmp(&right.canonical_path))
}

/// Test whether a path has a supported image extension.
///
/// Extension matching is ASCII case-insensitive and accepts PNG, JPEG, KTX2,
/// Basis Universal, OpenEXR, and HDR files.
pub fn is_supported_texture_path(path: &Path) -> bool {
    path.extension().and_then(OsStr::to_str).is_some_and(|ext| {
        TEXTURE_EXTENSIONS
            .iter()
            .any(|supported| ext.eq_ignore_ascii_case(supported))
    })
}

/// Normalize and deduplicate configured external roots.
///
/// Relative roots are resolved against `base_dir`, existing paths are
/// canonicalized, and missing paths are normalized lexically. The first
/// spelling of each resulting identity retains its position.
pub fn normalize_external_roots(
    base_dir: &Path,
    roots: impl IntoIterator<Item = PathBuf>,
) -> Vec<PathBuf> {
    let base_dir = if base_dir.as_os_str().is_empty() {
        PathBuf::new()
    } else {
        canonical_or_normalized(base_dir)
    };
    let mut seen = HashSet::new();
    roots
        .into_iter()
        .filter(|root| !root.as_os_str().is_empty())
        .filter_map(|root| {
            let path = if root.is_absolute() {
                root
            } else {
                base_dir.join(root)
            };
            let path = canonical_or_normalized(&path);
            seen.insert(path.clone()).then_some(path)
        })
        .collect()
}

fn canonical_or_normalized(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| normalize_path(path))
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if matches!(
                    normalized.components().next_back(),
                    Some(Component::Normal(_))
                ) {
                    normalized.pop();
                } else if !path.is_absolute() {
                    normalized.push(component);
                }
            }
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                normalized.push(component)
            }
        }
    }
    normalized
}

fn case_insensitive_path(path: &Path) -> String {
    path.to_string_lossy().to_lowercase()
}

/// The texture browser's thumbnail density.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextureViewMode {
    /// One compact row per texture.
    List,
    /// A grid of compact thumbnails.
    #[default]
    Small,
    /// A grid of large thumbnails.
    Large,
}

/// User-configurable texture library state persisted across sessions.
#[derive(Resource, Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct TextureLibrarySettings {
    pub external_roots: Vec<PathBuf>,
    pub view_mode: TextureViewMode,
}

impl TextureLibrarySettings {
    /// Normalize external roots relative to a process directory.
    pub fn normalize_against(&mut self, process_dir: &Path) {
        self.external_roots =
            normalize_external_roots(process_dir, std::mem::take(&mut self.external_roots));
    }

    /// Build an owned scan request from these settings.
    pub fn scan_request(&self) -> TextureScanRequest {
        TextureScanRequest::workshop(self.external_roots.clone())
    }
}

/// An error reading, writing, or encoding texture library settings.
#[derive(Debug)]
pub enum TextureSettingsError {
    Io(io::Error),
    Serialize(ron::Error),
    Deserialize(ron::error::SpannedError),
}

impl fmt::Display for TextureSettingsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(formatter),
            Self::Serialize(error) => error.fmt(formatter),
            Self::Deserialize(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for TextureSettingsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Serialize(error) => Some(error),
            Self::Deserialize(error) => Some(error),
        }
    }
}

impl From<io::Error> for TextureSettingsError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Return the per-user texture library settings path.
pub fn texture_library_settings_path() -> Option<PathBuf> {
    dirs::config_dir().map(|directory| {
        directory
            .join("hanabi-workshop")
            .join(TEXTURE_LIBRARY_SETTINGS_FILE)
    })
}

/// Encode texture library settings as pretty RON.
pub fn serialize_texture_library_settings(
    settings: &TextureLibrarySettings,
) -> Result<String, TextureSettingsError> {
    ron::ser::to_string_pretty(settings, ron::ser::PrettyConfig::default())
        .map_err(TextureSettingsError::Serialize)
}

/// Decode texture library settings from RON.
pub fn deserialize_texture_library_settings(
    source: &str,
) -> Result<TextureLibrarySettings, TextureSettingsError> {
    ron::from_str(source).map_err(TextureSettingsError::Deserialize)
}

/// Load texture library settings from an explicit path.
///
/// This path-based helper performs no logging and is suitable for tests or
/// alternate hosts.
pub fn load_texture_library_settings_from(
    path: &Path,
) -> Result<TextureLibrarySettings, TextureSettingsError> {
    let source = fs::read_to_string(path)?;
    deserialize_texture_library_settings(&source)
}

/// Save texture library settings to an explicit path.
///
/// The parent directory is created when necessary. This path-based helper
/// performs no logging and is suitable for tests or alternate hosts.
pub fn save_texture_library_settings_to(
    path: &Path,
    settings: &TextureLibrarySettings,
) -> Result<(), TextureSettingsError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serialize_texture_library_settings(settings)?)?;
    Ok(())
}

/// Load the per-user texture library settings.
///
/// Missing settings and unavailable platform configuration directories yield
/// defaults. Other failures are logged and also yield defaults.
pub fn load_texture_library_settings() -> TextureLibrarySettings {
    let Some(path) = texture_library_settings_path() else {
        info!("texture library settings unavailable: platform has no configuration directory");
        return TextureLibrarySettings::default();
    };
    let mut settings = match load_texture_library_settings_from(&path) {
        Ok(settings) => {
            info!("loaded texture library settings from {}", path.display());
            settings
        }
        Err(TextureSettingsError::Io(error)) if error.kind() == io::ErrorKind::NotFound => {
            info!(
                "texture library settings not found at {}; using defaults",
                path.display()
            );
            TextureLibrarySettings::default()
        }
        Err(error) => {
            warn!(
                "failed to load texture library settings from {}: {error}",
                path.display()
            );
            TextureLibrarySettings::default()
        }
    };
    let process_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    settings.normalize_against(&process_dir);
    settings
}

/// Persist the per-user texture library settings.
///
/// Failures are logged and do not terminate the application.
pub fn save_texture_library_settings(settings: &TextureLibrarySettings) {
    let Some(path) = texture_library_settings_path() else {
        warn!("cannot save texture library settings: platform has no configuration directory");
        return;
    };
    let process_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut normalized = settings.clone();
    normalized.normalize_against(&process_dir);
    match save_texture_library_settings_to(&path, &normalized) {
        Ok(()) => info!("saved texture library settings to {}", path.display()),
        Err(error) => warn!(
            "failed to save texture library settings to {}: {error}",
            path.display()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(path: &str, source: TextureSource, relative: &str) -> Candidate {
        Candidate {
            canonical_path: PathBuf::from(path),
            source,
            relative_display_path: PathBuf::from(relative),
        }
    }

    fn catalog_from_candidates(asset_root: &Path, candidates: Vec<Candidate>) -> TextureCatalog {
        let mut by_identity = HashMap::<PathBuf, TextureEntry>::new();
        for candidate in candidates {
            let entry = entry_from_candidate(candidate, asset_root);
            match by_identity.entry(entry.canonical_path.clone()) {
                Entry::Vacant(slot) => {
                    slot.insert(entry);
                }
                Entry::Occupied(mut slot) => {
                    if preferred_entry(&entry, slot.get()) == Ordering::Less {
                        slot.insert(entry);
                    }
                }
            }
        }
        let mut entries: Vec<_> = by_identity.into_values().collect();
        entries.sort_by(catalog_order);
        TextureCatalog { entries }
    }

    #[test]
    fn normalizes_relative_external_roots_and_removes_duplicates() {
        let roots = normalize_external_roots(
            Path::new("/workspace/project"),
            [
                PathBuf::from("../textures/./shared"),
                PathBuf::from("/workspace/textures/shared"),
                PathBuf::from("other/../local"),
            ],
        );
        assert_eq!(
            roots,
            vec![
                PathBuf::from("/workspace/textures/shared"),
                PathBuf::from("/workspace/project/local"),
            ]
        );
    }

    #[test]
    fn filters_extensions_case_insensitively() {
        for path in [
            "a.png", "a.JPG", "a.JpEg", "a.ktx2", "a.BASIS", "a.exr", "a.HDR",
        ] {
            assert!(is_supported_texture_path(Path::new(path)), "{path}");
        }
        for path in ["a", "a.gif", "a.png.txt", ".png"] {
            assert!(!is_supported_texture_path(Path::new(path)), "{path}");
        }
    }

    #[test]
    fn deduplicates_by_identity_with_source_precedence() {
        let identity = "/project/assets/textures/patterns/smoke.png";
        let catalog = catalog_from_candidates(
            Path::new("/project/assets"),
            vec![
                candidate(
                    identity,
                    TextureSource::External(PathBuf::from("/project")),
                    "assets/textures/patterns/smoke.png",
                ),
                candidate(
                    identity,
                    TextureSource::Project,
                    "textures/patterns/smoke.png",
                ),
                candidate(identity, TextureSource::Preset, "smoke.png"),
            ],
        );
        assert_eq!(catalog.entries.len(), 1);
        assert_eq!(catalog.entries[0].source, TextureSource::Preset);
        assert_eq!(
            catalog.entries[0].asset_path,
            AssetPath::from("textures/patterns/smoke.png")
        );
    }

    #[test]
    fn external_entries_persist_absolute_asset_paths() {
        let path = PathBuf::from("/art/textures/smoke.png");
        let catalog = catalog_from_candidates(
            Path::new("/project/assets"),
            vec![candidate(
                path.to_str().unwrap(),
                TextureSource::External(PathBuf::from("/art/textures")),
                "smoke.png",
            )],
        );
        assert_eq!(
            catalog.entries[0].asset_path,
            AssetPath::from_path_buf(path)
        );
    }

    #[test]
    fn sorts_stably_without_extension_case_sensitivity() {
        let catalog = catalog_from_candidates(
            Path::new("/project/assets"),
            vec![
                candidate("/project/assets/z.png", TextureSource::Project, "z.png"),
                candidate("/project/assets/B.png", TextureSource::Project, "B.png"),
                candidate("/project/assets/a.png", TextureSource::Project, "a.png"),
            ],
        );
        assert_eq!(
            catalog
                .entries
                .iter()
                .map(|entry| entry.display_name.as_str())
                .collect::<Vec<_>>(),
            ["a", "B", "z"]
        );
    }

    #[test]
    fn round_trips_settings_through_helpers() {
        let settings = TextureLibrarySettings {
            external_roots: vec![PathBuf::from("/art/textures")],
            view_mode: TextureViewMode::Large,
        };
        let encoded = serialize_texture_library_settings(&settings).unwrap();
        let decoded = deserialize_texture_library_settings(&encoded).unwrap();
        assert_eq!(decoded, settings);
    }

    #[test]
    fn defaults_missing_settings_fields() {
        let decoded = deserialize_texture_library_settings("()").unwrap();
        assert_eq!(decoded, TextureLibrarySettings::default());
    }
}
