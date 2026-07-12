//! Centralized bundled-resource root discovery.
//!
//! A single resolved root directory gives Bevy's [`AssetPlugin`], the texture
//! catalog scanner, and the example browser a consistent, relocatable base.
//! The root is the directory that *contains* `assets/` and `examples/`; callers
//! join the appropriate subdirectory themselves.
//!
//! Resolution uses [`bundled_root_candidates`] in priority order and accepts the
//! first candidate whose `assets/` subdirectory exists on disk. The deterministic
//! order is: an explicit developer/test override, the macOS app-bundle
//! `Contents/Resources` directory, the directory beside the executable (portable
//! archive layout), and the Cargo manifest directory (development/`cargo run`
//! layout). Launch current working directory is intentionally excluded; bundled
//! resources must be found regardless of where the user invoked the binary.
//!
//! [`AssetPlugin`]: bevy::asset::AssetPlugin

use std::path::{Path, PathBuf};

/// Environment variable that overrides bundled-resource root discovery.
///
/// When set to a non-empty absolute path, that path is used unconditionally as
/// the bundled root and all other candidates are skipped. Intended for
/// developers and automated tests that need to point the app at a synthetic
/// resource tree.
pub const RESOURCE_ROOT_ENV: &str = "HANABI_WORKSHOP_RESOURCE_ROOT";

/// All candidate bundled-resource roots in decreasing priority order.
///
/// Each entry is an absolute (or env-expanded) path to a directory that might
/// contain `assets/` and `examples/`. Callers typically pass this list to
/// [`resolve_bundled_root`] or iterate it themselves when searching for a
/// specific subdirectory.
///
/// Order:
/// 1. `$HANABI_WORKSHOP_RESOURCE_ROOT` — explicit developer/test override.
/// 2. `Contents/Resources` of the macOS app bundle whose `MacOS/` directory
///    contains the current executable.
/// 3. The directory containing the current executable (portable archive layout:
///    `assets/` sits beside the binary).
/// 4. `env!("CARGO_MANIFEST_DIR")` — development layout, valid when the binary
///    was compiled and run from a Cargo workspace.
pub fn bundled_root_candidates() -> Vec<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Ok(root) = std::env::var(RESOURCE_ROOT_ENV) {
        if !root.is_empty() {
            candidates.push(PathBuf::from(root));
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        // macOS app bundle: the executable sits at Contents/MacOS/<bin>; Resources
        // is the sibling of MacOS inside Contents.
        if let Some(parent) = exe.parent() {
            if parent.file_name().is_some_and(|n| n == "MacOS") {
                if let Some(contents) = parent.parent() {
                    candidates.push(contents.join("Resources"));
                }
            }
        }

        // Portable archive layout: assets/ sits beside the executable.
        if let Some(exe_dir) = exe.parent() {
            candidates.push(exe_dir.to_path_buf());
        }
    }

    // Development / cargo-run layout: assets/ is at the workspace root.
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")));

    candidates
}

/// Resolve the bundled-resource root.
///
/// Returns the first candidate from [`bundled_root_candidates`] whose `assets/`
/// subdirectory exists on disk, or `None` if no candidate qualifies.
pub fn resolve_bundled_root() -> Option<PathBuf> {
    resolve_bundled_root_from(bundled_root_candidates(), Path::is_dir)
}

/// Pure resolution helper used by tests.
///
/// Accepts a pre-computed candidate list and a predicate that checks whether a
/// given path is a directory. Using a predicate allows tests to inject a fake
/// filesystem without mutating global process state.
pub(crate) fn resolve_bundled_root_from(
    candidates: impl IntoIterator<Item = PathBuf>,
    is_dir: impl Fn(&Path) -> bool,
) -> Option<PathBuf> {
    candidates
        .into_iter()
        .find(|root| is_dir(&root.join("assets")))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn resolve(pairs: &[(&str, bool)]) -> Option<PathBuf> {
        let candidates = pairs.iter().map(|(p, _)| PathBuf::from(p));
        resolve_bundled_root_from(candidates, |path| {
            pairs.iter().any(|(p, has_assets)| {
                *has_assets && PathBuf::from(p).join("assets") == path
            })
        })
    }

    #[test]
    fn picks_first_candidate_with_assets() {
        let result = resolve(&[
            ("/no-assets", false),
            ("/has-assets", true),
            ("/also-has", true),
        ]);
        assert_eq!(result, Some(PathBuf::from("/has-assets")));
    }

    #[test]
    fn returns_none_when_no_candidate_has_assets() {
        let result = resolve(&[("/a", false), ("/b", false)]);
        assert!(result.is_none());
    }

    #[test]
    fn returns_none_for_empty_candidates() {
        let result = resolve_bundled_root_from([], |_| true);
        assert!(result.is_none());
    }

    #[test]
    fn first_candidate_wins_over_later_ones() {
        let result = resolve(&[("/first", true), ("/second", true)]);
        assert_eq!(result, Some(PathBuf::from("/first")));
    }

    #[test]
    fn candidates_list_is_non_empty_and_ends_with_manifest_dir() {
        let candidates = bundled_root_candidates();
        assert!(!candidates.is_empty(), "no candidates returned");
        assert_eq!(
            candidates.last().unwrap(),
            &PathBuf::from(env!("CARGO_MANIFEST_DIR")),
            "last candidate should always be the manifest directory"
        );
    }

    #[test]
    fn resolve_bundled_root_finds_workspace_in_dev() {
        // In a `cargo test` run the manifest dir's assets/ directory exists.
        let root = resolve_bundled_root();
        assert!(
            root.is_some(),
            "resolve_bundled_root returned None; expected to find assets/ under \
             one of: {:#?}",
            bundled_root_candidates()
        );
        assert!(
            root.unwrap().join("assets").is_dir(),
            "resolved root does not contain assets/"
        );
    }

    /// Ensure resolution is independent of the process working directory by
    /// supplying arbitrary fake paths as candidates.
    #[test]
    fn resolution_ignores_cwd() {
        // Use absolute paths that cannot exist on disk — the predicate below
        // simulates the filesystem independently of the real CWD.
        let fake_root = PathBuf::from("/definitely/not/a/real/path/42");
        let result = resolve_bundled_root_from(
            [fake_root.clone()],
            |path| path == fake_root.join("assets"),
        );
        assert_eq!(result, Some(fake_root));
    }
}
