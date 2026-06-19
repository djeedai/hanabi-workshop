//! The three modifier lists of a `bevy_hanabi` effect.

/// Which of the three modifier lists a modifier lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ModifierGroup {
    Init,
    Update,
    Render,
}

impl ModifierGroup {
    pub fn label(self) -> &'static str {
        match self {
            ModifierGroup::Init => "Init",
            ModifierGroup::Update => "Update",
            ModifierGroup::Render => "Render",
        }
    }

    /// Lowercase tag for this group (`init` / `update` / `render`).
    ///
    /// Used in hanabi's baked shader path
    /// (`hanabi/{asset}_{init|update|render}_{hash}.wgsl`) and as a stable key
    /// for per-group UI state.
    pub fn suffix(self) -> &'static str {
        match self {
            ModifierGroup::Init => "init",
            ModifierGroup::Update => "update",
            ModifierGroup::Render => "render",
        }
    }
}
