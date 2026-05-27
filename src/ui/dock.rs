use bevy::prelude::*;
use egui_dock::{DockState, NodeIndex};

/// Identifies the content of each dockable tab.
#[derive(Debug, Clone, PartialEq)]
pub enum EditorTab {
    /// A 3D render viewport (multiple allowed).
    Viewport(usize),
    /// List of loaded / in-project effects.
    EffectList,
    /// Property editor for the currently selected effect.
    Properties,
}

/// Bevy resource that owns the egui_dock layout state.
#[derive(Resource)]
pub struct EditorUiState {
    pub dock_state: DockState<EditorTab>,
}

impl Default for EditorUiState {
    fn default() -> Self {
        // Root: Viewport 0.
        let mut dock_state = DockState::new(vec![EditorTab::Viewport(0)]);
        let surface = dock_state.main_surface_mut();

        // Left panel: effect list.
        let [_left, center] =
            surface.split_left(NodeIndex::root(), 0.2, vec![EditorTab::EffectList]);

        // Right panel: properties (rightmost 25%).
        let [center, _right] =
            surface.split_right(center, 0.75, vec![EditorTab::Properties]);

        // Split the center horizontally: Viewport 0 on left, Viewport 1 on right.
        surface.split_right(center, 0.5, vec![EditorTab::Viewport(1)]);

        Self { dock_state }
    }
}

impl EditorUiState {
    pub fn reset_layout(&mut self) {
        *self = Self::default();
    }
}
