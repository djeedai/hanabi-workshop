//! Inner panels: viewports, properties, outline. Rendered inside each
//! document tab's nested dock area.

use std::collections::HashMap;

use bevy::prelude::*;
use bevy::shader::Shader;
use bevy_egui::egui;
use bevy_hanabi::EffectAsset;
use egui_dock::TabViewer;

use crate::document::{ModifierSelection, PanelKind, ViewportSizeRequests};
use crate::edits::EditRequest;
use crate::plugins::camera_control::CameraControlMessage;

mod graph;
mod outline;
mod properties;
mod properties_section;
mod shaders;
mod viewport;
mod wgsl_highlight;

pub struct PanelTabViewer<'w, 'wc, 'a, 'cw, 'cs> {
    pub doc_entity: Entity,
    pub viewport_textures: &'a HashMap<(Entity, usize), egui::TextureId>,
    pub size_requests: &'a mut ViewportSizeRequests,
    pub edits: &'a mut bevy::ecs::message::MessageWriter<'w, EditRequest>,
    pub cam_msgs: &'a mut bevy::ecs::message::MessageWriter<'wc, CameraControlMessage>,
    pub effects: &'a Assets<EffectAsset>,
    pub shaders: &'a Assets<Shader>,
    pub effect_handle: &'a Handle<EffectAsset>,
    /// The document's canonical edit graph, read directly by the Graph panel.
    pub graph: &'a crate::effect_graph::model::EffectGraph,
    pub selected_modifier: &'a mut Option<ModifierSelection>,
    pub type_registry: &'a AppTypeRegistry,
    /// Per-document node-graph view state (pan/zoom/positions/selection).
    pub graph_view: &'a mut crate::ui::widgets::node_graph::GraphView,
    /// Read-only ECS query for camera lookup by `(parent doc, viewport
    /// index)`. The viewport panel iterates this directly — no
    /// intermediate snapshot resource.
    pub cameras: &'a Query<'cw, 'cs, (&'static crate::document::ViewportCamera, &'static ChildOf)>,
}

impl<'w, 'wc, 'a, 'cw, 'cs> TabViewer for PanelTabViewer<'w, 'wc, 'a, 'cw, 'cs> {
    type Tab = PanelKind;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        match tab {
            PanelKind::Viewport(i) => format!("Viewport {}", i).into(),
            PanelKind::Details => "Details".into(),
            PanelKind::Effect => "Effect".into(),
            PanelKind::Properties => "Properties".into(),
            PanelKind::Shaders => "Shaders".into(),
            PanelKind::Graph => "Graph".into(),
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        match tab {
            PanelKind::Viewport(i) => {
                viewport::show(
                    ui,
                    self.doc_entity,
                    *i,
                    self.viewport_textures,
                    self.size_requests,
                    self.cam_msgs,
                    self.cameras,
                );
            }
            PanelKind::Details => properties::show(
                ui,
                self.doc_entity,
                self.graph,
                *self.selected_modifier,
                self.edits,
            ),
            PanelKind::Effect => outline::show(ui, self.effects, self.effect_handle),
            PanelKind::Properties => properties_section::show_panel(
                ui,
                self.doc_entity,
                self.graph,
                self.edits,
            ),
            PanelKind::Shaders => shaders::show(ui, self.effects, self.shaders, self.effect_handle),
            PanelKind::Graph => graph::show(
                ui,
                self.doc_entity,
                self.graph,
                self.effects,
                self.effect_handle,
                self.type_registry,
                self.edits,
                self.graph_view,
            ),
        }
    }

    /// Drop the tab-body inner margin for viewport panels so the 3D render
    /// fills the panel edge-to-edge; other panels keep the default padding
    /// so text content doesn't kiss the tab borders.
    fn tab_style_override(
        &self,
        tab: &Self::Tab,
        global_style: &egui_dock::TabStyle,
    ) -> Option<egui_dock::TabStyle> {
        if matches!(tab, PanelKind::Viewport(_) | PanelKind::Graph) {
            let mut s = global_style.clone();
            s.tab_body.inner_margin = egui::Margin::ZERO;
            Some(s)
        } else {
            None
        }
    }
}
