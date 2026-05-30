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

mod debug;
mod outline;
mod properties;
mod properties_section;
mod viewport;

pub struct PanelTabViewer<'w, 'wc, 'a, 'cw, 'cs> {
    pub doc_entity: Entity,
    pub viewport_textures: &'a HashMap<(Entity, usize), egui::TextureId>,
    pub size_requests: &'a mut ViewportSizeRequests,
    pub edits: &'a mut bevy::ecs::message::MessageWriter<'w, EditRequest>,
    pub cam_msgs: &'a mut bevy::ecs::message::MessageWriter<'wc, CameraControlMessage>,
    pub effects: &'a Assets<EffectAsset>,
    pub shaders: &'a Assets<Shader>,
    pub effect_handle: &'a Handle<EffectAsset>,
    pub selected_modifier: &'a mut Option<ModifierSelection>,
    /// Read-only ECS query for camera lookup by `(parent doc, viewport
    /// index)`. The viewport panel iterates this directly — no
    /// intermediate snapshot resource.
    pub cameras: &'a Query<
        'cw,
        'cs,
        (
            &'static crate::document::ViewportCamera,
            &'static ChildOf,
        ),
    >,
}

impl<'w, 'wc, 'a, 'cw, 'cs> TabViewer for PanelTabViewer<'w, 'wc, 'a, 'cw, 'cs> {
    type Tab = PanelKind;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        match tab {
            PanelKind::Viewport(i) => format!("Viewport {}", i).into(),
            PanelKind::Details => "Details".into(),
            PanelKind::Effect => "Effect".into(),
            PanelKind::Properties => "Properties".into(),
            PanelKind::Debug => "Debug".into(),
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
                self.effects,
                self.effect_handle,
                *self.selected_modifier,
                self.edits,
            ),
            PanelKind::Effect => outline::show(
                ui,
                self.doc_entity,
                self.effects,
                self.effect_handle,
                self.selected_modifier,
                self.edits,
            ),
            PanelKind::Properties => properties_section::show_panel(
                ui,
                self.doc_entity,
                self.effects,
                self.effect_handle,
                self.edits,
            ),
            PanelKind::Debug => {
                debug::show(ui, self.effects, self.shaders, self.effect_handle)
            }
        }
    }
}
