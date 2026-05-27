//! Inner panels: viewports, properties, outline. Rendered inside each
//! document tab's nested dock area.

use std::collections::HashMap;

use bevy::prelude::*;
use bevy_egui::egui;
use egui_dock::TabViewer;

use crate::document::{PanelKind, ViewportSizeRequests};
use crate::edits::EditRequest;

mod outline;
mod properties;
mod viewport;

pub struct PanelTabViewer<'w, 'a> {
    pub doc_entity: Entity,
    pub viewport_textures: &'a HashMap<(Entity, usize), egui::TextureId>,
    pub size_requests: &'a mut ViewportSizeRequests,
    pub edits: &'a mut bevy::ecs::message::MessageWriter<'w, EditRequest>,
}

impl<'w, 'a> TabViewer for PanelTabViewer<'w, 'a> {
    type Tab = PanelKind;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        match tab {
            PanelKind::Viewport(i) => format!("Viewport {}", i).into(),
            PanelKind::Properties => "Properties".into(),
            PanelKind::Outline => "Outline".into(),
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
                );
            }
            PanelKind::Properties => properties::show(ui, self.doc_entity, self.edits),
            PanelKind::Outline => outline::show(ui, self.doc_entity),
        }
    }
}
