//! Top-level tab viewer: each tab represents a document entity, and renders
//! the document's nested dock area in the tab body.

use std::collections::HashMap;

use bevy::prelude::*;
use bevy_egui::egui;
use egui_dock::TabViewer;

use crate::document::ViewportSizeRequests;
use crate::edits::EditRequest;

use super::{panels, DocTabState};

pub struct DocumentTabViewer<'w, 'a> {
    pub tab_states: &'a mut HashMap<Entity, DocTabState>,
    pub viewport_textures: &'a HashMap<(Entity, usize), egui::TextureId>,
    pub size_requests: &'a mut ViewportSizeRequests,
    pub edits: &'a mut bevy::ecs::message::MessageWriter<'w, EditRequest>,
}

impl<'w, 'a> TabViewer for DocumentTabViewer<'w, 'a> {
    type Tab = Entity;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        let Some(state) = self.tab_states.get(tab) else {
            return format!("[doc {:?}]", tab).into();
        };
        let prefix = if state.dirty { "* " } else { "" };
        format!("{prefix}{}", state.name).into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        let doc_entity = *tab;
        let Some(state) = self.tab_states.get_mut(&doc_entity) else {
            ui.label("(missing document)");
            return;
        };

        let mut inner_viewer = panels::PanelTabViewer {
            doc_entity,
            viewport_textures: self.viewport_textures,
            size_requests: &mut *self.size_requests,
            edits: self.edits,
        };

        egui_dock::DockArea::new(&mut state.dock)
            .id(egui::Id::new(("inner-dock", doc_entity)))
            .style(egui_dock::Style::from_egui(ui.style()))
            .show_inside(ui, &mut inner_viewer);
    }
}
