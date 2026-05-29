//! Top-level tab viewer: each tab represents a document entity, and
//! renders the document's nested dock area in the tab body. The tab
//! body has a playback toolbar (Play/Pause/Restart/Respawn) above the
//! inner DockArea. The toolbar lives at the document-tab level (not
//! inside a panel) because playback state is per-effect, not per-view.

use std::collections::HashMap;

use bevy::prelude::*;
use bevy_egui::egui;
use bevy_hanabi::EffectAsset;
use egui_dock::TabViewer;

use crate::document::ViewportSizeRequests;
use crate::edits::EditRequest;
use crate::playback::PlaybackCommand;

use super::{panels, DocTabState};

pub struct DocumentTabViewer<'we, 'wp, 'a> {
    pub tab_states: &'a mut HashMap<Entity, DocTabState>,
    pub viewport_textures: &'a HashMap<(Entity, usize), egui::TextureId>,
    pub size_requests: &'a mut ViewportSizeRequests,
    pub edits: &'a mut bevy::ecs::message::MessageWriter<'we, EditRequest>,
    pub playback: &'a mut bevy::ecs::message::MessageWriter<'wp, PlaybackCommand>,
    pub effects: &'a Assets<EffectAsset>,
}

impl<'we, 'wp, 'a> TabViewer for DocumentTabViewer<'we, 'wp, 'a> {
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

        draw_playback_toolbar(ui, doc_entity, &mut state.playing, self.playback);
        ui.separator();

        let DocTabState {
            dock,
            effect,
            selected_modifier,
            ..
        } = state;
        let mut inner_viewer = panels::PanelTabViewer {
            doc_entity,
            viewport_textures: self.viewport_textures,
            size_requests: &mut *self.size_requests,
            edits: self.edits,
            effects: self.effects,
            effect_handle: effect,
            selected_modifier,
        };

        egui_dock::DockArea::new(dock)
            .id(egui::Id::new(("inner-dock", doc_entity)))
            .style(egui_dock::Style::from_egui(ui.style()))
            .show_inside(ui, &mut inner_viewer);
    }
}

fn draw_playback_toolbar(
    ui: &mut egui::Ui,
    doc: Entity,
    playing: &mut bool,
    playback: &mut bevy::ecs::message::MessageWriter<PlaybackCommand>,
) {
    ui.horizontal(|ui| {
        let label = if *playing { "⏸ Pause" } else { "▶ Play" };
        if ui.button(label).clicked() {
            // Play/pause is state, not an action — mutate it directly.
            // The PlaybackState component is updated by the outer
            // draw_editor_ui after the dock pass closes.
            *playing = !*playing;
        }
        if ui.button("↻ Restart").clicked() {
            playback.write(PlaybackCommand::Restart(doc));
        }
        ui.separator();
        if ui
            .button("⟲ Respawn")
            .on_hover_text(
                "Despawn and recreate the ParticleEffect entity. Use if the \
                 preview doesn't reflect an asset change.",
            )
            .clicked()
        {
            playback.write(PlaybackCommand::Respawn(doc));
        }
    });
}
