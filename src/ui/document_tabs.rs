//! Top-level tab viewer: each tab represents a document entity, and
//! renders the document's nested dock area in the tab body. The tab
//! body has a playback toolbar (Play/Pause/Restart/Respawn) above the
//! inner DockArea. The toolbar lives at the document-tab level (not
//! inside a panel) because playback state is per-effect, not per-view.

use std::collections::HashMap;

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy_egui::egui;
use bevy_hanabi::EffectAsset;
use egui_dock::TabViewer;

use crate::document::{DocumentContent, DocumentUi, ViewportSizeRequests};
use crate::edits::EditRequest;
use crate::playback::{PlaybackCommand, PlaybackState};

use super::panels;

/// All ECS data the outer tab viewer needs from the system.
/// `#[derive(SystemParam)]` lets us pass this as a single argument to
/// the system without manually threading the `'w`/`'s` lifetimes of
/// each query — Bevy generates the borrow conjunction for us.
#[derive(SystemParam)]
pub struct TabViewerData<'w, 's> {
    pub docs: Query<
        'w,
        's,
        (
            &'static DocumentContent,
            &'static mut DocumentUi,
            &'static mut PlaybackState,
        ),
    >,
    /// Used by the viewport gizmo to derive world basis vectors per camera.
    pub cameras: Query<'w, 's, (&'static crate::document::ViewportCamera, &'static ChildOf)>,
    pub effects: Res<'w, Assets<EffectAsset>>,
}

/// Outer tab viewer. Each `title()` / `ui()` call acquires its own
/// short-lived per-tab borrow on `data.docs` and drops it before
/// returning, so successive tab renders don't conflict.
pub struct DocumentTabViewer<'we, 'wp, 'wc, 'a, 'w, 's> {
    pub data: &'a mut TabViewerData<'w, 's>,
    pub viewport_textures: &'a HashMap<(Entity, usize), egui::TextureId>,
    pub size_requests: &'a mut ViewportSizeRequests,
    pub edits: &'a mut bevy::ecs::message::MessageWriter<'we, EditRequest>,
    pub playback: &'a mut bevy::ecs::message::MessageWriter<'wp, PlaybackCommand>,
    pub cam_msgs: &'a mut bevy::ecs::message::MessageWriter<
        'wc,
        crate::plugins::camera_control::CameraControlMessage,
    >,
}

impl<'we, 'wp, 'wc, 'a, 'w, 's> TabViewer for DocumentTabViewer<'we, 'wp, 'wc, 'a, 'w, 's> {
    type Tab = Entity;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        let Ok((content, _, _)) = self.data.docs.get(*tab) else {
            return format!("[doc {:?}]", tab).into();
        };
        let prefix = if content.dirty() { "* " } else { "" };
        format!("{prefix}{}", content.name()).into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        let doc_entity = *tab;

        let Ok((content, mut ui_state, mut playback)) = self.data.docs.get_mut(doc_entity) else {
            ui.label("(missing document)");
            return;
        };

        draw_playback_toolbar(ui, doc_entity, &mut playback.playing, self.playback);
        ui.separator();

        // Field-split-borrow: dock for the inner DockArea, the rest
        // for the inner viewer.
        let DocumentUi {
            dock,
            selected_modifier,
        } = &mut *ui_state;
        let mut inner_viewer = panels::PanelTabViewer {
            doc_entity,
            viewport_textures: self.viewport_textures,
            size_requests: &mut *self.size_requests,
            edits: self.edits,
            cam_msgs: self.cam_msgs,
            effects: &self.data.effects,
            effect_handle: content.effect(),
            selected_modifier,
            cameras: &self.data.cameras,
        };

        egui_dock::DockArea::new(dock)
            .id(egui::Id::new(("inner-dock", doc_entity)))
            .style(egui_dock::Style::from_egui(ui.style()))
            .show_leaf_collapse_buttons(false)
            .show_leaf_close_all_buttons(false)
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
