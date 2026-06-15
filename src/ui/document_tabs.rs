//! Top-level tab viewer: each tab represents a document entity, and
//! renders the document's nested dock area in the tab body. The tab
//! body has a playback toolbar (Play/Pause/Restart/Respawn) above the
//! inner DockArea. The toolbar lives at the document-tab level (not
//! inside a panel) because playback state is per-effect, not per-view.

use std::collections::HashMap;

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy::shader::Shader;
use bevy_egui::egui;
use bevy_hanabi::EffectAsset;
use egui_dock::TabViewer;

use super::panels;
use crate::app_commands::AppCommand;
use crate::document::{DocumentContent, DocumentUi, ViewportSizeRequests};
use crate::edits::EditRequest;
use crate::playback::{PlaybackCommand, PlaybackState};
use crate::plugins::camera_control::CameraControlMessage;

/// All ECS data the outer tab viewer needs from the system.
/// `#[derive(SystemParam)]` lets us pass this as a single argument to
/// the system without manually threading the `'w`/`'s` lifetimes of
/// each query — Bevy generates the borrow conjunction for us. Bundling the
/// message writers here too means the whole borrow set shares one world
/// lifetime, so the viewer needs only a single `'w`.
#[derive(SystemParam)]
pub struct TabViewerData<'w, 's> {
    pub docs: Query<
        'w,
        's,
        (
            &'static DocumentContent,
            &'static mut DocumentUi,
            &'static mut PlaybackState,
            &'static crate::plugins::shader_errors::ShaderErrors,
        ),
    >,
    /// Used by the viewport gizmo to derive world basis vectors per camera.
    pub cameras: Query<'w, 's, (&'static crate::document::ViewportCamera, &'static ChildOf)>,
    pub effects: Res<'w, Assets<EffectAsset>>,
    /// Hanabi's per-effect baked WGSL is uploaded into `Assets<Shader>`
    /// by its `compile_effects` system. The Shaders panel reads them
    /// back by path (`hanabi/{name}_{phase}_{hash}.wgsl`).
    pub shaders: Res<'w, Assets<Shader>>,
    /// Source of truth for the set of known modifier types; read by
    /// the Effect panel's Add menu.
    pub type_registry: Res<'w, AppTypeRegistry>,
    pub edits: MessageWriter<'w, EditRequest>,
    pub playback: MessageWriter<'w, PlaybackCommand>,
    pub cam_msgs: MessageWriter<'w, CameraControlMessage>,
    pub app: MessageWriter<'w, AppCommand>,
}

/// Outer tab viewer. Each `title()` / `ui()` call acquires its own
/// short-lived per-tab borrow on `data.docs` and drops it before
/// returning, so successive tab renders don't conflict.
///
/// `'w`/`'s` are the world/state lifetimes Bevy gives every param of the owning
/// system; [`TabViewerData`] bundles all of them under one pair. `'a` is the
/// (shorter) borrow taken when the viewer is built inside the system body.
pub struct DocumentTabViewer<'a, 'w, 's> {
    pub data: &'a mut TabViewerData<'w, 's>,
    pub viewport_textures: &'a HashMap<(Entity, usize), egui::TextureId>,
    pub size_requests: &'a mut ViewportSizeRequests,
}

impl<'a, 'w, 's> TabViewer for DocumentTabViewer<'a, 'w, 's> {
    type Tab = Entity;

    /// Route the tab-bar close button through the app-command channel so the
    /// document entity is actually despawned. Returning `false` keeps the tab
    /// for now; `sync_document_tabs` removes it once the entity is gone. Without
    /// this, egui_dock would drop the tab from the dock while the entity lived
    /// on, and the tab would immediately reappear.
    fn on_close(&mut self, tab: &mut Self::Tab) -> egui_dock::tab_viewer::OnCloseResponse {
        self.data.app.write(AppCommand::CloseDocument(*tab));
        egui_dock::tab_viewer::OnCloseResponse::Ignore
    }

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        let Ok((content, _, _, errors)) = self.data.docs.get(*tab) else {
            return format!("[doc {:?}]", tab).into();
        };
        let dirty = if content.dirty() { "* " } else { "" };
        // Prefix with a warning glyph when any of this effect's shaders failed
        // to compile, so the error is noticeable even with the Shaders panel
        // hidden.
        if !errors.0.is_empty() {
            let mut text = egui::text::LayoutJob::default();
            text.append(
                &format!("{} ", crate::ui::icons::ICON_TRIANGLE_EXCLAMATION),
                0.0,
                egui::TextFormat {
                    color: egui::Color32::from_rgb(0xE5, 0x73, 0x73),
                    ..Default::default()
                },
            );
            text.append(&format!("{dirty}{}", content.name()), 0.0, Default::default());
            return text.into();
        }
        format!("{dirty}{}", content.name()).into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        let doc_entity = *tab;

        let Ok((content, mut ui_state, mut playback, errors)) = self.data.docs.get_mut(doc_entity)
        else {
            ui.label("(missing document)");
            return;
        };

        // Eliminate egui's default vertical item-spacing between the toolbar
        // frame, our 6 px gutter, and the inner dock — otherwise each
        // boundary adds ~3 px and the visible gap doubles.
        ui.spacing_mut().item_spacing.y = 0.0;

        // Toolbar gets its own mid-gray (panel_fill) background — the outer
        // document tab body is painted `extreme_bg_color` so the area around
        // the inner dock blends with the gutters, but we don't want the
        // toolbar to inherit that very dark color. `set_min_width` forces
        // the frame to span the tab's full width instead of hugging the
        // 3-button cluster in the middle.
        let panel_fill = ui.style().visuals.panel_fill;
        egui::Frame::new()
            .fill(panel_fill)
            .inner_margin(egui::Margin::symmetric(0, 7))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                draw_playback_toolbar(ui, doc_entity, &mut playback.playing, &mut self.data.playback);
            });
        // 6 px gutter painted in the same `extreme_bg_color` as the panel
        // separators, so the toolbar visually detaches from the inner dock.
        ui.allocate_space(egui::vec2(ui.available_width(), 6.0));

        // Field-split-borrow: dock for the inner DockArea, the rest
        // for the inner viewer.
        let DocumentUi {
            dock,
            graph_view,
        } = &mut *ui_state;
        let mut inner_viewer = panels::PanelTabViewer {
            doc_entity,
            viewport_textures: self.viewport_textures,
            size_requests: &mut *self.size_requests,
            edits: &mut self.data.edits,
            cam_msgs: &mut self.data.cam_msgs,
            effects: &self.data.effects,
            shaders: &self.data.shaders,
            shader_errors: &errors.0,
            effect_handle: content.effect(),
            graph: content.graph(),
            type_registry: &self.data.type_registry,
            cameras: &self.data.cameras,
            graph_view,
        };

        egui_dock::DockArea::new(dock)
            .id(egui::Id::new(("inner-dock", doc_entity)))
            .style(crate::ui::dock_style_for(ui.style()))
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
    use crate::ui::icons::{
        ICON_ARROWS_ROTATE, ICON_BACKWARD_STEP, ICON_PAUSE, ICON_PLAY, icon_button,
    };
    /// Side length, in points, of each square icon button.
    const BTN: f32 = 28.0;
    const GAP: f32 = 7.0;

    ui.horizontal(|ui| {
        // Zero egui's horizontal item-spacing so our explicit `GAP` is the
        // only space between buttons (otherwise default spacing.x ≈ 8 is
        // added on top of `GAP`).
        ui.spacing_mut().item_spacing.x = 0.0;
        // Center the 3-button cluster horizontally. Width = 3 buttons
        // + 2 gaps; subtract from available width and pad the lead.
        let cluster_w = 3.0 * BTN + 2.0 * GAP;
        let lead = ((ui.available_width() - cluster_w) * 0.5).max(0.0);
        ui.add_space(lead);

        if icon_button(ui, ICON_BACKWARD_STEP, BTN)
            .on_hover_text("Restart (rewind effect to t=0)")
            .clicked()
        {
            playback.write(PlaybackCommand::Restart(doc));
        }
        ui.add_space(GAP);

        let play_icon = if *playing { ICON_PAUSE } else { ICON_PLAY };
        let play_hover = if *playing { "Pause" } else { "Play" };
        if icon_button(ui, play_icon, BTN)
            .on_hover_text(play_hover)
            .clicked()
        {
            *playing = !*playing;
        }
        ui.add_space(GAP);

        if icon_button(ui, ICON_ARROWS_ROTATE, BTN)
            .on_hover_text(
                "Respawn (despawn and recreate the ParticleEffect entity; \
                 use if the preview doesn't reflect an asset change)",
            )
            .clicked()
        {
            playback.write(PlaybackCommand::Respawn(doc));
        }
    });
}
