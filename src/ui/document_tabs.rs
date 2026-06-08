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
use crate::document::{DocumentContent, DocumentUi, ViewportSizeRequests};
use crate::edits::EditRequest;
use crate::playback::{PlaybackCommand, PlaybackState};

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
    /// Hanabi's per-effect baked WGSL is uploaded into `Assets<Shader>`
    /// by its `compile_effects` system. The Shaders panel reads them
    /// back by path (`hanabi/{name}_{phase}_{hash}.wgsl`).
    pub shaders: Res<'w, Assets<Shader>>,
    /// Source of truth for the set of known modifier types; read by
    /// the Effect panel's Add menu.
    pub type_registry: Res<'w, AppTypeRegistry>,
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
                draw_playback_toolbar(ui, doc_entity, &mut playback.playing, self.playback);
            });
        // 6 px gutter painted in the same `extreme_bg_color` as the panel
        // separators, so the toolbar visually detaches from the inner dock.
        ui.allocate_space(egui::vec2(ui.available_width(), 6.0));

        // Field-split-borrow: dock for the inner DockArea, the rest
        // for the inner viewer.
        let DocumentUi {
            dock,
            selected_modifier,
            graph_view,
        } = &mut *ui_state;
        let mut inner_viewer = panels::PanelTabViewer {
            doc_entity,
            viewport_textures: self.viewport_textures,
            size_requests: &mut *self.size_requests,
            edits: self.edits,
            cam_msgs: self.cam_msgs,
            effects: &self.data.effects,
            shaders: &self.data.shaders,
            effect_handle: content.effect(),
            graph: content.graph(),
            selected_modifier,
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
