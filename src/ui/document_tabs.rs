//! Top-level tab viewer: one tab per document entity.
//!
//! Renders the document's nested dock area in the tab body. The tab body has a
//! playback toolbar (Play/Pause/Restart/Respawn) above the inner DockArea. The
//! toolbar lives at the document-tab level (not inside a panel) because
//! playback state is per-effect, not per-view.

use std::collections::HashMap;

use bevy::{ecs::system::SystemParam, prelude::*, shader::Shader};
use bevy_egui::egui;
use bevy_hanabi::EffectAsset;
use egui_dock::TabViewer;

use super::panels;
use crate::{
    app_commands::AppCommand,
    document::{DocumentContent, DocumentUi, ViewportSizeRequests},
    edits::EditRequest,
    playback::{PlaybackCommand, PlaybackState},
    plugins::camera_control::CameraControlMessage,
};

/// All ECS data the outer tab viewer needs from the system.
///
/// `#[derive(SystemParam)]` lets us pass this as a single argument to the
/// system without manually threading the `'w`/`'s` lifetimes of each query —
/// Bevy generates the borrow conjunction for us. Bundling the message writers
/// here too means the whole borrow set shares one world lifetime, so the viewer
/// needs only a single `'w`.
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
    /// Each document's spawned [`bevy_hanabi::ParticleEffect`] entity, used to
    /// read back the exact shaders hanabi compiled for that effect. The entity
    /// is a child of the document's [`crate::document::DocumentSceneRoot`].
    pub compiled_effects: Query<
        'w,
        's,
        (
            &'static ChildOf,
            &'static bevy_hanabi::CompiledParticleEffect,
        ),
    >,
    /// Resolves a scene root back to its owning document entity.
    pub scene_roots: Query<'w, 's, &'static ChildOf, With<crate::document::DocumentSceneRoot>>,
    pub effects: Res<'w, Assets<EffectAsset>>,
    /// Hanabi's per-effect baked WGSL is uploaded into `Assets<Shader>` by its
    /// `compile_effects` system; the Shaders panel reads the exact handles for
    /// each effect via
    /// [`bevy_hanabi::CompiledParticleEffect::get_configured_shaders`].
    pub shaders: Res<'w, Assets<Shader>>,
    /// Source of truth for the set of known modifier types; read by
    /// the Effect panel's Add menu.
    pub type_registry: Res<'w, AppTypeRegistry>,
    pub edits: MessageWriter<'w, EditRequest>,
    pub live_values: MessageWriter<'w, crate::proxy::LiveValueEdit>,
    pub playback: MessageWriter<'w, PlaybackCommand>,
    pub cam_msgs: MessageWriter<'w, CameraControlMessage>,
    pub app: MessageWriter<'w, AppCommand>,
    /// Requests thumbnail generation for effects shown in the Home browser.
    pub thumb_requests: MessageWriter<'w, crate::thumbnail::ThumbnailRequest>,
    /// Requests a full clear of the thumbnail cache from the Home browser.
    pub thumb_clear: MessageWriter<'w, crate::thumbnail::ClearThumbnailCache>,
    /// Bundled example effects listed by the Home tab's browser.
    pub examples: Res<'w, crate::effect_library::ExampleLibrary>,
    /// Recently opened/saved user files listed by the Home tab's browser.
    pub recents: Res<'w, crate::effect_library::RecentFiles>,
    pub texture_catalog: Res<'w, crate::asset_library::TextureCatalog>,
    pub texture_settings: Res<'w, crate::asset_library::TextureLibrarySettings>,
    pub texture_previews: ResMut<'w, crate::texture_preview::TexturePreviewCache>,
    pub asset_server: Res<'w, AssetServer>,
    pub texture_library: MessageWriter<'w, crate::asset_library::TextureLibraryCommand>,
    pub frame_count: Res<'w, bevy::diagnostic::FrameCount>,
}

/// Outer tab viewer.
///
/// Each `title()` / `ui()` call acquires its own short-lived per-tab borrow on
/// `data.docs` and drops it before returning, so successive tab renders don't
/// conflict.
///
/// `'w`/`'s` are the world/state lifetimes Bevy gives every param of the owning
/// system; [`TabViewerData`] bundles all of them under one pair. `'a` is the
/// (shorter) borrow taken when the viewer is built inside the system body.
pub struct DocumentTabViewer<'a, 'w, 's> {
    pub data: &'a mut TabViewerData<'w, 's>,
    pub viewport_textures: &'a HashMap<(Entity, usize), egui::TextureId>,
    /// Ready thumbnail textures for Home browser cards, keyed by effect path.
    pub thumbnail_textures: &'a crate::ui::home::ThumbnailTextures,
    pub size_requests: &'a mut ViewportSizeRequests,
    pub pending_dialogs: &'a mut crate::app_commands::PendingFileDialogs,
}

impl<'a, 'w, 's> TabViewer for DocumentTabViewer<'a, 'w, 's> {
    type Tab = crate::ui::OuterTab;

    /// The Home tab is not closable; documents route through the app-command
    /// channel.
    fn is_closeable(&self, tab: &Self::Tab) -> bool {
        tab.document().is_some()
    }

    /// Route the tab-bar close button through the app-command channel.
    ///
    /// So the document entity is actually despawned. Returning `Ignore` keeps
    /// the tab for now; `sync_document_tabs` removes it once the entity is
    /// gone. Without this, egui_dock would drop the tab from the dock while the
    /// entity lived on, and the tab would immediately reappear. The Home tab is
    /// never closed.
    fn on_close(&mut self, tab: &mut Self::Tab) -> egui_dock::tab_viewer::OnCloseResponse {
        if let Some(doc) = tab.document() {
            self.data.app.write(AppCommand::RequestCloseDocument(doc));
        }
        egui_dock::tab_viewer::OnCloseResponse::Ignore
    }

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        let Some(doc) = tab.document() else {
            return format!("{}  Home", crate::ui::icons::ICON_HOUSE).into();
        };
        let Ok((content, _, _, errors)) = self.data.docs.get(doc) else {
            return format!("[doc {:?}]", doc).into();
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
            text.append(
                &format!("{dirty}{}", content.name()),
                0.0,
                Default::default(),
            );
            return text.into();
        }
        format!("{dirty}{}", content.name()).into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        let Some(doc_entity) = tab.document() else {
            let recents = self.data.recents.entries();
            super::home::show(
                ui,
                &mut self.data.app,
                self.pending_dialogs,
                &self.data.examples.0,
                &recents,
                self.thumbnail_textures,
                &mut self.data.thumb_requests,
                &mut self.data.thumb_clear,
            );
            return;
        };

        // Resolve the shaders hanabi actually compiled for this document's
        // effect entity. Matching by entity (rather than by asset name)
        // sidesteps hanabi's source-keyed shader dedup, which can collapse two
        // documents with identical content onto a single shader named after
        // whichever compiled first.
        let effect_shaders = self
            .data
            .compiled_effects
            .iter()
            .find_map(|(child_of, compiled)| {
                let scene_root = self.data.scene_roots.get(child_of.parent()).ok()?;
                (scene_root.parent() == doc_entity)
                    .then(|| compiled.get_configured_shaders().cloned())
                    .flatten()
            });

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
                draw_playback_toolbar(
                    ui,
                    doc_entity,
                    &mut playback.playing,
                    &mut self.data.playback,
                );
            });
        // 6 px gutter painted in the same `extreme_bg_color` as the panel
        // separators, so the toolbar visually detaches from the inner dock.
        ui.allocate_space(egui::vec2(ui.available_width(), 6.0));

        // Field-split-borrow: dock for the inner DockArea, the rest
        // for the inner viewer.
        let DocumentUi {
            dock,
            graph_view,
            modifier_gizmo_node,
            modifier_gizmo_frame,
            show_viewport_grid,
        } = &mut *ui_state;
        let mut inner_viewer = panels::PanelTabViewer {
            doc_entity,
            viewport_textures: self.viewport_textures,
            size_requests: &mut *self.size_requests,
            edits: &mut self.data.edits,
            live_values: &mut self.data.live_values,
            cam_msgs: &mut self.data.cam_msgs,
            effects: &self.data.effects,
            shaders: &self.data.shaders,
            effect_shaders: effect_shaders.as_ref(),
            shader_errors: &errors.0,
            effect_handle: content.effect(),
            graph: content.graph(),
            type_registry: &self.data.type_registry,
            cameras: &self.data.cameras,
            graph_view,
            modifier_gizmo_node,
            modifier_gizmo_frame,
            frame_count: self.data.frame_count.0,
            graph_was_drawn: false,
            show_viewport_grid,
            pending_dialogs: &mut *self.pending_dialogs,
            texture_catalog: &self.data.texture_catalog,
            texture_settings: &self.data.texture_settings,
            texture_previews: &mut self.data.texture_previews,
            asset_server: &self.data.asset_server,
            texture_library: &mut self.data.texture_library,
        };

        egui_dock::DockArea::new(dock)
            .id(egui::Id::new(("inner-dock", doc_entity)))
            .style(crate::ui::dock_style_for(ui.style()))
            .show_leaf_collapse_buttons(false)
            .show_leaf_close_all_buttons(false)
            .show_inside(ui, &mut inner_viewer);
        if !inner_viewer.graph_was_drawn {
            *inner_viewer.modifier_gizmo_node = None;
            *inner_viewer.modifier_gizmo_frame = 0;
        }
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
