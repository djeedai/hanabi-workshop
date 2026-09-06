//! Inner panels: viewports, properties, outline.
//!
//! Rendered inside each document tab's nested dock area.

use std::collections::HashMap;

use bevy::{prelude::*, shader::Shader};
use bevy_egui::egui;
use bevy_hanabi::EffectAsset;
use egui_dock::TabViewer;

use crate::{
    document::{PanelKind, ViewportSizeRequests},
    edits::EditRequest,
    effect_graph::model::{EffectGraph, EmitterId},
    plugins::camera_control::CameraControlMessage,
};

mod assets;
mod graph;
mod material;
mod outline;
mod properties_section;
mod shaders;
mod value_edit;
mod viewport;
mod wgsl_highlight;

pub struct PanelTabViewer<'w, 'wc, 'a, 'cw, 'cs> {
    pub doc_entity: Entity,
    pub viewport_textures: &'a HashMap<(Entity, usize), egui::TextureId>,
    pub size_requests: &'a mut ViewportSizeRequests,
    pub edits: &'a mut bevy::ecs::message::MessageWriter<'w, EditRequest>,
    pub live_values: &'a mut bevy::ecs::message::MessageWriter<'w, crate::proxy::LiveValueEdit>,
    pub cam_msgs: &'a mut bevy::ecs::message::MessageWriter<'wc, CameraControlMessage>,
    pub emitters: &'a Assets<EffectAsset>,
    pub images: &'a Assets<Image>,
    pub shaders: &'a Assets<Shader>,
    /// The shaders hanabi compiled for the active emitter, read straight from
    /// its [`bevy_hanabi::CompiledParticleEffect`]. `None` until that emitter
    /// has been spawned and compiled at least once.
    pub emitter_shaders: Option<&'a bevy_hanabi::EffectShaders>,
    /// Shader compile errors for the active emitter only — pre-filtered by the
    /// caller from the document's full
    /// [`crate::plugins::shader_errors::ShaderErrors`].
    pub shader_errors: &'a [crate::plugins::shader_errors::ShaderCompileError],
    /// The active emitter's canonical preview asset handle. `None` until that
    /// emitter has baked successfully at least once.
    pub emitter_handle: Option<&'a Handle<EffectAsset>>,
    /// The document's canonical edit graph: every emitter pipeline plus the
    /// spawn source contexts and topology links between them.
    pub effect_graph: &'a EffectGraph,
    /// Failures from the latest strict bake of the authored graph.
    pub bake_errors: &'a [crate::effect_graph::bake::EffectBakeError],
    /// Emitter the Emitter/Properties/Material/Shaders/Graph panels
    /// currently operate on. Updated in place from the Graph panel's
    /// most-recently-interacted-emitter tracking (see [`graph::show`]).
    pub active_emitter: &'a mut EmitterId,
    pub type_registry: &'a AppTypeRegistry,
    /// Per-document node-graph view state (pan/zoom/positions/selection).
    pub graph_view: &'a mut hanabi_node_graph::GraphView,
    /// Modifier node whose contextual help should appear in the viewport.
    pub modifier_gizmo_node: &'a mut Option<crate::effect_graph::model::NodeId>,
    /// Last render frame in which the Graph panel refreshed the target.
    pub modifier_gizmo_frame: &'a mut u32,
    /// Current render frame, used to reject stale targets.
    pub frame_count: u32,
    /// Whether the Graph panel was visible in this dock pass.
    pub graph_was_drawn: bool,
    /// Whether the shared horizontal grid is visible in this document's
    /// viewports.
    pub show_viewport_grid: &'a mut bool,
    /// Read-only ECS query for camera lookup by `(parent doc, viewport
    /// index)`. The viewport panel iterates this directly — no
    /// intermediate snapshot resource.
    pub cameras: &'a Query<'cw, 'cs, (&'static crate::document::ViewportCamera, &'static ChildOf)>,
    /// Native file dialogs, so the Material panel can pop an image picker.
    pub pending_dialogs: &'a mut crate::app_commands::PendingFileDialogs,
    pub texture_catalog: &'a crate::asset_library::TextureCatalog,
    pub texture_settings: &'a crate::asset_library::TextureLibrarySettings,
    pub texture_previews: &'a mut crate::texture_preview::TexturePreviewCache,
    pub asset_server: &'a AssetServer,
    pub texture_library:
        &'a mut bevy::ecs::message::MessageWriter<'w, crate::asset_library::TextureLibraryCommand>,
}

impl<'w, 'wc, 'a, 'cw, 'cs> TabViewer for PanelTabViewer<'w, 'wc, 'a, 'cw, 'cs> {
    type Tab = PanelKind;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        let icon = panel_icon(tab);
        match tab {
            PanelKind::Viewport(i) => format!("{icon}  Viewport {i}").into(),
            PanelKind::Emitter => format!("{icon}  Emitter").into(),
            PanelKind::Properties => format!("{icon}  Properties").into(),
            PanelKind::Material => format!("{icon}  Material").into(),
            PanelKind::Assets => format!("{icon}  Assets").into(),
            PanelKind::Shaders => format!("{icon}  Shaders").into(),
            PanelKind::Graph => format!("{icon}  Graph").into(),
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
                    self.show_viewport_grid,
                );
            }
            PanelKind::Emitter => outline::show(
                ui,
                self.doc_entity,
                self.effect_graph,
                *self.active_emitter,
                self.emitters,
                self.emitter_handle,
                self.edits,
            ),
            PanelKind::Properties => properties_section::show_panel(
                ui,
                self.doc_entity,
                self.effect_graph,
                *self.active_emitter,
                self.edits,
            ),
            PanelKind::Material => material::show_panel(
                ui,
                self.doc_entity,
                self.effect_graph,
                *self.active_emitter,
                self.edits,
            ),
            PanelKind::Assets => assets::show(
                ui,
                self.texture_catalog,
                self.texture_settings,
                self.texture_previews,
                self.asset_server,
                self.texture_library,
                self.pending_dialogs,
            ),
            PanelKind::Shaders => shaders::show(
                ui,
                self.emitters,
                self.shaders,
                self.emitter_handle,
                self.emitter_shaders,
                self.shader_errors,
            ),
            PanelKind::Graph => {
                self.graph_was_drawn = true;
                if let Some(new_active) = graph::show(
                    ui,
                    self.doc_entity,
                    self.effect_graph,
                    self.bake_errors,
                    *self.active_emitter,
                    self.emitters,
                    self.emitter_handle,
                    self.type_registry,
                    self.edits,
                    self.live_values,
                    self.pending_dialogs,
                    self.texture_catalog,
                    self.texture_settings,
                    self.texture_previews,
                    self.asset_server,
                    self.images,
                    self.texture_library,
                    self.graph_view,
                    self.modifier_gizmo_node,
                    self.modifier_gizmo_frame,
                    self.frame_count,
                ) {
                    *self.active_emitter = new_active;
                }
            }
        }
    }

    /// Drop the tab-body inner margin for viewport panels.
    ///
    /// The 3D render fills the panel edge-to-edge; other panels keep the
    /// default padding so text content doesn't kiss the tab borders.
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

/// Font Awesome glyph shown on a panel's tab and in the View menu.
///
/// Centralizes the panel-to-icon mapping so tab titles and the menu stay in
/// sync.
pub(crate) fn panel_icon(panel: &PanelKind) -> char {
    use crate::ui::icons::{
        ICON_CIRCLE_NODES, ICON_CODE, ICON_CUBE, ICON_FOLDER_TREE, ICON_IMAGES, ICON_SLIDERS,
        ICON_SPRAY_CAN_SPARKLES,
    };
    match panel {
        PanelKind::Viewport(_) => ICON_CUBE,
        PanelKind::Emitter => ICON_SPRAY_CAN_SPARKLES,
        PanelKind::Properties => ICON_SLIDERS,
        PanelKind::Material => ICON_IMAGES,
        PanelKind::Assets => ICON_FOLDER_TREE,
        PanelKind::Shaders => ICON_CODE,
        PanelKind::Graph => ICON_CIRCLE_NODES,
    }
}

/// Render a collapsible section with a full-width, hover-highlighted header.
///
/// The header bar spans the panel width with a slightly lighter background that
/// brightens on hover and toggles the body when clicked. `id_salt` keeps the
/// open/closed state stable even when `label` text changes between frames.
pub(super) fn collapsing<R>(
    ui: &mut egui::Ui,
    id_salt: impl std::hash::Hash,
    label: &str,
    add: impl FnOnce(&mut egui::Ui) -> R,
) {
    use egui::collapsing_header::CollapsingState;

    let id = ui.make_persistent_id(id_salt);
    let mut state = CollapsingState::load_with_default_open(ui.ctx(), id, true);
    let openness = state.openness(ui.ctx());

    // Full-width clickable header bar.
    let height = ui
        .text_style_height(&egui::TextStyle::Button)
        .max(ui.spacing().interact_size.y);
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), height),
        egui::Sense::click(),
    );
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    if resp.clicked() {
        state.toggle(ui);
    }

    let visuals = ui.style().interact(&resp);
    ui.painter()
        .rect_filled(rect, visuals.corner_radius, visuals.weak_bg_fill);

    // Disclosure arrow, vertically centred at the left of the bar.
    let icon_size = ui.spacing().icon_width;
    let icon_rect = egui::Rect::from_center_size(
        egui::pos2(rect.left() + 4.0 + icon_size * 0.5, rect.center().y),
        egui::vec2(icon_size, icon_size),
    );
    paint_arrow(ui, icon_rect, openness, visuals.fg_stroke.color);

    ui.painter().text(
        egui::pos2(icon_rect.right() + 4.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        label,
        egui::TextStyle::Button.resolve(ui.style()),
        visuals.fg_stroke.color,
    );

    state.show_body_unindented(ui, add);
    state.store(ui.ctx());
}

/// Paint a triangular disclosure arrow that rotates with `openness`.
///
/// `openness` is `0.0` when collapsed (arrow points right) and `1.0` when fully
/// open (arrow points down), with intermediate values during the animation.
fn paint_arrow(ui: &egui::Ui, rect: egui::Rect, openness: f32, color: egui::Color32) {
    use std::f32::consts::TAU;
    let rect = egui::Rect::from_center_size(rect.center(), rect.size() * 0.75);
    let mut points = vec![rect.left_top(), rect.right_top(), rect.center_bottom()];
    let rotation =
        egui::emath::Rot2::from_angle(egui::remap(openness, 0.0..=1.0, -TAU / 4.0..=0.0));
    for p in &mut points {
        *p = rect.center() + rotation * (*p - rect.center());
    }
    ui.painter().add(egui::Shape::convex_polygon(
        points,
        color,
        egui::Stroke::NONE,
    ));
}
