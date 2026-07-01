//! Wires the editor systems and creates the initial demo document.

use bevy::{camera::visibility::RenderLayers, prelude::*};
use bevy_egui::{EguiGlobalSettings, EguiPrimaryContextPass, PrimaryEguiContext};

use crate::{
    app_commands::AppCommandPlugin,
    document::{
        ActiveDocument, DocumentRoot, DocumentViewports, RenderLayerPool, ViewportSizeRequests,
    },
    edits::{EditPlugin, EditSystems},
    playback::PlaybackPlugin,
    plugins::{
        reconcile::{TexturePlaceholder, reconcile_documents},
        viewport_resize::apply_viewport_resizes,
    },
    ui::draw_editor_ui,
};

pub struct EditorPlugin;

impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ActiveDocument>()
            .init_resource::<RenderLayerPool>()
            .init_resource::<DocumentViewports>()
            .init_resource::<ViewportSizeRequests>()
            .init_resource::<crate::ui::DocumentDock>()
            .init_resource::<TexturePlaceholder>()
            .insert_resource(crate::effect_library::load_recent_files())
            .insert_resource(crate::effect_library::ExampleLibrary(
                crate::effect_library::discover_examples(),
            ))
            .add_plugins(EditPlugin)
            .add_plugins(AppCommandPlugin)
            .add_plugins(PlaybackPlugin)
            .add_plugins(crate::proxy::ProxyPlugin)
            .add_plugins(crate::plugins::shader_errors::ShaderErrorPlugin)
            .add_plugins(crate::modifier_registry::ModifierRegistryPlugin)
            .add_plugins(hanabi_effect_graph::EffectGraphPlugin)
            .add_plugins(crate::plugins::camera_control::CameraControlPlugin)
            .add_plugins(crate::thumbnail::ThumbnailPlugin)
            .add_systems(
                Startup,
                (configure_egui, setup_primary_camera, create_document_root).chain(),
            )
            // Font registration: must run *after* `setup_primary_camera`
            // has spawned the `PrimaryEguiContext` entity (whose
            // required `EguiContext` component holds the egui context
            // we need to `set_fonts` on). With
            // `auto_create_primary_context = false`, bevy_egui's
            // `EguiStartupSet::InitContexts` system is gated off and
            // `.after(EguiStartupSet::InitContexts)` provides no real
            // ordering — we must order against `setup_primary_camera`
            // directly. Runs once at startup, never per-frame.
            .add_systems(
                Startup,
                crate::ui::icons::install_fonts.after(setup_primary_camera),
            )
            .add_systems(
                Update,
                (
                    reconcile_documents.after(EditSystems),
                    apply_viewport_resizes.after(reconcile_documents),
                    crate::ui::handle_history_shortcuts,
                    crate::ui::handle_save_shortcut,
                ),
            )
            .add_systems(EguiPrimaryContextPass, draw_editor_ui);
    }
}

fn configure_egui(mut egui_global_settings: ResMut<EguiGlobalSettings>) {
    egui_global_settings.auto_create_primary_context = false;
}

fn setup_primary_camera(mut commands: Commands) {
    commands.spawn((
        PrimaryEguiContext,
        Camera3d::default(),
        Camera::default(),
        RenderLayers::none(),
    ));
}

fn create_document_root(mut commands: Commands) {
    let root = commands
        .spawn((
            Name::new("DocumentRoot"),
            Transform::default(),
            Visibility::default(),
        ))
        .id();
    commands.insert_resource(DocumentRoot(root));
}
