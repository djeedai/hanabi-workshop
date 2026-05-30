//! Wires the editor systems and creates the initial demo document.

use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy_egui::{EguiGlobalSettings, EguiPrimaryContextPass, PrimaryEguiContext};
use bevy_hanabi::EffectAsset;

use crate::app_commands::{AppCommandPlugin, spawn_document};
use crate::document::{
    ActiveDocument, DocumentRoot, DocumentViewports, RenderLayerPool, ViewportSizeRequests,
};
use crate::edits::{EditPlugin, EditSystems};
use crate::playback::PlaybackPlugin;
use crate::plugins::{reconcile::reconcile_documents, viewport_resize::apply_viewport_resizes};
use crate::ui::draw_editor_ui;

pub struct EditorPlugin;

impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ActiveDocument>()
            .init_resource::<RenderLayerPool>()
            .init_resource::<DocumentViewports>()
            .init_resource::<ViewportSizeRequests>()
            .init_resource::<crate::ui::DocumentDock>()
            .add_plugins(EditPlugin)
            .add_plugins(AppCommandPlugin)
            .add_plugins(PlaybackPlugin)
            .add_plugins(crate::proxy::ProxyPlugin)
            .add_plugins(crate::modifier_registry::ModifierRegistryPlugin)
            .add_plugins(crate::plugins::camera_control::CameraControlPlugin)
            .add_systems(
                Startup,
                (
                    configure_egui,
                    setup_primary_camera,
                    create_document_root,
                    seed_demo_document,
                )
                    .chain(),
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

fn seed_demo_document(
    mut commands: Commands,
    mut effect_assets: ResMut<Assets<EffectAsset>>,
    mut layer_pool: ResMut<RenderLayerPool>,
    mut active: ResMut<ActiveDocument>,
    root: Res<DocumentRoot>,
) {
    let first = spawn_document(
        &mut commands,
        &mut layer_pool,
        root.0,
        "Untitled".to_string(),
        None,
        effect_assets.add(crate::demo_effect::demo_effect()),
    );
    spawn_document(
        &mut commands,
        &mut layer_pool,
        root.0,
        "Second".to_string(),
        None,
        effect_assets.add(crate::demo_effect::demo_effect()),
    );
    active.0 = Some(first);
}
