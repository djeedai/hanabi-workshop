use bevy::camera::{visibility::RenderLayers, RenderTarget};
use bevy::prelude::*;
use bevy::render::render_resource::{
    Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
};
use bevy_egui::{
    EguiGlobalSettings, EguiPrimaryContextPass, EguiTextureHandle, EguiUserTextures,
    PrimaryEguiContext,
};

use crate::ui::{draw_editor_ui, EditorUiState};

pub struct EditorPlugin;

/// Handles of the textures each viewport camera renders into.
#[derive(Resource, Clone)]
pub struct ViewportImages(pub Vec<Handle<Image>>);

#[derive(Component)]
struct Spinner;

impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EditorUiState>()
            .add_systems(Startup, setup_scene)
            .add_systems(Update, rotate_spinner)
            .add_systems(EguiPrimaryContextPass, draw_editor_ui);
    }
}

fn setup_scene(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut egui_user_textures: ResMut<EguiUserTextures>,
    mut egui_global_settings: ResMut<EguiGlobalSettings>,
) {
    // We spawn the primary egui camera ourselves below.
    egui_global_settings.auto_create_primary_context = false;

    let size = Extent3d {
        width: 512,
        height: 512,
        ..default()
    };

    let make_target = |images: &mut Assets<Image>,
                           egui_user_textures: &mut EguiUserTextures|
     -> Handle<Image> {
        let mut image = Image {
            texture_descriptor: TextureDescriptor {
                label: None,
                size,
                dimension: TextureDimension::D2,
                format: TextureFormat::Bgra8UnormSrgb,
                mip_level_count: 1,
                sample_count: 1,
                usage: TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_DST
                    | TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            },
            ..default()
        };
        image.resize(size);
        let handle = images.add(image);
        egui_user_textures.add_image(EguiTextureHandle::Strong(handle.clone()));
        handle
    };

    let viewport_a = make_target(&mut images, &mut egui_user_textures);
    let viewport_b = make_target(&mut images, &mut egui_user_textures);
    commands.insert_resource(ViewportImages(vec![viewport_a.clone(), viewport_b.clone()]));

    // Shared scene content.
    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.6, 0.4, 0.0)),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.85, 0.45, 0.2),
            ..default()
        })),
        Transform::IDENTITY,
        Spinner,
    ));

    // Viewport A: front view, dark blue clear.
    commands.spawn((
        Camera3d::default(),
        Camera {
            order: -2,
            clear_color: ClearColorConfig::Custom(Color::srgb(0.08, 0.10, 0.16)),
            ..default()
        },
        RenderTarget::Image(viewport_a.into()),
        Transform::from_xyz(0.0, 0.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Viewport B: side/top view, dark red clear — visibly different from A.
    commands.spawn((
        Camera3d::default(),
        Camera {
            order: -1,
            clear_color: ClearColorConfig::Custom(Color::srgb(0.16, 0.08, 0.10)),
            ..default()
        },
        RenderTarget::Image(viewport_b.into()),
        Transform::from_xyz(3.0, 2.5, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Primary egui camera (renders to the window). egui paints over its output.
    commands.spawn((
        PrimaryEguiContext,
        Camera3d::default(),
        Camera::default(),
        // Don't render the scene into this camera — keep its output empty.
        RenderLayers::none(),
    ));
}

fn rotate_spinner(time: Res<Time>, mut q: Query<&mut Transform, With<Spinner>>) {
    for mut t in &mut q {
        t.rotate_y(time.delta_secs() * 0.8);
        t.rotate_x(time.delta_secs() * 0.3);
    }
}

