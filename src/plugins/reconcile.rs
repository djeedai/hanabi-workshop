//! Per-document scene reconciliation (ECS form).
//!
//! Documents are entities. Each document entity has children:
//! a scene root (with light + placeholder mesh), and N viewport cameras
//! each rendering into its own `Image`. Hierarchical despawn cleans
//! everything up when a document entity is despawned.

use std::collections::HashSet;

use bevy::{
    camera::{RenderTarget, visibility::RenderLayers},
    prelude::*,
    render::render_resource::{
        Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    },
};
use bevy_egui::{EguiTextureHandle, EguiUserTextures};

use crate::{
    document::{
        DocumentContent, DocumentSceneRoot, DocumentUi, DocumentViewports, PanelKind,
        ViewportCamera, ViewportSlots,
    },
    proxy::ProxyEffect,
};

/// Ensure each document's child scene and viewport cameras match its dock.
///
/// Walks every document, reconciling against what `DocumentUi.dock` requests.
/// Also rebuilds the `DocumentViewports` cache for the UI to consume.
pub fn reconcile_documents(
    mut commands: Commands,
    mut docs: Query<(
        Entity,
        &DocumentContent,
        &DocumentUi,
        Option<&ProxyEffect>,
        Option<&Children>,
    )>,
    viewport_cams: Query<(Entity, &ViewportCamera)>,
    scene_roots: Query<Entity, With<DocumentSceneRoot>>,
    mut viewports: ResMut<DocumentViewports>,
    mut images: ResMut<Assets<Image>>,
    mut egui_user_textures: ResMut<EguiUserTextures>,
) {
    // Rebuild the UI lookup from scratch each frame; cheap (few docs, few
    // viewports).
    viewports.by_doc.clear();

    for (doc_entity, content, ui, proxy, children) in docs.iter_mut() {
        let layer = RenderLayers::layer(content.render_layer());
        let slots = viewports.by_doc.entry(doc_entity).or_default();

        let child_list: Vec<Entity> = children.map(|c| c.iter().collect()).unwrap_or_default();

        // Scene root spawning is deferred until the proxy exists,
        // because the `ParticleEffect` we instantiate references the
        // proxy handle (not the canonical one). `ensure_proxy`
        // installs the component once the canonical asset has loaded.
        if let Some(proxy) = proxy {
            ensure_scene_root(
                &mut commands,
                doc_entity,
                proxy,
                &child_list,
                &scene_roots,
                &layer,
            );
        }

        reconcile_viewports(
            &mut commands,
            doc_entity,
            ui,
            &child_list,
            &viewport_cams,
            &layer,
            slots,
            &mut images,
            &mut egui_user_textures,
        );
    }
}

fn ensure_scene_root(
    commands: &mut Commands,
    doc_entity: Entity,
    proxy: &ProxyEffect,
    children: &[Entity],
    scene_roots: &Query<Entity, With<DocumentSceneRoot>>,
    layer: &RenderLayers,
) {
    let already = children.iter().any(|c| scene_roots.get(*c).is_ok());
    if already {
        return;
    }

    let effect_handle = proxy.handle.clone();
    let scene_root = commands
        .spawn((
            DocumentSceneRoot,
            Transform::default(),
            Visibility::default(),
        ))
        .with_children(|p| {
            p.spawn((
                DirectionalLight {
                    illuminance: 10_000.0,
                    ..default()
                },
                Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.6, 0.4, 0.0)),
                layer.clone(),
            ));
            p.spawn((
                bevy_hanabi::ParticleEffect::new(effect_handle),
                // Required when the (proxy) asset declares any properties:
                // hanabi's `update_properties_from_asset` only updates the
                // GPU-side property buffer for entities that already have an
                // `EffectProperties` component. Without this, our promoted
                // synthetic properties default to zero on the GPU and the
                // effect renders nothing. Cheap to always include.
                bevy_hanabi::EffectProperties::default(),
                Transform::IDENTITY,
                layer.clone(),
            ));
        })
        .id();
    commands.entity(doc_entity).add_child(scene_root);
}

fn reconcile_viewports(
    commands: &mut Commands,
    doc_entity: Entity,
    ui: &DocumentUi,
    children: &[Entity],
    viewport_cams: &Query<(Entity, &ViewportCamera)>,
    layer: &RenderLayers,
    slots: &mut ViewportSlots,
    images: &mut Assets<Image>,
    egui_user_textures: &mut EguiUserTextures,
) {
    let wanted: HashSet<usize> = ui
        .dock
        .iter_all_tabs()
        .filter_map(|(_, tab)| match tab {
            PanelKind::Viewport(i) => Some(*i),
            _ => None,
        })
        .collect();

    let mut have: HashSet<usize> = HashSet::new();
    for child in children {
        if let Ok((cam_entity, vp_cam)) = viewport_cams.get(*child) {
            if wanted.contains(&vp_cam.viewport_index) {
                have.insert(vp_cam.viewport_index);
                slots
                    .images
                    .insert(vp_cam.viewport_index, vp_cam.image.clone());
                slots.cameras.insert(vp_cam.viewport_index, cam_entity);
            } else {
                commands.entity(cam_entity).despawn();
            }
        }
    }

    for index in wanted.difference(&have) {
        let handle = make_render_target(images, egui_user_textures);
        let cam =
            spawn_viewport_camera(commands, doc_entity, *index, handle.clone(), layer.clone());
        commands.entity(doc_entity).add_child(cam);
        slots.images.insert(*index, handle);
        slots.cameras.insert(*index, cam);
    }
}

fn make_render_target(
    images: &mut Assets<Image>,
    egui_user_textures: &mut EguiUserTextures,
) -> Handle<Image> {
    let size = Extent3d {
        width: 512,
        height: 512,
        ..default()
    };
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
}

fn spawn_viewport_camera(
    commands: &mut Commands,
    _doc_entity: Entity,
    viewport_index: usize,
    image: Handle<Image>,
    layer: RenderLayers,
) -> Entity {
    let angle = viewport_index as f32 * std::f32::consts::FRAC_PI_3;
    // Initial orbit state: target at origin, ~26° above equator, 4.47 units out.
    let target = Vec3::ZERO;
    let distance = 4.47;
    let yaw = angle;
    let pitch = (2.0_f32 / distance).asin();
    let cam = ViewportCamera {
        viewport_index,
        image: image.clone(),
        target,
        yaw,
        pitch,
        distance,
    };
    let eye = cam.eye();
    let clear = Color::srgb(0.08 + 0.02 * viewport_index as f32, 0.10, 0.16);

    commands
        .spawn((
            cam,
            Camera3d::default(),
            Camera {
                // Lower than the primary egui camera (order 0). All viewport
                // cameras share order -1; they render to separate targets so
                // ordering between them is irrelevant.
                order: -1,
                clear_color: ClearColorConfig::Custom(clear),
                ..default()
            },
            RenderTarget::Image(image.into()),
            Transform::from_translation(eye).looking_at(target, Vec3::Y),
            layer,
        ))
        .id()
}
