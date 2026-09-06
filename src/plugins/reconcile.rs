//! Per-document scene reconciliation (ECS form).
//!
//! Documents are entities. Each document entity has children:
//! a scene root (with light + placeholder mesh), and N viewport cameras
//! each rendering into its own `Image`. Hierarchical despawn cleans
//! everything up when a document entity is despawned.

use std::collections::{HashMap, HashSet};

use bevy::{
    asset::RenderAssetUsages,
    camera::{Hdr, RenderTarget, visibility::RenderLayers},
    mesh::PrimitiveTopology,
    post_process::bloom::Bloom,
    prelude::*,
    render::render_resource::{
        Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    },
};
use bevy_egui::{EguiTextureHandle, EguiUserTextures};
use bevy_hanabi::EffectMaterial;
use hanabi_effect_graph::model::EmitterId;

use crate::{
    document::{
        DocumentContent, DocumentSceneRoot, DocumentUi, DocumentViewports, EmitterSceneEntities,
        PanelKind, SceneEmitter, ViewportCamera, ViewportSlots,
    },
    effect_graph::bake::PlannedImage,
    proxy::ProxyEmitters,
};

/// A 1×1 white placeholder image for texture slots with no editor-side asset.
///
/// Host-supplied (runtime) and unbound slots have no asset to load in the
/// editor, but the emitter's material still needs a bound image per slot. This
/// shared handle fills those slots so the emitter renders (untextured) rather
/// than failing.
#[derive(Resource)]
pub struct TexturePlaceholder(pub Handle<Image>);

impl FromWorld for TexturePlaceholder {
    fn from_world(world: &mut World) -> Self {
        let mut images = world.resource_mut::<Assets<Image>>();
        let image = Image::new_fill(
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &[255, 255, 255, 255],
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
        );
        Self(images.add(image))
    }
}

/// Shared mesh and material for viewport grids.
#[derive(Resource)]
pub struct ViewportGridAssets {
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
}

impl FromWorld for ViewportGridAssets {
    fn from_world(world: &mut World) -> Self {
        const HALF_CELLS: i32 = 20;
        const SPACING: f32 = 0.5;

        let extent = HALF_CELLS as f32 * SPACING;
        let mut vertices = Vec::with_capacity((HALF_CELLS as usize * 2 + 1) * 4);
        for cell in -HALF_CELLS..=HALF_CELLS {
            let offset = cell as f32 * SPACING;
            vertices.extend([
                Vec3::new(-extent, 0.0, offset),
                Vec3::new(extent, 0.0, offset),
                Vec3::new(offset, 0.0, -extent),
                Vec3::new(offset, 0.0, extent),
            ]);
        }

        let mesh = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::RENDER_WORLD)
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
        let material = StandardMaterial {
            base_color: Color::srgba(0.55, 0.58, 0.62, 0.32),
            unlit: true,
            alpha_mode: AlphaMode::Blend,
            ..default()
        };

        let mesh = world.resource_mut::<Assets<Mesh>>().add(mesh);
        let material = world
            .resource_mut::<Assets<StandardMaterial>>()
            .add(material);
        Self { mesh, material }
    }
}

#[derive(Component)]
pub(crate) struct ViewportGrid;

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
        Option<&ProxyEmitters>,
        Option<&Children>,
    )>,
    viewport_cams: Query<(Entity, &ViewportCamera)>,
    scene_roots: Query<Entity, With<DocumentSceneRoot>>,
    viewport_grids: Query<Entity, With<ViewportGrid>>,
    mut viewports: ResMut<DocumentViewports>,
    mut images: ResMut<Assets<Image>>,
    mut egui_user_textures: ResMut<EguiUserTextures>,
    asset_server: Res<AssetServer>,
    placeholder: Res<TexturePlaceholder>,
    grid_assets: Res<ViewportGridAssets>,
) {
    // Rebuild the UI lookup from scratch each frame; cheap (few docs, few
    // viewports).
    viewports.by_doc.clear();

    for (doc_entity, content, ui, proxies, children) in docs.iter_mut() {
        let layer = RenderLayers::layer(content.render_layer());
        let slots = viewports.by_doc.entry(doc_entity).or_default();

        let child_list: Vec<Entity> = children.map(|c| c.iter().collect()).unwrap_or_default();

        reconcile_viewport_grid(
            &mut commands,
            doc_entity,
            ui.show_viewport_grid,
            &child_list,
            &viewport_grids,
            &layer,
            &grid_assets,
        );

        // Scene root spawning is deferred until every emitter's proxy exists,
        // because every `ParticleEffect` we instantiate references a proxy
        // handle (not a canonical one). `ensure_proxy` installs each entry
        // once its canonical asset has loaded.
        if let Some(proxies) = proxies {
            ensure_scene_root(
                &mut commands,
                doc_entity,
                content,
                proxies,
                &child_list,
                &scene_roots,
                &layer,
                &asset_server,
                &placeholder,
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

fn reconcile_viewport_grid(
    commands: &mut Commands,
    doc_entity: Entity,
    show_grid: bool,
    children: &[Entity],
    viewport_grids: &Query<Entity, With<ViewportGrid>>,
    layer: &RenderLayers,
    assets: &ViewportGridAssets,
) {
    let existing = children
        .iter()
        .find_map(|child| viewport_grids.get(*child).ok());

    match (show_grid, existing) {
        (true, None) => {
            let grid = commands
                .spawn((
                    Name::new("Viewport Grid"),
                    ViewportGrid,
                    Mesh3d(assets.mesh.clone()),
                    MeshMaterial3d(assets.material.clone()),
                    Transform::IDENTITY,
                    layer.clone(),
                ))
                .id();
            commands.entity(doc_entity).add_child(grid);
        }
        (false, Some(grid)) => {
            commands.entity(grid).despawn();
        }
        _ => {}
    }
}

/// Spawn a document's whole preview scene once every emitter has a proxy.
///
/// One `ParticleEffect` entity per baked emitter in the document's
/// `EffectGraph`, under a shared `DocumentSceneRoot`, plus a light. Waits until
/// *every* emitter has a built proxy (rather than spawning incrementally) so
/// the second `EffectParent`-wiring pass below can always resolve every parent
/// reference to a sibling that already exists — a GPU-driven child spawned
/// before its parent emitter existed would have nothing to attach to.
fn ensure_scene_root(
    commands: &mut Commands,
    doc_entity: Entity,
    content: &DocumentContent,
    proxies: &ProxyEmitters,
    children: &[Entity],
    scene_roots: &Query<Entity, With<DocumentSceneRoot>>,
    layer: &RenderLayers,
    asset_server: &AssetServer,
    placeholder: &TexturePlaceholder,
) {
    let already = children.iter().any(|c| scene_roots.get(*c).is_ok());
    if already {
        return;
    }

    let emitter_ids: Vec<EmitterId> = content.preview_emitter_ids().collect();
    if emitter_ids.is_empty() || !emitter_ids.iter().all(|id| proxies.contains(*id)) {
        return; // still waiting on one or more emitters' proxies
    }

    let scene_root = commands
        .spawn((
            DocumentSceneRoot,
            Transform::default(),
            Visibility::default(),
        ))
        .id();
    commands.entity(doc_entity).add_child(scene_root);

    let light = commands
        .spawn((
            DirectionalLight {
                illuminance: 10_000.0,
                ..default()
            },
            Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.6, 0.4, 0.0)),
            layer.clone(),
        ))
        .id();
    commands.entity(scene_root).add_child(light);

    // First pass: spawn one `ParticleEffect` per baked emitter, in document
    // order, collecting entity ids for the second (parent-wiring) pass.
    let mut entity_map: HashMap<EmitterId, Entity> = HashMap::with_capacity(emitter_ids.len());
    for &emitter in &emitter_ids {
        let (Some(instance), Some(record)) =
            (proxies.get(emitter), content.emitter_record(emitter))
        else {
            continue; // guarded above, but stay defensive
        };

        // One image handle per texture slot, ordered by slot index to match
        // the baked module's texture layout. Editor-known assets load from
        // disk; host (runtime) and unbound slots fall back to a white
        // placeholder.
        let images: Vec<Handle<Image>> = record
            .texture_plan
            .iter()
            .map(|planned| match planned {
                // The artist's chosen image lives outside the `assets/`
                // folder, so it is an "unapproved" path; `load_override`
                // opts these specific loads past the asset server's `Deny`
                // policy.
                PlannedImage::Asset(path) => asset_server
                    .load_builder()
                    .override_unapproved()
                    .load(path.clone()),
                PlannedImage::Runtime(_) | PlannedImage::Unbound => placeholder.0.clone(),
            })
            .collect();

        // Seed the instance's properties with the values tweaked since the
        // last structural rebake. hanabi's `update_properties_from_asset`
        // only *adds* missing properties (never overwrites), so pre-seeding
        // preserves live tweaks that the proxy asset's stale defaults would
        // otherwise revert.
        let seed_props: Vec<(String, bevy_hanabi::Value)> = instance
            .current_values
            .iter()
            .map(|(name, value)| (name.clone(), *value))
            .collect();

        let mut emitter_cmds = commands.spawn((
            SceneEmitter(emitter),
            bevy_hanabi::ParticleEffect::new(instance.handle.clone()),
            // Required when the (proxy) asset declares any properties: see
            // seed_props above.
            bevy_hanabi::EffectProperties::default().with_properties(seed_props),
            Transform::IDENTITY,
            layer.clone(),
        ));
        if !images.is_empty() {
            emitter_cmds.insert(EffectMaterial { images });
        }
        let emitter_entity = emitter_cmds.id();
        commands.entity(scene_root).add_child(emitter_entity);
        entity_map.insert(emitter, emitter_entity);
    }

    // Second pass: `EffectParent` is a plain data reference (not an ECS
    // parent/child relationship), so it's order-independent to attach — but
    // every sibling must already exist, hence the separate pass.
    for &emitter in &emitter_ids {
        if let Some(parent_emitter) = content.emitter_parent(emitter)
            && let (Some(&child_entity), Some(&parent_entity)) =
                (entity_map.get(&emitter), entity_map.get(&parent_emitter))
        {
            commands
                .entity(child_entity)
                .insert(bevy_hanabi::EffectParent::new(parent_entity));
        }
    }

    commands
        .entity(scene_root)
        .insert(EmitterSceneEntities(entity_map));
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
    let angle = 10.0_f32.to_radians() + viewport_index as f32 * std::f32::consts::FRAC_PI_3;
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
    let transform = cam.transform();

    commands
        .spawn((
            cam,
            Camera3d::default(),
            Camera {
                // Lower than the primary egui camera (order 0). All viewport
                // cameras share order -1; they render to separate targets so
                // ordering between them is irrelevant.
                order: -1,
                clear_color: ClearColorConfig::Custom(Color::BLACK),
                ..default()
            },
            // HDR + bloom so bright particle cores glow rather than reading as
            // flat quads; the camera's default tonemapping maps the HDR result.
            Hdr,
            Bloom {
                intensity: 0.25,
                ..default()
            },
            RenderTarget::Image(image.into()),
            transform,
            layer,
        ))
        .id()
}
