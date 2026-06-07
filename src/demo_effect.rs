//! Helpers for building demo `EffectAsset`s used by the startup seeds.
//!
//! Production-quality effect authoring will live in the Phase 5 properties
//! editor; this module just exists so new/seeded documents have something
//! visible to render.

use bevy::prelude::*;
use bevy_hanabi::prelude::*;

/// Build a small, visible demo effect: particles spawn over a sphere
/// surface with outward velocity, fall under gravity, and fade out.
///
/// The effect is deliberately varied so the node-graph editor exercises a
/// range of features: scalar and vector **properties**, an **operator**
/// sub-graph, modifiers carrying **enum** and **integral** (non-expr) fields,
/// and a few free literal nodes spanning the pin **type-color** palette.
pub fn demo_effect() -> EffectAsset {
    let mut gradient = bevy_hanabi::Gradient::new();
    gradient.add_key(0.0, Vec4::new(1.0, 0.5, 0.1, 1.0));
    gradient.add_key(0.5, Vec4::new(1.0, 0.2, 0.05, 0.7));
    gradient.add_key(1.0, Vec4::ZERO);

    let mut module = Module::default();

    // Properties: a Vec3 gravity and a scalar spawn speed. These surface as
    // property nodes in the graph, wired into the modifiers that consume them
    // (distinct accent + value-type color on the link).
    let gravity = module.add_property("gravity", Vec3::new(0.0, -2.0, 0.0).into());
    let spawn_speed = module.add_property("spawn_speed", 2.0_f32.into());

    let init_pos = SetPositionSphereModifier {
        center: module.lit(Vec3::ZERO),
        // radius = base * scale, a tiny operator sub-graph (f32) so the graph
        // shows a Binary node feeding a modifier input.
        radius: {
            let base = module.lit(0.4_f32);
            let scale = module.lit(1.25_f32);
            module.mul(base, scale)
        },
        dimension: ShapeDimension::Surface,
    };
    let init_vel = SetVelocitySphereModifier {
        center: module.lit(Vec3::ZERO),
        speed: module.prop(spawn_speed),
    };
    let lifetime = module.lit(2.5_f32);
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);
    let accel = AccelModifier::new(module.prop(gravity));

    // Free literal nodes spanning the pin type-color palette (integer,
    // unsigned, vec2, vec4). They aren't referenced by any modifier, so
    // codegen ignores them, but they render as nodes so the colors are
    // visible without needing a consuming modifier for each type.
    let _i = module.lit(7_i32);
    let _u = module.lit(3_u32);
    let _v2 = module.lit(Vec2::new(1.0, 2.0));
    let _v4 = module.lit(Vec4::new(0.2, 0.4, 0.6, 1.0));

    EffectAsset::new(8192, SpawnerSettings::rate(120.0.into()), module)
        .with_name("demo")
        .init(init_pos)
        .init(init_vel)
        .init(init_lifetime)
        .update(accel)
        // Render modifiers carrying enum / integral fields, for testing
        // non-expr field handling.
        .render(OrientModifier::new(OrientMode::FaceCameraPosition))
        .render(FlipbookModifier {
            sprite_grid_size: UVec2::new(4, 4),
        })
        // CpuValue modifier fields: a constant size and a uniform-range color,
        // exercising both `CpuValue::Single` and `CpuValue::Uniform` display.
        .render(SetSizeModifier {
            size: CpuValue::Single(Vec3::splat(0.1)),
        })
        .render(SetColorModifier {
            color: CpuValue::Uniform((
                Vec4::new(1.0, 0.5, 0.1, 1.0),
                Vec4::new(1.0, 0.9, 0.3, 1.0),
            )),
            blend: ColorBlendMode::Overwrite,
            mask: ColorBlendMask::RGBA,
        })
        .render(ColorOverLifetimeModifier {
            gradient,
            blend: ColorBlendMode::Overwrite,
            mask: ColorBlendMask::RGBA,
        })
}
