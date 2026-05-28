//! Helpers for building demo `EffectAsset`s used by the startup seeds.
//!
//! Production-quality effect authoring will live in the Phase 5 properties
//! editor; this module just exists so new/seeded documents have something
//! visible to render.

use bevy::prelude::*;
use bevy_hanabi::prelude::*;

/// Build a small, visible demo effect: particles spawn over a sphere
/// surface with outward velocity, fall under gravity, and fade out.
pub fn demo_effect() -> EffectAsset {
    let mut gradient = bevy_hanabi::Gradient::new();
    gradient.add_key(0.0, Vec4::new(1.0, 0.5, 0.1, 1.0));
    gradient.add_key(0.5, Vec4::new(1.0, 0.2, 0.05, 0.7));
    gradient.add_key(1.0, Vec4::ZERO);

    let mut module = Module::default();

    let init_pos = SetPositionSphereModifier {
        center: module.lit(Vec3::ZERO),
        radius: module.lit(0.5),
        dimension: ShapeDimension::Surface,
    };
    let init_vel = SetVelocitySphereModifier {
        center: module.lit(Vec3::ZERO),
        speed: module.lit(2.0),
    };
    let lifetime = module.lit(2.5_f32);
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime);
    let accel = AccelModifier::new(module.lit(Vec3::new(0.0, -2.0, 0.0)));

    EffectAsset::new(8192, SpawnerSettings::rate(120.0.into()), module)
        .with_name("demo")
        .init(init_pos)
        .init(init_vel)
        .init(init_lifetime)
        .update(accel)
        .render(ColorOverLifetimeModifier {
            gradient,
            blend: ColorBlendMode::Overwrite,
            mask: ColorBlendMask::RGBA,
        })
}
