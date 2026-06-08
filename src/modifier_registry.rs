//! Discovers modifier types and their factories via Bevy's
//! [`AppTypeRegistry`].
//!
//! Each modifier type contributes a [`ReflectModifier`] piece of
//! [`bevy_reflect::TypeData`] carrying:
//!
//! - a `factory: fn(&mut Module) -> BoxedAnyModifier` — sensible defaults for
//!   `ExprHandle` literals, the one bit of per-type knowledge that reflection
//!   alone cannot derive;
//! - the modifier's [`ModifierContext`] flags (cached at registration so we
//!   don't re-probe per frame).
//!
//! The editor reads `AppTypeRegistry` to enumerate modifier types,
//! display them in the Add menu, and instantiate them via the
//! factory. **No hard-coded list of modifier types is referenced
//! from any other module.**
//!
//! ## Today: bridge code
//!
//! `bevy_hanabi` 0.18 doesn't register its modifier types with the
//! type registry, let alone provide [`ReflectModifier`] data for
//! them. [`ModifierRegistryPlugin`] does both on Hanabi's behalf for
//! every built-in we want to expose. The entire
//! `register_builtin_modifiers` function is "upstream candidate" —
//! when Hanabi (or a third-party metadata crate) ships equivalent
//! registrations, we delete this function and the editor becomes
//! truly modifier-agnostic. No changes anywhere else.
//!
//! ## Tomorrow: user-defined modifiers
//!
//! Any user crate can ship a modifier type by:
//!
//! ```ignore
//! app.register_type::<MyModifier>();
//! crate::modifier_registry::insert_reflect_modifier::<MyModifier>(
//!     app, |module| { /* factory */ },
//! );
//! ```
//!
//! …and the editor picks it up automatically.

use std::any::TypeId;
use std::borrow::Cow;

use bevy::math::{Vec3, Vec4};
use bevy::prelude::*;
use bevy::reflect::{GetTypeRegistration, TypeRegistry};
use bevy::math::UVec2;
use bevy_hanabi::{
    AccelModifier, Attribute, ColorOverLifetimeModifier, FlipbookModifier, Gradient,
    LinearDragModifier, ModifierContext, Module, OrientMode, OrientModifier, SetAttributeModifier,
    SetColorModifier, SetPositionSphereModifier, SetSizeModifier, SetVelocitySphereModifier,
    ShapeDimension,
};

use crate::document::ModifierGroup;
use crate::modifier_ops::BoxedAnyModifier;

/// Builds a fresh modifier instance, allocating any required
/// `ExprHandle` literals into `module`.
pub type ModifierFactory = fn(&mut Module) -> BoxedAnyModifier;

/// Returns the set of attributes a modifier *fully overwrites*
/// (pure assignment in the generated WGSL, discarding any previous
/// value). Empty for read-modify-write modifiers like
/// [`AccelModifier`] / [`LinearDragModifier`], and for render-stage
/// modifiers (which write vertex shader variables rather than
/// particle attributes).
///
/// Used by the Effect panel's shadow detector to warn when a
/// modifier's output is fully obviated by a later overwrite of every
/// attribute it produces.
///
/// `Modifier::attributes()` deliberately can't be used for this:
/// upstream returns the modifier's *layout requirements* (reads AND
/// writes), so e.g. `SetVelocityCircleModifier` lists
/// `[POSITION, VELOCITY]` even though it only writes VELOCITY.
pub type ModifierOverwrites = fn(&dyn bevy::reflect::Reflect) -> Vec<Attribute>;

/// Type data attached to a modifier type's registration in
/// [`AppTypeRegistry`]. Carrying both fields here means the editor
/// never needs to construct an instance just to query
/// `Modifier::context()`.
#[derive(Clone, Copy)]
pub struct ReflectModifier {
    pub factory: ModifierFactory,
    pub context: ModifierContext,
    pub overwrites: ModifierOverwrites,
}

/// Short-lived view into a single modifier registration. Yielded by
/// [`iter_modifier_kinds`] / [`iter_modifier_kinds_for`].
pub struct ModifierKindView<'a> {
    pub type_id: TypeId,
    /// `reflect_short_type_path()` of the modifier struct (e.g.
    /// `"SetPositionSphereModifier"`). Used as the display-name
    /// lookup key.
    pub short_type_name: &'a str,
    pub reflect_modifier: &'a ReflectModifier,
}

impl ModifierKindView<'_> {
    pub fn display_name(&self) -> Cow<'static, str> {
        crate::ui::modifier_names::display_name_for_type(self.short_type_name)
    }

    pub fn context(&self) -> ModifierContext {
        self.reflect_modifier.context
    }

    #[allow(dead_code)]
    pub fn make(&self, module: &mut Module) -> BoxedAnyModifier {
        (self.reflect_modifier.factory)(module)
    }
}

/// Iterate every type in the registry that carries
/// [`ReflectModifier`] type data, sorted by short type name for
/// stable UI ordering. (Registry iteration is otherwise
/// HashMap-backed, hence unstable.)
pub fn iter_modifier_kinds(registry: &TypeRegistry) -> impl Iterator<Item = ModifierKindView<'_>> {
    let mut v: Vec<ModifierKindView<'_>> = registry
        .iter()
        .filter_map(|reg| {
            let rm = reg.data::<ReflectModifier>()?;
            Some(ModifierKindView {
                type_id: reg.type_id(),
                short_type_name: reg.type_info().type_path_table().short_path(),
                reflect_modifier: rm,
            })
        })
        .collect();
    v.sort_by_key(|k| k.short_type_name);
    v.into_iter()
}

/// Same as [`iter_modifier_kinds`] but filtered to modifiers valid in
/// the given group's context.
pub fn iter_modifier_kinds_for(
    registry: &TypeRegistry,
    group: ModifierGroup,
) -> impl Iterator<Item = ModifierKindView<'_>> {
    let flag: ModifierContext = group.into();
    iter_modifier_kinds(registry).filter(move |k| k.context().contains(flag))
}

/// Look up a modifier kind by `TypeId`. Returns `None` if the type
/// isn't registered or carries no [`ReflectModifier`] data.
pub fn get_modifier_kind(registry: &TypeRegistry, type_id: TypeId) -> Option<ModifierKindView<'_>> {
    let reg = registry.get(type_id)?;
    let rm = reg.data::<ReflectModifier>()?;
    Some(ModifierKindView {
        type_id: reg.type_id(),
        short_type_name: reg.type_info().type_path_table().short_path(),
        reflect_modifier: rm,
    })
}

/// Register `T` in the type registry (if not already) and attach a
/// [`ReflectModifier`] with the given factory and overwrite-set
/// callback. Probes the factory once in a throwaway `Module` to
/// cache the modifier's [`ModifierContext`].
///
/// `pub` so user crates can call this from their own plugins to
/// contribute custom modifier types without touching this file.
pub fn insert_reflect_modifier<T: GetTypeRegistration>(
    app: &mut App,
    factory: ModifierFactory,
    overwrites: ModifierOverwrites,
) {
    app.register_type::<T>();

    // Probe the factory once to derive the context.
    let mut scratch = Module::default();
    let context = match factory(&mut scratch) {
        BoxedAnyModifier::Plain(m) => m.context(),
        // Render modifiers' context() is generated by Hanabi's
        // `impl_mod_render!` macro and always returns Render.
        BoxedAnyModifier::Render(_) => ModifierContext::Render,
    };
    let rm = ReflectModifier {
        factory,
        context,
        overwrites,
    };

    let app_registry = app.world().resource::<AppTypeRegistry>();
    let mut registry = app_registry.write();
    match registry.get_mut(TypeId::of::<T>()) {
        Some(reg) => reg.insert(rm),
        None => warn!(
            "insert_reflect_modifier: type {} just registered but not found in AppTypeRegistry",
            std::any::type_name::<T>()
        ),
    }
}

/// Adds [`ReflectModifier`] type data for every built-in Hanabi
/// modifier the editor currently supports.
pub struct ModifierRegistryPlugin;

impl Plugin for ModifierRegistryPlugin {
    fn build(&self, app: &mut App) {
        register_builtin_modifiers(app);
    }
}

/// "Bridge" registration: bevy_hanabi 0.18 doesn't register its
/// modifier types or provide [`ReflectModifier`] data, so we do
/// both. Each line below is an upstream candidate — when Hanabi
/// ships the equivalent, delete the matching call here.
fn register_builtin_modifiers(app: &mut App) {
    insert_reflect_modifier::<SetAttributeModifier>(
        app,
        |m| {
            // Default to LIFETIME = 5.0; the Details panel lets the
            // user retarget the attribute and tune the value.
            let v = m.lit(5.0_f32);
            BoxedAnyModifier::Plain(Box::new(SetAttributeModifier::new(Attribute::LIFETIME, v)))
        },
        |m| {
            // Reads its own `attribute` field — the one slot whose
            // overwrite set depends on the instance, not the type.
            m.downcast_ref::<SetAttributeModifier>()
                .map(|s| vec![s.attribute])
                .unwrap_or_default()
        },
    );

    insert_reflect_modifier::<SetPositionSphereModifier>(
        app,
        |m| {
            let center = m.lit(Vec3::ZERO);
            let radius = m.lit(1.0_f32);
            BoxedAnyModifier::Plain(Box::new(SetPositionSphereModifier {
                center,
                radius,
                dimension: ShapeDimension::Volume,
            }))
        },
        |_| vec![Attribute::POSITION],
    );

    insert_reflect_modifier::<SetVelocitySphereModifier>(
        app,
        |m| {
            let center = m.lit(Vec3::ZERO);
            let speed = m.lit(1.0_f32);
            BoxedAnyModifier::Plain(Box::new(SetVelocitySphereModifier { center, speed }))
        },
        |_| vec![Attribute::VELOCITY],
    );

    insert_reflect_modifier::<AccelModifier>(
        app,
        |m| {
            BoxedAnyModifier::Plain(Box::new(AccelModifier::constant(
                m,
                Vec3::new(0.0, -9.81, 0.0),
            )))
        },
        // Read-modify-write: velocity += accel * dt. Not a pure
        // overwrite, so it doesn't shadow earlier velocity writers.
        |_| vec![],
    );

    insert_reflect_modifier::<LinearDragModifier>(
        app,
        |m| BoxedAnyModifier::Plain(Box::new(LinearDragModifier::constant(m, 1.0))),
        // Read-modify-write on velocity.
        |_| vec![],
    );

    insert_reflect_modifier::<SetColorModifier>(
        app,
        |_m| {
            BoxedAnyModifier::Render(Box::new(SetColorModifier::new(Vec4::new(
                1.0, 1.0, 1.0, 1.0,
            ))))
        },
        // Render-stage: writes vertex shader `color`, not the COLOR
        // particle attribute. Shadow analysis across render modifiers
        // would need a separate "vertex output" channel; out of scope
        // for now.
        |_| vec![],
    );

    insert_reflect_modifier::<SetSizeModifier>(
        app,
        |_m| {
            BoxedAnyModifier::Render(Box::new(SetSizeModifier {
                size: Vec3::new(0.1, 0.1, 0.1).into(),
            }))
        },
        // See SetColorModifier note — writes vertex `size`.
        |_| vec![],
    );

    insert_reflect_modifier::<OrientModifier>(
        app,
        |_m| {
            BoxedAnyModifier::Render(Box::new(OrientModifier::new(
                OrientMode::ParallelCameraDepthPlane,
            )))
        },
        |_| vec![],
    );

    insert_reflect_modifier::<FlipbookModifier>(
        app,
        |_m| {
            BoxedAnyModifier::Render(Box::new(FlipbookModifier {
                sprite_grid_size: UVec2::new(4, 4),
            }))
        },
        |_| vec![],
    );

    insert_reflect_modifier::<ColorOverLifetimeModifier>(
        app,
        |_m| {
            BoxedAnyModifier::Render(Box::new(ColorOverLifetimeModifier::new(Gradient::constant(
                Vec4::ONE,
            ))))
        },
        // Render-stage: writes vertex `color`, like SetColorModifier.
        |_| vec![],
    );
}
