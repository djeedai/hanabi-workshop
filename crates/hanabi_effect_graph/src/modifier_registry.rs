//! Discovers modifier types and their factories via [`AppTypeRegistry`].
//!
//! Modifier registration and the per-type [`ReflectModifier`] data (factory +
//! [`ModifierContext`]) are provided by `bevy_hanabi` itself via
//! [`register_modifiers`]. [`ModifierRegistryPlugin`] calls it so headless
//! tools (baking, tests) work without [`HanabiPlugin`].
//!
//! The editor reads `AppTypeRegistry` to enumerate modifier types, display them
//! in the Add menu, and instantiate them via the factory. **No hard-coded list
//! of modifier types is referenced from any other module.**
//!
//! On top of Hanabi's registration the plugin attaches a [`ModifierOverwrites`]
//! piece of type data to the modifiers that fully overwrite a particle
//! attribute. This editor-only metadata drives the Emitter panel's shadow
//! detector and has no upstream equivalent.
//!
//! Any user crate can contribute a custom modifier type with
//! [`register_reflect_modifier`] and the editor picks it up automatically.
//!
//! [`HanabiPlugin`]: bevy_hanabi::HanabiPlugin
//! [`register_reflect_modifier`]: bevy_hanabi::register_reflect_modifier

use std::{any::TypeId, borrow::Cow};

use bevy::{
    prelude::*,
    reflect::{Reflect, TypeRegistry},
};
use bevy_hanabi::{
    Attribute, EmitSpawnEventModifier, EventEmitCondition, ModifierContext, ReflectModifier,
    SetAttributeModifier, SetPositionCircleModifier, SetPositionCone3dModifier,
    SetPositionSphereModifier, SetVelocityCircleModifier, SetVelocitySphereModifier,
    SetVelocityTangentModifier, register_modifiers, register_reflect_modifier,
};

use crate::ModifierGroup;

/// Type data giving the attributes a modifier *fully overwrites*.
///
/// Reports the attributes assigned by pure assignment in the generated WGSL,
/// discarding any previous value. Empty for read-modify-write modifiers like
/// [`AccelModifier`] / [`LinearDragModifier`], and for render-stage modifiers
/// (which write vertex shader variables rather than particle attributes).
///
/// Attached as type data to a modifier's registration and read by the Effect
/// panel's shadow detector to warn when a modifier's output is fully obviated
/// by a later overwrite of every attribute it produces.
///
/// `Modifier::attributes()` deliberately can't be used for this: upstream
/// returns the modifier's *layout requirements* (reads AND writes), so e.g.
/// `SetVelocityCircleModifier` lists `[POSITION, VELOCITY]` even though it only
/// writes VELOCITY.
///
/// [`AccelModifier`]: bevy_hanabi::AccelModifier
/// [`LinearDragModifier`]: bevy_hanabi::LinearDragModifier
#[derive(Clone, Copy)]
pub struct ModifierOverwrites {
    pub overwrites: fn(&dyn Reflect) -> Vec<Attribute>,
}

/// Short-lived view into a single modifier registration.
///
/// Yielded by [`iter_modifier_kinds`] / [`iter_modifier_kinds_for`].
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
        crate::modifier_names::display_name_for_type(self.short_type_name)
    }

    pub fn context(&self) -> ModifierContext {
        self.reflect_modifier.context
    }
}

/// Iterate every registered modifier kind, sorted by short type name.
///
/// Yields every type in the registry that carries [`ReflectModifier`] type
/// data. Sorting gives stable UI ordering (registry iteration is otherwise
/// HashMap-backed, hence unstable).
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

/// Like [`iter_modifier_kinds`], filtered to a group's valid modifiers.
pub fn iter_modifier_kinds_for(
    registry: &TypeRegistry,
    group: ModifierGroup,
) -> impl Iterator<Item = ModifierKindView<'_>> {
    let flag: ModifierContext = group.into();
    iter_modifier_kinds(registry).filter(move |k| k.context().contains(flag))
}

/// Look up a modifier kind by `TypeId`.
///
/// Returns `None` if the type isn't registered or carries no
/// [`ReflectModifier`] data.
pub fn get_modifier_kind(registry: &TypeRegistry, type_id: TypeId) -> Option<ModifierKindView<'_>> {
    let reg = registry.get(type_id)?;
    let rm = reg.data::<ReflectModifier>()?;
    Some(ModifierKindView {
        type_id: reg.type_id(),
        short_type_name: reg.type_info().type_path_table().short_path(),
        reflect_modifier: rm,
    })
}

/// Registers modifier types plus editor-only shadow-detection metadata.
pub struct ModifierRegistryPlugin;

impl Plugin for ModifierRegistryPlugin {
    fn build(&self, app: &mut App) {
        register_modifiers(app.world().resource::<AppTypeRegistry>());
        register_missing_modifiers(app.world().resource::<AppTypeRegistry>());
        register_builtin_overwrites(app);
    }
}

/// Registers modifier types upstream's [`register_modifiers`] omits.
///
/// [`EmitSpawnEventModifier`] drives our GPU source-context / event-link
/// topology (see `bake::bake_modifier`'s `child_index` override) but isn't
/// covered by `bevy_hanabi`'s own registration function, so we register it
/// here ourselves via the same public [`register_reflect_modifier`] API
/// upstream uses internally.
fn register_missing_modifiers(type_registry: &AppTypeRegistry) {
    // `register_reflect_modifier` only attaches `ReflectModifier` type data to
    // an *already*-registered type (it looks the type up and warns instead of
    // registering it if missing), unlike upstream's own `register_modifiers`
    // which registers each type first. Mirror that here.
    type_registry.write().register::<EmitSpawnEventModifier>();
    register_reflect_modifier::<EmitSpawnEventModifier>(type_registry, |module| {
        let count = module.lit(1u32);
        Box::new(EmitSpawnEventModifier {
            condition: EventEmitCondition::Always,
            count,
            child_index: 0,
        })
    });
}

/// Attach [`ModifierOverwrites`] data to built-in fully-overwriting modifiers.
///
/// Modifiers without an entry are treated as read-modify-write (no shadowing),
/// which is the conservative default.
fn register_builtin_overwrites(app: &mut App) {
    let app_registry = app.world().resource::<AppTypeRegistry>();
    let mut registry = app_registry.write();
    let mut set = |type_id: TypeId, overwrites: fn(&dyn Reflect) -> Vec<Attribute>| {
        if let Some(reg) = registry.get_mut(type_id) {
            reg.insert(ModifierOverwrites { overwrites });
        }
    };

    // Reads its own `attribute` field — the one slot whose overwrite
    // set depends on the instance, not the type.
    set(TypeId::of::<SetAttributeModifier>(), |m| {
        m.downcast_ref::<SetAttributeModifier>()
            .map(|s| vec![s.attribute])
            .unwrap_or_default()
    });

    set(TypeId::of::<SetPositionSphereModifier>(), |_| {
        vec![Attribute::POSITION]
    });
    set(TypeId::of::<SetPositionCircleModifier>(), |_| {
        vec![Attribute::POSITION]
    });
    set(TypeId::of::<SetPositionCone3dModifier>(), |_| {
        vec![Attribute::POSITION]
    });

    set(TypeId::of::<SetVelocitySphereModifier>(), |_| {
        vec![Attribute::VELOCITY]
    });
    set(TypeId::of::<SetVelocityCircleModifier>(), |_| {
        vec![Attribute::VELOCITY]
    });
    set(TypeId::of::<SetVelocityTangentModifier>(), |_| {
        vec![Attribute::VELOCITY]
    });
}
