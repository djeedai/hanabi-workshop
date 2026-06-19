//! Discovers modifier types and their factories via Bevy's
//! [`AppTypeRegistry`].
//!
//! Modifier registration and the per-type [`ReflectModifier`] data
//! (factory + [`ModifierContext`]) are provided by `bevy_hanabi`
//! itself via [`register_modifiers`]. [`ModifierRegistryPlugin`]
//! calls it so headless tools (baking, tests) work without
//! [`HanabiPlugin`].
//!
//! The editor reads `AppTypeRegistry` to enumerate modifier types,
//! display them in the Add menu, and instantiate them via the
//! factory. **No hard-coded list of modifier types is referenced
//! from any other module.**
//!
//! On top of Hanabi's registration the plugin attaches a
//! [`ModifierOverwrites`] piece of type data to the modifiers that
//! fully overwrite a particle attribute. This editor-only metadata
//! drives the Effect panel's shadow detector and has no upstream
//! equivalent.
//!
//! Any user crate can contribute a custom modifier type with
//! [`register_reflect_modifier`]
//! and the editor picks it up automatically.
//!
//! [`HanabiPlugin`]: bevy_hanabi::HanabiPlugin
//! [`register_reflect_modifier`]: bevy_hanabi::register_reflect_modifier

use std::any::TypeId;
use std::borrow::Cow;

use bevy::prelude::*;
use bevy::reflect::{Reflect, TypeRegistry};
use bevy_hanabi::{
    Attribute, ModifierContext, ReflectModifier, SetAttributeModifier, SetPositionCircleModifier,
    SetPositionCone3dModifier, SetPositionSphereModifier, SetVelocityCircleModifier,
    SetVelocitySphereModifier, SetVelocityTangentModifier, register_modifiers,
};

use crate::ModifierGroup;

/// Returns the set of attributes a modifier *fully overwrites*
/// (pure assignment in the generated WGSL, discarding any previous
/// value). Empty for read-modify-write modifiers like
/// [`AccelModifier`] /
/// [`LinearDragModifier`], and for
/// render-stage modifiers (which write vertex shader variables rather
/// than particle attributes).
///
/// Attached as type data to a modifier's registration and read by the
/// Effect panel's shadow detector to warn when a modifier's output is
/// fully obviated by a later overwrite of every attribute it produces.
///
/// `Modifier::attributes()` deliberately can't be used for this:
/// upstream returns the modifier's *layout requirements* (reads AND
/// writes), so e.g. `SetVelocityCircleModifier` lists
/// `[POSITION, VELOCITY]` even though it only writes VELOCITY.
///
/// [`AccelModifier`]: bevy_hanabi::AccelModifier
/// [`LinearDragModifier`]: bevy_hanabi::LinearDragModifier
#[derive(Clone, Copy)]
pub struct ModifierOverwrites {
    pub overwrites: fn(&dyn Reflect) -> Vec<Attribute>,
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
        crate::modifier_names::display_name_for_type(self.short_type_name)
    }

    pub fn context(&self) -> ModifierContext {
        self.reflect_modifier.context
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

/// Adds modifier registration ([`register_modifiers`]) plus the
/// editor-only [`ModifierOverwrites`] shadow-detection metadata.
pub struct ModifierRegistryPlugin;

impl Plugin for ModifierRegistryPlugin {
    fn build(&self, app: &mut App) {
        register_modifiers(app.world().resource::<AppTypeRegistry>());
        register_builtin_overwrites(app);
    }
}

/// Attach [`ModifierOverwrites`] type data to the built-in modifiers
/// that fully overwrite a particle attribute. Modifiers without an
/// entry are treated as read-modify-write (no shadowing), which is the
/// conservative default.
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
