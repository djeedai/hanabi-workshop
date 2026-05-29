//! Debug panel — generated WGSL inspector.
//!
//! Shows the **assembled** WGSL shaders that hanabi compiles for the
//! current effect (init / update / render), pulled by path from
//! `Assets<Shader>`. Hanabi's `compile_effects` system bakes its
//! templates and uploads the result under paths of the form
//! `hanabi/{asset_name}_{init|update|render}_{hash}.wgsl`. We look
//! them up by prefix match on the asset's name.
//!
//! (The Particle layout view used to live here too; it moved to
//! the Outline panel since it's core to authoring, not just debug.
//! The helpers stayed here as `pub(super)` to keep one source of
//! truth.)

use bevy::prelude::*;
use bevy::shader::Shader;
use bevy_egui::egui;
use bevy_hanabi::{Attribute, EffectAsset};

pub fn show(
    ui: &mut egui::Ui,
    effects: &Assets<EffectAsset>,
    shaders: &Assets<Shader>,
    effect_handle: &Handle<EffectAsset>,
) {
    ui.heading("Debug");
    ui.separator();

    let Some(asset) = effects.get(effect_handle) else {
        ui.label("(effect asset not loaded yet)");
        return;
    };

    egui::ScrollArea::vertical().show(ui, |ui| {
        wgsl_section(ui, asset, shaders);
    });
}

pub(super) fn layout_section(ui: &mut egui::Ui, asset: &EffectAsset) {
    let layout = asset.particle_layout();
    egui::CollapsingHeader::new(format!(
        "Particle layout ({} bytes, align {})",
        layout.size(),
        layout.align()
    ))
    .default_open(true)
    .show(ui, |ui| {
        if layout.is_empty() {
            ui.weak("(empty layout)");
            return;
        }
        // Iterate all known attributes; only show the ones actually
        // present in the layout, sorted by offset.
        let mut rows: Vec<(u32, &'static str, &'static str, usize)> = Attribute::all()
            .iter()
            .filter_map(|attr| {
                let offset = layout.byte_offset(*attr)?;
                let value_type = attr.value_type();
                let type_name = value_type_short(&value_type);
                Some((offset, attr.name(), type_name, attr.size()))
            })
            .collect();
        rows.sort_by_key(|(off, _, _, _)| *off);

        egui::Grid::new("particle-layout-grid")
            .num_columns(4)
            .striped(true)
            .show(ui, |ui| {
                ui.strong("offset");
                ui.strong("attribute");
                ui.strong("type");
                ui.strong("size");
                ui.end_row();
                for (offset, name, ty, size) in rows {
                    ui.monospace(format!("{offset:>4}"));
                    ui.monospace(name);
                    ui.monospace(ty);
                    ui.monospace(format!("{size}"));
                    ui.end_row();
                }
            });
    });
}

fn value_type_short(vt: &bevy_hanabi::ValueType) -> &'static str {
    // `ValueType: Debug`, but the formatted form is verbose. Map to
    // WGSL-flavoured short names.
    use bevy_hanabi::{ScalarType, ValueType, VectorType};
    match vt {
        ValueType::Scalar(ScalarType::Float) => "f32",
        ValueType::Scalar(ScalarType::Int) => "i32",
        ValueType::Scalar(ScalarType::Uint) => "u32",
        ValueType::Scalar(ScalarType::Bool) => "bool",
        ValueType::Vector(v) => match *v {
            VectorType::VEC2F => "vec2<f32>",
            VectorType::VEC3F => "vec3<f32>",
            VectorType::VEC4F => "vec4<f32>",
            VectorType::VEC2I => "vec2<i32>",
            VectorType::VEC3I => "vec3<i32>",
            VectorType::VEC4I => "vec4<i32>",
            VectorType::VEC2U => "vec2<u32>",
            VectorType::VEC3U => "vec3<u32>",
            VectorType::VEC4U => "vec4<u32>",
            _ => "vec?",
        },
        ValueType::Matrix(_) => "mat?",
        _ => "?",
    }
}

fn wgsl_section(ui: &mut egui::Ui, asset: &EffectAsset, shaders: &Assets<Shader>) {
    egui::CollapsingHeader::new("Generated WGSL (compiled)")
        .default_open(true)
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(
                    "Baked WGSL uploaded to bevy_render by hanabi's compile_effects \
                     system. Empty until the effect has been spawned at least once.",
                )
                .small()
                .weak(),
            );
            ui.add_space(4.0);

            // Hanabi names its baked shaders `hanabi/{name}_{phase}_{hash}.wgsl`.
            // Match by prefix on the asset name + phase suffix; pick the most
            // recently inserted handle if more than one matches (rare; happens
            // briefly while the old shader hasn't been GC'd yet).
            let name = asset.name.as_str();
            phase_block(
                ui,
                "Init",
                find_shader(shaders, name, "init").unwrap_or_default(),
            );
            phase_block(
                ui,
                "Update",
                find_shader(shaders, name, "update").unwrap_or_default(),
            );
            phase_block(
                ui,
                "Render",
                find_shader(shaders, name, "render").unwrap_or_default(),
            );
        });
}

/// Locate the most-recently-added baked shader whose path matches
/// `hanabi/{name}_{phase}_*.wgsl`. We rely on `Assets::iter` order
/// being stable within a frame; if multiple shaders match (old +
/// new during a hot recompile), the last one wins.
fn find_shader(shaders: &Assets<Shader>, name: &str, phase: &str) -> Option<String> {
    let prefix = format!("hanabi/{name}_{phase}_");
    let mut best: Option<&str> = None;
    for (_id, shader) in shaders.iter() {
        if shader.path.starts_with(&prefix) {
            best = Some(shader.source.as_str());
        }
    }
    best.map(str::to_string)
}

/// Read-only collapsing code block. `code` is the assembled WGSL;
/// empty string renders as a "not yet compiled" placeholder.
fn phase_block(ui: &mut egui::Ui, label: &str, code: String) {
    egui::CollapsingHeader::new(label)
        .id_salt(("debug-wgsl", label))
        .default_open(false)
        .show(ui, |ui| {
            if code.trim().is_empty() {
                ui.weak("(not yet compiled — try spawning the effect)");
                return;
            }
            let line_count = code.lines().count();
            ui.weak(format!("{} lines", line_count));
            let mut display = code;
            egui::ScrollArea::both()
                .max_height(360.0)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut display)
                            .font(egui::TextStyle::Monospace)
                            .code_editor()
                            .desired_width(f32::INFINITY)
                            .desired_rows(12)
                            .interactive(false),
                    );
                });
        });
}
