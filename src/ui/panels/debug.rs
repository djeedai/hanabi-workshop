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
        paint_layout_strip(ui, asset);
    });
}

/// One byte-range in the particle layout — either a real attribute or
/// inter-attribute padding.
enum LayoutSegment {
    Attr {
        offset: u32,
        size: u32,
        name: &'static str,
        type_name: &'static str,
        color: egui::Color32,
    },
    Padding {
        offset: u32,
        size: u32,
    },
}

impl LayoutSegment {
    fn range(&self) -> (u32, u32) {
        match self {
            Self::Attr { offset, size, .. } | Self::Padding { offset, size } => (*offset, *size),
        }
    }
}

/// Render the particle layout as a horizontal strip of colored
/// segments, wrapping every `LINE_BYTES` bytes. Each segment's width
/// is proportional to the attribute's byte size; padding shows as a
/// dashed gray outline.
///
/// This is "VFX authoring info", not a debug table — it makes it
/// obvious when adding a modifier blows out the per-particle GPU
/// memory budget, or when an attribute order leaves wasted padding
/// holes.
fn paint_layout_strip(ui: &mut egui::Ui, asset: &EffectAsset) {
    /// Width of one display row, in bytes. Matches WGSL's std430
    /// 16-byte alignment boundary so the rows visually correspond
    /// to vec4 slots — the unit the GPU actually loads.
    const LINE_BYTES: u32 = 16;
    const ROW_HEIGHT: f32 = 26.0;
    const ROW_GAP: f32 = 3.0;

    let layout = asset.particle_layout();
    let total = layout.size() as u32;
    if total == 0 {
        return;
    }

    // Collect attribute intervals (offset, size, name, type, color),
    // sorted by offset.
    let mut intervals: Vec<(u32, u32, &'static str, &'static str)> = Attribute::all()
        .iter()
        .filter_map(|attr| {
            let offset = layout.byte_offset(*attr)?;
            Some((
                offset,
                attr.size() as u32,
                attr.name(),
                value_type_short(&attr.value_type()),
            ))
        })
        .collect();
    intervals.sort_by_key(|(o, _, _, _)| *o);

    // Build a contiguous segment list including padding holes.
    let mut segments: Vec<LayoutSegment> = Vec::with_capacity(intervals.len() * 2);
    let mut cursor = 0u32;
    for (offset, size, name, type_name) in intervals {
        if offset > cursor {
            segments.push(LayoutSegment::Padding {
                offset: cursor,
                size: offset - cursor,
            });
        }
        segments.push(LayoutSegment::Attr {
            offset,
            size,
            name,
            type_name,
            color: color_for(name),
        });
        cursor = offset + size;
    }
    if cursor < total {
        segments.push(LayoutSegment::Padding {
            offset: cursor,
            size: total - cursor,
        });
    }

    let n_rows = total.div_ceil(LINE_BYTES) as f32;
    let avail_w = ui.available_width();
    let height = n_rows * ROW_HEIGHT + (n_rows - 1.0).max(0.0) * ROW_GAP;
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(avail_w, height), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    let byte_w = avail_w / LINE_BYTES as f32;

    let mut hover_text: Option<String> = None;
    let hover_pos = response.hover_pos();

    for seg in &segments {
        let (offset, size) = seg.range();
        // Split the segment across line boundaries.
        let mut o = offset;
        let end = offset + size;
        while o < end {
            let row = (o / LINE_BYTES) as f32;
            let col = (o % LINE_BYTES) as f32;
            let span = (LINE_BYTES - (o % LINE_BYTES)).min(end - o);
            let piece_rect = egui::Rect::from_min_size(
                egui::pos2(
                    rect.left() + col * byte_w,
                    rect.top() + row * (ROW_HEIGHT + ROW_GAP),
                ),
                egui::vec2(span as f32 * byte_w, ROW_HEIGHT),
            );
            // 1px inset so adjacent segments don't visually merge.
            let inner = piece_rect.shrink(1.0);

            match seg {
                LayoutSegment::Attr {
                    name,
                    type_name,
                    color,
                    ..
                } => {
                    painter.rect_filled(inner, 3.0, *color);
                    // Label: only render if the piece is wide enough.
                    // 4-byte segments (one f32) are usually too narrow
                    // for both name and type; show name only there.
                    let label_w = inner.width();
                    let text = if label_w > 90.0 {
                        format!("{name}  {type_name}")
                    } else if label_w > 36.0 {
                        (*name).to_string()
                    } else if label_w > 18.0 {
                        // Just the type, shortened.
                        type_name
                            .trim_start_matches("vec")
                            .chars()
                            .take(2)
                            .collect()
                    } else {
                        String::new()
                    };
                    if !text.is_empty() {
                        painter.text(
                            inner.center(),
                            egui::Align2::CENTER_CENTER,
                            text,
                            egui::FontId::proportional(11.0),
                            text_color_for(*color),
                        );
                    }
                }
                LayoutSegment::Padding { .. } => {
                    paint_dashed_rect(
                        &painter,
                        inner,
                        ui.visuals().weak_text_color(),
                        1.0,
                    );
                    if inner.width() > 50.0 {
                        painter.text(
                            inner.center(),
                            egui::Align2::CENTER_CENTER,
                            format!("padding ({}B)", size),
                            egui::FontId::proportional(10.0),
                            ui.visuals().weak_text_color(),
                        );
                    }
                }
            }

            // Hover lookup.
            if let Some(p) = hover_pos {
                if piece_rect.contains(p) {
                    hover_text = Some(match seg {
                        LayoutSegment::Attr {
                            name,
                            type_name,
                            size,
                            offset,
                            ..
                        } => format!("{name}: {type_name} ({size}B @ offset {offset})"),
                        LayoutSegment::Padding { size, offset } => {
                            format!("padding ({size}B @ offset {offset})")
                        }
                    });
                }
            }

            o += span;
        }
    }

    if let Some(text) = hover_text {
        response.on_hover_text_at_pointer(text);
    }
}

/// Pick a stable color per attribute name from a small categorical
/// palette. Deterministic across runs so the user builds muscle memory
/// for which color is which attribute.
fn color_for(name: &str) -> egui::Color32 {
    const PALETTE: &[egui::Color32] = &[
        egui::Color32::from_rgb(0xE5, 0x73, 0x73), // red
        egui::Color32::from_rgb(0xFF, 0xB7, 0x4D), // orange
        egui::Color32::from_rgb(0xFF, 0xD5, 0x4F), // amber
        egui::Color32::from_rgb(0xAE, 0xD5, 0x81), // light green
        egui::Color32::from_rgb(0x4D, 0xB6, 0xAC), // teal
        egui::Color32::from_rgb(0x64, 0xB5, 0xF6), // blue
        egui::Color32::from_rgb(0x79, 0x86, 0xCB), // indigo
        egui::Color32::from_rgb(0xBA, 0x68, 0xC8), // purple
        egui::Color32::from_rgb(0xF0, 0x62, 0x92), // pink
        egui::Color32::from_rgb(0xA1, 0x88, 0x7F), // brown
        egui::Color32::from_rgb(0x90, 0xA4, 0xAE), // blue gray
    ];
    // djb2 hash → palette index.
    let mut h: u32 = 5381;
    for b in name.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u32);
    }
    PALETTE[(h as usize) % PALETTE.len()]
}

/// Pick black or white text for legibility on `bg` using sRGB luminance.
fn text_color_for(bg: egui::Color32) -> egui::Color32 {
    let r = bg.r() as f32 / 255.0;
    let g = bg.g() as f32 / 255.0;
    let b = bg.b() as f32 / 255.0;
    let lum = 0.299 * r + 0.587 * g + 0.114 * b;
    if lum > 0.55 {
        egui::Color32::from_gray(20)
    } else {
        egui::Color32::WHITE
    }
}

/// Draw a dashed rectangle outline. egui has no built-in dashed stroke,
/// so we draw each side with short alternating segments.
fn paint_dashed_rect(
    painter: &egui::Painter,
    rect: egui::Rect,
    color: egui::Color32,
    stroke_w: f32,
) {
    let stroke = egui::Stroke::new(stroke_w, color);
    let corners = [
        (rect.left_top(), rect.right_top()),
        (rect.right_top(), rect.right_bottom()),
        (rect.right_bottom(), rect.left_bottom()),
        (rect.left_bottom(), rect.left_top()),
    ];
    for (a, b) in corners {
        paint_dashed_line(painter, a, b, 4.0, 3.0, stroke);
    }
}

fn paint_dashed_line(
    painter: &egui::Painter,
    a: egui::Pos2,
    b: egui::Pos2,
    dash: f32,
    gap: f32,
    stroke: egui::Stroke,
) {
    let v = b - a;
    let len = v.length();
    if len < 0.001 {
        return;
    }
    let dir = v / len;
    let mut t = 0.0;
    while t < len {
        let t2 = (t + dash).min(len);
        painter.line_segment([a + dir * t, a + dir * t2], stroke);
        t = t2 + gap;
    }
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
