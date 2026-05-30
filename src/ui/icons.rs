//! Font Awesome 7 Free Solid font loader and small UI helpers.
//!
//! egui's bundled font set is missing many common UI glyphs (✕ ↑ ↓ …
//! all render as tofu). We bundle Font Awesome 7 Free Solid as an
//! egui fallback so any Font-Awesome `\u{...}` codepoint renders as
//! the corresponding icon, while ordinary text still uses the default
//! Ubuntu / Noto Emoji fonts.
//!
//! Codepoint constants live in [`crate::IconsFontAwesome7`] (an
//! auto-generated table from the FA 7 metadata, kept verbatim). Use
//! `ICON_FOO` (a `char`); for an egui button, wrap with
//! [`icon_button`] or `.to_string()`.

use std::sync::Arc;

use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};

pub use crate::IconsFontAwesome7::*;

/// Embedded Font Awesome 7 Free Solid OTF. Licensed under SIL OFL 1.1
/// (font file) + CC-BY 4.0 (icons). See `assets/fonts/` for license texts.
const FA_SOLID_OTF: &[u8] = include_bytes!("../../assets/fonts/Font Awesome 7 Free-Solid-900.otf");

/// Install Font Awesome as a fallback in both the Proportional and
/// Monospace families. Runs once in `Startup` after
/// `EguiStartupSet::InitContexts` has created the primary context.
pub fn install_fonts(mut contexts: EguiContexts) {
    let Ok(ctx) = contexts.ctx_mut() else {
        bevy::log::error!(
            "install_fonts: primary egui context not ready yet — FA icons will be tofu"
        );
        return;
    };

    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "fa-solid".to_owned(),
        Arc::new(
            egui::FontData::from_static(FA_SOLID_OTF).tweak(egui::FontTweak {
                y_offset_factor: 0.07,
                scale: 0.95,
                ..Default::default()
            }),
        ),
    );
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .push("fa-solid".to_owned());
    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .push("fa-solid".to_owned());
    ctx.set_fonts(fonts);
    bevy::log::info!(
        "install_fonts: registered Font Awesome 7 Free Solid ({} bytes)",
        FA_SOLID_OTF.len()
    );
}

/// Square icon button. `size` is the side length in points; the icon
/// glyph is sized to about 60% of that to leave breathing room.
/// Returns the standard egui `Response` so callers can chain
/// `.on_hover_text(...)` / `.clicked()`.
pub fn icon_button(ui: &mut egui::Ui, icon: char, size: f32) -> egui::Response {
    let text = egui::RichText::new(icon.to_string()).size(size * 0.55);
    ui.add_sized([size, size], egui::Button::new(text))
}

/// Like [`icon_button`] but rendered with the "selected" highlight
/// when `selected` is true — useful for toggle buttons (Play/Pause).
#[allow(dead_code)]
pub fn icon_toggle(ui: &mut egui::Ui, icon: char, selected: bool, size: f32) -> egui::Response {
    let text = egui::RichText::new(icon.to_string()).size(size * 0.55);
    ui.add_sized([size, size], egui::Button::selectable(selected, text))
}
