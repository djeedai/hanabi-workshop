//! The Home landing tab: create actions and an emitter browser.
//!
//! Rendered in the outer dock's non-closable [`OuterTab::Home`] tab. A left
//! "Create" column (~30%) offers New / Open / Import; a right "Browse" column
//! (~70%) will list bundled examples and recent files with thumbnail previews.
//!
//! [`OuterTab::Home`]: crate::ui::OuterTab::Home

use std::{collections::HashMap, path::PathBuf};

use bevy::ecs::message::MessageWriter;
use bevy_egui::egui;

use crate::{
    app_commands::{AppCommand, DialogKind, PendingFileDialogs},
    effect_library::EffectEntry,
    thumbnail::{ClearThumbnailCache, ThumbnailRequest},
    ui::icons::{
        ICON_FILE_CIRCLE_PLUS, ICON_FILE_IMPORT, ICON_FOLDER_OPEN, ICON_SPRAY_CAN_SPARKLES,
    },
};

/// Ready thumbnail textures, keyed by emitter path.
pub type ThumbnailTextures = HashMap<PathBuf, egui::TextureId>;

/// Draw the Home tab body.
///
/// Splits into a left create column and a right browser column. Create actions
/// emit [`AppCommand`]s / spawn file dialogs; browser cards emit
/// [`AppCommand::OpenFile`].
pub fn show(
    ui: &mut egui::Ui,
    app: &mut MessageWriter<AppCommand>,
    pending: &mut PendingFileDialogs,
    examples: &[EffectEntry],
    recents: &[EffectEntry],
    thumbnails: &ThumbnailTextures,
    requests: &mut MessageWriter<ThumbnailRequest>,
    clear: &mut MessageWriter<ClearThumbnailCache>,
) {
    let left_width = (ui.available_width() * 0.3).clamp(200.0, 420.0);
    egui::Panel::left("home_create")
        .resizable(false)
        .exact_size(left_width)
        .show_inside(ui, |ui| {
            show_create_column(ui, app, pending);
        });
    egui::CentralPanel::default().show_inside(ui, |ui| {
        show_browse_column(ui, app, examples, recents, thumbnails, requests, clear);
    });
}

/// Left column: primary create/open actions.
fn show_create_column(
    ui: &mut egui::Ui,
    app: &mut MessageWriter<AppCommand>,
    pending: &mut PendingFileDialogs,
) {
    ui.add_space(12.0);
    ui.heading("Create");
    ui.add_space(8.0);

    if action_button(
        ui,
        ICON_FILE_CIRCLE_PLUS,
        "New Effect",
        "Start from the demo emitter",
    )
    .clicked()
    {
        app.write(AppCommand::NewDocument);
    }
    if action_button(
        ui,
        ICON_FOLDER_OPEN,
        "Open…",
        "Open an existing .hnb effect graph",
    )
    .clicked()
    {
        pending.spawn(DialogKind::Open);
    }
    if action_button(
        ui,
        ICON_FILE_IMPORT,
        "Import…",
        "Import a baked EffectAsset .ron",
    )
    .clicked()
    {
        pending.spawn(DialogKind::Import);
    }
}

/// Right column: browser of existing emitters (examples + recents).
fn show_browse_column(
    ui: &mut egui::Ui,
    app: &mut MessageWriter<AppCommand>,
    examples: &[EffectEntry],
    recents: &[EffectEntry],
    thumbnails: &ThumbnailTextures,
    requests: &mut MessageWriter<ThumbnailRequest>,
    clear: &mut MessageWriter<ClearThumbnailCache>,
) {
    ui.add_space(12.0);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(ICON_SPRAY_CAN_SPARKLES.to_string()).size(18.0));
        ui.heading("Browse emitters");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .button(crate::ui::icons::ICON_ARROWS_ROTATE.to_string())
                .on_hover_text("Regenerate thumbnails (clear the cache and re-render)")
                .clicked()
            {
                clear.write(ClearThumbnailCache);
            }
        });
    });
    ui.add_space(8.0);
    ui.separator();
    ui.add_space(8.0);

    egui::ScrollArea::vertical()
        .auto_shrink(false)
        .show(ui, |ui| {
            collapsible_section(ui, "Examples", |ui| {
                if examples.is_empty() {
                    ui.add_space(4.0);
                    ui.weak("No bundled examples found.");
                } else {
                    entry_grid(ui, app, examples, thumbnails, requests);
                }
            });

            ui.add_space(8.0);
            collapsible_section(ui, "Recent", |ui| {
                if recents.is_empty() {
                    ui.add_space(4.0);
                    ui.weak("No recent emitters yet.");
                } else {
                    entry_grid(ui, app, recents, thumbnails, requests);
                }
            });
        });
}

/// A collapsible browser section with a styled header, open by default.
fn collapsible_section(ui: &mut egui::Ui, title: &str, body: impl FnOnce(&mut egui::Ui)) {
    egui::CollapsingHeader::new(egui::RichText::new(title).strong().size(15.0))
        .default_open(true)
        .show(ui, body);
}

/// A full-width, left-aligned action button with an icon.
fn action_button(ui: &mut egui::Ui, icon: char, label: &str, hint: &str) -> egui::Response {
    let text = format!("{icon}   {label}");
    let resp = ui.add_sized(
        [ui.available_width(), 40.0],
        egui::Button::new(egui::RichText::new(text).size(15.0)),
    );
    ui.add_space(4.0);
    resp.on_hover_text(hint)
}

/// A wrapping grid of clickable emitter cards; clicking opens the emitter.
fn entry_grid(
    ui: &mut egui::Ui,
    app: &mut MessageWriter<AppCommand>,
    entries: &[EffectEntry],
    thumbnails: &ThumbnailTextures,
    requests: &mut MessageWriter<ThumbnailRequest>,
) {
    ui.add_space(4.0);
    ui.horizontal_wrapped(|ui| {
        for entry in entries {
            let texture = thumbnails.get(&entry.path).copied();
            if texture.is_none() {
                requests.write(ThumbnailRequest(entry.path.clone()));
            }
            if effect_card(ui, entry, texture).clicked() {
                app.write(AppCommand::OpenFile(entry.path.clone()));
            }
        }
    });
}

/// Draw a single emitter card (thumbnail + name) and sense clicks.
fn effect_card(
    ui: &mut egui::Ui,
    entry: &EffectEntry,
    texture: Option<egui::TextureId>,
) -> egui::Response {
    /// Card outer size in points.
    const CARD: egui::Vec2 = egui::vec2(148.0, 132.0);
    /// Thumbnail height inside the card.
    const THUMB_H: f32 = 96.0;

    let (rect, resp) = ui.allocate_exact_size(CARD, egui::Sense::click());
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    let visuals = ui.style().interact(&resp);

    ui.painter().rect_filled(rect, 6.0, visuals.weak_bg_fill);

    // Thumbnail region: the rendered preview if ready, else a placeholder icon.
    let thumb = egui::Rect::from_min_size(
        rect.min + egui::vec2(6.0, 6.0),
        egui::vec2(rect.width() - 12.0, THUMB_H),
    );
    match texture {
        Some(id) => {
            ui.painter().image(
                id,
                thumb,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }
        None => {
            ui.painter()
                .rect_filled(thumb, 4.0, ui.style().visuals.extreme_bg_color);
            ui.painter().text(
                thumb.center(),
                egui::Align2::CENTER_CENTER,
                ICON_SPRAY_CAN_SPARKLES.to_string(),
                egui::FontId::proportional(28.0),
                ui.style().visuals.weak_text_color(),
            );
        }
    }

    // Name row below the thumbnail, truncated to the card width.
    ui.painter().text(
        egui::pos2(rect.left() + 8.0, thumb.bottom() + 6.0),
        egui::Align2::LEFT_TOP,
        truncate(&entry.name, 18),
        egui::TextStyle::Body.resolve(ui.style()),
        visuals.text_color(),
    );

    resp.on_hover_text(entry.path.display().to_string())
}

/// Truncate a label to `max` chars, appending an ellipsis when cut.
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let head: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{head}…")
    }
}
