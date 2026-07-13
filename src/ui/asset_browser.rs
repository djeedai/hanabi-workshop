//! Shared texture catalog browser UI.
//!
//! The dock panel and Image-node picker use the same searchable,
//! source-filtered list/grid renderer. Workshop owns the typed drag payload;
//! the reusable graph widget only reports domain-neutral drop targets.

use std::path::PathBuf;

use bevy::{asset::AssetPath, ecs::message::MessageWriter, prelude::AssetServer};
use bevy_egui::egui;

use crate::{
    app_commands::{DialogKind, PendingFileDialogs},
    asset_library::{
        TextureCatalog, TextureEntry, TextureLibraryCommand, TextureLibrarySettings, TextureSource,
        TextureViewMode,
    },
    texture_preview::{TexturePreviewCache, TexturePreviewState},
};

/// Payload carried while a texture card is dragged.
#[derive(Debug, Clone)]
pub struct TextureDragPayload {
    pub asset_path: AssetPath<'static>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
enum SourceFilter {
    #[default]
    All,
    Preset,
    Project,
    External(PathBuf),
}

#[derive(Debug, Clone, Default)]
struct BrowserState {
    search: String,
    source: SourceFilter,
}

/// Rendering options for one browser instance.
#[derive(Debug, Clone, Copy)]
pub struct BrowserOptions {
    pub id_salt: egui::Id,
    pub manage_sources: bool,
    pub draggable: bool,
    pub selectable: bool,
}

/// Draw a texture catalog and return a clicked texture, if selection is
/// enabled.
#[allow(clippy::too_many_arguments)]
pub fn show(
    ui: &mut egui::Ui,
    catalog: &TextureCatalog,
    settings: &TextureLibrarySettings,
    previews: &mut TexturePreviewCache,
    asset_server: &AssetServer,
    commands: &mut MessageWriter<TextureLibraryCommand>,
    pending: &mut PendingFileDialogs,
    options: BrowserOptions,
) -> Option<AssetPath<'static>> {
    let state_id = ui.make_persistent_id((options.id_salt, "state"));
    let mut state: BrowserState = ui
        .ctx()
        .data_mut(|data| data.get_temp(state_id))
        .unwrap_or_default();

    browser_toolbar(
        ui,
        settings,
        commands,
        pending,
        options.manage_sources,
        &mut state,
    );
    ui.separator();

    let needle = state.search.trim().to_lowercase();
    let entries: Vec<&TextureEntry> = catalog
        .entries
        .iter()
        .filter(|entry| source_matches(&state.source, &entry.source))
        .filter(|entry| {
            needle.is_empty()
                || entry.display_name.to_lowercase().contains(&needle)
                || entry
                    .relative_display_path
                    .to_string_lossy()
                    .to_lowercase()
                    .contains(&needle)
        })
        .collect();

    let selected = match settings.view_mode {
        TextureViewMode::List => show_list(
            ui,
            options.id_salt,
            &entries,
            previews,
            asset_server,
            options.draggable,
            options.selectable,
        ),
        TextureViewMode::Small => show_grid(
            ui,
            options.id_salt,
            &entries,
            previews,
            asset_server,
            88.0,
            82.0,
            options.draggable,
            options.selectable,
        ),
        TextureViewMode::Large => show_grid(
            ui,
            options.id_salt,
            &entries,
            previews,
            asset_server,
            144.0,
            132.0,
            options.draggable,
            options.selectable,
        ),
    };

    if entries.is_empty() {
        ui.centered_and_justified(|ui| ui.weak("No matching textures"));
    }
    ui.ctx().data_mut(|data| data.insert_temp(state_id, state));
    selected
}

fn browser_toolbar(
    ui: &mut egui::Ui,
    settings: &TextureLibrarySettings,
    commands: &mut MessageWriter<TextureLibraryCommand>,
    pending: &mut PendingFileDialogs,
    manage_sources: bool,
    state: &mut BrowserState,
) {
    use crate::ui::icons::{
        ICON_ARROWS_ROTATE, ICON_FOLDER_PLUS, ICON_GRIP, ICON_LIST, ICON_TABLE_CELLS_LARGE,
    };

    ui.horizontal_wrapped(|ui| {
        let controls_width = if manage_sources {
            ui.spacing().interact_size.x * 2.0 + ui.spacing().item_spacing.x * 2.0
        } else {
            0.0
        };
        let search_width = (ui.available_width() - controls_width).max(1.0);
        ui.add_sized(
            egui::vec2(search_width, ui.spacing().interact_size.y),
            egui::TextEdit::singleline(&mut state.search)
                .hint_text("Search textures")
                .desired_width(search_width),
        );
        if manage_sources
            && ui
                .button(ICON_FOLDER_PLUS.to_string())
                .on_hover_text("Add external folder")
                .clicked()
        {
            pending.spawn(DialogKind::AddTextureFolder);
        }
        if manage_sources
            && ui
                .button(ICON_ARROWS_ROTATE.to_string())
                .on_hover_text("Rescan texture folders")
                .clicked()
        {
            commands.write(TextureLibraryCommand::Rescan);
        }
    });
    ui.horizontal_wrapped(|ui| {
        source_filter(ui, settings, commands, &mut state.source, manage_sources);
        for (mode, icon, tooltip) in [
            (
                TextureViewMode::Large,
                ICON_TABLE_CELLS_LARGE,
                "Large thumbnails",
            ),
            (TextureViewMode::Small, ICON_GRIP, "Small thumbnails"),
            (TextureViewMode::List, ICON_LIST, "List"),
        ] {
            if ui
                .selectable_label(settings.view_mode == mode, icon.to_string())
                .on_hover_text(tooltip)
                .clicked()
            {
                commands.write(TextureLibraryCommand::SetViewMode(mode));
            }
        }
    });
}

fn source_filter(
    ui: &mut egui::Ui,
    settings: &TextureLibrarySettings,
    commands: &mut MessageWriter<TextureLibraryCommand>,
    filter: &mut SourceFilter,
    manage_sources: bool,
) {
    let current = match filter {
        SourceFilter::All => "All".to_string(),
        SourceFilter::Preset => "Presets".to_string(),
        SourceFilter::Project => "Project".to_string(),
        SourceFilter::External(root) => root
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("External")
            .to_string(),
    };
    let combo_width = ui.available_width().clamp(1.0, 160.0);
    egui::ComboBox::from_id_salt(("texture-source", ui.id()))
        .selected_text(current)
        .width(combo_width)
        .show_ui(ui, |ui| {
            ui.selectable_value(filter, SourceFilter::All, "All");
            ui.selectable_value(filter, SourceFilter::Preset, "Presets");
            ui.selectable_value(filter, SourceFilter::Project, "Project");
            for root in &settings.external_roots {
                ui.selectable_value(
                    filter,
                    SourceFilter::External(root.clone()),
                    root.file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("External"),
                );
            }
        });

    if manage_sources {
        egui::containers::menu::MenuButton::new("Folders").ui(ui, |ui| {
            if settings.external_roots.is_empty() {
                ui.weak("No external folders");
            }
            for root in &settings.external_roots {
                ui.horizontal(|ui| {
                    ui.label(root.display().to_string());
                    if ui.small_button("Remove").clicked() {
                        if *filter == SourceFilter::External(root.clone()) {
                            *filter = SourceFilter::All;
                        }
                        commands.write(TextureLibraryCommand::RemoveExternalRoot(root.clone()));
                        ui.close();
                    }
                });
            }
        });
    }
}

fn source_matches(filter: &SourceFilter, source: &TextureSource) -> bool {
    match (filter, source) {
        (SourceFilter::All, _) => true,
        (SourceFilter::Preset, TextureSource::Preset)
        | (SourceFilter::Project, TextureSource::Project) => true,
        (SourceFilter::External(left), TextureSource::External(right)) => left == right,
        _ => false,
    }
}

fn show_list(
    ui: &mut egui::Ui,
    browser_id: egui::Id,
    entries: &[&TextureEntry],
    previews: &mut TexturePreviewCache,
    asset_server: &AssetServer,
    draggable: bool,
    selectable: bool,
) -> Option<AssetPath<'static>> {
    let mut selected = None;
    egui::ScrollArea::vertical().show_rows(ui, 36.0, entries.len(), |ui, range| {
        for index in range {
            let entry = entries[index];
            let response = drag_source(ui, browser_id, entry, draggable, |ui| {
                texture_row(ui, entry, previews, asset_server)
            });
            if selectable && response.clicked() {
                selected = Some(entry.asset_path.clone_owned());
            }
        }
    });
    selected
}

#[allow(clippy::too_many_arguments)]
fn show_grid(
    ui: &mut egui::Ui,
    browser_id: egui::Id,
    entries: &[&TextureEntry],
    previews: &mut TexturePreviewCache,
    asset_server: &AssetServer,
    card_width: f32,
    preview_height: f32,
    draggable: bool,
    selectable: bool,
) -> Option<AssetPath<'static>> {
    let available_width = ui.available_width().max(1.0);
    let scale = (available_width / card_width).min(1.0);
    let card_width = card_width * scale;
    let preview_height = preview_height * scale;
    let gap = ui.spacing().item_spacing.x;
    let columns = ((ui.available_width() + gap) / (card_width + gap))
        .floor()
        .max(1.0) as usize;
    let rows = entries.len().div_ceil(columns);
    let row_height = preview_height + 42.0;
    let mut selected = None;
    egui::ScrollArea::vertical().show_rows(ui, row_height, rows, |ui, range| {
        for row in range {
            ui.horizontal(|ui| {
                for column in 0..columns {
                    let index = row * columns + column;
                    let Some(entry) = entries.get(index).copied() else {
                        break;
                    };
                    let response = drag_source(ui, browser_id, entry, draggable, |ui| {
                        texture_card(
                            ui,
                            entry,
                            previews,
                            asset_server,
                            card_width,
                            preview_height,
                        )
                    });
                    if selectable && response.clicked() {
                        selected = Some(entry.asset_path.clone_owned());
                    }
                }
            });
        }
    });
    selected
}

fn drag_source(
    ui: &mut egui::Ui,
    browser_id: egui::Id,
    entry: &TextureEntry,
    draggable: bool,
    draw: impl FnOnce(&mut egui::Ui) -> egui::Response,
) -> egui::Response {
    let entry_id = browser_id.with(("texture-entry", &entry.canonical_path));
    ui.scope_builder(egui::UiBuilder::new().id(entry_id), |ui| {
        if draggable {
            ui.dnd_drag_source(
                entry_id.with("drag"),
                TextureDragPayload {
                    asset_path: entry.asset_path.clone_owned(),
                },
                draw,
            )
            .inner
        } else {
            draw(ui)
        }
    })
    .inner
}

fn texture_row(
    ui: &mut egui::Ui,
    entry: &TextureEntry,
    previews: &mut TexturePreviewCache,
    asset_server: &AssetServer,
) -> egui::Response {
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 34.0), egui::Sense::click());
    let visuals = ui.style().interact(&response);
    if response.hovered() {
        ui.painter()
            .rect_filled(rect, visuals.corner_radius, visuals.weak_bg_fill);
    }
    let preview =
        egui::Rect::from_min_size(rect.min + egui::vec2(3.0, 3.0), egui::vec2(28.0, 28.0));
    paint_preview(ui, preview, entry, previews, asset_server);
    ui.painter().text(
        egui::pos2(preview.right() + 7.0, rect.center().y - 7.0),
        egui::Align2::LEFT_CENTER,
        &entry.display_name,
        egui::TextStyle::Body.resolve(ui.style()),
        visuals.text_color(),
    );
    ui.painter().text(
        egui::pos2(preview.right() + 7.0, rect.center().y + 8.0),
        egui::Align2::LEFT_CENTER,
        entry.relative_display_path.display().to_string(),
        egui::TextStyle::Small.resolve(ui.style()),
        ui.visuals().weak_text_color(),
    );
    response.on_hover_text(entry.canonical_path.display().to_string())
}

fn texture_card(
    ui: &mut egui::Ui,
    entry: &TextureEntry,
    previews: &mut TexturePreviewCache,
    asset_server: &AssetServer,
    width: f32,
    preview_height: f32,
) -> egui::Response {
    let size = egui::vec2(width, preview_height + 36.0);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    let visuals = ui.style().interact(&response);
    ui.painter()
        .rect_filled(rect, visuals.corner_radius, visuals.weak_bg_fill);
    let preview = egui::Rect::from_min_size(
        rect.min + egui::vec2(4.0, 4.0),
        egui::vec2((width - 8.0).max(1.0), preview_height),
    );
    paint_preview(ui, preview, entry, previews, asset_server);
    ui.painter().text(
        egui::pos2(rect.left() + 6.0, preview.bottom() + 9.0),
        egui::Align2::LEFT_CENTER,
        truncate(&entry.display_name, if width < 100.0 { 12 } else { 22 }),
        egui::TextStyle::Body.resolve(ui.style()),
        visuals.text_color(),
    );
    response.on_hover_text(format!(
        "{}\n{}",
        entry.relative_display_path.display(),
        source_label(&entry.source)
    ))
}

fn paint_preview(
    ui: &egui::Ui,
    rect: egui::Rect,
    entry: &TextureEntry,
    previews: &mut TexturePreviewCache,
    asset_server: &AssetServer,
) {
    ui.painter()
        .rect_filled(rect, 3.0, ui.visuals().extreme_bg_color);
    let state = previews.request_path(asset_server, &entry.asset_path);
    match state {
        TexturePreviewState::Ready(preview) => {
            ui.painter().image(
                preview.texture_id,
                rect,
                egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }
        TexturePreviewState::Failed { .. } => paint_status(ui, rect, "Missing"),
        TexturePreviewState::Loading { .. } => paint_status(ui, rect, "..."),
    }
}

fn paint_status(ui: &egui::Ui, rect: egui::Rect, text: &str) {
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        text,
        egui::TextStyle::Small.resolve(ui.style()),
        ui.visuals().weak_text_color(),
    );
}

fn source_label(source: &TextureSource) -> &str {
    match source {
        TextureSource::Preset => "Preset",
        TextureSource::Project => "Project",
        TextureSource::External(_) => "External",
    }
}

fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut value: String = text.chars().take(max.saturating_sub(1)).collect();
    value.push('…');
    value
}
