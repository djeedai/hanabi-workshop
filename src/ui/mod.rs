//! Top-level editor UI: menu bar + nested document dock.

use std::collections::{HashMap, HashSet};

use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};
use egui_dock::{DockArea, DockState, Style};

mod document_tabs;
pub mod graph_validation;
mod home;
pub mod icons;
pub use hanabi_effect_graph::modifier_names;
mod panels;
mod shortcuts;

pub use shortcuts::{handle_file_shortcuts, handle_history_shortcuts};

use crate::document::{
    ActiveDocument, DocumentRoot, DocumentViewports, FocusDocument, PanelKind, ViewportSizeRequests,
};

/// A tab in the outer dock: the singleton Home landing tab, or a document.
///
/// The outer dock is otherwise document-centric, but Home is a non-document
/// UI surface (no [`DocumentContent`], history, viewport, or render layer). It
/// is seeded once, always present, and not closable.
///
/// [`DocumentContent`]: crate::document::DocumentContent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OuterTab {
    /// The landing tab shown on startup: create actions + effect browser.
    Home,
    /// An open document, identified by its entity.
    Document(Entity),
}

impl OuterTab {
    /// The document entity this tab refers to, or `None` for
    /// [`OuterTab::Home`].
    pub fn document(self) -> Option<Entity> {
        match self {
            OuterTab::Home => None,
            OuterTab::Document(e) => Some(e),
        }
    }
}

/// Outer dock that hosts the Home tab plus one tab per open document.
///
/// Tabs may be torn off into floating windows for side-by-side document
/// comparison. The Home tab is seeded by default and never removed.
#[derive(Resource)]
pub struct DocumentDock {
    pub state: DockState<OuterTab>,
}

impl Default for DocumentDock {
    fn default() -> Self {
        Self {
            state: DockState::new(vec![OuterTab::Home]),
        }
    }
}

/// Egui pass: draws the menu and the outer document dock.
///
/// The inner [`DocumentTabViewer`] holds `&mut` references to the
/// queries — each call to `TabViewer::ui()` / `TabViewer::title()`
/// acquires its own short-lived `Mut<DocumentUi>` / `Mut<PlaybackState>`
/// for the tab being rendered and drops it on return, so successive
/// tabs don't conflict.
///
/// [`DocumentTabViewer`]: crate::ui::document_tabs::DocumentTabViewer
pub fn draw_editor_ui(
    mut contexts: EguiContexts,
    mut document_dock: ResMut<DocumentDock>,
    viewports: Res<DocumentViewports>,
    mut size_requests: ResMut<ViewportSizeRequests>,
    document_root: Option<Res<DocumentRoot>>,
    active: ResMut<ActiveDocument>,
    mut focus_reader: MessageReader<FocusDocument>,
    mut pending_dialogs: ResMut<crate::app_commands::PendingFileDialogs>,
    root_children: Query<&Children>,
    mut tab_data: document_tabs::TabViewerData,
    thumbnails: Res<crate::thumbnail::ThumbnailCache>,
    mut history_writer: bevy::ecs::message::MessageWriter<crate::edits::HistoryRequest>,
) -> Result {
    let Some(root) = document_root else {
        return Ok(());
    };
    let ordered_docs: Vec<Entity> = root_children
        .get(root.0)
        .map(|c| c.iter().collect())
        .unwrap_or_default();

    sync_document_tabs(&mut document_dock, &ordered_docs);

    // Honor one-shot focus requests (a document was just opened/created, or a
    // re-open was redirected to an already-open document): move dock focus to
    // the target tab so it becomes visible and active. The last request wins.
    if let Some(FocusDocument(target)) = focus_reader.read().last().copied()
        && let Some(path) = document_dock.state.find_tab(&OuterTab::Document(target))
    {
        document_dock
            .state
            .set_focused_node_and_surface(egui_dock::NodePath {
                surface: path.surface,
                node: path.node,
            });
        let _ = document_dock.state.set_active_tab(path);
    }

    let viewport_textures = resolve_viewport_textures(&mut contexts, &viewports);
    let thumbnail_textures = resolve_thumbnail_textures(&mut contexts, &thumbnails);

    // Document the menus act on: the tab the outer dock is actually showing.
    // Prefer the focused tab, fall back to the active (displayed) tab of the
    // first leaf, then to `ActiveDocument`, then to the first open document.
    // `ActiveDocument` alone is unreliable here — it lags a frame behind, and a
    // freshly opened document may be displayed before it has keyboard focus.
    let displayed_doc = document_dock
        .state
        .find_active_focused()
        .and_then(|(_, t)| t.document())
        .or_else(|| {
            document_dock
                .state
                .main_surface_mut()
                .find_active()
                .and_then(|(_, t)| t.document())
        })
        .or(active.0)
        .or_else(|| ordered_docs.first().copied());

    let active_has_path = displayed_doc
        .and_then(|e| tab_data.docs.get(e).ok())
        .map(|(c, _, _, _)| c.path().is_some())
        .unwrap_or(false);

    // Mutable handle to the displayed document's inner dock, so the View menu
    // can list its panels and re-open ones the user has closed. Borrows a
    // disjoint field of `tab_data` from `app` below, so both can be passed to
    // the menu.
    let mut active_ui = displayed_doc
        .and_then(|e| tab_data.docs.get_mut(e).ok())
        .map(|(_, ui, _, _)| ui);

    let ctx = contexts.ctx_mut()?;
    let mut root_ui = egui::Ui::new(
        ctx.clone(),
        "editor_root".into(),
        egui::UiBuilder::new()
            .layer_id(egui::LayerId::background())
            .max_rect(ctx.viewport_rect()),
    );
    draw_menu_bar(
        &mut root_ui,
        &mut tab_data.app,
        &mut pending_dialogs,
        &mut history_writer,
        displayed_doc,
        active_has_path,
        active_ui.as_deref_mut().map(|ui| &mut ui.dock),
    );
    drop(active_ui);

    let mut tab_viewer = document_tabs::DocumentTabViewer {
        data: &mut tab_data,
        viewport_textures: &viewport_textures,
        thumbnail_textures: &thumbnail_textures,
        size_requests: &mut size_requests,
        pending_dialogs: &mut pending_dialogs,
    };
    let mut dock_style = dock_style_for(ctx.global_style().as_ref());
    // Paint the *outer* document tab's body in the same `extreme_bg_color`
    // as the splitter gutters so the frame around each document's inner
    // dock blends with the gutter background. Inner panel tab-bodies keep
    // their default mid-gray (set via `dock_style_for` on the inner dock).
    dock_style.tab.tab_body.bg_fill = ctx.global_style().visuals.extreme_bg_color;
    // Zero the outer tab body's inner margin so the playback toolbar and
    // inner panel dock sit flush against the tab edges; the only inset
    // between them is our 6 px extreme-bg gutter strip.
    dock_style.tab.tab_body.inner_margin = egui::Margin::ZERO;
    DockArea::new(&mut document_dock.state)
        .style(dock_style)
        .show_leaf_collapse_buttons(false)
        .show_leaf_close_all_buttons(false)
        .show_inside(&mut root_ui, &mut tab_viewer);

    // Sync the displayed outer tab into ActiveDocument. Falls back from the
    // focused tab to the first leaf's active tab so the active document tracks
    // what's actually on screen even before a tab gains keyboard focus; resolves
    // to `None` only when no documents remain.
    let displayed = document_dock
        .state
        .find_active_focused()
        .and_then(|(_, tab)| tab.document())
        .or_else(|| {
            document_dock
                .state
                .main_surface_mut()
                .find_active()
                .and_then(|(_, tab)| tab.document())
        });
    let mut active = active;
    if active.0 != displayed {
        active.0 = displayed;
    }

    Ok(())
}

/// Shared egui_dock style for the document and panel docks.
///
/// Removes per-tab outlines, the hairline below the tab bar, the outer dock
/// border, and the tab-body stroke; adds a hover background highlight on tabs.
/// Used by both the outer document dock and each document's inner panel dock so
/// they feel visually consistent.
pub(crate) fn dock_style_for(style: &egui::Style) -> Style {
    let mut s = Style::from_egui(style);
    s.main_surface_border_stroke = egui::Stroke::NONE;
    s.tab.tab_body.stroke = egui::Stroke::NONE;
    s.tab_bar.hline_color = egui::Color32::TRANSPARENT;
    for ts in [
        &mut s.tab.active,
        &mut s.tab.inactive,
        &mut s.tab.focused,
        &mut s.tab.hovered,
        &mut s.tab.active_with_kb_focus,
        &mut s.tab.inactive_with_kb_focus,
        &mut s.tab.focused_with_kb_focus,
    ] {
        ts.outline_color = egui::Color32::TRANSPARENT;
    }
    s.tab.hovered.bg_fill = style.visuals.widgets.hovered.bg_fill;
    // Make the separator "gutter" between docked panels blend with the
    // menu/tab-bar background. Keep hover/drag colors visible so users can
    // still find and drag the splitter.
    s.separator.color_idle = style.visuals.extreme_bg_color;
    s.separator.width = 6.0;
    s
}

/// Panels offered by the View menu, in display order.
///
/// Each entry maps a [`PanelKind`] to its menu label. The menu uses these to
/// toggle panels in the active document's dock so a closed panel can be
/// re-opened.
const PANEL_MENU_ENTRIES: &[(PanelKind, &str)] = &[
    (PanelKind::Viewport(0), "Viewport"),
    (PanelKind::Graph, "Graph"),
    (PanelKind::Effect, "Effect"),
    (PanelKind::Properties, "Properties"),
    (PanelKind::Material, "Material"),
    (PanelKind::Shaders, "Shaders"),
];

fn draw_menu_bar(
    root_ui: &mut egui::Ui,
    app: &mut bevy::ecs::message::MessageWriter<crate::app_commands::AppCommand>,
    pending: &mut crate::app_commands::PendingFileDialogs,
    history: &mut bevy::ecs::message::MessageWriter<crate::edits::HistoryRequest>,
    active: Option<Entity>,
    active_has_path: bool,
    active_dock: Option<&mut DockState<PanelKind>>,
) {
    use crate::{
        app_commands::{AppCommand, DialogKind},
        edits::HistoryRequest,
    };

    // Minimum popup width so short labels don't produce cramped menus.
    const MENU_MIN_WIDTH: f32 = 180.0;

    // Match the dock's tab-bar background (egui's extreme_bg_color, also used
    // by egui_dock for the empty area beside tabs) and drop the default
    // bottom stroke so there's no visible seam between the menu and tabs.
    let visuals = &root_ui.style().visuals;
    let menu_frame = egui::Frame::default()
        .fill(visuals.extreme_bg_color)
        .inner_margin(egui::Margin::symmetric(8, 2))
        .stroke(egui::Stroke::NONE);
    egui::Panel::top("menu_bar")
        .frame(menu_frame)
        .show_separator_line(false)
        .show_inside(root_ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                // Widen the horizontal button padding so top-level menu
                // entries aren't visually cramped against each other.
                ui.spacing_mut().button_padding.x += 8.0;
                let (file_btn, _) = egui::containers::menu::MenuButton::new("File")
                    .config(egui::containers::menu::MenuConfig::new().style(menu_popup_style))
                    .ui(ui, |ui| {
                        ui.set_min_width(MENU_MIN_WIDTH);
                        if menu_item(
                            ui,
                            Some(icons::ICON_FILE),
                            "New",
                            shortcut_label(false, "N"),
                        )
                        .clicked()
                        {
                            app.write(AppCommand::NewDocument);
                            ui.close();
                        }
                        if menu_item(
                            ui,
                            Some(icons::ICON_FOLDER_OPEN),
                            "Open…",
                            shortcut_label(false, "O"),
                        )
                        .clicked()
                        {
                            pending.spawn(DialogKind::Open);
                            ui.close();
                        }
                        if menu_item(
                            ui,
                            Some(icons::ICON_FILE_IMPORT),
                            "Import…",
                            shortcut_label(true, "O"),
                        )
                        .clicked()
                        {
                            pending.spawn(DialogKind::Import);
                            ui.close();
                        }
                        ui.add_enabled_ui(active.is_some(), |ui| {
                            let save_btn = menu_item(
                                ui,
                                Some(icons::ICON_FLOPPY_DISK),
                                "Save",
                                shortcut_label(false, "S"),
                            );
                            if save_btn.clicked() {
                                // No path yet (never saved) falls back to Save As.
                                if active_has_path {
                                    app.write(AppCommand::SaveActive);
                                } else {
                                    pending.spawn(DialogKind::SaveAs);
                                }
                                ui.close();
                            }
                            if menu_item(ui, None, "Save As…", shortcut_label(true, "S")).clicked()
                            {
                                pending.spawn(DialogKind::SaveAs);
                                ui.close();
                            }
                            ui.separator();
                            if menu_item(
                                ui,
                                Some(icons::ICON_XMARK),
                                "Close",
                                shortcut_label(false, "W"),
                            )
                            .clicked()
                            {
                                if let Some(e) = active {
                                    app.write(AppCommand::RequestCloseDocument(e));
                                }
                                ui.close();
                            }
                        });
                        ui.separator();
                        if menu_item(
                            ui,
                            Some(icons::ICON_RIGHT_FROM_BRACKET),
                            "Exit",
                            shortcut_label(false, "Q"),
                        )
                        .clicked()
                        {
                            app.write(AppCommand::RequestQuit);
                            ui.close();
                        }
                    });
                let (edit_btn, _) = egui::containers::menu::MenuButton::new("Edit")
                    .config(egui::containers::menu::MenuConfig::new().style(menu_popup_style))
                    .ui(ui, |ui| {
                        ui.set_min_width(MENU_MIN_WIDTH);
                        ui.add_enabled_ui(active.is_some(), |ui| {
                            if menu_item(
                                ui,
                                Some(icons::ICON_ARROW_ROTATE_LEFT),
                                "Undo",
                                shortcut_label(false, "Z"),
                            )
                            .clicked()
                            {
                                if let Some(e) = active {
                                    history.write(HistoryRequest::Undo(e));
                                }
                                ui.close();
                            }
                            if menu_item(
                                ui,
                                Some(icons::ICON_ARROW_ROTATE_RIGHT),
                                "Redo",
                                shortcut_label(true, "Z"),
                            )
                            .clicked()
                            {
                                if let Some(e) = active {
                                    history.write(HistoryRequest::Redo(e));
                                }
                                ui.close();
                            }
                        });
                    });
                // Keep the View menu open while toggling panel checkboxes;
                // close it only when the user clicks outside.
                let (view_btn, _) = egui::containers::menu::MenuButton::new("View")
                    .config(
                        egui::containers::menu::MenuConfig::new()
                            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                            .style(menu_popup_style),
                    )
                    .ui(ui, |ui| {
                        ui.set_min_width(MENU_MIN_WIDTH);
                        if let Some(dock) = active_dock {
                            ui.label("Panels");
                            for (panel, label) in PANEL_MENU_ENTRIES {
                                let location = dock.find_tab(panel);
                                let mut open = location.is_some();
                                let text = format!("{}  {label}", panels::panel_icon(panel));
                                if ui.checkbox(&mut open, text).clicked() {
                                    match location {
                                        Some(loc) => {
                                            dock.remove_tab(loc);
                                        }
                                        None => dock.push_to_focused_leaf(panel.clone()),
                                    }
                                }
                            }
                        } else {
                            ui.label("No document open");
                        }
                    });
                // egui opens top-level menu-bar entries on click only (submenus
                // open on hover). Restore the conventional bar behaviour: once
                // one menu is open, hovering a sibling entry switches to it.
                switch_menu_bar_on_hover(ui, &[&file_btn, &edit_btn, &view_btn]);
            });
        });
}

/// Formats a menu accelerator label for the current platform.
///
/// macOS uses the native symbol form with no separators (e.g. `⇧⌘O`); other
/// platforms use the `Ctrl+Shift+O` style. `shift` prepends the Shift modifier.
fn shortcut_label(shift: bool, key: &str) -> String {
    if cfg!(target_os = "macos") {
        format!("{}⌘{key}", if shift { "⇧" } else { "" })
    } else {
        format!("Ctrl+{}{key}", if shift { "Shift+" } else { "" })
    }
}

/// Fixed width of the leading icon gutter in menu-popup entries, in points.
const MENU_ICON_GUTTER: f32 = 16.0;

/// Adds a menu-popup entry with a leading icon gutter and trailing shortcut.
///
/// Every entry reserves the same gutter on the left, drawing `icon` when given
/// and blank space otherwise, so labels line up whether or not they have an
/// icon. Returns the button [`Response`] so callers can test `.clicked()`.
///
/// [`Response`]: egui::Response
fn menu_item(
    ui: &mut egui::Ui,
    icon: Option<char>,
    label: &str,
    shortcut: String,
) -> egui::Response {
    use egui::AtomExt as _;

    let height = ui.text_style_height(&egui::TextStyle::Button);
    let glyph = icon.map(|c| c.to_string()).unwrap_or_default();
    let gutter = egui::RichText::new(glyph).atom_size(egui::vec2(MENU_ICON_GUTTER, height));
    ui.add(egui::Button::new((gutter, label)).shortcut_text(shortcut))
}

/// Styles menu popups: a darker fill and no border.
///
/// Composed on top of egui's default [`menu_style`] so button transparency and
/// stroke removal inside the menu are preserved; only the popup frame's fill
/// and outer stroke are overridden.
///
/// [`menu_style`]: egui::containers::menu::menu_style
fn menu_popup_style(style: &mut egui::Style) {
    egui::containers::menu::menu_style(style);
    style.visuals.window_stroke = egui::Stroke::NONE;
    style.visuals.window_fill = darken_popup_fill(style.visuals.window_fill);
}

/// Darkens a popup fill colour while keeping it fully opaque.
///
/// Scales only the RGB channels so the surface stays readable over any
/// background — [`egui::Color32::gamma_multiply`] would also scale alpha.
pub(crate) fn darken_popup_fill(fill: egui::Color32) -> egui::Color32 {
    egui::Color32::from_rgb(
        (fill.r() as f32 * 0.6) as u8,
        (fill.g() as f32 * 0.6) as u8,
        (fill.b() as f32 * 0.6) as u8,
    )
}

/// Switches the open menu-bar entry to whichever one the pointer is over.
///
/// egui's top-level [`MenuButton`] only toggles its popup on click, unlike
/// submenu buttons which also open on hover. This restores the conventional
/// menu-bar behaviour: while one entry's menu is open, moving the pointer onto
/// a sibling entry opens that one instead. Opening a popup implicitly closes
/// the previously open one, since at most one popup is open at a time.
///
/// [`MenuButton`]: egui::containers::menu::MenuButton
fn switch_menu_bar_on_hover(ui: &egui::Ui, buttons: &[&egui::Response]) {
    let ctx = ui.ctx();
    let Some(pointer) = ctx.pointer_hover_pos() else {
        return;
    };
    // Only switch while a menu is already open, so a plain hover (no menu open)
    // still requires a click to open the first menu.
    let popup_ids: Vec<egui::Id> = buttons.iter().map(|b| b.id.with("popup")).collect();
    let any_open = popup_ids.iter().any(|id| egui::Popup::is_id_open(ctx, *id));
    if !any_open {
        return;
    }
    for (button, popup_id) in buttons.iter().zip(&popup_ids) {
        if button.interact_rect.contains(pointer) && !egui::Popup::is_id_open(ctx, *popup_id) {
            egui::Popup::open_id(ctx, *popup_id);
            break;
        }
    }
}

/// Ensures the outer dock's tabs match the current set of documents.
///
/// The Home tab is never added or removed here — it is seeded by
/// [`DocumentDock::default`] and kept for the lifetime of the app.
fn sync_document_tabs(dock: &mut DocumentDock, ordered: &[Entity]) {
    let current: HashSet<Entity> = dock
        .state
        .iter_all_tabs()
        .filter_map(|(_, t)| t.document())
        .collect();
    let wanted: HashSet<Entity> = ordered.iter().copied().collect();

    for doc in ordered {
        if !current.contains(doc) {
            dock.state.push_to_focused_leaf(OuterTab::Document(*doc));
        }
    }

    let stale: Vec<Entity> = current.difference(&wanted).copied().collect();
    for doc in stale {
        // Re-find each stale document's current path rather than reusing a
        // pre-collected one: removing a tab shifts the indices of later tabs in
        // the same node, and a hardcoded `TabIndex(0)` would remove whatever is
        // first in the leaf — including the Home tab.
        while let Some(path) = dock.state.find_tab(&OuterTab::Document(doc)) {
            dock.state.remove_tab(path);
        }
    }
}

fn resolve_viewport_textures(
    contexts: &mut EguiContexts,
    viewports: &DocumentViewports,
) -> HashMap<(Entity, usize), egui::TextureId> {
    let mut out = HashMap::new();
    for (doc, slots) in &viewports.by_doc {
        for (vp_idx, handle) in &slots.images {
            if let Some(tex) = contexts.image_id(handle) {
                out.insert((*doc, *vp_idx), tex);
            }
        }
    }
    out
}

/// Resolve egui texture ids for every ready thumbnail, keyed by effect path.
fn resolve_thumbnail_textures(
    contexts: &mut EguiContexts,
    thumbnails: &crate::thumbnail::ThumbnailCache,
) -> crate::ui::home::ThumbnailTextures {
    let mut out = crate::ui::home::ThumbnailTextures::new();
    for (path, handle) in thumbnails.ready_handles() {
        if let Some(tex) = contexts.image_id(handle) {
            out.insert(path.clone(), tex);
        }
    }
    out
}
