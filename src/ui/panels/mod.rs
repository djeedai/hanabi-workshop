use bevy_egui::egui;
use egui_dock::TabViewer;

use crate::ui::EditorTab;

mod effect_list;
mod properties;
mod viewport;

pub struct EditorTabViewer<'a> {
    pub viewport_textures: &'a [egui::TextureId],
}

impl<'a> TabViewer for EditorTabViewer<'a> {
    type Tab = EditorTab;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        match tab {
            EditorTab::Viewport(i) => format!("Viewport {}", i).into(),
            EditorTab::EffectList => "Effects".into(),
            EditorTab::Properties => "Properties".into(),
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        match tab {
            EditorTab::Viewport(i) => {
                if let Some(tex) = self.viewport_textures.get(*i).copied() {
                    viewport::show(ui, tex);
                } else {
                    ui.label(format!("No texture bound for viewport {}", i));
                }
            }
            EditorTab::EffectList => effect_list::show(ui),
            EditorTab::Properties => properties::show(ui),
        }
    }
}
