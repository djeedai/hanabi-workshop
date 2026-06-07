//! World-space geometry and the world↔screen transform.
//!
//! The graph lives in an "infinite" `f64` world plane. egui paints in
//! `f32` screen space, so we convert at the rendering/hit-test boundary
//! only. Keeping all graph geometry in `f64` avoids precision loss at
//! deep zoom levels.

/// A point in the infinite `f64` world plane.
pub type WorldPos = glam::DVec2;

/// An axis-aligned rectangle in world space, stored as min corner + size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldRect {
    pub min: WorldPos,
    pub width: f64,
    pub height: f64,
}

impl WorldRect {
    pub fn new(min: WorldPos, width: f64, height: f64) -> Self {
        Self { min, width, height }
    }

    pub fn max(&self) -> WorldPos {
        self.min + WorldPos::new(self.width, self.height)
    }

    pub fn contains(&self, p: WorldPos) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max()).all()
    }
}

/// Maps between world coordinates (`f64`) and on-screen egui coordinates
/// (`f32`). `pan` is the world coordinate displayed at `origin` (the
/// canvas top-left); `zoom` is screen-pixels per world-unit.
#[derive(Debug, Clone, Copy)]
pub struct Transform {
    /// Screen position of the canvas top-left corner.
    pub origin: egui::Pos2,
    /// World coordinate shown at `origin`.
    pub pan: WorldPos,
    /// Screen pixels per world unit.
    pub zoom: f64,
}

impl Transform {
    pub fn new(origin: egui::Pos2, pan: WorldPos, zoom: f64) -> Self {
        Self { origin, pan, zoom }
    }

    pub fn world_to_screen(&self, w: WorldPos) -> egui::Pos2 {
        let d = ((w - self.pan) * self.zoom).as_vec2();
        self.origin + egui::vec2(d.x, d.y)
    }

    pub fn screen_to_world(&self, s: egui::Pos2) -> WorldPos {
        self.pan + self.screen_vec_to_world(s - self.origin)
    }

    pub fn world_len_to_screen(&self, len: f64) -> f32 {
        (len * self.zoom) as f32
    }

    pub fn screen_len_to_world(&self, len: f32) -> f64 {
        len as f64 / self.zoom
    }

    /// Convert a screen-space delta (egui `f32`) into a world-space vector.
    pub fn screen_vec_to_world(&self, v: egui::Vec2) -> WorldPos {
        glam::vec2(v.x, v.y).as_dvec2() / self.zoom
    }

    pub fn world_rect_to_screen(&self, r: WorldRect) -> egui::Rect {
        egui::Rect::from_min_max(self.world_to_screen(r.min), self.world_to_screen(r.max()))
    }
}
