//! Spline geometry for graph edges: cubic Béziers with tangents along the
//! flow axis — horizontal for the conventional left-to-right node links,
//! vertical for the top-to-bottom connections between stacked blocks.

use egui::epaint::{CubicBezierShape, PathStroke};
use egui::{Color32, Pos2, Stroke};

/// Swap a point's x and y. A vertical link is just a horizontal one
/// reflected across the diagonal.
fn swap_xy(p: Pos2) -> Pos2 {
    Pos2::new(p.y, p.x)
}

/// Control points `[from, c1, c2, to]` for a link with horizontal tangents.
fn horizontal_ctrl(from: Pos2, to: Pos2) -> [Pos2; 4] {
    let dx = (to.x - from.x).abs();
    // Tangent length grows with horizontal separation so close ports get
    // a gentle curve and distant ones a pronounced S.
    let handle = (dx * 0.5).clamp(24.0, 160.0);
    [
        from,
        Pos2::new(from.x + handle, from.y),
        Pos2::new(to.x - handle, to.y),
        to,
    ]
}

fn shape(points: [Pos2; 4], stroke: Stroke) -> CubicBezierShape {
    CubicBezierShape::from_points_stroke(points, false, Color32::TRANSPARENT, PathStroke::from(stroke))
}

/// Build a cubic Bézier from an output port (`from`) to an input port
/// (`to`), both in screen space, with horizontal control tangents.
pub fn link_curve(from: Pos2, to: Pos2, stroke: Stroke) -> CubicBezierShape {
    shape(horizontal_ctrl(from, to), stroke)
}

/// Build a cubic Bézier with *vertical* control tangents — the horizontal
/// curve with x and y swapped — for connections between stacked blocks.
pub fn link_curve_vertical(from: Pos2, to: Pos2, stroke: Stroke) -> CubicBezierShape {
    let ctrl = horizontal_ctrl(swap_xy(from), swap_xy(to)).map(swap_xy);
    shape(ctrl, stroke)
}
