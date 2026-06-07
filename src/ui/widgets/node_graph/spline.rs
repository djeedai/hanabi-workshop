//! Spline geometry for graph edges: horizontal-tangent cubic Béziers,
//! the conventional look for left-to-right node graphs.

use egui::epaint::{CubicBezierShape, PathStroke};
use egui::{Color32, Pos2, Stroke};

/// Build a cubic Bézier from an output port (`from`) to an input port
/// (`to`), both in screen space, with horizontal control tangents.
pub fn link_curve(from: Pos2, to: Pos2, stroke: Stroke) -> CubicBezierShape {
    let dx = (to.x - from.x).abs();
    // Tangent length grows with horizontal separation so close ports get
    // a gentle curve and distant ones a pronounced S.
    let handle = (dx * 0.5).clamp(24.0, 160.0);
    let c1 = Pos2::new(from.x + handle, from.y);
    let c2 = Pos2::new(to.x - handle, to.y);
    CubicBezierShape::from_points_stroke(
        [from, c1, c2, to],
        false,
        Color32::TRANSPARENT,
        PathStroke::from(stroke),
    )
}
