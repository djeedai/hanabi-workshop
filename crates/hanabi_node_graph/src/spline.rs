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

/// Parametric position of `p` projected onto the segment `a`→`b`, clamped to
/// `[0, 1]`. Used to drive an along-the-edge color gradient.
fn project_t(a: Pos2, b: Pos2, p: Pos2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_sq();
    if len2 <= f32::EPSILON {
        return 0.0;
    }
    ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
}

/// Gamma-space color lerp; good enough for an edge tint gradient.
fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let l = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round().clamp(0.0, 255.0) as u8;
    Color32::from_rgba_unmultiplied(
        l(a.r(), b.r()),
        l(a.g(), b.g()),
        l(a.b(), b.b()),
        l(a.a(), b.a()),
    )
}

/// Build a Bézier whose stroke fades from `c_from` at `from` to `c_to` at
/// `to`, projecting each tessellated point onto the endpoint axis.
fn grad_shape(
    points: [Pos2; 4],
    from: Pos2,
    to: Pos2,
    width: f32,
    c_from: Color32,
    c_to: Color32,
) -> CubicBezierShape {
    let stroke = PathStroke::new_uv(width, move |_bbox, p| lerp_color(c_from, c_to, project_t(from, to, p)));
    CubicBezierShape::from_points_stroke(points, false, Color32::TRANSPARENT, stroke)
}

/// Build a cubic Bézier from an output port (`from`) to an input port
/// (`to`), both in screen space, with horizontal control tangents.
pub fn link_curve(from: Pos2, to: Pos2, stroke: Stroke) -> CubicBezierShape {
    shape(horizontal_ctrl(from, to), stroke)
}

/// Horizontal link with a color gradient from `c_from` to `c_to` — used to
/// visualize an implicit type cast along the edge.
pub fn link_curve_grad(
    from: Pos2,
    to: Pos2,
    width: f32,
    c_from: Color32,
    c_to: Color32,
) -> CubicBezierShape {
    grad_shape(horizontal_ctrl(from, to), from, to, width, c_from, c_to)
}

/// Build a cubic Bézier with *vertical* control tangents — the horizontal
/// curve with x and y swapped — for connections between stacked blocks.
pub fn link_curve_vertical(from: Pos2, to: Pos2, stroke: Stroke) -> CubicBezierShape {
    let ctrl = horizontal_ctrl(swap_xy(from), swap_xy(to)).map(swap_xy);
    shape(ctrl, stroke)
}
