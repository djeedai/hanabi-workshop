//! Painting: background grid, node bodies, ports, edges and the live
//! link rubber-band. All geometry arrives in world space and is converted
//! to screen here; off-screen elements are culled against the canvas rect.

use std::collections::HashMap;

use egui::{Align2, Color32, CornerRadius, FontId, Pos2, Rect, Stroke, Vec2};

use super::layout::{
    NodeLayout, StackLayout, MEMBER_GAP, PORT_RADIUS, STACK_HEADER_H, STACK_PAD,
};
use super::spline;
use super::state::{GraphView, ReorderDrag};
use super::transform::{Transform, WorldPos, WorldRect};
use super::viewer::{Link, NodeId, PortSide, StackId, StackLink};

/// Colors used by the node-graph renderer, derived from egui visuals so
/// the widget blends with the host theme.
pub struct Palette {
    pub grid_minor: Color32,
    pub grid_major: Color32,
    pub node_bg: Color32,
    pub node_header: Color32,
    pub node_stroke: Color32,
    pub selected: Color32,
    pub text: Color32,
    pub port: Color32,
    pub link: Color32,
    pub stack_bg: Color32,
    pub stack_header: Color32,
    pub stack_stroke: Color32,
    /// Recessed background for inline value chips on input ports.
    pub value_bg: Color32,
}

impl Palette {
    pub fn from_visuals(v: &egui::Visuals) -> Self {
        let accent = v.selection.bg_fill;
        Self {
            grid_minor: v.extreme_bg_color.linear_multiply(1.6),
            grid_major: v.extreme_bg_color.linear_multiply(2.4),
            // Sit the node body between the (very dark) canvas and egui's
            // default widget fill, so nodes read as distinctly raised but
            // don't wash out against the dark canvas.
            node_bg: blend(v.extreme_bg_color, v.widgets.inactive.bg_fill, 0.45),
            node_header: v.widgets.active.bg_fill,
            node_stroke: v.widgets.noninteractive.bg_stroke.color,
            selected: accent,
            text: v.text_color(),
            port: v.widgets.active.fg_stroke.color,
            link: v.widgets.active.fg_stroke.color,
            stack_bg: v.extreme_bg_color.linear_multiply(1.3),
            stack_header: v.widgets.open.bg_fill,
            stack_stroke: v.widgets.noninteractive.bg_stroke.color,
            value_bg: v.extreme_bg_color,
        }
    }
}

/// Linear-ish lerp between two colors in gamma space (good enough for UI
/// tinting). `t = 0` yields `a`, `t = 1` yields `b`.
fn blend(a: Color32, b: Color32, t: f32) -> Color32 {
    let lerp = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round().clamp(0.0, 255.0) as u8;
    Color32::from_rgb(lerp(a.r(), b.r()), lerp(a.g(), b.g()), lerp(a.b(), b.b()))
}

/// Pick near-black or near-white text for legibility on `bg`, by perceived
/// (Rec. 601) luminance. Keeps header titles readable on arbitrary accent
/// colors supplied by the viewer.
fn contrast_text(bg: Color32) -> Color32 {
    let l = 0.299 * bg.r() as f32 + 0.587 * bg.g() as f32 + 0.114 * bg.b() as f32;
    if l > 140.0 {
        Color32::from_gray(20)
    } else {
        Color32::from_gray(240)
    }
}

/// Corner radii for a header strip: rounded on top to follow the node body,
/// square on the bottom so the header/body boundary is a clean straight line.
fn header_corners(rounding: f32) -> CornerRadius {
    let r = rounding.round().clamp(0.0, 255.0) as u8;
    CornerRadius {
        nw: r,
        ne: r,
        sw: 0,
        se: 0,
    }
}

/// Draw `text` clipped to a header strip so it never spills past the header
/// width, and skip it entirely once the header is too narrow to be useful.
fn draw_header_title(
    painter: &egui::Painter,
    header: Rect,
    text: &str,
    size: f32,
    color: Color32,
) {
    // Leave a little right margin so glyphs don't kiss the header edge.
    let text_rect = Rect::from_min_max(header.min, Pos2::new(header.max.x - 4.0, header.max.y));
    if text_rect.width() < 12.0 || size < 7.0 {
        return;
    }
    painter
        .with_clip_rect(text_rect)
        .text(
            Pos2::new(header.min.x + 6.0, header.center().y),
            Align2::LEFT_CENTER,
            text,
            FontId::proportional(size),
            color,
        );
}

/// Draw the background grid, culled and level-of-detail'd to the canvas.
pub fn draw_grid(painter: &egui::Painter, t: &Transform, rect: Rect, view: &GraphView) {
    if !view.grid.enabled {
        return;
    }
    let palette_minor = Palette::from_visuals(&painter.ctx().style().visuals).grid_minor;
    let palette_major = Palette::from_visuals(&painter.ctx().style().visuals).grid_major;

    let spacing = view.grid.spacing.max(f64::EPSILON);
    let major_every = view.grid.major_every.max(1) as f64;

    // Skip minor lines once they collapse below a few pixels; drop the
    // grid entirely if even majors would be too dense.
    let minor_px = t.world_len_to_screen(spacing);
    let major_px = t.world_len_to_screen(spacing * major_every);
    if major_px < 6.0 {
        return;
    }
    let draw_minor = minor_px >= 6.0;

    let top_left = t.screen_to_world(rect.min);
    let bottom_right = t.screen_to_world(rect.max);

    let start_i = (top_left.x / spacing).floor() as i64;
    let end_i = (bottom_right.x / spacing).ceil() as i64;
    for i in start_i..=end_i {
        let is_major = i.rem_euclid(major_every as i64) == 0;
        if !is_major && !draw_minor {
            continue;
        }
        let x = t
            .world_to_screen(WorldPos::new(i as f64 * spacing, 0.0))
            .x;
        let color = if is_major { palette_major } else { palette_minor };
        painter.line_segment(
            [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
            Stroke::new(1.0, color),
        );
    }

    let start_j = (top_left.y / spacing).floor() as i64;
    let end_j = (bottom_right.y / spacing).ceil() as i64;
    for j in start_j..=end_j {
        let is_major = j.rem_euclid(major_every as i64) == 0;
        if !is_major && !draw_minor {
            continue;
        }
        let y = t
            .world_to_screen(WorldPos::new(0.0, j as f64 * spacing))
            .y;
        let color = if is_major { palette_major } else { palette_minor };
        painter.line_segment(
            [Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)],
            Stroke::new(1.0, color),
        );
    }
}

/// Draw all edges between existing links. Selected links are drawn thicker
/// and in the selection color.
pub fn draw_links(
    painter: &egui::Painter,
    t: &Transform,
    layouts: &[NodeLayout],
    links: &[Link],
    selected: &std::collections::HashSet<Link>,
    palette: &Palette,
) {
    let by_id: HashMap<NodeId, &NodeLayout> = layouts.iter().map(|n| (n.id, n)).collect();
    let base_width = (t.world_len_to_screen(2.0)).clamp(1.0, 4.0);
    for link in links {
        let (Some(from_node), Some(to_node)) =
            (by_id.get(&link.from.node), by_id.get(&link.to.node))
        else {
            continue;
        };
        let (Some(from_w), Some(to_w)) = (
            from_node.port_center(link.from.port),
            to_node.port_center(link.to.port),
        ) else {
            continue;
        };
        let is_selected = selected.contains(link);
        let from_s = t.world_to_screen(from_w);
        let to_s = t.world_to_screen(to_w);
        if is_selected {
            painter.add(spline::link_curve(
                from_s,
                to_s,
                Stroke::new(base_width + 1.5, palette.selected),
            ));
            continue;
        }
        // Tint the edge by its endpoint colors: a solid color when both pins
        // match, a gradient when they differ. Falls back to the neutral link
        // color for ports with no color.
        let from_c = from_node.port_color(link.from.port).unwrap_or(palette.link);
        let to_c = to_node.port_color(link.to.port).unwrap_or(palette.link);
        let curve = if from_c == to_c {
            spline::link_curve(from_s, to_s, Stroke::new(base_width, from_c))
        } else {
            spline::link_curve_grad(from_s, to_s, base_width, from_c, to_c)
        };
        painter.add(curve);
    }
}

/// Draw the fixed vertical pipeline connections between stacks: a pin on
/// each connected stack edge plus a vertical spline between them. Purely
/// decorative — these are not hit-tested or selectable.
pub fn draw_stack_links(
    painter: &egui::Painter,
    t: &Transform,
    stacks: &[StackLayout],
    links: &[StackLink],
    palette: &Palette,
) {
    let by_id: HashMap<StackId, &StackLayout> = stacks.iter().map(|s| (s.id, s)).collect();
    let width = (t.world_len_to_screen(2.0)).clamp(1.0, 4.0);
    let pin_r = (t.world_len_to_screen(PORT_RADIUS)).clamp(2.0, 9.0);

    for link in links {
        let (Some(from), Some(to)) = (by_id.get(&link.from), by_id.get(&link.to)) else {
            continue;
        };
        let from_w = from.bottom_pin();
        let to_w = to.top_pin();
        let from_s = t.world_to_screen(from_w);
        let to_s = t.world_to_screen(to_w);

        let curve = spline::link_curve_vertical(from_s, to_s, Stroke::new(width, palette.link));
        painter.add(curve);

        for c in [from_s, to_s] {
            painter.circle_filled(c, pin_r, palette.port);
            painter.circle_stroke(c, pin_r, Stroke::new(1.0, palette.node_stroke));
        }
    }
}

/// Draw a translucent highlight disc at a hovered port, sized to the grab
/// tolerance so the pickable area is visible. Drawn on top of the pin.
pub fn draw_port_hover(painter: &egui::Painter, t: &Transform, center: WorldPos) {
    let r = super::layout::port_grab_radius_screen(t);
    let c = t.world_to_screen(center);
    painter.circle_filled(c, r, Color32::from_white_alpha(140));
}

/// Draw a small warning tooltip hovering above a port `pin` (screen space),
/// with a chevron on its bottom edge pointing down at the pin. The box is
/// offset left so the chevron sits a fixed inset from its left edge — i.e.
/// directly over the pin — and it holds still while the pointer keeps moving
/// during a drag. Styled as an error callout: dark-red border, a bright-red
/// accent bar down the left edge, and plain light text. Used for the reason a
/// dragged link can't connect to that port.
pub fn draw_tooltip(painter: &egui::Painter, pin: Pos2, text: &str) {
    let font = FontId::proportional(13.0);
    let text_color = Color32::from_rgb(0xF5, 0xF5, 0xF5);
    let galley = painter.layout_no_wrap(text.to_owned(), font.clone(), text_color);
    let bg = Color32::from_rgb(0x1E, 0x1E, 0x1E);
    let border = Color32::from_rgb(0x7A, 0x1F, 0x1F);
    let accent = Color32::from_rgb(0xE5, 0x48, 0x48);
    let stroke = Stroke::new(1.0, border);
    let radius = 4.0;

    // Error icon between the accent bar and the text.
    let icon = crate::IconsFontAwesome7::ICON_CIRCLE_EXCLAMATION.to_string();
    let icon_galley = painter.layout_no_wrap(icon, font, accent);
    let icon_gap = 6.0;
    let icon_w = icon_galley.size().x;
    let icon_h = icon_galley.size().y;

    let pad = Vec2::new(7.0, 4.0);
    let bar = 4.0; // width of the left accent bar
    let content_w = icon_w + icon_gap + galley.size().x;
    let content_h = galley.size().y.max(icon_h);
    let size = Vec2::new(bar + pad.x * 2.0 + content_w, content_h + pad.y * 2.0);

    // Chevron geometry: tip just above the pin, base on the box's bottom edge.
    let inset = 14.0; // chevron tip distance from the box's left edge
    let ch = 5.0; // chevron height
    let cw = 6.0; // chevron half-width
    let gap = 6.0; // pin → chevron tip
    let tip = Pos2::new(pin.x, pin.y - gap);
    let box_bottom = tip.y - ch;
    let min = Pos2::new(pin.x - inset, box_bottom - size.y);
    let rect = Rect::from_min_size(min, size);
    let base_l = Pos2::new(pin.x - cw, box_bottom);
    let base_r = Pos2::new(pin.x + cw, box_bottom);

    painter.rect_filled(rect, radius, bg);
    painter.rect_stroke(rect, radius, stroke, egui::StrokeKind::Inside);
    // Bright accent bar down the left edge, drawn *over* the border stroke so
    // its opaque fill covers the stroke's inner anti-aliased edge (otherwise a
    // faint dark seam shows between the dark border and the bright bar). Left
    // corners rounded to follow the box.
    let bar_rect = Rect::from_min_size(rect.min, Vec2::new(bar, rect.height()));
    let r = radius.round() as u8;
    painter.rect_filled(
        bar_rect,
        CornerRadius { nw: r, ne: 0, sw: r, se: 0 },
        accent,
    );
    // Chevron fill, raised slightly above the box's bottom edge so its opaque
    // body overwrites the straight bottom-border segment across the mouth —
    // leaving the box border and the chevron's two sides reading as one
    // continuous outline. The side strokes still start at the true base edge.
    painter.add(egui::Shape::convex_polygon(
        vec![
            tip,
            Pos2::new(base_l.x, box_bottom - 1.5),
            Pos2::new(base_r.x, box_bottom - 1.5),
        ],
        bg,
        Stroke::NONE,
    ));
    painter.line_segment([base_l, tip], stroke);
    painter.line_segment([tip, base_r], stroke);

    // Icon then text, vertically centered within the box.
    let cy = rect.center().y;
    let content_left = rect.min.x + bar + pad.x;
    painter.galley(
        Pos2::new(content_left, cy - icon_h * 0.5),
        icon_galley,
        accent,
    );
    painter.galley(
        Pos2::new(content_left + icon_w + icon_gap, cy - galley.size().y * 0.5),
        galley,
        text_color,
    );
}

/// Draw the in-progress link being dragged from a port to the cursor.
/// `anchor_is_input` flips the curve orientation so the tangents always run
/// output→input (the anchor's a destination when dragging out of an input).
/// `anchor_color`/`target_color` tint the two ends (anchor end vs. the
/// cursor/target end); differing colors preview the link as a gradient.
pub fn draw_pending_link(
    painter: &egui::Painter,
    t: &Transform,
    from_world: WorldPos,
    cursor: Pos2,
    anchor_is_input: bool,
    anchor_color: Color32,
    target_color: Color32,
) {
    let width = (t.world_len_to_screen(2.0)).clamp(1.0, 4.0);
    let anchor = t.world_to_screen(from_world);
    let (a_col, t_col) = (anchor_color, target_color);
    // Orient as output→input: when the anchor is an input, the cursor plays
    // the role of the upstream output, so the curve leaves the cursor and
    // enters the input from the left. The gradient runs from the output end
    // to the input end accordingly.
    let curve = if anchor_is_input {
        spline::link_curve_grad(cursor, anchor, width, t_col, a_col)
    } else {
        spline::link_curve_grad(anchor, cursor, width, a_col, t_col)
    };
    painter.add(curve);
}

/// Draw stack container frames (header + body) behind their member nodes.
pub fn draw_stacks(
    painter: &egui::Painter,
    t: &Transform,
    stacks: &[StackLayout],
    hovered: Option<super::viewer::StackId>,
    palette: &Palette,
) {
    let canvas = painter.clip_rect();
    let title_size = (t.world_len_to_screen(12.0)).clamp(7.0, 24.0);
    let rounding = (t.world_len_to_screen(6.0)).clamp(1.0, 10.0);

    for s in stacks {
        let screen = t.world_rect_to_screen(s.rect);
        if !canvas.intersects(screen) {
            continue;
        }

        painter.rect_filled(screen, rounding, palette.stack_bg);

        let header_h = t.world_len_to_screen(STACK_HEADER_H);
        let header = Rect::from_min_max(
            screen.min,
            Pos2::new(screen.max.x, (screen.min.y + header_h).min(screen.max.y)),
        );
        let header_color = s.accent.unwrap_or(palette.stack_header);
        painter.rect_filled(header, header_corners(rounding), header_color);

        let stroke = if hovered == Some(s.id) {
            Stroke::new(1.5, palette.stack_stroke.gamma_multiply(1.8))
        } else {
            Stroke::new(1.0, palette.stack_stroke)
        };
        painter.rect_stroke(screen, rounding, stroke, egui::StrokeKind::Inside);

        draw_header_title(
            painter,
            header,
            &s.title,
            title_size,
            contrast_text(header_color),
        );
    }
}

/// Draw every node body and its ports. Nodes in `selected` (the live
/// selection plus any under an in-progress marquee) get the selection
/// outline; `hovered` gets the lighter hover outline.
pub fn draw_nodes(
    painter: &egui::Painter,
    t: &Transform,
    layouts: &[NodeLayout],
    selected: &std::collections::HashSet<NodeId>,
    hovered: Option<NodeId>,
    hovered_port: Option<WorldPos>,
    palette: &Palette,
) {
    let canvas = painter.clip_rect();
    let title_size = (t.world_len_to_screen(13.0)).clamp(7.0, 26.0);
    let label_size = (t.world_len_to_screen(11.0)).clamp(6.0, 22.0);
    let port_r = (t.world_len_to_screen(PORT_RADIUS)).clamp(2.0, 9.0);
    let rounding = (t.world_len_to_screen(5.0)).clamp(1.0, 8.0);
    let show_labels = label_size >= 7.5;

    for node in layouts {
        let screen = t.world_rect_to_screen(node.rect);
        if !canvas.intersects(screen) {
            continue;
        }

        let is_selected = selected.contains(&node.id);
        let is_hovered = hovered == Some(node.id);

        // Body.
        painter.rect_filled(screen, rounding, palette.node_bg);

        // Header strip.
        let header_h = t.world_len_to_screen(super::layout::HEADER_H);
        let header = Rect::from_min_max(
            screen.min,
            Pos2::new(screen.max.x, (screen.min.y + header_h).min(screen.max.y)),
        );
        let header_color = node.accent.unwrap_or(palette.node_header);
        painter.rect_filled(header, header_corners(rounding), header_color);

        // Outline (selection color for selected / pending-marquee, lighter
        // for hover).
        let stroke = if is_selected {
            Stroke::new(2.0, palette.selected)
        } else if is_hovered {
            Stroke::new(1.5, palette.node_stroke.gamma_multiply(1.6))
        } else {
            Stroke::new(1.0, palette.node_stroke)
        };
        painter.rect_stroke(screen, rounding, stroke, egui::StrokeKind::Inside);

        // Title.
        draw_header_title(
            painter,
            header,
            &node.title,
            title_size,
            contrast_text(header_color),
        );

        // Ports.
        for (port, side) in node
            .inputs
            .iter()
            .map(|p| (p, PortSide::Input))
            .chain(node.outputs.iter().map(|p| (p, PortSide::Output)))
        {
            let c = t.world_to_screen(port.center);
            // Read-only display rows (non-connectable) draw no pin — just the
            // label and value chip below.
            if port.connectable {
                // Hover halo sits behind the pin marker (above the node body).
                if hovered_port == Some(port.center) {
                    draw_port_hover(painter, t, port.center);
                }
                let color = port.color.unwrap_or(palette.port);
                painter.circle_filled(c, port_r, color);
                painter.circle_stroke(c, port_r, Stroke::new(1.0, palette.node_stroke));
            }

            match side {
                PortSide::Output => {
                    if show_labels && !port.label.is_empty() {
                        painter.text(
                            Pos2::new(c.x - port_r - 3.0, c.y),
                            Align2::RIGHT_CENTER,
                            &port.label,
                            FontId::proportional(label_size),
                            palette.text,
                        );
                    }
                }
                PortSide::Input => {
                    // Label, then (if present) an inline value chip right after
                    // it — "name value" — so an inlined literal reads as a
                    // field on the pin without colliding with the right edge.
                    let mut x = c.x + port_r + 3.0;
                    if show_labels && !port.label.is_empty() {
                        let g = painter.layout_no_wrap(
                            port.label.to_string(),
                            FontId::proportional(label_size),
                            palette.text,
                        );
                        let w = g.size().x;
                        painter.galley(Pos2::new(x, c.y - g.size().y * 0.5), g, palette.text);
                        x += w + 5.0;
                    }
                    if show_labels {
                        if let Some(val) = &port.value {
                            let g = painter.layout_no_wrap(
                                val.to_string(),
                                FontId::monospace(label_size),
                                palette.text,
                            );
                            let pad = (t.world_len_to_screen(3.0)).clamp(1.5, 5.0);
                            let chip_w = g.size().x + pad * 2.0;
                            let chip_h = g.size().y + pad;
                            let chip_min = Pos2::new(x, c.y - chip_h * 0.5);
                            let chip_rect = Rect::from_min_size(chip_min, Vec2::new(chip_w, chip_h));
                            let rr = (t.world_len_to_screen(3.0)).clamp(1.0, 5.0);
                            painter.rect_filled(chip_rect, rr, palette.value_bg);
                            painter.rect_stroke(
                                chip_rect,
                                rr,
                                Stroke::new(1.0, palette.node_stroke),
                                egui::StrokeKind::Inside,
                            );
                            painter.galley(
                                Pos2::new(chip_min.x + pad, chip_min.y + pad * 0.5),
                                g,
                                palette.text,
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Draw the overlay for an in-progress stack-member reorder: a horizontal
/// drop indicator at the target slot and a translucent ghost of the dragged
/// member following the cursor.
pub fn draw_reorder_overlay(
    painter: &egui::Painter,
    t: &Transform,
    layouts: &[NodeLayout],
    stacks: &[StackLayout],
    rd: &ReorderDrag,
    cursor: Pos2,
    palette: &Palette,
) {
    let Some(stack) = stacks.iter().find(|s| s.id == rd.stack) else {
        return;
    };

    // Drop indicator: a line at the gap the member would land in. We map the
    // target index (computed over the *other* members) onto the current
    // member layout, which still includes the dragged node in its original
    // slot, so the line always falls between two real nodes — including when
    // hovering the dragged node's own slot.
    let members: Vec<&NodeLayout> = layouts
        .iter()
        .filter(|n| n.stack == Some(rd.stack))
        .collect();
    let n = members.len();
    let from = members.iter().position(|m| m.id == rd.node).unwrap_or(0);
    let ti = rd.target_index.min(n.saturating_sub(1));
    let gap = if ti <= from { ti } else { ti + 1 };

    let indicator_y = if members.is_empty() {
        stack.rect.min.y + STACK_HEADER_H + STACK_PAD
    } else if gap == 0 {
        members[0].rect.min.y - MEMBER_GAP * 0.5
    } else if gap >= n {
        members[n - 1].rect.max().y + MEMBER_GAP * 0.5
    } else {
        (members[gap - 1].rect.max().y + members[gap].rect.min.y) * 0.5
    };
    let y = t.world_to_screen(WorldPos::new(0.0, indicator_y)).y;
    let x0 = t.world_to_screen(stack.rect.min).x;
    let x1 = t.world_to_screen(stack.rect.max()).x;
    painter.line_segment(
        [Pos2::new(x0, y), Pos2::new(x1, y)],
        Stroke::new(2.0, palette.selected),
    );

    // Ghost of the dragged member, offset so its grab point tracks the
    // cursor.
    if let Some(node) = layouts.iter().find(|n| n.id == rd.node) {
        let min = t.screen_to_world(cursor) - rd.grab_offset;
        let ghost = WorldRect::new(min, node.rect.width, node.rect.height);
        let screen = t.world_rect_to_screen(ghost);
        let rounding = (t.world_len_to_screen(5.0)).clamp(1.0, 8.0);
        painter.rect_filled(screen, rounding, palette.node_bg.gamma_multiply(0.6));
        painter.rect_stroke(
            screen,
            rounding,
            Stroke::new(1.5, palette.selected),
            egui::StrokeKind::Inside,
        );
        let title_size = (t.world_len_to_screen(13.0)).clamp(7.0, 26.0);
        if title_size >= 7.0 {
            painter.text(
                Pos2::new(screen.min.x + 6.0, screen.min.y + title_size),
                Align2::LEFT_CENTER,
                &node.title,
                FontId::proportional(title_size),
                palette.text,
            );
        }
    }
}
