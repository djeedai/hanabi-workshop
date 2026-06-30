//! Painting: grid, node bodies, ports, edges and the link rubber-band.
//!
//! All geometry arrives in world space and is converted to screen here;
//! off-screen elements are culled against the canvas rect.

use std::{borrow::Cow, collections::HashMap};

use egui::{Align2, Color32, CornerRadius, FontId, Pos2, Rect, Stroke, Vec2};

use super::{
    layout::{
        MEMBER_GAP, NodeLayout, PORT_RADIUS, PORT_ROW_H, STACK_HEADER_H, STACK_PAD, StackLayout,
    },
    spline,
    state::{GraphView, ReorderDrag},
    transform::{Transform, WorldPos, WorldRect},
    viewer::{Link, NodeId, PortAddr, PortSide, StackId, StackLink},
};

/// Colors used by the node-graph renderer.
///
/// Derived from egui visuals so the widget blends with the host theme.
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
            grid_minor: v.extreme_bg_color.linear_multiply(2.2),
            grid_major: v.extreme_bg_color.linear_multiply(3.4),
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

/// Linear-ish lerp between two colors in gamma space.
///
/// Good enough for UI tinting. `t = 0` yields `a`, `t = 1` yields `b`.
fn blend(a: Color32, b: Color32, t: f32) -> Color32 {
    let lerp = |x: u8, y: u8| {
        (x as f32 + (y as f32 - x as f32) * t)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    Color32::from_rgb(lerp(a.r(), b.r()), lerp(a.g(), b.g()), lerp(a.b(), b.b()))
}

/// Pick near-black or near-white text for legibility on `bg`.
///
/// By perceived (Rec. 601) luminance. Keeps header titles readable on arbitrary
/// accent colors supplied by the viewer.
fn contrast_text(bg: Color32) -> Color32 {
    let l = 0.299 * bg.r() as f32 + 0.587 * bg.g() as f32 + 0.114 * bg.b() as f32;
    if l > 140.0 {
        Color32::from_gray(20)
    } else {
        Color32::from_gray(240)
    }
}

/// Corner radii for a header strip.
///
/// Rounded on top to follow the node body, square on the bottom so the
/// header/body boundary is a clean straight line.
fn header_corners(rounding: f32) -> CornerRadius {
    let r = rounding.round().clamp(0.0, 255.0) as u8;
    CornerRadius {
        nw: r,
        ne: r,
        sw: 0,
        se: 0,
    }
}

/// Draw `text` clipped to a header strip.
///
/// Never spills past the header width; skipped entirely once the header is too
/// narrow to be useful.
fn draw_header_title(
    painter: &egui::Painter,
    header: Rect,
    pad: f32,
    text: &str,
    size: f32,
    color: Color32,
) {
    // Leave a little right margin so glyphs don't kiss the header edge.
    let text_rect = Rect::from_min_max(header.min, Pos2::new(header.max.x - 4.0, header.max.y));
    if text_rect.width() < 12.0 || size < 2.0 {
        return;
    }
    painter.with_clip_rect(text_rect).text(
        Pos2::new(header.min.x + pad, header.center().y),
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
    let palette_minor = Palette::from_visuals(&painter.ctx().global_style().visuals).grid_minor;
    let palette_major = Palette::from_visuals(&painter.ctx().global_style().visuals).grid_major;

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
        let x = t.world_to_screen(WorldPos::new(i as f64 * spacing, 0.0)).x;
        let color = if is_major {
            palette_major
        } else {
            palette_minor
        };
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
        let y = t.world_to_screen(WorldPos::new(0.0, j as f64 * spacing)).y;
        let color = if is_major {
            palette_major
        } else {
            palette_minor
        };
        painter.line_segment(
            [Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)],
            Stroke::new(1.0, color),
        );
    }
}

/// Draw all edges between existing links.
///
/// Each edge is tinted by its endpoint pin colors. Selected links are
/// emphasized with a light halo and a thicker stroke while keeping their own
/// tint.
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
        let from_c = from_node.port_color(link.from.port).unwrap_or(palette.link);
        let to_c = to_node.port_color(link.to.port).unwrap_or(palette.link);
        let from_s = t.world_to_screen(from_w);
        let to_s = t.world_to_screen(to_w);

        // A selected edge keeps its own endpoint tint (so its data type stays
        // legible) and reads as *brighter*: a soft light halo behind it plus a
        // slightly thicker stroke. Recoloring it to the dark accent instead
        // would make selection look recessed.
        let is_selected = selected.contains(link);
        if is_selected {
            let halo = blend(palette.selected, Color32::WHITE, 0.55).gamma_multiply(0.5);
            painter.add(spline::link_curve(
                from_s,
                to_s,
                t.zoom as f32,
                Stroke::new(base_width + 5.0, halo),
            ));
        }
        let width = if is_selected {
            base_width + 1.0
        } else {
            base_width
        };
        // Tint the edge by its endpoint colors: a solid color when both pins
        // match, a gradient when they differ.
        let curve = if from_c == to_c {
            spline::link_curve(from_s, to_s, t.zoom as f32, Stroke::new(width, from_c))
        } else {
            spline::link_curve_grad(from_s, to_s, t.zoom as f32, width, from_c, to_c)
        };
        painter.add(curve);
    }
}

/// Draw the fixed vertical pipeline connections between stacks.
///
/// A pin on each connected stack edge plus a vertical spline between them.
/// Purely decorative — these are not hit-tested or selectable.
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

        let curve = spline::link_curve_vertical(
            from_s,
            to_s,
            t.zoom as f32,
            Stroke::new(width, palette.link),
        );
        painter.add(curve);

        for c in [from_s, to_s] {
            painter.circle_filled(c, pin_r, palette.port);
            painter.circle_stroke(c, pin_r, Stroke::new(1.0, palette.node_stroke));
        }
    }
}

/// Draw a translucent highlight disc at a hovered port.
///
/// Sized to the grab tolerance so the pickable area is visible. Drawn on top of
/// the pin.
pub fn draw_port_hover(painter: &egui::Painter, t: &Transform, center: WorldPos) {
    let r = super::layout::port_grab_radius_screen(t);
    let c = t.world_to_screen(center);
    painter.circle_filled(c, r, Color32::from_white_alpha(140));
}

/// Draw a small callout box pointing at a `pin` via a chevron.
///
/// The message is wrapped to the canvas width and the box is clamped to stay
/// fully on-screen — shifted horizontally and flipped above/below the pin as
/// needed — so a long callout near an edge stays readable. `accent` tints the
/// left bar and the leading `icon`; `border` strokes the box.
fn draw_callout(
    painter: &egui::Painter,
    pin: Pos2,
    text: &str,
    accent: Color32,
    border: Color32,
    icon: char,
) {
    let clip = painter.clip_rect();
    let font = FontId::proportional(13.0);
    let text_color = Color32::from_rgb(0xF5, 0xF5, 0xF5);
    let bg = Color32::from_rgb(0x1E, 0x1E, 0x1E);
    let stroke = Stroke::new(1.0, border);
    let radius = 4.0;

    // Icon between the accent bar and the text.
    let icon_galley = painter.layout_no_wrap(icon.to_string(), font.clone(), accent);
    let icon_gap = 6.0;
    let icon_w = icon_galley.size().x;
    let icon_h = icon_galley.size().y;

    let pad = Vec2::new(7.0, 4.0);
    let bar = 4.0; // width of the left accent bar
    let margin = 8.0; // keep the box this far from the canvas edges

    // Wrap the message so the box never grows wider than the canvas.
    let fixed_w = bar + pad.x * 2.0 + icon_w + icon_gap;
    let max_text_w = (clip.width() - margin * 2.0 - fixed_w).max(60.0);
    let galley = painter.layout(text.to_owned(), font, text_color, max_text_w);

    let content_h = galley.size().y.max(icon_h);
    let size = Vec2::new(fixed_w + galley.size().x, content_h + pad.y * 2.0);

    let ch = 5.0; // chevron height
    let cw = 6.0; // chevron half-width
    let gap = 6.0; // pin → chevron tip

    // Prefer the box above the pin (chevron pointing down); flip below when it
    // would clip the canvas top and there's room beneath.
    let above_top = pin.y - gap - ch - size.y;
    let room_below = pin.y + gap + ch + size.y <= clip.max.y - margin;
    let place_above = above_top >= clip.min.y + margin || !room_below;

    // Box left: start with the chevron a fixed inset from the left edge, then
    // clamp so the whole box fits horizontally.
    let min_x = (pin.x - 14.0)
        .min(clip.max.x - margin - size.x)
        .max(clip.min.x + margin);
    // Keep the chevron tip within the (possibly shifted) box span so it still
    // reads as belonging to the box.
    let tip_x = pin.x.clamp(min_x + cw + 2.0, min_x + size.x - cw - 2.0);

    let (min_y, mouth_y, tip_y, fill_y) = if place_above {
        let tip_y = pin.y - gap;
        let mouth_y = tip_y - ch; // box's bottom edge
        (mouth_y - size.y, mouth_y, tip_y, mouth_y - 1.5)
    } else {
        let tip_y = pin.y + gap;
        let mouth_y = tip_y + ch; // box's top edge
        (mouth_y, mouth_y, tip_y, mouth_y + 1.5)
    };

    let rect = Rect::from_min_size(Pos2::new(min_x, min_y), size);
    let tip = Pos2::new(tip_x, tip_y);
    let base_l = Pos2::new(tip_x - cw, mouth_y);
    let base_r = Pos2::new(tip_x + cw, mouth_y);

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
        CornerRadius {
            nw: r,
            ne: 0,
            sw: r,
            se: 0,
        },
        accent,
    );
    // Chevron fill, nudged just inside the box's edge so its opaque body
    // overwrites the straight border segment across the mouth — leaving the box
    // border and the chevron's two sides reading as one continuous outline.
    painter.add(egui::Shape::convex_polygon(
        vec![
            tip,
            Pos2::new(base_l.x, fill_y),
            Pos2::new(base_r.x, fill_y),
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

/// Error callout for why a dragged link can't connect to a port.
///
/// Dark-red border, bright-red accent bar.
pub fn draw_tooltip(painter: &egui::Painter, pin: Pos2, text: &str) {
    draw_callout(
        painter,
        pin,
        text,
        Color32::from_rgb(0xE5, 0x48, 0x48),
        Color32::from_rgb(0x7A, 0x1F, 0x1F),
        crate::icons::ICON_CIRCLE_EXCLAMATION,
    );
}

/// Warning callout (amber accent) shown when hovering a node's warning icon.
pub fn draw_warning(painter: &egui::Painter, pin: Pos2, text: &str) {
    draw_callout(
        painter,
        pin,
        text,
        Color32::from_rgb(0xFF, 0xB4, 0x32),
        Color32::from_rgb(0x7A, 0x5A, 0x1F),
        crate::icons::ICON_TRIANGLE_EXCLAMATION,
    );
}

/// Draw the in-progress link being dragged from a port to the cursor.
///
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
        spline::link_curve_grad(cursor, anchor, t.zoom as f32, width, t_col, a_col)
    } else {
        spline::link_curve_grad(anchor, cursor, t.zoom as f32, width, a_col, t_col)
    };
    painter.add(curve);
}

/// Draw stack container frames (header + body) behind their member nodes.
///
/// Stacks in `selected` (live selection plus any under an in-progress marquee)
/// get the selection outline; `hovered` gets the lighter hover outline.
pub fn draw_stacks(
    painter: &egui::Painter,
    t: &Transform,
    stacks: &[StackLayout],
    selected: &std::collections::HashSet<super::viewer::StackId>,
    hovered: Option<super::viewer::StackId>,
    hovered_add: Option<super::viewer::StackId>,
    palette: &Palette,
) {
    let canvas = painter.clip_rect();
    let title_size = (t.world_len_to_screen(12.0)).clamp(2.0, 24.0);
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

        let stroke = if selected.contains(&s.id) {
            Stroke::new(2.0, palette.selected)
        } else if hovered == Some(s.id) {
            Stroke::new(1.5, palette.stack_stroke.gamma_multiply(1.8))
        } else {
            Stroke::new(1.0, palette.stack_stroke)
        };
        painter.rect_stroke(screen, rounding, stroke, egui::StrokeKind::Inside);

        draw_header_title(
            painter,
            header,
            t.world_len_to_screen(6.0),
            &s.title,
            title_size,
            contrast_text(header_color),
        );

        // Bottom "Add" button. Only readable above a minimum zoom; below that
        // the glyph would be sub-pixel noise.
        let button = t.world_rect_to_screen(s.add_button);
        let btn_rounding = (rounding * 0.6).clamp(1.0, 6.0);
        let is_hovered_add = hovered_add == Some(s.id);
        let btn_fill = if is_hovered_add {
            palette.stack_header.gamma_multiply(1.4)
        } else {
            palette.stack_header
        };
        painter.rect_filled(button, btn_rounding, btn_fill);
        painter.rect_stroke(
            button,
            btn_rounding,
            Stroke::new(1.0, palette.stack_stroke),
            egui::StrokeKind::Inside,
        );
        let label_size = (button.height() * 0.62).clamp(0.0, 16.0);
        if label_size >= 7.0 {
            painter.text(
                button.center(),
                egui::Align2::CENTER_CENTER,
                "+ Add",
                egui::FontId::proportional(label_size),
                contrast_text(btn_fill),
            );
        }
    }
}

/// What `draw_nodes` observed under the pointer this frame.
///
/// For the caller to act on after painting (drawn-geometry-dependent hit
/// results).
#[derive(Default)]
pub struct NodePaint {
    /// The hovered warning icon's anchor + tooltip text, drawn last by caller.
    pub warning_tooltip: Option<(Pos2, Cow<'static, str>)>,
    /// Screen rect of every input value chip drawn this frame, so the caller
    /// can overlay a real editor on each.
    pub chips: Vec<super::response::ChipHit>,
}

/// Draw every node body and its ports.
///
/// Nodes in `selected` (the live selection plus any under an in-progress
/// marquee) get the selection outline; `hovered` gets the lighter hover
/// outline.
pub fn draw_nodes(
    painter: &egui::Painter,
    t: &Transform,
    layouts: &[NodeLayout],
    selected: &std::collections::HashSet<NodeId>,
    hovered: Option<NodeId>,
    hovered_port: Option<WorldPos>,
    hovered_close: Option<NodeId>,
    hover_pos: Option<Pos2>,
    palette: &Palette,
) -> NodePaint {
    let canvas = painter.clip_rect();
    let title_size = (t.world_len_to_screen(13.0)).clamp(2.0, 26.0);
    let label_size = (t.world_len_to_screen(11.0)).clamp(2.0, 22.0);
    let port_r = (t.world_len_to_screen(PORT_RADIUS)).clamp(2.0, 9.0);
    let rounding = (t.world_len_to_screen(5.0)).clamp(1.0, 8.0);
    // Keep port labels, value chips and their overlaid editors rendering down
    // to a low zoom; below this the node is too small for them to be useful.
    let show_labels = t.zoom >= 0.25;
    let mut result = NodePaint::default();

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

        // Header decorations: title (left), an optional warning badge right of
        // it, and an optional close button hugging the right edge. The title is
        // clipped to leave room for the close button so they never overlap.
        let title_color = contrast_text(header_color);
        let close_screen = node
            .close_button
            .map(|r| t.world_rect_to_screen(r))
            .filter(|r| r.width() >= 7.0 && canvas.intersects(*r));
        let title_limit = match close_screen {
            Some(r) => r.min.x - 4.0,
            None => header.max.x - 4.0,
        };
        let title_end = draw_node_title(
            painter,
            header,
            title_limit,
            t.world_len_to_screen(6.0),
            &node.title,
            title_size,
            title_color,
        );

        // Warning badge immediately right of the title.
        if let Some(text) = &node.warning {
            let icon_size = title_size;
            let gap = 5.0;
            let icon_x = (title_end + gap).min(title_limit - icon_size);
            if icon_size >= 7.0 && icon_x >= header.min.x {
                let amber = Color32::from_rgb(0xFF, 0xB4, 0x32);
                let g = painter.layout_no_wrap(
                    crate::icons::ICON_TRIANGLE_EXCLAMATION.to_string(),
                    FontId::proportional(icon_size),
                    amber,
                );
                let pos = Pos2::new(icon_x, header.center().y - g.size().y * 0.5);
                let icon_rect = Rect::from_min_size(pos, g.size());
                painter.galley(pos, g, amber);
                if hover_pos.is_some_and(|p| icon_rect.contains(p)) {
                    result.warning_tooltip = Some((
                        Pos2::new(icon_rect.center().x, icon_rect.min.y),
                        text.clone(),
                    ));
                }
            }
        }

        // Close button.
        if let Some(close_screen) = close_screen {
            let is_hovered_close = hovered_close == Some(node.id);
            if is_hovered_close {
                painter.rect_filled(
                    close_screen,
                    (rounding * 0.6).clamp(1.0, 4.0),
                    Color32::from_rgb(0xC0, 0x39, 0x2B),
                );
            }
            let glyph_color = if is_hovered_close {
                Color32::WHITE
            } else {
                title_color
            };
            let glyph_size = (close_screen.height() * 0.8).clamp(7.0, 18.0);
            painter.text(
                close_screen.center(),
                Align2::CENTER_CENTER,
                crate::icons::ICON_XMARK.to_string(),
                FontId::proportional(glyph_size),
                glyph_color,
            );
        }

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
                            Pos2::new(c.x - t.world_len_to_screen(PORT_RADIUS + 3.0), c.y),
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
                    // A collapsible row prefixes the label with a chevron.
                    let mut x = c.x + t.world_len_to_screen(PORT_RADIUS + 3.0);
                    let expanded = port.row_height > PORT_ROW_H + 0.5;
                    let mut chevron_rect = None;
                    if show_labels && port.collapsible {
                        let s = label_size;
                        let r = Rect::from_center_size(Pos2::new(x + s * 0.5, c.y), Vec2::splat(s));
                        draw_chevron(painter, r, expanded, palette.text);
                        chevron_rect = Some(r);
                        x = r.max.x + t.world_len_to_screen(3.0);
                    }
                    if show_labels && !port.label.is_empty() {
                        let g = painter.layout_no_wrap(
                            port.label.to_string(),
                            FontId::proportional(label_size),
                            palette.text,
                        );
                        let w = g.size().x;
                        painter.galley(Pos2::new(x, c.y - g.size().y * 0.5), g, palette.text);
                        x += w + t.world_len_to_screen(5.0);
                    }
                    if show_labels {
                        if let Some(val) = &port.value {
                            let pad = (t.world_len_to_screen(3.0)).clamp(1.5, 5.0);
                            // Node body inset on the right: chips and overlaid
                            // editors stay this far inside the border.
                            let node_clip = Rect::from_min_max(
                                screen.min,
                                Pos2::new(screen.max.x - pad * 2.0, screen.max.y),
                            );
                            let rr = (t.world_len_to_screen(3.0)).clamp(1.0, 5.0);
                            // An expanded collapsible row gets a full-width box
                            // spanning from below its label line to the row's
                            // bottom; a collapsed collapsible row gets a
                            // full-width single-line preview box after its label;
                            // a normal row keeps its chip on the label line.
                            let mut value_text = None;
                            let chip_rect = if expanded {
                                let row_h = t.world_len_to_screen(port.row_height) as f32;
                                let line_h = t.world_len_to_screen(PORT_ROW_H) as f32;
                                let top = c.y - line_h * 0.5 + line_h;
                                let left = screen.min.x + pad * 2.0;
                                Rect::from_min_max(
                                    Pos2::new(left, top),
                                    Pos2::new(node_clip.max.x, c.y - line_h * 0.5 + row_h),
                                )
                            } else if port.collapsible {
                                let chip_h = label_size + pad;
                                let chip_min = Pos2::new(x, c.y - chip_h * 0.5);
                                Rect::from_min_max(
                                    chip_min,
                                    Pos2::new(node_clip.max.x, chip_min.y + chip_h),
                                )
                            } else {
                                let g = painter.layout_no_wrap(
                                    val.to_string(),
                                    FontId::monospace(label_size),
                                    palette.text,
                                );
                                let chip_h = g.size().y + pad;
                                let chip_min = Pos2::new(x, c.y - chip_h * 0.5);
                                let full_w = g.size().x + pad * 2.0;
                                let chip_w = full_w.min((node_clip.max.x - chip_min.x).max(0.0));
                                let rect = Rect::from_min_size(chip_min, Vec2::new(chip_w, chip_h));
                                // Defer the value text until after the chip box is
                                // filled, so the fill doesn't paint over it.
                                value_text =
                                    Some((g, Pos2::new(chip_min.x + pad, chip_min.y + pad * 0.5)));
                                rect
                            };
                            let host_editor = expanded && !port.collapsible;
                            if !host_editor {
                                painter.rect_filled(chip_rect, rr, palette.value_bg);
                                painter.rect_stroke(
                                    chip_rect,
                                    rr,
                                    Stroke::new(1.0, palette.node_stroke),
                                    egui::StrokeKind::Inside,
                                );
                            }
                            if let Some((g, pos)) = value_text {
                                painter.with_clip_rect(chip_rect.intersect(canvas)).galley(
                                    pos,
                                    g,
                                    palette.text,
                                );
                            }
                            result.chips.push(super::response::ChipHit {
                                port: PortAddr::new(node.id, port.id),
                                rect: chip_rect,
                                font_size: label_size,
                                pad,
                                clip: node_clip,
                                chevron: chevron_rect,
                                expanded,
                            });
                        }
                    }
                }
            }
        }
    }

    result
}

/// Draw a collapse/expand chevron inside `rect` (▶ collapsed, ▼ expanded).
fn draw_chevron(painter: &egui::Painter, rect: Rect, expanded: bool, color: Color32) {
    let r = rect.shrink(rect.width() * 0.22);
    let pts = if expanded {
        vec![
            Pos2::new(r.min.x, r.min.y),
            Pos2::new(r.max.x, r.min.y),
            Pos2::new(r.center().x, r.max.y),
        ]
    } else {
        vec![
            Pos2::new(r.min.x, r.min.y),
            Pos2::new(r.max.x, r.center().y),
            Pos2::new(r.min.x, r.max.y),
        ]
    };
    painter.add(egui::Shape::convex_polygon(pts, color, Stroke::NONE));
}

/// Draw a node's header title clipped to `right_limit` (screen x).
///
/// Returns the screen x just past the painted glyphs so a badge can follow it,
/// or the text's left x when there's no room to draw.
fn draw_node_title(
    painter: &egui::Painter,
    header: Rect,
    right_limit: f32,
    pad: f32,
    text: &str,
    size: f32,
    color: Color32,
) -> f32 {
    let left = header.min.x + pad;
    let text_rect = Rect::from_min_max(header.min, Pos2::new(right_limit, header.max.y));
    if text_rect.width() < 12.0 || size < 2.0 {
        return left;
    }
    let galley = painter.layout_no_wrap(text.to_owned(), FontId::proportional(size), color);
    let w = galley.size().x;
    let h = galley.size().y;
    painter.with_clip_rect(text_rect).galley(
        Pos2::new(left, header.center().y - h * 0.5),
        galley,
        color,
    );
    (left + w).min(right_limit)
}
/// Draw the stack-member reorder overlay.
///
/// A drop indicator at the target slot and a translucent ghost of the dragged
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
        let title_size = (t.world_len_to_screen(13.0)).clamp(2.0, 26.0);
        if title_size >= 2.0 {
            painter.text(
                Pos2::new(
                    screen.min.x + t.world_len_to_screen(6.0),
                    screen.min.y + title_size,
                ),
                Align2::LEFT_CENTER,
                &node.title,
                FontId::proportional(title_size),
                palette.text,
            );
        }
    }
}
