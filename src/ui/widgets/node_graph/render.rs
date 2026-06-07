//! Painting: background grid, node bodies, ports, edges and the live
//! link rubber-band. All geometry arrives in world space and is converted
//! to screen here; off-screen elements are culled against the canvas rect.

use std::collections::HashMap;

use egui::{Align2, Color32, FontId, Pos2, Rect, Stroke};

use super::layout::{
    NodeLayout, StackLayout, MEMBER_GAP, PORT_RADIUS, STACK_HEADER_H, STACK_PAD,
};
use super::spline;
use super::state::{GraphView, ReorderDrag};
use super::transform::{Transform, WorldPos, WorldRect};
use super::viewer::{Link, NodeId, PortSide};

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
}

impl Palette {
    pub fn from_visuals(v: &egui::Visuals) -> Self {
        let accent = v.selection.bg_fill;
        Self {
            grid_minor: v.extreme_bg_color.linear_multiply(1.6),
            grid_major: v.extreme_bg_color.linear_multiply(2.4),
            node_bg: v.widgets.inactive.bg_fill,
            node_header: v.widgets.active.bg_fill,
            node_stroke: v.widgets.noninteractive.bg_stroke.color,
            selected: accent,
            text: v.text_color(),
            port: v.widgets.active.fg_stroke.color,
            link: v.widgets.active.fg_stroke.color,
            stack_bg: v.extreme_bg_color.linear_multiply(1.3),
            stack_header: v.widgets.open.bg_fill,
            stack_stroke: v.widgets.noninteractive.bg_stroke.color,
        }
    }
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
        let (color, width) = if is_selected {
            (palette.selected, base_width + 1.5)
        } else {
            (palette.link, base_width)
        };
        let curve = spline::link_curve(
            t.world_to_screen(from_w),
            t.world_to_screen(to_w),
            Stroke::new(width, color),
        );
        painter.add(curve);
    }
}

/// Draw the in-progress link being dragged from a port to the cursor.
pub fn draw_pending_link(
    painter: &egui::Painter,
    t: &Transform,
    from_world: WorldPos,
    cursor: Pos2,
    palette: &Palette,
) {
    let width = (t.world_len_to_screen(2.0)).clamp(1.0, 4.0);
    let curve = spline::link_curve(
        t.world_to_screen(from_world),
        cursor,
        Stroke::new(width, palette.link.gamma_multiply(0.8)),
    );
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
        painter.rect_filled(header, rounding, s.accent.unwrap_or(palette.stack_header));

        let stroke = if hovered == Some(s.id) {
            Stroke::new(1.5, palette.stack_stroke.gamma_multiply(1.8))
        } else {
            Stroke::new(1.0, palette.stack_stroke)
        };
        painter.rect_stroke(screen, rounding, stroke, egui::StrokeKind::Inside);

        if title_size >= 7.0 {
            painter.text(
                Pos2::new(screen.min.x + 6.0, header.center().y),
                Align2::LEFT_CENTER,
                &s.title,
                FontId::proportional(title_size),
                palette.text,
            );
        }
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
        painter.rect_filled(header, rounding, header_color);

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
        if title_size >= 7.0 {
            painter.text(
                Pos2::new(screen.min.x + 6.0, header.center().y),
                Align2::LEFT_CENTER,
                &node.title,
                FontId::proportional(title_size),
                palette.text,
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
            let color = port.color.unwrap_or(palette.port);
            painter.circle_filled(c, port_r, color);
            painter.circle_stroke(c, port_r, Stroke::new(1.0, palette.node_stroke));
            if show_labels && !port.label.is_empty() {
                let (anchor, x) = match side {
                    PortSide::Input => (Align2::LEFT_CENTER, c.x + port_r + 3.0),
                    PortSide::Output => (Align2::RIGHT_CENTER, c.x - port_r - 3.0),
                };
                painter.text(
                    Pos2::new(x, c.y),
                    anchor,
                    &port.label,
                    FontId::proportional(label_size),
                    palette.text,
                );
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
