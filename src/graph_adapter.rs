//! Read-only bridge from a Hanabi [`EffectAsset`] to the standalone
//! [`node_graph`](crate::ui::widgets::node_graph) widget.
//!
//! This module is a *consumer* of the widget: it implements
//! [`GraphViewer`] over the canonical `EffectAsset`/`Module`, so the
//! widget can render the document's real expression DAG and modifier
//! lists. It lives outside `node_graph/` precisely because the widget
//! must stay free of any `bevy_hanabi` import (see the plan's
//! compatibility section).
//!
//! Built over [`Module`] (read via the `proxy.rs` reflection helpers),
//! **not** over `bevy_hanabi::Graph` — that API can't read its own nodes
//! back. Everything here is read-only; structural editing is a later
//! phase gated on upstream API support.
//!
//! ## Identity scheme
//!
//! - **Expr node id** = the expression's `ExprHandle` id (1..N).
//! - **Modifier node id** = `MOD_BASE + group * GROUP_STRIDE + index`,
//!   keeping it disjoint from the small expr ids.
//! - **Stack id** = `group + 1` (Init=1, Update=2, Render=3).

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use bevy_egui::egui::Color32;
use bevy_hanabi::{
    EffectAsset, Expr, ExprHandle, Module, ScalarType, ToWgslString, ValueType, VectorType,
};

use crate::document::ModifierGroup;
use crate::proxy;
use crate::ui::modifier_names::display_name_for_modifier;
use crate::ui::widgets::node_graph::{
    GraphView, GraphViewer, Link, NodeDesc, NodeId, PortAddr, PortDesc, PortId, StackDesc, StackId,
    StackLink, WorldPos,
};

/// First node id reserved for modifier nodes; expr nodes use ids below it.
const MOD_BASE: u32 = 0x4000_0000;
/// Id span allotted to each modifier group.
const GROUP_STRIDE: u32 = 0x0001_0000;

/// Horizontal spacing between auto-layout columns (world units).
const COL_W: f64 = 220.0;
/// Vertical spacing between auto-layout rows (world units).
const ROW_H: f64 = 90.0;

// Rough geometry constants mirroring the widget's layout, used only to
// estimate stack heights when seeding so taller stacks don't pile on top of
// shorter ones. Kept local so the adapter stays decoupled from widget
// internals; exact agreement isn't needed — this only seeds initial spacing.
const EST_NODE_HEADER: f64 = 26.0;
const EST_ROW_H: f64 = 22.0;
const EST_NODE_BODY_PAD: f64 = 14.0;
const EST_STACK_HEADER: f64 = 24.0;
const EST_STACK_PAD: f64 = 8.0;
const EST_MEMBER_GAP: f64 = 6.0;
/// Vertical gap left between consecutive seeded stacks (world units).
const STACK_GAP: f64 = 48.0;

/// The three modifier groups, in execution order.
const GROUPS: [&str; 3] = ["Init", "Update", "Render"];

/// A precomputed, read-only snapshot of an [`EffectAsset`] as graph
/// topology the widget can render. Rebuilt each frame from the asset.
pub struct GraphSnapshot {
    /// All node ids in render order (expr nodes first, then members).
    order: Vec<NodeId>,
    /// Per-node descriptions.
    descs: HashMap<NodeId, NodeDesc>,
    /// All links (operand edges + modifier-field edges).
    links: Vec<Link>,
    /// The three modifier stacks.
    stacks: Vec<StackDesc>,
    /// Seed positions for free expr nodes (depth-layered auto-layout).
    expr_seed: Vec<(NodeId, WorldPos)>,
    /// Seed positions for the modifier stacks.
    stack_seed: Vec<(StackId, WorldPos)>,
}

fn expr_node_id(handle_id: u32) -> Option<NodeId> {
    NodeId::new(handle_id)
}

fn modifier_node_id(group: usize, index: usize) -> Option<NodeId> {
    let id = MOD_BASE + group as u32 * GROUP_STRIDE + index as u32;
    NodeId::new(id)
}

fn stack_id(group: usize) -> StackId {
    StackId::new(group as u32 + 1).unwrap()
}

/// Inverse of [`stack_id`]: map a stack id back to its modifier group, or
/// `None` if it isn't one of the three modifier stacks.
pub fn group_of_stack(stack: StackId) -> Option<ModifierGroup> {
    match stack.get() {
        1 => Some(ModifierGroup::Init),
        2 => Some(ModifierGroup::Update),
        3 => Some(ModifierGroup::Render),
        _ => None,
    }
}

/// Accent color for an expression node, by variant family.
fn expr_accent(expr: &Expr) -> Color32 {
    match expr {
        Expr::Literal(_) => Color32::from_rgb(90, 130, 80),
        Expr::Property(_) => Color32::from_rgb(150, 120, 60),
        Expr::Attribute(_) | Expr::ParentAttribute(_) => Color32::from_rgb(70, 110, 160),
        Expr::BuiltIn(_) => Color32::from_rgb(60, 130, 140),
        Expr::Unary { .. } | Expr::Binary { .. } | Expr::Ternary { .. } => {
            Color32::from_rgb(150, 110, 60)
        }
        Expr::Cast(_) => Color32::from_rgb(120, 90, 150),
        Expr::TextureSample(_) => Color32::from_rgb(140, 80, 120),
    }
}

/// Header accent for a modifier group's member nodes.
fn group_accent(group: usize) -> Color32 {
    match group {
        0 => Color32::from_rgb(120, 80, 130),
        1 => Color32::from_rgb(80, 110, 130),
        _ => Color32::from_rgb(120, 100, 70),
    }
}

/// Frame accent for a modifier stack.
fn stack_accent(group: usize) -> Color32 {
    match group {
        0 => Color32::from_rgb(80, 60, 90),
        1 => Color32::from_rgb(55, 75, 90),
        _ => Color32::from_rgb(85, 70, 50),
    }
}

/// Pin color for a value type, so compatible ports share a hue and links
/// between matching types read as a single color (a cast shows as a
/// gradient). Vectors share their scalar's hue, brightened with width.
fn value_type_color(vt: ValueType) -> Color32 {
    // Base hues per scalar family.
    const FLOAT: Color32 = Color32::from_rgb(0x5A, 0xB0, 0xE6); // blue
    const INT: Color32 = Color32::from_rgb(0x8C, 0xCB, 0x5E); // green
    const UINT: Color32 = Color32::from_rgb(0xB9, 0x8C, 0xE6); // purple
    const BOOL: Color32 = Color32::from_rgb(0xE0, 0x6C, 0x6C); // red
    match vt {
        ValueType::Scalar(ScalarType::Float) => FLOAT,
        ValueType::Scalar(ScalarType::Int) => INT,
        ValueType::Scalar(ScalarType::Uint) => UINT,
        ValueType::Scalar(ScalarType::Bool) => BOOL,
        ValueType::Vector(v) => match v {
            VectorType::VEC2F | VectorType::VEC3F | VectorType::VEC4F => FLOAT,
            VectorType::VEC2I | VectorType::VEC3I | VectorType::VEC4I => INT,
            VectorType::VEC2U | VectorType::VEC3U | VectorType::VEC4U => UINT,
            _ => Color32::GRAY,
        },
        ValueType::Matrix(_) => Color32::from_rgb(0xE0, 0xB0, 0x6C), // amber
        _ => Color32::GRAY,
    }
}

/// Best-effort value type of an expression. Properties resolve through the
/// module (their type lives on the property's default value); everything
/// else defers to [`Expr::value_type`], which may be `None` when the type
/// can't be inferred without more context.
fn expr_value_type(module: &Module, expr: &Expr) -> Option<ValueType> {
    match expr {
        Expr::Property(pe) => proxy::property_handle_of(pe)
            .and_then(|ph| module.get_property(ph))
            .map(|p| p.default_value().value_type()),
        other => other.value_type(),
    }
}

/// Value type of the expression a handle points at, if resolvable.
fn handle_value_type(module: &Module, handle: ExprHandle) -> Option<ValueType> {
    module.get(handle).and_then(|e| expr_value_type(module, e))
}

/// Pin color for the expression a handle points at.
fn handle_color(module: &Module, handle: ExprHandle) -> Option<Color32> {
    handle_value_type(module, handle).map(value_type_color)
}

/// Short, human-readable title for an expression node.
fn expr_title(expr: &Expr, module: &Module) -> String {
    match expr {
        Expr::Literal(l) => l.to_wgsl_string(),
        Expr::Attribute(a) => a
            .to_wgsl_string()
            .trim_start_matches("particle.")
            .to_string(),
        Expr::ParentAttribute(a) => {
            format!("parent.{}", a.to_wgsl_string().trim_start_matches("particle."))
        }
        Expr::Property(p) => proxy::property_handle_of(p)
            .and_then(|h| module.get_property(h))
            .map(|pr| format!("${}", pr.name()))
            .unwrap_or_else(|| "$prop".to_string()),
        Expr::BuiltIn(b) => b.to_wgsl_string(),
        Expr::Unary { op, .. } => format!("{op:?}"),
        Expr::Binary { op, .. } => format!("{op:?}"),
        Expr::Ternary { op, .. } => format!("{op:?}"),
        Expr::Cast(_) => "Cast".to_string(),
        Expr::TextureSample(_) => "TextureSample".to_string(),
    }
}

/// Operand input ports for an expression, paired with the handle each
/// consumes. Labels are operator-aware where the arity is known.
fn operand_ports(expr: &Expr) -> Vec<(Cow<'static, str>, ExprHandle)> {
    let handles = proxy::operand_handles(expr);
    let names: &[&'static str] = match expr {
        Expr::Unary { .. } => &["in"],
        Expr::Binary { .. } => &["lhs", "rhs"],
        Expr::Ternary { .. } => &["a", "b", "c"],
        _ => &[],
    };
    handles
        .into_iter()
        .enumerate()
        .map(|(i, h)| {
            let label: Cow<'static, str> = names
                .get(i)
                .map(|s| Cow::Borrowed(*s))
                .unwrap_or_else(|| Cow::Owned(format!("in{i}")));
            (label, h)
        })
        .collect()
}

/// Max displayed length of an inlined literal value chip. Longer values
/// are truncated with an ellipsis; full editing arrives in Phase 4.
const LITERAL_CHIP_MAX: usize = 18;

/// Tidy a wgsl literal for compact inline display: drop a `vecN<f32>(…)`
/// type-constructor prefix (keeping the components) and trim redundant
/// trailing dots on floats (`1.` → `1`, `0.,` → `0,`). Then truncate.
fn short_literal(value: &str) -> String {
    let trimmed = match value.find('(') {
        Some(open) if value[..open].contains('<') || value[..open].starts_with("vec") => {
            &value[open..]
        }
        _ => value,
    };
    let cleaned = trimmed
        .replace(".,", ",")
        .replace(".)", ")")
        .trim_end_matches('.')
        .to_string();

    if cleaned.chars().count() > LITERAL_CHIP_MAX {
        let head: String = cleaned.chars().take(LITERAL_CHIP_MAX - 1).collect();
        format!("{head}…")
    } else {
        cleaned
    }
}

/// Build the input [`PortDesc`] for a consumed operand, either as an inline
/// literal value chip (hiding the literal node and emitting no link) or as
/// a normal connection target (emitting a link from the operand's output).
#[allow(clippy::too_many_arguments)]
fn classify_input(
    label: Cow<'static, str>,
    operand: ExprHandle,
    consumer: NodeId,
    port_index: u16,
    module: &Module,
    literal_value: &HashMap<u32, String>,
    referenced_lits: &mut HashSet<u32>,
    links: &mut Vec<Link>,
) -> PortDesc {
    let mut port = PortDesc::new(label);
    if let Some(c) = handle_color(module, operand) {
        port = port.with_color(c);
    }
    if let Some(hid) = proxy::expr_handle_id(operand) {
        if let Some(val) = literal_value.get(&hid) {
            referenced_lits.insert(hid);
            return port.with_value(short_literal(val));
        }
        if let Some(src) = expr_node_id(hid) {
            links.push(Link {
                from: PortAddr::new(src, PortId::output(0)),
                to: PortAddr::new(consumer, PortId::input(port_index)),
            });
        }
    }
    port
}

impl GraphSnapshot {
    /// Build a snapshot from the canonical asset.
    pub fn build(asset: &EffectAsset) -> Self {
        let module = asset.module();
        let mut order = Vec::new();
        let mut descs = HashMap::new();
        let mut links = Vec::new();

        // --- Expr nodes ---------------------------------------------------
        let exprs = proxy::expressions(module);

        // Literal expressions are inlined onto their consumers' input pins,
        // so precompute their value strings up front.
        let mut literal_value: HashMap<u32, String> = HashMap::new();
        for (handle, expr) in &exprs {
            if let Expr::Literal(l) = expr {
                if let Some(hid) = proxy::expr_handle_id(*handle) {
                    literal_value.insert(hid, l.to_wgsl_string());
                }
            }
        }
        // Literal nodes actually consumed by some input pin; hidden below.
        let mut referenced_lits: HashSet<u32> = HashSet::new();

        for (handle, expr) in &exprs {
            let Some(hid) = proxy::expr_handle_id(*handle) else {
                continue;
            };
            let Some(id) = expr_node_id(hid) else {
                continue;
            };
            let inputs = operand_ports(expr)
                .into_iter()
                .enumerate()
                .map(|(k, (label, operand))| {
                    classify_input(
                        label,
                        operand,
                        id,
                        k as u16,
                        module,
                        &literal_value,
                        &mut referenced_lits,
                        &mut links,
                    )
                })
                .collect();
            let mut out = PortDesc::new("out");
            if let Some(c) = expr_value_type(module, expr).map(value_type_color) {
                out = out.with_color(c);
            }
            let desc = NodeDesc::new(expr_title(expr, module))
                .with_inputs(inputs)
                .with_outputs(vec![out])
                .with_accent(expr_accent(expr));
            descs.insert(id, desc);
            order.push(id);
        }

        // --- Modifier stacks ---------------------------------------------
        let groups: [Vec<&dyn bevy_hanabi::Modifier>; 3] = [
            asset.init_modifiers().collect(),
            asset.update_modifiers().collect(),
            asset.render_modifiers().map(|m| m.as_modifier()).collect(),
        ];

        let mut stacks = Vec::new();
        for (gi, mods) in groups.iter().enumerate() {
            let mut members = Vec::new();
            for (idx, m) in mods.iter().enumerate() {
                let Some(id) = modifier_node_id(gi, idx) else {
                    continue;
                };
                let inputs = proxy::modifier_expr_fields(m.as_reflect())
                    .into_iter()
                    .enumerate()
                    .map(|(k, (name, fh))| {
                        classify_input(
                            Cow::Owned(name),
                            fh,
                            id,
                            k as u16,
                            module,
                            &literal_value,
                            &mut referenced_lits,
                            &mut links,
                        )
                    })
                    .chain(
                        // Read-only display rows for non-expr fields (enums,
                        // integral grid sizes, etc.) so the modifier's full
                        // configuration is visible in the graph.
                        proxy::modifier_display_fields(m.as_reflect())
                            .into_iter()
                            .map(|(name, value)| PortDesc::new(name).display_value(value)),
                    )
                    .collect();
                let desc = NodeDesc::new(display_name_for_modifier(*m).into_owned())
                    .with_inputs(inputs)
                    .with_accent(group_accent(gi));
                descs.insert(id, desc);
                order.push(id);
                members.push(id);
            }
            stacks.push(
                StackDesc::new(stack_id(gi), GROUPS[gi])
                    .with_members(members)
                    .with_accent(stack_accent(gi)),
            );
        }

        // Hide literal nodes that were inlined into a consumer's value chip.
        // Orphan literals (never referenced) stay visible so nothing is lost.
        order.retain(|id| !referenced_lits.contains(&id.get()));
        for hid in &referenced_lits {
            if let Some(id) = expr_node_id(*hid) {
                descs.remove(&id);
            }
        }

        let (expr_seed, stack_seed) =
            seed_layout(module, &exprs, &stacks, &descs, &referenced_lits);

        Self {
            order,
            descs,
            links,
            stacks,
            expr_seed,
            stack_seed,
        }
    }

    /// Apply seed positions for any node/stack the view hasn't placed yet,
    /// so a freshly opened graph lays itself out instead of piling at the
    /// origin. User drags persist (only unset positions are seeded).
    pub fn seed_positions(&self, view: &mut GraphView) {
        for (id, pos) in &self.expr_seed {
            view.ensure_position(*id, *pos);
        }
        for (id, pos) in &self.stack_seed {
            view.ensure_stack_position(*id, *pos);
        }
    }
}

impl GraphViewer for GraphSnapshot {
    fn node_ids(&self) -> Vec<NodeId> {
        self.order.clone()
    }

    fn node(&self, id: NodeId) -> NodeDesc {
        self.descs
            .get(&id)
            .cloned()
            .unwrap_or_else(|| NodeDesc::new("?"))
    }

    fn links(&self) -> Vec<Link> {
        self.links.clone()
    }

    fn stacks(&self) -> Vec<StackDesc> {
        self.stacks.clone()
    }

    fn stack_links(&self) -> Vec<StackLink> {
        // The fixed particle pipeline: init feeds update feeds render.
        vec![
            StackLink {
                from: stack_id(0),
                to: stack_id(1),
            },
            StackLink {
                from: stack_id(1),
                to: stack_id(2),
            },
        ]
    }
}

/// Depth of an expression = longest operand chain below it (leaves are 0).
/// Memoized; a `visited` guard keeps a malformed cyclic arena from looping.
fn expr_depth(
    handle: ExprHandle,
    module: &Module,
    memo: &mut HashMap<u32, u32>,
    visited: &mut Vec<u32>,
) -> u32 {
    let Some(hid) = proxy::expr_handle_id(handle) else {
        return 0;
    };
    if let Some(d) = memo.get(&hid) {
        return *d;
    }
    if visited.contains(&hid) {
        return 0;
    }
    visited.push(hid);
    let depth = module
        .get(handle)
        .map(|expr| {
            proxy::operand_handles(expr)
                .into_iter()
                .map(|op| expr_depth(op, module, memo, visited))
                .max()
                .map(|d| d + 1)
                .unwrap_or(0)
        })
        .unwrap_or(0);
    visited.pop();
    memo.insert(hid, depth);
    depth
}

/// Compute seed positions: expr nodes laid out left→right by dependency
/// depth, stacks parked in a right-hand column. Inlined literal nodes
/// (`hidden`) are skipped — they never render.
fn seed_layout(
    module: &Module,
    exprs: &[(ExprHandle, Expr)],
    stacks: &[StackDesc],
    descs: &HashMap<NodeId, NodeDesc>,
    hidden: &HashSet<u32>,
) -> (Vec<(NodeId, WorldPos)>, Vec<(StackId, WorldPos)>) {
    let mut memo = HashMap::new();
    let mut visited = Vec::new();

    // Bucket expr nodes by depth, preserving handle order within a depth.
    let mut by_depth: HashMap<u32, Vec<NodeId>> = HashMap::new();
    let mut max_depth = 0u32;
    for (handle, _) in exprs {
        let Some(hid) = proxy::expr_handle_id(*handle) else {
            continue;
        };
        if hidden.contains(&hid) {
            continue;
        }
        let Some(id) = expr_node_id(hid) else {
            continue;
        };
        let d = expr_depth(*handle, module, &mut memo, &mut visited);
        max_depth = max_depth.max(d);
        by_depth.entry(d).or_default().push(id);
    }

    let mut expr_seed = Vec::new();
    for (depth, ids) in &by_depth {
        for (row, id) in ids.iter().enumerate() {
            let pos = WorldPos::new(*depth as f64 * COL_W + 40.0, row as f64 * ROW_H + 60.0);
            expr_seed.push((*id, pos));
        }
    }

    // Stack the stacks vertically with a uniform gap, advancing the cursor by
    // each stack's *estimated* height so a tall stack doesn't overlap the one
    // below nor leave a huge gap (the old fixed 300-unit pitch did both).
    let stack_x = (max_depth as f64 + 1.0) * COL_W + 120.0;
    let mut stack_seed = Vec::new();
    let mut cursor_y = 60.0;
    for stack in stacks {
        stack_seed.push((stack.id, WorldPos::new(stack_x, cursor_y)));
        cursor_y += estimated_stack_height(stack, descs) + STACK_GAP;
    }

    (expr_seed, stack_seed)
}

/// Estimate a stack's rendered height (world units) from its members' port
/// counts, mirroring the widget's layout math closely enough to seed spacing.
fn estimated_stack_height(stack: &StackDesc, descs: &HashMap<NodeId, NodeDesc>) -> f64 {
    let mut h = EST_STACK_HEADER + EST_STACK_PAD * 2.0;
    for (i, member) in stack.members.iter().enumerate() {
        if i > 0 {
            h += EST_MEMBER_GAP;
        }
        let rows = descs
            .get(member)
            .map(|d| d.inputs.len().max(d.outputs.len()))
            .unwrap_or(1);
        h += EST_NODE_HEADER + EST_NODE_BODY_PAD + rows as f64 * EST_ROW_H;
    }
    h
}
