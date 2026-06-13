//! Read-only bridge from an [`EffectGraph`] to the standalone
//! [`node_graph`](crate::ui::widgets::node_graph) widget.
//!
//! Implements [`GraphViewer`] directly over the canonical [`EffectGraph`], so
//! the widget renders the document's real graph — its nodes, ordered modifier
//! stacks, links, and inline-default value chips — with no intermediate
//! projection. (This replaces the old `graph_adapter`, which reconstructed graph
//! topology from the *baked* `EffectAsset` because the asset is not a graph.)
//!
//! The widget stays free of any `bevy_hanabi` import; this module is the
//! consumer that bridges the two. Node and stack ids map 1:1 onto the widget's
//! id types (both are `NonZeroU32`), and inline defaults — already modeled as
//! unlinked [`InputSlot`](super::model::InputSlot)s — render as value chips
//! without any literal-hiding pass.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use bevy::reflect::TypeRegistry;
use bevy_egui::egui::Color32;
use bevy_hanabi::{ScalarType, ToWgslString, Value, ValueType, VectorType};

use crate::document::ModifierGroup;
use crate::ui::modifier_names::display_name_for_type;
use crate::ui::widgets::node_graph::{
    GraphView, GraphViewer, Link, LinkVerdict, NodeDesc, NodeId as WNodeId, PortAddr, PortDesc,
    PortId, StackDesc, StackId as WStackId, StackLink, WorldPos,
};

use super::model::{
    EditValue, EffectGraph, ExprNode, GradientVec3, GradientVec4, GraphLink, GraphNode,
    ModifierNodeData, NodeId, NodePayload, PortRef, SharedStr, TextureValue,
};
use super::schema::{OUTPUT_PORT, expr_input_ports, modifier_schema};

/// Horizontal spacing between auto-layout columns (world units).
const COL_W: f64 = 220.0;
/// Vertical spacing between auto-layout rows (world units).
const ROW_H: f64 = 90.0;
/// Vertical gap left between consecutive seeded stacks (world units).
const STACK_GAP: f64 = 48.0;
// Rough geometry constants mirroring the widget's layout, used only to estimate
// stack heights when seeding so taller stacks don't pile on shorter ones.
const EST_NODE_HEADER: f64 = 26.0;
const EST_ROW_H: f64 = 22.0;
const EST_NODE_BODY_PAD: f64 = 14.0;
const EST_STACK_HEADER: f64 = 24.0;
const EST_STACK_PAD: f64 = 8.0;
const EST_MEMBER_GAP: f64 = 6.0;

/// Max displayed length of an inlined value chip; longer values are truncated.
const CHIP_MAX: usize = 18;

/// A read-only view of an [`EffectGraph`] as graph topology the widget can
/// render. Borrows the graph and the type registry (needed for modifier schemas
/// and display names); builds no precomputed snapshot.
pub struct GraphReader<'a> {
    graph: &'a EffectGraph,
    registry: &'a TypeRegistry,
    /// node id → `(group, index)` for stack members; drives accents, execution
    /// order, and which nodes float vs. live in a stack.
    member_of: HashMap<NodeId, (ModifierGroup, usize)>,
}

impl<'a> GraphReader<'a> {
    pub fn new(graph: &'a EffectGraph, registry: &'a TypeRegistry) -> Self {
        let mut member_of = HashMap::new();
        for stack in &graph.stacks {
            for (idx, &member) in stack.members.iter().enumerate() {
                member_of.insert(member, (stack.group, idx));
            }
        }
        Self {
            graph,
            registry,
            member_of,
        }
    }

    /// Apply seed positions for any node/stack the view hasn't placed yet, so a
    /// freshly opened graph lays itself out instead of piling at the origin.
    /// User drags persist (only unset positions are seeded).
    pub fn seed_positions(&self, view: &mut GraphView) {
        let (expr_seed, stack_seed) = self.seed_layout();
        for (id, pos) in expr_seed {
            view.ensure_position(id, pos);
        }
        for (id, pos) in stack_seed {
            view.ensure_stack_position(id, pos);
        }
    }

    /// The connectable input port names of a node, in order — operand ports for
    /// an expression, expression-field ports for a modifier. These come first in
    /// the node's input list, so their indices double as the widget port index.
    fn connectable_inputs(&self, node: &GraphNode) -> Vec<Cow<'static, str>> {
        match &node.payload {
            NodePayload::Expr(e) => expr_input_ports(e).iter().map(|s| Cow::Borrowed(*s)).collect(),
            NodePayload::Modifier(ModifierNodeData::Known { type_path, .. }) => self
                .schema_ports(type_path)
                .into_iter()
                .map(Cow::Owned)
                .collect(),
            NodePayload::Modifier(ModifierNodeData::Unknown { .. }) => Vec::new(),
        }
    }

    /// Map a widget link (output port → input port) back to a model
    /// [`GraphLink`], or `None` if either endpoint no longer resolves. The
    /// inverse of the index↔name mapping this reader builds for the widget:
    /// outputs are a node's single `out` port; inputs are looked up by their
    /// position in [`connectable_inputs`](Self::connectable_inputs).
    pub fn resolve_link(&self, from: PortAddr, to: PortAddr) -> Option<GraphLink> {
        let from_node = NodeId::new(from.node.get())?;
        let to_node = NodeId::new(to.node.get())?;
        let target = self.graph.node(to_node)?;
        let to_port = self
            .connectable_inputs(target)
            .get(to.port.index as usize)?
            .as_ref()
            .to_owned();
        Some(GraphLink {
            from: PortRef {
                node: from_node,
                port: OUTPUT_PORT.into(),
            },
            to: PortRef {
                node: to_node,
                port: SharedStr::from(to_port),
            },
        })
    }

    /// Expression-port field names of a modifier type, in declaration order.
    fn schema_ports(&self, type_path: &str) -> Vec<String> {
        self.registry
            .get_with_type_path(type_path)
            .and_then(|reg| modifier_schema(reg.type_info()))
            .map(|s| s.ports().map(|f| f.name.to_string()).collect())
            .unwrap_or_default()
    }

    /// The source node feeding `node`'s input `port`, if a link targets it.
    fn linked_source(&self, node: NodeId, port: &str) -> Option<NodeId> {
        self.graph
            .links
            .iter()
            .find(|l| l.to.node == node && &*l.to.port == port)
            .map(|l| l.from.node)
    }

    /// The inline-default literal for `node`'s input `port`, if declared.
    fn inline_default(&self, node: NodeId, port: &str) -> Option<Value> {
        self.graph
            .node(node)?
            .inputs
            .iter()
            .find(|s| &*s.name == port)
            .map(|s| s.default)
    }

    /// Output value type of an expression node (`None` for modifier nodes or
    /// when the type can't be inferred). Operators infer from their first
    /// operand; a `visited` set guards against malformed cyclic graphs.
    fn output_type(&self, node: NodeId) -> Option<ValueType> {
        self.output_type_rec(node, &mut Vec::new())
    }

    fn output_type_rec(&self, node: NodeId, visited: &mut Vec<NodeId>) -> Option<ValueType> {
        if visited.contains(&node) {
            return None;
        }
        visited.push(node);
        let result = match &self.graph.node(node)?.payload {
            NodePayload::Expr(e) => match e {
                ExprNode::Literal(v) => Some(v.value_type()),
                ExprNode::Property(pid) => {
                    self.graph.property(*pid).map(|p| p.default.value_type())
                }
                ExprNode::Attribute(a) | ExprNode::ParentAttribute(a) => Some(a.value_type()),
                ExprNode::BuiltIn(op) => Some(op.value_type()),
                ExprNode::Cast(vt) => Some(*vt),
                ExprNode::Unary(_) | ExprNode::Binary(_) | ExprNode::Ternary(_) => {
                    // Infer from the first operand (link source, else default).
                    let first = expr_input_ports(e).first().copied()?;
                    self.operand_type_rec(node, first, visited)
                }
            },
            NodePayload::Modifier(_) => None,
        };
        visited.pop();
        result
    }

    /// Value type flowing into `node`'s input `port`: the linked source's output
    /// type, or the inline default's type.
    fn operand_type(&self, node: NodeId, port: &str) -> Option<ValueType> {
        self.operand_type_rec(node, port, &mut Vec::new())
    }

    fn operand_type_rec(&self, node: NodeId, port: &str, visited: &mut Vec<NodeId>) -> Option<ValueType> {
        if let Some(src) = self.linked_source(node, port) {
            self.output_type_rec(src, visited)
        } else {
            self.inline_default(node, port).map(|v| v.value_type())
        }
    }

    /// Build a node's input ports (connectable expr ports first, then read-only
    /// config display rows for a modifier).
    fn input_ports(&self, node: &GraphNode) -> Vec<PortDesc> {
        let mut ports = Vec::new();
        for name in self.connectable_inputs(node) {
            let mut port = PortDesc::new(name.to_string());
            if let Some(t) = self.operand_type(node.id, &name) {
                port = port.with_color(value_type_color(t));
            }
            if self.linked_source(node.id, &name).is_some() {
                // Linked: a connection target; the link is emitted by `links()`.
                ports.push(port);
            } else if let Some(def) = self.inline_default(node.id, &name) {
                ports.push(port.with_value(short_literal(&def.to_wgsl_string())));
            } else {
                // Optional, unconnected port with no default.
                ports.push(port);
            }
        }
        // Read-only display rows for a modifier's non-expr configuration.
        if let NodePayload::Modifier(ModifierNodeData::Known { type_path, config }) = &node.payload {
            for field in self.config_fields(type_path) {
                if let Some(value) = config.get(field.as_str()) {
                    ports.push(PortDesc::new(field).display_value(format_config(value)));
                }
            }
        }
        ports
    }

    /// Config field names of a modifier type, in declaration order.
    fn config_fields(&self, type_path: &str) -> Vec<String> {
        self.registry
            .get_with_type_path(type_path)
            .and_then(|reg| modifier_schema(reg.type_info()))
            .map(|s| s.config().map(|f| f.name.to_string()).collect())
            .unwrap_or_default()
    }

    /// Whether linking `from → to` would close a cycle (i.e. `from` already
    /// depends transitively on `to`).
    fn would_cycle(&self, from: NodeId, to: NodeId) -> bool {
        let mut stack = vec![from];
        let mut seen: HashSet<NodeId> = HashSet::new();
        while let Some(n) = stack.pop() {
            if n == to {
                return true;
            }
            if !seen.insert(n) {
                continue;
            }
            for l in &self.graph.links {
                if l.to.node == n {
                    stack.push(l.from.node);
                }
            }
        }
        false
    }

    /// Execution rank `(group_order, index)` of a stacked modifier member, or
    /// `None` for a free expression node. Lower ranks run earlier.
    fn exec_rank(&self, node: NodeId) -> Option<(u32, usize)> {
        self.member_of
            .get(&node)
            .map(|(group, idx)| (group_order(*group), *idx))
    }

    /// Longest chain of *linked* operands below a node (leaves are 0). Inline
    /// defaults are not nodes and don't add depth.
    fn node_depth(&self, node: NodeId, memo: &mut HashMap<NodeId, u32>, visited: &mut Vec<NodeId>) -> u32 {
        if let Some(d) = memo.get(&node) {
            return *d;
        }
        if visited.contains(&node) {
            return 0;
        }
        visited.push(node);
        let depth = self
            .graph
            .node(node)
            .map(|n| {
                self.connectable_inputs(n)
                    .iter()
                    .filter_map(|p| self.linked_source(node, p))
                    .map(|src| self.node_depth(src, memo, visited))
                    .max()
                    .map(|d| d + 1)
                    .unwrap_or(0)
            })
            .unwrap_or(0);
        visited.pop();
        memo.insert(node, depth);
        depth
    }

    /// Compute seed positions: free expr nodes laid left→right by dependency
    /// depth; stacks parked in a right-hand column, stacked vertically.
    fn seed_layout(&self) -> (Vec<(WNodeId, WorldPos)>, Vec<(WStackId, WorldPos)>) {
        let mut memo = HashMap::new();
        let mut by_depth: HashMap<u32, Vec<NodeId>> = HashMap::new();
        let mut max_depth = 0u32;
        for node in &self.graph.nodes {
            // Modifier members are laid out by their stack, not as free nodes.
            if self.member_of.contains_key(&node.id) {
                continue;
            }
            let d = self.node_depth(node.id, &mut memo, &mut Vec::new());
            max_depth = max_depth.max(d);
            by_depth.entry(d).or_default().push(node.id);
        }

        let mut expr_seed = Vec::new();
        for (depth, ids) in &by_depth {
            for (row, id) in ids.iter().enumerate() {
                let pos = WorldPos::new(*depth as f64 * COL_W + 40.0, row as f64 * ROW_H + 60.0);
                expr_seed.push((wnode(*id), pos));
            }
        }

        let stack_x = (max_depth as f64 + 1.0) * COL_W + 120.0;
        let mut stack_seed = Vec::new();
        let mut cursor_y = 60.0;
        for stack in &self.graph.stacks {
            stack_seed.push((wstack(stack.id.0), WorldPos::new(stack_x, cursor_y)));
            cursor_y += self.estimated_stack_height(stack) + STACK_GAP;
        }
        (expr_seed, stack_seed)
    }

    /// Estimate a stack's rendered height from its members' port counts.
    fn estimated_stack_height(&self, stack: &super::model::GraphStack) -> f64 {
        let mut h = EST_STACK_HEADER + EST_STACK_PAD * 2.0;
        for (i, member) in stack.members.iter().enumerate() {
            if i > 0 {
                h += EST_MEMBER_GAP;
            }
            let rows = self
                .graph
                .node(*member)
                .map(|n| self.input_ports(n).len().max(1))
                .unwrap_or(1);
            h += EST_NODE_HEADER + EST_NODE_BODY_PAD + rows as f64 * EST_ROW_H;
        }
        h
    }
}

impl GraphViewer for GraphReader<'_> {
    fn node_ids(&self) -> Vec<WNodeId> {
        self.graph.nodes.iter().map(|n| wnode(n.id)).collect()
    }

    fn node(&self, id: WNodeId) -> NodeDesc {
        let Some(model_id) = NodeId::new(id.get()) else {
            return NodeDesc::new("?");
        };
        let Some(node) = self.graph.node(model_id) else {
            return NodeDesc::new("?");
        };
        match &node.payload {
            NodePayload::Expr(e) => {
                let mut out = PortDesc::new("out");
                if let Some(t) = self.output_type(model_id) {
                    out = out.with_color(value_type_color(t));
                }
                NodeDesc::new(self.expr_title(e))
                    .with_inputs(self.input_ports(node))
                    .with_outputs(vec![out])
                    .with_accent(expr_accent(e))
            }
            NodePayload::Modifier(data) => {
                let (title, type_path) = match data {
                    ModifierNodeData::Known { type_path, .. } => {
                        (display_name_for_type(base_name(type_path)).into_owned(), type_path)
                    }
                    ModifierNodeData::Unknown { type_path, .. } => {
                        (format!("{} (?)", base_name(type_path)), type_path)
                    }
                };
                let _ = type_path;
                let accent = self
                    .member_of
                    .get(&model_id)
                    .map(|(g, _)| group_accent(group_order(*g)))
                    .unwrap_or(Color32::DARK_GRAY);
                NodeDesc::new(title)
                    .with_inputs(self.input_ports(node))
                    .with_accent(accent)
            }
        }
    }

    fn links(&self) -> Vec<Link> {
        let mut out = Vec::new();
        for link in &self.graph.links {
            let Some(target) = self.graph.node(link.to.node) else {
                continue;
            };
            let names = self.connectable_inputs(target);
            let Some(index) = names.iter().position(|n| n.as_ref() == &*link.to.port) else {
                continue;
            };
            out.push(Link {
                from: PortAddr::new(wnode(link.from.node), PortId::output(0)),
                to: PortAddr::new(wnode(link.to.node), PortId::input(index as u16)),
            });
        }
        out
    }

    fn stacks(&self) -> Vec<StackDesc> {
        self.graph
            .stacks
            .iter()
            .map(|stack| {
                let members = stack.members.iter().map(|m| wnode(*m)).collect();
                StackDesc::new(wstack(stack.id.0), stack.group.label())
                    .with_members(members)
                    .with_accent(stack_accent(group_order(stack.group)))
            })
            .collect()
    }

    fn stack_links(&self) -> Vec<StackLink> {
        let id_of = |group: ModifierGroup| {
            self.graph
                .stacks
                .iter()
                .find(|s| s.group == group)
                .map(|s| wstack(s.id.0))
        };
        let mut links = Vec::new();
        if let (Some(init), Some(update)) = (id_of(ModifierGroup::Init), id_of(ModifierGroup::Update)) {
            links.push(StackLink { from: init, to: update });
        }
        if let (Some(update), Some(render)) =
            (id_of(ModifierGroup::Update), id_of(ModifierGroup::Render))
        {
            links.push(StackLink { from: update, to: render });
        }
        links
    }

    fn validate_link(&self, from: PortAddr, to: PortAddr) -> LinkVerdict {
        if from.node == to.node {
            return Err("a node can't feed its own input".into());
        }
        let (Some(from_id), Some(to_id)) = (NodeId::new(from.node.get()), NodeId::new(to.node.get()))
        else {
            return Ok(());
        };
        if self.would_cycle(from_id, to_id) {
            return Err("would create a cycle".into());
        }
        // Stacked modifiers run in a fixed order; values only flow forward.
        if let (Some(a), Some(b)) = (self.exec_rank(from_id), self.exec_rank(to_id)) {
            if a > b {
                return Err("a later stage can't feed an earlier one".into());
            }
        }
        // Type compatibility, with a few implicit casts.
        let from_ty = self.output_type(from_id);
        let to_ty = self
            .graph
            .node(to_id)
            .and_then(|n| self.connectable_inputs(n).get(to.port.index as usize).cloned())
            .and_then(|name| self.operand_type(to_id, &name));
        match (from_ty, to_ty) {
            (Some(ft), Some(tt)) => cast_verdict(ft, tt),
            _ => Ok(()),
        }
    }
}

impl GraphReader<'_> {
    /// Short, human-readable title for an expression node.
    fn expr_title(&self, expr: &ExprNode) -> String {
        match expr {
            ExprNode::Literal(v) => v.to_wgsl_string(),
            ExprNode::Attribute(a) => a.name().to_string(),
            ExprNode::ParentAttribute(a) => {
                format!("parent.{}", a.name())
            }
            ExprNode::Property(pid) => self
                .graph
                .property(*pid)
                .map(|p| format!("${}", p.name))
                .unwrap_or_else(|| "$prop".to_string()),
            ExprNode::BuiltIn(op) => op.to_wgsl_string(),
            ExprNode::Unary(op) => format!("{op:?}"),
            ExprNode::Binary(op) => format!("{op:?}"),
            ExprNode::Ternary(op) => format!("{op:?}"),
            ExprNode::Cast(_) => "Cast".to_string(),
        }
    }
}

/// Map a model node id to the widget's node id (both one-based `NonZeroU32`).
fn wnode(id: NodeId) -> WNodeId {
    WNodeId::new(id.get()).expect("node ids are non-zero")
}

/// Map a model stack id to the widget's stack id.
fn wstack(id: std::num::NonZeroU32) -> WStackId {
    WStackId::new(id.get()).expect("stack ids are non-zero")
}

/// Resolve the [`ModifierGroup`] for a widget stack id by matching it against
/// the graph's stacks. Returns `None` if the id has no corresponding stack
/// (e.g. a stale widget id after a structural change).
pub fn group_of_widget_stack(graph: &EffectGraph, stack: WStackId) -> Option<ModifierGroup> {
    graph
        .stacks
        .iter()
        .find(|s| s.id.0.get() == stack.get())
        .map(|s| s.group)
}

/// Execution order of a modifier group: Init < Update < Render.
fn group_order(group: ModifierGroup) -> u32 {
    match group {
        ModifierGroup::Init => 0,
        ModifierGroup::Update => 1,
        ModifierGroup::Render => 2,
    }
}

/// The last path segment of a type path, ignoring generics.
fn base_name(path: &str) -> &str {
    let head = path.split('<').next().unwrap_or(path);
    head.rsplit("::").next().unwrap_or(head)
}

/// Accent color for an expression node, by variant family.
fn expr_accent(expr: &ExprNode) -> Color32 {
    match expr {
        ExprNode::Literal(_) => Color32::from_rgb(90, 130, 80),
        ExprNode::Property(_) => Color32::from_rgb(150, 120, 60),
        ExprNode::Attribute(_) | ExprNode::ParentAttribute(_) => Color32::from_rgb(70, 110, 160),
        ExprNode::BuiltIn(_) => Color32::from_rgb(60, 130, 140),
        ExprNode::Unary(_) | ExprNode::Binary(_) | ExprNode::Ternary(_) => {
            Color32::from_rgb(150, 110, 60)
        }
        ExprNode::Cast(_) => Color32::from_rgb(120, 90, 150),
    }
}

/// Header accent for a modifier group's member nodes.
fn group_accent(group: u32) -> Color32 {
    match group {
        0 => Color32::from_rgb(120, 80, 130),
        1 => Color32::from_rgb(80, 110, 130),
        _ => Color32::from_rgb(120, 100, 70),
    }
}

/// Frame accent for a modifier stack.
fn stack_accent(group: u32) -> Color32 {
    match group {
        0 => Color32::from_rgb(80, 60, 90),
        1 => Color32::from_rgb(55, 75, 90),
        _ => Color32::from_rgb(85, 70, 50),
    }
}

/// Pin color for a value type, so compatible ports share a hue.
fn value_type_color(vt: ValueType) -> Color32 {
    const FLOAT: Color32 = Color32::from_rgb(0x5A, 0xB0, 0xE6);
    const INT: Color32 = Color32::from_rgb(0x8C, 0xCB, 0x5E);
    const UINT: Color32 = Color32::from_rgb(0xB9, 0x8C, 0xE6);
    const BOOL: Color32 = Color32::from_rgb(0xE0, 0x6C, 0x6C);
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
        ValueType::Matrix(_) => Color32::from_rgb(0xE0, 0xB0, 0x6C),
        _ => Color32::GRAY,
    }
}

/// Whether an output of type `from` may feed an input of type `to`: identical
/// types connect directly, a scalar splats into a vector of the same scalar.
fn cast_verdict(from: ValueType, to: ValueType) -> LinkVerdict {
    if from == to {
        return Ok(());
    }
    if let (ValueType::Scalar(s), ValueType::Vector(v)) = (from, to) {
        if v.elem_type() == s {
            return Ok(());
        }
    }
    Err(format!("no implicit cast {} → {}", type_short(from), type_short(to)).into())
}

/// WGSL-flavoured short name for a value type, for rejection messages.
fn type_short(vt: ValueType) -> &'static str {
    match vt {
        ValueType::Scalar(ScalarType::Float) => "f32",
        ValueType::Scalar(ScalarType::Int) => "i32",
        ValueType::Scalar(ScalarType::Uint) => "u32",
        ValueType::Scalar(ScalarType::Bool) => "bool",
        ValueType::Vector(v) => match v {
            VectorType::VEC2F => "vec2<f32>",
            VectorType::VEC3F => "vec3<f32>",
            VectorType::VEC4F => "vec4<f32>",
            VectorType::VEC2I => "vec2<i32>",
            VectorType::VEC3I => "vec3<i32>",
            VectorType::VEC4I => "vec4<i32>",
            VectorType::VEC2U => "vec2<u32>",
            VectorType::VEC3U => "vec3<u32>",
            VectorType::VEC4U => "vec4<u32>",
            _ => "vecN",
        },
        ValueType::Matrix(_) => "matrix",
        _ => "?",
    }
}

/// Tidy a wgsl literal for compact inline display and truncate.
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
    if cleaned.chars().count() > CHIP_MAX {
        let head: String = cleaned.chars().take(CHIP_MAX - 1).collect();
        format!("{head}…")
    } else {
        cleaned
    }
}

/// Compact display string for a modifier's non-expr config value.
fn format_config(value: &EditValue) -> String {
    match value {
        EditValue::Bool(b) => b.to_string(),
        EditValue::U32(n) => n.to_string(),
        EditValue::Scalar(v) => short_literal(&v.to_wgsl_string()),
        EditValue::UVec2(u) => format!("({}, {})", u.x, u.y),
        EditValue::Color(c) => short_literal(&Value::from(*c).to_wgsl_string()),
        EditValue::Attribute(a) => a.name().to_string(),
        EditValue::CpuVec3(cv) => format_cpu_vec3(cv),
        EditValue::CpuVec4(cv) => format_cpu_vec4(cv),
        EditValue::Gradient3(g) => match g {
            GradientVec3::Analytical(_) => "gradient".to_string(),
            GradientVec3::Lut(t) => format_texture(t),
        },
        EditValue::Gradient4(g) => match g {
            GradientVec4::Analytical(_) => "gradient".to_string(),
            GradientVec4::Lut(t) => format_texture(t),
        },
        EditValue::Texture(t) => format_texture(t),
        EditValue::Enum { variant, .. } => variant.to_string(),
        EditValue::Flags { bits, .. } => format!("0x{bits:X}"),
        EditValue::Raw(_) => "…".to_string(),
    }
}

fn format_cpu_vec3(cv: &bevy_hanabi::CpuValue<bevy::math::Vec3>) -> String {
    match cv {
        bevy_hanabi::CpuValue::Single(v) => short_literal(&Value::from(*v).to_wgsl_string()),
        bevy_hanabi::CpuValue::Uniform((a, b)) => format!(
            "{} … {}",
            short_literal(&Value::from(*a).to_wgsl_string()),
            short_literal(&Value::from(*b).to_wgsl_string())
        ),
        _ => "?".to_string(),
    }
}

fn format_cpu_vec4(cv: &bevy_hanabi::CpuValue<bevy::math::Vec4>) -> String {
    match cv {
        bevy_hanabi::CpuValue::Single(v) => short_literal(&Value::from(*v).to_wgsl_string()),
        bevy_hanabi::CpuValue::Uniform((a, b)) => format!(
            "{} … {}",
            short_literal(&Value::from(*a).to_wgsl_string()),
            short_literal(&Value::from(*b).to_wgsl_string())
        ),
        _ => "?".to_string(),
    }
}

fn format_texture(t: &TextureValue) -> String {
    match t {
        TextureValue::Asset(path) => path.to_string(),
        TextureValue::Slot { name } => format!("[{name}]"),
    }
}
