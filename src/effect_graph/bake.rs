//! Baking an [`EffectGraph`] into a runtime [`bevy_hanabi::EffectAsset`].
//!
//! The edit model expresses wiring with explicit [`GraphLink`]s and per-input
//! inline defaults; the runtime model wires expressions through `ExprHandle`
//! arena indices inside a [`Module`]. Baking rebuilds that arena: it walks the
//! expression nodes in dependency order, materializing each as a `Module`
//! expression and recording the resulting handle, then resolves every operand
//! to either a linked source node's handle or its inline-default literal.
//!
//! Properties bake per [`PropertyDef::exposed`]: an exposed property becomes a
//! real `Module` property (settable per instance at runtime), while an
//! edit-only property is inlined to a literal constant at each reference, so it
//! has no runtime cost.
//!
//! This module covers expression and property baking. Modifier instantiation
//! and final [`EffectAsset`] assembly build on the `NodeId → ExprHandle` map it
//! produces.

use std::collections::HashMap;

use bevy_hanabi::graph::expr::PropertyHandle;
use bevy_hanabi::{ExprHandle, Module};

use super::model::{EffectGraph, ExprNode, NodeId, NodePayload, PortRef, PropertyDef, PropertyId};
use super::schema::expr_input_ports;

/// What a [`BakeError`] is attributed to, so the UI can surface it in context
/// (e.g. highlight the offending node or property, or show a graph-level banner).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BakeSubject {
    /// A specific graph node (e.g. an expression with a missing operand).
    Node(NodeId),
    /// A specific user property (e.g. an exposed-name conflict).
    Property(PropertyId),
    /// The graph as a whole, with no single element to blame.
    Graph,
}

/// A problem encountered while baking, attributed to the element to blame.
/// Baking collects every error it can rather than stopping at the first.
#[derive(Debug, Clone, PartialEq)]
pub struct BakeError {
    pub subject: BakeSubject,
    pub message: String,
}

impl BakeError {
    fn node(node: NodeId, message: impl Into<String>) -> Self {
        Self {
            subject: BakeSubject::Node(node),
            message: message.into(),
        }
    }

    fn property(id: PropertyId, message: impl Into<String>) -> Self {
        Self {
            subject: BakeSubject::Property(id),
            message: message.into(),
        }
    }

    fn graph(message: impl Into<String>) -> Self {
        Self {
            subject: BakeSubject::Graph,
            message: message.into(),
        }
    }
}

/// Resolved property bindings produced by [`bake_properties`]: the runtime
/// handle of each exposed property, plus every property's definition indexed by
/// stable id (used to resolve [`ExprNode::Property`] references).
struct PropertyBindings<'a> {
    handles: HashMap<PropertyId, PropertyHandle>,
    defs: HashMap<PropertyId, &'a PropertyDef>,
}

/// Register exposed properties into `module` and index every property by its
/// stable id for later reference resolution.
///
/// Properties are referenced by id, not name, so display names are free to
/// collide. The one name constraint is on **exposed** properties: each becomes a
/// runtime `Module` property keyed by name, so a name shared by two exposed
/// properties is an inconsistency that blocks baking. It is reported as a
/// [`BakeError`] (never a panic — `Module::add_property` would panic on a
/// duplicate name, so the second add is skipped) so the author can fix it.
fn bake_properties<'a>(
    graph: &'a EffectGraph,
    module: &mut Module,
    errors: &mut Vec<BakeError>,
) -> PropertyBindings<'a> {
    let mut handles = HashMap::new();
    let mut defs = HashMap::with_capacity(graph.properties.len());
    let mut exposed_names: HashMap<&str, PropertyId> = HashMap::new();
    for prop in &graph.properties {
        if defs.insert(prop.id, prop).is_some() {
            // Two properties sharing an id is a structural inconsistency (ids are
            // unique by construction); references to it would be ambiguous.
            errors.push(BakeError::property(
                prop.id,
                format!("duplicate property id {}", prop.id.get()),
            ));
            continue;
        }
        if prop.exposed {
            let name: &str = &prop.name;
            if exposed_names.contains_key(name) {
                errors.push(BakeError::property(
                    prop.id,
                    format!("two exposed properties share the name '{name}'; rename one to bake"),
                ));
                continue;
            }
            exposed_names.insert(name, prop.id);
            let handle = module.add_property(name, prop.default);
            handles.insert(prop.id, handle);
        }
    }
    PropertyBindings { handles, defs }
}

/// Expression-node baking context: the graph, the property bindings, the
/// `Module` under construction, and the running `NodeId → ExprHandle` cache.
struct ExprBaker<'a, 'm> {
    graph: &'a EffectGraph,
    props: &'a PropertyBindings<'a>,
    module: &'m mut Module,
    handles: HashMap<NodeId, ExprHandle>,
    /// Nodes on the current DFS stack, for cycle detection.
    visiting: Vec<NodeId>,
}

impl ExprBaker<'_, '_> {
    /// Resolve a node to its `ExprHandle`, baking it (and its operands) on
    /// first visit and caching the result. Returns `None` once an error has
    /// been recorded for this subtree.
    fn resolve(&mut self, node_id: NodeId, errors: &mut Vec<BakeError>) -> Option<ExprHandle> {
        if let Some(h) = self.handles.get(&node_id) {
            return Some(*h);
        }
        if self.visiting.contains(&node_id) {
            errors.push(BakeError::node(node_id, "expression cycle"));
            return None;
        }

        let node = self.graph.node(node_id).or_else(|| {
            errors.push(BakeError::node(
                node_id,
                format!("link references missing node {}", node_id.get()),
            ));
            None
        })?;
        let NodePayload::Expr(expr) = &node.payload else {
            errors.push(BakeError::node(
                node_id,
                "expected an expression node as a link source",
            ));
            return None;
        };

        self.visiting.push(node_id);
        let handle = self.bake_expr(node_id, expr, errors);
        self.visiting.pop();

        if let Some(h) = handle {
            self.handles.insert(node_id, h);
        }
        handle
    }

    /// Bake one expression node, resolving its operand ports first.
    fn bake_expr(
        &mut self,
        node_id: NodeId,
        expr: &ExprNode,
        errors: &mut Vec<BakeError>,
    ) -> Option<ExprHandle> {
        let handle = match expr {
            ExprNode::Literal(v) => self.module.lit(*v),
            ExprNode::Property(id) => self.bake_property_ref(node_id, *id, errors)?,
            ExprNode::Attribute(a) => self.module.attr(*a),
            ExprNode::ParentAttribute(a) => self.module.parent_attr(*a),
            ExprNode::BuiltIn(op) => self.module.builtin(*op),
            ExprNode::Unary(op) => {
                let inner = self.operand(node_id, "in", errors)?;
                self.module.unary(*op, inner)
            }
            ExprNode::Binary(op) => {
                let lhs = self.operand(node_id, "lhs", errors)?;
                let rhs = self.operand(node_id, "rhs", errors)?;
                self.module.binary(*op, lhs, rhs)
            }
            ExprNode::Ternary(op) => {
                let a = self.operand(node_id, "a", errors)?;
                let b = self.operand(node_id, "b", errors)?;
                let c = self.operand(node_id, "c", errors)?;
                self.module.ternary(*op, a, b, c)
            }
            ExprNode::Cast(ty) => {
                let inner = self.operand(node_id, "in", errors)?;
                self.module.cast(inner, *ty)
            }
        };
        Some(handle)
    }

    /// Bake a property reference (by stable id): the property's runtime handle
    /// if exposed, otherwise its default value inlined as a literal. A reference
    /// to a missing or duplicate-named exposed property is reported, not fatal.
    fn bake_property_ref(
        &mut self,
        node_id: NodeId,
        id: PropertyId,
        errors: &mut Vec<BakeError>,
    ) -> Option<ExprHandle> {
        let Some(def) = self.props.defs.get(&id) else {
            errors.push(BakeError::node(
                node_id,
                format!("reference to unknown property id {}", id.get()),
            ));
            return None;
        };
        if def.exposed {
            // An exposed property with no handle was dropped as a duplicate-name
            // conflict during registration; that error is already recorded.
            let Some(handle) = self.props.handles.get(&id) else {
                return None;
            };
            Some(self.module.prop(*handle))
        } else {
            Some(self.module.lit(def.default))
        }
    }

    /// Resolve the value feeding input port `port` of `node_id`: the source of a
    /// link into that port if one exists, else the port's inline default
    /// literal. Errors if neither is available.
    fn operand(
        &mut self,
        node_id: NodeId,
        port: &str,
        errors: &mut Vec<BakeError>,
    ) -> Option<ExprHandle> {
        if let Some(source) = self.linked_source(node_id, port) {
            return self.resolve(source, errors);
        }
        if let Some(default) = self.inline_default(node_id, port) {
            return Some(self.module.lit(default));
        }
        errors.push(BakeError::node(
            node_id,
            format!("input port '{port}' is neither linked nor given a default"),
        ));
        None
    }

    /// The source node of the (single) link targeting `node_id`'s `port`.
    fn linked_source(&self, node_id: NodeId, port: &str) -> Option<NodeId> {
        let target = PortRef {
            node: node_id,
            port: port.into(),
        };
        self.graph
            .links
            .iter()
            .find(|l| l.to == target)
            .map(|l| l.from.node)
    }

    /// The inline default literal for `node_id`'s input `port`, if declared.
    fn inline_default(&self, node_id: NodeId, port: &str) -> Option<bevy_hanabi::Value> {
        let node = self.graph.node(node_id)?;
        node.inputs
            .iter()
            .find(|s| &*s.name == port)
            .map(|s| s.default)
    }
}

/// Build a [`Module`] from `graph`'s expression nodes and properties, returning
/// the module and the `NodeId → ExprHandle` map for every expression node that
/// is reachable from a modifier or another expression.
///
/// Only expression nodes reachable as operands or modifier inputs are
/// materialized; a dangling expression node with no consumer contributes
/// nothing to the arena. Errors (cycles, unknown properties, missing inputs,
/// wrong node kinds) are collected rather than fatal, so the caller can surface
/// all of them at once.
pub fn bake_module(
    graph: &EffectGraph,
) -> Result<(Module, HashMap<NodeId, ExprHandle>), Vec<BakeError>> {
    let mut module = Module::default();
    let mut errors = Vec::new();

    let props = bake_properties(graph, &mut module, &mut errors);

    let mut baker = ExprBaker {
        graph,
        props: &props,
        module: &mut module,
        handles: HashMap::new(),
        visiting: Vec::new(),
    };

    // Resolve every expression node that participates in the graph. A node is a
    // participant if it is the source or target of a link, or an operand-bearing
    // expression; resolving each pulls in its operand subtree transitively.
    let participants = expr_participants(graph);
    for node_id in participants {
        baker.resolve(node_id, &mut errors);
    }

    let handles = std::mem::take(&mut baker.handles);
    drop(baker);

    if errors.is_empty() {
        Ok((module, handles))
    } else {
        Err(errors)
    }
}

/// Expression nodes that must be materialized: every node that appears as a
/// link endpoint, plus every operand-bearing expression node (so a fully
/// inline-defaulted operator with no incoming links is still built).
fn expr_participants(graph: &EffectGraph) -> Vec<NodeId> {
    let mut seen = Vec::new();
    let push = |id: NodeId, seen: &mut Vec<NodeId>| {
        if !seen.contains(&id) {
            seen.push(id);
        }
    };
    for link in &graph.links {
        push(link.from.node, &mut seen);
        push(link.to.node, &mut seen);
    }
    for node in &graph.nodes {
        if let NodePayload::Expr(expr) = &node.payload
            && !expr_input_ports(expr).is_empty()
        {
            push(node.id, &mut seen);
        }
    }
    seen
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effect_graph::model::{
        EffectHeader, GraphLink, GraphNode, InputSlot, PortRef,
    };
    use bevy_hanabi::graph::expr::BinaryOperator;
    use bevy_hanabi::{
        Attribute, Expr, SimulationCondition, SimulationSpace, SpawnerSettings, Value,
    };

    fn header() -> EffectHeader {
        EffectHeader {
            name: "t".into(),
            capacity: 32,
            spawner: SpawnerSettings::rate(1.0.into()),
            simulation_space: SimulationSpace::Global,
            simulation_condition: SimulationCondition::Always,
            z_layer_2d: 0.0,
        }
    }

    fn graph_with(nodes: Vec<GraphNode>, links: Vec<GraphLink>, props: Vec<PropertyDef>) -> EffectGraph {
        let max = nodes.iter().map(|n| n.id.get()).max().unwrap_or(0);
        EffectGraph {
            header: header(),
            properties: props,
            nodes,
            stacks: vec![],
            links,
            next_id: max + 1,
        }
    }

    fn expr_node(id: u32, expr: ExprNode, inputs: Vec<InputSlot>) -> GraphNode {
        GraphNode {
            id: NodeId::new(id).unwrap(),
            payload: NodePayload::Expr(expr),
            inputs,
        }
    }

    fn pid(n: u32) -> PropertyId {
        PropertyId::new(n).unwrap()
    }

    fn prop_def(id: u32, name: &str, default: Value, exposed: bool) -> PropertyDef {
        PropertyDef {
            id: pid(id),
            name: name.into(),
            default,
            exposed,
        }
    }

    #[test]
    fn bakes_binary_with_link_and_inline_default() {
        // n1 = attr(position); n2 = n1 + lit(2.0 via inline default on rhs)
        let n1 = expr_node(1, ExprNode::Attribute(Attribute::POSITION), vec![]);
        let n2 = expr_node(
            2,
            ExprNode::Binary(BinaryOperator::Add),
            vec![InputSlot {
                name: "rhs".into(),
                default: Value::from(2.0f32),
            }],
        );
        let link = GraphLink {
            from: PortRef {
                node: NodeId::new(1).unwrap(),
                port: "out".into(),
            },
            to: PortRef {
                node: NodeId::new(2).unwrap(),
                port: "lhs".into(),
            },
        };
        let graph = graph_with(vec![n1, n2], vec![link], vec![]);

        let (module, handles) = bake_module(&graph).expect("bake");
        assert_eq!(handles.len(), 2);
        let top = handles[&NodeId::new(2).unwrap()];
        assert!(matches!(module.get(top), Some(Expr::Binary { .. })));
    }

    #[test]
    fn exposed_property_becomes_module_property() {
        // A property reference consumed by a unary so it participates in baking.
        let prop = expr_node(1, ExprNode::Property(pid(10)), vec![]);
        let unary = expr_node(
            2,
            ExprNode::Unary(bevy_hanabi::graph::expr::UnaryOperator::Abs),
            vec![],
        );
        let link = GraphLink {
            from: PortRef {
                node: NodeId::new(1).unwrap(),
                port: "out".into(),
            },
            to: PortRef {
                node: NodeId::new(2).unwrap(),
                port: "in".into(),
            },
        };
        let graph = graph_with(
            vec![prop, unary],
            vec![link],
            vec![prop_def(10, "speed", Value::from(4.0f32), true)],
        );

        let (module, _) = bake_module(&graph).expect("bake");
        assert_eq!(module.properties().len(), 1);
        assert_eq!(module.properties()[0].name(), "speed");
    }

    #[test]
    fn edit_only_property_is_inlined() {
        let n1 = expr_node(1, ExprNode::Property(pid(10)), vec![]);
        let unary = expr_node(2, ExprNode::Unary(bevy_hanabi::graph::expr::UnaryOperator::Abs), vec![]);
        let link = GraphLink {
            from: PortRef { node: NodeId::new(1).unwrap(), port: "out".into() },
            to: PortRef { node: NodeId::new(2).unwrap(), port: "in".into() },
        };
        let graph = graph_with(
            vec![n1, unary],
            vec![link],
            vec![prop_def(10, "tweak", Value::from(7.0f32), false)],
        );

        let (module, handles) = bake_module(&graph).expect("bake");
        assert!(module.properties().is_empty());
        let lit = handles[&NodeId::new(1).unwrap()];
        assert!(matches!(module.get(lit), Some(Expr::Literal(_))));
    }

    #[test]
    fn detects_cycle() {
        // n1(unary) -> n2(unary) -> n1 : a cycle.
        let n1 = expr_node(1, ExprNode::Unary(bevy_hanabi::graph::expr::UnaryOperator::Abs), vec![]);
        let n2 = expr_node(2, ExprNode::Unary(bevy_hanabi::graph::expr::UnaryOperator::Abs), vec![]);
        let links = vec![
            GraphLink {
                from: PortRef { node: NodeId::new(1).unwrap(), port: "out".into() },
                to: PortRef { node: NodeId::new(2).unwrap(), port: "in".into() },
            },
            GraphLink {
                from: PortRef { node: NodeId::new(2).unwrap(), port: "out".into() },
                to: PortRef { node: NodeId::new(1).unwrap(), port: "in".into() },
            },
        ];
        let graph = graph_with(vec![n1, n2], links, vec![]);

        let errors = bake_module(&graph).unwrap_err();
        assert!(errors.iter().any(|e| e.message.contains("cycle")));
    }

    #[test]
    fn unknown_property_errors() {
        let n1 = expr_node(1, ExprNode::Property(pid(99)), vec![]);
        let unary = expr_node(2, ExprNode::Unary(bevy_hanabi::graph::expr::UnaryOperator::Abs), vec![]);
        let link = GraphLink {
            from: PortRef { node: NodeId::new(1).unwrap(), port: "out".into() },
            to: PortRef { node: NodeId::new(2).unwrap(), port: "in".into() },
        };
        let graph = graph_with(vec![n1, unary], vec![link], vec![]);

        let errors = bake_module(&graph).unwrap_err();
        assert!(errors.iter().any(|e| {
            e.subject == BakeSubject::Node(NodeId::new(1).unwrap())
                && e.message.contains("unknown property")
        }));
    }

    #[test]
    fn duplicate_exposed_property_name_errors() {
        // Exposed properties become runtime Module properties keyed by name; a
        // name collision is an inconsistency that blocks baking (but never panics).
        let graph = graph_with(
            vec![],
            vec![],
            vec![
                prop_def(10, "dup", Value::from(1.0f32), true),
                prop_def(11, "dup", Value::from(2.0f32), true),
            ],
        );

        let errors = bake_module(&graph).unwrap_err();
        // The error is attributed to the conflicting (second) property so the UI
        // can link straight to it.
        assert!(errors.iter().any(|e| {
            e.subject == BakeSubject::Property(pid(11)) && e.message.contains("share the name 'dup'")
        }));
    }

    #[test]
    fn duplicate_edit_only_property_name_is_tolerated() {
        // Non-exposed properties are baked to literals and referenced by id, so a
        // shared display name is harmless and must not fail the bake.
        let graph = graph_with(
            vec![],
            vec![],
            vec![
                prop_def(10, "tweak", Value::from(1.0f32), false),
                prop_def(11, "tweak", Value::from(2.0f32), false),
            ],
        );

        let (module, _) = bake_module(&graph).expect("edit-only duplicates are harmless");
        assert!(module.properties().is_empty());
    }

    #[test]
    fn distinct_ids_resolve_independently() {
        // Two edit-only properties share a name but have distinct ids; each
        // reference resolves to its own value via id, not the shared name.
        let r1 = expr_node(1, ExprNode::Property(pid(10)), vec![]);
        let r2 = expr_node(2, ExprNode::Property(pid(11)), vec![]);
        let add = expr_node(3, ExprNode::Binary(BinaryOperator::Add), vec![]);
        let links = vec![
            GraphLink {
                from: PortRef { node: NodeId::new(1).unwrap(), port: "out".into() },
                to: PortRef { node: NodeId::new(3).unwrap(), port: "lhs".into() },
            },
            GraphLink {
                from: PortRef { node: NodeId::new(2).unwrap(), port: "out".into() },
                to: PortRef { node: NodeId::new(3).unwrap(), port: "rhs".into() },
            },
        ];
        let graph = graph_with(
            vec![r1, r2, add],
            links,
            vec![
                prop_def(10, "same", Value::from(1.0f32), false),
                prop_def(11, "same", Value::from(2.0f32), false),
            ],
        );

        let (module, handles) = bake_module(&graph).expect("bake");
        // Both references baked to distinct literal expressions.
        assert!(matches!(
            module.get(handles[&NodeId::new(1).unwrap()]),
            Some(Expr::Literal(_))
        ));
        assert!(matches!(
            module.get(handles[&NodeId::new(2).unwrap()]),
            Some(Expr::Literal(_))
        ));
    }
}
