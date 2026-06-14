//! Editor-level validity rules for effect-graph edits.
//!
//! These encode `bevy_hanabi` runtime constraints that the [`EffectGraph`] model
//! itself can legally represent but that produce a broken effect, so the editor
//! rejects them at edit time rather than waiting for the bake. They live in the
//! UI layer because they exist to gate *interactions* (a dragged link, a
//! create-node menu entry), not to describe the graph data.
//!
//! ## Properties in the render context
//!
//! `bevy_hanabi` 0.18 binds module properties only in the init/update compute
//! shaders; the render shader has none (see `hanabi_gaps.md` §6.3). An *exposed*
//! property that reaches a render modifier therefore bakes to an `Expr::Property`
//! the render shader can't resolve, and the effect silently stops rendering.
//! (Edit-only properties are inlined to literals at bake, so they're render-safe.)
//! [`link_routes_property_to_render`] rejects a dragged link that would create
//! this, and [`node_reaches_render`] lets the create-node menu hide exposed
//! property producers when the dangling input pin feeds render.

use std::collections::HashSet;

use crate::document::ModifierGroup;
use crate::effect_graph::model::{EffectGraph, ExprNode, NodeId, NodePayload};

/// Whether routing the output of `from` into an input of `to` would carry an
/// *exposed* property into the render context — which hanabi can't bind, so the
/// editor refuses such a link the same way it refuses an incompatible port type.
///
/// Evaluated against the *current* graph (the proposed link is not yet present):
/// the link feeds `from → to`, so it changes neither `from`'s upstream cone nor
/// `to`'s downstream cone. It is `true` exactly when `from` already carries an
/// exposed-property value *and* `to` already feeds the render stage.
pub fn link_routes_property_to_render(graph: &EffectGraph, from: NodeId, to: NodeId) -> bool {
    carries_exposed_property(graph, from) && node_reaches_render(graph, to)
}

/// True if `node` is itself a render-stack modifier, or transitively feeds one
/// through its (existing) output links.
pub fn node_reaches_render(graph: &EffectGraph, node: NodeId) -> bool {
    let render_members: HashSet<NodeId> = graph
        .stack(ModifierGroup::Render)
        .map(|s| s.members.iter().copied().collect())
        .unwrap_or_default();
    let mut stack = vec![node];
    let mut seen = HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if render_members.contains(&n) {
            return true;
        }
        for link in &graph.links {
            if link.from.node == n {
                stack.push(link.to.node);
            }
        }
    }
    false
}

/// True if `node` is an exposed-property expression node, or transitively
/// depends on one through its (existing) input links. Such a value cannot
/// legally enter the render context.
fn carries_exposed_property(graph: &EffectGraph, node: NodeId) -> bool {
    let mut stack = vec![node];
    let mut seen = HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if is_exposed_property(graph, n) {
            return true;
        }
        for link in &graph.links {
            if link.to.node == n {
                stack.push(link.from.node);
            }
        }
    }
    false
}

/// True for an `ExprNode::Property` node that references an exposed property.
/// Edit-only property references are inlined to literals at bake time, so they
/// are render-safe and excluded.
fn is_exposed_property(graph: &EffectGraph, node: NodeId) -> bool {
    match graph.node(node).map(|n| &n.payload) {
        Some(NodePayload::Expr(ExprNode::Property(id))) => {
            graph.property(*id).is_some_and(|p| p.exposed)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effect_graph::model::{
        GraphLink, GraphNode, GraphStack, InputSlot, ModifierNodeData, PortRef, PropertyDef,
    };
    use bevy_hanabi::graph::expr::BinaryOperator;
    use bevy_hanabi::Value;

    fn modifier_node(graph: &mut EffectGraph, group: ModifierGroup) -> NodeId {
        let id = graph.alloc_node_id();
        graph.nodes.push(GraphNode {
            id,
            payload: NodePayload::Modifier(ModifierNodeData::Known {
                type_path: "test::Modifier".into(),
                config: Default::default(),
            }),
            inputs: vec![InputSlot {
                name: "in".into(),
                default: Value::from(0.0f32),
            }],
        });
        let stack = graph.alloc_stack_id();
        graph.stacks.push(GraphStack {
            id: stack,
            group,
            members: vec![id],
        });
        id
    }

    fn property_node(graph: &mut EffectGraph, exposed: bool) -> NodeId {
        let pid = graph.alloc_property_id();
        graph.properties.push(PropertyDef {
            id: pid,
            name: "p".into(),
            default: Value::from(1.0f32),
            exposed,
        });
        let id = graph.alloc_node_id();
        graph.nodes.push(GraphNode {
            id,
            payload: NodePayload::Expr(ExprNode::Property(pid)),
            inputs: vec![],
        });
        id
    }

    fn link(graph: &mut EffectGraph, from: NodeId, to: NodeId, to_port: &str) {
        graph.links.push(GraphLink {
            from: PortRef { node: from, port: "out".into() },
            to: PortRef { node: to, port: to_port.into() },
        });
    }

    #[test]
    fn exposed_property_directly_into_render_is_rejected() {
        let mut g = EffectGraph::empty();
        let prop = property_node(&mut g, true);
        let render = modifier_node(&mut g, ModifierGroup::Render);
        assert!(link_routes_property_to_render(&g, prop, render));
    }

    #[test]
    fn edit_only_property_into_render_is_allowed() {
        let mut g = EffectGraph::empty();
        let prop = property_node(&mut g, false);
        let render = modifier_node(&mut g, ModifierGroup::Render);
        // Edit-only properties inline to literals at bake, so they're render-safe.
        assert!(!link_routes_property_to_render(&g, prop, render));
    }

    #[test]
    fn exposed_property_into_init_is_allowed() {
        let mut g = EffectGraph::empty();
        let prop = property_node(&mut g, true);
        // A render stack exists but isn't on the path.
        modifier_node(&mut g, ModifierGroup::Render);
        let init = modifier_node(&mut g, ModifierGroup::Init);
        assert!(!link_routes_property_to_render(&g, prop, init));
    }

    #[test]
    fn exposed_property_reaches_render_transitively() {
        let mut g = EffectGraph::empty();
        let prop = property_node(&mut g, true);
        let render = modifier_node(&mut g, ModifierGroup::Render);

        // An intermediate expression node fed by the exposed property.
        let mid = g.alloc_node_id();
        g.nodes.push(GraphNode {
            id: mid,
            payload: NodePayload::Expr(ExprNode::Binary(BinaryOperator::Add)),
            inputs: vec![
                InputSlot { name: "lhs".into(), default: Value::from(0.0f32) },
                InputSlot { name: "rhs".into(), default: Value::from(0.0f32) },
            ],
        });
        link(&mut g, prop, mid, "lhs");

        // Proposed link: the intermediate node into the render modifier. The
        // property taint reaches render through `mid`.
        assert!(link_routes_property_to_render(&g, mid, render));

        // Equivalently, proposing the property into `mid` (which already feeds
        // render) is also rejected.
        link(&mut g, mid, render, "in");
        assert!(link_routes_property_to_render(&g, prop, mid));
    }
}
