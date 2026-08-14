use std::collections::HashMap;

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;

use crate::types::{Edge, EdgeKind, Node, NodeKind};

/// A causal Directed Acyclic Graph (DAG) for the SCM.
///
/// Wraps `petgraph::DiGraph` with a name-based index for ergonomic access
/// and provides causal-inference-specific operations (parents, ancestors, etc.).
#[derive(Debug, Clone)]
pub struct CausalDag {
    pub(crate) graph: DiGraph<Node, Edge>,
    pub(crate) index: HashMap<String, NodeIndex>,
}

impl CausalDag {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            index: HashMap::new(),
        }
    }

    /// Add a node. Returns its `NodeIndex`. Panics if a node with the same name
    /// already exists — names must be unique.
    pub fn add_node(&mut self, name: &str, kind: NodeKind, asset: Option<&str>) -> NodeIndex {
        assert!(
            !self.index.contains_key(name),
            "duplicate node name: {name}"
        );
        let node = Node {
            name: name.to_string(),
            kind,
            asset: asset.map(|s| s.to_string()),
        };
        let idx = self.graph.add_node(node);
        self.index.insert(name.to_string(), idx);
        idx
    }

    /// Add a directed edge from `from` to `to`. Both must already exist in the graph.
    pub fn add_edge(&mut self, from: &str, to: &str, kind: EdgeKind, lag: i32) {
        let &from_idx = self.index.get(from).unwrap_or_else(|| {
            panic!("add_edge: source node '{from}' not found");
        });
        let &to_idx = self.index.get(to).unwrap_or_else(|| {
            panic!("add_edge: target node '{to}' not found");
        });
        self.graph.add_edge(from_idx, to_idx, Edge { kind, lag });
    }

    /// Look up a node index by name.
    pub fn node_index(&self, name: &str) -> Option<NodeIndex> {
        self.index.get(name).copied()
    }

    /// Get the `Node` data for a given index.
    pub fn node(&self, idx: NodeIndex) -> &Node {
        &self.graph[idx]
    }

    /// Get the `Node` data by name.
    pub fn node_by_name(&self, name: &str) -> Option<&Node> {
        self.index.get(name).map(|&idx| &self.graph[idx])
    }

    /// Number of nodes in the DAG.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges in the DAG.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// All node names.
    pub fn node_names(&self) -> Vec<&str> {
        self.graph
            .node_indices()
            .map(|idx| self.graph[idx].name.as_str())
            .collect()
    }

    /// Direct parents of `node` (nodes with an edge INTO this node).
    pub fn parents(&self, name: &str) -> Vec<&str> {
        let &idx = match self.index.get(name) {
            Some(i) => i,
            None => return vec![],
        };
        self.graph
            .neighbors_directed(idx, Direction::Incoming)
            .map(|i| self.graph[i].name.as_str())
            .collect()
    }

    /// Direct children of `node` (nodes this node has an edge TO).
    pub fn children(&self, name: &str) -> Vec<&str> {
        let &idx = match self.index.get(name) {
            Some(i) => i,
            None => return vec![],
        };
        self.graph
            .neighbors_directed(idx, Direction::Outgoing)
            .map(|i| self.graph[i].name.as_str())
            .collect()
    }

    /// All ancestors of `node` (transitive parents). Does not include the node itself.
    pub fn ancestors(&self, name: &str) -> Vec<&str> {
        let &start = match self.index.get(name) {
            Some(i) => i,
            None => return vec![],
        };
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![start];
        while let Some(current) = stack.pop() {
            for parent in self.graph.neighbors_directed(current, Direction::Incoming) {
                if visited.insert(parent) {
                    stack.push(parent);
                }
            }
        }
        visited
            .into_iter()
            .map(|i| self.graph[i].name.as_str())
            .collect()
    }

    /// All descendants of `node` (transitive children). Does not include the node itself.
    pub fn descendants(&self, name: &str) -> Vec<&str> {
        let &start = match self.index.get(name) {
            Some(i) => i,
            None => return vec![],
        };
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![start];
        while let Some(current) = stack.pop() {
            for child in self.graph.neighbors_directed(current, Direction::Outgoing) {
                if visited.insert(child) {
                    stack.push(child);
                }
            }
        }
        visited
            .into_iter()
            .map(|i| self.graph[i].name.as_str())
            .collect()
    }

    /// Check if the graph is a valid DAG (no cycles).
    pub fn is_dag(&self) -> bool {
        petgraph::algo::toposort(&self.graph, None).is_ok()
    }

    /// Get all edges as `(from_name, to_name, &Edge)` triples.
    pub fn edges(&self) -> Vec<(&str, &str, &Edge)> {
        self.graph
            .edge_references()
            .map(|e| {
                let from = self.graph[e.source()].name.as_str();
                let to = self.graph[e.target()].name.as_str();
                (from, to, e.weight())
            })
            .collect()
    }

    /// Get nodes filtered by kind.
    pub fn nodes_of_kind(&self, kind: NodeKind) -> Vec<&str> {
        self.graph
            .node_indices()
            .filter(|&idx| self.graph[idx].kind == kind)
            .map(|idx| self.graph[idx].name.as_str())
            .collect()
    }

    /// A copy of the graph with all edges OUT of `name` removed (Pearl's G_T̄,
    /// used by the graphical IV exclusion test). Node indices are preserved.
    pub fn without_outgoing_edges(&self, name: &str) -> CausalDag {
        let mut out = self.clone();
        if let Some(&idx) = out.index.get(name) {
            out.graph
                .retain_edges(|g, e| g.edge_endpoints(e).is_none_or(|(src, _)| src != idx));
        }
        out
    }

    /// Check if there is a directed path from `from` to `to`.
    pub fn has_directed_path(&self, from: &str, to: &str) -> bool {
        let &from_idx = match self.index.get(from) {
            Some(i) => i,
            None => return false,
        };
        let &to_idx = match self.index.get(to) {
            Some(i) => i,
            None => return false,
        };
        petgraph::algo::has_path_connecting(&self.graph, from_idx, to_idx, None)
    }
}

impl Default for CausalDag {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_chain() -> CausalDag {
        // A -> B -> C
        let mut dag = CausalDag::new();
        dag.add_node("A", NodeKind::GlobalFactor, None);
        dag.add_node("B", NodeKind::GlobalFactor, None);
        dag.add_node("C", NodeKind::AssetReturn, Some("eth"));
        dag.add_edge("A", "B", EdgeKind::Causal, 0);
        dag.add_edge("B", "C", EdgeKind::Causal, 0);
        dag
    }

    #[test]
    fn test_node_count() {
        let dag = simple_chain();
        assert_eq!(dag.node_count(), 3);
        assert_eq!(dag.edge_count(), 2);
    }

    #[test]
    fn test_parents_children() {
        let dag = simple_chain();
        assert_eq!(dag.parents("B"), vec!["A"]);
        assert_eq!(dag.children("B"), vec!["C"]);
        assert!(dag.parents("A").is_empty());
        assert!(dag.children("C").is_empty());
    }

    #[test]
    fn test_ancestors_descendants() {
        let dag = simple_chain();
        let anc: Vec<&str> = dag.ancestors("C");
        assert!(anc.contains(&"A"));
        assert!(anc.contains(&"B"));
        assert_eq!(anc.len(), 2);

        let desc: Vec<&str> = dag.descendants("A");
        assert!(desc.contains(&"B"));
        assert!(desc.contains(&"C"));
        assert_eq!(desc.len(), 2);
    }

    #[test]
    fn test_is_dag() {
        let dag = simple_chain();
        assert!(dag.is_dag());
    }

    #[test]
    fn test_has_directed_path() {
        let dag = simple_chain();
        assert!(dag.has_directed_path("A", "C"));
        assert!(!dag.has_directed_path("C", "A"));
    }

    #[test]
    fn test_nodes_of_kind() {
        let dag = simple_chain();
        let factors = dag.nodes_of_kind(NodeKind::GlobalFactor);
        assert_eq!(factors.len(), 2);
        let returns = dag.nodes_of_kind(NodeKind::AssetReturn);
        assert_eq!(returns.len(), 1);
    }

    #[test]
    #[should_panic(expected = "duplicate node name")]
    fn test_duplicate_node_panics() {
        let mut dag = CausalDag::new();
        dag.add_node("A", NodeKind::GlobalFactor, None);
        dag.add_node("A", NodeKind::GlobalFactor, None);
    }
}
