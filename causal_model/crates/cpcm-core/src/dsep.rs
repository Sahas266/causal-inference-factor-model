use std::collections::{HashSet, VecDeque};

use petgraph::graph::NodeIndex;
use petgraph::Direction;

use crate::dag::CausalDag;

/// Direction from which a node is reached during the Bayes-Ball traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Arrival {
    /// Reached via a parent (travelling downward along an edge).
    FromParent,
    /// Reached via a child (travelling upward against an edge).
    FromChild,
}

/// Test whether `x` and `y` are **d-separated** given the conditioning set `z`.
///
/// Uses the Bayes-Ball algorithm (Shachter 1998). Returns `true` if every
/// path between `x` and `y` is blocked by `z`.
///
/// The three junction rules:
/// - **Chain** (A → B → C): blocked if B ∈ Z
/// - **Fork** (A ← B → C): blocked if B ∈ Z
/// - **Collider** (A → B ← C): blocked if B ∉ Z and no descendant of B ∈ Z
pub fn d_separated(dag: &CausalDag, x: &str, y: &str, z: &[&str]) -> bool {
    let x_idx = match dag.node_index(x) {
        Some(i) => i,
        None => return true, // non-existent nodes are trivially separated
    };
    let y_idx = match dag.node_index(y) {
        Some(i) => i,
        None => return true,
    };

    if x_idx == y_idx {
        return false; // a node is never d-separated from itself
    }

    let z_set: HashSet<NodeIndex> = z
        .iter()
        .filter_map(|name| dag.node_index(name))
        .collect();

    // Pre-compute: which nodes have a descendant in Z?
    // A node has a "descendant in Z" if it is in Z or any of its descendants is in Z.
    let descendant_in_z = compute_descendant_in_z(dag, &z_set);

    // Bayes-Ball: BFS from x, tracking (node, arrival_direction).
    // A node can be visited twice (once from parent, once from child) with different
    // propagation rules, so we track visited as (NodeIndex, Arrival).
    let mut visited: HashSet<(NodeIndex, Arrival)> = HashSet::new();
    let mut queue: VecDeque<(NodeIndex, Arrival)> = VecDeque::new();

    // Start from x, respecting conditioning rules for x itself.
    let x_is_conditioned = z_set.contains(&x_idx);

    if !x_is_conditioned {
        // X is NOT conditioned: can propagate to children (like a fork/chain source)
        for child in dag.graph.neighbors_directed(x_idx, Direction::Outgoing) {
            let state = (child, Arrival::FromParent);
            if visited.insert(state) {
                queue.push_back(state);
            }
        }
        // Can also propagate up to parents
        for parent in dag.graph.neighbors_directed(x_idx, Direction::Incoming) {
            let state = (parent, Arrival::FromChild);
            if visited.insert(state) {
                queue.push_back(state);
            }
        }
    } else {
        // X IS conditioned: can only propagate up to parents (collider activation)
        if descendant_in_z.contains(&x_idx) {
            for parent in dag.graph.neighbors_directed(x_idx, Direction::Incoming) {
                let state = (parent, Arrival::FromChild);
                if visited.insert(state) {
                    queue.push_back(state);
                }
            }
        }
    }

    while let Some((current, arrival)) = queue.pop_front() {
        if current == y_idx {
            return false; // y is reachable => not d-separated
        }

        let is_conditioned = z_set.contains(&current);

        match arrival {
            Arrival::FromParent => {
                // We arrived at `current` travelling downward (from a parent).
                // Chain / Fork outgoing: if current NOT in Z, pass through to children.
                if !is_conditioned {
                    for child in dag.graph.neighbors_directed(current, Direction::Outgoing)
                    {
                        let state = (child, Arrival::FromParent);
                        if visited.insert(state) {
                            queue.push_back(state);
                        }
                    }
                }
                // Collider bounce: if current is conditioned or has descendant in Z,
                // we can go UP to parents.
                if is_conditioned || descendant_in_z.contains(&current) {
                    for parent in
                        dag.graph.neighbors_directed(current, Direction::Incoming)
                    {
                        let state = (parent, Arrival::FromChild);
                        if visited.insert(state) {
                            queue.push_back(state);
                        }
                    }
                }
            }
            Arrival::FromChild => {
                // We arrived at `current` travelling upward (from a child).
                // Fork: if current NOT in Z, pass through to other children.
                if !is_conditioned {
                    for child in dag.graph.neighbors_directed(current, Direction::Outgoing)
                    {
                        let state = (child, Arrival::FromParent);
                        if visited.insert(state) {
                            queue.push_back(state);
                        }
                    }
                }
                // Chain going further up: if current NOT in Z, pass to parents.
                if !is_conditioned {
                    for parent in
                        dag.graph.neighbors_directed(current, Direction::Incoming)
                    {
                        let state = (parent, Arrival::FromChild);
                        if visited.insert(state) {
                            queue.push_back(state);
                        }
                    }
                }
            }
        }
    }

    true // y was never reached => d-separated
}

/// Compute the set of nodes that either are in Z or have at least one descendant in Z.
fn compute_descendant_in_z(dag: &CausalDag, z_set: &HashSet<NodeIndex>) -> HashSet<NodeIndex> {
    let mut result: HashSet<NodeIndex> = z_set.clone();

    // For each node in Z, walk up to all ancestors and mark them.
    for &z_node in z_set {
        let mut stack = vec![z_node];
        while let Some(current) = stack.pop() {
            for parent in dag.graph.neighbors_directed(current, Direction::Incoming) {
                if result.insert(parent) {
                    stack.push(parent);
                }
            }
        }
    }

    result
}

/// Find all nodes reachable from `x` that are NOT d-separated from `x` given `z`.
/// Useful for understanding which variables `x` can still influence.
pub fn d_connected_set(dag: &CausalDag, x: &str, z: &[&str]) -> Vec<String> {
    let names = dag.node_names();
    names
        .into_iter()
        .filter(|&name| name != x && !d_separated(dag, x, name, z))
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EdgeKind, NodeKind};

    /// Chain: A → B → C
    /// B blocks the path when conditioned on.
    #[test]
    fn test_chain_unconditioned() {
        let mut dag = CausalDag::new();
        dag.add_node("A", NodeKind::GlobalFactor, None);
        dag.add_node("B", NodeKind::GlobalFactor, None);
        dag.add_node("C", NodeKind::AssetReturn, None);
        dag.add_edge("A", "B", EdgeKind::Causal, 0);
        dag.add_edge("B", "C", EdgeKind::Causal, 0);

        // Unconditioned: A and C are d-connected (path A→B→C is open)
        assert!(!d_separated(&dag, "A", "C", &[]));
    }

    #[test]
    fn test_chain_conditioned_on_mediator() {
        let mut dag = CausalDag::new();
        dag.add_node("A", NodeKind::GlobalFactor, None);
        dag.add_node("B", NodeKind::GlobalFactor, None);
        dag.add_node("C", NodeKind::AssetReturn, None);
        dag.add_edge("A", "B", EdgeKind::Causal, 0);
        dag.add_edge("B", "C", EdgeKind::Causal, 0);

        // Conditioned on B: path is blocked
        assert!(d_separated(&dag, "A", "C", &["B"]));
    }

    /// Fork: A ← B → C
    /// B is a common cause. Conditioning on B blocks the path.
    #[test]
    fn test_fork_unconditioned() {
        let mut dag = CausalDag::new();
        dag.add_node("A", NodeKind::AssetReturn, None);
        dag.add_node("B", NodeKind::GlobalFactor, None);
        dag.add_node("C", NodeKind::AssetReturn, None);
        dag.add_edge("B", "A", EdgeKind::Causal, 0);
        dag.add_edge("B", "C", EdgeKind::Causal, 0);

        // Unconditioned: A and C are d-connected via B
        assert!(!d_separated(&dag, "A", "C", &[]));
    }

    #[test]
    fn test_fork_conditioned_on_common_cause() {
        let mut dag = CausalDag::new();
        dag.add_node("A", NodeKind::AssetReturn, None);
        dag.add_node("B", NodeKind::GlobalFactor, None);
        dag.add_node("C", NodeKind::AssetReturn, None);
        dag.add_edge("B", "A", EdgeKind::Causal, 0);
        dag.add_edge("B", "C", EdgeKind::Causal, 0);

        // Conditioned on B: path is blocked
        assert!(d_separated(&dag, "A", "C", &["B"]));
    }

    /// Collider: A → B ← C
    /// Unconditioned: path is blocked (collider). Conditioning on B opens it.
    #[test]
    fn test_collider_unconditioned() {
        let mut dag = CausalDag::new();
        dag.add_node("A", NodeKind::GlobalFactor, None);
        dag.add_node("B", NodeKind::AssetReturn, None);
        dag.add_node("C", NodeKind::GlobalFactor, None);
        dag.add_edge("A", "B", EdgeKind::Causal, 0);
        dag.add_edge("C", "B", EdgeKind::Causal, 0);

        // Unconditioned: A and C are d-separated (collider blocks)
        assert!(d_separated(&dag, "A", "C", &[]));
    }

    #[test]
    fn test_collider_conditioned_on_collider() {
        let mut dag = CausalDag::new();
        dag.add_node("A", NodeKind::GlobalFactor, None);
        dag.add_node("B", NodeKind::AssetReturn, None);
        dag.add_node("C", NodeKind::GlobalFactor, None);
        dag.add_edge("A", "B", EdgeKind::Causal, 0);
        dag.add_edge("C", "B", EdgeKind::Causal, 0);

        // Conditioned on B: collider is opened
        assert!(!d_separated(&dag, "A", "C", &["B"]));
    }

    /// Collider with descendant conditioning:
    /// A → B ← C, B → D. Conditioning on D (descendant of collider) opens A--C.
    #[test]
    fn test_collider_conditioned_on_descendant() {
        let mut dag = CausalDag::new();
        dag.add_node("A", NodeKind::GlobalFactor, None);
        dag.add_node("B", NodeKind::AssetReturn, None);
        dag.add_node("C", NodeKind::GlobalFactor, None);
        dag.add_node("D", NodeKind::AssetCovariate, None);
        dag.add_edge("A", "B", EdgeKind::Causal, 0);
        dag.add_edge("C", "B", EdgeKind::Causal, 0);
        dag.add_edge("B", "D", EdgeKind::Causal, 0);

        // Conditioning on D (descendant of collider B) opens the path
        assert!(!d_separated(&dag, "A", "C", &["D"]));
    }

    /// No edges at all: all pairs are d-separated.
    #[test]
    fn test_disconnected_graph() {
        let mut dag = CausalDag::new();
        dag.add_node("A", NodeKind::GlobalFactor, None);
        dag.add_node("B", NodeKind::GlobalFactor, None);
        dag.add_node("C", NodeKind::GlobalFactor, None);

        assert!(d_separated(&dag, "A", "B", &[]));
        assert!(d_separated(&dag, "A", "C", &[]));
        assert!(d_separated(&dag, "B", "C", &[]));
    }

    /// Diamond: A → B, A → C, B → D, C → D
    /// Multiple paths.
    #[test]
    fn test_diamond() {
        let mut dag = CausalDag::new();
        dag.add_node("A", NodeKind::GlobalFactor, None);
        dag.add_node("B", NodeKind::GlobalFactor, None);
        dag.add_node("C", NodeKind::GlobalFactor, None);
        dag.add_node("D", NodeKind::AssetReturn, None);
        dag.add_edge("A", "B", EdgeKind::Causal, 0);
        dag.add_edge("A", "C", EdgeKind::Causal, 0);
        dag.add_edge("B", "D", EdgeKind::Causal, 0);
        dag.add_edge("C", "D", EdgeKind::Causal, 0);

        // A → D through both B and C
        assert!(!d_separated(&dag, "A", "D", &[]));

        // Condition on B: still connected via A → C → D
        assert!(!d_separated(&dag, "A", "D", &["B"]));

        // Condition on B and C: both paths blocked
        assert!(d_separated(&dag, "A", "D", &["B", "C"]));
    }

    #[test]
    fn test_d_connected_set() {
        let mut dag = CausalDag::new();
        dag.add_node("A", NodeKind::GlobalFactor, None);
        dag.add_node("B", NodeKind::GlobalFactor, None);
        dag.add_node("C", NodeKind::AssetReturn, None);
        dag.add_node("D", NodeKind::AssetReturn, None); // isolated
        dag.add_edge("A", "B", EdgeKind::Causal, 0);
        dag.add_edge("B", "C", EdgeKind::Causal, 0);

        let connected = d_connected_set(&dag, "A", &[]);
        assert!(connected.contains(&"B".to_string()));
        assert!(connected.contains(&"C".to_string()));
        assert!(!connected.contains(&"D".to_string()));
    }
}
