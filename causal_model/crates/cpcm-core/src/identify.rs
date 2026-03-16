use std::collections::HashSet;

use crate::dag::CausalDag;
use crate::dsep::d_separated;
use crate::types::NodeKind;

/// Result of checking whether a causal effect is identified.
#[derive(Debug, Clone)]
pub struct IdentificationResult {
    pub treatment: String,
    pub outcome: String,
    pub identified: bool,
    pub method: IdentificationMethod,
    pub adjustment_set: Vec<String>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdentificationMethod {
    /// Identified via the backdoor criterion with an adjustment set.
    Backdoor,
    /// Identified via an instrumental variable.
    Iv { instrument: String },
    /// Not identified — confounding cannot be blocked.
    NotIdentified,
}

/// Find a valid **backdoor adjustment set** for the effect of `treatment` on `outcome`.
///
/// The backdoor criterion (Pearl 1995) says: a set S satisfies the backdoor criterion
/// relative to (X, Y) if:
/// 1. No node in S is a descendant of X.
/// 2. S blocks every path between X and Y that contains an arrow INTO X (backdoor paths).
///
/// We search for a minimal set among the observed (non-latent) non-descendant nodes of X.
/// Returns `None` if no valid set exists (e.g., unblocked latent confounding).
pub fn backdoor_adjustment_set(
    dag: &CausalDag,
    treatment: &str,
    outcome: &str,
) -> Option<Vec<String>> {
    let descendants_of_treatment: HashSet<String> = dag
        .descendants(treatment)
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    // Candidate nodes: observed (not UnobservedShock), not the treatment, not the outcome,
    // and not a descendant of treatment.
    let candidates: Vec<String> = dag
        .node_names()
        .into_iter()
        .filter(|&name| {
            name != treatment
                && name != outcome
                && !descendants_of_treatment.contains(name)
                && dag.node_by_name(name).map_or(false, |n| {
                    n.kind != NodeKind::UnobservedShock
                })
        })
        .map(|s| s.to_string())
        .collect();

    // Try the empty set first (no confounders).
    if blocks_all_backdoor_paths(dag, treatment, outcome, &[]) {
        return Some(vec![]);
    }

    // Try each single candidate.
    for c in &candidates {
        let set = vec![c.as_str()];
        if blocks_all_backdoor_paths(dag, treatment, outcome, &set) {
            return Some(vec![c.clone()]);
        }
    }

    // Try pairs.
    for i in 0..candidates.len() {
        for j in (i + 1)..candidates.len() {
            let set = vec![candidates[i].as_str(), candidates[j].as_str()];
            if blocks_all_backdoor_paths(dag, treatment, outcome, &set) {
                return Some(vec![candidates[i].clone(), candidates[j].clone()]);
            }
        }
    }

    // Try the full candidate set as a last resort.
    let full_set: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();
    if blocks_all_backdoor_paths(dag, treatment, outcome, &full_set) {
        return Some(candidates);
    }

    None // unidentified
}

/// Check if `adjustment_set` blocks all backdoor paths from `treatment` to `outcome`.
///
/// A backdoor path is any path that starts with an arrow INTO `treatment`.
/// We check this by: in a modified graph where we remove all outgoing edges from
/// treatment, is treatment d-separated from outcome given the adjustment set?
///
/// Shortcut: we check d-separation in the original graph between all parents of
/// treatment and the outcome, conditioning on the adjustment set + treatment.
fn blocks_all_backdoor_paths(
    dag: &CausalDag,
    treatment: &str,
    outcome: &str,
    adjustment_set: &[&str],
) -> bool {
    let parents = dag.parents(treatment);

    if parents.is_empty() {
        // No parents = no backdoor paths = always identified with empty set.
        return true;
    }

    // For each parent of treatment, check if it's d-separated from outcome
    // given the adjustment set (we don't include treatment itself in the
    // conditioning set for this check — we want to see if there's a non-causal
    // path from parent-of-treatment to outcome that bypasses treatment).
    //
    // More precisely: remove the treatment→outcome edge conceptually and check if
    // treatment is d-separated from outcome given the adjustment set.
    // We approximate this by checking: for each parent P of treatment, is P d-separated
    // from outcome given {adjustment_set ∪ treatment}?
    let mut cond: Vec<&str> = adjustment_set.to_vec();
    if !cond.contains(&treatment) {
        cond.push(treatment);
    }

    for parent in &parents {
        // Skip if parent is an unobserved shock — can't condition on it.
        if let Some(node) = dag.node_by_name(parent) {
            if node.kind == NodeKind::UnobservedShock {
                // Unobserved confounder: check if it's already blocked by the set.
                // If it has a path to outcome not through treatment, we can't block it.
                if !d_separated(dag, parent, outcome, &cond) {
                    return false;
                }
                continue;
            }
        }

        if !d_separated(dag, parent, outcome, &cond) {
            return false;
        }
    }

    true
}

/// Result of IV validity check.
#[derive(Debug, Clone)]
pub struct IvValidityResult {
    pub instrument: String,
    pub treatment: String,
    pub outcome: String,
    pub relevant: bool,
    pub excludable: bool,
    pub valid: bool,
    pub reason: String,
}

/// Check whether `instrument` is a valid IV for the effect of `treatment` on `outcome`.
///
/// Two conditions (graph-based):
/// 1. **Relevance**: instrument is NOT d-separated from treatment (unconditionally).
///    There must be an active path from Z to X.
/// 2. **Exclusion restriction**: instrument IS d-separated from outcome given treatment.
///    The only path from Z to Y goes through X.
pub fn check_iv_validity(
    dag: &CausalDag,
    instrument: &str,
    treatment: &str,
    outcome: &str,
) -> IvValidityResult {
    // Relevance: Z ⊥̸ X | ∅  (Z is NOT d-separated from X unconditionally)
    let relevant = !d_separated(dag, instrument, treatment, &[]);

    // Exclusion: Z ⊥ Y | X  (Z IS d-separated from Y given X)
    let excludable = d_separated(dag, instrument, outcome, &[treatment]);

    let valid = relevant && excludable;

    let reason = if valid {
        "Valid IV: relevant and excludable".to_string()
    } else if !relevant {
        format!(
            "Invalid IV: {} is d-separated from {} (no relevance)",
            instrument, treatment
        )
    } else {
        format!(
            "Invalid IV: {} is NOT d-separated from {} given {} (exclusion violated)",
            instrument, outcome, treatment
        )
    };

    IvValidityResult {
        instrument: instrument.to_string(),
        treatment: treatment.to_string(),
        outcome: outcome.to_string(),
        relevant,
        excludable,
        valid,
        reason,
    }
}

/// Run identification analysis for all treatment-outcome pairs in the DAG.
///
/// Treats all `GlobalFactor` and `MacroFactor` nodes as potential treatments,
/// and all `AssetReturn` nodes as outcomes.
pub fn identify_all_effects(dag: &CausalDag) -> Vec<IdentificationResult> {
    let treatments: Vec<String> = dag
        .node_names()
        .into_iter()
        .filter(|&name| {
            dag.node_by_name(name).map_or(false, |n| {
                matches!(n.kind, NodeKind::GlobalFactor | NodeKind::MacroFactor)
            })
        })
        .map(|s| s.to_string())
        .collect();

    let outcomes: Vec<String> = dag
        .nodes_of_kind(NodeKind::AssetReturn)
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    let instruments: Vec<String> = dag
        .nodes_of_kind(NodeKind::Instrument)
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    let mut results = Vec::new();

    for treatment in &treatments {
        for outcome in &outcomes {
            // Try backdoor first.
            if let Some(adj_set) = backdoor_adjustment_set(dag, treatment, outcome) {
                results.push(IdentificationResult {
                    treatment: treatment.clone(),
                    outcome: outcome.clone(),
                    identified: true,
                    method: IdentificationMethod::Backdoor,
                    adjustment_set: adj_set,
                    reason: "Identified via backdoor criterion".to_string(),
                });
                continue;
            }

            // Try each instrument.
            let mut found_iv = false;
            for iv in &instruments {
                let validity = check_iv_validity(dag, iv, treatment, outcome);
                if validity.valid {
                    results.push(IdentificationResult {
                        treatment: treatment.clone(),
                        outcome: outcome.clone(),
                        identified: true,
                        method: IdentificationMethod::Iv {
                            instrument: iv.clone(),
                        },
                        adjustment_set: vec![iv.clone()],
                        reason: format!("Identified via IV: {iv}"),
                    });
                    found_iv = true;
                    break;
                }
            }

            if !found_iv {
                results.push(IdentificationResult {
                    treatment: treatment.clone(),
                    outcome: outcome.clone(),
                    identified: false,
                    method: IdentificationMethod::NotIdentified,
                    adjustment_set: vec![],
                    reason: "No valid backdoor set or instrument found".to_string(),
                });
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::CausalDag;
    use crate::types::{EdgeKind, NodeKind};

    /// Simple identified case: X → Y with no confounders.
    #[test]
    fn test_backdoor_no_confounders() {
        let mut dag = CausalDag::new();
        dag.add_node("X", NodeKind::GlobalFactor, None);
        dag.add_node("Y", NodeKind::AssetReturn, Some("eth"));
        dag.add_edge("X", "Y", EdgeKind::Causal, 0);

        let adj = backdoor_adjustment_set(&dag, "X", "Y");
        assert!(adj.is_some());
        assert!(adj.unwrap().is_empty()); // empty set suffices
    }

    /// Confounded case: X → Y, Z → X, Z → Y.
    /// Conditioning on Z blocks the backdoor path X ← Z → Y.
    #[test]
    fn test_backdoor_with_confounder() {
        let mut dag = CausalDag::new();
        dag.add_node("X", NodeKind::GlobalFactor, None);
        dag.add_node("Y", NodeKind::AssetReturn, Some("eth"));
        dag.add_node("Z", NodeKind::MacroFactor, None);
        dag.add_edge("X", "Y", EdgeKind::Causal, 0);
        dag.add_edge("Z", "X", EdgeKind::Causal, 0);
        dag.add_edge("Z", "Y", EdgeKind::Causal, 0);

        let adj = backdoor_adjustment_set(&dag, "X", "Y");
        assert!(adj.is_some());
        let set = adj.unwrap();
        assert!(set.contains(&"Z".to_string()));
    }

    /// Valid IV: Z → X → Y, no direct Z → Y.
    #[test]
    fn test_iv_valid() {
        let mut dag = CausalDag::new();
        dag.add_node("Z", NodeKind::Instrument, None);
        dag.add_node("X", NodeKind::GlobalFactor, None);
        dag.add_node("Y", NodeKind::AssetReturn, Some("eth"));
        dag.add_edge("Z", "X", EdgeKind::Causal, 1);
        dag.add_edge("X", "Y", EdgeKind::Causal, 0);

        let result = check_iv_validity(&dag, "Z", "X", "Y");
        assert!(result.valid);
        assert!(result.relevant);
        assert!(result.excludable);
    }

    /// Invalid IV: Z → X → Y, but also Z → Y (violates exclusion).
    #[test]
    fn test_iv_exclusion_violated() {
        let mut dag = CausalDag::new();
        dag.add_node("Z", NodeKind::Instrument, None);
        dag.add_node("X", NodeKind::GlobalFactor, None);
        dag.add_node("Y", NodeKind::AssetReturn, Some("eth"));
        dag.add_edge("Z", "X", EdgeKind::Causal, 0);
        dag.add_edge("X", "Y", EdgeKind::Causal, 0);
        dag.add_edge("Z", "Y", EdgeKind::Causal, 0); // violates exclusion

        let result = check_iv_validity(&dag, "Z", "X", "Y");
        assert!(!result.valid);
        assert!(result.relevant);
        assert!(!result.excludable);
    }

    /// Invalid IV: Z is disconnected from X (no relevance).
    #[test]
    fn test_iv_no_relevance() {
        let mut dag = CausalDag::new();
        dag.add_node("Z", NodeKind::Instrument, None);
        dag.add_node("X", NodeKind::GlobalFactor, None);
        dag.add_node("Y", NodeKind::AssetReturn, Some("eth"));
        dag.add_edge("X", "Y", EdgeKind::Causal, 0);
        // Z has no edges — not relevant

        let result = check_iv_validity(&dag, "Z", "X", "Y");
        assert!(!result.valid);
        assert!(!result.relevant);
    }

    /// identify_all_effects with a simple DAG.
    #[test]
    fn test_identify_all_simple() {
        let mut dag = CausalDag::new();
        dag.add_node("F1", NodeKind::GlobalFactor, None);
        dag.add_node("eth_return", NodeKind::AssetReturn, Some("eth"));
        dag.add_edge("F1", "eth_return", EdgeKind::Causal, 0);

        let results = identify_all_effects(&dag);
        assert_eq!(results.len(), 1);
        assert!(results[0].identified);
        assert_eq!(results[0].method, IdentificationMethod::Backdoor);
    }
}
