use serde::{Deserialize, Serialize};

/// Unique node identifier (index into the graph).
pub type NodeId = usize;

/// What role this node plays in the SCM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeKind {
    /// Global DeFi factor (LiqFlow, StableFlow, etc.) — affects all assets.
    GlobalFactor,
    /// Per-asset fundamental (whale_conc, protocol_rev, etc.).
    AssetCovariate,
    /// Log return of an asset.
    AssetReturn,
    /// Instrumental variable used for 2SLS identification.
    Instrument,
    /// Latent / unobserved shock (exploits, halts, cascades).
    UnobservedShock,
    /// Macroeconomic factor from FRED (interest rates, inflation, etc.).
    MacroFactor,
}

/// A node in the causal DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub name: String,
    pub kind: NodeKind,
    /// `None` for global/macro factors; `Some("eth")` for asset-specific nodes.
    pub asset: Option<String>,
}

/// Semantic type of a directed edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeKind {
    /// X causes Y.
    Causal,
    /// Z is an instrument for the X → Y relationship.
    Instrumental,
    /// Bidirectional association (latent common cause).
    Confounded,
}

/// A directed edge in the causal DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub kind: EdgeKind,
    /// Temporal lag in days. 0 = contemporaneous, 1 = X_{t-1} → Y_t.
    pub lag: i32,
}

impl std::fmt::Display for NodeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeKind::GlobalFactor => write!(f, "GlobalFactor"),
            NodeKind::AssetCovariate => write!(f, "AssetCovariate"),
            NodeKind::AssetReturn => write!(f, "AssetReturn"),
            NodeKind::Instrument => write!(f, "Instrument"),
            NodeKind::UnobservedShock => write!(f, "UnobservedShock"),
            NodeKind::MacroFactor => write!(f, "MacroFactor"),
        }
    }
}

impl std::fmt::Display for EdgeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EdgeKind::Causal => write!(f, "Causal"),
            EdgeKind::Instrumental => write!(f, "Instrumental"),
            EdgeKind::Confounded => write!(f, "Confounded"),
        }
    }
}
