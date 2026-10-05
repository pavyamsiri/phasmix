//! Common types and utilities.
use core::cmp;

/// Result of a global optimizer and additional diagnostics.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// The best parameters.
    pub params: Vec<f64>,
    /// The best cost.
    pub cost: f64,
    /// The number of function evaluations.
    pub nfev: u64,
}

pub(crate) struct OrderedPoint {
    pub(crate) cost: f64,
    pub(crate) point: Vec<f64>,
}

impl PartialEq for OrderedPoint {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for OrderedPoint {}

impl PartialOrd for OrderedPoint {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedPoint {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        // Max-heap by cost — so the *worst* kept point is always at the top
        self.cost
            .partial_cmp(&other.cost)
            .unwrap_or(cmp::Ordering::Equal)
    }
}
