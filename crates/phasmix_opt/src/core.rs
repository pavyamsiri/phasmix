//! Common types and utilities.
use core::cmp;

use thiserror::Error;

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

/// A failure in the objective or the optimizer.
#[derive(Debug, Error)]
pub enum OptimizationError<C, O> {
    /// An objective evaluation failed, including during local polishing.
    CostFunction(C),
    /// The optimizer configuration or bookkeeping is invalid.
    Optimizer(O),
}

/// A point in parameter space and its cost.
/// Used so that points in parameter space can be ordered.
pub(crate) struct OrderedPoint {
    /// The cost of the point.
    pub(crate) cost: f64,
    /// The point in parameter space.
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
