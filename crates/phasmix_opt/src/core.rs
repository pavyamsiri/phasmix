//! Common types and utilities.

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
