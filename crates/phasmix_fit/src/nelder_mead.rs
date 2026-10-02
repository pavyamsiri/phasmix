//! A single bounded local search in scaled, free-parameter coordinates.

use super::{FitOptimizationResult, OptimizerDiagnostics};
use basin::{BoxConstraints, CostFunction};
use core::fmt;
use phasmix_deopt::{DifferentialEvolutionError, OptimizationError};

#[derive(Clone, Debug)]
struct ScaledObjective<'prob, C> {
    objective: &'prob C,
    template: Vec<f64>,
    bounds: &'prob [(f64, f64)],
    free_indices: Vec<usize>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl<C> ScaledObjective<'_, C> {
    fn expand(&self, scaled: &[f64]) -> Vec<f64> {
        let mut full = self.template.clone();
        for (&index, &value) in self.free_indices.iter().zip(scaled) {
            let (lower, upper) = self.bounds[index];
            full[index] = ((1.0 - value).mul_add(lower, value * upper)).clamp(lower, upper);
        }
        full
    }
}

impl<C: CostFunction<Param = Vec<f64>, Output = f64>> CostFunction for ScaledObjective<'_, C> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = C::Error;

    fn cost(&self, param: &Self::Param) -> Result<f64, Self::Error> {
        self.objective.cost(&self.expand(param))
    }
}

impl<C: CostFunction<Param = Vec<f64>, Output = f64>> BoxConstraints for ScaledObjective<'_, C> {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

pub(super) fn minimize<C>(
    objective: &C,
    bounds: &[(f64, f64)],
    warm_start: Option<&[f64]>,
    max_iter: usize,
) -> Result<FitOptimizationResult, OptimizationError<C::Error>>
where
    C: CostFunction<Param = Vec<f64>, Output = f64> + Clone + fmt::Debug + BoxConstraints,
    C::Error: fmt::Display,
{
    let Some(start) = warm_start else {
        return Err(DifferentialEvolutionError::InvalidWarmStart.into());
    };
    if start.len() != bounds.len()
        || !start
            .iter()
            .zip(bounds)
            .all(|(&value, &(lower, upper))| value.is_finite() && value >= lower && value <= upper)
    {
        return Err(DifferentialEvolutionError::InvalidWarmStart.into());
    }
    for (index, &(lower, upper)) in bounds.iter().enumerate() {
        if !lower.is_finite() || !upper.is_finite() || !(upper - lower).is_finite() || lower > upper
        {
            return Err(DifferentialEvolutionError::InvalidBound { index }.into());
        }
    }
    let free_indices: Vec<_> = bounds
        .iter()
        .enumerate()
        .filter_map(|(index, &(lower, upper))| (lower < upper).then_some(index))
        .collect();
    if free_indices.is_empty() {
        let params = start.to_vec();
        let cost = objective
            .cost(&params)
            .map_err(OptimizationError::CostFunction)?;
        return Ok(FitOptimizationResult {
            params,
            cost,
            diagnostics: OptimizerDiagnostics {
                success: cost.is_finite(),
                nfev: 1,
                nit: 0,
                message: "All parameters fixed".into(),
            },
        });
    }
    let scaled: Vec<_> = free_indices
        .iter()
        .map(|&index| {
            let (lower, upper) = bounds[index];
            (start[index] - lower) / (upper - lower)
        })
        .collect();
    let mut simplex = vec![scaled.clone()];
    for index in 0..scaled.len() {
        let mut vertex = scaled.clone();
        vertex[index] += if vertex[index] <= 0.95 { 0.05 } else { -0.05 };
        simplex.push(vertex);
    }
    let scaled_objective = ScaledObjective {
        objective,
        template: start.to_vec(),
        bounds,
        lower: vec![0.0; free_indices.len()],
        upper: vec![1.0; free_indices.len()],
        free_indices,
    };
    let result = basin::Executor::new(
        scaled_objective.clone(),
        basin::NelderMead::standard().projected(),
        basin::BasicSimplexState::from_simplex(simplex),
    )
    .max_iter(u64::try_from(max_iter).expect("iteration count fits in u64"))
    .terminate_on(basin::SimplexTolerance::new(1e-8, 1e-8))
    .run()
    .map_err(OptimizationError::CostFunction)?;
    Ok(FitOptimizationResult {
        params: scaled_objective.expand(result.best_param()),
        cost: result.best_cost(),
        diagnostics: OptimizerDiagnostics {
            success: result.best_cost().is_finite()
                && matches!(
                    result.reason,
                    basin::TerminationReason::SimplexTolerance
                        | basin::TerminationReason::SolverConverged
                ),
            nfev: result.cost_evals(),
            nit: result.iter(),
            message: format!("Nelder-Mead: {:?}", result.reason),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;
    use core::convert::Infallible;
    use std::sync::Mutex;

    #[derive(Clone, Debug)]
    struct Objective {
        lower: Vec<f64>,
        upper: Vec<f64>,
        evaluations: Arc<Mutex<Vec<Vec<f64>>>>,
    }

    impl CostFunction for Objective {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, param: &Self::Param) -> Result<f64, Self::Error> {
            self.evaluations.lock().unwrap().push(param.clone());
            Ok(((param[0] - 2e-9) / 1e-9).powi(2) + ((param[2] - 150.0) / 50.0).powi(2))
        }
    }

    impl BoxConstraints for Objective {
        fn lower(&self) -> &Vec<f64> {
            &self.lower
        }
        fn upper(&self) -> &Vec<f64> {
            &self.upper
        }
    }

    fn objective() -> Objective {
        Objective {
            lower: vec![1e-9, 7.0, 100.0],
            upper: vec![3e-9, 7.0, 200.0],
            evaluations: Arc::default(),
        }
    }

    #[test]
    fn single_local_search_scales_free_parameters_and_preserves_fixed_values() {
        let objective = objective();
        let bounds = [(1e-9, 3e-9), (7.0, 7.0), (100.0, 200.0)];
        let start = [3e-9, 7.0, 200.0];
        let result = minimize(&objective, &bounds, Some(&start), 200).unwrap();
        assert!(result.cost < 1e-12);
        assert!(result.diagnostics.success);
        assert!(result.diagnostics.nit > 0);
        let evaluations = objective.evaluations.lock().unwrap().clone();
        assert_eq!(evaluations[0], start);
        assert_eq!(
            result.diagnostics.nfev,
            u64::try_from(evaluations.len()).unwrap()
        );
        for point in evaluations {
            assert!(
                point
                    .iter()
                    .zip(bounds)
                    .all(|(&value, (lower, upper))| value >= lower && value <= upper)
            );
        }
    }

    #[test]
    fn iteration_limit_does_not_trigger_global_sampling() {
        let objective = objective();
        let result = minimize(
            &objective,
            &[(1e-9, 3e-9), (7.0, 7.0), (100.0, 200.0)],
            Some(&[3e-9, 7.0, 200.0]),
            1,
        )
        .unwrap();
        assert!(result.diagnostics.nfev <= 7);
        assert!(!result.diagnostics.success);
        assert_eq!(result.diagnostics.nit, 1);
        assert!(result.diagnostics.message.contains("MaxIter"));
        assert!(result.cost <= 2.0);
    }

    #[test]
    fn all_fixed_parameters_need_one_evaluation() {
        let objective = objective();
        let start = [2e-9, 7.0, 150.0];
        let bounds = [(2e-9, 2e-9), (7.0, 7.0), (150.0, 150.0)];
        let result = minimize(&objective, &bounds, Some(&start), 200).unwrap();
        assert_eq!(result.diagnostics.nfev, 1);
        assert!(result.diagnostics.success);
        assert_eq!(result.params, start);
    }

    #[test]
    fn invalid_or_missing_starts_fail_before_evaluation() {
        let objective = objective();
        let bounds = [(1e-9, 3e-9), (7.0, 7.0), (100.0, 200.0)];
        for start in [
            None,
            Some(&[][..]),
            Some(&[f64::NAN, 7.0, 150.0][..]),
            Some(&[2e-9, 8.0, 150.0][..]),
        ] {
            assert!(matches!(
                minimize(&objective, &bounds, start, 200),
                Err(OptimizationError::DifferentialEvolution(
                    DifferentialEvolutionError::InvalidWarmStart
                ))
            ));
        }
        assert!(objective.evaluations.lock().unwrap().is_empty());
    }
}
