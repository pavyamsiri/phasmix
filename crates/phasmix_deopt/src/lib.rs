extern crate alloc;

use basin::CostFunction;
#[cfg(test)]
use core::convert;
use core::{array, error::Error, fmt};
use rand::{RngExt as _, distr::Uniform, seq::index::sample};
use rayon::prelude::*;

#[derive(Debug)]
pub struct OptimizationResult {
    pub params: Vec<f64>,
    pub cost: f64,
    pub nfev: u64,
}

#[derive(Debug, Clone)]
pub struct DifferentialEvolution {
    pub pop_size: usize,
    pub max_iter: usize,
    pub atol: f64,
    pub rtol: f64,
}

/// A failure in the objective or the differential evolution optimizer.
#[derive(Debug)]
pub enum OptimizationError<E> {
    /// An objective evaluation failed, including during local polishing.
    CostFunction(E),
    /// The optimizer configuration or bookkeeping is invalid.
    DifferentialEvolution(DifferentialEvolutionError),
}

/// Errors specific to differential evolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifferentialEvolutionError {
    /// `rand/1` requires at least four population members.
    PopulationTooSmall { size: usize },
    /// At least one parameter is required.
    EmptyBounds,
    /// A bound is non-finite, reversed, or cannot be sampled safely.
    InvalidBound { index: usize },
    /// Polishing must use the same bounds as the global search.
    InconsistentBoxConstraints,
    /// Tolerances must be finite and nonnegative.
    InvalidTolerance,
    /// The evaluation count cannot be represented as a `u64`.
    EvaluationCountOverflow,
    /// No population member is available for selection.
    EmptyPopulation,
    /// Warm start has the wrong length, non-finite values, or violates bounds.
    InvalidWarmStart,
}

impl fmt::Display for DifferentialEvolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PopulationTooSmall { size } => {
                write!(f, "population size {size} is smaller than four")
            }
            Self::EmptyBounds => f.write_str("at least one parameter bound is required"),
            Self::InvalidBound { index } => {
                write!(f, "invalid or unsampleable bound at index {index}")
            }
            Self::InconsistentBoxConstraints => {
                f.write_str("objective box constraints do not match the search bounds")
            }
            Self::InvalidTolerance => f.write_str("tolerances must be finite and nonnegative"),
            Self::EvaluationCountOverflow => f.write_str("function evaluation count exceeds u64"),
            Self::InvalidWarmStart => {
                f.write_str("warm start must match bounds and contain finite in-bounds values")
            }
            Self::EmptyPopulation => f.write_str("cannot select from an empty population"),
        }
    }
}

impl Error for DifferentialEvolutionError {}

impl<E: fmt::Display> fmt::Display for OptimizationError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CostFunction(error) => write!(f, "objective evaluation failed: {error}"),
            Self::DifferentialEvolution(error) => error.fmt(f),
        }
    }
}

impl<E: Error + 'static> Error for OptimizationError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::CostFunction(error) => Some(error),
            Self::DifferentialEvolution(error) => Some(error),
        }
    }
}

impl<E> From<DifferentialEvolutionError> for OptimizationError<E> {
    fn from(error: DifferentialEvolutionError) -> Self {
        Self::DifferentialEvolution(error)
    }
}

type EvaluatedPopulation = Vec<(Vec<f64>, f64)>;

const fn replace_nan(cost: f64) -> f64 {
    if cost.is_nan() { f64::INFINITY } else { cost }
}

impl DifferentialEvolution {
    fn validate<C>(
        &self,
        cost_func: &C,
        bounds: &[(f64, f64)],
    ) -> Result<(), DifferentialEvolutionError>
    where
        C: basin::BoxConstraints<Param = Vec<f64>>,
    {
        if self.pop_size < 4 {
            return Err(DifferentialEvolutionError::PopulationTooSmall {
                size: self.pop_size,
            });
        }
        if bounds.is_empty() {
            return Err(DifferentialEvolutionError::EmptyBounds);
        }
        if !self.atol.is_finite() || self.atol < 0.0 || !self.rtol.is_finite() || self.rtol < 0.0 {
            return Err(DifferentialEvolutionError::InvalidTolerance);
        }
        for (index, &(lower, upper)) in bounds.iter().enumerate() {
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                return Err(DifferentialEvolutionError::InvalidBound { index });
            }
        }
        #[expect(
            clippy::float_cmp,
            reason = "global search and polishing must use identical bounds"
        )]
        let matching_bounds = cost_func.lower().len() == bounds.len()
            && cost_func.upper().len() == bounds.len()
            && bounds.iter().enumerate().all(|(index, &(lower, upper))| {
                cost_func.lower()[index] == lower && cost_func.upper()[index] == upper
            });
        if !matching_bounds {
            return Err(DifferentialEvolutionError::InconsistentBoxConstraints);
        }
        Ok(())
    }

    fn converged(&self, population: &[(Vec<f64>, f64)], population_size: f64) -> bool {
        let mean = population.iter().map(|(_, cost)| *cost).sum::<f64>() / population_size;
        let variance = population
            .iter()
            .map(|(_, cost)| {
                let residual = cost - mean;
                residual * residual
            })
            .sum::<f64>()
            / population_size;
        let std = variance.sqrt();
        mean.is_finite() && std.is_finite() && std <= self.rtol.mul_add(mean.abs(), self.atol)
    }

    fn evaluate_population<C>(
        cost_func: &C,
        population: Vec<Vec<f64>>,
    ) -> Result<EvaluatedPopulation, OptimizationError<C::Error>>
    where
        C: CostFunction<Param = Vec<f64>, Output = f64> + Sync,
        C::Error: Send,
    {
        population
            .into_par_iter()
            .map(|parameters| {
                let cost = replace_nan(
                    cost_func
                        .cost(&parameters)
                        .map_err(OptimizationError::CostFunction)?,
                );
                Ok((parameters, cost))
            })
            .collect()
    }

    fn polish<C>(
        cost_func: &C,
        best_member: &[f64],
        best_cost: f64,
        nfev: u64,
    ) -> Result<OptimizationResult, OptimizationError<C::Error>>
    where
        C: CostFunction<Param = Vec<f64>, Output = f64> + Clone + basin::BoxConstraints,
    {
        let polished = basin::Executor::new(
            cost_func.clone(),
            basin::NelderMead::standard().projected(),
            basin::BasicSimplexState::new(best_member.to_owned()),
        )
        .max_iter(200)
        .run()
        .map_err(OptimizationError::CostFunction)?;
        let nfev = nfev
            .checked_add(polished.cost_evals())
            .ok_or(DifferentialEvolutionError::EvaluationCountOverflow)?;
        let polished_cost = replace_nan(polished.best_cost());
        if polished_cost <= best_cost {
            Ok(OptimizationResult {
                params: polished.best_param().to_owned(),
                cost: polished_cost,
                nfev,
            })
        } else {
            Ok(OptimizationResult {
                params: best_member.to_owned(),
                cost: best_cost,
                nfev,
            })
        }
    }

    /// Globally minimize the given cost function using differential evolution.
    ///
    /// Initial and trial costs are evaluated in parallel using Rayon. Each
    /// generation proposes trials from a fixed population and applies selection
    /// only after all trial evaluations succeed. Random draws remain serial.
    ///
    /// NaN costs are treated as positive infinity. Existing infinite costs
    /// are preserved. Local polishing uses the objective's box constraints.
    ///
    /// # Errors
    /// Returns an error for invalid settings or bounds, an overflowing evaluation
    /// count, or an objective failure during initialization, evolution, or polishing.
    pub fn minimize<C>(
        &self,
        cost_func: &C,
        bounds: &[(f64, f64)],
    ) -> Result<OptimizationResult, OptimizationError<C::Error>>
    where
        C: CostFunction<Param = Vec<f64>, Output = f64>
            + Clone
            + fmt::Debug
            + Sync
            + Send
            + basin::BoxConstraints,
        C::Error: Send + fmt::Display,
    {
        self.minimize_with_warm_start(cost_func, bounds, None)
    }

    /// Append a warm start to the configured random population.
    ///
    /// # Errors
    /// Returns an error for invalid warm starts, settings, or objective failures.
    pub fn minimize_with_warm_start<C>(
        &self,
        cost_func: &C,
        bounds: &[(f64, f64)],
        warm_start: Option<&[f64]>,
    ) -> Result<OptimizationResult, OptimizationError<C::Error>>
    where
        C: CostFunction<Param = Vec<f64>, Output = f64>
            + Clone
            + fmt::Debug
            + Sync
            + Send
            + basin::BoxConstraints,
        C::Error: Send + fmt::Display,
    {
        self.validate(cost_func, bounds)?;
        if let Some(point) = warm_start
            && (point.len() != bounds.len()
                || !point
                    .iter()
                    .zip(bounds)
                    .all(|(&value, &(lb, ub))| value.is_finite() && value >= lb && value <= ub))
        {
            return Err(DifferentialEvolutionError::InvalidWarmStart.into());
        }
        let population_size = self
            .pop_size
            .checked_add(usize::from(warm_start.is_some()))
            .ok_or(DifferentialEvolutionError::EvaluationCountOverflow)?;
        let evaluations_per_generation = u64::try_from(population_size)
            .map_err(|_error| DifferentialEvolutionError::EvaluationCountOverflow)?;
        let mut nfev = evaluations_per_generation;

        // Step 1: Create initial population
        let mut rng = rand::rng();
        let uniform_bounds = bounds
            .iter()
            .enumerate()
            .map(|(index, &(lower, upper))| {
                Uniform::new_inclusive(lower, upper)
                    .map_err(|_error| DifferentialEvolutionError::InvalidBound { index })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut initial_population: Vec<Vec<f64>> = (0..self.pop_size)
            .map(|_| {
                uniform_bounds
                    .iter()
                    .map(|current_bounds| rng.sample(current_bounds))
                    .collect()
            })
            .collect();
        if let Some(point) = warm_start {
            initial_population.push(point.to_vec());
        }
        let mut population = Self::evaluate_population(cost_func, initial_population)?;

        let num_members = population.len();
        // Population lengths need only an approximate floating-point representation
        // for convergence statistics, unlike the exact integer evaluation counter.
        #[expect(
            clippy::cast_precision_loss,
            reason = "population length is used only for statistical averaging"
        )]
        let population_size_f64 = num_members as f64;

        // Step 2: Create donor vectors
        // v[i] = z[a] + F * (z[b] - z[c]) where a, b and c are not equal to i
        let weight = 0.7;
        let crossover = 0.7;
        for _ in 0..self.max_iter {
            let next_nfev = nfev
                .checked_add(evaluations_per_generation)
                .ok_or(DifferentialEvolutionError::EvaluationCountOverflow)?;
            // Generate trials in order, without changing any parent this generation.
            let mut trials = Vec::with_capacity(num_members);
            for i in 0..num_members {
                let indices = sample(&mut rng, num_members - 1, 3);
                // Sampling three indices is safe after validating pop_size >= 4.
                let [a_index, b_index, c_index] = array::from_fn(|position| {
                    let index = indices.index(position);
                    if index >= i { index + 1 } else { index }
                });
                let donor: Vec<_> = population[a_index]
                    .0
                    .iter()
                    .enumerate()
                    .map(|(param_index, value)| {
                        value
                            + weight
                                * (population[b_index].0[param_index]
                                    - population[c_index].0[param_index])
                    })
                    .collect();
                let forced = rng.random_range(0..bounds.len());
                let new_pop: Vec<_> = population[i]
                    .0
                    .iter()
                    .enumerate()
                    .map(|(param_index, value)| {
                        if param_index == forced || rng.random::<f64>() < crossover {
                            donor[param_index].clamp(bounds[param_index].0, bounds[param_index].1)
                        } else {
                            *value
                        }
                    })
                    .collect();

                trials.push(new_pop);
            }

            // Evaluate the complete generation before committing any updates.
            let evaluated_trials = Self::evaluate_population(cost_func, trials)?;
            for (parent, trial) in population.iter_mut().zip(evaluated_trials) {
                if trial.1 <= parent.1 {
                    *parent = trial;
                }
            }
            let converged = self.converged(&population, population_size_f64);
            nfev = next_nfev;
            if converged {
                break;
            }
        }

        // Step 4: Select best cost
        let (best_member, best_cost) = population
            .iter()
            .min_by(|(_, left), (_, right)| left.total_cmp(right))
            .ok_or(DifferentialEvolutionError::EmptyPopulation)?;

        // Step 5: Polish best result
        Self::polish(cost_func, best_member, *best_cost, nfev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argmin_testfunctions::rosenbrock;

    #[derive(Debug, Clone)]
    struct Rosenbrock {
        lb: Vec<f64>,
        ub: Vec<f64>,
    }

    impl basin::CostFunction for Rosenbrock {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = convert::Infallible;
        fn cost(&self, param: &Self::Param) -> Result<Self::Output, Self::Error> {
            Ok(rosenbrock(param))
        }
    }

    impl basin::BoxConstraints for Rosenbrock {
        fn lower(&self) -> &Self::Param {
            &self.lb
        }

        fn upper(&self) -> &Self::Param {
            &self.ub
        }
    }

    #[test]
    fn de_smoke() {
        let de = DifferentialEvolution {
            pop_size: 90,
            max_iter: 100,
            atol: 0.0,
            rtol: 0.01,
        };

        let prob = Rosenbrock {
            lb: vec![0.0, 0.0],
            ub: vec![100.0, 100.0],
        };
        let bounds = vec![(0.0, 100.0), (0.0, 100.0)];

        let res = de.minimize(&prob, &bounds);

        assert!(res.is_ok());

        let Ok(res) = res else {
            panic!("Assert should've caught it earlier.");
        };

        println!("cost = {}", res.cost);
        println!("params = {:?}", res.params);
        println!("nfev = {}", res.nfev);
    }
}

#[cfg(test)]
mod warm_start_tests {
    use super::*;
    use alloc::sync::Arc;
    use core::convert::Infallible;
    use std::sync::Mutex;

    #[derive(Clone, Debug)]
    struct RecordingObjective {
        lower: Vec<f64>,
        upper: Vec<f64>,
        evaluations: Arc<Mutex<Vec<Vec<f64>>>>,
    }

    impl CostFunction for RecordingObjective {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, param: &Vec<f64>) -> Result<f64, Self::Error> {
            self.evaluations.lock().unwrap().push(param.clone());
            Ok(param.iter().map(|value| (value - 0.123).powi(2)).sum())
        }
    }
    impl basin::BoxConstraints for RecordingObjective {
        fn lower(&self) -> &Vec<f64> {
            &self.lower
        }
        fn upper(&self) -> &Vec<f64> {
            &self.upper
        }
    }
    fn objective() -> RecordingObjective {
        RecordingObjective {
            lower: vec![0.0],
            upper: vec![1.0],
            evaluations: Arc::default(),
        }
    }

    #[test]
    fn warm_start_adds_population_member_and_evaluation() {
        let objective = objective();
        let optimizer = DifferentialEvolution {
            pop_size: 4,
            max_iter: 1,
            atol: 0.0,
            rtol: 0.0,
        };
        let result = optimizer
            .minimize_with_warm_start(&objective, &[(0.0, 1.0)], Some(&[0.123]))
            .unwrap();
        let points = objective.evaluations.lock().unwrap().clone();
        assert!(points[..5].contains(&vec![0.123]));
        assert_eq!(result.nfev, u64::try_from(points.len()).unwrap());
        assert_eq!(result.params, vec![0.123]);
        assert!(points.len() >= 10);
    }

    #[test]
    fn invalid_warm_starts_fail_before_evaluation() {
        let objective = objective();
        let optimizer = DifferentialEvolution {
            pop_size: 4,
            max_iter: 0,
            atol: 0.0,
            rtol: 0.0,
        };
        for point in [
            vec![],
            vec![0.0, 0.0],
            vec![f64::NAN],
            vec![f64::INFINITY],
            vec![-0.1],
            vec![1.1],
        ] {
            assert!(matches!(
                optimizer.minimize_with_warm_start(&objective, &[(0.0, 1.0)], Some(&point)),
                Err(OptimizationError::DifferentialEvolution(
                    DifferentialEvolutionError::InvalidWarmStart
                ))
            ));
        }
        assert!(objective.evaluations.lock().unwrap().is_empty());
    }
}
