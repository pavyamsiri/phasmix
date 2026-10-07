extern crate alloc;

use crate::core::{OptimizationError, OptimizationResult};
use basin::CostFunction;
#[cfg(test)]
use core::convert;
use core::default;
use core::{array, fmt};
use rand::{Rng, RngExt as _, SeedableRng as _, rngs::StdRng, seq::index::sample};
use rayon::prelude::*;
use thiserror::Error;

#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub enum MutationStrategy {
    /// The mutant vector is the current best vector plus one random difference vector.
    #[default]
    Best1,
    /// The mutant vector is the current best vector plus two random difference vectors.
    Best2,
    /// The mutant vector is a random population vector plus one random difference vector.
    Rand1,
    /// The mutant vector is a random population vector plus two random difference vectors.
    Rand2,
    /// The mutant vector is a random population vector shifted towards the current best
    /// vector and perturbed by one random difference vector.
    RandToBest,
    /// The mutant vector is the current target vector shifted towards the current best
    /// vector and perturbed by one random difference vector.
    CurrentToBest,
}

#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub enum CrossoverStrategy {
    /// Each parameter is selected between the mutant and the target independently according to the crossover rate.
    /// One parameter is guaranteed to be from the mutant.
    #[default]
    Binomial,
    // NOTE: Not implemented yet.
    // A random starting parameter is selected from the mutant, then consecutive parameters are taken from the mutant
    // while the crossover condition succeeds. The parameter array is periodic such that there are no issues with going
    // out of bounds. The remaining parameters are taken from the target.
    // Exponential,
}

#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub enum BoundaryStrategy {
    /// If the trial vector is out of bounds, it is assigned infinite cost and therefore rejected in favour of the target
    /// vector. Target vectors are guaranteed to be within bounds.
    #[default]
    Reject,
    // NOTE: Not implemented yet.
    // If the trial vector is out of bounds, each out of bounds parameter is replaced by a newly sampled in-bounds parameter.
    // Comparison with the target vector then proceeds as normal.
    // Resample,
    // If the trial vector is out of bounds, each out-of-bounds parameter is reflected across its violated boundary until
    // it lies within bounds.
    // Comparison with the target vector then proceeds as normal.
    // Reflect,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default)]
pub enum InitializationStrategy {
    LatinHyperCube,
    Sobol,
    #[default]
    Independent,
}

impl InitializationStrategy {
    fn generate_initial_population(
        self,
        pop_size: usize,
        num_parameters: usize,
        rng: &mut impl Rng,
    ) -> Vec<Vec<f64>> {
        match self {
            InitializationStrategy::LatinHyperCube => todo!(),
            InitializationStrategy::Sobol => todo!(),
            InitializationStrategy::Independent => (0..pop_size)
                .map(|_| {
                    (0..num_parameters)
                        .map(|_| rng.random_range(0.0..=1.0))
                        .collect()
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DifferentialEvolutionConfig {
    /// Number of random population members per parameter, excluding the warm start.
    pub pop_size_factor: usize,
    pub max_iter: usize,
    pub crossover_rate: f32,
    pub mutation_factor: f64,
    pub atol: f64,
    pub rtol: f64,
    pub mutation: MutationStrategy,
    pub crossover: CrossoverStrategy,
    pub boundary: BoundaryStrategy,
    pub initializer: InitializationStrategy,
}

impl default::Default for DifferentialEvolutionConfig {
    fn default() -> Self {
        Self {
            pop_size_factor: 4,
            max_iter: 10,
            crossover_rate: 0.7,
            mutation_factor: 0.5,
            atol: 0.0,
            rtol: 0.01,
            mutation: MutationStrategy::default(),
            crossover: CrossoverStrategy::default(),
            boundary: BoundaryStrategy::default(),
            initializer: InitializationStrategy::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DifferentialEvolution {
    pop_size_factor: usize,
    max_iter: usize,
    crossover_rate: f32,
    mutation_factor: f64,
    atol: f64,
    rtol: f64,
    mutation: MutationStrategy,
    crossover: CrossoverStrategy,
    boundary: BoundaryStrategy,
    initializer: InitializationStrategy,
}

impl DifferentialEvolution {
    /// Initialize a new differential evolution optimizer given a configuration.
    ///
    /// # Errors
    /// Initialization will fail if any of the configuration is invalid:
    /// - `pop_size_factor` is too small; less than 4.
    /// - `atol` or `rtol` are invalid tolerances; non-finite or negative.
    /// - `crossover_rate` is invalid; should be a probability in the range [0.0, 1.0].
    /// - `mutation_factor` is invalid; non-finite or negative.
    pub fn new(
        config: DifferentialEvolutionConfig,
    ) -> Result<DifferentialEvolution, DifferentialEvolutionInitializationError> {
        if config.pop_size_factor < 4 {
            return Err(
                DifferentialEvolutionInitializationError::PopulationTooSmall {
                    size: config.pop_size_factor as u8,
                },
            );
        }

        if !config.atol.is_finite() || config.atol < 0.0 {
            return Err(DifferentialEvolutionInitializationError::InvalidAbsoluteTolerance);
        }

        if !config.rtol.is_finite() || config.rtol < 0.0 {
            return Err(DifferentialEvolutionInitializationError::InvalidRelativeTolerance);
        }

        if !config.crossover_rate.is_finite() || !(0.0..=1.0).contains(&config.crossover_rate) {
            return Err(DifferentialEvolutionInitializationError::InvalidCrossoverRate);
        }

        if !config.mutation_factor.is_finite() || config.mutation_factor < 0.0 {
            return Err(DifferentialEvolutionInitializationError::InvalidMutationFactor);
        }

        Ok(Self {
            pop_size_factor: config.pop_size_factor,
            max_iter: config.max_iter,
            crossover_rate: config.crossover_rate,
            mutation_factor: config.mutation_factor,
            atol: config.atol,
            rtol: config.rtol,
            mutation: config.mutation,
            crossover: config.crossover,
            boundary: config.boundary,
            initializer: config.initializer,
        })
    }
}

/// Errors when creating a differential evolution optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum DifferentialEvolutionInitializationError {
    /// The optimizer only supports `popsize` greater than or equal to four.
    /// This is so that all strategies can be supported.
    #[error("Differential evolution requires at least four population members.")]
    PopulationTooSmall { size: u8 },
    /// Absolute tolerance must be finite and nonnegative.
    #[error("Absolute tolerance must be finite and nonnegative.")]
    InvalidAbsoluteTolerance,
    /// Relative tolerance must be finite and nonnegative.
    #[error("Relative tolerance must be finite and nonnegative.")]
    InvalidRelativeTolerance,
    /// Crossover rate must be a probability i.e. in the range [0, 1].
    #[error("The crossover rate must be a probability within the range [0, 1].")]
    InvalidCrossoverRate,
    /// Mutation factor must be finite and nonnegative.
    #[error("The mutation factor must be finite and nonnegative.")]
    InvalidMutationFactor,
}

/// Errors specific to differential evolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum DifferentialEvolutionError {
    #[error("Bounds are invalid in some way.")]
    InvalidBounds,
    #[error("Warm start was invalid.")]
    InvalidWarmStart,
    #[error("The evaluation count has overflowed. This is not realistically possible.")]
    EvaluationCountOverflow,
    #[error(
        "The population was empty. This is not possible and can only happen due to programmer error."
    )]
    EmptyPopulation,
}

impl<C> From<DifferentialEvolutionError> for OptimizationError<C, DifferentialEvolutionError> {
    fn from(error: DifferentialEvolutionError) -> Self {
        Self::Optimizer(error)
    }
}

const fn replace_nan(cost: f64) -> f64 {
    if cost.is_nan() { f64::INFINITY } else { cost }
}

impl DifferentialEvolution {
    fn converged(&self, costs: &[f64], population_size: f64) -> bool {
        let mean = costs.iter().sum::<f64>() / population_size;
        let variance = costs
            .iter()
            .map(|cost| {
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
        population: &[Vec<f64>],
    ) -> Result<Vec<f64>, OptimizationError<C::Error, DifferentialEvolutionError>>
    where
        C: CostFunction<Param = Vec<f64>, Output = f64> + Sync + basin::BoxConstraints,
        C::Error: Send,
    {
        population
            .par_iter()
            .map(|parameters| {
                let cost = replace_nan(
                    cost_func
                        .cost(
                            &parameters
                                .iter()
                                .zip(cost_func.lower().iter())
                                .zip(cost_func.upper().iter())
                                .map(|((val, lower), upper)| lower + val * (upper - lower))
                                .collect(),
                        )
                        .map_err(OptimizationError::CostFunction)?,
                );
                Ok(cost)
            })
            .collect()
    }

    fn polish<C>(
        cost_func: &C,
        best_member: &[f64],
        best_cost: f64,
        nfev: u64,
    ) -> Result<OptimizationResult, OptimizationError<C::Error, DifferentialEvolutionError>>
    where
        C: CostFunction<Param = Vec<f64>, Output = f64> + Clone + basin::BoxConstraints,
    {
        let best_member: Vec<_> = best_member
            .iter()
            .zip(cost_func.lower())
            .zip(cost_func.upper())
            .map(|((val, lower), upper)| lower + val * (upper - lower))
            .collect();
        let polished = basin::Executor::new(
            cost_func.clone(),
            basin::NelderMead::standard().projected(),
            basin::BasicSimplexState::new(best_member.clone()),
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
                params: best_member,
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
        seed: Option<u64>,
    ) -> Result<OptimizationResult, OptimizationError<C::Error, DifferentialEvolutionError>>
    where
        C: CostFunction<Param = Vec<f64>, Output = f64>
            + Clone
            + fmt::Debug
            + Sync
            + Send
            + basin::BoxConstraints,
        C::Error: Send + fmt::Display,
    {
        self.minimize_with_warm_start(cost_func, seed, None)
    }

    /// Append a warm start to the configured random population.
    ///
    /// # Errors
    /// Returns an error for invalid warm starts, settings, or objective failures.
    pub fn minimize_with_warm_start<C>(
        &self,
        cost_func: &C,
        seed: Option<u64>,
        warm_start: Option<&[f64]>,
    ) -> Result<OptimizationResult, OptimizationError<C::Error, DifferentialEvolutionError>>
    where
        C: CostFunction<Param = Vec<f64>, Output = f64>
            + Clone
            + fmt::Debug
            + Sync
            + Send
            + basin::BoxConstraints,
        C::Error: Send + fmt::Display,
    {
        // Check that bounds are valid: finite
        let lower_bounds = cost_func.lower();
        let upper_bounds = cost_func.upper();

        // Upper and lower bounds must be the same shape.
        if lower_bounds.is_empty() || lower_bounds.len() != upper_bounds.len() {
            return Err(DifferentialEvolutionError::InvalidBounds.into());
        }
        let num_parameters = lower_bounds.len();

        // Lower bound must be lower than the upper bound and both must be finite.
        for (lb, ub) in lower_bounds.iter().zip(upper_bounds.iter()) {
            if !lb.is_finite() || !ub.is_finite() {
                return Err(DifferentialEvolutionError::InvalidBounds.into());
            }
            if lb > ub || !(ub - lb).is_finite() {
                return Err(DifferentialEvolutionError::InvalidBounds.into());
            }
        }

        // Check warm start
        if let Some(point) = warm_start
            && (point.len() != num_parameters
                || !point
                    .iter()
                    .zip(lower_bounds.iter())
                    .zip(upper_bounds.iter())
                    .all(|((value, lb), ub)| value >= lb && value <= ub))
        {
            return Err(DifferentialEvolutionError::InvalidWarmStart.into());
        }

        let population_size = self
            .pop_size_factor
            .checked_mul(num_parameters)
            .ok_or(DifferentialEvolutionError::EvaluationCountOverflow)?
            .checked_add(usize::from(warm_start.is_some()))
            .ok_or(DifferentialEvolutionError::EvaluationCountOverflow)?;

        let evaluations_per_generation = u64::try_from(population_size)
            .map_err(|_error| DifferentialEvolutionError::EvaluationCountOverflow)?;

        let mut nfev = evaluations_per_generation;

        let mut rng =
            seed.map_or_else(|| StdRng::from_rng(&mut rand::rng()), StdRng::seed_from_u64);
        // Step 1: Create initial population
        let mut population = self.initializer.generate_initial_population(
            population_size - usize::from(warm_start.is_some()),
            num_parameters,
            &mut rng,
        );
        if let Some(point) = warm_start {
            population.push(
                point
                    .iter()
                    .zip(lower_bounds.iter())
                    .zip(upper_bounds.iter())
                    .map(|((val, lower), upper)| {
                        if lower == upper {
                            0.0
                        } else {
                            (val - lower) / (upper - lower)
                        }
                    })
                    .collect(),
            );
        }

        let mut costs = Self::evaluate_population(cost_func, &population)?;

        #[expect(
            clippy::cast_precision_loss,
            reason = "population length is used only for statistical averaging"
        )]
        let num_members_f64 = population_size as f64;

        // Step 2: Create donor vectors
        // v[i] = z[a] + F * (z[b] - z[c]) where a, b and c are not equal to i
        let weight = self.mutation_factor;
        let crossover = self.crossover_rate.into();

        // Iterate over generations
        let mut trials = Vec::with_capacity(population_size);
        for _ in 0..self.max_iter {
            // Add the expected number of function evaluations
            let next_nfev = nfev
                .checked_add(evaluations_per_generation)
                .ok_or(DifferentialEvolutionError::EvaluationCountOverflow)?;

            trials.clear();

            // Generate trials in order, without changing any parent this generation.
            for i in 0..population_size {
                let indices = sample(&mut rng, population_size - 1, 3);
                // Sampling three indices is safe after validating pop_size >= 4.
                let [a_index, b_index, c_index] = array::from_fn(|position| {
                    let index = indices.index(position);
                    if index >= i { index + 1 } else { index }
                });
                let donor: Vec<_> = population[a_index]
                    .iter()
                    .enumerate()
                    .map(|(param_index, value)| {
                        value
                            + weight
                                * (population[b_index][param_index]
                                    - population[c_index][param_index])
                    })
                    .collect();
                let forced = rng.random_range(0..num_parameters);
                let new_pop: Vec<_> = population[i]
                    .iter()
                    .enumerate()
                    .map(|(param_index, value)| {
                        if param_index == forced || rng.random::<f64>() < crossover {
                            // TODO: Handle boundary here
                            donor[param_index].clamp(0.0, 1.0)
                        } else {
                            *value
                        }
                    })
                    .collect();

                trials.push(new_pop);
            }

            // Evaluate the complete generation before committing any updates.
            let trial_costs = Self::evaluate_population(cost_func, &trials)?;
            for ((parent, parent_cost), (trial, trial_cost)) in population
                .iter_mut()
                .zip(costs.iter_mut())
                .zip(trials.iter().zip(trial_costs))
            {
                if trial_cost <= *parent_cost {
                    trial.clone_into(parent);
                    *parent_cost = trial_cost;
                }
            }
            let converged = self.converged(&costs, num_members_f64);
            nfev = next_nfev;
            if converged {
                break;
            }
        }

        // Step 4: Select best cost
        let (best_member, best_cost) = population
            .iter()
            .zip(costs)
            .min_by(|(_, left), (_, right)| left.total_cmp(right))
            .ok_or(DifferentialEvolutionError::EmptyPopulation)?;

        // Step 5: Polish best result
        Self::polish(cost_func, best_member, best_cost, nfev)
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
        let de = DifferentialEvolution::new(DifferentialEvolutionConfig {
            pop_size_factor: 90,
            max_iter: 100,
            atol: 0.0,
            rtol: 0.01,
            ..Default::default()
        })
        .expect("construction should always pass.");

        let prob = Rosenbrock {
            lb: vec![0.0, 0.0],
            ub: vec![100.0, 100.0],
        };
        let res = de.minimize(&prob, Some(879_123));
        let res = res.expect("test does not pass if this is an error.");

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
        let optimizer = DifferentialEvolution::new(DifferentialEvolutionConfig {
            pop_size_factor: 4,
            max_iter: 1,
            atol: 0.0,
            rtol: 0.0,
            ..Default::default()
        })
        .expect("construction should always pass.");
        let result = optimizer
            .minimize_with_warm_start(&objective, Some(883_331), Some(&[0.123]))
            .unwrap();
        let points = objective.evaluations.lock().unwrap().clone();
        assert!(points[..5].contains(&vec![0.123]));
        assert_eq!(result.nfev, u64::try_from(points.len()).unwrap());
        assert_eq!(result.params, vec![0.123]);
        assert!(points.len() >= 10);
    }

    #[test]
    fn physical_warm_start_with_fixed_bounds() {
        let objective = RecordingObjective {
            lower: vec![10.0, 3.0],
            upper: vec![20.0, 3.0],
            evaluations: Arc::default(),
        };
        let optimizer = DifferentialEvolution::new(DifferentialEvolutionConfig {
            max_iter: 1,
            atol: 0.0,
            rtol: 0.0,
            ..Default::default()
        })
        .unwrap();
        let result = optimizer
            .minimize_with_warm_start(&objective, Some(883_331), Some(&[15.0, 3.0]))
            .unwrap();
        let points = objective.evaluations.lock().unwrap();
        assert!(points[..9].contains(&vec![15.0, 3.0]));
        assert!(
            points
                .iter()
                .all(|point| { (10.0..=20.0).contains(&point[0]) && point[1] == 3.0 })
        );
        assert_eq!(result.nfev, u64::try_from(points.len()).unwrap());
        assert!((10.0..=20.0).contains(&result.params[0]));
        assert_eq!(result.params[1], 3.0);
        let expected_cost: f64 = result
            .params
            .iter()
            .map(|value| (value - 0.123).powi(2))
            .sum();
        assert_eq!(result.cost, expected_cost);
    }

    #[test]
    fn polishing_fallback_returns_physical_parameters() {
        let objective = RecordingObjective {
            lower: vec![10.0],
            upper: vec![20.0],
            evaluations: Arc::default(),
        };
        // Force the fallback independently of the local optimizer's convergence.
        let result = DifferentialEvolution::polish(&objective, &[0.5], -1.0, 0).unwrap();
        assert_eq!(result.params, vec![15.0]);
        assert_eq!(result.cost, -1.0);
    }

    #[test]
    fn invalid_bounds_fail_before_evaluation() {
        let optimizer = DifferentialEvolution::new(DifferentialEvolutionConfig::default()).unwrap();
        for (lower, upper) in [(vec![], vec![]), (vec![-f64::MAX], vec![f64::MAX])] {
            let objective = RecordingObjective {
                lower,
                upper,
                evaluations: Arc::default(),
            };
            assert!(matches!(
                optimizer.minimize(&objective, Some(1)),
                Err(OptimizationError::Optimizer(
                    DifferentialEvolutionError::InvalidBounds
                ))
            ));
            assert!(objective.evaluations.lock().unwrap().is_empty());
        }
    }

    #[test]
    fn invalid_warm_starts_fail_before_evaluation() {
        let objective = objective();
        let optimizer = DifferentialEvolution::new(DifferentialEvolutionConfig {
            pop_size_factor: 4,
            max_iter: 0,
            atol: 0.0,
            rtol: 0.0,
            ..Default::default()
        })
        .expect("construction should always pass.");
        for point in [
            vec![],
            vec![0.0, 0.0],
            vec![f64::NAN],
            vec![f64::INFINITY],
            vec![-0.1],
            vec![1.1],
        ] {
            assert!(matches!(
                optimizer.minimize_with_warm_start(&objective, Some(12333), Some(&point)),
                Err(OptimizationError::Optimizer(
                    DifferentialEvolutionError::InvalidWarmStart
                ))
            ));
        }
        assert!(objective.evaluations.lock().unwrap().is_empty());
    }
}
