extern crate alloc;

use crate::core::{OptimizationError, OptimizationResult};
use basin::CostFunction;
use core::array;
use core::cmp;
#[cfg(test)]
use core::convert;
use core::default;
use core::fmt;
use rand::{Rng, RngExt as _, SeedableRng as _, rngs::StdRng, seq::index::sample};
use rayon::prelude::*;
use thiserror::Error;

// Best2 excludes target and best before sampling four distinct members;
// Rand2 excludes the target before sampling five distinct members.
const MIN_POPULATION_SIZE: usize = 6;

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default)]
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

impl MutationStrategy {
    fn generate_mutant(
        self,
        population: &[Vec<f64>],
        mutation_factor: f64,
        target_index: usize,
        best_index: usize,
        rng: &mut impl Rng,
    ) -> Vec<f64> {
        assert!(
            (0..population.len()).contains(&target_index),
            "target index must be within bounds."
        );
        assert!(
            (0..population.len()).contains(&best_index),
            "best index must be within bounds."
        );

        let exclude_target = &[target_index];
        let exclude_target_and_best = &match target_index.cmp(&best_index) {
            cmp::Ordering::Less => [target_index, best_index],
            cmp::Ordering::Equal => [target_index, population.len()],
            cmp::Ordering::Greater => [best_index, target_index],
        };

        match self {
            MutationStrategy::Best1 => {
                let [r1, r2] = Self::sample_indices(population.len(), exclude_target_and_best, rng);
                itertools::izip!(
                    population[best_index].iter(),
                    population[r1].iter(),
                    population[r2].iter()
                )
                .map(|(best_val, r1_val, r2_val)| best_val + mutation_factor * (r1_val - r2_val))
                .collect()
            }

            MutationStrategy::Best2 => {
                let [r1, r2, r3, r4] =
                    Self::sample_indices(population.len(), exclude_target_and_best, rng);
                itertools::izip!(
                    population[best_index].iter(),
                    population[r1].iter(),
                    population[r2].iter(),
                    population[r3].iter(),
                    population[r4].iter()
                )
                .map(|(best_val, r1_val, r2_val, r3_val, r4_val)| {
                    best_val + mutation_factor * (r1_val + r2_val - r3_val - r4_val)
                })
                .collect()
            }
            MutationStrategy::Rand1 => {
                let [r0, r1, r2] = Self::sample_indices(population.len(), exclude_target, rng);
                itertools::izip!(
                    population[r0].iter(),
                    population[r1].iter(),
                    population[r2].iter()
                )
                .map(|(r0_val, r1_val, r2_val)| r0_val + mutation_factor * (r1_val - r2_val))
                .collect()
            }
            MutationStrategy::Rand2 => {
                let [r0, r1, r2, r3, r4] =
                    Self::sample_indices(population.len(), exclude_target, rng);
                itertools::izip!(
                    population[r0].iter(),
                    population[r1].iter(),
                    population[r2].iter(),
                    population[r3].iter(),
                    population[r4].iter()
                )
                .map(|(r0_val, r1_val, r2_val, r3_val, r4_val)| {
                    r0_val + mutation_factor * (r1_val + r2_val - r3_val - r4_val)
                })
                .collect()
            }
            MutationStrategy::RandToBest => {
                let [r0, r1, r2] =
                    Self::sample_indices(population.len(), exclude_target_and_best, rng);
                itertools::izip!(
                    population[r0].iter(),
                    population[r1].iter(),
                    population[r2].iter(),
                    population[best_index].iter()
                )
                .map(|(r0_val, r1_val, r2_val, best_val)| {
                    r0_val + mutation_factor * (best_val - r0_val + r1_val - r2_val)
                })
                .collect()
            }
            MutationStrategy::CurrentToBest => {
                let [r1, r2] = Self::sample_indices(population.len(), exclude_target_and_best, rng);
                itertools::izip!(
                    population[target_index].iter(),
                    population[r1].iter(),
                    population[r2].iter(),
                    population[best_index].iter()
                )
                .map(|(target_val, r1_val, r2_val, best_val)| {
                    target_val + mutation_factor * (best_val - target_val + r1_val - r2_val)
                })
                .collect()
            }
        }
    }

    fn sample_indices<const N: usize>(
        num_population: usize,
        excluding: &[usize],
        rng: &mut impl Rng,
    ) -> [usize; N] {
        debug_assert!(excluding.is_sorted(), "`excluding` must be sorted.");
        debug_assert!(
            excluding.windows(2).all(|w| w[0] != w[1]),
            "`excluding` must contain no duplicates."
        );
        let num_excluded = excluding
            .iter()
            .take_while(|&&idx| idx < num_population)
            .count();

        let mut sampled = sample(rng, num_population - num_excluded, N).into_iter();

        array::from_fn(|_| {
            let mut idx = sampled.next().unwrap();

            for &excluded_idx in excluding {
                if idx >= excluded_idx {
                    idx += 1;
                }
            }
            idx
        })
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default)]
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

impl CrossoverStrategy {
    fn crossover(
        self,
        mutant: &mut [f64],
        target: &[f64],
        crossover_rate: f64,
        rng: &mut impl Rng,
    ) {
        assert_eq!(
            mutant.len(),
            target.len(),
            "`mutant` and `target` must be the same length."
        );
        let num_parameters = mutant.len();
        match self {
            CrossoverStrategy::Binomial => {
                let forced = rng.random_range(0..num_parameters);
                for (param_idx, (mutant_val, target_val)) in
                    mutant.iter_mut().zip(target.iter()).enumerate()
                {
                    if param_idx == forced || rng.random::<f64>() < crossover_rate {
                    } else {
                        *mutant_val = *target_val;
                    }
                }
            }
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default)]
pub enum BoundaryStrategy {
    /// If the trial vector is out of bounds, each out of bounds parameter is replaced by a newly sampled in-bounds parameter.
    /// Comparison with the target vector then proceeds as normal.
    Resample,
    /// Reflect out-of-bounds parameters across the bounds, including repeated
    /// overshoots. Non-finite parameters are resampled because reflection is undefined.
    #[default]
    Reflect,
}

impl BoundaryStrategy {
    fn bound(
        self,
        trial: &mut [f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        rng: &mut impl Rng,
    ) {
        assert_eq!(
            lower_bounds.len(),
            upper_bounds.len(),
            "`lower_bounds` and `upper_bounds` must be the same length."
        );
        assert_eq!(
            trial.len(),
            upper_bounds.len(),
            "`trial` and `upper_bounds` must be the same length."
        );
        match self {
            BoundaryStrategy::Resample => {
                for (val, lb, ub) in itertools::izip!(
                    trial.iter_mut(),
                    lower_bounds.iter().copied(),
                    upper_bounds.iter().copied()
                ) {
                    debug_assert!(lb <= ub, "lower bound must be lower than upper bound");
                    if !(lb..=ub).contains(val) {
                        *val = rng.random_range(lb..=ub);
                    }
                }
            }
            BoundaryStrategy::Reflect => {
                for (val, lb, ub) in itertools::izip!(
                    trial.iter_mut(),
                    lower_bounds.iter().copied(),
                    upper_bounds.iter().copied()
                ) {
                    if (lb..=ub).contains(val) {
                        continue;
                    }
                    let width = ub - lb;
                    if width == 0.0 {
                        *val = lb;
                    } else if !val.is_finite() {
                        *val = rng.random_range(lb..=ub);
                    } else {
                        // Fold a period of twice the width into the bounded interval.
                        // Reduce operands separately to avoid overflowing val - lb.
                        let period = 2.0 * width;
                        let reflected = if period.is_finite() {
                            let offset =
                                (val.rem_euclid(period) - lb.rem_euclid(period)).rem_euclid(period);
                            if offset <= width {
                                lb + offset
                            } else {
                                ub - (offset - width)
                            }
                        } else {
                            // Halve coordinates when twice the width would overflow.
                            let offset = ((*val * 0.5).rem_euclid(width)
                                - (lb * 0.5).rem_euclid(width))
                            .rem_euclid(width);
                            if offset <= width * 0.5 {
                                2.0f64.mul_add(offset, lb)
                            } else {
                                2.0f64.mul_add(-width.mul_add(-0.5, offset), ub)
                            }
                        };
                        *val = reflected.clamp(lb, ub);
                    }
                }
            }
        }
    }
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
    /// Must be at least six to support every mutation strategy in one dimension.
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
            pop_size_factor: MIN_POPULATION_SIZE,
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
    /// - `pop_size_factor` is too small; less than 6.
    /// - `atol` or `rtol` are invalid tolerances; non-finite or negative.
    /// - `crossover_rate` is invalid; should be a probability in the range [0.0, 1.0].
    /// - `mutation_factor` is invalid; non-finite or negative.
    pub fn new(
        config: DifferentialEvolutionConfig,
    ) -> Result<DifferentialEvolution, DifferentialEvolutionInitializationError> {
        if config.pop_size_factor < MIN_POPULATION_SIZE {
            return Err(
                DifferentialEvolutionInitializationError::PopulationTooSmall {
                    size: config.pop_size_factor,
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
    /// The population factor must be at least six to support all mutation strategies.
    #[error("Differential evolution requires a population factor of at least six, but got {size}.")]
    PopulationTooSmall { size: usize },
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
                        .cost(parameters)
                        .map_err(OptimizationError::CostFunction)?,
                );
                Ok(cost)
            })
            .collect()
    }

    fn polish<C>(
        cost_func: &C,
        best_member: Vec<f64>,
        best_cost: f64,
        nfev: u64,
    ) -> Result<OptimizationResult, OptimizationError<C::Error, DifferentialEvolutionError>>
    where
        C: CostFunction<Param = Vec<f64>, Output = f64> + Clone + basin::BoxConstraints,
    {
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
        let mut population: Vec<Vec<f64>> = self
            .initializer
            .generate_initial_population(
                population_size - usize::from(warm_start.is_some()),
                num_parameters,
                &mut rng,
            )
            .into_iter()
            .map(|params| {
                params
                    .into_iter()
                    .zip(lower_bounds.iter())
                    .zip(upper_bounds.iter())
                    .map(|((val, lower), upper)| lower + val * (upper - lower))
                    .collect()
            })
            .collect();
        if let Some(point) = warm_start {
            population.push(point.to_vec());
        }

        let mut costs = Self::evaluate_population(cost_func, &population)?;

        #[expect(
            clippy::cast_precision_loss,
            reason = "population length is used only for statistical averaging"
        )]
        let num_members_f64 = population_size as f64;

        let mut best_index = costs
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .ok_or(DifferentialEvolutionError::EmptyPopulation)?;

        // Iterate over generations
        let mut trials = Vec::with_capacity(population_size);
        for _ in 0..self.max_iter {
            // Add the expected number of function evaluations
            let next_nfev = nfev
                .checked_add(evaluations_per_generation)
                .ok_or(DifferentialEvolutionError::EvaluationCountOverflow)?;

            trials.clear();

            // Generate trials in order, without changing any parent this generation.
            for target_index in 0..population_size {
                // Step 2: Create mutant vectors using mutation strategy
                let mut mutant = self.mutation.generate_mutant(
                    &population,
                    self.mutation_factor,
                    target_index,
                    best_index,
                    &mut rng,
                );
                // Step 3: Crossover mutant with target
                self.crossover.crossover(
                    &mut mutant,
                    &population[target_index],
                    f64::from(self.crossover_rate),
                    &mut rng,
                );
                // Step 3.5: Handle out of bounds values
                self.boundary
                    .bound(&mut mutant, lower_bounds, upper_bounds, &mut rng);

                trials.push(mutant);
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

            // Update best index
            best_index = costs
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index)
                .ok_or(DifferentialEvolutionError::EmptyPopulation)?;
        }

        // Step 4: Select best cost
        let (best_member, best_cost) = population
            .into_iter()
            .zip(costs)
            .min_by(|(_, left), (_, right)| left.total_cmp(right))
            .ok_or(DifferentialEvolutionError::EmptyPopulation)?;

        // Step 5: Polish best result
        Self::polish(cost_func, best_member, best_cost, nfev)
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::*;

    #[test]
    fn reflection_handles_both_bounds_and_multiple_overshoots() {
        let mut trial = vec![9.0, 21.0, -15.0, 45.0, 10.0, 20.0, 15.0];
        let mut rng = StdRng::seed_from_u64(1);
        BoundaryStrategy::Reflect.bound(&mut trial, &[10.0; 7], &[20.0; 7], &mut rng);
        assert_eq!(
            trial,
            vec![11.0, 19.0, 15.0, 15.0, 10.0, 20.0, 15.0],
            "reflection must fold repeated overshoots and preserve in-bounds values"
        );
    }

    #[test]
    fn reflection_handles_fixed_bounds_and_nonfinite_values() {
        let mut trial = vec![100.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        let mut rng = StdRng::seed_from_u64(1);
        BoundaryStrategy::Reflect.bound(
            &mut trial,
            &[3.0, 10.0, 10.0, 10.0],
            &[3.0, 20.0, 20.0, 20.0],
            &mut rng,
        );
        assert_eq!(trial[0], 3.0, "fixed bounds must be preserved");
        assert!(
            trial[1..].iter().all(|val| (10.0..=20.0).contains(val)),
            "non-finite values must be repaired"
        );
    }

    #[test]
    fn reflection_avoids_overflow_for_large_finite_values() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut trial = vec![f64::MAX, -f64::MAX];
        BoundaryStrategy::Reflect.bound(&mut trial, &[-1.0e308; 2], &[0.0; 2], &mut rng);
        assert!(
            trial.iter().all(|val| (-1.0e308..=0.0).contains(val)),
            "reflection must work when twice the width overflows"
        );
        let mut trial = vec![f64::MAX];
        BoundaryStrategy::Reflect.bound(&mut trial, &[-1.0e308], &[-9.0e307], &mut rng);
        assert!(
            (-1.0e308..=-9.0e307).contains(&trial[0]),
            "reflection must work when subtracting the lower bound would overflow"
        );
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
    fn shared_population_minimum_supports_every_strategy() {
        for mutation in [
            MutationStrategy::Best1,
            MutationStrategy::Best2,
            MutationStrategy::Rand1,
            MutationStrategy::Rand2,
            MutationStrategy::RandToBest,
            MutationStrategy::CurrentToBest,
        ] {
            for pop_size_factor in 0..MIN_POPULATION_SIZE {
                assert!(
                    matches!(
                        DifferentialEvolution::new(DifferentialEvolutionConfig {
                            pop_size_factor,
                            mutation,
                            ..Default::default()
                        }),
                        Err(DifferentialEvolutionInitializationError::PopulationTooSmall { size })
                            if size == pop_size_factor
                    ),
                    "every strategy must reject factors below the shared minimum"
                );
            }
            let objective = objective();
            let optimizer = DifferentialEvolution::new(DifferentialEvolutionConfig {
                mutation,
                max_iter: 2,
                atol: 0.0,
                rtol: 0.0,
                ..Default::default()
            })
            .unwrap();
            let result = optimizer.minimize(&objective, Some(883_331)).unwrap();
            let points = objective.evaluations.lock().unwrap();
            assert_eq!(
                result.nfev,
                u64::try_from(points.len()).unwrap(),
                "evaluation accounting must hold at the shared minimum"
            );
        }
    }

    #[test]
    fn warm_start_adds_population_member_and_evaluation() {
        let objective = objective();
        let optimizer = DifferentialEvolution::new(DifferentialEvolutionConfig {
            pop_size_factor: MIN_POPULATION_SIZE,
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
        assert!(points[..7].contains(&vec![0.123]));
        assert_eq!(result.nfev, u64::try_from(points.len()).unwrap());
        assert_eq!(result.params, vec![0.123]);
        assert!(points.len() >= 14);
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
        assert!(points[..13].contains(&vec![15.0, 3.0]));
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
        let result = DifferentialEvolution::polish(&objective, vec![15.0], -1.0, 0).unwrap();
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
            pop_size_factor: MIN_POPULATION_SIZE,
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
