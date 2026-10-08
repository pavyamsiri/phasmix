//! Python-facing optimizer configuration inputs.
//!
//! These are extracted directly from the attributes of the Python dataclasses
//! in `phasmix.optimizers`; they are not additional Python extension classes.
//! Strategy conversion and native construction happen at the binding boundary.
//! Exponential crossover is rejected until implemented by the backend.

use phasmix_fit::GlobalOptimizer;
use phasmix_opt::differential_evolution::{
    BoundaryStrategy, CrossoverStrategy, DifferentialEvolution,
    DifferentialEvolutionConfig as NativeDEConfig, MutationStrategy,
};
use phasmix_opt::initialization::InitializationStrategy;
use phasmix_opt::tiktak::TikTak;
use pyo3::{
    FromPyObject,
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};

/// Input fields of `phasmix.optimizers.DifferentialEvolutionConfig`.
#[derive(Debug, Clone, FromPyObject)]
pub(crate) struct DifferentialEvolutionConfig {
    pub pop_size_factor: usize,
    pub max_iter: usize,
    pub max_local_iter: usize,
    pub mutation_factor: f64,
    pub crossover_rate: f32,
    pub atol: f64,
    pub rtol: f64,
    pub mutation: String,
    pub crossover: String,
    pub boundary: String,
    pub initializer: String,
}

/// Input fields of `phasmix.optimizers.TikTakConfig`.
#[derive(Debug, Clone, FromPyObject)]
pub(crate) struct TikTakConfig {
    pub log_num_samples: u8,
    pub keep_ratio: f32,
    pub min_weight: f64,
    pub max_weight: f64,
}

/// Input fields of `phasmix.optimizers.NelderMeadConfig`.
#[derive(Debug, Clone, FromPyObject)]
pub(crate) struct NelderMeadConfig {
    pub max_iter: usize,
}

/// Build a native optimizer from a concrete Python config dataclass.
pub(crate) fn build_optimizer<const N: usize>(
    input: &Bound<'_, PyAny>,
) -> PyResult<GlobalOptimizer<N>> {
    let configs = input.py().import("phasmix.optimizers")?;
    if input.is_instance(&configs.getattr("DifferentialEvolutionConfig")?)? {
        let config = input.extract::<DifferentialEvolutionConfig>()?;
        let mutation = match config.mutation.as_str() {
            "best1" => MutationStrategy::Best1,
            "best2" => MutationStrategy::Best2,
            "rand1" => MutationStrategy::Rand1,
            "rand2" => MutationStrategy::Rand2,
            "randtobest1" => MutationStrategy::RandToBest,
            "currenttobest1" => MutationStrategy::CurrentToBest,
            _ => return Err(PyValueError::new_err("Unsupported mutation strategy")),
        };
        let crossover = match config.crossover.as_str() {
            "binomial" => CrossoverStrategy::Binomial,
            "exponential" => {
                return Err(PyValueError::new_err(
                    "Exponential crossover is not implemented by the Rust backend",
                ));
            }
            _ => return Err(PyValueError::new_err("Unsupported crossover strategy")),
        };
        let boundary = match config.boundary.as_str() {
            "reflect" => BoundaryStrategy::Reflect,
            "resample" => BoundaryStrategy::Resample,
            _ => return Err(PyValueError::new_err("Unsupported boundary strategy")),
        };
        let initializer = match config.initializer.as_str() {
            "latinhypercube" => InitializationStrategy::LatinHyperCube,
            "sobol" => InitializationStrategy::Sobol,
            "random" => InitializationStrategy::Independent,
            _ => return Err(PyValueError::new_err("Unsupported initializer strategy")),
        };
        let optimizer = DifferentialEvolution::new(NativeDEConfig {
            pop_size_factor: config.pop_size_factor,
            max_iter: config.max_iter,
            max_local_iter: config.max_local_iter,
            mutation_factor: config.mutation_factor,
            crossover_rate: config.crossover_rate,
            atol: config.atol,
            rtol: config.rtol,
            mutation,
            crossover,
            boundary,
            initializer,
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        return Ok(GlobalOptimizer::DifferentialEvolution(optimizer));
    }
    if input.is_instance(&configs.getattr("TikTakConfig")?)? {
        let config = input.extract::<TikTakConfig>()?;
        // Guard the native constructor's assertions at the Python boundary.
        if !(1..16).contains(&config.log_num_samples)
            || !config.keep_ratio.is_finite()
            || !(0.0 < config.keep_ratio && config.keep_ratio <= 1.0)
            || !config.min_weight.is_finite()
            || !config.max_weight.is_finite()
            || !(0.0 <= config.min_weight
                && config.min_weight < config.max_weight
                && config.max_weight <= 1.0)
        {
            return Err(PyValueError::new_err(
                "Invalid TikTak sampling or interpolation configuration",
            ));
        }
        return Ok(GlobalOptimizer::TikTak(TikTak::new(
            config.log_num_samples,
            config.keep_ratio,
            config.min_weight,
            config.max_weight,
        )));
    }
    if input.is_instance(&configs.getattr("NelderMeadConfig")?)? {
        let config = input.extract::<NelderMeadConfig>()?;
        if config.max_iter == 0 {
            return Err(PyValueError::new_err(
                "Nelder-Mead max_iter must be positive",
            ));
        }
        return Ok(GlobalOptimizer::NelderMead {
            max_iter: config.max_iter,
        });
    }
    Err(PyTypeError::new_err(
        "optimizer must be a phasmix.optimizers configuration",
    ))
}
