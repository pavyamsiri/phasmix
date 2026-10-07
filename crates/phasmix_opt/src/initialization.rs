use rand::{Rng, RngExt as _, seq::SliceRandom as _};
use thiserror::Error;

/// Errors generating an initial population.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum InitializationError {
    #[error("Rounded population size overflowed.")]
    PopulationSizeOverflow,
    #[error("Sobol supports at most 2^16 points.")]
    SobolSampleLimit,
    #[error("Sobol supports at most 256 dimensions.")]
    SobolDimensionLimit,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default)]
pub enum InitializationStrategy {
    /// Sample once per equally sized stratum in each parameter, independently
    /// shuffling the strata across population members for each parameter.
    #[default]
    LatinHyperCube,
    /// Scrambled Sobol points, rounding the count up to the next power of two.
    Sobol,
    /// Independently sample every coordinate uniformly in the unit interval.
    Independent,
}

impl InitializationStrategy {
    /// Generate points in the unit hypercube. Sobol rounds the requested count up
    /// to a power of two and uses a fixed scramble seed of zero.
    ///
    /// # Errors
    /// Returns an error if the rounded count overflows or Sobol exceeds its
    /// supported limits of 2^16 points and 256 dimensions.
    pub fn generate_initial_population(
        self,
        pop_size: usize,
        num_parameters: usize,
        rng: &mut impl Rng,
    ) -> Result<Vec<Vec<f64>>, InitializationError> {
        Ok(match self {
            InitializationStrategy::LatinHyperCube => {
                let mut population = vec![vec![0.0; num_parameters]; pop_size];
                if pop_size == 0 {
                    return Ok(population);
                }
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "population size is used to divide the unit interval into strata"
                )]
                let stratum_width = 1.0 / pop_size as f64;
                let mut strata: Vec<_> = (0..pop_size).collect();
                for parameter in 0..num_parameters {
                    strata.shuffle(rng);
                    for (member, &stratum) in population.iter_mut().zip(&strata) {
                        #[expect(
                            clippy::cast_precision_loss,
                            reason = "stratum index is used only to locate a sample within the unit interval"
                        )]
                        let sample = (stratum as f64 + rng.random::<f64>()) * stratum_width;
                        member[parameter] = sample;
                    }
                }
                population
            }
            InitializationStrategy::Sobol => {
                if pop_size == 0 {
                    return Ok(Vec::new());
                }
                let count = pop_size
                    .checked_next_power_of_two()
                    .ok_or(InitializationError::PopulationSizeOverflow)?;
                if count > (1 << 16) {
                    return Err(InitializationError::SobolSampleLimit);
                }
                if num_parameters > sobol_burley::NUM_DIMENSIONS as usize {
                    return Err(InitializationError::SobolDimensionLimit);
                }
                (0..count)
                    .map(|index| {
                        let mut point = Vec::with_capacity(num_parameters);
                        for dimension_set in 0..num_parameters.div_ceil(4) {
                            point.extend(
                                sobol_burley::sample_4d(index as u32, dimension_set as u32, 0)
                                    .into_iter()
                                    .map(f64::from),
                            );
                        }
                        point.truncate(num_parameters);
                        point
                    })
                    .collect()
            }
            InitializationStrategy::Independent => (0..pop_size)
                .map(|_| {
                    (0..num_parameters)
                        .map(|_| rng.random_range(0.0..=1.0))
                        .collect()
                })
                .collect(),
        })
    }
}

#[cfg(test)]
mod initialization_tests {
    use super::*;
    use rand::{SeedableRng as _, rngs::StdRng};

    #[test]
    fn sobol_rounds_counts_and_matches_backend() {
        let mut rng = StdRng::seed_from_u64(1);
        for (requested, expected) in [(0, 0), (1, 1), (6, 8), (8, 8), (9, 16)] {
            let points = InitializationStrategy::Sobol
                .generate_initial_population(requested, 5, &mut rng)
                .unwrap();
            assert_eq!(
                points.len(),
                expected,
                "Sobol must round up only when needed"
            );
            for (index, point) in points.iter().enumerate() {
                assert_eq!(
                    point.len(),
                    5,
                    "partial dimension batches must be truncated"
                );
                for (dimension, &value) in point.iter().enumerate() {
                    assert_eq!(
                        value,
                        f64::from(sobol_burley::sample(index as u32, dimension as u32, 0)),
                        "shared sampling must preserve the backend sequence"
                    );
                }
            }
        }
    }

    #[test]
    fn sobol_rejects_unsupported_sizes() {
        let mut rng = StdRng::seed_from_u64(1);
        for (count, dimensions, expected) in [
            (usize::MAX, 1, InitializationError::PopulationSizeOverflow),
            ((1 << 16) + 1, 1, InitializationError::SobolSampleLimit),
            (8, 257, InitializationError::SobolDimensionLimit),
        ] {
            assert_eq!(
                InitializationStrategy::Sobol
                    .generate_initial_population(count, dimensions, &mut rng),
                Err(expected),
                "unsupported Sobol requests must fail before sampling"
            );
        }
    }

    #[test]
    fn latin_hypercube_covers_each_stratum_in_every_parameter() {
        let mut rng = StdRng::seed_from_u64(883_331);
        let population = InitializationStrategy::LatinHyperCube
            .generate_initial_population(16, 4, &mut rng)
            .unwrap();
        assert_eq!(population.len(), 16, "population size must be preserved");
        assert!(
            population.iter().all(|point| point.len() == 4),
            "every point must have the requested dimension"
        );
        let mut assignments = Vec::new();
        for parameter in 0..4 {
            let mut counts = [0; 16];
            let strata: Vec<_> = population
                .iter()
                .map(|point| {
                    let value = point[parameter];
                    assert!(
                        (0.0..1.0).contains(&value),
                        "samples must lie in the unit interval"
                    );
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "sample is in [0, 1), so its stratum index is in 0..16"
                    )]
                    let stratum = (value * 16.0) as usize;
                    counts[stratum] += 1;
                    stratum
                })
                .collect();
            assert_eq!(
                counts, [1; 16],
                "each parameter must sample every stratum once"
            );
            assignments.push(strata);
        }
        assert!(
            assignments.windows(2).all(|pair| pair[0] != pair[1]),
            "parameters must shuffle strata independently for this seed"
        );
        let repeated = InitializationStrategy::LatinHyperCube
            .generate_initial_population(16, 4, &mut StdRng::seed_from_u64(883_331))
            .unwrap();
        assert_eq!(
            population, repeated,
            "seeded initialization must be reproducible"
        );
    }

    #[test]
    fn latin_hypercube_supports_empty_and_single_member_populations() {
        let mut rng = StdRng::seed_from_u64(1);
        assert!(
            InitializationStrategy::LatinHyperCube
                .generate_initial_population(0, 2, &mut rng)
                .unwrap()
                .is_empty(),
            "empty population must stay empty"
        );
        let population = InitializationStrategy::LatinHyperCube
            .generate_initial_population(1, 2, &mut rng)
            .unwrap();
        assert_eq!(
            population.len(),
            1,
            "single-member population must be preserved"
        );
        assert!(
            population[0].iter().all(|value| (0.0..1.0).contains(value)),
            "single-member samples must lie in the unit interval"
        );
    }
}
