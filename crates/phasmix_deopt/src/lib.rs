use argmin_testfunctions::rosenbrock;
use basin::CostFunction;
use core::{cmp, convert, fmt};
use phasmix_core::usize_to_f64;
use rand::{RngExt as _, distr::Uniform, seq::index::sample};
use rayon::prelude::*;

#[derive(Debug)]
pub struct OptimizationResult {
    pub params: Vec<f64>,
    pub cost: f64,
    pub nfev: u64,
}

pub struct DifferentialEvolution {
    pub pop_size: usize,
    pub max_iter: usize,
    pub atol: f64,
    pub rtol: f64,
}

impl DifferentialEvolution {
    /// Globally minimize the given cost function using differential evolution.
    ///
    /// # Panics
    /// Can panic. TODO: Avoid panicking.
    ///
    /// # Errors
    /// Can error. TODO: Need to implement proper error handling.
    ///
    pub fn minimize<C>(
        &self,
        cost_func: &C,
        bounds: &[(f64, f64)],
    ) -> Result<OptimizationResult, C::Error>
    where
        C: CostFunction<Param = Vec<f64>, Output = f64>
            + Clone
            + fmt::Debug
            + Sync
            + Send
            + basin::BoxConstraints,
        C::Error: Send + fmt::Display,
    {
        // Step 1: Create initial population
        let mut rng = rand::rng();
        let uniform_bounds = bounds
            .iter()
            .map(|(lb, ub)| Uniform::new_inclusive(lb, ub))
            .collect::<Result<Vec<_>, _>>()
            .expect("bounds are invalid.");
        let populations: Vec<_> = (0..self.pop_size)
            .map(|_| {
                let parameters = uniform_bounds
                    .iter()
                    .map(|current_bounds| rng.sample(current_bounds))
                    .collect::<Vec<_>>();
                let cost = cost_func.cost(&parameters)?;
                Ok((parameters, cost))
            })
            .collect::<Result<Vec<_>, _>>()?;
        println!("populations = {populations:?}");

        let mut next_population = populations.clone();

        // Step 2: Create donor vectors
        // v[i] = z[a] + F * (z[b] - z[c]) where a, b and c are not equal to i
        let weight = 0.7;
        let crossover = 0.7;
        let max_generations = 100;
        let mut num_generations = 0;
        for _ in 0..max_generations {
            for i in 0..populations.len() {
                let indices = sample(&mut rng, populations.len() - 1, 3);
                let [a_index, b_index, c_index] = indices
                    .into_iter()
                    .map(|j| if j >= i { j + 1 } else { j })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                let donor: Vec<_> = populations[a_index]
                    .0
                    .iter()
                    .enumerate()
                    .map(|(param_index, value)| {
                        value
                            + weight
                                * (populations[b_index].0[param_index]
                                    - populations[c_index].0[param_index])
                    })
                    .collect();
                let forced = rng.sample(Uniform::new(0, bounds.len()).unwrap());
                let new_pop: Vec<_> = populations[i]
                    .0
                    .iter()
                    .enumerate()
                    .map(|(param_index, value)| {
                        let keep_value: f64 = rng.random();
                        if param_index == forced || keep_value < crossover {
                            donor[param_index].clamp(bounds[param_index].0, bounds[param_index].1)
                        } else {
                            *value
                        }
                    })
                    .collect();

                // Step 3: Compare new vector's cost with old cost
                let new_cost = cost_func.cost(&new_pop).unwrap_or(f64::INFINITY);
                if new_cost <= populations[i].1 {
                    next_population[i] = (new_pop, new_cost);
                }
            }
            let mean_costs = next_population.iter().map(|(_, cost)| *cost).sum::<f64>()
                / (next_population.len() as f64);
            let variance = next_population
                .iter()
                .map(|(_, cost)| {
                    let residual = cost - mean_costs;
                    residual * residual
                })
                .sum::<f64>()
                / (next_population.len() as f64);
            let std_costs = variance.sqrt();
            let converged = std_costs <= self.rtol.mul_add(mean_costs.abs(), self.atol);
            println!(
                "convergence ({converged}): {std_costs} <= {} + {} * {}",
                self.atol,
                self.rtol,
                mean_costs.abs()
            );
            num_generations += 1;
            if converged {
                break;
            }
        }

        // Step 4: Select best cost
        println!("next population = {next_population:?}");
        let (best_index, (best_member, best_cost)) = next_population
            .iter()
            .enumerate()
            .min_by(|(_, (_, a_cost)), (_, (_, b_cost))| a_cost.partial_cmp(b_cost).unwrap())
            .unwrap();
        println!("best member = {best_member:?}");
        println!("best cost = {best_cost}");
        println!("best index = {best_index}");
        println!(
            "best neighbourhood = {:?}",
            &next_population[best_index - 1..best_index + 3]
        );

        let nfev: u64 = (self.pop_size * (num_generations + 1)).try_into().unwrap();

        // Step 5: Polish best result
        match basin::Executor::new(
            cost_func.clone(),
            basin::NelderMead::standard().projected(),
            basin::BasicSimplexState::new(best_member.to_owned()),
        )
        .max_iter(200)
        .run()
        {
            Ok(res) => {
                let best_param = res.best_param();
                Ok(OptimizationResult {
                    cost: res.best_cost(),
                    params: best_param.to_owned(),
                    nfev: nfev + res.cost_evals(),
                })
            }
            Err(err) => panic!("restart failed: {err}"),
        }
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

        let Ok(res) = res;

        println!("nfev = {}", res.nfev);
    }
}
