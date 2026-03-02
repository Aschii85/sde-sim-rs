pub mod euler;
pub mod runge_kutta;

use crate::filtration::ScenarioFiltration;
use crate::proc::ProcessUniverse;
use crate::rng::sobol::SobolEngine;
use crate::rng::{BaseRng, pseudo::PseudoRng, sobol::SobolRng};
use rand::Rng;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use ordered_float::OrderedFloat;
use std::collections::HashMap;
use polars::prelude::IntoLazy;

/// Run a batch of simulation paths in parallel and return a concatenated DataFrame.
///
/// Each scenario is executed independently on its own `ScenarioFiltration`.  The
/// numerical scheme and RNG algorithm are chosen by the `scheme`/`rng_method`
/// parameters.  When `rng_method` is "sobol" a shared Sobol engine is used to
/// avoid rebuilding the sequence for every path.

pub fn simulate(
    process_universe: &ProcessUniverse,
    timesteps: Vec<OrderedFloat<f64>>,
    initial_values: HashMap<String, f64>,
    num_scenarios: u64,
    scheme: &str,
    rng_method: &str,
) -> polars::prelude::LazyFrame {
    let mut rng = rand::rng();
    let random_seed: u64 = rng.random();
    let times = timesteps;
    let num_time_deltas = times.len() - 1;
    let sobol_increments = process_universe.stochastic_registry.len();
    let sobol_dims = num_time_deltas * sobol_increments;

    // shared Sobol engine (only used when rng_method == "sobol")
    let shared_engine = match rng_method {
        "sobol" => Some(Arc::new(Mutex::new(SobolEngine::new(sobol_dims)))),
        _ => None,
    };

    let dfs: Vec<polars::prelude::LazyFrame> = (0..num_scenarios)
        .into_par_iter()
        .map(|s_idx| {
            // build a fresh filtration for this scenario
            let local_process_universe = process_universe.clone();
            let mut filtration = ScenarioFiltration::new(
                s_idx as i32,
                local_process_universe.clone(),
                times.clone(),
                initial_values.clone(),

            );

            // every scenario gets its own RNG instance
            let mut local_rng: Box<dyn BaseRng> = match rng_method {
                "sobol" => Box::new(SobolRng::new(
                    s_idx as u64 + random_seed,
                    Arc::clone(
                        shared_engine
                            .as_ref()
                            .expect("Sobol engine not initialized"),
                    ),
                    sobol_increments,
                    times.len(),
                )),
                _ => Box::new(PseudoRng::new(s_idx as u64 + random_seed, sobol_increments)),
            };

            for t_idx in 0..num_time_deltas {
                match scheme {
                    "euler" => euler::euler_iteration(
                        &mut filtration,
                        &local_process_universe,
                        t_idx,
                        local_rng.as_mut(),
                    ),
                    "runge-kutta" => runge_kutta::runge_kutta_iteration(
                        &mut filtration,
                        &local_process_universe,
                        t_idx,
                        local_rng.as_mut(),
                    ),
                    _ => unimplemented!(),
                }
            }

            filtration.to_dataframe().lazy()
        })
        .collect();

    // stack all of the individual scenario frames together
    polars::prelude::concat(&dfs, polars::prelude::UnionArgs::default()).expect("failed to concatenate results")
}
