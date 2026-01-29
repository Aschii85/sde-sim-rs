pub mod euler;
pub mod runge_kutta;

use crate::filtration::Filtration;
use crate::proc::LevyProcess;
use crate::rng::sobol::SobolEngine;
use crate::rng::{BaseRng, pseudo::PseudoRng, sobol::SobolRng};
use rand::Rng;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

// TODO add seed?
pub fn simulate(filtration: &mut Filtration, scheme: &str, rng_method: &str) {
    let mut rng = rand::rng();
    let random_seed: u64 = rng.random();

    let processes = &filtration.processes_universe.processes.clone();
    let num_increments = filtration.processes_universe.num_stochastic_increments;
    let times = &filtration.times.clone();
    let num_time_deltas = times.len() - 1;

    // 1. Calculate total dimensions needed for one path
    let dims = (times.len() - 1) * num_increments;

    // 2. Create the shared engine
    let shared_engine = match rng_method {
        "sobol" => Some(Arc::new(Mutex::new(SobolEngine::new(dims)))),
        _ => None,
    };

    filtration
        .scenario_partitions()
        .enumerate()
        .collect::<Vec<_>>() // Optional: helps Rayon with indexing performance
        .into_par_iter() // Converts std iterator to Rayon's parallel iterator
        .for_each(|(s_idx, scenario_slice)| {
            let mut local_processes: Vec<Box<LevyProcess>> = processes
                .iter()
                .map(|p| p.clone()) // Ensure LevyProcess implements Clone
                .collect();
            let mut local_rng: Box<dyn BaseRng> = match rng_method {
                "sobol" => Box::new(SobolRng::new(
                    s_idx as u64 + random_seed,
                    Arc::clone(
                        shared_engine
                            .as_ref()
                            .expect("Sobol engine not initialized"),
                    ),
                    num_increments,
                    times.len(),
                )),
                _ => Box::new(PseudoRng::new(s_idx as u64 + random_seed, num_increments)),
            };
            for t_idx in 0..num_time_deltas {
                match scheme {
                    "euler" => euler::euler_iteration(
                        scenario_slice,
                        &mut local_processes,
                        times,
                        t_idx,
                        local_rng.as_mut(),
                    ),
                    "runge-kutta" => runge_kutta::runge_kutta_iteration(
                        scenario_slice,
                        &mut local_processes,
                        times,
                        t_idx,
                        local_rng.as_mut(),
                    ),
                    _ => unimplemented!(),
                }
            }
        });
}
