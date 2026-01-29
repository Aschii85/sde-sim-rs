pub mod euler;
pub mod runge_kutta;

use crate::filtration::Filtration;
use crate::rng::Rng;
use ordered_float::OrderedFloat;

pub fn simulate(filtration: &mut Filtration, rng: &mut dyn Rng, scheme: &str) {
    let num_scenarios = filtration.scenarios.len();
    let num_time_deltas = filtration.times.len() - 1;

    let mut rk_scratchpad = if scheme == "runge-kutta" {
        Some(Filtration::new(
            vec![OrderedFloat(0.0)],
            vec![0],
            filtration.processes.to_vec(),
            None,
        ))
    } else {
        None
    };
    for scenario_idx in 0..num_scenarios {
        for time_idx in 0..num_time_deltas {
            match scheme {
                "euler" => {
                    euler::euler_iteration(filtration, scenario_idx, time_idx, rng);
                }
                "runge-kutta" => {
                    runge_kutta::runge_kutta_iteration(
                        filtration,
                        scenario_idx,
                        time_idx,
                        rng,
                        rk_scratchpad.as_mut().unwrap(),
                    );
                }
                _ => panic!("Unknown scheme"),
            }
        }
    }
}
