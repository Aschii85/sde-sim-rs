use crate::proc::{Process, ProcessUniverse};
use crate::rng::BaseRng;
use crate::filtration::ScenarioFiltration;

pub fn runge_kutta_iteration(
    filtration: &mut ScenarioFiltration,
    process_universe: &mut ProcessUniverse,
    t_idx: usize,
    rng: &mut dyn BaseRng,
) {
    let num_processes = process_universe.processes.len();
    let current_time = filtration.times[t_idx];
    let dt = (filtration.times[t_idx + 1] - filtration.times[t_idx]).into_inner();
    let sqrt_dt = dt.sqrt();

    // Derive the random variable sk (±1) for the stochastic RK scheme.
    let sk = if rng.sample(t_idx, 0) > 0.5 {
        1.0
    } else {
        -1.0
    };

    let mut current_values = vec![0.0; num_processes];
    for p_idx in 0..num_processes {
        current_values[p_idx] = filtration.get(t_idx, p_idx).clone();
    }
    let mut intermediate_values = current_values.clone();
    let mut k1 = vec![0.0; num_processes];
    let mut k2 = vec![0.0; num_processes];

    // --- STAGE 1: Compute k1 for Levy Processes ---
    for p_idx in 0..num_processes {
        if let Process::Levy(levy) = &mut process_universe.processes[p_idx] {
            let mut step_k1 = 0.0;
            for inc_idx in 0..levy.incrementors.len() {
                let c = (levy.coefficients[inc_idx]).eval(filtration, current_time).unwrap();
                let d = levy.incrementors[inc_idx].sample(t_idx, rng, filtration);

                step_k1 += if inc_idx == 0 {
                    c * d
                } else {
                    c * (d - sk * sqrt_dt)
                };
            }
            k1[p_idx] = step_k1;
            intermediate_values[p_idx] += step_k1;
            filtration.set(t_idx, p_idx, intermediate_values[p_idx].clone());
        }
    }

    // --- STAGE 2: Compute k2 for Levy Processes ---
    for p_idx in 0..num_processes {
        if let Process::Levy(levy) = &mut process_universe.processes[p_idx] {
            let mut step_k2 = 0.0;
            for inc_idx in 0..levy.incrementors.len() {
                let c = (levy.coefficients[inc_idx]).eval(filtration, current_time).unwrap();
                let d = levy.incrementors[inc_idx].sample(t_idx, rng, filtration);

                step_k2 += if inc_idx == 0 {
                    c * d
                } else {
                    c * (d + sk * sqrt_dt)
                };
            }
            k2[p_idx] = step_k2;
        }
    }

    // --- FINAL UPDATE: Average Levy stages & Evaluate Algebraic processes ---
    for p_idx in 0..num_processes {
        match &mut process_universe.processes[p_idx] {
            Process::Levy(_) => {
                let val = current_values[p_idx] + 0.5 * (k1[p_idx] + k2[p_idx]);
                filtration.set(t_idx + 1, p_idx, val);
            }
            Process::Algebraic(alg) => {
                let val = alg.coefficients[0].eval(filtration, current_time).unwrap();
                filtration.set(t_idx + 1, p_idx, val);
            }
        }
    }
}
