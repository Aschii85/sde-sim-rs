use crate::proc::{Process, ProcessUniverse};
use crate::rng::BaseRng;
use crate::filtration::ScenarioFiltration;

pub fn runge_kutta_iteration(
    filtration: &mut ScenarioFiltration,
    process_universe: &ProcessUniverse,
    t_idx: usize,
    rng: &mut dyn BaseRng,
) {
    let num_processes = process_universe.processes.len();
    let current_time = filtration.times[t_idx];
    let next_time = filtration.times[t_idx + 1];
    let dt = (next_time - current_time).into_inner();
    let sqrt_dt = dt.sqrt();

    // 1. Generate the sk random variable (±1) for the stochastic correction
    let sk = if rng.sample(t_idx, 0) > 0.5 { 1.0 } else { -1.0 };

    // 2. Pre-sample all increments for this step.
    // k1 and k2 MUST use the same dW and dN values.
    let mut step_increments = Vec::with_capacity(num_processes);
    for p_idx in 0..num_processes {
        let mut incs = Vec::new();
        if let Process::Levy(levy) = &process_universe.processes[p_idx] {
            for incr in &levy.incrementors {
                incs.push(incr.sample(t_idx, filtration, rng));
            }
        }
        step_increments.push(incs);
    }

    // Capture state at t_idx to avoid repetitive filtration lookups
    let mut x_t = vec![0.0; num_processes];
    for i in 0..num_processes {
        x_t[i] = filtration.get(t_idx, i);
    }

    // --- STAGE 1: Compute k1 ---
    let mut k1 = vec![0.0; num_processes];
    for p_idx in 0..num_processes {
        if let Process::Levy(levy) = &process_universe.processes[p_idx] {
            for (inc_idx, &d) in step_increments[p_idx].iter().enumerate() {
                let c = levy.coefficients[inc_idx].eval(current_time, filtration).unwrap();
                k1[p_idx] += c * d;
            }
        }
    }

    // --- STAGE 2: Compute k2 ---
    // We evaluate coefficients at the "probed" state (t + dt, x + k1 + perturbation)
    let mut k2 = vec![0.0; num_processes];
    
    // First, set a temporary "probed" state in the filtration for t+1
    for p_idx in 0..num_processes {
        if let Process::Levy(levy) = &process_universe.processes[p_idx] {
            // Find the diffusion perturbation (only if dW exists in this process)
            let mut perturbation = 0.0;
            for (inc_idx, incr) in levy.incrementors.iter().enumerate() {
                if incr.is_wiener() {
                    // This is the core of the Stochastic RK Strong Order 1.0 logic
                    perturbation += levy.coefficients[inc_idx].eval(current_time, filtration).unwrap() * sk * sqrt_dt;
                }
            }
            filtration.set(t_idx + 1, p_idx, x_t[p_idx] + k1[p_idx] + perturbation);
        }
    }

    // Now compute k2 using the probed state
    for p_idx in 0..num_processes {
        if let Process::Levy(levy) = &process_universe.processes[p_idx] {
            for (inc_idx, &d) in step_increments[p_idx].iter().enumerate() {
                // Evaluates coefficient at next_time using the state we just set at t+1
                let c = levy.coefficients[inc_idx].eval(next_time, filtration).unwrap();
                k2[p_idx] += c * d;
            }
        }
    }

    // --- FINAL UPDATE: Settle Levy Processes ---
    for p_idx in &process_universe.levy_process_indices {
        let final_val = x_t[*p_idx] + 0.5 * (k1[*p_idx] + k2[*p_idx]);
        filtration.set(t_idx + 1, *p_idx, final_val);
    }

    // --- FINAL UPDATE: Settle Algebraic processes ---
    // These must be calculated last so they see the final converged Levy values at t+1
    for p_idx in &process_universe.algebraic_process_indices {
        if let Process::Algebraic(alg) = &process_universe.processes[*p_idx] {
            let val = alg.coefficients[0].eval(next_time, filtration).unwrap();
            filtration.set(t_idx + 1, *p_idx, val);
        }   
    }
}