use crate::proc::LevyProcess;
use crate::rng::BaseRng;
use ordered_float::OrderedFloat;

pub fn runge_kutta_iteration(
    scenario_data: &mut [f64], // Slice for JUST this scenario
    processes: &mut [Box<LevyProcess>],
    times: &[OrderedFloat<f64>],
    time_idx: usize,
    rng: &mut dyn BaseRng,
) {
    let num_processes = processes.len();
    let current_time = times[time_idx];
    let dt = (times[time_idx + 1] - times[time_idx]).into_inner();
    let sqrt_dt = dt.sqrt();

    let offset = time_idx * num_processes;
    let next_offset = (time_idx + 1) * num_processes;

    // 1. Capture the current state (S_t)
    let current_values = scenario_data[offset..offset + num_processes].to_vec();

    // We need a random variable sk (±1) for the stochastic RK scheme.
    // To keep it consistent with the RNG system, we'll derive it from the RNG.
    let sk = if rng.sample(time_idx, 0) > 0.5 {
        1.0
    } else {
        -1.0
    };

    let mut k1 = vec![0.0; num_processes];
    let mut intermediate_values = current_values.clone();

    // --- STAGE 1: Compute k1 ---
    for p_idx in 0..num_processes {
        let mut step_k1 = 0.0;
        let num_incs = processes[p_idx].incrementors.len();

        for inc_idx in 0..num_incs {
            // Evaluate coefficients at current state
            let c = (processes[p_idx].coefficients[inc_idx])(&current_values, current_time);
            let d = processes[p_idx].incrementors[inc_idx].sample(time_idx, rng);

            // Scheme logic: if it's the first incrementor (usually dt), apply normally.
            // Otherwise, apply the drift/diffusion adjustment for RK SDEs.
            step_k1 += if inc_idx == 0 {
                c * d
            } else {
                c * (d - sk * sqrt_dt)
            };
        }
        k1[p_idx] = step_k1;
        // Update intermediate "scratchpad" state for Stage 2
        intermediate_values[p_idx] += step_k1;
    }

    // --- STAGE 2: Compute k2 ---
    let mut k2 = vec![0.0; num_processes];
    for p_idx in 0..num_processes {
        let mut step_k2 = 0.0;
        let num_incs = processes[p_idx].incrementors.len();

        for inc_idx in 0..num_incs {
            // Evaluate coefficients at the PREDICTED intermediate state
            let c = (processes[p_idx].coefficients[inc_idx])(&intermediate_values, current_time);

            // Note: sample() returns the cached value for the same time_idx,
            // ensuring we use the same dW for both stages.
            let d = processes[p_idx].incrementors[inc_idx].sample(time_idx, rng);

            step_k2 += if inc_idx == 0 {
                c * d
            } else {
                c * (d + sk * sqrt_dt)
            };
        }
        k2[p_idx] = step_k2;
    }

    // --- FINAL UPDATE: Average the stages ---
    for p_idx in 0..num_processes {
        let val = current_values[p_idx] + 0.5 * (k1[p_idx] + k2[p_idx]);
        scenario_data[next_offset + p_idx] = val;
    }
}
