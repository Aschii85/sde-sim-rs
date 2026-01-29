use crate::filtration::Filtration;
use crate::rng::Rng;

pub fn runge_kutta_iteration(
    filtration: &mut Filtration,
    scenario_idx: usize,
    time_idx: usize,
    rng: &mut dyn Rng,
    scratchpad: &mut Filtration,
) {
    let time_start = filtration.times[time_idx];
    let time_end = filtration.times[time_idx + 1];
    let sqrt_dt = (time_end - time_start).sqrt();

    let num_processes = filtration.processes.len();

    let sk = if rand::random_bool(0.5) { 1.0 } else { -1.0 };
    let k1 = &mut vec![0.0; num_processes];
    let k2 = &mut vec![0.0; num_processes];

    // Stage 1
    for process_idx in 0..num_processes {
        let mut step_k1 = 0.0;
        let num_incrementors = filtration.processes[process_idx].incrementors.len();
        for inc_idx in 0..num_incrementors {
            let coeff = &filtration.processes[process_idx].coefficients[inc_idx];
            let c = coeff(filtration, time_start, time_idx, scenario_idx);
            let incrementor = &mut filtration.processes[process_idx].incrementors[inc_idx];
            let d = incrementor.sample(time_idx, scenario_idx, rng);
            step_k1 += if inc_idx == 0 {
                c * d
            } else {
                c * (d - sk * sqrt_dt)
            };
        }
        scratchpad.set(
            0,
            0,
            process_idx,
            filtration.get(scenario_idx, time_idx, process_idx) + step_k1,
        );
        k1[process_idx] = step_k1;
    }

    // Stage 2
    for process_idx in 0..num_processes {
        let mut step_k2 = 0.0;
        let num_incrementors = filtration.processes[process_idx].incrementors.len();
        for inc_idx in 0..num_incrementors {
            let coeff = &filtration.processes[process_idx].coefficients[inc_idx];
            let c = coeff(scratchpad, time_start, 0, 0);
            let incrementor = &mut filtration.processes[process_idx].incrementors[inc_idx];
            let d = incrementor.sample(time_idx, scenario_idx, rng);
            step_k2 += if inc_idx == 0 {
                c * d
            } else {
                c * (d + sk * sqrt_dt)
            };
        }
        k2[process_idx] = step_k2;
    }
    for process_idx in 0..num_processes {
        let val = filtration.get(scenario_idx, time_idx, process_idx)
            + 0.5 * (k1[process_idx] + k2[process_idx]);
        filtration.set(scenario_idx, time_idx + 1, process_idx, val);
    }
}
