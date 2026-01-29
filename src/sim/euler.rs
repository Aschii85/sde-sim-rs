use crate::proc::LevyProcess;
use crate::rng::BaseRng;
use ordered_float::OrderedFloat;

pub fn euler_iteration(
    scenario_data: &mut [f64],
    processes: &mut [Box<LevyProcess>],
    times: &[OrderedFloat<f64>],
    time_idx: usize,
    rng: &mut dyn BaseRng,
) {
    let num_processes = processes.len();
    let current_time = times[time_idx];
    let offset = time_idx * num_processes;
    let current_step_values = scenario_data[offset..offset + num_processes].to_vec();
    #[allow(clippy::needless_range_loop)]
    for p_idx in 0..num_processes {
        let current_val_idx = offset + p_idx;
        let mut val = scenario_data[current_val_idx];
        for inc_idx in 0..processes[p_idx].incrementors.len() {
            // Pass the local copy to the coefficient function
            let c = (processes[p_idx].coefficients[inc_idx])(&current_step_values, current_time);
            let x = processes[p_idx].incrementors[inc_idx].sample(time_idx, rng);
            val += c * x;
        }
        let next_val_idx = (time_idx + 1) * num_processes + p_idx;
        scenario_data[next_val_idx] = val;
    }
}
