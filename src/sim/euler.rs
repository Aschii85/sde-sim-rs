use crate::filtration::Filtration;
use crate::rng::Rng;

pub fn euler_iteration(
    filtration: &mut Filtration,
    scenario_idx: usize,
    time_idx: usize,
    rng: &mut dyn Rng,
) {
    let time = filtration.times[time_idx];
    let num_processes = filtration.processes.len();
    for process_idx in 0..num_processes {
        let mut val = filtration.get(scenario_idx, time_idx, process_idx);
        let num_incrementors = filtration.processes[process_idx].incrementors.len();
        for inc_idx in 0..num_incrementors {
            let c = (filtration.processes[process_idx].coefficients[inc_idx])(
                filtration,
                time,
                time_idx,
                scenario_idx,
            );
            let incrementor = &mut filtration.processes[process_idx].incrementors[inc_idx];
            let x = incrementor.sample(time_idx, scenario_idx, rng);
            val += c * x;
        }
        filtration.set(scenario_idx, time_idx + 1, process_idx, val);
    }
}
