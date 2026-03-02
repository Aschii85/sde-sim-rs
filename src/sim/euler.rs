use crate::filtration::ScenarioFiltration;
use crate::proc::{Process, ProcessUniverse};
use crate::rng::BaseRng;

pub fn euler_iteration(
    filtration: &mut ScenarioFiltration,
    process_universe: &mut ProcessUniverse,
    t_idx: usize,
    rng: &mut dyn BaseRng,
) {
    let current_time = filtration.times[t_idx];
    let next_time = filtration.times[t_idx + 1];

    // 1. First Pass: Compute all SDE-based (Levy) updates
    // We calculate the delta and store it in t_idx + 1
    for p_idx in 0..process_universe.processes.len() {
        if let Process::Levy(levy) = &mut process_universe.processes[p_idx] {
            let mut val = filtration.get(t_idx, p_idx);
            
            for inc_idx in 0..levy.incrementors.len() {
                // eval updates the internal Slab pointers using t_idx data
                let c = levy.coefficients[inc_idx].eval(filtration, current_time).unwrap();
                let x = levy.incrementors[inc_idx].sample(t_idx, rng, filtration);
                val += c * x;
            }
            filtration.set(t_idx + 1, p_idx, val);
        }
    }

    // 2. Second Pass: Compute Algebraic updates
    // These depend on the results of the Levy updates at the SAME time step (t_idx + 1)
    for p_idx in 0..process_universe.processes.len() {
        if let Process::Algebraic(alg) = &mut process_universe.processes[p_idx] {
            // NOTE: We use next_time and the data already set in next_t_idx
            let val = alg.coefficients[0].eval(filtration, next_time).unwrap(); // TODO: SHOULD BE NEXT???
            filtration.set(t_idx + 1, p_idx, val);
        }
    }
}
