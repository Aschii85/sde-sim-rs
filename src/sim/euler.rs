use crate::filtration::ScenarioFiltration;
use crate::proc::{Process, ProcessUniverse};
use crate::rng::BaseRng;

pub fn euler_iteration(
    filtration: &mut ScenarioFiltration,
    process_universe: &ProcessUniverse,
    t_idx: usize,
    rng: &mut dyn BaseRng,
) {
    let current_time = filtration.times[t_idx];
    let next_time = filtration.times[t_idx + 1];

    // 1. First Pass: Compute all SDE-based (Levy) updates
    for p_idx in &process_universe.levy_process_indices {
        if let Process::Levy(levy) = &process_universe.processes[*p_idx] {
            let mut val = filtration.get(t_idx, *p_idx);
            for inc_idx in 0..levy.incrementors.len() {
                // eval updates the internal Slab pointers using t_idx data
                let c = levy.coefficients[inc_idx].eval(current_time, filtration).unwrap();
                let x = levy.incrementors[inc_idx].sample(t_idx, filtration, rng);
                val += c * x;
            }
            filtration.set(t_idx + 1, *p_idx, val);
        }
    }

    // --- PASS 2: Evaluate Algebraic processes using next, t + 1, values ---
    for p_idx in &process_universe.algebraic_process_indices {
        if let Process::Algebraic(alg) = &process_universe.processes[*p_idx] {
            let val = alg.coefficients[0].eval(next_time, filtration).unwrap();
            filtration.set(t_idx + 1, *p_idx, val);
        }   
    }
}
