use crate::filtration::Filtration;
use crate::process::LevyProcess;
use crate::rng::Rng;
use ordered_float::OrderedFloat;

pub fn simulate(
    filtration: &mut Filtration,
    processes: &mut Vec<Box<LevyProcess>>,
    time_steps: &[OrderedFloat<f64>],
    scenarios: &i32,
    rng: &mut dyn Rng,
    scheme: &str,
) {
    let num_p = processes.len();
    let num_s = *scenarios as usize;

    let process_indices: Vec<usize> = processes
        .iter()
        .map(|p| {
            filtration
                .process_name_idx_map
                .get(&p.name)
                .copied()
                .expect("Process name missing in filtration")
        })
        .collect();

    let mut current_values = vec![0.0; num_p];
    let mut k1 = vec![0.0; num_p];
    let mut k2 = vec![0.0; num_p];
    let mut results = vec![0.0; num_p]; // Buffer for Euler results

    let mut rk_scratchpad = if scheme == "runge-kutta" {
        Some(Filtration::new(
            vec![OrderedFloat(0.0)],
            vec![0],
            processes.iter().map(|p| p.name.clone()).collect(),
            ndarray::Array3::<f64>::zeros((1, 1, num_p)),
            None,
        ))
    } else {
        None
    };

    for scenario in 1..=*scenarios {
        for ts in time_steps.windows(2) {
            match scheme {
                "euler" => {
                    euler_iteration(
                        filtration,
                        processes,
                        &process_indices,
                        ts[0],
                        scenario,
                        rng,
                        &mut current_values,
                        &mut results,
                        num_s,
                    );
                }
                "runge-kutta" => {
                    runge_kutta_iteration(
                        filtration,
                        processes,
                        &process_indices,
                        ts[0],
                        ts[1],
                        scenario,
                        rng,
                        rk_scratchpad.as_mut().unwrap(),
                        &mut current_values,
                        &mut k1,
                        &mut k2,
                        num_s,
                    );
                }
                _ => panic!("Unknown scheme"),
            }
        }
    }
}

pub fn runge_kutta_iteration(
    filtration: &mut Filtration,
    processes: &mut Vec<Box<LevyProcess>>,
    process_indices: &[usize],
    t_start: OrderedFloat<f64>,
    t_end: OrderedFloat<f64>,
    scenario: i32,
    rng: &mut dyn Rng,
    scratchpad: &mut Filtration,
    current_values: &mut Vec<f64>,
    k1: &mut Vec<f64>,
    k2: &mut Vec<f64>,
    num_scenarios: usize,
) {
    let dt = (t_end - t_start).0;
    let sqrt_dt = dt.sqrt();
    let num_p = processes.len();

    let time_idx = *filtration.time_idx_map.get(&t_start).unwrap();
    let scenario_idx = (scenario - 1) as usize;
    let base_offset = (time_idx * num_scenarios * num_p) + (scenario_idx * num_p);

    // Copy current step values (read-only borrow ends here)
    current_values.copy_from_slice(
        &filtration.raw_values.as_slice().unwrap()[base_offset..base_offset + num_p],
    );

    let sk = if rand::random_bool(0.5) { 1.0 } else { -1.0 };

    // Stage 1: Evaluation (Uses filtration as &)
    for (i, process) in processes.iter_mut().enumerate() {
        let mut step_k1 = 0.0;
        for (inc_idx, incrementor) in process.incrementors.iter_mut().enumerate() {
            let c =
                (process.coefficients[inc_idx])(filtration, current_values, t_start.0, scenario);
            let d = incrementor.sample(time_idx, scenario_idx, rng);
            step_k1 += if inc_idx == 0 {
                c * d
            } else {
                c * (d - sk * sqrt_dt)
            };
        }
        k1[i] = step_k1;
        scratchpad.raw_values[[0, 0, i]] = current_values[process_indices[i]] + step_k1;
    }

    // Stage 2: Evaluation (Uses scratchpad as &)
    let scratch_ptr = scratchpad.raw_values.as_slice().unwrap();
    for (i, process) in processes.iter_mut().enumerate() {
        let mut step_k2 = 0.0;
        for (inc_idx, incrementor) in process.incrementors.iter_mut().enumerate() {
            let c = (process.coefficients[inc_idx])(scratchpad, scratch_ptr, 0.0, 0);
            let d = incrementor.sample(time_idx, scenario_idx, rng);
            step_k2 += if inc_idx == 0 {
                c * d
            } else {
                c * (d + sk * sqrt_dt)
            };
        }
        k2[i] = step_k2;
    }

    // Write-back: Mutable borrow occurs only here at the end
    let next_offset = base_offset + (num_scenarios * num_p);
    let target_slice = filtration.raw_values.as_slice_mut().unwrap();
    for (i, &p_idx) in process_indices.iter().enumerate() {
        target_slice[next_offset + p_idx] = current_values[p_idx] + 0.5 * (k1[i] + k2[i]);
    }
}

pub fn euler_iteration(
    filtration: &mut Filtration,
    processes: &mut Vec<Box<LevyProcess>>,
    process_indices: &[usize],
    t_start: OrderedFloat<f64>,
    scenario: i32,
    rng: &mut dyn Rng,
    current_values: &mut Vec<f64>,
    results: &mut Vec<f64>,
    num_scenarios: usize,
) {
    let num_p = processes.len();
    let time_idx = *filtration.time_idx_map.get(&t_start).unwrap();
    let scenario_idx = (scenario - 1) as usize;
    let base_offset = (time_idx * num_scenarios * num_p) + (scenario_idx * num_p);

    // Copy current step values (read-only borrow ends here)
    current_values.copy_from_slice(
        &filtration.raw_values.as_slice().unwrap()[base_offset..base_offset + num_p],
    );

    // Perform all calculations into the 'results' buffer first
    for (i, process) in processes.iter_mut().enumerate() {
        let p_idx = process_indices[i];
        let mut val = current_values[p_idx];

        for (inc_idx, incrementor) in process.incrementors.iter_mut().enumerate() {
            // Immutable borrow of filtration happens here
            let c =
                (process.coefficients[inc_idx])(filtration, current_values, t_start.0, scenario);
            let x = incrementor.sample(time_idx, scenario_idx, rng);
            val += c * x;
        }
        results[i] = val;
    }

    // Now borrow filtration mutably to write back
    let next_offset = base_offset + (num_scenarios * num_p);
    let target_slice = filtration.raw_values.as_slice_mut().unwrap();
    for (i, &p_idx) in process_indices.iter().enumerate() {
        target_slice[next_offset + p_idx] = results[i];
    }
}
