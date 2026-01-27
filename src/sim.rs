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
    // PRE-ALLOCATION: Create the temporary filtration once here.
    // This removes the libc malloc/free and HashMap initialization from the inner loop.
    let mut rk_scratchpad = if scheme == "runge-kutta" {
        Some(Filtration::new(
            vec![OrderedFloat(0.0)], // Placeholder time
            vec![0],                 // Placeholder scenario
            processes.iter().map(|p| p.name.clone()).collect(),
            ndarray::Array3::<f64>::zeros((1, 1, processes.len())),
            None,
        ))
    } else {
        None
    };

    for scenario in 1..=*scenarios {
        for ts in time_steps.windows(2) {
            match scheme {
                "euler" => {
                    euler_iteration(filtration, processes, ts[0], ts[1], scenario, rng);
                }
                "runge-kutta" => {
                    runge_kutta_iteration_reused(
                        filtration,
                        processes,
                        ts[0],
                        ts[1],
                        scenario,
                        rng,
                        rk_scratchpad.as_mut().unwrap(),
                    );
                }
                _ => panic!("Unknown scheme: {}", scheme),
            }
        }
    }
}

pub fn runge_kutta_iteration_reused(
    filtration: &mut Filtration,
    processes: &mut Vec<Box<LevyProcess>>,
    t_start: OrderedFloat<f64>,
    t_end: OrderedFloat<f64>,
    scenario: i32,
    rng: &mut dyn Rng,
    scratchpad: &mut Filtration,
) {
    let dt = (t_end - t_start).into_inner();
    let sqrt_dt = dt.sqrt();
    let num_p = processes.len();

    // Using a fixed-size vector to avoid per-call heap allocation
    let mut k1 = vec![0.0; num_p];
    let mut k2 = vec![0.0; num_p];

    let sk = if rand::random_bool(0.5) { 1.0 } else { -1.0 };

    // Stage 1: Calculate k1 and populate scratchpad
    for (i, process) in processes.iter_mut().enumerate() {
        let mut step_k1 = 0.0;

        for idx in 0..process.coefficients.len() {
            let c = process.coefficients[idx](filtration, t_start, scenario);
            let d = process.incrementors[idx].sample(scenario, t_start, t_end, rng);
            step_k1 += if idx == 0 {
                c * d
            } else {
                c * (d - sk * sqrt_dt)
            };
        }

        k1[i] = step_k1;
        let x_old = filtration
            .value(t_start, scenario, &process.name)
            .unwrap_or(0.0);

        // REUSE: We assume the scratchpad is set up for scenario '0' and index '0'
        // to minimize internal HashMap lookups within the scratchpad itself.
        scratchpad.set_value(OrderedFloat(0.0), 0, &process.name, x_old + step_k1);
    }

    // Stage 2: Calculate k2
    for (i, process) in processes.iter_mut().enumerate() {
        let mut step_k2 = 0.0;
        for idx in 0..process.coefficients.len() {
            // Note: We evaluate coefficients at the scratchpad's pseudo-time/scenario
            let c = process.coefficients[idx](scratchpad, OrderedFloat(0.0), 0);
            let d = process.incrementors[idx].sample(scenario, t_start, t_end, rng);
            step_k2 += if idx == 0 {
                c * d
            } else {
                c * (d + sk * sqrt_dt)
            };
        }
        k2[i] = step_k2;
    }

    // Final Update
    for (i, process) in processes.iter().enumerate() {
        let x_old = filtration
            .value(t_start, scenario, &process.name)
            .unwrap_or(0.0);
        filtration.set_value(
            t_end,
            scenario,
            &process.name,
            x_old + 0.5 * (k1[i] + k2[i]),
        );
    }
}

pub fn euler_iteration(
    filtration: &mut Filtration,
    processes: &mut Vec<Box<LevyProcess>>,
    t_start: OrderedFloat<f64>,
    t_end: OrderedFloat<f64>,
    scenario: i32,
    rng: &mut dyn Rng,
) {
    for process in processes.iter_mut() {
        let mut result = filtration
            .value(t_start, scenario, &process.name)
            .unwrap_or(0.0);
        for idx in 0..process.coefficients.len() {
            let c = process.coefficients[idx](filtration, t_start, scenario);
            let x = process.incrementors[idx].sample(scenario, t_start, t_end, rng);
            result += c * x;
        }
        filtration.set_value(t_end, scenario, &process.name, result);
    }
}
