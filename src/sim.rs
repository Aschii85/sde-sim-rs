use crate::filtration::Filtration;
use crate::process::Process;
use crate::rng::Rng;
use ordered_float::OrderedFloat;

pub fn euler_iteration(
    filtration: &mut Filtration,
    processes: &mut Vec<Box<dyn Process>>,
    t_start: OrderedFloat<f64>,
    t_end: OrderedFloat<f64>,
    scenario: i32,
    rng: &mut dyn Rng,
) {
    for process in processes.iter_mut() {
        let mut result = filtration
            .value(t_start, scenario, process.name())
            .unwrap_or(0.0);

        // Use the simple zip approach to get coefficients and incrementors
        let num_coeffs = process.coefficients().len();
        for idx in 0..num_coeffs {
            // Reverting to the 3-argument signature: (filtration, time, scenario)
            let c = process.coefficients()[idx](filtration, t_start, scenario);
            let x = process.incrementors()[idx].sample(scenario, t_start, t_end, rng);
            result += c * x;
        }
        filtration.set_value(t_end, scenario, process.name(), result);
    }
}

pub fn runge_kutta_iteration(
    filtration: &mut Filtration,
    processes: &mut Vec<Box<dyn Process>>,
    t_start: OrderedFloat<f64>,
    t_end: OrderedFloat<f64>,
    scenario: i32,
    rng: &mut dyn Rng,
) {
    let dt = (t_end - t_start).into_inner();
    let sqrt_dt = dt.sqrt();
    let mut k1 = vec![0.0; processes.len()];
    let mut k2 = vec![0.0; processes.len()];

    // Original temporary filtration logic
    let mut filtration_plus_k1 = Filtration::new(
        vec![t_end],
        vec![scenario],
        processes.iter().map(|p| p.name().clone()).collect(),
        ndarray::Array3::<f64>::zeros((1, 1, processes.len())),
        None,
    );

    let sk = if rand::random_bool(0.5) { 1.0 } else { -1.0 };

    for (i, process) in processes.iter_mut().enumerate() {
        for idx in 0..process.coefficients().len() {
            let c = process.coefficients()[idx](filtration, t_start, scenario);
            let d = process.incrementors()[idx].sample(scenario, t_start, t_end, rng);
            k1[i] += if idx == 0 {
                c * d
            } else {
                c * (d - sk * sqrt_dt)
            };
        }
        let x_old = filtration
            .value(t_start, scenario, process.name())
            .unwrap_or(0.0);
        filtration_plus_k1.set_value(t_end, scenario, process.name(), x_old + k1[i]);
    }

    for (i, process) in processes.iter_mut().enumerate() {
        for idx in 0..process.coefficients().len() {
            let c = process.coefficients()[idx](&filtration_plus_k1, t_end, scenario);
            let d = process.incrementors()[idx].sample(scenario, t_start, t_end, rng);
            k2[i] += if idx == 0 {
                c * d
            } else {
                c * (d + sk * sqrt_dt)
            };
        }
    }

    for (i, process) in processes.iter().enumerate() {
        let x_old = filtration
            .value(t_start, scenario, process.name())
            .unwrap_or(0.0);
        filtration.set_value(
            t_end,
            scenario,
            process.name(),
            x_old + 0.5 * (k1[i] + k2[i]),
        );
    }
}

pub fn simulate(
    filtration: &mut Filtration,
    processes: &mut Vec<Box<dyn Process>>,
    time_steps: &[OrderedFloat<f64>],
    scenarios: &i32,
    rng: &mut dyn Rng,
    scheme: &str,
) {
    for scenario in 1..=*scenarios {
        for ts in time_steps.windows(2) {
            match scheme {
                "euler" => euler_iteration(filtration, processes, ts[0], ts[1], scenario, rng),
                "runge-kutta" => {
                    runge_kutta_iteration(filtration, processes, ts[0], ts[1], scenario, rng)
                }
                _ => panic!("Unknown scheme"),
            }
        }
    }
}
