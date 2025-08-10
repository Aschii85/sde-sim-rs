use crate::filtration::Filtration;
use crate::process::Process;
use crate::rng::Rng;
use ordered_float::OrderedFloat;

/// Performs a single **Euler-Maruyama** iteration for the given processes.
///
/// This function advances the state of each stochastic process from `t_start` to `t_end`
/// for a single scenario. It calculates the new value by adding the weighted increments
/// (drift and diffusion) to the current value.
///
/// # Arguments
///
/// * `filtration` - A mutable reference to the `Filtration` storing the process values.
/// * `processes` - A mutable vector of boxed `Process` trait objects.
/// * `t_start` - The starting time of the iteration interval.
/// * `t_end` - The ending time of the iteration interval.
/// * `scenario` - The identifier for the current simulation path.
/// * `rng` - A mutable reference to a `Rng` trait object for sampling random numbers.
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
        for idx in 0..process.coefficients().len() {
            let c = process.coefficients()[idx](filtration, t_start, scenario);
            let x = process.incrementors()[idx].sample(scenario, t_start, t_end, rng);
            result += c * x;
        }
        filtration.set_value(t_end, scenario, process.name(), result);
    }
}

/// Performs a first-order **Runge-Kutta** scheme iteration for the given processes.
///
/// This function advances the state of each stochastic process from `t_start` to `t_end`
/// for a single scenario using a Runge-Kutta method. It involves calculating two intermediate
/// values, `k1` and `k2`, to achieve a higher order of accuracy than the Euler-Maruyama scheme.
///
/// **NOTE:** The current implementation assumes the time incrementor (`dt`) is the first
/// coefficient in the process definition. This is a known limitation.
///
/// # Arguments
///
/// * `filtration` - A mutable reference to the `Filtration` storing the process values.
/// * `processes` - A mutable vector of boxed `Process` trait objects.
/// * `t_start` - The starting time of the iteration interval.
/// * `t_end` - The ending time of the iteration interval.
/// * `scenario` - The identifier for the current simulation path.
/// * `rng` - A mutable reference to a `Rng` trait object for sampling random numbers.
pub fn runge_kutta_iteration(
    filtration: &mut Filtration,
    processes: &mut Vec<Box<dyn Process>>,
    t_start: OrderedFloat<f64>,
    t_end: OrderedFloat<f64>,
    scenario: i32,
    rng: &mut dyn Rng,
) {
    let sqrt_dt = (*(t_end - t_start)).sqrt();
    let mut k1 = vec![0.0; processes.len()];
    let mut k2 = vec![0.0; processes.len()];
    let mut filtration_plus_k1_at_t_end = Filtration::new(
        vec![t_end],
        vec![scenario],
        processes.iter().map(|p| p.name().clone()).collect(),
        ndarray::Array3::<f64>::zeros((1, 1, processes.len())),
        None,
    );
    let sk = if rand::random_bool(0.5) { 1.0 } else { -1.0 };
    // Calculate k1
    for (i, process) in processes.iter_mut().enumerate() {
        for idx in 0..process.coefficients().len() {
            let c = process.coefficients()[idx](filtration, t_start, scenario);
            let d = process.incrementors()[idx].sample(scenario, t_start, t_end, rng);
            // NOTE: This requires the time incrementor is first. Do something more sophisticated...
            k1[i] += if idx == 0 {
                c * d
            } else {
                c * (d - sk * sqrt_dt)
            };
        }
        filtration_plus_k1_at_t_end.set_value(
            t_end,
            scenario,
            process.name(),
            filtration.value(t_start, scenario, process.name()).unwrap() + k1[i],
        );
    }
    // Calculate k2
    for (i, process) in processes.iter_mut().enumerate() {
        for idx in 0..process.coefficients().len() {
            let c = process.coefficients()[idx](&filtration_plus_k1_at_t_end, t_end, scenario);
            let d = process.incrementors()[idx].sample(scenario, t_start, t_end, rng);
            // NOTE: This requires the time incrementor is first. Do something more sophisticated...
            k2[i] += if idx == 0 {
                c * d
            } else {
                c * (d + sk * sqrt_dt)
            };
        }
    }
    for (i, process) in processes.iter().enumerate() {
        filtration.set_value(
            t_end,
            scenario,
            process.name(),
            filtration.value(t_start, scenario, process.name()).unwrap() + 0.5 * (k1[i] + k2[i]),
        );
    }
}

/// Simulates a set of stochastic processes over time and scenarios.
///
/// This is the main simulation function that iterates through all specified scenarios
/// and time steps, applying the chosen numerical scheme (`euler` or `runge-kutta`)
/// to advance the state of each process.
///
/// # Arguments
///
/// * `filtration` - A mutable reference to the `Filtration` instance to store the results.
/// * `processes` - A mutable vector of boxed `Process` trait objects representing the SDEs.
/// * `time_steps` - A slice of `OrderedFloat<f64>` values for the time points of the simulation.
/// * `scenarios` - The total number of scenarios (simulation paths) to run.
/// * `rng` - A mutable reference to a `Rng` trait object for random number generation.
/// * `scheme` - A string slice indicating the numerical scheme to use ("euler" or "runge-kutta").
pub fn simulate(
    filtration: &mut Filtration,
    processes: &mut Vec<Box<dyn Process>>,
    time_steps: &Vec<OrderedFloat<f64>>,
    scenarios: &i32,
    rng: &mut dyn Rng,
    scheme: &str,
) {
    for scenario in 1..=*scenarios {
        for ts in time_steps.windows(2) {
            match scheme {
                "euler" => {
                    euler_iteration(filtration, processes, ts[0], ts[1], scenario, rng);
                }
                "runge-kutta" => {
                    runge_kutta_iteration(filtration, processes, ts[0], ts[1], scenario, rng);
                }
                _ => panic!("Unknown scheme: {}", scheme),
            }
        }
    }
}
