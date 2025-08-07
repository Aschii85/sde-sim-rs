use crate::filtration::Filtration;
use crate::process::Process;
use crate::rng::Rng;
use ordered_float::OrderedFloat;

// TODO: Implement Runge-Kutta, Millstein, etc. iterations.

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
            .value(t_start, scenario, &process.name())
            .unwrap_or(0.0);
        for idx in 0..process.coefficients().len() {
            let c = process.coefficients()[idx](&filtration, t_start, scenario);
            let x = process.incrementors()[idx].sample(scenario, t_start, t_end, rng);
            result += c * x;
        }
        filtration.set_value(t_end, scenario, &process.name(), result);
    }
}

// First-order Runge-Kutta scheme of string order 1 iteration
pub fn runge_kutta_iteration(
    filtration: &mut Filtration,
    processes: &mut Vec<Box<dyn Process>>,
    t_start: OrderedFloat<f64>,
    t_end: OrderedFloat<f64>,
    scenario: i32,
    rng: &mut dyn Rng,
) {
    // let h = t_end - t_start;
    // let mut k1 = vec![0.0; processes.len()];
    // let mut k2 = vec![0.0; processes.len()];
    // // Calculate k1
    // for i in 0..processes.len() {
    //     let process = &processes[i];
    //     let mut result = filtration
    //         .value(t_start, scenario, &process.name())
    //         .unwrap_or(0.0);
    //     for idx in 0..process.coefficients().len() {
    //         let c = process.coefficients()[idx](&filtration, t_start, scenario);
    //         let x = process.incrementors()[idx].sample(scenario, t_start.0, t_end.0, rng);
    //         result += c * x;
    //     }
    //     k1[i] = result;
    // }
    // for (process, rng) in processes.iter_mut().zip(rngs.iter_mut()) {
    //     let mut result = filtration
    //         .value(t_start, scenario, &process.name())
    //         .unwrap_or(0.0);
    //     for idx in 0..process.coefficients().len() {
    //         let c = process.coefficients()[idx](&filtration, t_start, scenario);
    //         let x = process.incrementors()[idx].sample(scenario, t_start.0, t_end.0, rng);
    //         result += c * x;
    //     }
    // }
}

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
                "runge_kutta" => {
                    runge_kutta_iteration(filtration, processes, ts[0], ts[1], scenario, rng);
                }
                _ => panic!("Unknown scheme: {}", scheme),
            }
        }
    }
}
