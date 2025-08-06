use crate::filtration::Filtration;
use crate::process::Process;
use ordered_float::OrderedFloat;
use rand::Rng;

pub fn euler_scheme_iteration(
    processes: &mut Vec<Box<dyn Process>>,
    filtration: &mut Filtration,
    t_start: OrderedFloat<f64>,
    t_end: OrderedFloat<f64>,
    scenario: i32,
    rngs: &mut Vec<impl Rng>,
) {
    for (process, rng) in processes.iter_mut().zip(rngs.iter_mut()) {
        let mut result = filtration
            .value(t_start, scenario, &process.name())
            .unwrap_or(0.0);
        for idx in 0..process.coefficients().len() {
            let c = process.coefficients()[idx](&filtration, t_start, scenario);
            let x = process.incrementors()[idx].sample(scenario, t_start.0, t_end.0, rng);
            result += c * x;
        }
        filtration.set_value(t_end, scenario, &process.name(), result);
    }
}

pub fn simulate(
    filtration: &mut Filtration,
    processes: &mut Vec<Box<dyn Process>>,
    time_steps: &Vec<OrderedFloat<f64>>,
    scenarios: &i32,
    rngs: &mut Vec<impl Rng>,
) {
    for scenario in 1..=*scenarios {
        for ts in time_steps.windows(2) {
            euler_scheme_iteration(processes, filtration, ts[0], ts[1], scenario, rngs);
        }
    }
}
