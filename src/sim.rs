use rand::Rng;
use crate::filtration::Filtration;
use crate::process::levy::LevyLike;

pub fn euler_scheme_iteration(
    processes: &Vec<Box<dyn LevyLike>>,
    filtration: &mut Filtration,
    t_start: f64,
    t_end: f64,
    scenario: i32,
    rngs: &mut Vec<impl Rng>,
)
{
    let dt = t_end - t_start;
    for (process, rng) in processes.iter().zip(rngs.iter_mut()) {
        // Try to downcast to MarkovProcess
        let mut result = filtration.value(t_start, scenario, process.name().clone());
        for (coeff, incrementor) in process.coefficients().iter().zip(process.incrementors().iter()) {
            result += coeff(&filtration, t_start, scenario) * incrementor.sample(dt, rng);
        }
        filtration.set_value(t_end, scenario, process.name().clone(), result);
    }
}

pub fn simulate(
    filtration: &mut Filtration,
    processes: &Vec<Box<dyn LevyLike>>,
    time_steps: &Vec<f64>,
    scenarios: &i32,
    rngs: &mut Vec<impl Rng>,
)
{
    for scenario in 1..=*scenarios {
        for ts in time_steps.windows(2){
            euler_scheme_iteration(
                &processes,
                filtration,
                ts[0],
                ts[1],
                scenario,
                rngs,
            );
        }
    }
}
