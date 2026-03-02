use ordered_float::OrderedFloat;
use polars::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

use sde_sim_rs::proc::util::parse_equations;
use sde_sim_rs::sim::simulate;

fn main() {
    // ────── configuration ──────
    let initial_values = HashMap::from([
        ("X1".to_string(), 100.0),
        ("X2".to_string(), 0.0),
    ]);

    let processes_equations = vec![
        "dX1 = ( sin(t) ) * dt + (0.01 * X1) * dW1".to_string(),
        "X2 = max(X1 - 100.0, 0.0)".to_string(),
    ];

    let scheme = "euler"; // other valid value: "runge-kutta"
    let rng_method = "pseudo"; // other valid value: "sobol"
    let scenarios: u64 = 10_000;

    // build a uniformly spaced time vector, identical to what the Python
    // wrapper accepts as `time_steps: Vec<f64>`.
    let dt = 0.1;
    let t_start = 0.0;
    let t_end = 100.0;
    let time_steps: Vec<f64> = (0..)
        .map(|i| t_start + i as f64 * dt)
        .take_while(|t| *t <= t_end)
        .collect();

    // convert the floats to `OrderedFloat` for internal use
    let ordered_steps: Vec<OrderedFloat<f64>> =
        time_steps.iter().copied().map(OrderedFloat).collect();

    // parse the equations into a ProcessUniverse (same work done in Python)
    let mut universe = parse_equations(&processes_equations, ordered_steps.clone())
        .expect("failed to parse process equations");

    // run the actual simulation; this mirrors the body of `simulate_py`
    let start = Instant::now();
    println!("running {} scenarios with {} rng...", scenarios, rng_method);
    let df: DataFrame = simulate(
        &mut universe,
        ordered_steps.clone(),
        initial_values.clone(),
        scenarios,
        scheme,
        rng_method,
    )
    .collect()
    .expect("failed to collect results");

    let elapsed = start.elapsed();
    println!("completed in {:.3}s", elapsed.as_secs_f64());

    // print a small portion of the output frame
    println!("{:#?}", df.head(Some(10)));
}
