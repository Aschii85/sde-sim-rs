use ordered_float::OrderedFloat;
use polars::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

use sde_sim_rs::filtration::Filtration;
use sde_sim_rs::proc::util::parse_equations;
use sde_sim_rs::sim::simulate;

fn main() {
    // Simulation Parameters
    let dt: f64 = 0.1;
    let t_start: f64 = 0.0;
    let t_end: f64 = 100.0;
    let scenarios: i32 = 10000;
    let initial_values = HashMap::from([("X1".to_string(), 1.0), ("X2".to_string(), 1.0)]);
    let equations = [
        "dX1 = (0.005 * X1) * dt + (0.02 * X1) * dW1".to_string(),
        "dX2 = (0.005 * X2) * dt + (0.02 * X1) * dW1 + (0.01 * X2) * dW2 + (1) * dJ1(0.5)"
            .to_string(),
    ];
    let scheme = "runge-kutta"; // "euler" or "runge-kutta"
    let rng_method = "sobol"; // "pseudo" or "sobol"

    // 1. Prepare Time Steps
    let time_steps: Vec<OrderedFloat<f64>> = (0..)
        .map(|i| OrderedFloat(t_start + i as f64 * dt))
        .take_while(|t| t.0 <= t_end)
        .collect();

    // 2. Parse equations
    let universe =
        parse_equations(&equations, time_steps.clone()).expect("Failed to parse equations");

    // 3. Initialize Filtration
    let mut filtration = Filtration::new(
        universe,
        time_steps.clone(),
        (1..=scenarios).collect(),
        Some(initial_values),
    );

    // Run Simulation
    let before = Instant::now();
    println!("Starting simulation with {} RNG...", rng_method);
    simulate(&mut filtration, scheme, rng_method);

    let duration = before.elapsed();
    println!(
        "Simulation completed in {:.4} seconds.\n",
        duration.as_secs_f64()
    );

    let df: DataFrame = filtration.to_dataframe();
    println!("{}", df);

    assert!(duration.as_secs_f64() > 0.0);
}
