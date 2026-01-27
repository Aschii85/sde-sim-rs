use ordered_float::OrderedFloat;
use polars::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

use sde_sim_rs::filtration::Filtration;
use sde_sim_rs::process::util::parse_equations;
use sde_sim_rs::rng::{PseudoRng, Rng, SobolRng};
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
    let rng_scheme = "sobol"; // "pseudo" or "sobol"

    // 1. Prepare Time Steps
    let time_steps: Vec<OrderedFloat<f64>> = (0..)
        .map(|i| OrderedFloat(t_start + i as f64 * dt))
        .take_while(|t| t.0 <= t_end)
        .collect();

    // 2. Pre-extract process names for Filtration initialization
    let process_names: Vec<String> = equations
        .iter()
        .map(|eq| {
            eq.split('=')
                .next()
                .unwrap_or("")
                .trim()
                .trim_start_matches('d')
                .to_string()
        })
        .collect();

    // 3. Initialize Filtration (Container must exist for the parser to resolve indices)
    let mut filtration = Filtration::new(
        time_steps.clone(),
        (1..=scenarios).collect(),
        process_names.clone(),
        ndarray::Array3::<f64>::zeros((time_steps.len(), scenarios as usize, process_names.len())),
        Some(initial_values),
    );

    // 4. Parse equations using the filtration reference
    let mut processes =
        parse_equations(&equations, &filtration).expect("Failed to parse equations");

    // 5. Setup RNG
    let mut rng: Box<dyn Rng> = if rng_scheme == "sobol" {
        Box::new(SobolRng::new(
            processes
                .iter_mut()
                .flat_map(|p| p.incrementors.iter_mut().map(|i| i.name().clone()))
                .collect::<Vec<String>>(),
            time_steps.clone(),
        ))
    } else {
        Box::new(PseudoRng::new(
            processes
                .iter_mut()
                .flat_map(|p| p.incrementors.iter_mut().map(|i| i.name().clone()))
                .collect::<Vec<String>>(),
        ))
    };

    // Run Simulation
    let before = Instant::now();
    println!("Starting simulation...");
    simulate(
        &mut filtration,
        &mut processes,
        &time_steps,
        &scenarios,
        &mut *rng,
        scheme,
    );

    let duration = before.elapsed();
    println!(
        "Simulation completed in {:.4} seconds.\n",
        duration.as_secs_f64()
    );

    let df: DataFrame = filtration.to_dataframe();
    println!("{}", df);

    assert!(duration.as_secs_f64() > 0.0);
}
