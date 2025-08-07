#![allow(unused_imports)]
use ordered_float::OrderedFloat;
use polars::prelude::*;
use rand;
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
    let scenarios: i32 = 1000;
    let initial_values = HashMap::from([("X1".to_string(), 1.0), ("X2".to_string(), 1.0)]);
    let equations = [
        "dX1 = (0.005 * X1) * dt + (0.02 * X1) * dW1".to_string(),
        "dX2 = (0.005 * X2) * dt + (0.02 * X1) * dW1 + (0.01 * X2) * dW2".to_string(),
    ];
    let mut processes = parse_equations(&equations).expect("Failed to parse equations");
    let scheme = "euler"; // "euler" or "runge_kutta"
    let rng_scheme = "sobol"; // "pseudo" or "sobol"

    // Start Setup
    let time_steps: Vec<OrderedFloat<f64>> = (0..)
        .map(|i| OrderedFloat(t_start + i as f64 * dt))
        .take_while(|t| t.0 <= t_end)
        .collect();
    let mut filtration = Filtration::new(
        time_steps.clone(),
        (1..=scenarios).collect(),
        processes.iter().map(|p| p.name().clone()).collect(),
        ndarray::Array3::<f64>::zeros((time_steps.len(), scenarios as usize, processes.len())),
        Some(initial_values),
    );
    let mut rng: Box<dyn Rng> = if rng_scheme == "sobol" {
        Box::new(SobolRng::new(
            processes
                .iter_mut()
                .flat_map(|p| p.incrementors().iter_mut().map(|i| i.name().clone()))
                .collect::<Vec<String>>(),
        ))
    } else {
        Box::new(PseudoRng::new(
            processes
                .iter_mut()
                .flat_map(|p| p.incrementors().iter_mut().map(|i| i.name().clone()))
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
        &scheme,
    );
    print!(
        "Simulation completed in {} seconds.\n",
        before.elapsed().as_secs_f64()
    );
    let df: DataFrame = filtration.to_dataframe();
    println!("{}", df);
    assert!(before.elapsed().as_secs_f64() > 0.0);
}
