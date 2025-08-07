#![allow(unused_imports)]
pub mod filtration;
pub mod process;
pub mod sim;

use ordered_float::OrderedFloat;
use polars::prelude::*;
use rand;
use std::collections::HashMap;
use std::time::Instant;

use crate::filtration::Filtration;
use crate::process::util::parse_equations;
use crate::process::{Process, ito::ItoProcess, levy::LevyProcess};
use crate::sim::simulate;

fn main() {
    let dt: f64 = 0.1;
    let t_start: f64 = 0.0;
    let t_end: f64 = 100.0;
    let scenarios: i32 = 1000;
    let equations = [
        "dX1 = (0.005 * X1) * dt + (0.02 * X1) * dW1".to_string(),
        "dX2 = (0.005 * X2) * dt + (0.02 * X1) * dW1 + (0.01 * X2) * dW2".to_string(),
    ];
    let mut processes = parse_equations(&equations).expect("Failed to parse equations");
    let time_steps: Vec<OrderedFloat<f64>> = (0..)
        .map(|i| OrderedFloat(t_start + i as f64 * dt))
        .take_while(|t| t.0 <= t_end)
        .collect();
    let mut filtration = Filtration::new(
        time_steps.clone(),
        (1..=scenarios).collect(),
        processes.iter().map(|p| p.name().clone()).collect(),
        ndarray::Array3::<f64>::zeros((time_steps.len(), scenarios as usize, processes.len())),
        Some(HashMap::from([
            ("X1".to_string(), 1.0),
            ("X2".to_string(), 1.0),
        ])),
    );

    // Create a Vec of ThreadRngs, each owned and mutable
    let mut rngs: Vec<rand::rngs::ThreadRng> = (0..processes.len())
        .map(|_| rand::rngs::ThreadRng::default())
        .collect();

    let before = Instant::now();
    println!("Starting simulation...");
    simulate(
        &mut filtration,
        &mut processes,
        &time_steps,
        &scenarios,
        &mut rngs,
    );
    print!(
        "Simulation completed in {} seconds.\n",
        before.elapsed().as_secs_f64()
    );
    let df: DataFrame = filtration.to_dataframe();
    println!("{}", df);
}
