#![allow(unused_imports)]
pub mod filtration;
pub mod process;
pub mod sim;

use plotly::{
    Plot, Scatter,
    common::{DashType, Line},
};
use polars::prelude::*;
use rand;
use std::collections::HashMap;
use std::time::Instant;

use crate::filtration::Filtration;
use crate::process::util::parse_equations;
use crate::process::{Process, ito::ItoProcess, levy::LevyProcess};
use crate::sim::simulate;

fn plot_scenarios(df: &DataFrame) -> polars::prelude::PolarsResult<()> {
    let mut plot = Plot::new();

    // Get the columns as Series
    let scenario_col = df.column("scenario")?.i32()?;
    let time_col = df.column("time")?.f64()?;
    let value_col = df.column("value")?.f64()?;
    let process_name_col = df.column("process_name")?.str()?; // Use .str() for string column in Polars

    // Determine unique scenarios and process names
    let max_scenario = scenario_col.max().unwrap_or(0);
    let unique_process_names: Vec<&str> = process_name_col
        .into_iter()
        .filter_map(|s| s)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    // Define the mapping from process_name to DashType
    let dash_map: HashMap<&str, DashType> = [
        ("X1", DashType::Solid),
        ("X2", DashType::Dot),
        ("X3", DashType::Dash),
        ("X4", DashType::LongDash),
        ("X5", DashType::DashDot),
        ("X6", DashType::LongDashDot),
        // Add more mappings as needed
    ]
    .iter()
    .cloned()
    .collect();

    for scenario in 1..=max_scenario {
        let scenario_mask = scenario_col.equal(scenario);
        for process_name in &unique_process_names {
            let process_mask = process_name_col.equal(*process_name);
            let combined_mask = &scenario_mask & &process_mask;
            // Apply the mask to get filtered data
            let time: Vec<f64> = time_col
                .filter(&combined_mask)?
                .into_no_null_iter()
                .collect();
            let value: Vec<f64> = value_col
                .filter(&combined_mask)?
                .into_no_null_iter()
                .collect();
            if time.is_empty() {
                continue; // Skip if no data for this combination
            }

            // Get the dash type from the map, default to Solid if not found
            let dash_type = dash_map.get(*process_name).unwrap_or(&DashType::Solid);
            let trace = Scatter::new(time, value)
                .name(format!("Scenario {} - {}", scenario, process_name))
                .line(Line::new().dash(dash_type.clone())); // Apply the dash type
            plot.add_trace(trace);
        }
    }

    plot.write_html("sandbox/output.html");
    println!("Plot with dash types saved to output_with_dash.html");
    Ok(())
}

fn main() {
    let dt: f64 = 0.1;
    let t_start: f64 = 0.0;
    let t_end: f64 = 100.0;
    let scenarios: i32 = 1000;
    let equations = [
        "dX1 = (0.005 * X1) * dt + (0.02 * X1) * dW1".to_string(),
        "dX2 = (0.005 * X2) * dt + (0.02 * X2) * dW1".to_string(),
    ];
    let levy_processes = parse_equations(&equations).expect("Failed to parse equations");
    // Convert Vec<LevyProcess> to Vec<Box<dyn Process>>
    let mut processes: Vec<Box<dyn Process>> = levy_processes
        .into_iter()
        .map(|p| Box::new(p) as Box<dyn Process>)
        .collect();
    let time_steps: Vec<f64> = (0..)
        .map(|i| t_start + i as f64 * dt)
        .take_while(|&t| t <= t_end)
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
    plot_scenarios(&df).unwrap();
}
