pub mod filtration;
pub mod process;
pub mod sim;

use polars::prelude::*;
use plotly::{Plot, Scatter};
use rand;
use std::time::Instant;

use crate::filtration::Filtration;
use crate::process::{Process, levy::LevyProcess, increment::Increment, ito::ItoProcess};
use crate::sim::simulate;

fn plot_scenarios(df: &DataFrame) -> polars::prelude::PolarsResult<()> {
    let mut plot = Plot::new();
    let scenario_col = df.column("scenario")?.i32()?;
    let max_scenario = scenario_col.max().unwrap();
    for scenario in 1..=max_scenario {
        let mask = scenario_col.equal(scenario);
        let time: Vec<f64> = df.column("time")?.f64()?.into_no_null_iter()
            .zip(mask.into_no_null_iter())
            .filter_map(|(t, m)| if m { Some(t) } else { None })
            .collect();
        let state: Vec<f64> = df.column("value")?.f64()?.into_no_null_iter()
            .zip(mask.into_no_null_iter())
            .filter_map(|(s, m)| if m { Some(s) } else { None })
            .collect();
        let trace = Scatter::new(time, state).name(format!("Scenario {}", scenario));
        plot.add_trace(trace);
    }
    plot.write_html("sandbox/output.html");
    println!("Plot saved to output.html");
    Ok(())
}

fn main() {
    // Wiener process: X_{t+dt} - X_t = a * dt + b * sqrt(dt) * N(0,1)
    let dt: f64 = 1.0;
    let t_start: f64 = 0.0;
    let t_end: f64 = 100.0;
    let scenarios: i32 = 1000;
    // let processes: Vec<Box<dyn Process>> = vec![
    //     Box::new(LevyProcess::new(
    //         "X1".to_string(),
    //         vec![
    //             Box::new(|f: &Filtration, t: f64, s: i32| 0.005 * f.value(t, s, "X1".to_string())), 
    //             Box::new(|f: &Filtration, t: f64, s: i32| 0.02 * f.value(t, s, "X1".to_string())),
    //         ],
    //         vec![Increment::Time, Increment::Wiener],
    //     ).unwrap()),
    //     Box::new(LevyProcess::new(
    //         "X2".to_string(),
    //         vec![
    //             Box::new(|f: &Filtration, t: f64, s: i32| 0.005 * f.value(t, s, "X1".to_string())), 
    //             Box::new(|f: &Filtration, t: f64, s: i32| 0.02 * f.value(t, s, "X2".to_string())),
    //         ],
    //         vec![Increment::Time, Increment::Wiener],
    //     ).unwrap())];
    let processes: Vec<Box<dyn Process>> = vec![
        Box::new(
            ItoProcess::from_string(
                "X1".to_string(),
                "(0.005 * X) * dt + (0.02 * X) * dW".to_string(),
            ).unwrap()
        ),
        Box::new(
            ItoProcess::from_string(
                "X2".to_string(),
                "(0.001 * X) * dt + (0.02 * X) * dW".to_string(),
            ).unwrap()
        ),
    ];
    let time_steps: Vec<f64> = (0..).map(|i| t_start + i as f64 * dt).take_while(|&t| t <= t_end).collect();
    let mut filtration = Filtration::new(
        time_steps.clone(),
        (1..=scenarios).collect(),
        processes.iter().map(|p| p.name().clone()).collect(),
        ndarray::Array3::<f64>::zeros((time_steps.len(), scenarios as usize, processes.len())),
    );
    for scenario in 1..=scenarios {
        filtration.set_value(t_start, scenario, "X1".to_string(), 1.0);
        filtration.set_value(t_start, scenario, "X2".to_string(), 0.5);
    }

    // Create a Vec of ThreadRngs, each owned and mutable
    let mut rngs: Vec<rand::rngs::ThreadRng> = (0..processes.len())
        .map(|_| rand::rngs::ThreadRng::default())
        .collect();
    
    let before = Instant::now();
    println!("Starting simulation...");
    simulate(
        &mut filtration,
        &processes,
        &time_steps,
        &scenarios,
        &mut rngs,
    );
    print!("Simulation completed in {} seconds.\n", before.elapsed().as_secs_f64());
    let df: DataFrame = filtration.to_dataframe();
    println!("{}", df);
    plot_scenarios(&df).unwrap();
}
