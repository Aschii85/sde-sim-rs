use plotters::prelude::*;
use sobol::{Sobol, params::JoeKuoD6};
use statrs::distribution::{ContinuousCDF, Normal};
use std::f64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // GBM 1 parameters
    let s0_1 = 100.0;
    let mu_1 = 0.05;
    let sigma_1: f64 = 0.2;

    // GBM 2 parameters
    let s0_2 = 120.0;
    let mu_2 = 0.08;
    let sigma_2: f64 = 0.3;

    // Simulation parameters
    let num_steps = 252; // Number of trading days in a year
    let dt = 1.0 / num_steps as f64;
    let normal_dist = Normal::new(0.0, 1.0).unwrap();

    // Sobol sequence generator for two dimensions
    let dimension = 2;
    let params = JoeKuoD6::minimal();
    let mut sobol_iter = Sobol::<f64>::new(dimension, &params).skip(1); // Skip the first point

    // Store the paths
    let mut path_1: Vec<f64> = vec![s0_1];
    let mut path_2: Vec<f64> = vec![s0_2];

    let mut current_s1 = s0_1;
    let mut current_s2 = s0_2;

    for _ in 0..num_steps {
        if let Some(point) = sobol_iter.next() {
            // Get quasi-random numbers for each GBM
            let u1 = point[0];
            let u2 = point[1];

            // Transform to standard normal variables directly.
            let z1 = normal_dist.inverse_cdf(u1);
            let z2 = normal_dist.inverse_cdf(u2);

            // Apply the GBM formula
            current_s1 = current_s1
                * f64::exp((mu_1 - 0.5 * sigma_1.powi(2)) * dt + sigma_1 * z1 * dt.sqrt());
            current_s2 = current_s2
                * f64::exp((mu_2 - 0.5 * sigma_2.powi(2)) * dt + sigma_2 * z2 * dt.sqrt());

            path_1.push(current_s1);
            path_2.push(current_s2);
        }
    }

    println!("Path 1: {:?}", path_1);
    println!("Path 2: {:?}", path_2);

    // Plotting the paths
    let root = BitMapBackend::new("gbm_paths.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Geometric Brownian Motion Paths",
            ("sans-serif", 50).into_font(),
        )
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..num_steps, 0.0..150.0)?; // Adjust y-axis range as needed

    chart.configure_mesh().draw()?;

    // Plot path 1
    chart
        .draw_series(LineSeries::new(
            (0..).zip(path_1.iter()).map(|(x, y)| (x, *y)),
            &RED,
        ))?
        .label("Path 1")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Plot path 2
    chart
        .draw_series(LineSeries::new(
            (0..).zip(path_2.iter()).map(|(x, y)| (x, *y)),
            &BLUE,
        ))?
        .label("Path 2")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    println!("Plot saved to gbm_paths.png");

    Ok(())
}
