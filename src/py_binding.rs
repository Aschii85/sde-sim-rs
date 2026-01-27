use ordered_float::OrderedFloat;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use std::collections::HashMap;

use crate::filtration::Filtration;
use crate::rng::{PseudoRng, Rng, SobolRng};
use crate::sim::simulate;

#[pyfunction]
#[pyo3(name = "simulate")]
pub fn simulate_py(
    processes_equations: Vec<String>,
    time_steps: Vec<f64>,
    scenarios: i32,
    initial_values: HashMap<String, f64>,
    rng_method: String,
    scheme: String,
) -> PyResult<PyDataFrame> {
    let time_steps_ordered: Vec<OrderedFloat<f64>> =
        time_steps.iter().copied().map(OrderedFloat).collect();

    // 1. Pre-extract names from equations to initialize Filtration indices.
    // Equations are "dX = ...", so we split by '=' and trim 'd'.
    let process_names: Vec<String> = processes_equations
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

    // 2. Initialize Filtration FIRST so it can provide indices to the parser.
    let mut filtration = Filtration::new(
        time_steps_ordered.clone(),
        (1..=scenarios).collect(),
        process_names.clone(),
        ndarray::Array3::<f64>::zeros((
            time_steps_ordered.len(),
            scenarios as usize,
            process_names.len(),
        )),
        Some(initial_values),
    );

    // 3. Parse equations using the filtration reference for "Resolve Once" optimization.
    let mut processes = crate::process::util::parse_equations(&processes_equations, &filtration)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to parse process equations: {}",
                e
            ))
        })?;

    // 4. Set up RNG
    let mut rng: Box<dyn Rng> = if rng_method == "sobol" {
        Box::new(SobolRng::new(
            processes
                .iter_mut()
                .flat_map(|p| p.incrementors().iter_mut().map(|i| i.name().clone()))
                .collect::<Vec<String>>(),
            time_steps_ordered.clone(),
        ))
    } else {
        Box::new(PseudoRng::new(
            processes
                .iter_mut()
                .flat_map(|p| p.incrementors().iter_mut().map(|i| i.name().clone()))
                .collect::<Vec<String>>(),
        ))
    };

    // 5. Run simulation
    simulate(
        &mut filtration,
        &mut processes,
        &time_steps_ordered,
        &scenarios,
        &mut *rng,
        &scheme,
    );

    let df: DataFrame = filtration.to_dataframe();
    Ok(PyDataFrame(df))
}

#[pymodule]
fn sde_sim_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_py, m)?)?;
    Ok(())
}
