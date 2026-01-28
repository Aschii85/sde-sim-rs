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

    let mut processes =
        crate::process::util::parse_equations(&processes_equations, time_steps_ordered.clone())
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Failed to parse process equations: {}",
                    e
                ))
            })?;

    let num_incrementors = crate::process::util::num_incrementors();
    let mut rng: Box<dyn Rng> = if rng_method == "sobol" {
        Box::new(SobolRng::new(num_incrementors, time_steps_ordered.len()))
    } else {
        Box::new(PseudoRng::new(num_incrementors))
    };

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
