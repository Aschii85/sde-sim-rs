use crate::sim::simulate;
use ordered_float::OrderedFloat;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use std::collections::HashMap;

#[pyfunction]
#[pyo3(name = "simulate")]
pub fn simulate_py(
    py: Python<'_>,
    processes_equations: Vec<String>,
    time_steps: Vec<f64>,
    scenarios: i32,
    initial_values: HashMap<String, f64>,
    rng_method: String,
    scheme: String,
) -> PyResult<PyDataFrame> {
    // Basic validation for scenario count
    if scenarios <= 0 {
        return Err(PyValueError::new_err(
            "scenarios must be a positive integer",
        ));
    }

    let time_steps_ordered: Vec<OrderedFloat<f64>> =
        time_steps.iter().copied().map(OrderedFloat).collect();

    // 1. Parse equations and map internal errors to Python ValueErrors
    let processes =
        crate::proc::util::parse_equations(&processes_equations, time_steps_ordered.clone())
            .map_err(|e| PyValueError::new_err(format!("Failed to parse equations: {}", e)))?;

    // 2. Run simulation while releasing the GIL
    // We map simulation errors to PyRuntimeError
    let df = py
        .allow_threads(|| {
            simulate(
                &processes,
                time_steps_ordered,
                initial_values,
                scenarios as u64,
                &scheme,
                &rng_method,
            )
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Simulation failed: {}", e)))?;

    // 3. Collect the LazyFrame into a DataFrame
    // Polars errors are converted to Python-friendly messages
    let collected_df = df
        .collect()
        .map_err(|e| PyRuntimeError::new_err(format!("Polars collection error: {}", e)))?;

    Ok(PyDataFrame(collected_df))
}

#[pymodule]
fn sde_sim_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_py, m)?)?;
    Ok(())
}
