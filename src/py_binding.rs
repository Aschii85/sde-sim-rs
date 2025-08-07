use ordered_float::OrderedFloat;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use std::collections::HashMap;

use crate::filtration::Filtration;
use crate::sim::simulate;

#[pyfunction]
#[pyo3(name = "simulate")]
pub fn simulate_py(
    processes_equations: Vec<String>,
    time_steps: Vec<f64>,
    scenarios: i32,
    initial_values: HashMap<String, f64>,
) -> PyResult<PyDataFrame> {
    // Use parse_equations from levy.rs
    let mut processes =
        crate::process::util::parse_equations(&processes_equations).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to parse process equations: {}",
                e
            ))
        })?;
    let time_steps_ordered: Vec<OrderedFloat<f64>> =
        time_steps.iter().copied().map(OrderedFloat).collect();
    let mut filtration = Filtration::new(
        time_steps_ordered.clone(),
        (1..=scenarios).collect(),
        processes.iter().map(|p| p.name().clone()).collect(),
        ndarray::Array3::<f64>::zeros((
            time_steps_ordered.len(),
            scenarios as usize,
            processes.len(),
        )),
        Some(initial_values),
    );

    // Create a Vec of ThreadRngs, each owned and mutable
    let mut rngs: Vec<rand::rngs::ThreadRng> = (0..processes.len())
        .map(|_| rand::rngs::ThreadRng::default())
        .collect();

    simulate(
        &mut filtration,
        &mut processes,
        &time_steps_ordered,
        &scenarios,
        &mut rngs,
    );
    let df: DataFrame = filtration.to_dataframe();
    Ok(PyDataFrame(df))
}

#[pymodule]
fn sde_sim_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_py, m)?)?;
    Ok(())
}
