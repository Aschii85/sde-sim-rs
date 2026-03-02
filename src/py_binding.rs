use crate::sim::simulate;
use ordered_float::OrderedFloat;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use std::collections::HashMap;

#[pyfunction]
#[pyo3(name = "simulate")]
pub fn simulate_py(
    py: Python<'_>, // Added this to handle GIL release
    processes_equations: Vec<String>,
    time_steps: Vec<f64>,
    scenarios: i32,
    initial_values: HashMap<String, f64>,
    rng_method: String,
    scheme: String,
) -> PyResult<PyDataFrame> {
    let time_steps_ordered: Vec<OrderedFloat<f64>> =
        time_steps.iter().copied().map(OrderedFloat).collect();

    // 1. Heavy parsing done while holding the GIL (purely CPU bound, usually fast)
    let mut processes = crate::proc::util::parse_equations(
        &processes_equations,
        time_steps_ordered.clone(),
    )
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Failed to parse process equations: {}",
            e
        ))
    })?;

    // build the universe and hand off to the shared simulation routine
    // (the simulator takes ownership of the process universe and time vector)
    let df: LazyFrame = py
        .allow_threads(|| {
            simulate(
                &mut processes,
                time_steps_ordered.clone(),
                initial_values,
                scenarios as u64,
                &scheme,
                &rng_method,
            )
        });

    Ok(PyDataFrame(df.collect().unwrap()))
}

#[pymodule]
fn sde_sim_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_py, m)?)?;
    Ok(())
}
