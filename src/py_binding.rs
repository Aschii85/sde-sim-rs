use crate::filtration::Filtration;
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
    let processes =
        crate::proc::util::parse_equations(&processes_equations, time_steps_ordered.clone())
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Failed to parse process equations: {}",
                    e
                ))
            })?;

    let mut filtration = Filtration::new(
        processes,
        time_steps_ordered.clone(),
        (1..=scenarios).collect(),
        Some(initial_values),
    );

    // 2. Release the GIL so Rayon can scale across all cores
    py.allow_threads(|| {
        simulate(&mut filtration, &scheme, &rng_method);
    });

    // 3. Convert back to Polars (happens after threads join)
    let df: DataFrame = filtration.to_dataframe();
    Ok(PyDataFrame(df))
}

#[pymodule]
fn sde_sim_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_py, m)?)?;
    Ok(())
}
