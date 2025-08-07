use crate::filtration::Filtration;
use crate::process::Process;
use crate::process::increment::{Incrementor, TimeIncrementor, WienerIncrementor};
use crate::process::ito::ItoProcess;
use crate::process::levy::LevyProcess;
use evalexpr;
use evalexpr::ContextWithMutableVariables;
use lazy_static::lazy_static;
use ordered_float::OrderedFloat;
use regex::Regex;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

lazy_static! {
    static ref WIENER_PROCESSES: Mutex<HashMap<String, Arc<Mutex<WienerIncrementor>>>> =
        Mutex::new(HashMap::new());
}

/// Parses a single SDE string into a LevyProcess.
/// Assumes that WIENER_PROCESSES map has been pre-populated.
fn parse_equation(equation: &str) -> Result<Box<dyn Process>, String> {
    // 1. Get the process name (e.g., "X1" from "dX1 = ...")
    let name_re = Regex::new(r"^\s*d([a-zA-Z0-9_]+)").unwrap();
    let name = name_re
        .captures(equation)
        .and_then(|caps| caps.get(1).map(|m| m.as_str().to_string()))
        .ok_or_else(|| "Could not parse process name (e.g., dX1) from equation.".to_string())?;

    // 2. Regex to capture coefficient expressions and their corresponding differentials (e.g., "dt", "dW1").
    let term_re = Regex::new(r"\(((?:[^()]+|\((?R)\))*)\)\s*\*\s*(d[tWa-zA-Z0-9_]+)").unwrap();
    let wiener_map = WIENER_PROCESSES.lock().unwrap();
    let mut coefficients: Vec<Box<dyn Fn(&Filtration, OrderedFloat<f64>, i32) -> f64>> = Vec::new();
    let mut incrementors: Vec<Box<dyn Incrementor>> = Vec::new();
    let mut wiener_incrementors: Vec<Box<Arc<Mutex<WienerIncrementor>>>> = Vec::new();
    let mut is_ito_process: bool = true;
    let mut drift_term_idx: i32 = -1;

    for (idx, caps) in term_re.captures_iter(equation).enumerate() {
        let expression_str = caps.get(1).map_or("", |m| m.as_str()).to_string();
        let all_process_names: Vec<String> = Regex::new(r"X\w*")
            .unwrap()
            .find_iter(&expression_str)
            .map(|m| m.as_str().to_string())
            .collect();
        // Build the closure for the coefficient function
        let expression =
            evalexpr::build_operator_tree::<evalexpr::DefaultNumericTypes>(&expression_str)
                .map_err(|e| format!("Failed to parse expression '{}': {}", expression_str, e))?;
        // Create a closure that evaluates the expression with the current filtration and time
        let coeff_fn = Box::new(move |f: &Filtration, t: OrderedFloat<f64>, s: i32| {
            use evalexpr::Value;
            let mut context = evalexpr::HashMapContext::new();
            context
                .set_value("t".to_string(), Value::from_float(t.0))
                .ok();
            for process_name in &all_process_names {
                context
                    .set_value(
                        process_name.clone(),
                        Value::from_float(f.value(t, s, process_name).unwrap()),
                    )
                    .ok();
            }
            expression.eval_float_with_context(&context).unwrap_or(0.0)
        });
        coefficients.push(coeff_fn);
        // Determine the type of term (drift, diffusion, etc...)
        let term_name = caps.get(2).map_or("", |m| m.as_str()).to_string();
        match term_name.as_ref() {
            "dt" => {
                // Drift term
                incrementors.push(Box::new(TimeIncrementor::new()));
                drift_term_idx = idx as i32;
            }
            _ if term_name.starts_with("dW") => {
                // Diffusion term
                let wiener_name = term_name.trim_start_matches('d').to_string();
                let wiener_process = wiener_map.get(&wiener_name).ok_or_else(|| {
                    format!(
                        "Wiener process '{}' not found. Ensure it is declared in one of the equations.",
                        wiener_name
                    )
                })?;
                // Share the Arc<Mutex<WienerIncrementor>> for shared ownership
                incrementors.push(Box::new(Arc::clone(wiener_process)));
                wiener_incrementors.push(Box::new(Arc::clone(wiener_process)));
            }
            _ => {
                is_ito_process = false;
                // return Err(format!(
                //     "Unsupported term '{}' found in equation.",
                //     term_name
                // ));
            }
        }
    }
    if is_ito_process {
        let drift_term = if drift_term_idx != -1 {
            coefficients.remove(drift_term_idx as usize)
        } else {
            Box::new(
                |_filtration: &Filtration, // Arguments are prefixed with '_' because they are unused
                 _time: OrderedFloat<f64>,
                 _index: i32|
                 -> f64 {
                    0.0 // The closure always returns 0.0
                },
            )
        };
        Ok(Box::new(ItoProcess::new(
            name,
            drift_term, // The first coefficient is the drift
            coefficients,
            wiener_incrementors,
        )?))
    } else {
        Ok(Box::new(LevyProcess::new(
            name,
            coefficients,
            incrementors,
        )?))
    }
}

/// Parses a slice of equation strings, discovers all Wiener processes,
/// and returns a vector of the resulting LevyProcesses.
pub fn parse_equations(equations: &[String]) -> Result<Vec<Box<dyn Process>>, String> {
    // 1. First pass: discover all unique Wiener processes and populate the global map
    let dw_regex = Regex::new(r"d(W[a-zA-Z0-9_]*)").unwrap();
    let mut wiener_map = WIENER_PROCESSES.lock().unwrap();

    for equation in equations {
        for cap in dw_regex.captures_iter(equation) {
            // cap[0] is the full match like "dW1", cap[1] is just "W1"
            let var_name = cap.get(1).unwrap().as_str().to_string();
            wiener_map
                .entry(var_name)
                .or_insert_with(|| Arc::new(Mutex::new(WienerIncrementor::new())));
        }
    }
    // Drop the lock so parse_equation can acquire it
    drop(wiener_map);

    // 2. Second pass: parse each equation individually
    let mut processes: Vec<Box<dyn Process>> = Vec::new();
    for equation in equations {
        let process = parse_equation(equation)?;
        processes.push(process);
    }

    Ok(processes)
}
