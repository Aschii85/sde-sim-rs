use crate::filtration::Filtration;
use crate::process::increment::{Incrementor, TimeIncrementor, WienerIncrementor};
use crate::process::{Process, levy::LevyProcess};
use evalexpr;
use evalexpr::ContextWithMutableVariables;
use lazy_static::lazy_static;
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
fn parse_equation(equation: &str) -> Result<LevyProcess, String> {
    // 1. Get the process name (e.g., "X1" from "dX1 = ...")
    let name_re = Regex::new(r"^\s*d([a-zA-Z0-9_]+)").unwrap();
    let name = name_re
        .captures(equation)
        .and_then(|caps| caps.get(1).map(|m| m.as_str().to_string()))
        .ok_or_else(|| "Could not parse process name (e.g., dX1) from equation.".to_string())?;

    // 2. Regex to capture coefficient expressions and their corresponding differentials (e.g., "dt", "dW1").
    let term_re = Regex::new(r"\(((?:[^()]+|\((?R)\))*)\)\s*\*\s*(d[tWa-zA-Z0-9_]+)").unwrap();

    let mut coefficients: Vec<Box<dyn Fn(&Filtration, f64, i32) -> f64>> = Vec::new();
    let mut term_names: Vec<String> = Vec::new();

    for caps in term_re.captures_iter(equation) {
        let expression_str = caps.get(1).map_or("", |m| m.as_str()).to_string();
        let term_name = caps.get(2).map_or("", |m| m.as_str()).to_string();

        // Build the closure for the coefficient function
        let expression =
            evalexpr::build_operator_tree::<evalexpr::DefaultNumericTypes>(&expression_str)
                .map_err(|e| format!("Failed to parse expression '{}': {}", expression_str, e))?;

        let process_name = name.clone();
        let coeff_fn = Box::new(move |f: &Filtration, t: f64, s: i32| {
            use evalexpr::Value;
            let mut context = evalexpr::HashMapContext::new();
            // Set the variable for this process (e.g., X1, X2, ...)
            context
                .set_value(
                    process_name.clone(),
                    Value::from_float(f.value(t, s, process_name.clone())),
                )
                .ok();
            context
                .set_value("t".to_string(), Value::from_float(t))
                .ok();
            expression.eval_float_with_context(&context).unwrap_or(0.0)
        });

        coefficients.push(coeff_fn);
        term_names.push(term_name);
    }

    if term_names.is_empty() {
        return Err(format!(
            "No terms like '(expr) * dt' or '(expr) * dW1' found in equation for {}",
            name
        ));
    }

    // 3. Build the corresponding incrementors vector
    let mut incrementors: Vec<Box<dyn Incrementor>> = Vec::new();
    let wiener_map = WIENER_PROCESSES.lock().unwrap();

    for term_name in &term_names {
        if term_name == "dt" {
            // Drift term
            incrementors.push(Box::new(TimeIncrementor::new()));
        } else if term_name.starts_with("dW") {
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
        } else {
            return Err(format!(
                "Unsupported term '{}' found in equation.",
                term_name
            ));
        }
    }

    // 4. Create the final LevyProcess
    LevyProcess::new(name, coefficients, incrementors)
}

/// Parses a slice of equation strings, discovers all Wiener processes,
/// and returns a vector of the resulting LevyProcesses.
pub fn parse_equations(equations: &[String]) -> Result<Vec<LevyProcess>, String> {
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
    let mut processes: Vec<LevyProcess> = Vec::new();
    for equation in equations {
        let process = parse_equation(equation)?;
        processes.push(process);
    }

    Ok(processes)
}
