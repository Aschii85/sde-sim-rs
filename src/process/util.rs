use crate::filtration::Filtration;
use crate::process::Process;
use crate::process::increment::{Incrementor, TimeIncrementor, WienerIncrementor};
use crate::process::ito::ItoProcess;
use crate::process::levy::LevyProcess;
use evalexpr;
use ordered_float::OrderedFloat;
use regex;

/// Parses a single SDE string into a LevyProcess.
fn parse_equation(equation: &str) -> Result<Box<dyn Process>, String> {
    use evalexpr::ContextWithMutableVariables;
    // 1. Get the process name (e.g., "X1" from "dX1 = ...")
    let name_re = regex::Regex::new(r"^\s*d([a-zA-Z0-9_]+)").unwrap();
    let name = name_re
        .captures(equation)
        .and_then(|caps| caps.get(1).map(|m| m.as_str().to_string()))
        .ok_or_else(|| "Could not parse process name (e.g., dX1) from equation.".to_string())?;

    // 2. Regex to capture coefficient expressions and their corresponding differentials (e.g., "dt", "dW1").
    let term_re =
        regex::Regex::new(r"\(((?:[^()]+|\((?R)\))*)\)\s*\*\s*(d[tWa-zA-Z0-9_]+)").unwrap();
    let mut coefficients: Vec<Box<dyn Fn(&Filtration, OrderedFloat<f64>, i32) -> f64>> = Vec::new();
    let mut incrementors: Vec<Box<dyn Incrementor>> = Vec::new();
    let mut is_ito_process: bool = true;
    for caps in term_re.captures_iter(equation) {
        let expression_str = caps.get(1).map_or("", |m| m.as_str()).to_string();
        let all_process_names: Vec<String> = regex::Regex::new(r"X\w*")
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
                incrementors.insert(0, Box::new(TimeIncrementor::new("t".to_string())));
            }
            _ if term_name.starts_with("dW") => {
                incrementors.push(Box::new(WienerIncrementor::new(
                    term_name.trim_start_matches('d').to_string(),
                )));
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
        Ok(Box::new(ItoProcess::new(name, coefficients, incrementors)?))
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
    if equations.is_empty() {
        return Err("No equations provided to parse.".to_string());
    }
    let mut processes: Vec<Box<dyn Process>> = Vec::new();
    for equation in equations {
        let process = parse_equation(equation)?;
        processes.push(process);
    }
    Ok(processes)
}
