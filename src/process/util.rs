use crate::filtration::Filtration;
use crate::process::CoefficientFn;
use crate::process::Process;
use crate::process::increment::{Incrementor, TimeIncrementor, WienerIncrementor};
use crate::process::ito::ItoProcess;
use crate::process::levy::LevyProcess;
use evalexpr;
use ordered_float::OrderedFloat;
use regex;

/// Parses a single Stochastic Differential Equation (SDE) string into a `Box<dyn Process>`.
///
/// This function takes a string representation of an SDE, extracts the process name,
/// parses its drift and diffusion coefficients, and identifies the corresponding
/// incrementors (e.g., `dt`, `dW`). It determines if the process is an Ito process
/// (only `dt` and `dW` terms) or a more general Levy process.
///
/// # Arguments
///
/// * `equation` - A string slice representing a single SDE. The expected format is
///   `"dX = (drift_expression)*dt + (diffusion_expression)*dW"`, where `dX` is the
///   differential of the process, `drift_expression` and `diffusion_expression` are
///   mathematical expressions that can include `t` (time) and other process names (e.g., `X1`).
///
/// # Returns
///
/// A `Result` which is:
/// * `Ok(Box<dyn Process>)` - A boxed trait object representing either an `ItoProcess`
///   or a `LevyProcess`, depending on the parsed terms.
/// * `Err(String)` - An error message if parsing fails (e.g., malformed equation,
///   unparsable expressions, unknown terms).
///
/// # Examples
///
/// ```
/// // Example of how this might be used (assuming `sde_sim_rs` is in scope)
/// // let eq = "dX = (0.5*X)*dt + (0.2*X)*dW1".to_string();
/// // let process = parse_equation(&eq).unwrap();
/// // println!("Parsed process name: {}", process.name());
/// ```
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
    let process_names_re = regex::Regex::new(r"X\w*").unwrap();
    let mut coefficients: Vec<Box<CoefficientFn>> = Vec::new();
    let mut incrementors: Vec<Box<dyn Incrementor>> = Vec::new();
    let mut is_ito_process: bool = true;
    for caps in term_re.captures_iter(equation) {
        let expression_str = caps.get(1).map_or("", |m| m.as_str()).to_string();
        let all_process_names: Vec<String> = process_names_re
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

/// Parses a single Stochastic Differential Equation (SDE) string into a `Box<dyn Process>`.
///
/// This function takes a string representation of an SDE, extracts the process name,
/// parses its drift and diffusion coefficients, and identifies the corresponding
/// incrementors (e.g., `dt`, `dW`). It determines if the process is an Ito process
/// (only `dt` and `dW` terms) or a more general Levy process.
///
/// # Equation Format
///
/// The equation must be in the form: `d{ProcessName} = ({expression})*d{Incrementor} + ...`.
///
/// * **`{ProcessName}`**: The name of the process (e.g., `X1`, `X_t`).
/// * **`{expression}`**: A mathematical expression for the coefficient. This expression
///   can use the current time (`t`) and the values of other processes (e.g., `X1`).
/// * **`{Incrementor}`**: The differential term. Currently, only `dt` (for the drift term)
///   and `dW` (for Wiener processes, e.g., `dW1`, `dW2`) are supported.
///
/// # Examples
///
/// * **Geometric Brownian Motion:** `dX = (0.5 * X) * dt + (0.2 * X) * dW1`
/// * **Ornstein-Uhlenbeck Process:** `dX = (theta * (mu - X)) * dt + (sigma) * dW1`
/// * **Two-Factor Model:** `dX1 = (alpha) * dt + (beta * X2) * dW1`
///
/// # Arguments
///
/// * `equation` - A string slice representing a single SDE.
///
/// # Returns
///
/// A `Result` which is:
/// * `Ok(Box<dyn Process>)` - A boxed trait object representing an `ItoProcess`
///   or a `LevyProcess`.
/// * `Err(String)` - An error message if parsing fails.
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
