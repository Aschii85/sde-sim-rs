use crate::filtration::Filtration;
use crate::process::CoefficientFn;
use crate::process::Process;
use crate::process::increment::{Incrementor, JumpIncrementor, TimeIncrementor, WienerIncrementor};
use crate::process::levy::LevyProcess;
use evalexpr;
use ordered_float::OrderedFloat;
use regex;

fn parse_equation(equation: &str) -> Result<Box<dyn Process>, String> {
    // 1. Get the process name
    let name_re = regex::Regex::new(r"^\s*d([a-zA-Z0-9_]+)").unwrap();
    let name = name_re
        .captures(equation)
        .and_then(|caps| caps.get(1).map(|m| m.as_str().to_string()))
        .ok_or_else(|| "Could not parse process name (e.g., dX1) from equation.".to_string())?;

    let process_names_re = regex::Regex::new(r"X\w*").unwrap();
    let mut coefficients: Vec<Box<CoefficientFn>> = Vec::new();
    let mut incrementors: Vec<Box<dyn Incrementor>> = Vec::new();
    let mut coefficient_expressions: Vec<String> = Vec::new();

    // Remove the equation prefix "dX = "
    let rhs = equation
        .split('=')
        .nth(1)
        .ok_or("No '=' found in equation")?
        .trim();

    // Split by '+' to get individual terms, but need to be careful with nested parens
    let mut current_term = String::new();
    let mut paren_depth = 0;

    for ch in rhs.chars() {
        match ch {
            '(' => {
                paren_depth += 1;
                current_term.push(ch);
            }
            ')' => {
                paren_depth -= 1;
                current_term.push(ch);
            }
            '+' if paren_depth == 0 => {
                // End of term
                if !current_term.trim().is_empty() {
                    parse_single_term(
                        &current_term,
                        &mut coefficients,
                        &mut incrementors,
                        &mut coefficient_expressions,
                        &process_names_re,
                    )?;
                }
                current_term.clear();
            }
            _ => current_term.push(ch),
        }
    }

    // Don't forget the last term
    if !current_term.trim().is_empty() {
        parse_single_term(
            &current_term,
            &mut coefficients,
            &mut incrementors,
            &mut coefficient_expressions,
            &process_names_re,
        )?;
    }

    let mut process = Box::new(LevyProcess::new(name, coefficients, incrementors)?);

    let incrementor_names: Vec<String> = process
        .incrementors()
        .iter()
        .map(|inc| inc.name().clone())
        .collect();

    println!(
        "Parsed process: name='{}', coefficients={:?}, incrementors={:?}",
        process.name(),
        coefficient_expressions,
        incrementor_names
    );
    Ok(process)
}

fn parse_single_term(
    term: &str,
    coefficients: &mut Vec<Box<CoefficientFn>>,
    incrementors: &mut Vec<Box<dyn Incrementor>>,
    coefficient_expressions: &mut Vec<String>,
    process_names_re: &regex::Regex,
) -> Result<(), String> {
    use evalexpr::ContextWithMutableVariables;
    let term = term.trim();

    // Extract coefficient expression (balanced parentheses)
    let mut paren_depth = 0;
    let mut coeff_start = 0;
    let mut coeff_end = 0;
    let mut chars = term.char_indices();

    while let Some((i, ch)) = chars.next() {
        match ch {
            '(' => {
                if paren_depth == 0 {
                    coeff_start = i + 1;
                }
                paren_depth += 1;
            }
            ')' => {
                paren_depth -= 1;
                if paren_depth == 0 {
                    coeff_end = i;
                    break;
                }
            }
            _ => {}
        }
    }

    let expression_str = term[coeff_start..coeff_end].trim().to_string();
    let rest = term[coeff_end + 1..].trim();

    // Extract incrementor term (dt, dW1, dJ1(0.1), etc.)
    let term_re = regex::Regex::new(r"^.*?(d[tWJ][a-zA-Z0-9_]*)(?:\(([^)]+)\))?").unwrap();
    let caps = term_re
        .captures(rest)
        .ok_or_else(|| format!("Could not parse incrementor from term: {}", rest))?;

    let term_name = caps.get(1).unwrap().as_str();
    let param_str = caps.get(2).map(|m| m.as_str());

    // Build coefficient closure
    let all_process_names: Vec<String> = process_names_re
        .find_iter(&expression_str)
        .map(|m| m.as_str().to_string())
        .collect();
    let expression =
        evalexpr::build_operator_tree::<evalexpr::DefaultNumericTypes>(&expression_str)
            .map_err(|e| format!("Failed to parse expression '{}': {}", expression_str, e))?;

    let expression_str_clone = expression_str.clone();
    let coeff_fn = Box::new(move |f: &Filtration, t: OrderedFloat<f64>, s: i32| {
        use evalexpr::Value;
        let mut context = evalexpr::HashMapContext::new();
        context
            .set_value("t".to_string(), Value::from_float(t.0))
            .ok();
        for process_name in &all_process_names {
            if let Ok(val) = f.value(t, s, process_name) {
                context
                    .set_value(process_name.clone(), Value::from_float(val))
                    .ok();
            }
        }
        match expression.eval_with_context(&context) {
            Ok(val) => {
                let result = match val {
                    evalexpr::Value::Float(f) => f,
                    evalexpr::Value::Int(i) => i as f64,
                    _ => 0.0,
                };
                result
            }
            Err(e) => {
                eprintln!(
                    "Coefficient evaluation error for '{}': {}",
                    expression_str_clone, e
                );
                0.0
            }
        }
    });

    coefficients.push(coeff_fn);
    coefficient_expressions.push(format!("({}) * {}", expression_str, term_name));

    // Create incrementor
    let incrementor: Box<dyn Incrementor> = match term_name {
        "dt" => Box::new(TimeIncrementor::new()),
        _ if term_name.starts_with("dW") => Box::new(WienerIncrementor::new(term_name.to_string())),
        _ if term_name.starts_with("dJ") => {
            let lambda = param_str
                .and_then(|s| s.parse::<f64>().ok())
                .ok_or_else(|| {
                    format!(
                        "Jump term '{}' requires a numeric lambda, e.g., dJ(0.5)",
                        term_name
                    )
                })?;
            Box::new(JumpIncrementor::new(term_name.to_string(), lambda))
        }
        _ => {
            return Err(format!(
                "Unsupported term '{}' found in equation.",
                term_name
            ));
        }
    };
    incrementors.push(incrementor);

    Ok(())
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
