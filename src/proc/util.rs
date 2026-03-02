use crate::func::Function;
use crate::proc::{AlgebraicProcess, LevyProcess, Process, ProcessUniverse, increment::*};
use ordered_float::OrderedFloat;
use regex::Regex;
use std::collections::HashMap;


/// Result of the parsing process
pub fn parse_equations(
    equations: &[String],
    process_names: &[String],
    timesteps: Vec<OrderedFloat<f64>>,
) -> Result<ProcessUniverse, String> {
    // Local registry to track stochastic incrementors (dW, dJ) per simulation run
    let mut stochastic_registry: HashMap<String, usize> = HashMap::new();
    let process_registry: HashMap<String, usize> = process_names
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.clone(), idx))
        .collect();
    let mut processes = Vec::with_capacity(equations.len());
    for eq in equations {
        processes.push(parse_single_equation(
            eq,
            process_names,
            timesteps.clone(),
            &mut stochastic_registry,
        )?);
    }

    Ok(ProcessUniverse {
        processes,
        process_registry,
        stochastic_registry,
    })
}

fn parse_single_equation(
    equation: &str,
    all_process_names: &[String],
    timesteps: Vec<OrderedFloat<f64>>,
    registry: &mut HashMap<String, usize>,
) -> Result<Process, String> {
    let parts: Vec<&str> = equation.split('=').collect();
    if parts.len() != 2 {
        return Err("Missing '='".into());
    }

    let lhs = parts[0].trim();
    let rhs = parts[1].trim();

    let process_name = lhs.strip_prefix('d').unwrap_or(lhs);
    if process_name.is_empty() || !all_process_names.contains(&process_name.to_string()) {
        return Err(format!(
            "Invalid process name '{}' in equation '{}', not specified in initial values!",
            process_name, equation
        ));
    }

    if lhs.starts_with('d') {
        // Levy process: each term in rhs has the form `(expr) * dIncr`.
        let mut coefficients: Vec<Box<Function>> = Vec::new();
        let mut incrementors: Vec<Box<dyn Incrementor>> = Vec::new();
        // Pattern to catch (coeff) * dIncr
        // Matches dt, dW<name>, or dJ<name>(...) with any content inside parentheses
        let term_pattern =
            Regex::new(r"\(([^)]*(?:\([^)]*\)[^)]*)*)\)\s*\*\s*(d(?:[tW]\w*|J\w*\([^)]*\)))").unwrap();
        for cap in term_pattern.captures_iter(rhs) {
            let expr_str = &cap[1];
            let inc_str = &cap[2];
            // create a Function object for the coefficient expression
            let coeff_fn = Box::new(Function::new(expr_str)?);
            // handle the incrementor and indexing
            let incr = build_incrementor(inc_str, timesteps.clone(), registry)?;
            coefficients.push(coeff_fn);
            incrementors.push(incr);
        }
        let levy_process = LevyProcess::new(process_name.to_string(), coefficients, incrementors)?;
        Ok(Process::Levy(Box::new(levy_process)))
    } else {
        // Algebraic process just has a single deterministic coefficient
        let coeff_fn = Box::new(Function::new(rhs)?);
        let algebraic_process = AlgebraicProcess {
            name: process_name.to_string(),
            coefficients: vec![coeff_fn],
        };
        Ok(Process::Algebraic(Box::new(algebraic_process)))
    }
}

fn build_incrementor(
    inc_str: &str,
    timesteps: Vec<OrderedFloat<f64>>,
    registry: &mut HashMap<String, usize>,
) -> Result<Box<dyn Incrementor>, String> {
    if inc_str == "dt" {
        return Ok(Box::new(TimeIncrementor::new(timesteps)));
    }
    // Assign a 0-based index for stochastic dimensions only
    let next_idx = registry.len();
    let incrementor_idx = *registry.entry(inc_str.to_string()).or_insert(next_idx);
    if inc_str.starts_with("dW") {
        Ok(Box::new(WienerIncrementor::new(incrementor_idx, timesteps)))
    } else if inc_str.starts_with("dJ") {
        let re = Regex::new(r"dJ\w*\(([^)]+)\)").unwrap();
        let lambda_expr = re
            .captures(inc_str)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str())
            .ok_or_else(|| "Lambda expression missing in dJ".to_string())?;
        // create a Function wrapper for the lambda expression
        let lambda_fn = Box::new(Function::new(lambda_expr)?);
        Ok(Box::new(JumpIncrementor::new(
            incrementor_idx,
            lambda_fn,
            timesteps,
        )))
    } else {
        Err(format!("Unknown incrementor type: {}", inc_str))
    }
}
