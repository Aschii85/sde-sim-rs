use crate::proc::{CoefficientFn, LevyProcess, increment::*};
use fasteval::{Compiler, Evaler, Instruction, Slab};
use ordered_float::OrderedFloat;
use regex::Regex;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
enum ResolvedVar {
    Time,
    Process(usize),
}

struct CompiledCoefficient {
    instruction: Instruction,
    slab: Slab,
    bound_vars: Vec<(String, ResolvedVar)>,
}

impl CompiledCoefficient {
    fn eval(&self, current_values: &[f64], t: f64) -> f64 {
        let mut cb = |name: &str, _args: Vec<f64>| -> Option<f64> {
            for (var_name, binding) in &self.bound_vars {
                if name == var_name {
                    return match binding {
                        ResolvedVar::Time => Some(t),
                        ResolvedVar::Process(idx) => Some(current_values[*idx]),
                    };
                }
            }
            None
        };
        self.instruction.eval(&self.slab, &mut cb).unwrap_or(0.0)
    }
}

/// Result of the parsing process
pub struct ProcessesUniverse {
    pub processes: Vec<Box<LevyProcess>>,
    pub num_stochastic_increments: usize,
}

pub fn parse_equations(
    equations: &[String],
    timesteps: Vec<OrderedFloat<f64>>,
) -> Result<ProcessesUniverse, String> {
    // Local registry to track stochastic incrementors (dW, dJ) per simulation run
    let mut stochastic_registry: HashMap<String, usize> = HashMap::new();

    let process_names: Vec<String> = equations
        .iter()
        .map(|eq| {
            eq.split('=')
                .next()
                .unwrap_or("")
                .trim()
                .trim_start_matches('d')
                .to_string()
        })
        .collect();

    let mut processes = Vec::with_capacity(equations.len());
    for eq in equations {
        processes.push(parse_single_equation(
            eq,
            &process_names,
            timesteps.clone(),
            &mut stochastic_registry,
        )?);
    }

    Ok(ProcessesUniverse {
        processes,
        num_stochastic_increments: stochastic_registry.len(),
    })
}

fn parse_single_equation(
    equation: &str,
    all_process_names: &[String],
    timesteps: Vec<OrderedFloat<f64>>,
    registry: &mut HashMap<String, usize>,
) -> Result<Box<LevyProcess>, String> {
    let parts: Vec<&str> = equation.split('=').collect();
    if parts.len() != 2 {
        return Err("Missing '='".into());
    }

    let name = parts[0].trim().trim_start_matches('d').to_string();
    let rhs = parts[1].trim();

    let mut coefficients: Vec<Arc<CoefficientFn>> = Vec::new();
    let mut incrementors: Vec<Box<dyn Incrementor>> = Vec::new();

    // Pattern to catch (coeff) * dIncr
    let term_pattern =
        Regex::new(r"\(([^)]*(?:\([^)]*\)[^)]*)*)\)\s*\*\s*(d[tWJ][\w\(\d\.\)]*)").unwrap();

    for cap in term_pattern.captures_iter(rhs) {
        let expr_str = &cap[1];
        let inc_str = &cap[2];

        // 1. Handle the math coefficient
        let mut slab = Slab::new();
        let parser = fasteval::Parser::new();
        let expr = parser
            .parse(expr_str, &mut slab.ps)
            .map_err(|e| format!("{:?}", e))?;
        let instruction = expr.from(&slab.ps).compile(&slab.ps, &mut slab.cs);

        let mut bound_vars = Vec::new();
        if expr_str.contains('t') {
            bound_vars.push(("t".to_string(), ResolvedVar::Time));
        }
        for (idx, p_name) in all_process_names.iter().enumerate() {
            if expr_str.contains(p_name) {
                bound_vars.push((p_name.clone(), ResolvedVar::Process(idx)));
            }
        }

        let compiled = Arc::new(CompiledCoefficient {
            instruction,
            slab,
            bound_vars,
        });
        let compiled_clone = Arc::clone(&compiled);
        let coeff_fn: Arc<CoefficientFn> = Arc::new(move |v, t| compiled_clone.eval(v, t.0));

        // 2. Handle the incrementor and indexing
        let incr = build_incrementor(inc_str, timesteps.clone(), registry)?;

        coefficients.push(coeff_fn);
        incrementors.push(incr);
    }

    Ok(Box::new(LevyProcess::new(
        name,
        coefficients,
        incrementors,
    )?))
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
        let lambda = re
            .captures(inc_str)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().parse::<f64>())
            .transpose()
            .map_err(|_| "Invalid lambda in dJ(...)".to_string())?
            .ok_or_else(|| "Lambda value missing in dJ".to_string())?;

        Ok(Box::new(JumpIncrementor::new(
            incrementor_idx,
            lambda,
            timesteps,
        )))
    } else {
        Err(format!("Unknown incrementor type: {}", inc_str))
    }
}
