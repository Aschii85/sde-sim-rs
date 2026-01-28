use crate::process::{CoefficientFn, LevyProcess, increment::*};
use fasteval::{Compiler, Evaler, Instruction, Slab};
use ordered_float::OrderedFloat;
use regex::Regex;
use std::sync::Arc;
use std::sync::Mutex;

enum ResolvedVar {
    Time,
    Process(usize),
}

struct CompiledCoefficient {
    instruction: Instruction,
    slab: Slab,
    // Store which names we found and where they point
    bound_vars: Vec<(String, ResolvedVar)>,
}

// Global mutable vector wrapped in a Mutex
static _INCREMENTORS: Mutex<Vec<String>> = Mutex::new(Vec::new());

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

pub fn parse_equations(
    equations: &[String],
    timesteps: Vec<OrderedFloat<f64>>,
) -> Result<Vec<Box<LevyProcess>>, String> {
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
        processes.push(parse_equation(eq, &process_names, timesteps.clone())?);
    }
    Ok(processes)
}

pub fn parse_equation(
    equation: &str,
    all_process_names: &[String],
    timesteps: Vec<OrderedFloat<f64>>,
) -> Result<Box<LevyProcess>, String> {
    let parts: Vec<&str> = equation.split('=').collect();
    if parts.len() != 2 {
        return Err("Missing '='".into());
    }

    let name = parts[0].trim().trim_start_matches('d').to_string();
    let rhs = parts[1].trim();

    let mut coefficients: Vec<Arc<CoefficientFn>> = Vec::new();
    let mut incrementors: Vec<Box<dyn Incrementor>> = Vec::new();

    let term_pattern =
        Regex::new(r"\(([^)]*(?:\([^)]*\)[^)]*)*)\)\s*\*\s*(d[tWJ][\w\(\d\.\)]*)").unwrap();

    for cap in term_pattern.captures_iter(rhs) {
        let expr_str = &cap[1];
        let inc_str = &cap[2];

        let mut slab = Slab::new();
        let parser = fasteval::Parser::new();
        let expr = parser
            .parse(expr_str, &mut slab.ps)
            .map_err(|e| format!("{:?}", e))?;
        let instruction = expr.from(&slab.ps).compile(&slab.ps, &mut slab.cs);

        // Map any variables found in the string to our indices
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

        let coeff_fn: Arc<CoefficientFn> = Arc::new(move |f, t, t_idx, s_idx| {
            compiled_clone.eval(f.get_processes_slice(s_idx, t_idx), t.0)
        });

        coefficients.push(coeff_fn);
        incrementors.push(parse_incrementor(inc_str, timesteps.clone())?);
    }

    Ok(Box::new(LevyProcess::new(
        name,
        coefficients,
        incrementors,
    )?))
}

fn parse_incrementor(
    inc_str: &str,
    timesteps: Vec<OrderedFloat<f64>>,
) -> Result<Box<dyn Incrementor>, String> {
    let incrementor_idx = if inc_str != "dt" {
        let mut list = _INCREMENTORS.lock().unwrap();
        if let Some(index) = list.iter().position(|s| s == inc_str) {
            index
        } else {
            list.push(inc_str.to_string());
            list.len() - 1
        }
    } else {
        0
    };

    if inc_str == "dt" {
        Ok(Box::new(TimeIncrementor::new(timesteps)))
    } else if inc_str.starts_with("dW") {
        Ok(Box::new(WienerIncrementor::new(incrementor_idx, timesteps)))
    } else if inc_str.starts_with("dJ") {
        let re = Regex::new(r"dJ\w*\(([^)]+)\)").unwrap();
        let val = re
            .captures(inc_str)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().parse::<f64>())
            .transpose()
            .map_err(|_| "Bad lambda".to_string())?
            .ok_or_else(|| "Lambda missing".to_string())?;
        Ok(Box::new(JumpIncrementor::new(
            incrementor_idx,
            val,
            timesteps,
        )))
    } else {
        Err(format!("Unknown incrementor: {}", inc_str))
    }
}

pub fn num_incrementors() -> usize {
    let list = _INCREMENTORS.lock().unwrap();
    list.len()
}
