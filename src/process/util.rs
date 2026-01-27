use crate::filtration::Filtration;
use crate::process::{CoefficientFn, Process, increment::*, levy::LevyProcess};
use fasteval::{Compiler, Evaler, Instruction, Slab};
use ordered_float::OrderedFloat;
use regex::Regex;
use std::sync::Arc;

/// Optimized coefficient that bypasses string lookups in the hot loop.
struct CompiledCoefficient {
    instruction: Instruction,
    slab: Slab,
    // (Variable name in SDE string, resolved index in Filtration)
    var_index_map: Vec<(String, usize)>,
}

impl CompiledCoefficient {
    fn eval(&self, f: &Filtration, t: OrderedFloat<f64>, s: i32) -> f64 {
        let mut ns = |name: &str, _args: Vec<f64>| -> Option<f64> {
            if name == "t" {
                return Some(t.0);
            }

            // Linear search of dependencies: Very fast for N < 5
            for (var_name, idx) in &self.var_index_map {
                if name == var_name {
                    return f.value_by_index(t, s, *idx).ok();
                }
            }
            None
        };
        self.instruction.eval(&self.slab, &mut ns).unwrap_or(0.0)
    }
}

pub fn parse_equations(
    equations: &[String],
    filtration: &Filtration,
) -> Result<Vec<Box<dyn Process>>, String> {
    let mut processes = Vec::with_capacity(equations.len());
    for eq in equations {
        processes.push(parse_equation(eq, filtration)?);
    }
    Ok(processes)
}

pub fn parse_equation(equation: &str, filtration: &Filtration) -> Result<Box<dyn Process>, String> {
    let parts: Vec<&str> = equation.split('=').collect();
    if parts.len() != 2 {
        return Err("Missing '='".into());
    }

    let name = parts[0].trim().trim_start_matches('d').to_string();
    let rhs = parts[1].trim();

    let mut coefficients: Vec<Box<CoefficientFn>> = Vec::new();
    let mut incrementors: Vec<Box<dyn Incrementor>> = Vec::new();

    let term_pattern =
        Regex::new(r"\(([^)]*(?:\([^)]*\)[^)]*)*)\)\s*\*\s*(d[tWJ][\w\(\d\.\)]*)").unwrap();
    let var_pattern = Regex::new(r"[XYZ]\w*").unwrap();

    for cap in term_pattern.captures_iter(rhs) {
        let expr_str = &cap[1];
        let inc_str = &cap[2];

        let mut slab = Slab::new();
        let instruction = fasteval::Parser::new()
            .parse(expr_str, &mut slab.ps)
            .map_err(|e| format!("Fasteval error: {:?}", e))?
            .from(&slab.ps)
            .compile(&slab.ps, &mut slab.cs);

        // MAP STRINGS TO INDICES ONCE
        let mut var_index_map = Vec::new();
        for m in var_pattern.find_iter(expr_str) {
            let var_name = m.as_str();
            if let Some(idx) = filtration.get_process_index(var_name) {
                var_index_map.push((var_name.to_string(), idx));
            }
        }

        let compiled = Arc::new(CompiledCoefficient {
            instruction,
            slab,
            var_index_map,
        });
        let compiled_clone = Arc::clone(&compiled);
        let coeff_fn: Box<CoefficientFn> = Box::new(move |f, t, s| compiled_clone.eval(f, t, s));

        coefficients.push(coeff_fn);
        incrementors.push(parse_incrementor(inc_str)?);
    }

    Ok(Box::new(LevyProcess::new(
        name,
        coefficients,
        incrementors,
    )?))
}

fn parse_incrementor(s: &str) -> Result<Box<dyn Incrementor>, String> {
    if s == "dt" {
        Ok(Box::new(TimeIncrementor::new()))
    } else if s.starts_with("dW") {
        Ok(Box::new(WienerIncrementor::new(s.to_string())))
    } else if s.starts_with("dJ") {
        let re = Regex::new(r"dJ\w*\(([^)]+)\)").unwrap();
        let val = re
            .captures(s)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().parse::<f64>())
            .transpose()
            .map_err(|_| "Bad lambda".to_string())?
            .ok_or_else(|| "Lambda missing".to_string())?;
        Ok(Box::new(JumpIncrementor::new(s.to_string(), val)))
    } else {
        Err(format!("Unknown: {}", s))
    }
}
