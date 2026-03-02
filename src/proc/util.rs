use crate::func::Function;
use crate::proc::{AlgebraicProcess, LevyProcess, Process, ProcessUniverse, increment::*};
use ordered_float::OrderedFloat;
use std::collections::HashMap;

// Fixed nom imports
use nom::{
    bytes::complete::take_while1,
    character::complete::char,
    sequence::delimited,
    IResult,
    Parser, // Required to use .parse()
};

/// Parser that recognizes either a non-parenthesis string OR a nested balanced pair.
fn balanced_parens(input: &str) -> IResult<&str, String> {
    let mut result = String::new();
    let mut remaining = input;

    while !remaining.is_empty() {
        if remaining.starts_with('(') {
            // Use .parse(remaining) instead of calling the result of delimited like a function
            let (rest, inside) = delimited(char('('), balanced_parens, char(')'))
                .parse(remaining)?;
            result.push('(');
            result.push_str(&inside);
            result.push(')');
            remaining = rest;
        } else if remaining.starts_with(')') {
            break; 
        } else {
            let (rest, text) = take_while1(|c| c != '(' && c != ')')(remaining)?;
            result.push_str(text);
            remaining = rest;
        }
    }
    Ok((remaining, result))
}

/// Robustly extracts the content inside dN...( [content] )
fn extract_lambda(input: &str) -> Result<String, String> {
    let start_bracket = input.find('(').ok_or("Missing '(' in dN incrementor")?;
    let content_block = &input[start_bracket..];
    
    // Removed the turbofish ::<_> and used .parse()
    match delimited(char('('), balanced_parens, char(')')).parse(content_block) {
        Ok((_, content)) => Ok(content.trim().to_string()),
        Err(_) => Err(format!("Unbalanced parentheses in jump term: {}", input)),
    }
}

pub fn parse_equations(
    equations: &[String],
    timesteps: Vec<OrderedFloat<f64>>,
) -> Result<ProcessUniverse, String> {
    let mut stochastic_registry: HashMap<String, usize> = HashMap::new();
    let mut processes = Vec::with_capacity(equations.len());
    for eq in equations {
        processes.push(parse_single_equation(
            eq,
            timesteps.clone(),
            &mut stochastic_registry,
        )?);
    }
    Ok(ProcessUniverse::new(processes, stochastic_registry))
}

fn parse_single_equation(
    equation: &str,
    timesteps: Vec<OrderedFloat<f64>>,
    stochastic_registry: &mut HashMap<String, usize>,
) -> Result<Process, String> {
    let parts: Vec<&str> = equation.split('=').collect();
    if parts.len() != 2 { return Err("Missing '='".into()); }

    let lhs = parts[0].trim();
    let rhs = parts[1].trim();
    let process_name = lhs.strip_prefix('d').unwrap_or(lhs);
    
    if lhs.starts_with('d') {
        let mut coefficients = Vec::new();
        let mut incrementors = Vec::new();

        let mut current_rhs = rhs;
        while let Some(start_idx) = current_rhs.find('(') {
            // Corrected to use .parse()
            let (after_coeff, coeff_content) = delimited(char('('), balanced_parens, char(')'))
                .parse(&current_rhs[start_idx..])
                .map_err(|_| "Unbalanced parentheses in coefficient")?;
            
            let trimmed_after = after_coeff.trim_start();
            if !trimmed_after.starts_with('*') { break; }
            
            let after_star = trimmed_after[1..].trim_start();
            
            let (remaining, inc_str) = if after_star.starts_with("dN") {
                let d_start = after_star.find('(').ok_or("dN missing opening bracket")?;
                let (rest, _inside) = delimited(char('('), balanced_parens, char(')'))
                    .parse(&after_star[d_start..])
                    .map_err(|_| "Unbalanced parentheses in dN intensity")?;
                
                let full_inc = &after_star[..after_star.len() - rest.len()];
                (rest, full_inc)
            } else {
                let end = after_star.find(' ').unwrap_or(after_star.len());
                (&after_star[end..], &after_star[..end])
            };

            let coeff_fn = Box::new(Function::new(coeff_content.trim())
                .map_err(|e| format!("Math error in coefficient: {}", e))?);
            
            let incr = build_incrementor(inc_str, timesteps.clone(), stochastic_registry)?;
            
            coefficients.push(coeff_fn);
            incrementors.push(incr);
            current_rhs = remaining;
        }
        
        let levy_process = LevyProcess::new(process_name.to_string(), coefficients, incrementors)?;
        Ok(Process::Levy(Box::new(levy_process)))
    } else {
        let coeff_fn = Box::new(Function::new(rhs)?);
        Ok(Process::Algebraic(Box::new(AlgebraicProcess {
            name: process_name.to_string(),
            coefficients: vec![coeff_fn],
        })))
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

    let next_idx = registry.len();
    let incrementor_idx = *registry.entry(inc_str.to_string()).or_insert(next_idx);

    if inc_str.starts_with("dW") {
        Ok(Box::new(WienerIncrementor::new(incrementor_idx, timesteps)))
    } else if inc_str.starts_with("dN") {
        let lambda_expr = extract_lambda(inc_str)?;

        let lambda_fn = Box::new(Function::new(&lambda_expr)
            .map_err(|e| format!("Math error in jump lambda '{}': {}", lambda_expr, e))?);

        Ok(Box::new(PoissonJumpIncrementor::new(incrementor_idx, lambda_fn, timesteps)))
    } else {
        Err(format!("Unknown incrementor type: {}", inc_str))
    }
}