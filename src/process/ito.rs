use meval::{Expr, Context};
use regex::Regex;
use std::collections::HashMap;

use crate::filtration::Filtration;
use crate::process::Process;
use crate::process::increment::Increment;

pub struct ItoProcess {
    name: String,
    coefficients: Vec<Box<dyn Fn(&Filtration, f64, i32) -> f64>>,
    incrementors: Vec<Increment>,
}

impl Process for ItoProcess {
    fn name(&self) -> &String {
        &self.name
    }

    // Update return type to match the new Box<dyn Fn> type
    fn coefficients(&self) -> &Vec<Box<dyn Fn(&Filtration, f64, i32) -> f64>> {
        &self.coefficients

    }

    fn incrementors(&self) -> &Vec<Increment> {
        &self.incrementors
    }
}

impl ItoProcess {
    pub fn new(
        name: String,
        drift: Box<dyn Fn(&Filtration, f64, i32) -> f64>,
        diffusion: Box<dyn Fn(&Filtration, f64, i32) -> f64>,
    ) -> Result<Self, String> {
        // Store the boxed closures directly
        let coefficients = vec![drift, diffusion]; // Clone to store in coefficients vector
        let incrementors = vec![
            Increment::Time,
            Increment::Wiener,
        ];
        Ok(ItoProcess {
            name,
            coefficients,
            incrementors
        })
    }

    pub fn from_string(name: String, equation: String) -> Result<Self, String> {
        let re = Regex::new(r"\(((?:[^()]+|\((?R)\))*)\)\s*\*\s*(d[tW]\w*)")
            .map_err(|e| format!("Failed to compile regex: {}", e))?;
        // Change HashMap value type to Box<dyn Fn>
        let mut terms_map: HashMap<String, Box<dyn Fn(&Filtration, f64, i32) -> f64>> = HashMap::new();
        for caps in re.captures_iter(&equation.clone()) {
            let expression_str = caps.get(1).map_or("", |m| m.as_str()).to_string();
            let expression = expression_str.parse::<Expr>()
                .map_err(|e| format!("Failed to parse expression '{}': {}", expression_str, e))?;
            let term = caps.get(2).map_or("", |m| m.as_str());
            let closure_name = name.clone();
            let exp_fun = move |f: &Filtration, t: f64, s: i32| {
                let mut context = Context::new();
                context.var("X", f.value(t, s, closure_name.to_string()));
                expression.eval_with_context(&context)
                    .expect(&format!("Failed to evaluate expression '{}' at t={}, s={}: {:?}", expression_str, t, s, expression))
            };
            // Box the closure before inserting into the HashMap
            terms_map.insert(term.to_string(), Box::new(exp_fun));
        }

        let drift_fn_box = terms_map.remove("dt")
            .ok_or("Drift term 'dt' not found in equation. Ensure equation includes a 'dt' term like '(expr) * dt'.".to_string())?;
        let diffusion_fn_box = terms_map.remove("dW")
            .ok_or("Diffusion term 'dW' not found in equation. Ensure equation includes a 'dW' term like '(expr) * dW'.".to_string())?;

        // Clone the Box<dyn Fn> before passing to new.
        // This is necessary because `new` takes ownership of the Box, and we need to keep the original in the map.
        ItoProcess::new(name, Box::new(drift_fn_box), Box::new(diffusion_fn_box))
    }

    pub fn drift(&self) -> &Box<dyn Fn(&Filtration, f64, i32) -> f64> {
        &self.coefficients[0]
    }

    pub fn diffusion(&self) -> &Box<dyn Fn(&Filtration, f64, i32) -> f64> {
        &self.coefficients[1]
    }
}
