use evalexpr;
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
        let coefficients = vec![drift, diffusion];
        let incrementors = vec![Increment::Time, Increment::Wiener];
        Ok(ItoProcess {
            name,
            coefficients,
            incrementors,
        })
    }

    /// Creates an `ItoProcess` instance by parsing a string equation.
    /// This method uses `mlua` (LuaJIT) to evaluate the expressions.
    ///
    /// The equation string should be in the format `"(expression_for_drift) * dt + (expression_for_diffusion) * dW"`.
    /// Variables `X`, `t`, and `s` can be used in the expressions, where:
    /// - `X` represents `f.value(t, s, self.name())`
    /// - `t` represents the current time
    /// - `s` represents the current state/sample index
    ///
    /// # Arguments
    /// * `name` - The name of the process.
    /// * `equation` - The string representation of the Ito process equation.
    ///
    /// # Returns
    /// A `Result` indicating success (`ItoProcess`) or failure (`String` error message).
    pub fn from_string(name: String, equation: String) -> Result<Self, String> {
        // Regex to capture expressions and their corresponding terms (e.g., "dt", "dW").
        // It looks for patterns like "(expression) * dt" or "(expression) * dW".
        let re = Regex::new(r"\(((?:[^()]+|\((?R)\))*)\)\s*\*\s*(d[tW]\w*)")
            .map_err(|e| format!("Failed to compile regex: {}", e))?;
        let mut terms_map: HashMap<String, Box<dyn Fn(&Filtration, f64, i32) -> f64>> =
            HashMap::new();
        for caps in re.captures_iter(&equation.clone()) {
            let _name = name.clone(); // Clone the name to use in the closure
            let expression_str = caps.get(1).map_or("", |m| m.as_str()).to_string();
            let expression =
                evalexpr::build_operator_tree::<evalexpr::DefaultNumericTypes>(&expression_str)
                    .unwrap();
            let exp_fun = Box::new(move |f: &Filtration, t: f64, s: i32| {
                let context: evalexpr::HashMapContext<evalexpr::DefaultNumericTypes> =
                    evalexpr::context_map! {
                        "X" => float f.value(t, s, _name.clone()),
                    }
                    .unwrap();
                expression.eval_float_with_context(&context).unwrap()
            });
            // Box the closure before inserting into the HashMap
            let term = caps.get(2).map_or("", |m| m.as_str()).to_string();
            terms_map.insert(term, exp_fun);
        }

        // Retrieve the drift and diffusion closures from the map.
        let drift_fn_box = terms_map.remove("dt")
            .ok_or("Drift term 'dt' not found in equation. Ensure equation includes a 'dt' term like '(expr) * dt'.".to_string())?;
        let diffusion_fn_box = terms_map.remove("dW")
            .ok_or("Diffusion term 'dW' not found in equation. Ensure equation includes a 'dW' term like '(expr) * dW'.".to_string())?;

        // Create and return the new ItoProcess instance.
        ItoProcess::new(name, drift_fn_box, diffusion_fn_box)
    }

    pub fn drift(&self) -> &Box<dyn Fn(&Filtration, f64, i32) -> f64> {
        &self.coefficients[0]
    }

    pub fn diffusion(&self) -> &Box<dyn Fn(&Filtration, f64, i32) -> f64> {
        &self.coefficients[1]
    }
}

/// Factory Functions for Specific Ito Processes
// These functions create specific instances of ItoProcess with predefined drift and diffusion coefficients.
// They are prefered  over using `from_string` for common processes, as they are more efficient
// computationally (around 2/4-times faster for meval/luajit).
pub fn geometric_brownian_motion(name: String, mu: f64, sigma: f64) -> ItoProcess {
    let _name = name.clone();
    let drift = Box::new(move |f: &Filtration, t: f64, s: i32| mu * f.value(t, s, _name.clone()));
    let _name = name.clone();
    let diffusion =
        Box::new(move |f: &Filtration, t: f64, s: i32| sigma * f.value(t, s, _name.clone()));
    ItoProcess::new(name.clone(), drift, diffusion).unwrap()
}
