use mlua::{Lua, Function};
use regex::Regex;
use std::collections::HashMap;
use std::rc::Rc; // For shared ownership of the Lua state in a single-threaded context

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

        // Create a single Lua instance for this ItoProcess.
        // Rc is used to allow multiple closures to share ownership of the Lua state
        // in a single-threaded context.
        let lua = Rc::new(Lua::new());
        let mut terms_map: HashMap<String, Box<dyn Fn(&Filtration, f64, i32) -> f64>> = HashMap::new();

        // Iterate over all matches in the equation string.
        for caps in re.captures_iter(&equation) {
            let expression_str = caps.get(1).map_or("", |m| m.as_str()).to_string();
            let term = caps.get(2).map_or("", |m| m.as_str());
            let closure_process_name = name.clone(); // Clone the process name for use in the closure
            let current_lua = Rc::clone(&lua); // Clone the Rc to capture in the upcoming closure

            // Compile the Lua expression into a `mlua::Function` once.
            // This function will be reused for every evaluation, improving performance.
            // We wrap the expression in 'return' to make it a valid Lua chunk that returns a value.
            let compiled_function: Function = current_lua.load(&format!("return {}", expression_str))
                .into_function()
                .map_err(|e| format!("Failed to compile Lua expression '{}': {}", expression_str, e))?;

            // Create a boxed closure for each term (drift or diffusion).
            // This closure will be responsible for evaluating the Lua expression
            // with the current filtration, time, and state.
            let exp_fun = move |f: &Filtration, t: f64, s: i32| {
                // Get a reference to the shared Lua state.
                let lua_ref = &current_lua;

                // Set the variables 'X', 't', and 's' in the Lua global environment.
                // The compiled Lua function will then be able to access these variables.
                lua_ref.globals().set("X", f.value(t, s, closure_process_name.clone()))
                    .expect("Failed to set X variable in Lua");
                lua_ref.globals().set("t", t)
                    .expect("Failed to set t variable in Lua");
                lua_ref.globals().set("s", s)
                    .expect("Failed to set s variable in Lua");

                // Call the pre-compiled Lua function.
                // It takes no direct arguments as it relies on the global variables set above.
                match compiled_function.call::<f64>(()) { // Corrected call syntax
                    Ok(result) => result,
                    Err(e) => {
                        // Log the error and panic, consistent with the original `meval` behavior.
                        eprintln!("Failed to evaluate Lua expression '{}' at t={}, s={}: {}", expression_str, t, s, e);
                        panic!("Failed to evaluate Lua expression: {}", e);
                    }
                }
            };
            terms_map.insert(term.to_string(), Box::new(exp_fun));
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
