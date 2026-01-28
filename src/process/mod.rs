pub mod increment;
pub mod util;

use crate::filtration::Filtration;
use crate::process::increment::Incrementor;

// Updated: Now takes the slice of values for O(1) math evaluation
pub type CoefficientFn = dyn Fn(&Filtration, &[f64], f64, i32) -> f64 + Send + Sync;

pub struct LevyProcess {
    pub name: String,
    pub coefficients: Vec<Box<CoefficientFn>>,
    pub incrementors: Vec<Box<dyn Incrementor>>,
}

impl LevyProcess {
    pub fn new(
        name: String,
        coefficients: Vec<Box<CoefficientFn>>,
        incrementors: Vec<Box<dyn Incrementor>>,
    ) -> Result<Self, String> {
        if coefficients.len() != incrementors.len() {
            return Err("Number of coefficients must match incrementors".into());
        }
        Ok(Self {
            name,
            coefficients,
            incrementors,
        })
    }
}
