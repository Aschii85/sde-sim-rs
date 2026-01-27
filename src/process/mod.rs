pub mod increment;
pub mod util;

use crate::filtration::Filtration;
use crate::process::increment::Incrementor;
use ordered_float::OrderedFloat;

pub type CoefficientFn = dyn Fn(&Filtration, OrderedFloat<f64>, i32) -> f64;

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
            return Err("coefficients and incrementors must have the same length".to_string());
        }
        Ok(Self {
            name,
            coefficients,
            incrementors,
        })
    }
}
