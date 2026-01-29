pub mod increment;
pub mod util;

use ordered_float::OrderedFloat;
use std::sync::Arc;

pub type CoefficientFn = dyn Fn(&[f64], OrderedFloat<f64>) -> f64 + Send + Sync;

pub struct LevyProcess {
    pub name: String,
    pub coefficients: Vec<Arc<CoefficientFn>>,
    pub incrementors: Vec<Box<dyn increment::Incrementor>>,
}

impl Clone for LevyProcess {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            coefficients: self.coefficients.clone(),
            incrementors: self.incrementors.iter().map(|i| i.clone_box()).collect(),
        }
    }
}

impl LevyProcess {
    pub fn new(
        name: String,
        coefficients: Vec<Arc<CoefficientFn>>,
        incrementors: Vec<Box<dyn increment::Incrementor>>,
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
