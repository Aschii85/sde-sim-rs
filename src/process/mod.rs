pub mod increment;
pub mod util;

use crate::filtration::Filtration;
use crate::process::increment::Incrementor;
use ordered_float::OrderedFloat;
use std::sync::Arc;

// Updated: Now takes the slice of values for O(1) math evaluation
pub type CoefficientFn = dyn Fn(&Filtration, OrderedFloat<f64>, usize, usize) -> f64 + Send + Sync;

pub struct LevyProcess {
    pub name: String,
    pub coefficients: Vec<Arc<CoefficientFn>>,
    pub incrementors: Vec<Box<dyn Incrementor>>,
}

impl Clone for LevyProcess {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            // Arc cloning is cheap (ref-count increment)
            coefficients: self.coefficients.clone(),
            // Box cloning uses our custom clone_box helper
            incrementors: self.incrementors.iter().map(|i| i.clone_box()).collect(),
        }
    }
}

impl LevyProcess {
    pub fn new(
        name: String,
        coefficients: Vec<Arc<CoefficientFn>>,
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
