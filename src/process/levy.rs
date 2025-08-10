use crate::process::CoefficientFn;
use crate::process::Process;
use crate::process::increment::Incrementor;

pub struct LevyProcess {
    name: String,
    coefficients: Vec<Box<CoefficientFn>>,
    incrementors: Vec<Box<dyn Incrementor>>,
}

impl Process for LevyProcess {
    fn name(&self) -> &String {
        &self.name
    }

    fn coefficients(&self) -> &Vec<Box<CoefficientFn>> {
        &self.coefficients
    }

    fn incrementors(&mut self) -> &mut Vec<Box<dyn Incrementor>> {
        &mut self.incrementors
    }
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
