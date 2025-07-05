use crate::filtration::Filtration;
use crate::process::Process;
use crate::process::increment::Increment;

pub struct LevyProcess {
    name: String,
    coefficients: Vec<Box<dyn Fn(&Filtration, f64, i32) -> f64>>,
    incrementors: Vec<Increment>,
}

impl Process for LevyProcess {
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


impl LevyProcess {
    pub fn new(
        name: String,
        coefficients: Vec<Box<dyn Fn(&Filtration, f64, i32) -> f64>>,
        incrementors: Vec<Increment>,
    ) -> Result<Self, String> {
        if coefficients.len() != incrementors.len() {
            return Err("coefficients and incrementors must have the same length".to_string());
        }
        Ok(LevyProcess {
            name,
            coefficients,
            incrementors,
        })
    }
}
