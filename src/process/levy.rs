use crate::process::increment::Increment;
use crate::filtration::Filtration;

pub struct LevyProcess {
    name: String,
    coefficients: Vec<fn(&Filtration, f64, i32) -> f64>,
    incrementors: Vec<Increment>,
}
pub trait LevyLike {
    fn name(&self) -> &String;
    fn coefficients(&self) -> &Vec<fn(&Filtration, f64, i32) -> f64>;
    fn incrementors(&self) -> &Vec<Increment>;
}

impl LevyLike for LevyProcess {
    fn name(&self) -> &String {
        &self.name
    }

    fn coefficients(&self) -> &Vec<fn(&Filtration, f64, i32) -> f64> {
        &self.coefficients
    }

    fn incrementors(&self) -> &Vec<Increment> {
        &self.incrementors
    }
}


impl LevyProcess {
    pub fn new(
        name: String,
        coefficients: Vec<fn(&Filtration, f64, i32) -> f64>,
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
