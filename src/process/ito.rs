use crate::process::CoefficientFn;
use crate::process::Process;
use crate::process::increment::Incrementor;

pub struct ItoProcess {
    name: String,
    coefficients: Vec<Box<CoefficientFn>>,
    incrementors: Vec<Box<dyn Incrementor>>,
}

impl Process for ItoProcess {
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

impl ItoProcess {
    pub fn new(
        name: String,
        coefficients: Vec<Box<CoefficientFn>>,
        incrementors: Vec<Box<dyn Incrementor>>,
    ) -> Result<Self, String> {
        if coefficients.len() != incrementors.len() {
            return Err("coefficients and incrementors must have the same length".to_string());
        }
        for incrementor in &incrementors {
            if !incrementor.name().starts_with("W") && incrementor.name() != "t" {
                return Err(format!(
                    "All incrementors must be Wiener processes or the time increment 't', not '{}'!",
                    incrementor.name()
                ));
            }
        }
        Ok(Self {
            name,
            coefficients,
            incrementors,
        })
    }
}
