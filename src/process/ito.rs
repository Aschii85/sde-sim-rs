use crate::filtration::Filtration;
use crate::process::levy::LevyLike;
use crate::process::increment::Increment;

pub struct ItoProcess {
    name: String,
    drift: fn(&Filtration, f64, i32) -> f64,
    diffusion: fn(&Filtration, f64, i32) -> f64,
    coefficients: Vec<fn(&Filtration, f64, i32) -> f64>,
    incrementors: Vec<Increment>,
}

impl LevyLike for ItoProcess {
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

impl ItoProcess {
    pub fn new(
        name: String,
        drift: fn(&Filtration, f64, i32) -> f64,
        diffusion: fn(&Filtration, f64, i32) -> f64,
    ) -> Result<Self, String> {
        let coefficients = vec![drift, diffusion];
        let incrementors = vec![
            Increment::Time,
            Increment::Wiener,
        ];
        Ok(ItoProcess {
            name,
            drift,
            diffusion,
            coefficients,
            incrementors
        })
    }

    pub fn drift(&self) -> &fn(&Filtration, f64, i32) -> f64 {
        &self.drift
    }

    pub fn diffusion(&self) -> &fn(&Filtration, f64, i32) -> f64 {
        &self.diffusion
    }
}
