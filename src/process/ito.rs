use crate::filtration::Filtration;
use crate::process::Process;
use crate::process::increment::{Incrementor, TimeIncrementor, WienerIncrementor};

pub struct ItoProcess {
    name: String,
    coefficients: Vec<Box<dyn Fn(&Filtration, f64, i32) -> f64>>,
    incrementors: Vec<Box<dyn Incrementor>>,
}

impl Process for ItoProcess {
    fn name(&self) -> &String {
        &self.name
    }

    fn coefficients(&self) -> &Vec<Box<dyn Fn(&Filtration, f64, i32) -> f64>> {
        &self.coefficients
    }

    fn incrementors(&mut self) -> &mut Vec<Box<dyn Incrementor>> {
        &mut self.incrementors
    }
}

impl ItoProcess {
    pub fn new(
        name: String,
        drift: Box<dyn Fn(&Filtration, f64, i32) -> f64>,
        diffusion: Box<dyn Fn(&Filtration, f64, i32) -> f64>,
    ) -> Result<Self, String> {
        let coefficients = vec![drift, diffusion];
        let incrementors: Vec<Box<dyn Incrementor>> = vec![
            Box::new(TimeIncrementor::new()),
            Box::new(WienerIncrementor::new()),
        ];
        Ok(ItoProcess {
            name,
            coefficients,
            incrementors,
        })
    }

    pub fn drift(&self) -> &Box<dyn Fn(&Filtration, f64, i32) -> f64> {
        &self.coefficients[0]
    }

    pub fn diffusion(&self) -> &Box<dyn Fn(&Filtration, f64, i32) -> f64> {
        &self.coefficients[1]
    }
}

/// Factory Functions for Specific Ito Processes
// These functions create specific instances of ItoProcess with predefined drift and diffusion coefficients.
// They are prefered  over using `from_string` for common processes, as they are more efficient
// computationally (around 2/4-times faster for meval/luajit).
pub fn geometric_brownian_motion(name: String, mu: f64, sigma: f64) -> ItoProcess {
    let _name = name.clone();
    let drift = Box::new(move |f: &Filtration, t: f64, s: i32| mu * f.value(t, s, _name.clone()));
    let _name = name.clone();
    let diffusion =
        Box::new(move |f: &Filtration, t: f64, s: i32| sigma * f.value(t, s, _name.clone()));
    ItoProcess::new(name.clone(), drift, diffusion).unwrap()
}
