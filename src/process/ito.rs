use crate::filtration::Filtration;
use ordered_float::OrderedFloat;
use crate::process::Process;
use crate::process::increment::{Incrementor, TimeIncrementor, WienerIncrementor};

pub struct ItoProcess {
    name: String,
    coefficients: Vec<Box<dyn Fn(&Filtration, OrderedFloat<f64>, i32) -> f64>>,
    incrementors: Vec<Box<dyn Incrementor>>,
}

impl Process for ItoProcess {
    fn name(&self) -> &String {
        &self.name
    }

    fn coefficients(&self) -> &Vec<Box<dyn Fn(&Filtration, OrderedFloat<f64>, i32) -> f64>> {
        &self.coefficients
    }

    fn incrementors(&mut self) -> &mut Vec<Box<dyn Incrementor>> {
        &mut self.incrementors
    }
}

impl ItoProcess {
    pub fn new(
        name: String,
        drift: Box<dyn Fn(&Filtration, OrderedFloat<f64>, i32) -> f64>,
        diffusion: Box<dyn Fn(&Filtration, OrderedFloat<f64>, i32) -> f64>,
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

    pub fn drift(&self) -> &Box<dyn Fn(&Filtration, OrderedFloat<f64>, i32) -> f64> {
        &self.coefficients[0]
    }

    pub fn diffusion(&self) -> &Box<dyn Fn(&Filtration, OrderedFloat<f64>, i32) -> f64> {
        &self.coefficients[1]
    }
}
