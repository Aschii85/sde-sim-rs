use crate::filtration::Filtration;
use crate::process::Process;
use crate::process::increment::{Incrementor, TimeIncrementor, WienerIncrementor};
use ordered_float::OrderedFloat;
use std::sync::{Arc, Mutex};

pub struct ItoProcess {
    name: String,
    coefficients: Vec<Box<dyn Fn(&Filtration, OrderedFloat<f64>, i32) -> f64>>,
    incrementors: Vec<Box<dyn Incrementor>>, // All incrementors (trait objects)
    diffusion_incrementors: Vec<Box<Arc<Mutex<WienerIncrementor>>>>, // Only diffusion incrementors (concrete)
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
        diffusion_coefficients: Vec<Box<dyn Fn(&Filtration, OrderedFloat<f64>, i32) -> f64>>,
        diffusion_incrementors: Vec<Box<Arc<Mutex<WienerIncrementor>>>>,
    ) -> Result<Self, String> {
        let coefficients = std::iter::once(drift)
            .chain(diffusion_coefficients.into_iter())
            .collect();

        let incrementors: Vec<Box<dyn Incrementor>> =
            std::iter::once(Box::new(TimeIncrementor::new()) as Box<dyn Incrementor>)
                .chain(diffusion_incrementors.iter().map(|w| {
                    // Clone each WienerIncrementor for trait object storage
                    // You must implement Clone for WienerIncrementor
                    w.clone() as Box<dyn Incrementor>
                }))
                .collect();

        Ok(ItoProcess {
            name,
            coefficients,
            incrementors,
            diffusion_incrementors,
        })
    }

    pub fn drift(&self) -> &Box<dyn Fn(&Filtration, OrderedFloat<f64>, i32) -> f64> {
        &self.coefficients[0]
    }

    pub fn diffusion_coefficients(
        &self,
    ) -> &[Box<dyn Fn(&Filtration, OrderedFloat<f64>, i32) -> f64>] {
        &self.coefficients[1..]
    }

    pub fn diffusion_incrementors(&mut self) -> &mut Vec<Box<Arc<Mutex<WienerIncrementor>>>> {
        &mut self.diffusion_incrementors
    }
}
