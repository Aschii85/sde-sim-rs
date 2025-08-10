pub mod increment;
pub mod ito;
pub mod levy;
pub mod util;

use crate::filtration::Filtration;
use crate::process::increment::Incrementor;
use ordered_float::OrderedFloat;

pub type CoefficientFn = dyn Fn(&Filtration, OrderedFloat<f64>, i32) -> f64;

pub trait Process {
    fn name(&self) -> &String;
    fn coefficients(&self) -> &Vec<Box<CoefficientFn>>;
    fn incrementors(&mut self) -> &mut Vec<Box<dyn Incrementor>>;
}
