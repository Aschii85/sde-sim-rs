pub mod ito;
pub mod increment;
pub mod levy;

use crate::filtration::Filtration;
use crate::process::increment::Increment;

pub trait Process {
    fn name(&self) -> &String;
    fn coefficients(&self) -> &Vec<fn(&Filtration, f64, i32) -> f64>;
    fn incrementors(&self) -> &Vec<Increment>;
}