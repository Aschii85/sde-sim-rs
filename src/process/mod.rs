pub mod increment;
pub mod ito;
pub mod levy;
pub mod util;

use crate::filtration::Filtration;
use crate::process::increment::Incrementor;

pub trait Process {
    fn name(&self) -> &String;
    fn coefficients(&self) -> &Vec<Box<dyn Fn(&Filtration, f64, i32) -> f64>>;
    fn incrementors(&mut self) -> &mut Vec<Box<dyn Incrementor>>;
}
