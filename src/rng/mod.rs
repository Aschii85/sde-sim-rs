pub mod pseudo;
pub mod sobol;

/// Trait for generating random or quasi-random numbers.
pub trait BaseRng {
    fn sample(&mut self, time_idx: usize, increment_idx: usize) -> f64;
}

/// Caches the generated random numbers for the current time step.
struct StepCache {
    time_idx: Option<usize>,
    values: Vec<f64>,
}
