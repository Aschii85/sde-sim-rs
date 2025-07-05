use once_cell::sync::Lazy;
use rand::distributions::Distribution;
use rand::Rng;
use statrs::distribution::Normal;

pub enum Increment {
    Time,
    Wiener,
    // Add more variants as needed, e.g., Poisson, Gamma, etc.
}

// Use a standard normal distribution for Wiener process sampling
static NORMAL_STD: Lazy<Normal> = Lazy::new(|| Normal::standard());

impl Increment {
    pub fn sample<R: Rng + ?Sized>(&self, dt: f64, rng: &mut R) -> f64 {
        match self {
            Increment::Time => {
                dt
            }
            Increment::Wiener => {
                dt.sqrt() * NORMAL_STD.sample(rng)
            }
            // Add more variants as needed
        }
    }
}
