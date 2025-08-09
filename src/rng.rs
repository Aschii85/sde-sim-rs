use lru;
use ordered_float::OrderedFloat;
use rand;
use sobol;

/// Trait for generating random or quasi-random numbers for stochastic simulation.
///
/// Implementors of this trait provide a method to sample a random number
/// for a specific increment within a given time interval and scenario.
pub trait Rng {
    /// Samples a random number.
    ///
    /// The random number should correspond to the increment for a given scenario,
    /// time interval, and increment name.
    ///
    /// # Arguments
    ///
    /// * `scenario` - The identifier for the simulation path.
    /// * `t_start` - The start time of the interval.
    /// * `t_end` - The end time of the interval.
    /// * `increment_name` - The name of the increment (e.g., "dW1").
    ///
    /// # Returns
    ///
    /// A `f64` value representing the sampled random number.
    fn sample(
        &mut self,
        scenario: i32,
        t_start: OrderedFloat<f64>,
        t_end: OrderedFloat<f64>,
        increment_name: &str,
    ) -> f64;
}

/// A pseudorandom number generator for stochastic simulations.
///
/// This struct uses a standard thread-local RNG to generate a new set of
/// pseudorandom numbers for each unique time interval and scenario. It
/// caches the generated numbers to ensure consistency if the same interval
/// is sampled multiple times.
pub struct PseudoRng {
    cache: lru::LruCache<
        (i32, OrderedFloat<f64>, OrderedFloat<f64>),
        std::collections::HashMap<String, f64>,
    >,
    increment_names: Vec<String>,
    rng: Box<dyn rand::RngCore>,
}

impl PseudoRng {
    /// Creates a new `PseudoRng` instance.
    ///
    /// # Arguments
    ///
    /// * `increment_names` - A list of the names of the increments that will be sampled.
    pub fn new(increment_names: Vec<String>) -> Self
    where
        Self: Sized,
    {
        Self {
            cache: lru::LruCache::new(std::num::NonZeroUsize::new(1).unwrap()),
            increment_names,
            rng: Box::new(rand::rngs::ThreadRng::default()),
        }
    }
}

impl Rng for PseudoRng {
    /// Samples a pseudorandom number.
    ///
    /// If the `(scenario, t_start, t_end)` key is not in the cache, it generates
    /// a new set of random numbers for all increments in that interval, stores them,
    /// and then returns the requested increment's value.
    fn sample(
        &mut self,
        scenario: i32,
        t_start: OrderedFloat<f64>,
        t_end: OrderedFloat<f64>,
        increment_name: &str,
    ) -> f64 {
        let key = (scenario, t_start, t_end);
        if !self.cache.contains(&key) {
            let mut rns = std::collections::HashMap::new();
            for increment_name in &self.increment_names {
                let random_number: f64 = self.rng.next_u64() as f64 / u64::MAX as f64;
                rns.insert(increment_name.clone(), random_number);
            }
            self.cache.put(key, rns);
        }
        self.cache
            .get(&key)
            .unwrap()
            .get(increment_name)
            .cloned()
            .unwrap_or(0.0)
    }
}

/// A scrambled Sobol sequence generator for quasi-random sampling.
///
/// This struct provides a low-discrepancy sequence of numbers, which is beneficial
/// for Monte Carlo simulations as it can lead to faster convergence. It generates
/// a complete sequence for all time steps and increments upfront for each scenario
/// and uses a scrambler to remove potential biases.
pub struct SobolRng {
    cache: lru::LruCache<
        (i32, OrderedFloat<f64>, OrderedFloat<f64>),
        std::collections::HashMap<String, f64>,
    >,
    increment_names: Vec<String>,
    timesteps: Vec<OrderedFloat<f64>>,
    rng: Box<std::iter::Skip<sobol::Sobol<f64>>>,
    scrambler: Box<dyn Scrambler>,
}

impl SobolRng {
    /// Creates a new `SobolRng` instance.
    ///
    /// # Arguments
    ///
    /// * `increment_names` - A list of the names of the increments.
    /// * `timesteps` - The complete list of time points for the simulation.
    pub fn new(increment_names: Vec<String>, timesteps: Vec<OrderedFloat<f64>>) -> Self
    where
        Self: Sized,
    {
        let dims = (timesteps.len() - 1) * increment_names.len();
        let params = sobol::params::JoeKuoD6::extended(); // Supports up to 21201 dimensions
        let sobol_iter = sobol::Sobol::<f64>::new(dims.clone(), &params);
        Self {
            cache: lru::LruCache::new(std::num::NonZeroUsize::new(timesteps.len() - 1).unwrap()),
            increment_names,
            timesteps,
            rng: Box::new(sobol_iter.skip(5)),
            scrambler: Box::new(XORScrambler::new()),
        }
    }
}

impl Rng for SobolRng {
    /// Samples a quasi-random number from the Sobol sequence.
    ///
    /// This method populates the cache for an entire scenario at once by
    /// generating a scrambled Sobol sequence for all increments and time steps.
    /// It then retrieves the value for the requested increment.
    fn sample(
        &mut self,
        scenario: i32,
        t_start: OrderedFloat<f64>,
        t_end: OrderedFloat<f64>,
        increment_name: &str,
    ) -> f64 {
        let key = (scenario, t_start, t_end);
        if !self.cache.contains(&key) {
            if let Some(random_numbers) = self.rng.next() {
                let scrambled_numbers = self.scrambler.scramble(random_numbers.to_vec());
                for (idx, ts) in self.timesteps.windows(2).enumerate() {
                    let mut rns = std::collections::HashMap::new();
                    for (jdx, increment_name) in self.increment_names.iter().enumerate() {
                        rns.insert(
                            increment_name.clone(),
                            scrambled_numbers[jdx * self.increment_names.len() + idx],
                        );
                    }
                    self.cache.put((scenario, ts[0], ts[1]), rns);
                }
            }
        }
        self.cache
            .get(&key)
            .unwrap()
            .get(increment_name)
            .cloned()
            .unwrap_or(0.0)
    }
}

/* Scramblers to remove bias from the sobol sampler.
* TODO: Implement an Owen scrambler.
*/

/// Trait for scrambling low-discrepancy sequences.
///
/// Scrambling is used to remove the bias inherent in quasi-random sequences,
/// making them more robust for certain types of simulations.
trait Scrambler {
    /// Scrambles a vector of numbers.
    ///
    /// # Arguments
    ///
    /// * `values` - A vector of `f64` values from a low-discrepancy sequence.
    ///
    /// # Returns
    ///
    /// A new `Vec<f64>` with the scrambled values.
    fn scramble(&mut self, values: Vec<f64>) -> Vec<f64>;
}

/// An XOR scrambler for low-discrepancy sequences.
///
/// This scrambler applies a bitwise XOR operation to the mantissa of each
/// floating-point number, using a pseudorandom offset to generate a new,
/// scrambled sequence.
struct XORScrambler {
    rng: rand::rngs::ThreadRng,
}

impl XORScrambler {
    /// Creates a new `XORScrambler` instance.
    pub fn new() -> Self
    where
        Self: Sized,
    {
        let rng = rand::rngs::ThreadRng::default();
        Self { rng }
    }
}

impl Scrambler for XORScrambler {
    /// Scrambles a vector of numbers using a bitwise XOR operation.
    fn scramble(&mut self, values: Vec<f64>) -> Vec<f64> {
        use rand::Rng;
        const MANTISSA_MASK: u64 = 0x000F_FFFF_FFFF_FFFF; // 48 bits
        let mut scrambled = Vec::new();
        for value in values {
            let offset = self.rng.random::<u64>() & MANTISSA_MASK;
            scrambled.push(f64::from_bits(value.to_bits() ^ offset));
        }
        scrambled
    }
}
