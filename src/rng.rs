use lru;
use ordered_float::OrderedFloat;
use rand;
use sobol;

pub trait Rng {
    fn sample(
        &mut self,
        scenario: i32,
        t_start: OrderedFloat<f64>,
        t_end: OrderedFloat<f64>,
        increment_name: &str,
    ) -> f64;
}

// (Pseudo) random sequence generator
pub struct PseudoRng {
    cache: lru::LruCache<
        (i32, OrderedFloat<f64>, OrderedFloat<f64>),
        std::collections::HashMap<String, f64>,
    >,
    increment_names: Vec<String>,
    rng: Box<dyn rand::RngCore>,
}

impl PseudoRng {
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

// (XOR) Scrambled Sobol sequence generator
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

trait Scrambler {
    fn scramble(&mut self, values: Vec<f64>) -> Vec<f64>;
}

struct XORScrambler {
    rng: rand::rngs::ThreadRng,
}

impl XORScrambler {
    pub fn new() -> Self
    where
        Self: Sized,
    {
        let rng = rand::rngs::ThreadRng::default();
        Self { rng }
    }
}

impl Scrambler for XORScrambler {
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
