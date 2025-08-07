use lru;
use ordered_float::OrderedFloat;
use rand;
use sobol;

pub trait Rng {
    fn new(increment_names: Vec<String>) -> Self
    where
        Self: Sized;
    fn sample(
        &mut self,
        scenario: i32,
        t_start: OrderedFloat<f64>,
        t_end: OrderedFloat<f64>,
        increment_name: &str,
    ) -> f64;
}

pub struct PseudoRng {
    cache: lru::LruCache<
        (i32, OrderedFloat<f64>, OrderedFloat<f64>),
        std::collections::HashMap<String, f64>,
    >,
    increment_names: Vec<String>,
    rng: Box<dyn rand::RngCore>,
}

impl Rng for PseudoRng {
    fn new(increment_names: Vec<String>) -> Self {
        Self {
            cache: lru::LruCache::new(std::num::NonZeroUsize::new(1).unwrap()),
            increment_names,
            rng: Box::new(rand::rngs::ThreadRng::default()),
        }
    }
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
            .unwrap()
    }
}

pub struct SobolRng {
    cache: lru::LruCache<
        (i32, OrderedFloat<f64>, OrderedFloat<f64>),
        std::collections::HashMap<String, f64>,
    >,
    increment_names: Vec<String>,
    rng: Box<std::iter::Skip<sobol::Sobol<f64>>>,
}

impl Rng for SobolRng {
    fn new(increment_names: Vec<String>) -> Self {
        use rand::Rng;
        let params = sobol::params::JoeKuoD6::minimal();
        let sobol_iter = sobol::Sobol::<f64>::new(increment_names.len(), &params);
        let rand_skip = rand::rngs::ThreadRng::default().gen_range(100..=99999); // Skip a random number of initial points
        Self {
            cache: lru::LruCache::new(std::num::NonZeroUsize::new(1).unwrap()),
            increment_names,
            rng: Box::new(sobol_iter.skip(rand_skip)),
        }
    }
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
            if let Some(random_numbers) = self.rng.next() {
                for (increment_name, random_number) in
                    self.increment_names.iter().zip(random_numbers.iter())
                {
                    rns.insert(increment_name.clone(), *random_number);
                }
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
