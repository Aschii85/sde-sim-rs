use crate::rng::Rng;
use lru;
use ordered_float::OrderedFloat;
use statrs::distribution::{ContinuousCDF, DiscreteCDF, Normal, Poisson};

pub trait Incrementor {
    fn sample(
        &mut self,
        scenario: i32,
        t_start: OrderedFloat<f64>,
        t_end: OrderedFloat<f64>,
        rng: &mut dyn Rng,
    ) -> f64;
    fn name(&self) -> &String;
}

#[derive(Clone)]
pub struct TimeIncrementor {
    name: String,
}

impl TimeIncrementor {
    pub fn new() -> Self
    where
        Self: Sized,
    {
        Self {
            name: "dt".to_string(),
        }
    }
}

impl Incrementor for TimeIncrementor {
    fn sample(
        &mut self,
        _scenario: i32,
        t_start: OrderedFloat<f64>,
        t_end: OrderedFloat<f64>,
        _rng: &mut dyn Rng,
    ) -> f64 {
        (t_end - t_start).into_inner()
    }
    fn name(&self) -> &String {
        &self.name
    }
}

#[derive(Clone)]
pub struct WienerIncrementor {
    cache: lru::LruCache<(i32, OrderedFloat<f64>, OrderedFloat<f64>), f64>,
    name: String,
    dist: Normal,
}

impl WienerIncrementor {
    pub fn new(name: String) -> Self
    where
        Self: Sized,
    {
        let capacity = match std::num::NonZeroUsize::new(1) {
            Some(c) => c,
            None => panic!("Capacity must be non-zero"),
        };
        Self {
            cache: lru::LruCache::new(capacity),
            name,
            dist: Normal::standard(),
        }
    }
}

impl Incrementor for WienerIncrementor {
    fn sample(
        &mut self,
        scenario: i32,
        t_start: OrderedFloat<f64>,
        t_end: OrderedFloat<f64>,
        rng: &mut dyn Rng,
    ) -> f64 {
        // Convert time to integer milliseconds for caching
        let key = (scenario, t_start, t_end);
        if !self.cache.contains(&key) {
            let q = rng.sample(scenario, t_start, t_end, &self.name);
            let increment = (t_end - t_start).sqrt() * self.dist.inverse_cdf(q);
            self.cache.put(key, increment);
        }
        match self.cache.get(&key) {
            Some(val) => *val,
            None => 0.0,
        }
    }
    fn name(&self) -> &String {
        &self.name
    }
}

#[derive(Clone)]
pub struct JumpIncrementor {
    cache: lru::LruCache<(i32, OrderedFloat<f64>, OrderedFloat<f64>), f64>,
    name: String,
    lambda: f64,
}

impl JumpIncrementor {
    pub fn new(name: String, lambda: f64) -> Self
    where
        Self: Sized,
    {
        let capacity = match std::num::NonZeroUsize::new(1) {
            Some(c) => c,
            None => panic!("Capacity must be non-zero"),
        };
        Self {
            cache: lru::LruCache::new(capacity),
            name,
            lambda,
        }
    }
}

impl Incrementor for JumpIncrementor {
    fn sample(
        &mut self,
        scenario: i32,
        t_start: OrderedFloat<f64>,
        t_end: OrderedFloat<f64>,
        rng: &mut dyn Rng,
    ) -> f64 {
        let key = (scenario, t_start, t_end);
        if !self.cache.contains(&key) {
            let u = rng.sample(scenario, t_start, t_end, &self.name);
            let effective_lambda = self.lambda * (t_end - t_start).into_inner();
            let num_jumps = match Poisson::new(effective_lambda) {
                Ok(dist) => {
                    let mut result = 0.0;
                    for _jump_count in 0.. {
                        if dist.cdf(_jump_count) >= u {
                            result = _jump_count as f64;
                            break;
                        }
                    }
                    result
                }
                Err(_) => 0.0,
            };
            self.cache.put(key, num_jumps);
        }
        match self.cache.get(&key) {
            Some(val) => *val,
            None => 0.0,
        }
    }
    fn name(&self) -> &String {
        &self.name
    }
}
