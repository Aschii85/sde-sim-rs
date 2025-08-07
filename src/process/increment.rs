use crate::rng::Rng;
use lru;
use once_cell::sync::Lazy;
use ordered_float::OrderedFloat;
use statrs::distribution::{ContinuousCDF, Normal};

// TODO: Add other increments such as jumps

// Use a single standard normal distribution for Wiener process sampling
static NORMAL_STD: Lazy<Normal> = Lazy::new(|| Normal::standard());

pub trait Incrementor {
    fn new(name: String) -> Self
    where
        Self: Sized;
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
    cache: lru::LruCache<(i32, OrderedFloat<f64>, OrderedFloat<f64>), f64>,
    name: String,
}

impl Incrementor for TimeIncrementor {
    fn new(name: String) -> Self
    where
        Self: Sized,
    {
        let capacity = std::num::NonZeroUsize::new(1).unwrap();
        Self {
            cache: lru::LruCache::new(capacity),
            name,
        }
    }
    fn sample(
        &mut self,
        scenario: i32,
        t_start: OrderedFloat<f64>,
        t_end: OrderedFloat<f64>,
        _rng: &mut dyn Rng,
    ) -> f64 {
        let key = (scenario, t_start, t_end);
        if !self.cache.contains(&key) {
            let increment = (t_end - t_start).into_inner();
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
pub struct WienerIncrementor {
    cache: lru::LruCache<(i32, OrderedFloat<f64>, OrderedFloat<f64>), f64>,
    name: String,
}

impl Incrementor for WienerIncrementor {
    fn new(name: String) -> Self
    where
        Self: Sized,
    {
        let capacity = std::num::NonZeroUsize::new(1).unwrap();
        Self {
            cache: lru::LruCache::new(capacity),
            name,
        }
    }
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
            let increment = (t_end - t_start).sqrt() * NORMAL_STD.inverse_cdf(q);
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
