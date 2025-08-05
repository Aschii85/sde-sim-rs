use lru::LruCache;
use once_cell::sync::Lazy;
use rand::RngCore;
use rand::distributions::Distribution;
use statrs::distribution::Normal;
use std::num::NonZeroUsize;

// Use a standard normal distribution for Wiener process sampling
static NORMAL_STD: Lazy<Normal> = Lazy::new(|| Normal::standard());

pub trait Incrementor {
    fn sample(&mut self, scenario: i32, t_start: f64, t_end: f64, rng: &mut dyn RngCore) -> f64;
}

pub struct TimeIncrementor {}

impl Incrementor for TimeIncrementor {
    fn sample(&mut self, _scenario: i32, t_start: f64, t_end: f64, _rng: &mut dyn RngCore) -> f64 {
        t_end - t_start
    }
}

impl TimeIncrementor {
    pub fn new() -> Self {
        Self {}
    }
}

pub struct WienerIncrementor {
    cache: LruCache<(i32, i64, i64), f64>,
}

impl Incrementor for WienerIncrementor {
    fn sample(&mut self, scenario: i32, t_start: f64, t_end: f64, rng: &mut dyn RngCore) -> f64 {
        // Convert time to integer milliseconds for caching
        let t_start_ms = (t_start * 1000.0) as i64;
        let t_end_ms = (t_end * 1000.0) as i64;
        let key = (scenario, t_start_ms, t_end_ms);

        if !self.cache.contains(&key) {
            let increment = (t_end - t_start).sqrt() * NORMAL_STD.sample(rng);
            self.cache.put(key, increment);
        }
        *self.cache.get(&key).unwrap()
    }
}

impl WienerIncrementor {
    pub fn new() -> Self {
        let capacity = NonZeroUsize::new(10).unwrap();
        Self {
            cache: LruCache::new(capacity),
        }
    }
}
