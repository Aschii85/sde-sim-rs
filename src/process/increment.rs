use crate::rng::Rng;
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

/// Simple register to cache the very last calculated value.
#[derive(Clone)]
struct LastValue {
    scenario: i32,
    t_start: OrderedFloat<f64>,
    t_end: OrderedFloat<f64>,
    value: f64,
}

#[derive(Clone)]
pub struct TimeIncrementor {
    name: String,
}

impl TimeIncrementor {
    pub fn new() -> Self {
        Self {
            name: "dt".to_string(),
        }
    }
}

impl Incrementor for TimeIncrementor {
    fn sample(
        &mut self,
        _s: i32,
        t0: OrderedFloat<f64>,
        t1: OrderedFloat<f64>,
        _rng: &mut dyn Rng,
    ) -> f64 {
        (t1 - t0).into_inner()
    }
    fn name(&self) -> &String {
        &self.name
    }
}

#[derive(Clone)]
pub struct WienerIncrementor {
    last: Option<LastValue>,
    name: String,
    dist: Normal,
}

impl WienerIncrementor {
    pub fn new(name: String) -> Self {
        Self {
            last: None,
            name,
            dist: Normal::standard(),
        }
    }
}

impl Incrementor for WienerIncrementor {
    #[inline]
    fn sample(
        &mut self,
        scenario: i32,
        t_start: OrderedFloat<f64>,
        t_end: OrderedFloat<f64>,
        rng: &mut dyn Rng,
    ) -> f64 {
        // Check "Register" instead of Hash Table
        if let Some(ref last) = self.last {
            if last.scenario == scenario && last.t_start == t_start && last.t_end == t_end {
                return last.value;
            }
        }

        let q = rng.sample(scenario, t_start, t_end, &self.name);
        let increment = (t_end - t_start).sqrt() * self.dist.inverse_cdf(q);

        self.last = Some(LastValue {
            scenario,
            t_start,
            t_end,
            value: increment,
        });
        increment
    }
    fn name(&self) -> &String {
        &self.name
    }
}

#[derive(Clone)]
pub struct JumpIncrementor {
    last: Option<LastValue>,
    name: String,
    lambda: f64,
}

impl JumpIncrementor {
    pub fn new(name: String, lambda: f64) -> Self {
        Self {
            last: None,
            name,
            lambda,
        }
    }
}

impl Incrementor for JumpIncrementor {
    #[inline]
    fn sample(
        &mut self,
        scenario: i32,
        t_start: OrderedFloat<f64>,
        t_end: OrderedFloat<f64>,
        rng: &mut dyn Rng,
    ) -> f64 {
        if let Some(ref last) = self.last {
            if last.scenario == scenario && last.t_start == t_start && last.t_end == t_end {
                return last.value;
            }
        }

        let u = rng.sample(scenario, t_start, t_end, &self.name);
        let dt = (t_end - t_start).into_inner();
        let effective_lambda = self.lambda * dt;

        let num_jumps = if let Ok(dist) = Poisson::new(effective_lambda) {
            let mut result = 0.0;
            // Removed redundant p_sum assignment
            for j in 0..1000 {
                if dist.cdf(j) >= u {
                    result = j as f64;
                    break;
                }
            }
            result
        } else {
            0.0
        };

        self.last = Some(LastValue {
            scenario,
            t_start,
            t_end,
            value: num_jumps,
        });
        num_jumps
    }
    fn name(&self) -> &String {
        &self.name
    }
}
