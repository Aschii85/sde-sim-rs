use crate::rng::BaseRng;
use ordered_float::OrderedFloat;

pub trait Incrementor: Send + Sync + std::fmt::Debug {
    fn sample(&mut self, time_idx: usize, rng: &mut dyn BaseRng) -> f64;
    fn clone_box(&self) -> Box<dyn Incrementor>;
}

impl Clone for Box<dyn Incrementor> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

#[derive(Clone, Debug)]
pub struct TimeIncrementor {
    dts: Vec<f64>,
}

impl TimeIncrementor {
    pub fn new(timesteps: Vec<OrderedFloat<f64>>) -> Self {
        let dts: Vec<f64> = timesteps
            .windows(2)
            .map(|w| (w[1] - w[0]).into_inner())
            .collect();
        Self { dts }
    }
}

impl Incrementor for TimeIncrementor {
    #[inline]
    fn sample(&mut self, time_idx: usize, _rng: &mut dyn BaseRng) -> f64 {
        self.dts[time_idx]
    }
    fn clone_box(&self) -> Box<dyn Incrementor> {
        Box::new(self.clone())
    }
}

#[derive(Clone, Debug)]
pub struct WienerIncrementor {
    idx: usize,
    sqrt_dts: Vec<f64>,
}

impl WienerIncrementor {
    pub fn new(idx: usize, timesteps: Vec<OrderedFloat<f64>>) -> Self {
        let sqrt_dts: Vec<f64> = timesteps
            .windows(2)
            .map(|w| (w[1] - w[0]).into_inner())
            .map(|dt| dt.sqrt())
            .collect();
        Self { idx, sqrt_dts }
    }
}

impl Incrementor for WienerIncrementor {
    #[inline]
    fn sample(&mut self, time_idx: usize, rng: &mut dyn BaseRng) -> f64 {
        let q = rng.sample(time_idx, self.idx);
        let increment = self.sqrt_dts[time_idx] * fast_inverse_normal_cdf(q);
        increment
    }
    fn clone_box(&self) -> Box<dyn Incrementor> {
        Box::new(Self {
            idx: self.idx,
            sqrt_dts: self.sqrt_dts.clone(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct JumpIncrementor {
    lambda: f64,
    idx: usize,
    dts: Vec<f64>,
}

impl JumpIncrementor {
    pub fn new(idx: usize, lambda: f64, timesteps: Vec<OrderedFloat<f64>>) -> Self {
        let dts: Vec<f64> = timesteps
            .windows(2)
            .map(|w| (w[1] - w[0]).into_inner())
            .collect();
        Self { lambda, idx, dts }
    }
}

impl Incrementor for JumpIncrementor {
    #[inline]
    fn sample(&mut self, time_idx: usize, rng: &mut dyn BaseRng) -> f64 {
        let u = rng.sample(time_idx, self.idx);
        let effective_lambda = self.lambda * self.dts[time_idx];
        let num_jumps = fast_inverse_poisson_cdf(u, effective_lambda) as f64;
        num_jumps
    }
    fn clone_box(&self) -> Box<dyn Incrementor> {
        Box::new(Self {
            lambda: self.lambda,
            idx: self.idx,
            dts: self.dts.clone(),
        })
    }
}

// Inverse cdff functions
#[inline]
fn fast_inverse_normal_cdf(p: f64) -> f64 {
    // High-precision approximation (Acklam's or similar)
    // For brevity, here is a standard efficient approximation
    // often used in high-performance simulators:
    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let x = t - ((c2 * t + c1) * t + c0) / (((d3 * t + d2) * t + d1) * t + 1.0);
    if p < 0.5 { -x } else { x }
}

#[inline]
fn fast_inverse_poisson_cdf(u: f64, lambda: f64) -> u64 {
    if lambda <= 0.0 {
        return 0;
    }
    // Initial probability P(X=0) = e^(-lambda)
    let mut p = (-lambda).exp();
    let mut f = p; // Cumulative distribution function value
    let mut k = 0;
    // Iterate until the cumulative probability exceeds our uniform sample
    // A cap of 200 is used for numerical safety, though for small lambda
    // it will resolve much earlier.
    while u > f && k < 200 {
        k += 1;
        // Recurrence: P(X=k) = P(X=k-1) * lambda / k
        p *= lambda / (k as f64);
        f += p;
    }
    k
}
