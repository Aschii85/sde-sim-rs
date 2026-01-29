use crate::rng::BaseRng;
use rand::{Rng as RandRng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sobol::params::JoeKuoD6;
use std::sync::{Arc, Mutex, OnceLock};

static SOBOL_PARAMS: OnceLock<JoeKuoD6> = OnceLock::new();

/// The internal "Engine" that is shared across all scenarios.
pub struct SobolEngine {
    sobol_iter: Box<dyn Iterator<Item = Vec<f64>> + Send>,
}

impl SobolEngine {
    pub fn new(dims: usize) -> Self {
        let params = SOBOL_PARAMS.get_or_init(JoeKuoD6::extended);
        let sobol_iter = sobol::Sobol::<f64>::new(dims, params).skip(5);
        Self {
            sobol_iter: Box::new(sobol_iter),
        }
    }

    pub fn next_path(&mut self) -> Option<Vec<f64>> {
        self.sobol_iter.next()
    }
}

/// The lightweight RNG wrapper created per scenario.
pub struct SobolRng {
    num_increments: usize,
    values: Vec<f64>,
}

impl SobolRng {
    pub fn new(
        seed: u64,
        engine: Arc<Mutex<SobolEngine>>,
        num_increments: usize,
        num_timesteps: usize,
    ) -> Self {
        let raw = {
            let mut lock = engine.lock().unwrap();
            lock.next_path().expect("Sobol sequence exhausted")
        };
        let dims = (num_timesteps - 1) * num_increments;
        let scrambler = RandomShiftScrambler::new(dims, seed);
        let scrambled = scrambler.scramble(raw);

        Self {
            num_increments,
            values: scrambled,
        }
    }
}

impl BaseRng for SobolRng {
    fn sample(&mut self, time_idx: usize, increment_idx: usize) -> f64 {
        self.values[time_idx * self.num_increments + increment_idx]
    }
}

struct RandomShiftScrambler {
    shift: Vec<f64>,
}

impl RandomShiftScrambler {
    fn new(dims: usize, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let shift = (0..dims).map(|_| rng.random::<f64>()).collect();
        Self { shift }
    }

    fn scramble(&self, mut values: Vec<f64>) -> Vec<f64> {
        for (val, &s) in values.iter_mut().zip(self.shift.iter()) {
            *val = (*val + s).fract();
        }
        values
    }
}
