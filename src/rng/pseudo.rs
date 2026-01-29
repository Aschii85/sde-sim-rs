use crate::rng::{Rng, StepCache};
use rand::{Rng as RandRng, SeedableRng};
use rand_chacha::ChaCha8Rng;

// --- Pseudo RNG ---

pub struct PseudoRng {
    last_step: Option<StepCache>,
    num_increments: usize,
    rng: ChaCha8Rng,
}

impl PseudoRng {
    pub fn new(num_increments: usize) -> Self {
        Self {
            last_step: None,
            num_increments,
            rng: ChaCha8Rng::from_os_rng(),
        }
    }

    fn refresh_cache(&mut self, time_idx: usize, scenario_idx: usize) {
        let mut values = Vec::with_capacity(self.num_increments);
        for _ in 0..self.num_increments {
            values.push(self.rng.random::<f64>());
        }
        self.last_step = Some(StepCache {
            time_idx: Some(time_idx),
            scenario_idx,
            values,
        });
    }
}

impl Rng for PseudoRng {
    fn sample(&mut self, time_idx: usize, scenario_idx: usize, increment_idx: usize) -> f64 {
        let is_cached = self
            .last_step
            .as_ref()
            .is_some_and(|c| c.scenario_idx == scenario_idx && c.time_idx == Some(time_idx));

        if !is_cached {
            self.refresh_cache(time_idx, scenario_idx);
        }

        self.last_step
            .as_ref()
            .unwrap()
            .values
            .get(increment_idx)
            .copied()
            .unwrap_or(0.0)
    }
}
