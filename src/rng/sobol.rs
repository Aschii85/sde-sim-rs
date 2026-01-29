use crate::rng::{Rng, StepCache};
use rand::{Rng as RandRng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sobol;

// --- Sobol RNG ---

pub struct SobolRng {
    last_step: Option<StepCache>,
    num_increments: usize,
    sobol_iter: Box<std::iter::Skip<sobol::Sobol<f64>>>,
    scrambler: XORScrambler,
}

impl SobolRng {
    pub fn new(num_increments: usize, num_timesteps: usize) -> Self {
        let dims = (num_timesteps - 1) * num_increments;
        let params = sobol::params::JoeKuoD6::extended();
        let sobol_iter = sobol::Sobol::<f64>::new(dims, &params);

        Self {
            last_step: None,
            num_increments,
            sobol_iter: Box::new(sobol_iter.skip(5)),
            scrambler: XORScrambler::new(),
        }
    }

    fn refresh_cache(&mut self, scenario_idx: usize) {
        // If scenario changed, grab next Sobol vector for the entire path
        if let Some(raw) = self.sobol_iter.next() {
            let current_scrambled_path = self.scrambler.scramble(raw);
            self.last_step = Some(StepCache {
                time_idx: None,
                scenario_idx,
                values: current_scrambled_path,
            });
        }
    }
}

impl Rng for SobolRng {
    fn sample(&mut self, time_idx: usize, scenario_idx: usize, increment_idx: usize) -> f64 {
        let is_cached = self
            .last_step
            .as_ref()
            .map_or(false, |c| c.scenario_idx == scenario_idx);

        if !is_cached {
            self.refresh_cache(scenario_idx);
        }

        self.last_step
            .as_ref()
            .unwrap()
            .values
            .get(time_idx * self.num_increments + increment_idx)
            .copied()
            .unwrap_or(0.0)
    }
}

// --- Scrambler ---

struct XORScrambler {
    rng: ChaCha8Rng,
}

impl XORScrambler {
    fn new() -> Self {
        Self {
            rng: ChaCha8Rng::from_os_rng(),
        }
    }

    fn scramble(&mut self, mut values: Vec<f64>) -> Vec<f64> {
        const MANTISSA_MASK: u64 = 0x000F_FFFF_FFFF_FFFF;
        for val in values.iter_mut() {
            let offset = self.rng.random::<u64>() & MANTISSA_MASK;
            *val = f64::from_bits(val.to_bits() ^ offset);
        }
        values
    }
}
