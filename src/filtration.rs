use fxhash::FxHashMap;
use ordered_float::OrderedFloat;
use std::collections::HashMap;

use crate::process::LevyProcess;

pub struct Filtration {
    pub scenarios: Vec<i32>,
    pub times: Vec<OrderedFloat<f64>>,
    pub processes: Vec<Box<LevyProcess>>,
    raw_values: Vec<f64>,
}

impl Filtration {
    pub fn new(
        times: Vec<OrderedFloat<f64>>,
        scenarios: Vec<i32>,
        processes: Vec<Box<LevyProcess>>,
        initial_values: Option<HashMap<String, f64>>,
    ) -> Self {
        let raw_values = vec![0.0; times.len() * scenarios.len() * processes.len()];
        let mut f = Filtration {
            scenarios,
            times,
            processes,
            raw_values,
        };

        if let Some(values) = initial_values {
            let fx_values: FxHashMap<String, f64> = values.into_iter().collect();
            f.set_initial_values(fx_values);
        }
        f
    }

    #[inline]
    pub fn get(&self, scenario_idx: usize, time_idx: usize, process_idx: usize) -> f64 {
        self.raw_values[scenario_idx * self.times.len() * self.processes.len()
            + time_idx * self.processes.len()
            + process_idx]
    }

    #[inline]
    pub fn get_processes_slice(&self, scenario_idx: usize, time_idx: usize) -> &[f64] {
        let start = (scenario_idx * self.times.len() * self.processes.len())
            + (time_idx * self.processes.len());
        let end = start + self.processes.len();
        &self.raw_values[start..end]
    }

    #[inline]
    pub fn set(&mut self, scenario_idx: usize, time_idx: usize, process_idx: usize, val: f64) {
        self.raw_values[scenario_idx * self.times.len() * self.processes.len()
            + time_idx * self.processes.len()
            + process_idx] = val;
    }

    pub fn set_initial_values(&mut self, values: FxHashMap<String, f64>) {
        let initial_vals: Vec<f64> = self
            .processes
            .iter()
            .map(|p| values.get(&p.name).copied().unwrap_or(0.0))
            .collect();
        for scenario_idx in 0..self.scenarios.len() {
            for (process_idx, &val) in initial_vals.iter().enumerate() {
                self.set(scenario_idx, 0, process_idx, val);
            }
        }
    }

    pub fn to_dataframe(&self) -> polars::prelude::DataFrame {
        let row_count = self.times.len() * self.scenarios.len() * self.processes.len();
        let mut times = Vec::with_capacity(row_count);
        let mut scenarios = Vec::with_capacity(row_count);
        let mut process_names = Vec::with_capacity(row_count);
        for &scenario in self.scenarios.iter() {
            for time in self.times.iter() {
                for process in self.processes.iter() {
                    times.push(time.0);
                    scenarios.push(scenario);
                    process_names.push(process.name.clone());
                }
            }
        }
        polars::prelude::df!["time"=>times, "scenario"=>scenarios, "process_name"=>process_names, "value"=>&self.raw_values]
            .expect("DF error")
    }
}
