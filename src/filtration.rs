use crate::proc::ProcessUniverse;
use ordered_float::OrderedFloat;
use std::{collections::HashMap};

pub struct ScenarioFiltration {
    pub scenario: i32,
    pub times: Vec<OrderedFloat<f64>>,
    pub process_universe: ProcessUniverse,
    raw_values: Vec<f64>,
    time_registry: HashMap<OrderedFloat<f64>, usize>,
}

impl ScenarioFiltration {
    pub fn new(
        scenario: i32,
        process_universe: ProcessUniverse,
        times: Vec<OrderedFloat<f64>>,
        initial_values: Option<HashMap<String, f64>>,
    ) -> Self {
        let raw_values = vec![0.0; times.len() * process_universe.processes.len()];
        let time_registry = times.iter().enumerate().map(|(i, t)| (*t, i)).collect();
        let mut scenario_filtration = ScenarioFiltration {
            scenario,
            process_universe,
            times,
            raw_values,
            time_registry,
        };
        if let Some(values) = initial_values {
            for (process_name, val) in values {
                if let Some(process_idx) = scenario_filtration.process_universe.process_registry.get(&process_name) {
                    scenario_filtration.set(0, *process_idx, val);
                }
            }
        }
        scenario_filtration
    }

    #[inline]
    pub fn get(&self, time_idx: usize, process_idx: usize) -> f64 {
        self.raw_values[time_idx * self.process_universe.processes.len() + process_idx]
    }

    #[inline]
    pub fn set(&mut self, time_idx: usize, process_idx: usize, val: f64) {
        let idx = time_idx * self.process_universe.processes.len() + process_idx;
        self.raw_values[idx] = val;
    }

    pub fn get_time_idx(&self, time: OrderedFloat<f64>) -> Option<&usize> {
        self.time_registry.get(&time)
    }

    pub fn to_dataframe(&self) -> polars::prelude::DataFrame {
        let num_procs = self.process_universe.processes.len();
        let row_count = self.times.len() * num_procs;

        let mut scenarios = Vec::with_capacity(row_count);
        let mut times = Vec::with_capacity(row_count);
        let mut process_names = Vec::with_capacity(row_count);

        for time in self.times.iter() {
            for process in self.process_universe.processes.iter() {
                scenarios.push(self.scenario);
                times.push(time.0);
                process_names.push(process.name().to_string());
            }
        }

        polars::prelude::df![
            "scenario" => scenarios,
            "time" => times,
            "process_name" => process_names,
            "value" => &self.raw_values
        ]
        .expect("DF error")
    }
}
