use crate::proc::ProcessUniverse;
use ordered_float::OrderedFloat;
use polars::prelude::*;
use std::collections::BTreeMap;
use std::collections::HashMap;

pub struct ScenarioFiltrationCache {
    pub time: OrderedFloat<f64>,
    pub values: BTreeMap<String, f64>,
}

pub struct ScenarioFiltration {
    pub scenario: i32,
    pub times: Vec<OrderedFloat<f64>>,
    pub process_universe: ProcessUniverse,
    raw_values: Vec<f64>,
    time_registry: HashMap<OrderedFloat<f64>, usize>,
    pub cache: ScenarioFiltrationCache,
}

impl ScenarioFiltration {
    pub fn new(
        scenario: i32,
        process_universe: ProcessUniverse,
        times: Vec<OrderedFloat<f64>>,
        initial_values: HashMap<String, f64>,
    ) -> Self {
        let raw_values = vec![0.0; times.len() * process_universe.processes.len()];
        let time_registry = times.iter().enumerate().map(|(i, t)| (*t, i)).collect();
        let value_cache = ScenarioFiltrationCache {
            time: times[0],
            values: BTreeMap::new(),
        };
        let mut scenario_filtration = ScenarioFiltration {
            scenario,
            process_universe,
            times,
            raw_values,
            time_registry,
            cache: value_cache,
        };
        for (process_name, val) in initial_values.into_iter() {
            if let Some(process_idx) = scenario_filtration
                .process_universe
                .process_registry
                .get(&process_name)
            {
                scenario_filtration.set(0, *process_idx, val);
            }
        }
        scenario_filtration.refresh_cache(scenario_filtration.times[0]);
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

    pub fn refresh_cache(&mut self, time: OrderedFloat<f64>) {
        self.cache.time = time;
        self.cache.values.insert("t".to_string(), time.into_inner());
        let t_idx = self.get_time_idx(time).copied().unwrap_or(0);
        for (p_name, p_idx) in self.process_universe.process_registry.iter() {
            self.cache
                .values
                .insert(p_name.clone(), self.get(t_idx, *p_idx));
        }
    }

    pub fn to_lazyframe(&self) -> LazyFrame {
        let num_procs = self.process_universe.processes.len();
        let num_times = self.times.len();

        // 1. Fixed PlSmallStr by adding .into()
        // and using StringChunked::from_iter for cleaner collection
        let process_names: Series = StringChunked::from_iter(
            self.times
                .iter()
                .flat_map(|_| self.process_universe.processes.iter().map(|p| p.name())),
        )
        .with_name("process_name".into())
        .into_series();

        // 2. Fixed Float64Chunked collection
        // We use Float64Chunked::from_iter and .into() for the name
        let times: Series = Float64Chunked::from_iter(
            self.times
                .iter()
                .flat_map(|t| std::iter::repeat_n(Some(t.0), num_procs)),
        )
        .with_name("time".into())
        .into_series();

        // 3. Build the DataFrame
        // Note: The df! macro in 0.51 also expects PlSmallStr for column names
        // but the macro usually handles string literals via internal conversion.
        df![
            "scenario" => [self.scenario].repeat(num_procs * num_times),
            "time" => times,
            "process_name" => process_names,
            "value" => &self.raw_values
        ]
        .expect("Failed to create DataFrame")
        .lazy()
    }
}
