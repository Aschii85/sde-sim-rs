use ndarray::Array3;
use ordered_float::OrderedFloat;
use std::collections::HashMap;

pub struct Filtration {
    times: Vec<OrderedFloat<f64>>,
    scenarios: Vec<i32>,
    process_names: Vec<String>,
    raw_values: Array3<f64>,
    time_idx_map: HashMap<OrderedFloat<f64>, usize>,
    scenario_idx_map: HashMap<i32, usize>,
    process_name_idx_map: HashMap<String, usize>,
}

impl Filtration {
    pub fn new(
        times: Vec<OrderedFloat<f64>>,
        scenarios: Vec<i32>,
        process_names: Vec<String>,
        raw_values: Array3<f64>,
        initial_values: Option<HashMap<String, f64>>,
    ) -> Self {
        let time_idx_map = times.iter().enumerate().map(|(i, t)| (*t, i)).collect();
        let scenario_idx_map = scenarios.iter().enumerate().map(|(i, &s)| (s, i)).collect();
        let process_name_idx_map = process_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();

        let mut f = Filtration {
            times,
            scenarios,
            process_names,
            raw_values,
            time_idx_map,
            scenario_idx_map,
            process_name_idx_map,
        };
        if let Some(values) = initial_values {
            f.set_initial_values(values);
        }
        f
    }

    /// Resolves a process name to its internal index.
    /// Used by the parser to bake indices into the simulation logic.
    pub fn get_process_index(&self, name: &str) -> Option<usize> {
        self.process_name_idx_map.get(name).copied()
    }

    /// HIGH PERFORMANCE PATH: Access values via pre-resolved process index.
    #[inline]
    pub fn value_by_index(
        &self,
        time: OrderedFloat<f64>,
        scenario: i32,
        process_idx: usize,
    ) -> Result<f64, String> {
        let t_idx = *self.time_idx_map.get(&time).ok_or("Time not found")?;
        let s_idx = *self
            .scenario_idx_map
            .get(&scenario)
            .ok_or("Scenario not found")?;
        Ok(self.raw_values[[t_idx, s_idx, process_idx]])
    }

    #[inline]
    fn indices(
        &self,
        time: OrderedFloat<f64>,
        scenario: i32,
        process_name: &str,
    ) -> Option<(usize, usize, usize)> {
        let &time_idx = self.time_idx_map.get(&time)?;
        let &scenario_idx = self.scenario_idx_map.get(&scenario)?;
        let &process_idx = self.process_name_idx_map.get(process_name)?;
        Some((time_idx, scenario_idx, process_idx))
    }

    pub fn value(
        &self,
        time: OrderedFloat<f64>,
        scenario: i32,
        process_name: &str,
    ) -> Result<f64, String> {
        match self.indices(time, scenario, process_name) {
            Some(idx) => Ok(self.raw_values[idx]),
            None => Err(format!("No value found for process: {}", process_name)),
        }
    }

    pub fn set_value(
        &mut self,
        time: OrderedFloat<f64>,
        scenario: i32,
        process_name: &str,
        val: f64,
    ) {
        if let Some(idx) = self.indices(time, scenario, process_name) {
            self.raw_values[idx] = val;
        }
    }

    pub fn set_initial_values(&mut self, values: HashMap<String, f64>) {
        let initial_time = self.times[0];
        for scenario in self.scenarios.clone() {
            for process_name in self.process_names.clone() {
                let val = values.get(&process_name).copied().unwrap_or(0.0);
                self.set_value(initial_time, scenario, &process_name, val);
            }
        }
    }

    pub fn to_dataframe(&self) -> polars::prelude::DataFrame {
        let mut time = Vec::new();
        let mut scenario = Vec::new();
        let mut process_name = Vec::new();
        let mut value = Vec::new();
        for (t_idx, t) in self.times.iter().enumerate() {
            for (s_idx, &s) in self.scenarios.iter().enumerate() {
                for (p_idx, pname) in self.process_names.iter().enumerate() {
                    time.push(t.0);
                    scenario.push(s);
                    process_name.push(pname.clone());
                    value.push(self.raw_values[[t_idx, s_idx, p_idx]]);
                }
            }
        }
        polars::prelude::df!["time"=>time, "scenario"=>scenario, "process_name"=>process_name, "value"=>value]
            .expect("DF error")
    }
}
