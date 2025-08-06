use ndarray::Array3;
use ordered_float::OrderedFloat;
use polars;
use std::collections::HashMap;

pub struct Filtration {
    times: Vec<OrderedFloat<f64>>, // List of times as OrderedFloat
    scenarios: Vec<i32>,           // List of scenario identifiers
    process_names: Vec<String>,    // List of process names
    raw_values: Array3<f64>, // 3D array to hold values indexed by (time, scenario, process_name)
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
        let time_idx_map = times
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i))
            .collect();
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
            None => Err(format!(
                "No value found for time: {:?}, scenario: {}, process_name: {}",
                time, scenario, process_name
            )),
        }
    }
    /// Sets the value at the given (time, scenario, process_name) to new_value
    pub fn set_value(
        &mut self,
        time: OrderedFloat<f64>,
        scenario: i32,
        process_name: &str,
        new_value: f64,
    ) {
        if let Some(idx) = self.indices(time, scenario, process_name) {
            self.raw_values[idx] = new_value;
        }
    }

    pub fn set_initial_values(&mut self, values: HashMap<String, f64>) {
        let initial_time = self.times[0];
        let process_names: Vec<String> = self.process_names.iter().cloned().collect();
        let scenarios: Vec<i32> = self.scenarios.iter().copied().collect();
        for scenario in scenarios {
            for process_name in &process_names {
                let val = values.get(process_name.as_str()).copied().unwrap_or(0.0);
                self.set_value(initial_time, scenario, process_name.as_str(), val);
            }
        }
    }

    /// Converts this Filtration into a Polars DataFrame
    pub fn to_dataframe(&self) -> polars::prelude::DataFrame {
        let n_times = self.times.len();
        let n_scenarios = self.scenarios.len();
        let n_processes = self.process_names.len();
        let total = n_times * n_scenarios * n_processes;
        let mut time = Vec::with_capacity(total);
        let mut scenario = Vec::with_capacity(total);
        let mut process_name = Vec::with_capacity(total);
        let mut value = Vec::with_capacity(total);
        for (t_idx, t) in self.times.iter().enumerate() {
            for (s_idx, &s) in self.scenarios.iter().enumerate() {
                for (p_idx, pname) in self.process_names.iter().enumerate() {
                    time.push(t.0); // Access the inner f64 value
                    scenario.push(s);
                    process_name.push(pname.clone());
                    value.push(self.raw_values[[t_idx, s_idx, p_idx]]);
                }
            }
        }
        polars::prelude::df![
            "time" => time,
            "scenario" => scenario,
            "process_name" => process_name,
            "value" => value,
        ]
        .expect("Failed to create DataFrame")
    }
}
