use ndarray::Array3;
use polars;
use std::collections::HashMap;

pub struct Filtration
// Represents the state of a Markov process
{
    times: Vec<f64>,            // List of times
    scenarios: Vec<i32>,        // List of scenario identifiers
    process_names: Vec<String>, // List of process names
    raw_values: Array3<f64>,    // 3D array to hold values indexed by (time, scenario, process_name)
}

impl Filtration {
    pub fn new(
        times: Vec<f64>,
        scenarios: Vec<i32>,
        process_names: Vec<String>,
        raw_values: Array3<f64>,
        initial_values: Option<HashMap<String, f64>>,
    ) -> Self {
        let mut f = Filtration {
            times,
            scenarios,
            process_names,
            raw_values,
        };
        match initial_values {
            Some(values) => {
                f.set_initial_values(values);
            }
            None => {}
        }
        f
    }

    fn indices(
        &self,
        time: f64,
        scenario: i32,
        process_name: &String,
    ) -> Option<(usize, usize, usize)> {
        let time_idx = self.times.iter().position(|&t| t == time)?;
        let scenario_idx = self.scenarios.iter().position(|&s| s == scenario)?;
        let process_idx = self.process_names.iter().position(|n| n == process_name)?;
        Some((time_idx, scenario_idx, process_idx))
    }

    pub fn value(&self, time: f64, scenario: i32, process_name: String) -> f64 {
        let idx = self.indices(time, scenario, &process_name).unwrap();
        self.raw_values[idx]
    }
    /// Sets the value at the given (time, scenario, process_name) to new_value
    pub fn set_value(&mut self, time: f64, scenario: i32, process_name: String, new_value: f64) {
        let idx = self.indices(time, scenario, &process_name).unwrap();
        self.raw_values[idx] = new_value;
    }

    pub fn set_initial_values(&mut self, values: HashMap<String, f64>) {
        let initial_time = self.times[0].clone();
        let scenarios = self.scenarios.clone();
        let process_names = self.process_names.clone();
        for scenario in scenarios.iter() {
            for process_name in process_names.iter() {
                self.set_value(
                    initial_time,
                    *scenario,
                    process_name.clone(),
                    values[process_name.as_str()],
                );
            }
        }
    }

    /// Converts this Filtration into a Polars DataFrame
    pub fn to_dataframe(&self) -> polars::prelude::DataFrame {
        let mut time = Vec::new();
        let mut scenario = Vec::new();
        let mut process_name = Vec::new();
        let mut value = Vec::new();
        for (t_idx, &t) in self.times.iter().enumerate() {
            for (s_idx, &s) in self.scenarios.iter().enumerate() {
                for (p_idx, pname) in self.process_names.iter().enumerate() {
                    time.push(t);
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
