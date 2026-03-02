pub mod increment;
pub mod util;

use crate::func::Function;
use std::collections::HashMap;

#[derive(Clone)]
pub struct AlgebraicProcess {
    pub name: String,
    pub coefficients: Vec<Box<Function>>,
}

pub struct LevyProcess {
    pub name: String,
    pub coefficients: Vec<Box<Function>>,
    pub incrementors: Vec<Box<dyn increment::Incrementor>>,
}

impl Clone for LevyProcess {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            coefficients: self.coefficients.clone(),
            incrementors: self.incrementors.iter().map(|i| i.clone_box()).collect(),
        }
    }
}

impl LevyProcess {
    pub fn new(
        name: String,
        coefficients: Vec<Box<Function>>,
        incrementors: Vec<Box<dyn increment::Incrementor>>,
    ) -> Result<Self, String> {
        if coefficients.len() != incrementors.len() {
            return Err("Number of coefficients must match incrementors".into());
        }
        Ok(Self {
            name,
            coefficients,
            incrementors,
        })
    }
}

#[derive(Clone)]
pub enum Process {
    Algebraic(Box<AlgebraicProcess>),
    Levy(Box<LevyProcess>),
}

impl Process {
    pub fn name(&self) -> &str {
        match self {
            Process::Levy(p) => &p.name,
            Process::Algebraic(p) => &p.name,
        }
    }
}


#[derive(Clone)]
pub struct ProcessUniverse {
    pub processes: Vec<Process>,
    pub process_registry: HashMap<String, usize>,
    pub stochastic_registry: HashMap<String, usize>,
    pub levy_process_indices: Vec<usize>,
    pub algebraic_process_indices: Vec<usize>,
}

impl ProcessUniverse {
    pub fn new(processes: Vec<Process>, stochastic_registry: HashMap<String, usize>) -> Self {
        let mut levy_process_indices = Vec::new();
        let mut algebraic_process_indices = Vec::new();
        let mut process_registry = HashMap::with_capacity(processes.len());
        for (idx, proc) in processes.iter().enumerate() {
            process_registry.insert(proc.name().to_string(), idx);
            match proc {
                Process::Levy(_) => levy_process_indices.push(idx),
                Process::Algebraic(_) => algebraic_process_indices.push(idx),
            }
        }
        Self {
            processes,
            process_registry,
            stochastic_registry,
            levy_process_indices,
            algebraic_process_indices,
        }
    }
}