use fasteval::{Compiler, Evaler, Instruction, Slab};
use ordered_float::OrderedFloat;
use crate::filtration::ScenarioFiltration;
use std::collections::BTreeMap;

pub struct Function {
    instruction: Instruction,
    slab: Slab,
    values_cache: (OrderedFloat<f64>, BTreeMap<String, f64>),
    expr_str: String,
}

impl Clone for Function {
    fn clone(&self) -> Self {
        // We re-parse AND re-compile.
        // This is slow, but it only happens once per thread.
        Self::new(&self.expr_str).expect("Failed to re-compile on clone")
    }
}

impl Function {
    /// Creates a new Function by parsing the string and resolving variable names
    /// against the provided filtration object.
    pub fn new(expr_str: &str) -> Result<Self, String> {
        // Parse and Compile
        let parser = fasteval::Parser::new();
        let mut slab = Slab::new();
        let expr = parser
            .parse(expr_str, &mut slab.ps)
            .map_err(|e| format!("Parse Error: {:?}", e))?;
        let instruction = expr.from(&slab.ps).compile(&slab.ps, &mut slab.cs);
        Ok(Self {
            instruction: instruction,
            slab: slab,
            values_cache: (OrderedFloat(f64::NEG_INFINITY), BTreeMap::new()),
            expr_str: expr_str.to_string(),
        })
    }

    pub fn eval(&mut self, filtration: &ScenarioFiltration, t: OrderedFloat<f64>) -> Result<f64, fasteval::Error> {
        // 1. Try to get a mutable reference from the cache immediately.
        // If it exists, this is a Zero-Allocation, Zero-Clone path.
        if t != self.values_cache.0 {
            // Cache Invalidation: Clear the map if the time has changed.
            self.values_cache.0 = t;
            self.values_cache.1.clear();
            self.values_cache.1.insert("t".to_string(), t.into_inner());
            let t_idx = filtration.get_time_idx(t).copied().unwrap_or(0);
            for (p_name, p_idx) in filtration.process_universe.process_registry.iter() {
                self.values_cache.1.insert(p_name.clone(), filtration.get(t_idx, *p_idx));
            }
        } 
        self.instruction.eval(&self.slab, &mut self.values_cache.1)
    }
}