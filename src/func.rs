use fasteval::{Compiler, Evaler, Instruction, Slab};
use ordered_float::OrderedFloat;
use crate::filtration::ScenarioFiltration;


pub struct Function {
    instruction: Instruction,
    slab: Slab,
    expr_str: String,
}

impl Clone for Function {
    fn clone(&self) -> Self {
        Self::new(&self.expr_str).expect("Failed to re-compile on clone")
    }
}

impl Function {
    pub fn new(expr_str: &str) -> Result<Self, String> {
        let parser = fasteval::Parser::new();
        let mut slab = Slab::new();
        let expr = parser
            .parse(expr_str, &mut slab.ps)
            .map_err(|e| format!("Parse Error: {:?}", e))?;
        let instruction = expr.from(&slab.ps).compile(&slab.ps, &mut slab.cs);
        Ok(Self {
            instruction: instruction,
            slab: slab,
            expr_str: expr_str.to_string(),
        })
    }

    pub fn eval(&self, t: OrderedFloat<f64>, filtration: &mut ScenarioFiltration) -> Result<f64, fasteval::Error> {
        if t != filtration.cache.time {
            filtration.refresh_cache(t);
        } 
        self.instruction.eval(&self.slab, &mut filtration.cache.values)
    }
}