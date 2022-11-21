//
// Alex Stone, Testing
//
pub type Value = gloss_tensor::Tensor<f64>;

pub trait ComputationNode {
    fn evaluate(&self) -> Value;
    fn requires_grad(&self) -> bool;
    fn backward(&mut self, gradient: Value); // weird placeholder
}

#[derive(Debug)]
pub struct Leaf {
    value: Value,
    requires_grad: bool,
    grad: Option<Value>
}
impl Leaf {
    pub fn new(value: Value, requires_grad: bool) -> Self {
        let grad = Some(gloss_tensor::full(value.shape(), 0.0));
        Self {
            value,
            requires_grad,
            grad
        }
    }
    pub fn variable(value: Value, requires_grad: bool) -> Self {
        Self::new(value, requires_grad)
    }
    pub fn constant(value: Value, requires_grad: bool) -> Self {
        Self {
            value,
            requires_grad,
            grad: None
        }
    }
}
impl ComputationNode for Leaf {
    fn evaluate(&self) -> Value { self.value.clone() }
    fn requires_grad(&self) -> bool { self.requires_grad }
    fn backward(&mut self, gradient: Value) {
        if self.requires_grad {
            // this unwrap is safe because it will only ever happen if this leaf type is a variable
            self.grad = Some(gloss_tensor::add(&(self.grad.as_ref().unwrap()), &gradient).unwrap());
        }
        else {
            panic!("Something terribley wrong has happened");
        }
    }
}
