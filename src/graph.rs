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
    grad: Value
}
impl Leaf {
    pub fn new(value: Value, requires_grad: bool) -> Self {
        let grad = gloss_tensor::full(value.shape(), 0.0);
        Self {
            value,
            requires_grad,
            grad
        }
    }
}
impl ComputationNode for Leaf {
    fn evaluate(&self) -> Value { self.value.clone() }
    fn requires_grad(&self) -> bool { self.requires_grad }
    fn backward(&mut self, gradient: Value) {
        self.grad = gloss_tensor::add(&self.grad, &gradient).unwrap();
    }
}
