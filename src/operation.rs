use crate::graph::{ComputationNode, Value};

pub struct Addition<'a> {
    refa: &'a mut dyn ComputationNode,
    refb: &'a mut dyn ComputationNode,
}
impl<'a> Addition<'a> { // this creates a new Self given args of a generic lifetime
    pub fn new(a: &'a mut dyn ComputationNode, b: &'a mut dyn ComputationNode) -> Self {
        Self {
            refa: a,
            refb: b,
        }
    }
}
impl<'a> ComputationNode for Addition<'a> { // this is just a generic lifetime implementation
    fn evaluate(&self) -> Value {
        gloss_tensor::add(&self.refa.evaluate(), &self.refb.evaluate()).unwrap()
    }
    fn requires_grad(&self) -> bool {
        self.refa.requires_grad() || self.refb.requires_grad()
    }
    fn backward(&mut self, gradient: Value) -> () {
        if self.refa.requires_grad() {
            self.refa.backward(gradient.clone());
        }
        if self.refb.requires_grad() {
            self.refb.backward(gradient.clone());
        }
    }
}
pub struct Multiplication<'a> {
    refa: &'a mut dyn ComputationNode,
    refb: &'a mut dyn ComputationNode,
}
impl<'a> Multiplication<'a> { // this creates a new Self given args of a generic lifetime
    pub fn new(a: &'a mut dyn ComputationNode, b: &'a mut dyn ComputationNode) -> Self {
        Self {
            refa: a,
            refb: b,
        }
    }
}
impl<'a> ComputationNode for Multiplication<'a> { // this is just a generic lifetime implementation
    fn evaluate(&self) -> Value {
        gloss_tensor::mul(&self.refa.evaluate(), &self.refb.evaluate()).unwrap()
    }
    fn requires_grad(&self) -> bool {
        self.refa.requires_grad() || self.refb.requires_grad()
    }
    fn backward(&mut self, gradient: Value) -> () {
        if self.refa.requires_grad() {
            self.refa.backward(gloss_tensor::mul(&gradient.clone(), &self.refb.evaluate()).unwrap());
        }
        if self.refb.requires_grad() {
            self.refb.backward(gloss_tensor::mul(&gradient.clone(), &self.refa.evaluate()).unwrap());
        }
    }
}
/*
 * AB
 */
pub struct Matmul<'a> {
    refa: &'a mut dyn ComputationNode,
    refb: &'a mut dyn ComputationNode,
}
impl<'a> Matmul<'a> { // this creates a new Self given args of a generic lifetime
    pub fn new(a: &'a mut dyn ComputationNode, b: &'a mut dyn ComputationNode) -> Self {
        Self {
            refa: a,
            refb: b,
        }
    }
}
impl<'a> ComputationNode for Matmul<'a> { // this is just a generic lifetime implementation
    fn evaluate(&self) -> Value {
        gloss_tensor::matmul(&self.refa.evaluate(), &self.refb.evaluate()).unwrap()
    }
    fn requires_grad(&self) -> bool {
        self.refa.requires_grad() || self.refb.requires_grad()
    }
    fn backward(&mut self, gradient: Value) -> () {
        if self.refa.requires_grad() {
            self.refa.backward(gloss_tensor::matmul(&gradient, 
                                                    &gloss_tensor::transpose(&self.refb.evaluate())
                                                    .unwrap()).unwrap());
        }
        if self.refb.requires_grad() {
            self.refb.backward(gloss_tensor::matmul(&gradient, 
                                                    &gloss_tensor::transpose(&self.refa.evaluate())
                                                    .unwrap()).unwrap());
        }
    }
}
pub struct Subtraction<'a> {
    refa: &'a mut dyn ComputationNode,
    refb: &'a mut dyn ComputationNode,
}
impl<'a> Subtraction<'a> { // this creates a new Self given args of a generic lifetime
    pub fn new(a: &'a mut dyn ComputationNode, b: &'a mut dyn ComputationNode) -> Self {
        Self {
            refa: a,
            refb: b,
        }
    }
}
impl<'a> ComputationNode for Subtraction<'a> { // this is just a generic lifetime implementation
    fn evaluate(&self) -> Value {
        gloss_tensor::sub(&self.refa.evaluate(), &self.refb.evaluate()).unwrap()
    }
    fn requires_grad(&self) -> bool {
        self.refa.requires_grad() || self.refb.requires_grad()
    }
    fn backward(&mut self, gradient: Value) -> () {
        if self.refa.requires_grad() {
            self.refa.backward(gradient.clone());
        }
        if self.refb.requires_grad() {
            self.refb.backward(gradient.clone());
        }
    }
}



