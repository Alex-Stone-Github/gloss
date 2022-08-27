//
// Alex Stone, Testing
//

type Value = f64;


pub trait ComputationGraphNode {
    fn evaluate(&self) -> Value;
    fn requires_grad(&self) -> bool;
    fn backward(&mut self) -> (); // weird placeholder
}



pub struct Leaf {
    value: Value,
    requires_grad: bool,
}
impl Leaf {
    pub fn new(value: Value, requires_grad: bool) -> Self {
        Self {
            value,
            requires_grad,
        }
    }
}
impl ComputationGraphNode for Leaf {
    fn evaluate(&self) -> Value { self.value }
    fn requires_grad(&self) -> bool { self.requires_grad }
    fn backward(&mut self) {
        self.value = 3.3;
        return ();
    }
}


// one thing to note about this struct is that it can only exist when its internal references live
// as long as the struct instance
pub struct Addition<'a> {
    refa: &'a dyn ComputationGraphNode,
    refb: &'a dyn ComputationGraphNode,
}
impl<'a> Addition<'a> { // this creates a new Self given args of a generic lifetime
    pub fn new(a: &'a dyn ComputationGraphNode, b: &'a dyn ComputationGraphNode) -> Self {
        Self {
            refa: a,
            refb: b,
        }
    }
}
impl<'a> ComputationGraphNode for Addition<'a> { // this is just a generic lifetime implementation
    fn evaluate(&self) -> Value {
        self.refa.evaluate() + self.refb.evaluate()
    }
    fn requires_grad(&self) -> bool {
        self.refa.requires_grad() || self.refb.requires_grad()
    }
    fn backward(&mut self) -> () {
        ()
    }
}






