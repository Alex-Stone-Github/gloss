pub trait Evaluable {
    fn eval(&self) -> f32;
}
// Ops
pub struct Addition<'a> {
    a: &'a dyn Evaluable,
    b: &'a dyn Evaluable,
}
impl Addition {
    pub fn new(a: &dyn Evaluable, b: &dyn Evaluable) -> Self {
        Self{a: a, b: b}
    }
}
impl Evaluable for Addition {
    fn eval(&self) -> f32 {
        self.a.eval() + self.b.eval()
    }
}

// Leaf
pub struct Leaf {
    value: f32,
    pub requires_grad: bool,
}
impl Leaf {
    pub fn new(value: f32) -> Self {
        Self{value: value, requires_grad: false}
    }
}
impl Evaluable for Leaf {
    fn eval(&self) -> f32 {
        self.value
    }
}
