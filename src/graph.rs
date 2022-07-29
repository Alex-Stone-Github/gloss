pub trait Evaluable {
    fn eval(&self) -> f32;
}

// Leaf
pub struct Leaf {
    value: f32,
    requires_grad: bool,  // naughty dead code will be used later
}
impl Leaf {
    pub fn new(value: f32) -> Self {
        Self{value, requires_grad: false}
    }
}
impl Evaluable for Leaf {
    fn eval(&self) -> f32 {
        self.value
    }
}


// Ops
pub struct Addition<'a> {
    a: &'a dyn Evaluable,
    b: &'a dyn Evaluable,
}
impl<'a> Addition<'a> {
    pub fn new(a: &'a dyn Evaluable, b: &'a dyn Evaluable) -> Self {
        Self{a, b}
    }
}
impl<'a> Evaluable for Addition<'a> {
    fn eval(&self) -> f32 {
        self.a.eval() + self.b.eval()
    }
}
pub struct Subtraction<'a> {
    a: &'a dyn Evaluable,
    b: &'a dyn Evaluable,
}
impl<'a> Subtraction<'a> {
    pub fn new(a: &'a dyn Evaluable, b: &'a dyn Evaluable) -> Self {
        Self{a, b}
    }
}
impl<'a> Evaluable for Subtraction<'a> {
    fn eval(&self) -> f32 {
        self.a.eval() - self.b.eval()
    }
}
pub struct Multiplication<'a> {
    a: &'a dyn Evaluable,
    b: &'a dyn Evaluable,
}
impl<'a> Multiplication<'a> {
    pub fn new(a: &'a dyn Evaluable, b: &'a dyn Evaluable) -> Self {
        Self{a, b}
    }
}
impl<'a> Evaluable for Multiplication<'a> {
    fn eval(&self) -> f32 {
        self.a.eval() * self.b.eval()
    }
}
