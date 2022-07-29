mod graph;
use graph::Evaluable;

pub fn hi() {
    let a: graph::Leaf = graph::Leaf::new(20.0);
    let b: graph::Leaf = graph::Leaf::new(30.0);
    let adder = graph::Multiplication::new(&a, &b);

    println!("{}", a.eval());
    println!("{}", b.eval());
    println!("{}", adder.eval());
}


