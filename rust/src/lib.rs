mod graph;
use graph::Evaluable;

pub fn hi() {
    let a: graph::Leaf = graph::Leaf::new(20.0);
    let b: graph::Leaf = graph::Leaf::new(30.0);
    let add: graph::Addition = graph::Addition
        ::new(Box::new(&a), Box::new(b));
    println!("{}", add.eval());
}


