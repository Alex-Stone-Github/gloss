
pub use gloss_tensor::*;

pub mod graph;
pub mod operation;

pub fn hi() {
    println!("Hi");

    let mut leaf1 = graph::Leaf::new(gloss_tensor::full(&[1], 3.3), true);
    let mut leaf2 = graph::Leaf::new(gloss_tensor::full(&[1], 6.6), false);
    let mut leaf3 = graph::Leaf::new(gloss_tensor::full(&[1], 2.0), true);
    {
        let mut opr_mul = operation::Addition::new(&mut leaf1, &mut leaf2);
        let mut opr_add = operation::Multiplication::new(&mut opr_mul, &mut leaf3);
        opr_add.backward(gloss_tensor::full(&[1], 1.0));
    }
    use crate::graph::ComputationNode;
    println!("{:?}", leaf1);
    println!("{:?}", leaf2);
    println!("{:?}", leaf3);

    // tranpose test
    let t = gloss_tensor::range(&[3, 2]).map(|x| x + 1);
    println!("{:?}", t);
    let transposed = gloss_tensor::transpose(&t).unwrap();
    println!("{:?}", transposed);
    let retransposed = gloss_tensor::transpose(&transposed).unwrap();
    println!("{:?}", retransposed);


    {
        let mut w = graph::Leaf::variable(gloss_tensor::random_norm(&[3, 3]), true);
        let mut x = graph::Leaf::constant(gloss_tensor::random_norm(&[3, 1]), false);
        {
            let mut y = operation::Matmul::new(&mut w, &mut x);
            println!("done");
            y.backward(gloss_tensor::full(&[3, 1], 1.0));
        }
        println!("{:?}", x);
        println!("{:?}", w);

    }


}

pub fn constant(value: graph::Value) -> graph::Leaf {
    graph::Leaf::new(value, false) // does not require grad
}
pub fn variable(value: graph::Value) -> graph::Leaf {
    graph::Leaf::new(value, true) // does require grad
}





/*
#[cfg(test)]
mod test {
    #[test]
    fn correct_values_and_requires_grads() {
        let mut leaf1 = crate::graph::Leaf::new(gloss_tensor::full(&[1], 3.3), true);
        let mut leaf2 = crate::graph::Leaf::new(gloss_tensor::full(&[1], 6.6), true);

        {
            let mut opr = crate::operation::Addition::new(&mut leaf1, &mut leaf2);
        }

        use crate::graph::ComputationNode;
        println!("{:?}", leaf1);
        println!("{:?}", leaf2);
    }
}
*/
