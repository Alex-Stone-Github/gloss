mod graph;

pub fn hi() {
    println!("Hi");
}

#[cfg(test)]
mod test {
    #[test]
    fn correct_values_and_requires_grads() {
        let leaf1 = crate::graph::Leaf::new(3.3, true);
        let leaf2 = crate::graph::Leaf::new(6.6, true);

        let opr = crate::graph::Addition::new(&leaf1, &leaf2);


        {
            use crate::graph::ComputationGraphNode;
            println!("{}{}", leaf1.requires_grad(), leaf1.evaluate());
            assert_eq!(leaf1.requires_grad(), true);
            assert_eq!(leaf1.evaluate(), 3.3);
            println!("{}{}", leaf2.requires_grad(), leaf2.evaluate());
            assert_eq!(leaf2.requires_grad(), true);
            assert_eq!(leaf2.evaluate(), 6.6);
            println!("{}{}", opr.requires_grad(), opr.evaluate());
            assert_eq!(opr.requires_grad(), true);
            assert_eq!(opr.evaluate(), 9.9);
        }
    }
}
