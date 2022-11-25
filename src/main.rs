/*
This file is just going to be used for testing out features during development.
*/

fn main() {
    //gloss::hi();

    let mut w = gloss::variable(gloss_tensor::random_norm(&[3, 3]));
    let mut x = gloss::constant(gloss_tensor::from_flat_vec(&[3, 1], vec![
        1.0,
        2.0,
        3.0
    ]).unwrap());
    let y = gloss_tensor::from_flat_vec(&[3, 1], vec![
        3.0,
        2.0,
        1.0
    ]).unwrap();

    {
        use gloss::graph::ComputationNode;
        let mut output = gloss::operation::Matmul::new(&mut w, &mut x);
        let error = gloss_tensor::sub(&y, &output.evaluate()).unwrap();
        output.backward(error);
    }
    println!("{:?}", w);
    println!("{:?}", x);
    println!("{:?}", y);
    {
    }
}
