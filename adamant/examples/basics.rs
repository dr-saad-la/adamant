//! Basic tensor operations example
//!
//! This example demonstrates creating tensors and performing simple operations.

use adamant::Tensor;

fn main() {
    // Create a 2x3 tensor with default values
    let mut tensor = Tensor::<f32>::new(&[2, 3]);
    println!("New tensor: {:?}", tensor);

    // Fill with data
    for i in 0..2 {
        for j in 0..3 {
            tensor.set(&[i, j], (i * 3 + j) as f32).unwrap();
        }
    }
    println!("Filled tensor: {:?}", tensor);

    // Reshape to 3x2
    let reshaped = tensor.reshape(&[3, 2]).unwrap();
    println!("Reshaped tensor: {:?}", reshaped);
}