//! Adamant is a high-performance tensor library written in pure Rust.
//!
//! This library provides efficient data structures and operations for numerical
//! computing, with no external dependencies. It supports various memory layouts,
//! views, slicing, and mathematical operations on multi-dimensional arrays.

pub mod core;

// Re-export commonly used items for convenience
pub use crate::core::tensors::Tensor;
pub use crate::core::shape::MemoryLayout;


pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
