//! Core functionality for the Adamant tensor library.
//!
//! This module contains the fundamental data structures and operations that
//! power the Adamant library, including tensor implementation, shape management,
//! storage backends, and error handling.

pub mod error;
pub mod shape;
pub mod storage;
pub mod tensors;

// Re-exports of core types for convenience
pub use self::error::{TensorError, TensorResult};
pub use self::shape::{MemoryLayout, TensorShape};
pub use self::storage::TensorStorage;
pub use self::tensors::Tensor;