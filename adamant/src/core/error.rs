//! Error types and utilities for the Adamant tensor library.
//!
//! This module provides the error types and result wrappers used throughout
//! the library for handling various exceptional conditions.

use std::fmt;

/// Error types for tensor operations
#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    /// Error related to incompatible shapes
    ShapeError(String),
    /// Error related to invalid indices
    IndexError(String),
    /// Error related to data type conversions
    TypeError(String),
    /// Error related to unsupported operations
    OperationError(String),
    /// Error related to memory access
    MemoryError(String),
    /// Other errors
    Other(String),
}

/// Result type for tensor operations
pub type TensorResult<T> = Result<T, TensorError>;