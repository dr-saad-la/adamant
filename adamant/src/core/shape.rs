//! Shape and memory layout management for tensors.
//!
//! This module provides structures and utilities for managing tensor dimensions,
//! strides, and memory layouts, which determine how multi-dimensional data is
//! stored in memory.
#![allow(unused_imports)]
use crate::core::error::{TensorError, TensorResult};

/// Memory layout options for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayout {
    /// Row-major (C-style) layout - elements in the same row are contiguous
    RowMajor,
    /// Column-major (Fortran-style) layout - elements in the same column are contiguous
    ColumnMajor,
}

impl Default for MemoryLayout {
    fn default() -> Self {
        // Default to row-major as it's most common in Rust
        MemoryLayout::RowMajor
    }
}

/// Represents the shape and strides of a tensor
///
/// The `TensorShape` struct maintains the dimensions of a tensor along with the strides
/// (steps between elements) for each dimension based on the memory layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    /// The dimensions of the tensor (e.g., [2, 3, 4] for a 2×3×4 tensor)
    dims: Vec<usize>,
    /// The strides for each dimension (used for indexing)
    strides: Vec<usize>,
    /// The memory layout used by this tensor
    layout: MemoryLayout,
}

impl TensorShape {
    /// Creates a new TensorShape with the given dimensions using the default layout (row-major)
    ///
    /// # Examples
    ///
    /// ```
    /// use adamant::core::shape::TensorShape;
    ///
    /// let shape = TensorShape::new(&[2, 3]);
    /// assert_eq!(shape.dims(), &[2, 3]);
    /// assert_eq!(shape.strides(), &[3, 1]); // Row-major strides
    /// ```
    pub fn new(dims: &[usize]) -> Self {
        Self::with_layout(dims, MemoryLayout::default())
    }

    /// Creates a new TensorShape with the given dimensions and memory layout
    ///
    /// # Examples
    ///
    /// ```
    /// use adamant::core::shape::{TensorShape, MemoryLayout};
    ///
    /// let row_major = TensorShape::with_layout(&[2, 3], MemoryLayout::RowMajor);
    /// assert_eq!(row_major.strides(), &[3, 1]);
    ///
    /// let col_major = TensorShape::with_layout(&[2, 3], MemoryLayout::ColumnMajor);
    /// assert_eq!(col_major.strides(), &[1, 2]);
    /// ```
    pub fn with_layout(dims: &[usize], layout: MemoryLayout) -> Self {
        let dims = dims.to_vec();
        let strides = match layout {
            MemoryLayout::RowMajor => Self::compute_row_major_strides(&dims),
            MemoryLayout::ColumnMajor => Self::compute_column_major_strides(&dims),
        };

        TensorShape { dims, strides, layout }
    }

    /// Computes strides for row-major (C-style) layout
    ///
    /// Row-major layout stores elements of the same row contiguously in memory.
    /// This is the default layout in C, C++, and Rust.
    fn compute_row_major_strides(dims: &[usize]) -> Vec<usize> {
        if dims.is_empty() {
            return Vec::new();
        }

        let mut strides = vec![1; dims.len()];

        // Compute strides right-to-left (excluding the last dimension)
        for i in (0..dims.len()-1).rev() {
            strides[i] = strides[i+1] * dims[i+1];
        }

        strides
    }

    /// Computes strides for column-major (Fortran-style) layout
    ///
    /// Column-major layout stores elements of the same column contiguously in memory.
    /// This is the default layout in Fortran, MATLAB, and many scientific computing libraries.
    fn compute_column_major_strides(dims: &[usize]) -> Vec<usize> {
        if dims.is_empty() {
            return Vec::new();
        }

        let mut strides = vec![1; dims.len()];

        // Compute strides left-to-right (excluding the first dimension)
        for i in 1..dims.len() {
            strides[i] = strides[i-1] * dims[i-1];
        }

        strides
    }

    /// Gets the memory layout used by this tensor
    #[inline]
    pub fn layout(&self) -> MemoryLayout {
        self.layout
    }

    /// Gets the total number of elements in the tensor
    ///
    /// This is the product of all dimension sizes.
    #[inline]
    pub fn size(&self) -> usize {
        self.dims.iter().product()
    }

    /// Gets the rank (number of dimensions) of the tensor
    #[inline]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Gets the dimensions of the tensor
    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Gets the strides of the tensor
    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Calculates the flat index for the given multi-dimensional indices
    ///
    /// # Errors
    ///
    /// Returns a `TensorError::IndexError` if:
    /// - The number of indices doesn't match the tensor's rank
    /// - Any index is out of bounds for its dimension
    pub fn get_flat_index(&self, indices: &[usize]) -> TensorResult<usize> {
        if indices.len() != self.rank() {
            return Err(TensorError::IndexError(format!(
                "Expected {} indices, got {}",
                self.rank(), indices.len()
            )));
        }

        // Check if any index is out of bounds
        for (i, (&idx, &dim)) in indices.iter().zip(self.dims.iter()).enumerate() {
            if idx >= dim {
                return Err(TensorError::IndexError(format!(
                    "Index {} is out of bounds for dimension {} with size {}",
                    idx, i, dim
                )));
            }
        }

        // Calculate the flat index using strides
        let flat_idx: usize = indices.iter()
            .zip(self.strides.iter())
            .map(|(&idx, &stride)| idx * stride)
            .sum();

        if flat_idx >= self.size() {
            return Err(TensorError::IndexError(format!(
                "Computed flat index {} is out of bounds for tensor with size {}",
                flat_idx, self.size()
            )));
        }

        Ok(flat_idx)
    }

    /// Checks if this shape is compatible with another shape for broadcasting
    pub fn is_broadcast_compatible_with(&self, other: &TensorShape) -> bool {
        // Broadcasting rules: dimensions align from the right
        // Each dimension must be either equal or one of them must be 1
        let self_dims = self.dims();
        let other_dims = other.dims();

        let self_rank = self_dims.len();
        let other_rank = other_dims.len();

        let max_rank = std::cmp::max(self_rank, other_rank);

        for i in 0..max_rank {
            // Get dimensions, treating missing dimensions as 1
            let self_dim = if i < self_rank { self_dims[self_rank - 1 - i] } else { 1 };
            let other_dim = if i < other_rank { other_dims[other_rank - 1 - i] } else { 1 };

            // Dimensions must be equal or one of them must be 1
            if self_dim != other_dim && self_dim != 1 && other_dim != 1 {
                return false;
            }
        }

        true
    }
}