//! Tensor implementation for the Adamant library.
//!
//! This module provides the core Tensor data structure, which represents
//! an n-dimensional array with various numeric operations. The implementation
//! supports both row-major (C-style) and column-major (Fortran-style) memory layouts,
//! flexible shape manipulation, views, slices, and efficient element access.
#![allow(unused_imports)]
#![allow(dead_code)]
use std::fmt;
use std::ops::{Add, Mul, Sub, Div};
use std::sync::Arc;
use std::marker::PhantomData;

use crate::core::error::{TensorError, TensorResult};
use crate::core::shape::{MemoryLayout, TensorShape};
use crate::core::storage::TensorStorage;

/// A view of a tensor that can reference a subset of the original tensor's data
///
/// The `TensorView` struct provides a lightweight view into a tensor,
/// allowing operations on subsets of tensor data without copying.
#[derive(Clone)]
pub struct TensorView<'a, T> {
    /// The shape of the view
    shape: TensorShape,
    /// Reference to the original tensor's data
    data: &'a [T],
    /// Starting offset in the original tensor's data
    offset: usize,
}

impl<'a, T> TensorView<'a, T> {
    /// Creates a new tensor view
    pub fn new(data: &'a [T], shape: TensorShape, offset: usize) -> TensorResult<Self> {
        let required_size = offset + shape.size();
        if required_size > data.len() {
            return Err(TensorError::IndexError(format!(
                "View would access {} elements, but tensor only has {} elements",
                required_size, data.len()
            )));
        }

        Ok(TensorView { shape, data, offset })
    }

    /// Gets the shape of the view
    #[inline]
    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    /// Gets the dimensions of the view
    #[inline]
    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Gets the total number of elements in the view
    #[inline]
    pub fn size(&self) -> usize {
        self.shape.size()
    }

    /// Gets the rank (number of dimensions) of the view
    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Gets an element by multi-dimensional index
    pub fn get(&self, indices: &[usize]) -> TensorResult<&T> {
        let flat_idx = self.shape.get_flat_index(indices)?;
        let abs_idx = self.offset + flat_idx;

        if abs_idx >= self.data.len() {
            return Err(TensorError::IndexError(format!(
                "Computed absolute index {} is out of bounds for tensor data with length {}",
                abs_idx, self.data.len()
            )));
        }

        Ok(&self.data[abs_idx])
    }
}

/// The base tensor data structure
///
/// `Tensor<T>` represents an n-dimensional array of elements of type `T`.
/// It supports various memory layouts and mathematical operations.
#[derive(Clone)]
pub struct Tensor<T> {
    /// The shape of the tensor
    shape: TensorShape,
    /// The actual data storage
    storage: TensorStorage<T>,
}

impl<T: Clone + Default> Tensor<T> {
    /// Creates a new tensor with the given shape and default layout, filled with the default value
    ///
    /// # Examples
    ///
    /// ```
    /// use adamant::core::tensors::Tensor;
    ///
    /// let tensor = Tensor::<f32>::new(&[2, 3]);
    /// assert_eq!(tensor.dims(), &[2, 3]);
    /// assert_eq!(tensor.size(), 6);
    /// ```
    pub fn new(dims: &[usize]) -> Self {
        Self::with_layout(dims, MemoryLayout::default())
    }

    /// Creates a new tensor with the given shape and memory layout, filled with the default value
    ///
    /// # Examples
    ///
    /// ```
    /// use adamant::core::tensors::Tensor;
    /// use adamant::core::shape::MemoryLayout;
    ///
    /// let tensor = Tensor::<f32>::with_layout(&[2, 3], MemoryLayout::ColumnMajor);
    /// assert_eq!(tensor.layout(), MemoryLayout::ColumnMajor);
    /// ```
    pub fn with_layout(dims: &[usize], layout: MemoryLayout) -> Self {
        let shape = TensorShape::with_layout(dims, layout);
        let size = shape.size();
        let data = vec![T::default(); size];

        Tensor {
            shape,
            storage: TensorStorage::new(data),
        }
    }

    /// Creates a new tensor filled with the specified value
    ///
    /// # Examples
    ///
    /// ```
    /// use adamant::core::tensors::Tensor;
    ///
    /// let tensor = Tensor::filled_with(&[2, 3], 42.0);
    /// for i in 0..2 {
    ///     for j in 0..3 {
    ///         assert_eq!(*tensor.get(&[i, j]).unwrap(), 42.0);
    ///     }
    /// }
    /// ```
    pub fn filled_with(dims: &[usize], value: T) -> Self {
        Self::filled_with_layout(dims, value, MemoryLayout::default())
    }

    /// Creates a new tensor with the specified layout, filled with the specified value
    pub fn filled_with_layout(dims: &[usize], value: T, layout: MemoryLayout) -> Self {
        let shape = TensorShape::with_layout(dims, layout);
        let size = shape.size();
        let data = vec![value; size];

        Tensor {
            shape,
            storage: TensorStorage::new(data),
        }
    }

    /// Creates a new tensor from existing data with the given shape and default layout
    ///
    /// # Examples
    ///
    /// ```
    /// use adamant::core::tensors::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let tensor = Tensor::from_vec(data, &[2, 3]).unwrap();
    /// assert_eq!(*tensor.get(&[0, 0]).unwrap(), 1.0);
    /// assert_eq!(*tensor.get(&[1, 2]).unwrap(), 6.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a `TensorError::ShapeError` if the length of `data` does not match
    /// the number of elements required by `dims`.
    pub fn from_vec(data: Vec<T>, dims: &[usize]) -> TensorResult<Self> {
        Self::from_vec_with_layout(data, dims, MemoryLayout::default())
    }

    /// Creates a new tensor from existing data with the given shape and memory layout
    ///
    /// # Examples
    ///
    /// ```
    /// use adamant::core::tensors::Tensor;
    /// use adamant::core::shape::MemoryLayout;
    ///
    /// // In column-major format, this is [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    /// let col_data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    /// let tensor = Tensor::from_vec_with_layout(
    ///     col_data, &[2, 3], MemoryLayout::ColumnMajor
    /// ).unwrap();
    ///
    /// // Access is always logical, regardless of layout
    /// assert_eq!(*tensor.get(&[0, 0]).unwrap(), 1.0);
    /// assert_eq!(*tensor.get(&[1, 2]).unwrap(), 6.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a `TensorError::ShapeError` if the length of `data` does not match
    /// the number of elements required by `dims`.
    pub fn from_vec_with_layout(data: Vec<T>, dims: &[usize], layout: MemoryLayout) -> TensorResult<Self> {
        let shape = TensorShape::with_layout(dims, layout);
        let expected_size = shape.size();

        if data.len() != expected_size {
            return Err(TensorError::ShapeError(format!(
                "Data length {} does not match expected size {} for shape {:?}",
                data.len(), expected_size, dims
            )));
        }

        Ok(Tensor {
            shape,
            storage: TensorStorage::new(data),
        })
    }

    /// Gets the memory layout used by this tensor
    #[inline]
    pub fn layout(&self) -> MemoryLayout {
        self.shape.layout()
    }

    /// Creates a tensor with the same data but in the specified memory layout
    ///
    /// # Examples
    ///
    /// ```
    /// use adamant::core::tensors::Tensor;
    /// use adamant::core::shape::MemoryLayout;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let row_tensor = Tensor::from_vec(data, &[2, 3]).unwrap();
    /// let col_tensor = row_tensor.to_layout(MemoryLayout::ColumnMajor);
    ///
    /// // Access is consistent regardless of layout
    /// assert_eq!(*col_tensor.get(&[0, 0]).unwrap(), 1.0);
    /// assert_eq!(*col_tensor.get(&[1, 2]).unwrap(), 6.0);
    ///
    /// // But the raw data is different
    /// assert_eq!(col_tensor.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    /// ```
    pub fn to_layout(&self, layout: MemoryLayout) -> Self {
        if self.layout() == layout {
            // If already in the requested layout, just clone
            return self.clone();
        }

        match self.rank() {
            0 => {
                // Scalar tensor - layout doesn't matter
                let mut new_data = Vec::with_capacity(1);
                if let Some(value) = self.storage.data().get(0) {
                    new_data.push(value.clone());
                }
                Tensor::from_vec_with_layout(new_data, &[], layout)
                    .expect("Scalar conversion should always succeed")
            },
            1 => {
                // Vector - layout doesn't matter
                let new_data = self.storage.data().to_vec();
                Tensor::from_vec_with_layout(new_data, self.dims(), layout)
                    .expect("Vector conversion should maintain size")
            },
            2 => {
                // Matrix - special case for efficiency
                self.convert_matrix_layout(layout)
            },
            _ => {
                // General case - create new storage with data in the target layout
                let mut new_data = Vec::with_capacity(self.size());

                // For row-major to column-major or vice versa, we need to iterate
                // through elements in the target layout order
                if (self.layout() == MemoryLayout::RowMajor && layout == MemoryLayout::ColumnMajor) ||
                    (self.layout() == MemoryLayout::ColumnMajor && layout == MemoryLayout::RowMajor) {
                    let _new_shape = TensorShape::with_layout(self.dims(), layout);

                    // Efficiently convert layouts without individual get() calls
                    let old_shape = &self.shape;
                    let old_data = self.data();

                    // Pre-allocate array for current indices
                    let mut indices = vec![0; self.rank()];

                    // Iterate through all elements in target layout order
                    for _ in 0..self.size() {
                        // Calculate flat index in old layout
                        // let old_flat_idx = indices.iter()
                        //     .zip(old_shape.strides().iter())
                        //     .map(|(&idx, &stride)| idx * stride)
                        //     .sum();
                        let old_flat_idx: usize = indices.iter()
                            .zip(old_shape.strides().iter())
                            .map(|(&idx, &stride)| idx * stride)
                            .sum();

                        // Get the element and add to new data
                        new_data.push(old_data.get(old_flat_idx).unwrap().clone());
                        // Increment indices based on target layout
                        self.increment_indices(&mut indices, layout);
                    }
                } else {
                    // Just clone the data if layouts are the same
                    new_data = self.storage.data().to_vec();
                }

                Tensor::from_vec_with_layout(new_data, self.dims(), layout)
                    .expect("Layout conversion should maintain tensor size")
            }
        }
    }

    /// Helper method to increment indices in the specified layout order
    fn increment_indices(&self, indices: &mut [usize], layout: MemoryLayout) {
        let dims = self.dims();
        match layout {
            MemoryLayout::RowMajor => {
                // For row-major, increment from right to left
                let mut i = dims.len() - 1;
                loop {
                    indices[i] += 1;
                    if indices[i] < dims[i] {
                        break;
                    }
                    indices[i] = 0;
                    if i == 0 {
                        // This should never happen if we're iterating the correct number of times
                        break;
                    }
                    i -= 1;
                }
            }
            MemoryLayout::ColumnMajor => {
                // For column-major, increment from left to right
                let mut i = 0;
                loop {
                    indices[i] += 1;
                    if indices[i] < dims[i] {
                        break;
                    }
                    indices[i] = 0;
                    if i == dims.len() - 1 {
                        // This should never happen if we're iterating the correct number of times
                        break;
                    }
                    i += 1;
                }
            }
        }
    }

    /// Specialized, optimized method for converting 2D matrix layouts
    fn convert_matrix_layout(&self, layout: MemoryLayout) -> Self {
        let dims = self.dims();
        if dims.len() != 2 {
            panic!("convert_matrix_layout should only be called for matrices");
        }

        let rows = dims[0];
        let cols = dims[1];
        let mut new_data = Vec::with_capacity(rows * cols);

        if self.layout() == MemoryLayout::RowMajor && layout == MemoryLayout::ColumnMajor {
            // Row-major to column-major
            for c in 0..cols {
                for r in 0..rows {
                    let old_idx = r * cols + c;
                    new_data.push(self.storage.data()[old_idx].clone());
                }
            }
        } else if self.layout() == MemoryLayout::ColumnMajor && layout == MemoryLayout::RowMajor {
            // Column-major to row-major
            for r in 0..rows {
                for c in 0..cols {
                    let old_idx = c * rows + r;
                    new_data.push(self.storage.data()[old_idx].clone());
                }
            }
        } else {
            // Same layout, just clone
            new_data = self.storage.data().to_vec();
        }

        Tensor::from_vec_with_layout(new_data, dims, layout)
            .expect("Matrix layout conversion should maintain size")
    }

    /// Gets the shape of the tensor
    #[inline]
    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    /// Gets the dimensions of the tensor
    #[inline]
    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Gets the total number of elements in the tensor
    #[inline]
    pub fn size(&self) -> usize {
        self.shape.size()
    }

    /// Gets the rank (number of dimensions) of the tensor
    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Gets a reference to the underlying data
    #[inline]
    pub fn data(&self) -> &[T] {
        self.storage.data()
    }

    /// Gets a mutable reference to the underlying data (if possible)
    ///
    /// Returns `None` if the tensor has shared storage.
    #[inline]
    pub fn data_mut(&mut self) -> Option<&mut [T]> {
        self.storage.data_mut()
    }

    /// Creates a tensor with the same data but reshaped to new dimensions
    ///
    /// # Examples
    ///
    /// ```
    /// use adamant::core::tensors::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let tensor = Tensor::from_vec(data, &[2, 3]).unwrap();
    /// let reshaped = tensor.reshape(&[3, 2]).unwrap();
    ///
    /// assert_eq!(reshaped.dims(), &[3, 2]);
    /// assert_eq!(reshaped.size(), 6);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a `TensorError::ShapeError` if the new shape requires a different
    /// number of elements than the current shape.
    pub fn reshape(&self, new_dims: &[usize]) -> TensorResult<Self> {
        let new_shape = TensorShape::with_layout(new_dims, self.layout());

        if new_shape.size() != self.shape.size() {
            return Err(TensorError::ShapeError(format!(
                "Cannot reshape tensor of size {} to new shape with size {}",
                self.shape.size(), new_shape.size()
            )));
        }

        // Create a new tensor with the same storage but different shape
        Ok(Tensor {
            shape: new_shape,
            storage: self.storage.clone(),
        })
    }

    /// Creates a view of this tensor
    ///
    /// A view is a lightweight reference to the tensor's data with potentially different shape.
    pub fn view(&self) -> TensorView<T> {
        TensorView::new(self.storage.data(), self.shape.clone(), 0)
            .expect("View of full tensor should always be valid")
    }

    /// Creates a slice of this tensor along specified dimensions
    ///
    /// # Examples
    ///
    /// ```
    /// use adamant::core::tensors::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    /// let tensor = Tensor::from_vec(data, &[2, 2, 3]).unwrap();
    ///
    /// // Get the slice for the first element of the first dimension (a 2×3 matrix)
    /// let slice = tensor.slice(&[0], &[1]).unwrap();
    /// assert_eq!(slice.dims(), &[2, 3]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns errors if the indices are out of bounds or the dimensions are invalid.
    pub fn slice(&self, start_indices: &[usize], sizes: &[usize]) -> TensorResult<TensorView<T>> {
        // Validate that we don't have too many indices
        if start_indices.len() > self.rank() {
            return Err(TensorError::IndexError(format!(
                "Too many start indices: got {}, but tensor rank is {}",
                start_indices.len(), self.rank()
            )));
        }

        if sizes.len() > self.rank() {
            return Err(TensorError::IndexError(format!(
                "Too many size values: got {}, but tensor rank is {}",
                sizes.len(), self.rank()
            )));
        }

        // Special case for partial indexing with specific semantics:
        // If we provide only the first few dimensions, we're selecting a subregion
        // and want to drop the fully-sliced dimensions
        if start_indices.len() < self.rank() && sizes.len() <= start_indices.len() {
            // This is a partial indexing case like tensor.slice(&[0], &[1])
            // We want to reduce the dimensionality

            // Determine full start indices
            let mut full_start_indices = vec![0; self.rank()];
            for (i, &idx) in start_indices.iter().enumerate() {
                full_start_indices[i] = idx;
            }

            // Determine output shape
            let mut result_dims = Vec::new();

            // The first 'sizes.len()' dimensions are specified by the user
            for i in 0..sizes.len() {
                if sizes[i] > 1 {
                    result_dims.push(sizes[i]);
                }
            }

            // The remaining dimensions are copied from the original tensor
            for i in start_indices.len()..self.rank() {
                result_dims.push(self.dims()[i]);
            }

            // Check bounds with the actual sizes used
            let mut full_sizes = vec![0; self.rank()];
            for i in 0..sizes.len() {
                full_sizes[i] = sizes[i];
            }
            for i in sizes.len()..self.rank() {
                full_sizes[i] = self.dims()[i];
            }

            for i in 0..self.rank() {
                if full_start_indices[i] >= self.dims()[i] {
                    return Err(TensorError::IndexError(format!(
                        "Start index {} is out of bounds for dimension {} with size {}",
                        full_start_indices[i], i, self.dims()[i]
                    )));
                }

                if full_start_indices[i] + full_sizes[i] > self.dims()[i] {
                    return Err(TensorError::IndexError(format!(
                        "Slice exceeds bounds: start={} + size={} > dimension_size={} for dimension {}",
                        full_start_indices[i], full_sizes[i], self.dims()[i], i
                    )));
                }
            }

            // Calculate the offset into the data
            let offset = full_start_indices.iter()
                .zip(self.shape.strides().iter())
                .map(|(idx, stride)| idx * stride)
                .sum();

            // Create new shape for the slice
            let slice_shape = TensorShape::with_layout(&result_dims, self.layout());

            // Create the view
            return TensorView::new(self.storage.data(), slice_shape, offset);
        }

        // Normal case: explicit indices for all dimensions or preserving dimensionality
        let mut full_start_indices = vec![0; self.rank()];
        let mut full_sizes = vec![0; self.rank()];

        // Fill start indices (defaulting to 0 for missing values)
        for (i, &idx) in start_indices.iter().enumerate() {
            full_start_indices[i] = idx;
        }

        // Fill sizes (defaulting to remaining size in that dimension)
        for i in 0..self.rank() {
            let size = if i < sizes.len() {
                sizes[i]
            } else {
                // Default to full size
                self.dims()[i] - full_start_indices[i]
            };

            full_sizes[i] = size;
        }

        // Check bounds
        for i in 0..self.rank() {
            if full_start_indices[i] >= self.dims()[i] {
                return Err(TensorError::IndexError(format!(
                    "Start index {} is out of bounds for dimension {} with size {}",
                    full_start_indices[i], i, self.dims()[i]
                )));
            }

            if full_start_indices[i] + full_sizes[i] > self.dims()[i] {
                return Err(TensorError::IndexError(format!(
                    "Slice exceeds bounds: start={} + size={} > dimension_size={} for dimension {}",
                    full_start_indices[i], full_sizes[i], self.dims()[i], i
                )));
            }
        }

        // Calculate the offset into the data
        let offset = full_start_indices.iter()
            .zip(self.shape.strides().iter())
            .map(|(idx, stride)| idx * stride)
            .sum();

        // Create new shape for the slice
        let slice_shape = TensorShape::with_layout(&full_sizes, self.layout());

        // Create the view
        TensorView::new(self.storage.data(), slice_shape, offset)
    }

    /// Gets an element by multi-dimensional index
    ///
    /// # Examples
    ///
    /// ```
    /// use adamant::core::tensors::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let tensor = Tensor::from_vec(data, &[2, 3]).unwrap();
    ///
    /// assert_eq!(*tensor.get(&[0, 0]).unwrap(), 1.0);
    /// assert_eq!(*tensor.get(&[1, 2]).unwrap(), 6.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a `TensorError::IndexError` if:
    /// - The number of indices doesn't match the tensor's rank
    /// - Any index is out of bounds for its dimension
    #[inline]
    pub fn get(&self, indices: &[usize]) -> TensorResult<&T> {
        let flat_idx = self.shape.get_flat_index(indices)?;

        // This should never fail if get_flat_index passed
        Ok(&self.storage.data()[flat_idx])
    }

    /// Sets an element by multi-dimensional index
    ///
    /// # Examples
    ///
    /// ```
    /// use adamant::core::tensors::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let mut tensor = Tensor::from_vec(data, &[2, 3]).unwrap();
    ///
    /// tensor.set(&[0, 0], 10.0).unwrap();
    /// tensor.set(&[1, 2], 20.0).unwrap();
    ///
    /// assert_eq!(*tensor.get(&[0, 0]).unwrap(), 10.0);
    /// assert_eq!(*tensor.get(&[1, 2]).unwrap(), 20.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a `TensorError::IndexError` if:
    /// - The number of indices doesn't match the tensor's rank
    /// - Any index is out of bounds for its dimension
    ///
    /// Returns a `TensorError::OperationError` if:
    /// - The tensor has shared storage and cannot be modified
    pub fn set(&mut self, indices: &[usize], value: T) -> TensorResult<()> {
        let flat_idx = self.shape.get_flat_index(indices)?;

        match self.storage.data_mut() {
            Some(data) => {
                data[flat_idx] = value;
                Ok(())
            },
            None => Err(TensorError::OperationError(
                "Cannot modify tensor with shared storage".to_string()
            )),
        }
    }

    /// Creates a copy of this tensor with owned storage
    ///
    /// If the tensor already has owned storage, this is a shallow clone.
    /// If the tensor has shared storage, this creates a deep copy.
    pub fn to_owned(&self) -> Self {
        Tensor {
            shape: self.shape.clone(),
            storage: self.storage.clone().into_owned(),
        }
    }

    /// Creates a copy of this tensor with shared storage
    ///
    /// This enables efficient sharing of tensor data between multiple tensors.
    pub fn to_shared(&self) -> Self {
        match &self.storage {
            TensorStorage::Owned(data) => {
                Tensor {
                    shape: self.shape.clone(),
                    storage: TensorStorage::Shared(Arc::new(data.clone())),
                }
            },
            TensorStorage::Shared(_) => self.clone(),
        }
    }
}

// Implementation of mathematical operations

impl<T: Clone + Default + Add<Output = T>> Tensor<T> {
    /// Adds another tensor to this one (element-wise)
    ///
    /// # Examples
    ///
    /// ```
    /// use adamant::core::tensors::Tensor;
    ///
    /// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    /// let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();
    /// let c = a.add(&b).unwrap();
    ///
    /// assert_eq!(*c.get(&[0]).unwrap(), 5.0);
    /// assert_eq!(*c.get(&[1]).unwrap(), 7.0);
    /// assert_eq!(*c.get(&[2]).unwrap(), 9.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a `TensorError::ShapeError` if the shapes are not compatible.
    pub fn add(&self, other: &Tensor<T>) -> TensorResult<Tensor<T>> {
        self.element_wise_op(other, |a, b| a.clone() + b.clone())
    }
}

impl<T: Clone + Default + Sub<Output = T>> Tensor<T> {
    /// Subtracts another tensor from this one (element-wise)
    ///
    /// # Examples
    ///
    /// ```
    /// use adamant::core::tensors::Tensor;
    ///
    /// let a = Tensor::from_vec(vec![5.0, 7.0, 9.0], &[3]).unwrap();
    /// let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    /// let c = a.sub(&b).unwrap();
    ///
    /// assert_eq!(*c.get(&[0]).unwrap(), 4.0);
    /// assert_eq!(*c.get(&[1]).unwrap(), 5.0);
    /// assert_eq!(*c.get(&[2]).unwrap(), 6.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a `TensorError::ShapeError` if the shapes are not compatible.
    pub fn sub(&self, other: &Tensor<T>) -> TensorResult<Tensor<T>> {
        self.element_wise_op(other, |a, b| a.clone() - b.clone())
    }
}

impl<T: Clone + Default + Mul<Output = T>> Tensor<T> {
    /// Multiplies this tensor by another (element-wise)
    ///
    /// # Examples
    ///
    /// ```
    /// use adamant::core::tensors::Tensor;
    ///
    /// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    /// let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();
    /// let c = a.mul(&b).unwrap();
    ///
    /// assert_eq!(*c.get(&[0]).unwrap(), 4.0);
    /// assert_eq!(*c.get(&[1]).unwrap(), 10.0);
    /// assert_eq!(*c.get(&[2]).unwrap(), 18.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a `TensorError::ShapeError` if the shapes are not compatible.
    pub fn mul(&self, other: &Tensor<T>) -> TensorResult<Tensor<T>> {
        self.element_wise_op(other, |a, b| a.clone() * b.clone())
    }
}

impl<T: Clone + Default + Div<Output = T>> Tensor<T> {
    /// Divides this tensor by another (element-wise)
    ///
    /// # Examples
    ///
    /// ```
    /// use adamant::core::tensors::Tensor;
    ///
    /// let a = Tensor::from_vec(vec![8.0, 6.0, 4.0], &[3]).unwrap();
    /// let b = Tensor::from_vec(vec![2.0, 3.0, 2.0], &[3]).unwrap();
    /// let c = a.div(&b).unwrap();
    ///
    /// assert_eq!(*c.get(&[0]).unwrap(), 4.0);
    /// assert_eq!(*c.get(&[1]).unwrap(), 2.0);
    /// assert_eq!(*c.get(&[2]).unwrap(), 2.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a `TensorError::ShapeError` if the shapes are not compatible.
    pub fn div(&self, other: &Tensor<T>) -> TensorResult<Tensor<T>> {
        self.element_wise_op(other, |a, b| a.clone() / b.clone())
    }
}

impl<T: Clone + Default> Tensor<T> {
    /// Generic element-wise operation with another tensor
    ///
    /// This is a helper method used by add, sub, mul, div, etc.
    fn element_wise_op<F>(&self, other: &Tensor<T>, op: F) -> TensorResult<Tensor<T>>
    where
        F: Fn(&T, &T) -> T
    {
        // Check if shapes are compatible for broadcasting
        if !self.shape.is_broadcast_compatible_with(&other.shape) {
            return Err(TensorError::ShapeError(format!(
                "Cannot perform element-wise operation with incompatible shapes {:?} and {:?}",
                self.dims(), other.dims()
            )));
        }

        // For now, we'll implement the simple case where shapes are identical
        // Broadcasting will be added in a future version
        if self.dims() != other.dims() {
            return Err(TensorError::OperationError(
                "Broadcasting not yet implemented".to_string()
            ));
        }

        // Create result tensor with the same shape
        let mut result = Tensor::new(self.dims());

        // Perform element-wise operation
        for i in 0..self.size() {
            let a = &self.storage.data()[i];
            let b = &other.storage.data()[i];

            if let Some(result_data) = result.data_mut() {
                result_data[i] = op(a, b);
            } else {
                return Err(TensorError::OperationError(
                    "Cannot modify result tensor".to_string()
                ));
            }
        }

        Ok(result)
    }

    /// Apply a function to each element of the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use adamant::core::tensors::Tensor;
    ///
    /// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    /// let b = a.map(|x| x * 2.0);
    ///
    /// assert_eq!(*b.get(&[0]).unwrap(), 2.0);
    /// assert_eq!(*b.get(&[1]).unwrap(), 4.0);
    /// assert_eq!(*b.get(&[2]).unwrap(), 6.0);
    /// ```
    pub fn map<F, U>(&self, f: F) -> Tensor<U>
    where
        F: Fn(&T) -> U,
        U: Clone + Default,
    {
        let mut new_data = Vec::with_capacity(self.size());

        for i in 0..self.size() {
            new_data.push(f(&self.storage.data()[i]));
        }

        Tensor::from_vec_with_layout(new_data, self.dims(), self.layout())
            .expect("Mapping should preserve tensor size")
    }

    /// Create a deep copy of the tensor
    pub fn clone_deep(&self) -> Self {
        Tensor {
            shape: self.shape.clone(),
            storage: TensorStorage::new(self.storage.data().to_vec()),
        }
    }
}

// Matrix multiplication (for rank-2 tensors)
impl<T: Clone + Default + Add<Output = T> + Mul<Output = T> + From<u8>> Tensor<T> {
    /// Performs matrix multiplication
    ///
    /// # Examples
    ///
    /// ```
    /// use adamant::core::tensors::Tensor;
    ///
    /// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    /// let c = a.matmul(&b).unwrap();
    ///
    /// assert_eq!(*c.get(&[0, 0]).unwrap(), 19.0);  // 1*5 + 2*7
    /// assert_eq!(*c.get(&[0, 1]).unwrap(), 22.0);  // 1*6 + 2*8
    /// assert_eq!(*c.get(&[1, 0]).unwrap(), 43.0);  // 3*5 + 4*7
    /// assert_eq!(*c.get(&[1, 1]).unwrap(), 50.0);  // 3*6 + 4*8
    /// ```
    ///
    /// # Errors
    ///
    /// Returns errors if:
    /// - Either tensor is not of rank 2 (not a matrix)
    /// - The inner dimensions don't match (first tensor columns != second tensor rows)
    pub fn matmul(&self, other: &Tensor<T>) -> TensorResult<Tensor<T>> {
        // Check if both tensors are matrices (rank 2)
        if self.rank() != 2 || other.rank() != 2 {
            return Err(TensorError::ShapeError(format!(
                "Matrix multiplication requires both tensors to be matrices, got ranks {} and {}",
                self.rank(), other.rank()
            )));
        }

        // Check if inner dimensions match
        let self_dims = self.dims();
        let other_dims = other.dims();

        if self_dims[1] != other_dims[0] {
            return Err(TensorError::ShapeError(format!(
                "Inner dimensions for matrix multiplication must match, got {}×{} and {}×{}",
                self_dims[0], self_dims[1], other_dims[0], other_dims[1]
            )));
        }

        // Result shape: [self_rows, other_cols]
        let result_shape = vec![self_dims[0], other_dims[1]];
        let mut result = Tensor::new(&result_shape);

        let zero = T::from(0);

        // Perform matrix multiplication
        for i in 0..self_dims[0] {
            for j in 0..other_dims[1] {
                let mut sum = zero.clone();

                for k in 0..self_dims[1] {
                    let a = self.get(&[i, k])?;
                    let b = other.get(&[k, j])?;
                    sum = sum + a.clone() * b.clone();
                }

                result.set(&[i, j], sum)?;
            }
        }

        Ok(result)
    }
}

// Basic display implementation
impl<T: fmt::Debug> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor {:?} ({:?}) [", self.shape.dims(), self.shape.layout())?;

        // For simplicity, just print the first few elements
        const MAX_DISPLAY: usize = 10;
        let data = self.storage.data();
        let display_count = std::cmp::min(data.len(), MAX_DISPLAY);

        for i in 0..display_count {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:?}", data[i])?;
        }

        if data.len() > MAX_DISPLAY {
            write!(f, ", ... ({} more elements)", data.len() - MAX_DISPLAY)?;
        }

        write!(f, "]")
    }
}

// Display implementation for TensorView
impl<'a, T: fmt::Debug> fmt::Debug for TensorView<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TensorView {:?} [", self.shape.dims())?;

        // Limit the number of elements to display
        const MAX_DISPLAY: usize = 8;

        // Get the first few elements
        let mut count = 0;
        let mut indices = vec![0; self.rank()];

        while count < MAX_DISPLAY && count < self.size() {
            if count > 0 {
                write!(f, ", ")?;
            }

            match self.get(&indices) {
                Ok(val) => write!(f, "{:?}", val)?,
                Err(_) => write!(f, "<?>")?
            }

            count += 1;

            // Increment indices
            if count < self.size() {
                let mut dim = self.rank() - 1;
                loop {
                    indices[dim] += 1;
                    if indices[dim] < self.dims()[dim] {
                        break;
                    }
                    indices[dim] = 0;
                    if dim == 0 {
                        break;
                    }
                    dim -= 1;
                }
            }
        }

        if self.size() > MAX_DISPLAY {
            write!(f, ", ... ({} more elements)", self.size() - MAX_DISPLAY)?;
        }

        write!(f, "]")
    }
}

// ------
// Tests
// ------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::<f32>::new(&[2, 3]);
        assert_eq!(tensor.dims(), &[2, 3]);
        assert_eq!(tensor.size(), 6);
        assert_eq!(tensor.rank(), 2);
    }

    #[test]
    fn test_tensor_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, &[2, 3]).unwrap();
        assert_eq!(tensor.dims(), &[2, 3]);
        assert_eq!(tensor.size(), 6);
    }

    #[test]
    fn test_tensor_reshape() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, &[2, 3]).unwrap();
        let reshaped = tensor.reshape(&[3, 2]).unwrap();
        assert_eq!(reshaped.dims(), &[3, 2]);
        assert_eq!(reshaped.size(), 6);
    }

    #[test]
    fn test_tensor_get() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, &[2, 3]).unwrap();

        assert_eq!(*tensor.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*tensor.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(*tensor.get(&[0, 2]).unwrap(), 3.0);
        assert_eq!(*tensor.get(&[1, 0]).unwrap(), 4.0);
        assert_eq!(*tensor.get(&[1, 1]).unwrap(), 5.0);
        assert_eq!(*tensor.get(&[1, 2]).unwrap(), 6.0);
    }

    #[test]
    fn test_tensor_set() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut tensor = Tensor::from_vec(data, &[2, 3]).unwrap();

        tensor.set(&[0, 0], 10.0).unwrap();
        tensor.set(&[1, 2], 20.0).unwrap();

        assert_eq!(*tensor.get(&[0, 0]).unwrap(), 10.0);
        assert_eq!(*tensor.get(&[1, 2]).unwrap(), 20.0);
    }

    #[test]
    fn test_index_errors() {
        let tensor = Tensor::<f32>::new(&[2, 3]);

        assert!(tensor.get(&[0, 0, 0]).is_err());  // Too many indices
        assert!(tensor.get(&[0]).is_err());        // Too few indices
        assert!(tensor.get(&[2, 0]).is_err());     // First index out of bounds
        assert!(tensor.get(&[0, 3]).is_err());     // Second index out of bounds
    }

    #[test]
    fn test_row_major_layout() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec_with_layout(data, &[2, 3], MemoryLayout::RowMajor).unwrap();

        assert_eq!(*tensor.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*tensor.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(*tensor.get(&[0, 2]).unwrap(), 3.0);
        assert_eq!(*tensor.get(&[1, 0]).unwrap(), 4.0);
        assert_eq!(*tensor.get(&[1, 1]).unwrap(), 5.0);
        assert_eq!(*tensor.get(&[1, 2]).unwrap(), 6.0);

        // Verify strides are correct for row-major
        assert_eq!(tensor.shape().strides(), &[3, 1]);
    }

    #[test]
    fn test_column_major_layout() {
        let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let tensor = Tensor::from_vec_with_layout(data, &[2, 3], MemoryLayout::ColumnMajor).unwrap();

        assert_eq!(*tensor.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*tensor.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(*tensor.get(&[0, 2]).unwrap(), 3.0);
        assert_eq!(*tensor.get(&[1, 0]).unwrap(), 4.0);
        assert_eq!(*tensor.get(&[1, 1]).unwrap(), 5.0);
        assert_eq!(*tensor.get(&[1, 2]).unwrap(), 6.0);

        // Verify strides are correct for column-major
        assert_eq!(tensor.shape().strides(), &[1, 2]);
    }

    #[test]
    fn test_layout_conversion() {
        // Create a row-major tensor
        let row_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let row_tensor = Tensor::from_vec_with_layout(row_data, &[2, 3], MemoryLayout::RowMajor).unwrap();

        // Convert to column-major
        let col_tensor = row_tensor.to_layout(MemoryLayout::ColumnMajor);

        // Verify layout changed
        assert_eq!(col_tensor.layout(), MemoryLayout::ColumnMajor);

        // Verify data access is consistent
        assert_eq!(*col_tensor.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*col_tensor.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(*col_tensor.get(&[0, 2]).unwrap(), 3.0);
        assert_eq!(*col_tensor.get(&[1, 0]).unwrap(), 4.0);
        assert_eq!(*col_tensor.get(&[1, 1]).unwrap(), 5.0);
        assert_eq!(*col_tensor.get(&[1, 2]).unwrap(), 6.0);

        // The raw data should now be in column-major order
        let expected_col_data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        assert_eq!(col_tensor.data(), &expected_col_data);
    }

    #[test]
    fn test_tensor_view() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, &[2, 3]).unwrap();

        let view = tensor.view();

        assert_eq!(view.dims(), tensor.dims());
        assert_eq!(*view.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*view.get(&[1, 2]).unwrap(), 6.0);
    }

    #[test]
    fn test_tensor_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, &[2, 3]).unwrap();

        // Get the first row
        let slice = tensor.slice(&[0, 0], &[1, 3]).unwrap();

        assert_eq!(slice.dims(), &[1, 3]);
        assert_eq!(*slice.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*slice.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(*slice.get(&[0, 2]).unwrap(), 3.0);
    }

    #[test]
    fn test_tensor_add() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let c = a.add(&b).unwrap();

        assert_eq!(*c.get(&[0, 0]).unwrap(), 6.0);
        assert_eq!(*c.get(&[0, 1]).unwrap(), 8.0);
        assert_eq!(*c.get(&[1, 0]).unwrap(), 10.0);
        assert_eq!(*c.get(&[1, 1]).unwrap(), 12.0);
    }

    #[test]
    fn test_tensor_mul() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let c = a.mul(&b).unwrap();

        assert_eq!(*c.get(&[0, 0]).unwrap(), 5.0);
        assert_eq!(*c.get(&[0, 1]).unwrap(), 12.0);
        assert_eq!(*c.get(&[1, 0]).unwrap(), 21.0);
        assert_eq!(*c.get(&[1, 1]).unwrap(), 32.0);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let c = a.matmul(&b).unwrap();

        assert_eq!(*c.get(&[0, 0]).unwrap(), 19.0);  // 1*5 + 2*7
        assert_eq!(*c.get(&[0, 1]).unwrap(), 22.0);  // 1*6 + 2*8
        assert_eq!(*c.get(&[1, 0]).unwrap(), 43.0);  // 3*5 + 4*7
        assert_eq!(*c.get(&[1, 1]).unwrap(), 50.0);  // 3*6 + 4*8
    }

    #[test]
    fn test_tensor_map() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let b = a.map(|&x| x * 2.0);

        assert_eq!(*b.get(&[0, 0]).unwrap(), 2.0);
        assert_eq!(*b.get(&[0, 1]).unwrap(), 4.0);
        assert_eq!(*b.get(&[1, 0]).unwrap(), 6.0);
        assert_eq!(*b.get(&[1, 1]).unwrap(), 8.0);
    }

    #[test]
    fn test_filled_with() {
        let tensor = Tensor::filled_with(&[2, 3], 7.0);

        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(*tensor.get(&[i, j]).unwrap(), 7.0);
            }
        }
    }

    #[test]
    fn test_shared_storage() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let shared = a.to_shared();

        // Should have same values
        assert_eq!(*shared.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*shared.get(&[1, 1]).unwrap(), 4.0);

        // But modification should not be possible
        let mut shared_clone = shared.clone();
        assert!(shared_clone.set(&[0, 0], 10.0).is_err());
    }

    #[test]
    fn test_to_owned() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let shared = a.to_shared();
        let owned = shared.to_owned();

        // Should have same values
        assert_eq!(*owned.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*owned.get(&[1, 1]).unwrap(), 4.0);

        // But modification should now be possible
        let mut owned_mut = owned.clone();
        assert!(owned_mut.set(&[0, 0], 10.0).is_ok());
        assert_eq!(*owned_mut.get(&[0, 0]).unwrap(), 10.0);
    }

    #[test]
    fn test_broadcast_compatibility() {
        let shape1 = TensorShape::new(&[2, 3]);
        let shape2 = TensorShape::new(&[3]);
        let shape3 = TensorShape::new(&[4, 3]);

        assert!(shape1.is_broadcast_compatible_with(&shape2));
        assert!(!shape1.is_broadcast_compatible_with(&shape3));
    }
}