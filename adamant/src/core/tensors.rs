//! Tensor implementation for the Adamant library.
//!
//! This module provides the core Tensor data structure, which represents
//! an n-dimensional array with various numeric operations.

use std::fmt;
use std::ops::{Index, IndexMut};
use std::sync::Arc;

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
    /// Other errors
    Other(String),
}

/// Result type for tensor operations
pub type TensorResult<T> = Result<T, TensorError>;

/// Represents the shape and strides of a tensor
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    /// The dimensions of the tensor (e.g., [2, 3, 4] for a 2×3×4 tensor)
    dims: Vec<usize>,
    /// The strides for each dimension (used for indexing)
    strides: Vec<usize>,
}

impl TensorShape {
    /// Create a new TensorShape with the given dimensions
    pub fn new(dims: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&dims);
        TensorShape { dims, strides }
    }

    /// Compute strides for the given dimensions (row-major order)
    fn compute_strides(dims: &[usize]) -> Vec<usize> {
        // Start with stride 1 for the last dimension
        let mut strides = vec![1];

        // Compute remaining strides right-to-left
        for i in (0..dims.len() - 1).rev() {
            let stride = strides[0] * dims[i + 1];
            strides.insert(0, stride);
        }

        strides
    }

    /// Get the number of elements in the tensor
    pub fn size(&self) -> usize {
        self.dims.iter().product()
    }

    /// Get the rank (number of dimensions) of the tensor
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Get the dimensions of the tensor
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Get the strides of the tensor
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
}

/// Represents a storage backend for tensor data
#[derive(Debug, Clone)]
pub enum TensorStorage<T> {
    /// Owned storage, where the tensor owns the data
    Owned(Vec<T>),
    /// Shared storage, where the data is shared between multiple tensors
    Shared(Arc<Vec<T>>),
    // Future: add more storage types (e.g., mapped memory, GPU storage)
}

impl<T: Clone> TensorStorage<T> {
    /// Create a new owned storage with the given data
    pub fn new(data: Vec<T>) -> Self {
        TensorStorage::Owned(data)
    }

    /// Create a new shared storage with the given data
    pub fn shared(data: Vec<T>) -> Self {
        TensorStorage::Shared(Arc::new(data))
    }

    /// Get a reference to the underlying data
    pub fn data(&self) -> &[T] {
        match self {
            TensorStorage::Owned(vec) => vec,
            TensorStorage::Shared(arc) => arc,
        }
    }

    /// Get a mutable reference to the underlying data
    pub fn data_mut(&mut self) -> Option<&mut [T]> {
        match self {
            TensorStorage::Owned(vec) => Some(vec),
            TensorStorage::Shared(_) => None,  // Can't mutate shared data
        }
    }

    /// Make this storage owned (cloning if necessary)
    pub fn into_owned(self) -> Self {
        match self {
            TensorStorage::Owned(_) => self,
            TensorStorage::Shared(arc) => TensorStorage::Owned(arc.to_vec()),
        }
    }
}

/// A multidimensional array that supports various numeric operations
#[derive(Clone)]
pub struct Tensor<T> {
    /// The shape of the tensor
    shape: TensorShape,
    /// The actual data storage
    storage: TensorStorage<T>,
}

impl<T: Clone + Default> Tensor<T> {
    /// Create a new tensor with the given shape, filled with the default value
    pub fn new(dims: &[usize]) -> Self {
        let shape = TensorShape::new(dims.to_vec());
        let size = shape.size();
        let data = vec![T::default(); size];

        Tensor {
            shape,
            storage: TensorStorage::new(data),
        }
    }

    /// Create a new tensor from existing data and shape
    pub fn from_vec(data: Vec<T>, dims: &[usize]) -> TensorResult<Self> {
        let shape = TensorShape::new(dims.to_vec());
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

    /// Get the shape of the tensor
    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    /// Get the dimensions of the tensor
    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Get the number of elements in the tensor
    pub fn size(&self) -> usize {
        self.shape.size()
    }

    /// Get the rank (number of dimensions) of the tensor
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Get a reference to the underlying data
    pub fn data(&self) -> &[T] {
        self.storage.data()
    }

    /// Get a mutable reference to the underlying data (if possible)
    pub fn data_mut(&mut self) -> Option<&mut [T]> {
        self.storage.data_mut()
    }

    /// Reshape the tensor to a new shape with the same total size
    pub fn reshape(&self, new_dims: &[usize]) -> TensorResult<Self> {
        let new_shape = TensorShape::new(new_dims.to_vec());

        if new_shape.size() != self.shape.size() {
            return Err(TensorError::ShapeError(format!(
                "Cannot reshape tensor of size {} to new shape with size {}",
                self.shape.size(), new_shape.size()
            )));
        }

        // For now, we clone the data to ensure correct layout
        // In the future, we could optimize this to avoid the clone in some cases
        Ok(Tensor {
            shape: new_shape,
            storage: self.storage.clone(),
        })
    }

    /// Get an element by multi-dimensional index
    pub fn get(&self, indices: &[usize]) -> TensorResult<&T> {
        let flat_idx = self.get_flat_index(indices)?;
        Ok(&self.storage.data()[flat_idx])
    }

    /// Set an element by multi-dimensional index
    pub fn set(&mut self, indices: &[usize], value: T) -> TensorResult<()> {
        let flat_idx = self.get_flat_index(indices)?;

        match self.storage.data_mut() {
            Some(data) => {
                data[flat_idx] = value;
                Ok(())
            },
            None => Err(TensorError::OperationError(
                "Cannot modify shared tensor data".to_string()
            )),
        }
    }

    /// Convert multi-dimensional indices to a flat index
    fn get_flat_index(&self, indices: &[usize]) -> TensorResult<usize> {
        if indices.len() != self.shape.rank() {
            return Err(TensorError::IndexError(format!(
                "Expected {} indices, got {}",
                self.shape.rank(), indices.len()
            )));
        }

        // Check if any index is out of bounds
        for (i, (&idx, &dim)) in indices.iter().zip(self.shape.dims().iter()).enumerate() {
            if idx >= dim {
                return Err(TensorError::IndexError(format!(
                    "Index {} is out of bounds for dimension {} with size {}",
                    idx, i, dim
                )));
            }
        }

        // Calculate the flat index using strides
        let flat_idx = indices.iter()
            .zip(self.shape.strides().iter())
            .map(|(&idx, &stride)| idx * stride)
            .sum();

        Ok(flat_idx)
    }
}

// Basic display implementation
impl<T: fmt::Debug> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor {:?} [", self.shape.dims())?;

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

// More implementations can be added later (operations, indexing, etc.)

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
}