//! Storage backends for tensor data.
//!
//! This module provides different storage options for tensor data,
//! including owned and shared storage to enable efficient views and operations.

use std::sync::Arc;

/// Represents a storage backend for tensor data
///
/// This enum provides different storage options for tensor data,
/// including owned and shared storage to enable efficient views and operations.
#[derive(Debug, Clone)]
pub enum TensorStorage<T> {
    /// Owned storage, where the tensor owns the data
    Owned(Vec<T>),
    /// Shared storage, where the data is shared between multiple tensors
    Shared(Arc<Vec<T>>),
    // Future: adding more storage types (e.g., mapped memory, GPU storage)
}

impl<T> TensorStorage<T> {
    /// Creates a new owned storage with the given data
    pub fn new(data: Vec<T>) -> Self {
        TensorStorage::Owned(data)
    }

    /// Creates a new shared storage with the given data
    pub fn shared(data: Vec<T>) -> Self {
        TensorStorage::Shared(Arc::new(data))
    }

    /// Gets a reference to the underlying data
    #[inline]
    pub fn data(&self) -> &[T] {
        match self {
            TensorStorage::Owned(vec) => vec,
            TensorStorage::Shared(arc) => arc,
        }
    }

    /// Gets a mutable reference to the underlying data
    ///
    /// Returns `None` if the storage is shared, as shared data cannot be modified.
    #[inline]
    pub fn data_mut(&mut self) -> Option<&mut [T]> {
        match self {
            TensorStorage::Owned(vec) => Some(vec),
            TensorStorage::Shared(_) => None,  // Can't mutate shared data
        }
    }

    /// Returns true if the storage is shared
    #[inline]
    pub fn is_shared(&self) -> bool {
        matches!(self, TensorStorage::Shared(_))
    }

    /// Returns the length of the stored data
    #[inline]
    pub fn len(&self) -> usize {
        self.data().len()
    }

    /// Returns true if the storage is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data().is_empty()
    }
}

impl<T: Clone> TensorStorage<T> {
    /// Makes this storage owned (cloning if necessary)
    ///
    /// If the storage is already owned, returns self.
    /// If the storage is shared, creates a new owned storage with cloned data.
    pub fn into_owned(self) -> Self {
        match self {
            TensorStorage::Owned(_) => self,
            TensorStorage::Shared(arc) => TensorStorage::Owned(arc.to_vec()),
        }
    }
}
