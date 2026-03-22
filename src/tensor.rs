//! Tensor types for MNN inference.
//!
//! This module provides safe wrappers around MNN tensor operations,
//! including creating, reading, and writing tensor data.

use crate::backend::DataType;
use crate::config::DataFormat;
use crate::error::{MnnError, MnnResult};
use mnn_sys::MNNTensor;
use std::ffi::c_void;
use std::marker::PhantomData;

/// Information about a tensor's shape and type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Name of the tensor (may be empty)
    pub name: String,

    /// Shape of the tensor (dimensions)
    pub shape: Vec<i32>,

    /// Data type of tensor elements
    pub dtype: DataType,

    /// Data format (layout)
    pub format: DataFormat,
}

impl TensorInfo {
    /// Get the total number of elements in the tensor.
    pub fn element_count(&self) -> i32 {
        self.shape.iter().product()
    }

    /// Get the size in bytes of the tensor data.
    pub fn byte_size(&self) -> usize {
        self.element_count() as usize * self.dtype.size()
    }
}

/// A multi-dimensional array for neural network operations.
///
/// Tensors are the primary data structure for MNN inference,
/// holding input and output data for models.
pub struct Tensor {
    inner: *mut MNNTensor,
    /// Name of the tensor (if any)
    name: Option<String>,
}

// Safety: Tensor operations are thread-safe through MNN's internal synchronization
unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl Tensor {
    /// Create a tensor wrapper around an existing MNN tensor pointer.
    ///
    /// # Safety
    /// The pointer must be valid and remain valid for the lifetime of this tensor.
    pub(crate) unsafe fn from_ptr_with_name(ptr: *mut MNNTensor, name: Option<String>) -> Self {
        Self { inner: ptr, name }
    }

    /// Get the shape of the tensor.
    pub fn shape(&self) -> Vec<i32> {
        unsafe {
            let dim_count = mnn_sys::mnn_tensor_get_dimensions(self.inner);
            if dim_count <= 0 {
                return Vec::new();
            }

            let mut shape = Vec::with_capacity(dim_count as usize);
            for i in 0..dim_count {
                let dim = mnn_sys::mnn_tensor_get_dim(self.inner, i);
                shape.push(dim);
            }
            shape
        }
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        unsafe { mnn_sys::mnn_tensor_get_dimensions(self.inner) as usize }
    }

    /// Get the size of a specific dimension.
    ///
    /// # Arguments
    /// * `axis` - The dimension index (0-based)
    ///
    /// # Returns
    /// The size of the dimension, or an error if the axis is out of bounds.
    pub fn dim(&self, axis: usize) -> MnnResult<i32> {
        let shape = self.shape();
        if axis >= shape.len() {
            return Err(MnnError::index_out_of_bounds(axis, 0, shape.len() as i32));
        }
        Ok(shape[axis])
    }

    /// Get the data type of the tensor.
    pub fn dtype(&self) -> DataType {
        DataType::Float32 // Default for now
    }

    /// Get the data format of the tensor.
    pub fn format(&self) -> DataFormat {
        unsafe {
            let dim_type = mnn_sys::mnn_tensor_get_dimension_type(self.inner);
            match dim_type {
                0 => DataFormat::Nhwc,
                1 => DataFormat::Nc4hw4,
                2 => DataFormat::Nchw,
                _ => DataFormat::Nhwc,
            }
        }
    }

    /// Get the total number of elements in the tensor.
    pub fn element_count(&self) -> i32 {
        unsafe { mnn_sys::mnn_tensor_get_element_count(self.inner) }
    }

    /// Get the size of the tensor data in bytes.
    pub fn byte_size(&self) -> usize {
        unsafe { mnn_sys::mnn_tensor_get_size(self.inner) as usize }
    }

    /// Get the name of the tensor.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Write data to the tensor.
    ///
    /// # Arguments
    /// * `data` - The data to write
    ///
    /// # Errors
    /// Returns an error if the data size doesn't match.
    pub fn write<T: TensorData>(&self, data: &[T]) -> MnnResult<()> {
        if data.is_empty() {
            return Err(MnnError::EmptyData);
        }

        let expected_count = self.element_count() as usize;
        if data.len() != expected_count {
            return Err(MnnError::shape_mismatch(
                &[expected_count as i32],
                &[data.len() as i32],
            ));
        }

        let host_data = unsafe { mnn_sys::mnn_tensor_get_host_data(self.inner) };
        if host_data.is_null() {
            return Err(MnnError::tensor_error("Tensor has no host data"));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const c_void,
                host_data,
                data.len() * std::mem::size_of::<T>(),
            );
        }

        Ok(())
    }

    /// Read data from the tensor.
    ///
    /// # Returns
    /// A vector containing the tensor data.
    pub fn read<T: TensorData>(&self) -> MnnResult<Vec<T>> {
        let count = self.element_count() as usize;
        let mut data = vec![T::default(); count];

        let host_data = unsafe { mnn_sys::mnn_tensor_get_host_data(self.inner) };
        if host_data.is_null() {
            return Err(MnnError::tensor_error("Tensor has no host data"));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                host_data,
                data.as_mut_ptr() as *mut c_void,
                count * std::mem::size_of::<T>(),
            );
        }

        Ok(data)
    }

    /// Get a mutable reference to the tensor's host data.
    ///
    /// # Safety
    /// The returned slice is valid only as long as no other operations
    /// are performed on the tensor.
    pub unsafe fn as_slice_mut<T: TensorData>(&mut self) -> MnnResult<&mut [T]> {
        let count = self.element_count() as usize;
        let ptr = unsafe { mnn_sys::mnn_tensor_get_host_data(self.inner) };

        if ptr.is_null() {
            return Err(MnnError::tensor_error("Tensor has no host data"));
        }

        Ok(unsafe { std::slice::from_raw_parts_mut(ptr as *mut T, count) })
    }

    /// Get a reference to the tensor's host data.
    ///
    /// # Safety
    /// The returned slice is valid only as long as no other operations
    /// are performed on the tensor.
    pub unsafe fn as_slice<T: TensorData>(&self) -> MnnResult<&[T]> {
        let count = self.element_count() as usize;
        let ptr = unsafe { mnn_sys::mnn_tensor_get_host_data(self.inner) };

        if ptr.is_null() {
            return Err(MnnError::tensor_error("Tensor has no host data"));
        }

        Ok(unsafe { std::slice::from_raw_parts(ptr as *const T, count) })
    }

    /// Get the tensor info.
    pub fn info(&self) -> TensorInfo {
        TensorInfo {
            name: self.name.clone().unwrap_or_default(),
            shape: self.shape(),
            dtype: self.dtype(),
            format: self.format(),
        }
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape())
            .field("dtype", &self.dtype())
            .field("format", &self.format())
            .field("name", &self.name)
            .finish()
    }
}

/// Trait for types that can be stored in a tensor.
///
/// This trait is implemented for primitive numeric types that MNN supports.
pub trait TensorData: Default + Clone + Copy + 'static {
    /// Get the MNN data type for this Rust type.
    fn dtype() -> DataType;
}

impl TensorData for f32 {
    fn dtype() -> DataType {
        DataType::Float32
    }
}

impl TensorData for f64 {
    fn dtype() -> DataType {
        DataType::Float64
    }
}

impl TensorData for i32 {
    fn dtype() -> DataType {
        DataType::Int32
    }
}

impl TensorData for i16 {
    fn dtype() -> DataType {
        DataType::Int16
    }
}

impl TensorData for u8 {
    fn dtype() -> DataType {
        DataType::UInt8
    }
}

#[cfg(feature = "fp16")]
impl TensorData for half::f16 {
    fn dtype() -> DataType {
        DataType::Float16
    }
}

#[cfg(feature = "int8")]
impl TensorData for i8 {
    fn dtype() -> DataType {
        DataType::Int8
    }
}

/// A view into a tensor's data without ownership.
///
/// This is useful for zero-copy access to tensor data.
pub struct TensorView<'a> {
    inner: *mut MNNTensor,
    _marker: PhantomData<&'a Tensor>,
}

impl std::fmt::Debug for TensorView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorView").finish_non_exhaustive()
    }
}

impl<'a> TensorView<'a> {
    /// Create a view from a tensor reference.
    pub fn from_tensor(tensor: &'a Tensor) -> Self {
        Self {
            inner: tensor.inner,
            _marker: PhantomData,
        }
    }

    /// Get the shape of the tensor.
    pub fn shape(&self) -> Vec<i32> {
        unsafe {
            let dim_count = mnn_sys::mnn_tensor_get_dimensions(self.inner);
            if dim_count <= 0 {
                return Vec::new();
            }

            let mut shape = Vec::with_capacity(dim_count as usize);
            for i in 0..dim_count {
                let dim = mnn_sys::mnn_tensor_get_dim(self.inner, i);
                shape.push(dim);
            }
            shape
        }
    }

    /// Get the data type.
    pub fn dtype(&self) -> DataType {
        DataType::Float32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_data_types() {
        assert_eq!(f32::dtype(), DataType::Float32);
        assert_eq!(i32::dtype(), DataType::Int32);
        assert_eq!(u8::dtype(), DataType::UInt8);
    }

    #[test]
    fn test_tensor_info() {
        let info = TensorInfo {
            name: "test".to_string(),
            shape: vec![1, 3, 224, 224],
            dtype: DataType::Float32,
            format: DataFormat::Nchw,
        };

        assert_eq!(info.element_count(), 1 * 3 * 224 * 224);
        assert_eq!(info.byte_size(), 1 * 3 * 224 * 224 * 4);
    }
}