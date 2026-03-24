//! Tensor types for MNN inference.
//!
//! This module provides safe wrappers around MNN tensor operations,
//! including creating, reading, and writing tensor data.

use crate::backend::{DataType, BackendType};
use crate::config::DataFormat;
use crate::error::{MnnError, MnnResult};
use mnn_rs_sys::MNNTensor;
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

    /// Create a tensor wrapper around an existing MNN tensor pointer (public version).
    ///
    /// # Safety
    /// The pointer must be valid and remain valid for the lifetime of this tensor.
    pub unsafe fn from_ptr(ptr: *mut MNNTensor, name: Option<String>) -> Self {
        Self { inner: ptr, name }
    }

    /// Get the mutable raw pointer to the underlying MNN tensor.
    pub fn inner_mut(&mut self) -> *mut MNNTensor {
        self.inner
    }

    /// Get the raw pointer to the underlying MNN tensor.
    pub fn as_ptr(&self) -> *const MNNTensor {
        self.inner
    }

    /// Get the shape of the tensor.
    pub fn shape(&self) -> Vec<i32> {
        unsafe {
            let dim_count = mnn_rs_sys::mnn_tensor_get_dimensions(self.inner);
            if dim_count <= 0 {
                return Vec::new();
            }

            let mut shape = Vec::with_capacity(dim_count as usize);
            for i in 0..dim_count {
                let dim = mnn_rs_sys::mnn_tensor_get_dim(self.inner, i);
                shape.push(dim);
            }
            shape
        }
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        unsafe { mnn_rs_sys::mnn_tensor_get_dimensions(self.inner) as usize }
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
        unsafe {
            let type_code = mnn_rs_sys::mnn_tensor_get_type_code(self.inner);
            DataType::from_type_code(type_code)
        }
    }

    /// Get the data format of the tensor.
    pub fn format(&self) -> DataFormat {
        unsafe {
            let dim_type = mnn_rs_sys::mnn_tensor_get_dimension_type(self.inner);
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
        unsafe { mnn_rs_sys::mnn_tensor_get_element_count(self.inner) }
    }

    /// Get the size of the tensor data in bytes.
    pub fn byte_size(&self) -> usize {
        unsafe { mnn_rs_sys::mnn_tensor_get_size(self.inner) as usize }
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

        let host_data = unsafe { mnn_rs_sys::mnn_tensor_get_host_data(self.inner) };
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

        let host_data = unsafe { mnn_rs_sys::mnn_tensor_get_host_data(self.inner) };
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
        let ptr = unsafe { mnn_rs_sys::mnn_tensor_get_host_data(self.inner) };

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
        let ptr = unsafe { mnn_rs_sys::mnn_tensor_get_host_data(self.inner) };

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

    // ========================================================================
    // GPU Memory Operations
    // ========================================================================

    /// Copy data from a host tensor to this tensor (potentially on device).
    ///
    /// # Arguments
    /// * `host_tensor` - The host tensor to copy from
    ///
    /// # Returns
    /// Ok(()) on success, or an error on failure.
    pub fn copy_from_host(&mut self, host_tensor: &Tensor) -> MnnResult<()> {
        let result = unsafe {
            mnn_rs_sys::mnn_tensor_copy_from_host(self.inner, host_tensor.inner)
        };

        if result != mnn_rs_sys::MNN_ERROR_NONE as i32 {
            return Err(MnnError::internal("Failed to copy from host tensor"));
        }

        Ok(())
    }

    /// Copy data from this tensor (potentially on device) to a host tensor.
    ///
    /// # Arguments
    /// * `host_tensor` - The host tensor to copy to
    ///
    /// # Returns
    /// Ok(()) on success, or an error on failure.
    pub fn copy_to_host(&self, host_tensor: &mut Tensor) -> MnnResult<()> {
        let result = unsafe {
            mnn_rs_sys::mnn_tensor_copy_to_host(host_tensor.inner, self.inner)
        };

        if result != mnn_rs_sys::MNN_ERROR_NONE as i32 {
            return Err(MnnError::internal("Failed to copy to host tensor"));
        }

        Ok(())
    }

    /// Create a device tensor with the given shape and format.
    ///
    /// # Arguments
    /// * `shape` - The tensor shape
    /// * `format` - The data format (NHWC, NCHW, etc.)
    /// * `dtype` - The data type
    ///
    /// # Returns
    /// A new device tensor on success, or an error on failure.
    pub fn create_device(
        shape: &[i32],
        format: DataFormat,
        dtype: DataType,
    ) -> MnnResult<Tensor> {
        if shape.is_empty() {
            return Err(MnnError::internal("Shape cannot be empty"));
        }

        let type_code = dtype.to_type_code();
        let format_code = format.to_mnn();

        let inner = unsafe {
            mnn_rs_sys::mnn_tensor_create_device(
                shape.as_ptr(),
                shape.len() as i32,
                type_code,
                format_code,
            )
        };

        if inner.is_null() {
            return Err(MnnError::internal("Failed to create device tensor"));
        }

        Ok(unsafe { Tensor::from_ptr(inner, None) })
    }

    /// Clone this tensor.
    ///
    /// # Arguments
    /// * `deep_copy` - If true, copy data; if false, only copy metadata
    ///
    /// # Returns
    /// A cloned tensor on success, or an error on failure.
    pub fn clone(&self, deep_copy: bool) -> MnnResult<Tensor> {
        let inner = unsafe {
            mnn_rs_sys::mnn_tensor_clone(self.inner, if deep_copy { 1 } else { 0 })
        };

        if inner.is_null() {
            return Err(MnnError::internal("Failed to clone tensor"));
        }

        Ok(unsafe { Tensor::from_ptr(inner, None) })
    }

    /// Get the device ID for this tensor (for GPU tensors).
    ///
    /// # Returns
    /// The device ID, or 0 if not a GPU tensor or unknown.
    pub fn device_id(&self) -> u64 {
        unsafe { mnn_rs_sys::mnn_tensor_device_id(self.inner) }
    }

    /// Get the backend type for this tensor.
    ///
    /// # Returns
    /// The backend type (CPU, CUDA, OpenCL, etc.).
    pub fn backend(&self) -> BackendType {
        let backend_code = unsafe { mnn_rs_sys::mnn_tensor_get_backend(self.inner) };
        BackendType::from_mnn_type(backend_code)
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
            let dim_count = mnn_rs_sys::mnn_tensor_get_dimensions(self.inner);
            if dim_count <= 0 {
                return Vec::new();
            }

            let mut shape = Vec::with_capacity(dim_count as usize);
            for i in 0..dim_count {
                let dim = mnn_rs_sys::mnn_tensor_get_dim(self.inner, i);
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

impl TensorInfo {
    /// Get the data type.
    pub fn dtype(&self) -> DataType {
        self.dtype
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