//! Utility functions and helpers for MNN.
//!
//! This module provides various utility functions for working with MNN.

use crate::config::DataFormat;
use crate::error::{MnnError, MnnResult};

/// Calculate the size of a tensor given its shape and element size.
///
/// # Arguments
/// * `shape` - The tensor shape
/// * `element_size` - Size of each element in bytes
///
/// # Returns
/// Total size in bytes.
#[inline]
pub fn calculate_tensor_size(shape: &[i32], element_size: usize) -> usize {
    shape.iter().map(|&d| d as usize).product::<usize>() * element_size
}

/// Calculate the total number of elements in a tensor.
///
/// # Arguments
/// * `shape` - The tensor shape
///
/// # Returns
/// Total number of elements.
#[inline]
pub fn calculate_element_count(shape: &[i32]) -> i32 {
    shape.iter().product()
}

/// Convert between NHWC and NCHW formats.
///
/// # Arguments
/// * `data` - The input data
/// * `shape` - The tensor shape
/// * `from` - Source format
/// * `to` - Target format
///
/// # Returns
/// The converted data.
pub fn convert_format<T: Copy>(
    data: &[T],
    shape: &[i32],
    from: DataFormat,
    to: DataFormat,
) -> MnnResult<Vec<T>> {
    if from == to {
        return Ok(data.to_vec());
    }

    let n = shape[0] as usize;

    match (from, to) {
        (DataFormat::Nhwc, DataFormat::Nchw) => {
            // NHWC -> NCHW
            if shape.len() != 4 {
                return Err(MnnError::invalid_input(
                    "Format conversion requires 4D tensor",
                ));
            }
            let h = shape[1] as usize;
            let w = shape[2] as usize;
            let c = shape[3] as usize;

            let mut result = Vec::with_capacity(data.len());

            for n_idx in 0..n {
                for c_idx in 0..c {
                    for h_idx in 0..h {
                        for w_idx in 0..w {
                            let src_idx = n_idx * h * w * c + h_idx * w * c + w_idx * c + c_idx;
                            result.push(data[src_idx]);
                        }
                    }
                }
            }

            Ok(result)
        }
        (DataFormat::Nchw, DataFormat::Nhwc) => {
            // NCHW -> NHWC
            if shape.len() != 4 {
                return Err(MnnError::invalid_input(
                    "Format conversion requires 4D tensor",
                ));
            }
            let c = shape[1] as usize;
            let h = shape[2] as usize;
            let w = shape[3] as usize;

            let mut result = Vec::with_capacity(data.len());

            for n_idx in 0..n {
                for h_idx in 0..h {
                    for w_idx in 0..w {
                        for c_idx in 0..c {
                            let src_idx = n_idx * c * h * w + c_idx * h * w + h_idx * w + w_idx;
                            result.push(data[src_idx]);
                        }
                    }
                }
            }

            Ok(result)
        }
        _ => Err(MnnError::unsupported(format!(
            "Cannot convert from {:?} to {:?}",
            from, to
        ))),
    }
}

/// Image preprocessing utilities.
pub mod image {
    use crate::error::{MnnError, MnnResult};

    /// Normalize an image with mean and standard deviation.
    ///
    /// # Arguments
    /// * `data` - The image data (HWC format)
    /// * `mean` - Mean values for each channel
    /// * `std` - Standard deviation for each channel
    ///
    /// # Returns
    /// Normalized data.
    pub fn normalize(data: &mut [f32], mean: &[f32], std: &[f32]) -> MnnResult<()> {
        if mean.len() != std.len() {
            return Err(MnnError::invalid_input(
                "Mean and std must have the same length",
            ));
        }

        let channels = mean.len();
        if data.len() % channels != 0 {
            return Err(MnnError::invalid_input(
                "Data length must be divisible by number of channels",
            ));
        }

        for chunk in data.chunks_mut(channels) {
            for (i, val) in chunk.iter_mut().enumerate() {
                *val = (*val - mean[i]) / std[i];
            }
        }

        Ok(())
    }

    /// Resize an image to a target size using bilinear interpolation.
    ///
    /// This is a simple implementation. For production use, consider
    /// using the `image` crate.
    ///
    /// # Arguments
    /// * `data` - The image data (HWC format)
    /// * `src_height` - Source height
    /// * `src_width` - Source width
    /// * `dst_height` - Target height
    /// * `dst_width` - Target width
    /// * `channels` - Number of channels
    ///
    /// # Returns
    /// Resized image data.
    pub fn resize_bilinear(
        data: &[f32],
        src_height: usize,
        src_width: usize,
        dst_height: usize,
        dst_width: usize,
        channels: usize,
    ) -> Vec<f32> {
        let src_row_stride = src_width * channels;
        let dst_row_stride = dst_width * channels;

        let mut result = vec![0.0f32; dst_height * dst_width * channels];

        let y_ratio = src_height as f32 / dst_height as f32;
        let x_ratio = src_width as f32 / dst_width as f32;

        for dst_y in 0..dst_height {
            let src_y_f = dst_y as f32 * y_ratio;
            let src_y0 = src_y_f.floor() as usize;
            let src_y1 = (src_y0 + 1).min(src_height - 1);
            let y_frac = src_y_f - src_y0 as f32;

            for dst_x in 0..dst_width {
                let src_x_f = dst_x as f32 * x_ratio;
                let src_x0 = src_x_f.floor() as usize;
                let src_x1 = (src_x0 + 1).min(src_width - 1);
                let x_frac = src_x_f - src_x0 as f32;

                let dst_idx = dst_y * dst_row_stride + dst_x * channels;

                let src_idx00 = src_y0 * src_row_stride + src_x0 * channels;
                let src_idx01 = src_y0 * src_row_stride + src_x1 * channels;
                let src_idx10 = src_y1 * src_row_stride + src_x0 * channels;
                let src_idx11 = src_y1 * src_row_stride + src_x1 * channels;

                for c in 0..channels {
                    let v00 = data[src_idx00 + c];
                    let v01 = data[src_idx01 + c];
                    let v10 = data[src_idx10 + c];
                    let v11 = data[src_idx11 + c];

                    let v0 = v00 * (1.0 - x_frac) + v01 * x_frac;
                    let v1 = v10 * (1.0 - x_frac) + v11 * x_frac;
                    let v = v0 * (1.0 - y_frac) + v1 * y_frac;

                    result[dst_idx + c] = v;
                }
            }
        }

        result
    }

    /// Convert HWC format to CHW format.
    pub fn hwc_to_chw(data: &[f32], height: usize, width: usize, channels: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; data.len()];

        for h in 0..height {
            for w in 0..width {
                for c in 0..channels {
                    let src_idx = (h * width + w) * channels + c;
                    let dst_idx = c * height * width + h * width + w;
                    result[dst_idx] = data[src_idx];
                }
            }
        }

        result
    }

    /// Convert CHW format to HWC format.
    pub fn chw_to_hwc(data: &[f32], height: usize, width: usize, channels: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; data.len()];

        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    let src_idx = c * height * width + h * width + w;
                    let dst_idx = (h * width + w) * channels + c;
                    result[dst_idx] = data[src_idx];
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_tensor_size() {
        let shape = [1, 3, 224, 224];
        let size = calculate_tensor_size(&shape, 4);
        assert_eq!(size, 1 * 3 * 224 * 224 * 4);
    }

    #[test]
    fn test_calculate_element_count() {
        let shape = [1, 3, 224, 224];
        let count = calculate_element_count(&shape);
        assert_eq!(count, 1 * 3 * 224 * 224);
    }
}