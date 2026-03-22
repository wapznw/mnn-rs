//! Integration tests for MNN.
//!
//! These tests require a valid MNN model file to run.
//! Tests are skipped if no model is available.

use mnn::prelude::*;

/// Check if a test model is available
fn test_model_available() -> Option<&'static str> {
    // Check for test model in common locations
    if std::path::Path::new("tests/models/test.mnn").exists() {
        Some("tests/models/test.mnn")
    } else if std::path::Path::new("test.mnn").exists() {
        Some("test.mnn")
    } else {
        None
    }
}

#[test]
fn test_version() {
    let v = mnn::version();
    // Version should be a string (may be "unknown" if MNN not linked)
    assert!(!v.is_empty());
}

#[test]
fn test_available_backends() {
    let backends = mnn::available_backends();
    // CPU should always be available (even if not linked properly)
    // At minimum, this tests that the function doesn't crash
    println!("Available backends: {:?}", backends);
}

#[test]
fn test_schedule_config_default() {
    let config = ScheduleConfig::default();
    assert_eq!(config.num_threads, 4);
}

#[test]
fn test_schedule_config_builder() {
    let config = ScheduleConfigBuilder::new()
        .backend(BackendType::CPU)
        .num_threads(8)
        .memory_mode(MemoryMode::High)
        .precision_mode(PrecisionMode::Low)
        .build();

    assert_eq!(config.num_threads, 8);
    assert_eq!(config.backend_config.memory_mode, MemoryMode::High);
    assert_eq!(config.backend_config.precision_mode, PrecisionMode::Low);
}

#[test]
fn test_data_type_size() {
    assert_eq!(DataType::Float32.size(), 4);
    assert_eq!(DataType::Int32.size(), 4);
    assert_eq!(DataType::Float64.size(), 8);
    assert_eq!(DataType::UInt8.size(), 1);
    assert_eq!(DataType::Int16.size(), 2);
}

#[test]
fn test_data_type_properties() {
    assert!(DataType::Float32.is_float());
    assert!(!DataType::Int32.is_float());
    assert!(DataType::Int32.is_integer());
    assert!(DataType::Float32.is_signed());
    assert!(!DataType::UInt8.is_signed());
}

#[test]
fn test_data_format() {
    assert_eq!(DataFormat::Nhwc.name(), "NHWC");
    assert_eq!(DataFormat::Nchw.name(), "NCHW");
    assert_eq!(DataFormat::Nc4hw4.name(), "NC4HW4");
}

#[test]
fn test_memory_mode() {
    let normal = MemoryMode::Normal;
    let low = MemoryMode::Low;
    let high = MemoryMode::High;

    // Ensure they're distinct
    assert_ne!(normal, low);
    assert_ne!(normal, high);
    assert_ne!(low, high);
}

#[test]
fn test_precision_mode() {
    let normal = PrecisionMode::Normal;
    let low = PrecisionMode::Low;
    let high = PrecisionMode::High;
    let low_bf16 = PrecisionMode::LowBf16;

    // Ensure they're distinct
    assert_ne!(normal, low);
    assert_ne!(normal, high);
    assert_ne!(normal, low_bf16);
}

#[test]
fn test_tensor_data_trait() {
    // Test that TensorData trait is implemented for expected types
    assert_eq!(f32::dtype(), DataType::Float32);
    assert_eq!(f64::dtype(), DataType::Float64);
    assert_eq!(i32::dtype(), DataType::Int32);
    assert_eq!(i16::dtype(), DataType::Int16);
    assert_eq!(u8::dtype(), DataType::UInt8);
}

#[test]
fn test_error_types() {
    let err = MnnError::invalid_model("test error");
    assert!(err.to_string().contains("test error"));

    let err = MnnError::shape_mismatch(&[1, 2, 3], &[1, 2, 4]);
    assert!(err.to_string().contains("expected"));
    assert!(err.to_string().contains("actual"));
}

#[test]
fn test_utility_functions() {
    let shape = [1, 3, 224, 224];

    let size = mnn::calculate_tensor_size(&shape, 4);
    assert_eq!(size, 1 * 3 * 224 * 224 * 4);

    let count = mnn::calculate_element_count(&shape);
    assert_eq!(count, 1 * 3 * 224 * 224);
}

// Model-dependent tests (require a test model)

#[test]
#[ignore = "Requires test model file"]
fn test_interpreter_from_file() {
    if let Some(model_path) = test_model_available() {
        let result = Interpreter::from_file(model_path);
        assert!(result.is_ok());
    }
}

#[test]
#[ignore = "Requires test model file"]
fn test_session_creation() {
    if let Some(model_path) = test_model_available() {
        let interpreter = Interpreter::from_file(model_path).unwrap();
        let config = ScheduleConfig::default();
        let result = interpreter.create_session(config);
        assert!(result.is_ok());
    }
}

#[test]
#[ignore = "Requires test model file"]
fn test_session_run() {
    if let Some(model_path) = test_model_available() {
        let interpreter = Interpreter::from_file(model_path).unwrap();
        let config = ScheduleConfig::default();
        let mut session = interpreter.create_session(config).unwrap();

        let result = session.run();
        assert!(result.is_ok());
        assert!(session.has_run());
    }
}

// Async tests (require async feature)

#[cfg(feature = "async")]
mod async_tests {
    use super::*;

    #[tokio::test]
    async fn test_async_interpreter_creation() {
        if let Some(model_path) = test_model_available() {
            let result = mnn::AsyncInterpreter::from_file(model_path).await;
            assert!(result.is_ok());
        }
    }
}