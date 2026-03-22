//! Asynchronous API for MNN inference.
//!
//! This module provides async versions of the MNN API using tokio.
//! It's useful for integrating MNN into async applications without
//! blocking the executor.

#[cfg(feature = "async")]
mod inner {
    use crate::config::ScheduleConfig;
    use crate::error::{MnnError, MnnResult};
    use crate::interpreter::AsyncInterpreter;
    use crate::session::Session;
    use crate::tensor::Tensor;
    use std::sync::Arc;

    /// Run inference asynchronously.
    ///
    /// This is a convenience function that runs a session's inference
    /// on a blocking thread pool.
    ///
    /// # Arguments
    /// * `session` - The session to run
    ///
    /// # Returns
    /// Ok(()) on success, or an error.
    pub async fn run_session_async(session: &mut Session) -> MnnResult<()> {
        session.run_async().await
    }

    /// Async context for batch inference.
    ///
    /// This struct manages an interpreter and sessions for batch processing.
    pub struct AsyncBatchInference {
        interpreter: AsyncInterpreter,
        config: ScheduleConfig,
        max_concurrent: usize,
    }

    impl AsyncBatchInference {
        /// Create a new batch inference context.
        ///
        /// # Arguments
        /// * `interpreter` - The async interpreter
        /// * `config` - Session configuration
        /// * `max_concurrent` - Maximum concurrent inference tasks
        pub fn new(
            interpreter: AsyncInterpreter,
            config: ScheduleConfig,
            max_concurrent: usize,
        ) -> Self {
            Self {
                interpreter,
                config,
                max_concurrent,
            }
        }

        /// Run inference on a batch of inputs.
        ///
        /// This method processes inputs concurrently up to `max_concurrent` tasks.
        ///
        /// # Arguments
        /// * `inputs` - Iterator of input data
        /// * `process` - Function to prepare inputs and process outputs
        ///
        /// # Returns
        /// A vector of results, one for each input.
        pub async fn run_batch<F, T, R>(
            &self,
            inputs: impl IntoIterator<Item = T> + Send,
            mut process: F,
        ) -> MnnResult<Vec<MnnResult<R>>>
        where
            F: FnMut(&mut Session, T) -> MnnResult<R> + Send + Clone,
            T: Send + 'static,
            R: Send + 'static,
        {
            use tokio::sync::Semaphore;

            let semaphore = Arc::new(Semaphore::new(self.max_concurrent));
            let mut handles = Vec::new();

            for input in inputs {
                let interpreter = self.interpreter.clone();
                let config = self.config.clone();
                let sem = Arc::clone(&semaphore);
                let proc = process.clone();

                let handle = tokio::spawn(async move {
                    let _permit = sem.acquire().await.map_err(|e| {
                        MnnError::AsyncError(format!("Semaphore error: {}", e))
                    })?;

                    let mut session = interpreter.create_session(config.clone()).await?;
                    proc(&mut session, input)
                });

                handles.push(handle);
            }

            let results: Vec<MnnResult<MnnResult<R>>> = futures::future::join_all(handles)
                .await
                .into_iter()
                .map(|r| r.map_err(|e| MnnError::AsyncError(e.to_string())))
                .collect();

            Ok(results.into_iter().map(|r| r?).collect())
        }
    }

    /// A pool of sessions for concurrent inference.
    pub struct SessionPool {
        interpreter: AsyncInterpreter,
        config: ScheduleConfig,
        pool: Vec<Session>,
    }

    impl SessionPool {
        /// Create a new session pool.
        ///
        /// # Arguments
        /// * `interpreter` - The async interpreter
        /// * `config` - Session configuration
        /// * `pool_size` - Number of sessions in the pool
        pub async fn new(
            interpreter: AsyncInterpreter,
            config: ScheduleConfig,
            pool_size: usize,
        ) -> MnnResult<Self> {
            let mut pool = Vec::with_capacity(pool_size);

            for _ in 0..pool_size {
                let session = interpreter.create_session(config.clone()).await?;
                pool.push(session);
            }

            Ok(Self {
                interpreter,
                config,
                pool,
            })
        }

        /// Acquire a session from the pool.
        ///
        /// Returns None if the pool is empty.
        pub fn acquire(&mut self) -> Option<Session> {
            self.pool.pop()
        }

        /// Return a session to the pool.
        pub fn release(&mut self, session: Session) {
            self.pool.push(session);
        }

        /// Get the number of available sessions.
        pub fn available(&self) -> usize {
            self.pool.len()
        }
    }
}

#[cfg(feature = "async")]
pub use inner::*;