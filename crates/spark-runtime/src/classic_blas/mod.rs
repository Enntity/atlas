// SPDX-License-Identifier: AGPL-3.0-only
//! Backend-owned classic BLAS resources. Every stream has an independent workspace.
use crate::gpu::{Bf16Gemm, Bf16GemmNumerics};
use anyhow::{Result, ensure};
use parking_lot::Mutex;
use std::{collections::HashMap, sync::Arc};
#[cfg(feature = "cuda")]
pub(crate) mod ffi;
#[cfg(any(feature = "cuda", test))]
mod provider;
#[cfg(any(feature = "cuda", test))]
use provider::validate_provider_pair;
pub(crate) trait ClassicBlasIo: Send + Sync {
    fn check_context(&self, context: u64) -> Result<()>;
    fn create(&self) -> Result<u64>;
    fn alloc(&self, bytes: usize) -> Result<u64>;
    fn configure(
        &self,
        handle: u64,
        stream: u64,
        workspace: u64,
        numerics: Bf16GemmNumerics,
    ) -> Result<()>;
    fn launch(&self, handle: u64, request: Bf16Gemm) -> Result<()>;
    fn synchronize(&self, context: u64, stream: u64) -> Result<()>;
    fn cleanup(
        &self,
        context: u64,
        stream: u64,
        handle: u64,
        workspace: u64,
        pending: bool,
    ) -> Result<()>;
}
struct OwnedClassic<I: ClassicBlasIo> {
    io: Arc<I>,
    context: u64,
    stream: u64,
    handle: u64,
    workspace: u64,
    pending: bool,
    configured: Option<Bf16GemmNumerics>,
    workspace_bytes: usize,
}
impl<I: ClassicBlasIo> Drop for OwnedClassic<I> {
    fn drop(&mut self) {
        if let Err(error) = self.io.cleanup(
            self.context,
            self.stream,
            self.handle,
            self.workspace,
            self.pending,
        ) {
            tracing::error!(%error, "classic BF16 resource cleanup failed; retaining library for outstanding work");
            // A failed synchronization must not unload code or reclaim live workspace.
            std::mem::forget(self.io.clone());
        }
    }
}
pub(crate) struct ClassicBf16Pool<I: ClassicBlasIo> {
    context: u64,
    io: Arc<I>,
    streams: Mutex<HashMap<u64, OwnedClassic<I>>>,
}
impl<I: ClassicBlasIo> ClassicBf16Pool<I> {
    pub(crate) fn new(context: u64, io: Arc<I>) -> Self {
        Self {
            context,
            io,
            streams: Mutex::new(HashMap::new()),
        }
    }
    pub(crate) fn quiesce(&self) -> Result<()> {
        let mut streams = self.streams.lock();
        for owned in streams.values_mut() {
            if owned.pending {
                self.io.synchronize(self.context, owned.stream)?;
                owned.pending = false;
            }
        }
        Ok(())
    }
    pub(crate) fn launch(&self, request: Bf16Gemm) -> Result<()> {
        request.validate()?;
        let (compute, algorithm, math, bytes) = request
            .numerics
            .profile()
            .ok_or_else(|| anyhow::anyhow!("classic pool requires explicit classic profile"))?;
        ensure!(bytes > 0, "classic profile requires owned workspace");
        self.io.check_context(self.context)?;
        let mut streams = self.streams.lock();
        if let std::collections::hash_map::Entry::Vacant(entry) = streams.entry(request.stream) {
            let mut owned = OwnedClassic {
                io: self.io.clone(),
                context: self.context,
                stream: request.stream,
                handle: self.io.create()?,
                workspace: 0,
                pending: false,
                configured: None,
                workspace_bytes: bytes,
            };
            owned.workspace = self.io.alloc(bytes)?;
            self.io.configure(
                owned.handle,
                owned.stream,
                owned.workspace,
                request.numerics,
            )?;
            owned.configured = Some(request.numerics);
            tracing::info!(
                context = self.context,
                stream = owned.stream,
                compute,
                algorithm,
                math_mode = math,
                workspace_bytes = bytes,
                "native shared classic BF16 stream configured"
            );
            entry.insert(owned);
        }
        let owned = streams
            .get_mut(&request.stream)
            .expect("stream resource inserted");
        ensure!(
            owned.workspace_bytes == bytes,
            "classic stream workspace profile differs"
        );
        if owned.configured != Some(request.numerics) {
            // Configuration may partly succeed. Force a retry after any failure,
            // including a request for the previously configured profile.
            owned.configured = None;
            self.io.configure(
                owned.handle,
                owned.stream,
                owned.workspace,
                request.numerics,
            )?;
            owned.configured = Some(request.numerics);
        }
        // A failed library call may still enqueue work; cleanup always waits afterwards.
        owned.pending = true;
        self.io.launch(owned.handle, request)
    }
}
#[cfg(test)]
mod tests;
