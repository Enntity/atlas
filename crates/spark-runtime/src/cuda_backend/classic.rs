// SPDX-License-Identifier: AGPL-3.0-only
use super::AtlasCudaBackend;
use crate::gpu::Bf16GemmNumerics;
use anyhow::{ensure, Result};
impl AtlasCudaBackend {
    pub(super) fn dispatch_bf16(&self, request: crate::gpu::Bf16Gemm) -> Result<()> {
        request.validate()?;
        if request.numerics.profile().is_some() {
            self.require_bf16_numerics(request.numerics)?;
            return self
                .classic_blas
                .get()
                .expect("classic capability initialized")
                .launch(request);
        }
        crate::cublaslt::bf16_gemm_act_weight_t(
            request.input.0,
            request.weight.0,
            request.output.0,
            request.rows,
            request.columns,
            request.inner,
            request.stream,
        )
    }

    pub(super) fn quiesce_classic_blas(&self) -> Result<()> {
        if let Some(pool) = self.classic_blas.get() {
            pool.quiesce()?;
        }
        Ok(())
    }

    pub(super) fn require_bf16_numerics(&self, numerics: Bf16GemmNumerics) -> Result<()> {
        if numerics.profile().is_some() && self.classic_blas.get().is_none() {
            let mut current = 0;
            let status = unsafe { super::cuCtxGetCurrent(&mut current) };
            ensure!(
                status == 0 && current == self.cuda_ctx,
                "classic capability CUDA context differs from backend owner"
            );
            let io = std::sync::Arc::new(crate::classic_blas::ffi::CublasIo::load(numerics)?);
            let pool = crate::classic_blas::ClassicBf16Pool::new(self.cuda_ctx, io);
            let _ = self.classic_blas.set(pool);
        }
        Ok(())
    }
}
