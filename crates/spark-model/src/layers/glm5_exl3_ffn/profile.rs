// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit diagnostic timing; inactive profiling submits no GPU operations.
use super::*;
use std::time::Instant;

#[derive(Clone, Copy)]
pub(super) struct ProfileMetadata {
    pub rows: usize,
    pub layer: Option<usize>,
    pub rank: usize,
    pub world: usize,
    pub routes: usize,
    pub role: &'static str,
    pub precision: Option<[u32; 8]>,
    pub thin_capacity: Option<u32>,
}
pub(super) struct FfnProfile<'a> {
    gpu: &'a dyn GpuBackend,
    stream: u64,
    enabled: bool,
    metadata: ProfileMetadata,
}
impl<'a> FfnProfile<'a> {
    pub(super) fn new(
        gpu: &'a dyn GpuBackend,
        stream: u64,
        enabled: bool,
        metadata: ProfileMetadata,
    ) -> Self {
        Self {
            gpu,
            stream,
            enabled,
            metadata,
        }
    }
    pub(super) fn measure<T>(
        &self,
        stage: &'static str,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        if !self.enabled {
            return operation();
        }
        self.gpu.synchronize(self.stream)?;
        let start = Instant::now();
        let value = operation()?;
        self.gpu.synchronize(self.stream)?;
        let m = self.metadata;
        let router_gemm = (stage == "router_gemm_transport")
            .then_some(spark_runtime::gpu::Bf16GemmNumerics::CublasLt);
        let shared_gemm = if m.role == "shared_reconstructed"
            && matches!(stage, "merged_gate_up_gemm_split" | "down_projection")
        {
            spark_runtime::gpu::Bf16GemmNumerics::Glm53Classic.profile()
        } else {
            None
        };
        tracing::info!(target: "atlas::exl3_ffn_profile", stage, elapsed_us = start.elapsed().as_micros() as u64,
            rows = m.rows, layer = ?m.layer, routes = m.routes, rank = m.rank, world = m.world, stream = self.stream,
            role = m.role, mma_profiles = ?m.precision, thin_capacity = ?m.thin_capacity,
            shared_gemm = ?shared_gemm, router_gemm = ?router_gemm,
            "EXL3 FFN diagnostic stage (synchronized; mixed routing policy, no occupancy download)");
        Ok(value)
    }
}
