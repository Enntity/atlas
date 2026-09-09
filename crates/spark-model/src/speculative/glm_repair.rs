// SPDX-License-Identifier: AGPL-3.0-only

//! Explicit GLM-only capability. No new wire command: both ranks bind these
//! borrowed inputs to their own live request before the existing E1 executes.

use crate::{layer::ForwardContext, speculative::ProposerState};
use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

#[derive(Clone, Copy, Debug)]
pub struct RepairSpan {
    pub ptr: DevicePtr,
    pub bytes: usize,
}

pub struct RepairInput<'a> {
    pub token: u32,
    pub tokens: &'a [u32],
    pub prompt_len: usize,
    pub position: usize,
    pub drafts: usize,
    pub generation: u64,
    pub capture_generation: u64,
    pub captured_rows: usize,
    pub context_tokens: usize,
    pub capture: RepairSpan,
    pub normalized: RepairSpan,
    pub bonus: RepairSpan,
    pub hidden_row: usize,
}

pub trait GlmPairRepair: Send + Sync {
    fn paired_handoff(&self) -> Option<&dyn GlmPairedHandoff> {
        None
    }
    /// No mutation, allocation or GPU operation. Head invokes before E1.
    fn validate_prepare(
        &self,
        input: &RepairInput<'_>,
        state: &dyn ProposerState,
        ctx: &ForwardContext,
    ) -> Result<()>;
    /// Revalidates, writes KV, then publishes the next proposal descriptor.
    /// A failed writer leaves an explicitly failed, non-resumable phase.
    fn prepare(
        &self,
        input: &RepairInput<'_>,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()>;
}

/// Optional GLM-only request-owned path; legacy and other proposers lack it.
pub trait GlmPairedHandoff: Send + Sync {
    /// Read-only candidate identity, not a reservation or published lease.
    fn validate_allocation(&self, gpu: &dyn spark_runtime::gpu::GpuBackend) -> Result<usize>;
    fn validate_verify(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        tokens: &[u32],
        state: &dyn ProposerState,
        ctx: &ForwardContext,
    ) -> Result<()>;
    fn validate_propose(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        seed: u32,
        state: &dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(u64, u64)>;
    /// Latch a selected ownership/issued-command failure before any producer exists.
    fn fail_session(&self, gpu: &dyn spark_runtime::gpu::GpuBackend) -> Result<()>;
    fn commit_target(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        committed: usize,
        width: usize,
        completed: bool,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
    ) -> Result<()>;
    fn begin_verify(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        tokens: &[u32],
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
    ) -> Result<()>;
    fn publish_verify(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        tokens: &[u32],
        predictions: &[u32],
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
    ) -> Result<()>;
    fn record_verify(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        base: usize,
        tokens: &[u32],
        accepted: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
    ) -> Result<()>;
    fn validate_cold(
        &self,
        state: &dyn ProposerState,
        slot: usize,
        prompt: usize,
        ctx: &ForwardContext,
    ) -> Result<()>;
    fn retire(
        &self,
        state: &mut dyn ProposerState,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
    ) -> Result<Option<usize>>;
    fn close(&self, gpu: &dyn spark_runtime::gpu::GpuBackend, secondary_stream: u64) -> Result<()>;
    fn propose_owned(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        token: u32,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Vec<u32>>;
    fn begin_decode(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        token: u32,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
    ) -> Result<()>;
    fn publish_decode(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        token: u32,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()>;
    fn quarantine(
        &self,
        state: &mut dyn ProposerState,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
    ) -> Result<()>;
    fn prime(
        &self,
        input: &crate::model::GlmPairedInput<'_>,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()>;
}
