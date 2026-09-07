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
