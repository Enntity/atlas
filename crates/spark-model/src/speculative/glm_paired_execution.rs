// SPDX-License-Identifier: AGPL-3.0-only
//! Optional actual GLM model execution; validation is not a reservation.
use crate::traits::SequenceState;
use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

pub(crate) mod sealed {
    pub trait Sealed {}
}

/// Only the actual TransformerModel implements this capability. No factory
/// selects its paired constructor yet; other models return None.
pub trait GlmPairedExecution: sealed::Sealed + Send + Sync {
    /// Local health only; no synchronization, host drain or pair-release authority.
    fn check_communication_health(&self) -> Result<()>;
    /// Strict local joins; caller retains owners and owns the supervised pair release.
    fn quiesce(&self) -> Result<()>;
    fn validate_cold_prefill(&self, seq: &SequenceState, tokens: &[u32]) -> Result<()>;
    fn cold_prefill(&self, seq: &mut SequenceState, tokens: &[u32]) -> Result<DevicePtr>;
    fn validate_bootstrap(&self, seq: &SequenceState, token: u32) -> Result<()>;
    fn bootstrap(&self, seq: &mut SequenceState, token: u32) -> Result<DevicePtr>;
    fn validate_verify(&self, seq: &SequenceState, tokens: &[u32]) -> Result<()>;
    fn validate_propose(
        &self,
        seq: &SequenceState,
        seed: u32,
        position: usize,
        drafts: usize,
        grammar: Option<&[i32]>,
    ) -> Result<()>;
    /// Revalidate before issuing selected F5; caller owns the later verdict.
    fn verify(&self, seq: &mut SequenceState, tokens: &[u32]) -> Result<Vec<u32>>;
    fn propose(
        &self,
        seq: &mut SequenceState,
        seed: u32,
        position: usize,
        drafts: usize,
        grammar: Option<&[i32]>,
    ) -> Result<Vec<u32>>;
}
