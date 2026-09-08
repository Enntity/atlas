// SPDX-License-Identifier: AGPL-3.0-only
//! Optional actual GLM model execution; validation is not a reservation.
use crate::traits::SequenceState;
use anyhow::Result;

pub(crate) mod sealed {
    pub trait Sealed {}
}

/// Only the actual TransformerModel implements this capability. No factory
/// selects its paired constructor yet; other models return None.
pub trait GlmPairedExecution: sealed::Sealed + Send + Sync {
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
