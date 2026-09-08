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
    /// Transport checkpoint is deliberately not implemented by validation alone.
    fn verify(&self, _seq: &mut SequenceState, _tokens: &[u32]) -> Result<Vec<u32>> {
        anyhow::bail!("paired command transport is not enabled")
    }
    fn propose(
        &self,
        _seq: &mut SequenceState,
        _seed: u32,
        _position: usize,
        _drafts: usize,
        _grammar: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        anyhow::bail!("paired command transport is not enabled")
    }
}
