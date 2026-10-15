// SPDX-License-Identifier: AGPL-3.0-only

//! F4 — DFlash speculative-decoding server stub (first pass, deferred).
//! Defines the draft-verify interface without a production loop.
//! ponytail: interface only; full loop needs GPU window and M1 qualifier.

/// Draft-verify loop interface for DSpark (gamma=4, K=3) + MTP (K=2).
/// Real implementation binds draft KV pool, runs verify against main model,
/// and accepts/rejects tokens. This stub holds the contract.
pub struct DsflashStub {
    pub gamma: usize,
    pub verify_k: usize,
    pub shared_kv: bool,
}

impl DsflashStub {
    pub fn new(gamma: usize, verify_k: usize, shared_kv: bool) -> Self {
        Self {
            gamma,
            verify_k,
            shared_kv,
        }
    }
    /// Placeholder verify step — returns accepted count (stub = 0 until loop built).
    pub fn verify_step(&self, _main_hidden: &[f32], _draft_hidden: &[f32]) -> usize {
        // ponytail: stub — full layered verify (o_proj → reduced attn → FFN → L3
        // routing → MoE → logits) deferred until M1 clears.
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stub_constructs() {
        let s = DsflashStub::new(4, 3, false);
        assert_eq!(s.gamma, 4);
    }
}
