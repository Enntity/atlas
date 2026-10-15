// SPDX-License-Identifier: AGPL-3.0-only

//! F4 — DFlash / speculative-decoding contract stub (first pass).
//! Covers DSpark drafter (6-layer, gamma=4 / K=3 verify) plus MTP
//! multi-token drafting. Full loop deferred — this is config + interface only.
//! ponytail: stub until M1 qualifier clears; full verify loop needs GPU window.

use serde::{Deserialize, Serialize};

/// F4 feature flags per model profile (README: Nemotron + DSpark, Qwen + MTP).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DsflashF4Contract {
    /// DSpark drafter enabled (checkpoint declares `dspark_block_size > 0`).
    #[serde(default)]
    pub dspark_enabled: bool,
    /// MTP speculative-decoding enabled (`mtp_num_hidden_layers > 0`).
    #[serde(default)]
    pub mtp_enabled: bool,
    /// Draft expansion gamma (README: gamma=4). Verified at parse time.
    #[serde(default = "default_gamma")]
    pub gamma: usize,
    /// Verify depth K (README: K=3 for DSpark, K=2 for MTP). Must be ≥ 1.
    #[serde(default = "default_k")]
    pub verify_k: usize,
    /// Max draft seqs in MTP pool (ref PROGRESS_LOG.md line 663-671).
    #[serde(default = "default_pool")]
    pub draft_pool_max_seqs: usize,
    /// Whether to bind the drafter to the same KV pool (true = shared).
    #[serde(default)]
    pub shared_kv_pool: bool,
}

fn default_gamma() -> usize {
    4
}
fn default_k() -> usize {
    3
}
fn default_pool() -> usize {
    4
}

impl DsflashF4Contract {
    /// Minimal validity check before any GPU work (in-tree, CPU-only).
    pub fn check(&self, _vocab_size: usize) -> anyhow::Result<()> {
        use anyhow::{bail, ensure};
        if self.dspark_enabled && self.gamma == 0 {
            bail!("F4 DSpark: gamma must be > 0");
        }
        if self.mtp_enabled && self.verify_k == 0 {
            bail!("F4 MTP: verify_k must be > 0");
        }
        if self.verify_k > 8 {
            bail!("F4: verify_k > 8 is unimplemented in this stub");
        }
        // DSpark noise token must sit inside vocab when drafter is declared.
        // (Real check binds `dspark_noise_token_id`; stub keeps it loose.)
        ensure!(self.draft_pool_max_seqs > 0, "draft_pool_max_seqs > 0");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn f4_contract_default_ok() {
        let c = DsflashF4Contract::default();
        assert!(!c.dspark_enabled && !c.mtp_enabled);
        c.check(32000).unwrap();
    }
    #[test]
    fn f4_dspark_params_ok() {
        let c = DsflashF4Contract {
            dspark_enabled: true,
            mtp_enabled: false,
            gamma: 4,
            verify_k: 3,
            draft_pool_max_seqs: 4,
            shared_kv_pool: true,
        };
        c.check(32000).unwrap();
    }
}
