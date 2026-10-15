// SPDX-License-Identifier: AGPL-3.0-only

//! Long-copy verify rows borrowed from idle slots (`speculative::long_copy`).
//!
//! The DFlash verify pools hold γ = `num_intermediates` rows per MTP slot,
//! slot-major in one region per SSM layer: KDA fold records at
//! `kda_records(slot) + t × row`, conv snapshots at `conv_intermediate(slot,
//! t)` = `(slot × γ + t) × conv`. The rows are per-verify scratch: a verify
//! writes them, its commit (`commit_kda_records`) reads them, and nothing in
//! them survives the step. Only a lone owner verifies a long copy, so rather
//! than sizing every slot for `long + 1` rows (+0.3 GB a rank at 15 drafts,
//! +0.9 GB at 31), a verify of more than γ rows runs on the owner's block and
//! the blocks of the slots after it, up to (not including) the MTP dummy: the
//! slot-major addresses `slot × γ + t` continue into them, so the verify, the
//! records commit and the finish-leaf save all address the window unchanged.
//! The pools stay at the shipped size.
//!
//! Borrowing never depends on timing:
//! - The head lends rows only while the owner is the model's only sequence:
//!   every other claimable slot is on the free list (`sole_claimed`). The
//!   scheduler asks before it proposes and again before it verifies
//!   (`Model::lone_verify_draft_capacity`), so a request admitted since the
//!   proposal cuts the verify back to γ rows; `prepare_verify_rows` refuses
//!   a long verify the window does not hold.
//! - The worker rank keeps a finished sequence's slot claimed until the head
//!   reuses it, so it checks only the window bound and follows the head.
//! - The commit of borrowed rows runs on the secondary stream. The next slot
//!   claim on either rank orders its `reset_slot` (which zeroes the slot's
//!   conv snapshots) after it (`fence_lent_rows`).
//!
//! Only the KDA records rollback (`--ssm-rollback-mode records`, GLM) borrows.
//! Under the H snapshot rollback a slot holds γ−1 H rows against γ conv rows,
//! and that rollback reads them by slot (`async_chkpt`), so a window of `n`
//! slots would hold n(γ−1) H rows, not nγ − 1. Those modes keep γ rows.

use anyhow::{Result, ensure};
use atlas_core::config::LayerType;
use std::sync::atomic::Ordering;

use super::ssm_pool::SsmStatePool;
use super::types::TransformerModel;
use crate::layer::SsmLayerState;
use crate::traits::{Model, SequenceState};

impl SsmStatePool {
    /// Rows a verify by the sequence in `slot` can reach: its own block and
    /// every later MTP slot's under KDA records, else its own γ.
    pub(crate) fn window_rows(&self, slot: usize) -> usize {
        if self.kda_record_pools.is_empty() || slot >= self.mtp_slots {
            return self.num_intermediates;
        }
        (self.mtp_slots - slot) * self.num_intermediates
    }

    /// `slot` is claimed and every other claimable slot is free.
    pub(crate) fn sole_claimed(&self, slot: usize) -> bool {
        let free = self.free_slots.lock();
        free.len() + 1 == self.max_slots && !free.contains(&slot)
    }

    /// Rows a verify by the sequence in `slot` may hold now: the window
    /// while it is the only sequence, else its own γ.
    pub(crate) fn lone_verify_rows(&self, slot: usize) -> usize {
        if self.sole_claimed(slot) {
            self.window_rows(slot)
        } else {
            self.num_intermediates
        }
    }
}

impl TransformerModel {
    /// Point `seq`'s conv snapshot rows at a `k`-row verify: its own γ rows,
    /// or for a long verify the borrowed window's first `k` (the records
    /// pointer already starts the window). Called on both ranks before the
    /// verify; refuses a long verify the rows do not hold.
    pub(super) fn prepare_verify_rows(&self, seq: &mut SequenceState, k: usize) -> Result<()> {
        let pool = &self.ssm_pool;
        let gamma = pool.num_intermediates;
        if pool.kda_record_pools.is_empty() {
            return Ok(());
        }
        if k > gamma {
            let slot = seq.slot_idx;
            ensure!(
                k <= pool.window_rows(slot),
                "long verify of {k} rows: slot {slot} reaches {} (γ {gamma})",
                pool.window_rows(slot)
            );
            let head = self.comm.as_ref().is_none_or(|c| c.rank() == 0);
            ensure!(
                !head || pool.sole_claimed(slot),
                "long verify of {k} rows: slot {slot} is not the only sequence, \
                 so the rows past its own {gamma} are not free"
            );
            pool.rows_lent.store(true, Ordering::Release);
        }
        self.set_conv_rows(seq, k.max(gamma))
    }

    /// Back to the slot's own γ conv rows once a long verify committed.
    pub(super) fn release_verify_rows(&self, seq: &mut SequenceState) -> Result<()> {
        if self.ssm_pool.kda_record_pools.is_empty() {
            return Ok(());
        }
        self.set_conv_rows(seq, self.ssm_pool.num_intermediates)
    }

    fn set_conv_rows(&self, seq: &mut SequenceState, rows: usize) -> Result<()> {
        let slot = seq.slot_idx;
        let mut ssm_layer_idx = 0usize;
        for (i, state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            let ssm = state
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;
            let inters = &mut ssm.conv_state_intermediates;
            if !inters.is_empty() && inters.len() != rows {
                *inters = (0..rows)
                    .map(|t| self.ssm_pool.conv_intermediate(ssm_layer_idx, slot, t))
                    .collect();
            }
            ssm_layer_idx += 1;
        }
        Ok(())
    }

    /// Before a slot claim: order it after the commit of borrowed rows (on
    /// the secondary stream), which the claimed slot's `reset_slot` would
    /// otherwise race. GPU-side; a no-op unless rows were lent since.
    pub(super) fn fence_lent_rows(&self) -> Result<()> {
        if self.ssm_pool.rows_lent.swap(false, Ordering::AcqRel) {
            self.sync_secondary()?;
        }
        Ok(())
    }

    /// Drafts a lone verify by the sequence in `slot` may hold now
    /// ([`lone_capacity`]), within the DFlash hidden-save front.
    pub(super) fn lone_verify_draft_capacity_dispatch(&self, slot: usize) -> usize {
        let front = if self.dflash_hidden_save.is_some() {
            self.dflash_front_rows
        } else {
            usize::MAX
        };
        lone_capacity(
            crate::speculative::long_copy::drafts(),
            self.ssm_pool.verify_draft_capacity(slot),
            self.ssm_pool.num_intermediates,
            self.ssm_pool.lone_verify_rows(slot).min(front),
        )
    }
}

/// Drafts a lone verify may hold: its slot's own `own`, or with long copies
/// on (`long` > 0) a window of `rows` verify rows past the slot's γ.
fn lone_capacity(long: usize, own: usize, gamma: usize, rows: usize) -> usize {
    if long == 0 || rows <= gamma {
        own
    } else {
        rows - 1
    }
}

#[cfg(test)]
#[path = "ssm_borrow_tests.rs"]
mod tests;
