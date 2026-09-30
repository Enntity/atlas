// SPDX-License-Identifier: AGPL-3.0-only

//! KV admission for a prefill chunk: every rank reserves the chunk's blocks
//! and all ranks agree on the outcome BEFORE its forward pass issues a
//! collective.
//!
//! The head cannot decide alone: each rank evicts from its own prefix cache
//! (rank-local, and it diverges — see the F83 note in `prefix_lookup`), so
//! free-plus-evictable capacity differs per rank. Each rank therefore tries
//! its own reservation and one min-vote decides for all. On refusal every
//! rank undoes its reservation and returns [`KvAdmissionRefused`] from the
//! same point, having mutated no sequence or GPU state for the chunk, so the
//! head can free a victim and re-send the chunk while the worker simply
//! waits for its next command (`spark-server` `prefill_preempt`).

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::prefix_cache::PrefixCache;

use super::block_mgmt::ensure_blocks_through_prefill;
use super::types::TransformerModel;
use crate::traits::SequenceState;

#[cfg(test)]
#[path = "kv_admission_tests.rs"]
mod tests;

/// Every rank refused a prefill chunk's KV reservation before its forward
/// pass and rolled it back; `by_peer` when this rank alone could have
/// admitted it.
#[derive(Debug)]
pub struct KvAdmissionRefused {
    pub by_peer: bool,
}

impl std::fmt::Display for KvAdmissionRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let by = if self.by_peer {
            "a peer rank"
        } else {
            "this rank"
        };
        write!(
            f,
            "KV cache exhausted: prefill chunk refused by {by} before its forward pass"
        )
    }
}

impl std::error::Error for KvAdmissionRefused {}

/// Whether `e` is an agreed, rolled-back refusal (safe to retry the chunk).
pub fn is_kv_admission_refused(e: &anyhow::Error) -> bool {
    e.downcast_ref::<KvAdmissionRefused>().is_some()
}

/// EP worker: an agreed refusal is not fatal. The head re-sends the chunk
/// after preempting, or releases the slot (`0xFFFFFFF1`); keep serving.
pub(super) fn worker_step_outcome(step: Result<bool>) -> Result<bool> {
    match step {
        Err(e) if is_kv_admission_refused(&e) => {
            tracing::warn!("EP worker: {e:#}; waiting for the head's retry or release");
            Ok(true)
        }
        step => step,
    }
}

/// EP worker: the head sends a chunk from its own recorded progress (a retry
/// resumes past a completed tail-split half), so this rank must stand there
/// too; anything else would desync the chunk's collectives.
pub(super) fn check_worker_chunk_start(seq: &SequenceState, chunk_start: usize) -> Result<()> {
    anyhow::ensure!(
        seq.seq_len == chunk_start,
        "EP worker: prefill chunk starts at {chunk_start} but slot {} is at {} \
         (head/worker prefill progress diverged)",
        seq.slot_idx,
        seq.seq_len
    );
    Ok(())
}

/// Reserve `seq`'s blocks through `abs_block_idx`; `vote` turns this rank's
/// success into the all-rank verdict. Transactional: unless every rank
/// admits, this call's blocks (and HSS disk ids) are released again.
#[allow(clippy::too_many_arguments)]
pub(crate) fn admit(
    seq: &mut SequenceState,
    abs_block_idx: usize,
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn PrefixCache,
    gpu: &dyn GpuBackend,
    stream: u64,
    kv_poison: bool,
    vote: impl FnOnce(bool) -> Result<bool>,
) -> Result<()> {
    let (blocks, disk_ids) = (seq.block_table.len(), seq.disk_block_ids.len());
    let local = ensure_blocks_through_prefill(
        seq,
        abs_block_idx,
        kv_cache,
        prefix_cache,
        gpu,
        stream,
        kv_poison,
    );
    if vote(local.is_ok())? && local.is_ok() {
        return Ok(());
    }
    kv_cache.free_blocks(&seq.block_table.split_off(blocks));
    let ids = seq
        .disk_block_ids
        .split_off(disk_ids.min(seq.disk_block_ids.len()));
    if !ids.is_empty() {
        let _ = spark_storage::with_local(|hss| {
            for &id in &ids {
                hss.dec_disk_ref(id);
            }
            Ok(())
        });
    }
    match local {
        Err(e) if !format!("{e:#}").contains("KV cache exhausted") => Err(e),
        local => Err(KvAdmissionRefused {
            by_peer: local.is_ok(),
        }
        .into()),
    }
}

impl TransformerModel {
    /// [`admit`] the blocks for positions `..end_pos` of this chunk. A
    /// multi-rank world always votes, so every rank issues the same
    /// collectives whether or not it had to allocate.
    pub(in crate::model) fn reserve_prefill_blocks(
        &self,
        seq: &mut SequenceState,
        end_pos: usize,
        kv_cache: &mut PagedKvCache,
        stream: u64,
    ) -> Result<()> {
        let last_block = (end_pos - 1) / kv_cache.block_size();
        admit(
            seq,
            last_block,
            kv_cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
            stream,
            self.levers.kv_poison,
            |ok| {
                if self.multi_rank_protocol_active() {
                    Ok(self.ep_min_u32(ok as u32)? == 1)
                } else {
                    Ok(ok)
                }
            },
        )
    }
}
