// SPDX-License-Identifier: AGPL-3.0-only

//! Serving-path side of the prefix cache's NVMe spill tier
//! (`ATLAS_KV_NVME_DIR`; see `spark_runtime::prefix_cache::nvme`): paging an
//! evicted prefix back in before the prefix lookup, and the cross-rank
//! agreement that keeps spill-tier restores from diverging between ranks.
//!
//! Construction and env parsing live in `factory/build/kv_nvme.rs`.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::prefix_cache::{NvmePrefixTier, PrefixCache, RestorePlan};

use super::prefix_blocks::alloc_block_evicting;
use super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    fn nvme_tier(&self) -> Option<&dyn NvmePrefixTier> {
        self.prefix_cache.nvme().filter(|t| t.is_enabled())
    }

    /// How many blocks of `plan.disk` are worth reading back.
    ///
    /// Pure attention: all of them (every restored block is a skipped prefill
    /// block). Hybrid SSM (GLM-5.3's KDA layers): a prefill can only resume
    /// from an SSM snapshot anchor — without one the matched KV is recomputed
    /// anyway ("Prefix cache hit … but no SSM snapshot"), and KV past the
    /// deepest anchor is replayed through every layer regardless. So restore
    /// exactly up to the deepest usable anchor (resident OR spill-tiered), and
    /// nothing when there is none.
    fn nvme_blocks_worth_restoring(
        &self,
        tier: &dyn NvmePrefixTier,
        plan: &RestorePlan,
        tokens: &[u32],
        seq: &SequenceState,
        bs: usize,
    ) -> usize {
        let run = plan.disk.len();
        if self.config.num_ssm_layers() == 0 {
            return run;
        }
        let limit = plan.resident_tokens + run * bs;
        let depth = tier.snapshot_anchor_depth(tokens, limit, seq.session_hash, seq.adapter_id);
        if depth <= plan.resident_tokens || depth < crate::model::mtp_carry::marconi_min_tokens() {
            return 0;
        }
        (depth - plan.resident_tokens).div_ceil(bs).min(run)
    }

    /// Page the on-disk continuation of `tokens`' cached prefix back into
    /// fresh KV blocks, so the resident-only `lookup` that follows matches it.
    /// No-op when the tier is off. See [`restore_prefix`].
    pub(in crate::model) fn nvme_restore_prefix(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        kv_cache: &mut PagedKvCache,
        stream: u64,
    ) {
        let Some(tier) = self.nvme_tier() else {
            return;
        };
        let bs = kv_cache.block_size();
        let t0 = std::time::Instant::now();
        let Some(r) = restore_prefix(
            self.prefix_cache.as_ref(),
            kv_cache,
            self.gpu.as_ref(),
            tokens,
            seq.adapter_id,
            stream,
            |plan| self.nvme_blocks_worth_restoring(tier, plan, tokens, seq, bs),
        ) else {
            return;
        };
        if r.wanted == 0 {
            tracing::debug!(
                "NVMe prefix restore skipped: {} on-disk blocks past {} resident tokens but no \
                 usable SSM anchor — recompute",
                r.on_disk,
                r.resident_tokens,
            );
            return;
        }
        let s = tier.nvme_stats();
        tracing::info!(
            "NVMe prefix restore: {}/{} blocks ({} tokens after {} resident) in {:.1} ms{}; \
             tier {}/{} slots, {} spills, {} restores, {} failures",
            r.restored,
            r.wanted,
            r.restored * bs,
            r.resident_tokens,
            t0.elapsed().as_secs_f64() * 1e3,
            if r.failed {
                " — a record FAILED (I/O or verification): recomputing from there"
            } else if r.allocated < r.wanted {
                " — KV pool exhausted: recomputing the rest"
            } else {
                ""
            },
            s.slots_used,
            s.max_slots,
            s.spills,
            s.restores,
            s.spill_failures + s.restore_failures,
        );
    }

    /// Whether ranks must agree on a spill-tier restore: a multi-rank world
    /// with either spill tier on. There a fault-in / read can succeed on one
    /// rank and fail on another, and a diverged Marconi skip means diverged
    /// `proc_count`s → mismatched collectives → deadlock. Off (single rank or
    /// no tier) this is `false` and costs nothing, so the default path is
    /// unchanged.
    fn spill_tier_agreement_active(&self) -> bool {
        self.multi_rank_protocol_active()
            && (self.ssm_tier_store.is_some() || self.nvme_tier().is_some())
    }

    /// All-rank agreement on the Marconi restore decision. Every rank calls
    /// this exactly once per chunk-0 prefix lookup (with `eligible = false`
    /// when it has no anchor) so the collectives pair up. Returns `true` only
    /// when EVERY rank is eligible at the SAME depth; otherwise all ranks take
    /// the no-restore path together (a recompute, never a divergence).
    pub(in crate::model) fn agree_marconi_restore(
        &self,
        depth: usize,
        eligible: bool,
    ) -> Result<bool> {
        if !self.spill_tier_agreement_active() {
            return Ok(eligible);
        }
        let local = if eligible {
            u32::try_from(depth).unwrap_or(u32::MAX - 1).max(1)
        } else {
            0
        };
        let (min, max) = agree_min_max(local, |v| self.ep_min_u32(v))?;
        if min != max {
            tracing::info!(
                "spill-tier rank agreement: Marconi anchor differs across ranks (local={local}, \
                 min={min}, max={max}) — all ranks recompute this prefix"
            );
        }
        Ok(eligible && min == max)
    }
}

/// What one [`restore_prefix`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RestoreOutcome {
    pub resident_tokens: usize,
    pub on_disk: usize,
    pub wanted: usize,
    pub allocated: usize,
    pub restored: usize,
    pub failed: bool,
}

/// The restore cycle: plan (pins the path) → allocate up to `want(plan)`
/// blocks (evicting/spilling others; the pin protects this path) → read and
/// verify → promote in the tree → hand back everything not adopted.
///
/// Best effort by construction: any shortfall (no free blocks, I/O error, a
/// record failing verification) only shortens the match; the prefill
/// recomputes the rest, and unverified bytes are never adopted. `None` when
/// there is nothing on disk to restore (or the tier is off).
pub(crate) fn restore_prefix(
    prefix_cache: &dyn PrefixCache,
    kv_cache: &mut PagedKvCache,
    gpu: &dyn GpuBackend,
    tokens: &[u32],
    adapter_id: u64,
    stream: u64,
    want: impl FnOnce(&RestorePlan) -> usize,
) -> Option<RestoreOutcome> {
    let tier = prefix_cache.nvme().filter(|t| t.is_enabled())?;
    if !kv_cache.nvme_attached() {
        return None;
    }
    let bs = kv_cache.block_size();
    let plan = tier.plan_restore(tokens, bs, adapter_id);
    if plan.disk.is_empty() {
        return None;
    }
    let wanted = want(&plan).min(plan.disk.len());
    let mut blocks = Vec::with_capacity(wanted);
    while blocks.len() < wanted {
        match alloc_block_evicting(kv_cache, prefix_cache, gpu) {
            Some(b) => blocks.push(b),
            None => break,
        }
    }
    let (mut ok, mut failed) = kv_cache.nvme_read(&plan.disk[..blocks.len()], &blocks, gpu, stream);
    // Slotted index tails: a restored block is a shared cached block and must
    // own NO tail (the index kernels then skip it). Its tail was released when
    // it was last freed, but that release reaches the device map only with the
    // next publish — do it now, on the stream the prefill will read on.
    if ok > 0
        && let Err(e) = kv_cache.lend_tail_slots(&[], gpu, stream)
    {
        tracing::warn!("NVMe prefix restore: tail-map publish failed ({e:#}) — recomputing");
        (ok, failed) = (0, false);
    }
    let give_back = tier.complete_restore(tokens, bs, adapter_id, &plan, &blocks[..ok], failed);
    // Allocated but not adopted (read failed / not reached): ref 1 → free.
    for &b in blocks[ok..].iter().chain(&give_back) {
        kv_cache.return_evicted_block(b);
    }
    Some(RestoreOutcome {
        resident_tokens: plan.resident_tokens,
        on_disk: plan.disk.len(),
        wanted,
        allocated: blocks.len(),
        restored: ok
            - give_back
                .iter()
                .filter(|b| blocks[..ok].contains(b))
                .count(),
        failed,
    })
}

/// `(min, max)` of `local` across ranks from a min-reduction alone
/// (`max = !min(!v)`). Both reductions always run, on every rank.
fn agree_min_max(local: u32, mut ep_min: impl FnMut(u32) -> Result<u32>) -> Result<(u32, u32)> {
    let min = ep_min(local)?;
    let max = !ep_min(!local)?;
    Ok((min, max))
}

#[cfg(test)]
#[path = "kv_nvme_tests.rs"]
mod tests;
