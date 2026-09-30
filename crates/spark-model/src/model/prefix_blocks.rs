// SPDX-License-Identifier: AGPL-3.0-only

//! Prefix-cache ↔ KV-pool block ownership: applying evictions (including the
//! NVMe spill tier's write-before-free), evict-until-free allocation, and the
//! ref obligations of `insert` / prefix hits. Split out of `block_mgmt.rs`
//! (500-LoC cap); re-exported there so existing imports stand.

use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::prefix_cache::PrefixCache;

use super::types::TransformerModel;

impl TransformerModel {
    /// [`apply_evicted_blocks`] against this model's prefix cache and GPU.
    pub(crate) fn apply_evicted(
        &self,
        evicted: spark_runtime::prefix_cache::EvictedBlocks,
        kv_cache: &mut PagedKvCache,
    ) {
        apply_evicted_blocks(
            evicted,
            kv_cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
        );
    }
}

/// Apply an `EvictedBlocks` result to the production cache and the HSS
/// orchestrator. Physical blocks return to the free list; disk-block IDs get
/// `dec_disk_ref`'d (Phase 6.1.e). When HSS isn't engaged the disk vec is
/// empty and this becomes a thin loop over the physical blocks.
///
/// NVMe spill tier: the blocks named in `evicted.spill` are written to disk
/// FIRST — before any of them can reach the free list and be overwritten (on
/// the fast path: gathered into staging that a queued write owns). A write
/// that fails degrades to a plain eviction (the tree drops that node).
pub(crate) fn apply_evicted_blocks(
    evicted: spark_runtime::prefix_cache::EvictedBlocks,
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn PrefixCache,
    gpu: &dyn GpuBackend,
) {
    if !evicted.spill.is_empty() {
        let failed = kv_cache.nvme_write(&evicted.spill, gpu, gpu.default_stream());
        drop_failed_spills(&failed, kv_cache, prefix_cache);
    }
    let free_before = kv_cache.num_free_blocks();
    let n_evicted = evicted.physical.len();
    for block in &evicted.physical {
        kv_cache.return_evicted_block(*block);
    }
    // Reclaim accounting. A node hands back exactly ONE ref, so a block that a
    // LIVE sequence also holds correctly survives its node's eviction and is
    // freed later by that sequence — `gained < n_evicted` is therefore normal
    // under load and is NOT a leak. (This line previously claimed "no future
    // eviction can release" them, which was true only while the cache's ref
    // could land on a block no node referenced; that mismatch is fixed, so the
    // shortfall now just measures how much of the LRU tail is still in use.)
    let gained = kv_cache.num_free_blocks().saturating_sub(free_before);
    if gained < n_evicted {
        tracing::debug!(
            "prefix-cache evict reclaimed {gained}/{n_evicted} blocks (free={}): \
             the rest are still held by live sequences and free with them",
            kv_cache.num_free_blocks(),
        );
    }
    if !evicted.disk_block_ids.is_empty()
        && let Some(res) = spark_storage::with_local(|hss| {
            for id in &evicted.disk_block_ids {
                // dec_disk_ref returns the new refcount; discarded here.
                let _new_refcount = hss.dec_disk_ref(*id);
            }
            Ok(())
        })
        && let Err(e) = res
    {
        // Errors here are advisory — orchestrator absent shouldn't block
        // the cache eviction path. Log and continue.
        tracing::debug!("apply_evicted_blocks: spark_storage::with_local closure: {e:#}");
    }
}

/// Spill writes that did not reach the disk: the tree forgets those nodes (a
/// plain eviction). The fast path reports a failure after the fact, so this
/// also runs around a restore.
pub(crate) fn drop_failed_spills(
    failed: &[spark_runtime::prefix_cache::SpillOrder],
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn PrefixCache,
) {
    if !failed.is_empty()
        && let Some(tier) = prefix_cache.nvme()
    {
        for block in tier.spill_failed(failed) {
            kv_cache.return_evicted_block(block);
        }
    }
}

/// Allocate one block, evicting from the prefix cache as many times as it takes.
///
/// Evicting a radix node returns only the CACHE's reference to its block, so if a
/// live sequence still holds that block nothing is freed — one `evict(1)` can and
/// does free ZERO blocks. Both allocation paths used to evict once and then call
/// `alloc_block()`, which then failed outright with "KV cache exhausted: no free
/// blocks" while the cache still held plenty of evictable entries. Observed on the
/// agentic bench as a task failure, immediately preceded by the giveaway pair:
///
/// ```text
/// DEBUG prefix-cache evict reclaimed 0/1 blocks (free=0)
/// ERROR alloc failed in ensure_blocks_through_decode: ... free_blocks=0 ...
/// ```
///
/// Keep evicting until a block comes free or the cache has nothing left to give;
/// every iteration removes at least one node from a finite tree, so it terminates.
/// `None` means genuinely out of capacity — the caller reports exhaustion.
pub(crate) fn alloc_block_evicting(
    kv_cache: &mut PagedKvCache,
    prefix_cache: &dyn PrefixCache,
    gpu: &dyn GpuBackend,
) -> Option<u32> {
    if let Some(b) = kv_cache.try_alloc_block() {
        return Some(b);
    }
    let mut evicted_nodes = 0usize;
    loop {
        let evicted = prefix_cache.evict(kv_cache.evict_batch());
        if evicted.is_empty() {
            if evicted_nodes > 0 {
                tracing::debug!(
                    "alloc: evicted {evicted_nodes} prefix-cache node(s) without freeing a \
                     block (every one is still held by a live sequence) — out of capacity"
                );
            }
            return None;
        }
        evicted_nodes += evicted.len();
        apply_evicted_blocks(evicted, kv_cache, prefix_cache, gpu);
        if let Some(b) = kv_cache.try_alloc_block() {
            if evicted_nodes > 1 {
                tracing::debug!(
                    "alloc: freed a block after evicting {evicted_nodes} prefix-cache node(s)"
                );
            }
            return Some(b);
        }
    }
}

/// Apply the disk-ref obligation reported by a `prefix_cache.insert*` call.
/// The cache returns the disk_block_ids on which it newly took an ownership
/// ref (a node was created OR an existing node had its disk_block_id
/// populated for the first time). The caller `inc_disk_ref`s each one so
/// the swap allocator's refcount matches the cache's reachability.
///
/// Without this, the cache stores disk_block_ids whose only live ref is
/// the sequence's; when `free_sequence` decs, the ID is reclaimed by the
/// swap allocator while the cache still references it — the next prefix
/// hit then trips `inc_disk_ref` on a freed ID and panics the scheduler
/// thread (Issue #17, panic at `high_speed_swap.rs:167`).
pub(crate) fn cache_acquires_disk_refs(newly_acquired: &[u32]) {
    if newly_acquired.is_empty() {
        return;
    }
    if let Some(res) = spark_storage::with_local(|hss| {
        for &id in newly_acquired {
            if id != u32::MAX {
                hss.inc_disk_ref(id);
            }
        }
        Ok(())
    }) && let Err(e) = res
    {
        tracing::debug!("cache_acquires_disk_refs: spark_storage::with_local: {e:#}");
    }
}

/// Apply BOTH ref obligations reported by a `prefix_cache.insert*` call: the
/// disk-side refs and the cache's own KV ref on every block whose radix node
/// this insert created.
///
/// The KV half must be taken here, against the blocks the CACHE stored, rather
/// than later against the finishing sequence's `block_table`: a node that
/// already existed keeps its original block, so the sequence's block at that
/// position can be a different one entirely (no cache hit, or a swap-file
/// restore). Referencing the sequence's block left the node's block
/// unreferenced, and evicting that node then stole a live sequence's ref —
/// freeing a block still in use and handing it to a second owner. See
/// `InsertAcquired`.
pub(crate) fn cache_acquires_refs(
    acquired: &spark_runtime::prefix_cache::InsertAcquired,
    kv_cache: &mut PagedKvCache,
) {
    cache_acquires_disk_refs(&acquired.disk_block_ids);
    // Acquire before release: a block that is both (e.g. a partial slot re-set to
    // the same block via different paths) must never transiently hit 0 and get
    // handed to another sequence.
    for &block in &acquired.blocks {
        kv_cache.inc_ref(block);
    }
    for &block in &acquired.released_blocks {
        kv_cache.dec_ref(block);
    }
}

/// Phase 6.1.e: bump disk-side refcounts for blocks reused from a prefix-cache
/// hit, and push the disk_block_ids onto the sequence's history. The cache's
/// own ref keeps these slots alive across eviction; we add the seq's ref so
/// `free_sequence` can dec_disk_ref it on exit.
///
/// `matched_disk_block_ids` parallels `matched_blocks` when the entries were
/// inserted under HSS (every entry is a live disk_id). When HSS wasn't
/// engaged at insert time the slice is empty — the per-layer offload helper
/// will alloc fresh disk_ids and stream the data to disk on the first decode
/// step that touches each block.
pub(crate) fn reuse_prefix_match_disk_ids(
    matched_disk_block_ids: &[u32],
    seq_disk_block_ids: &mut Vec<u32>,
) {
    if matched_disk_block_ids.is_empty() {
        return;
    }
    if let Some(res) = spark_storage::with_local(|hss| {
        for &id in matched_disk_block_ids {
            if id == u32::MAX {
                // Mixed-mode entry — skip; the catch-up offload will populate.
                continue;
            }
            hss.inc_disk_ref(id);
            seq_disk_block_ids.push(id);
        }
        Ok(())
    }) && let Err(e) = res
    {
        tracing::debug!("reuse_prefix_match_disk_ids: spark_storage::with_local: {e:#}");
    }
}
