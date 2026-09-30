# GLM-5.3-Flash: prefix cache spilled to NVMe

Status: implemented on `feat/glm-nvme-cache`, compile- and unit-tested on a
CPU-only Mac. **Nothing here has run on a GPU yet.** Opt-in; no behaviour
change without the flags below.

## Goal

Today, when the GPU prefix cache is full, an idle conversation's KV blocks are
deleted, and so are its KDA (SSM) snapshots once the 16 snapshot slots are
full. When the user comes back, the whole conversation is prefilled again,
which takes tens of seconds to minutes at GLM-5.3 context lengths.

With this change, those blocks and snapshots are written to the node-local
NVMe instead of being deleted. A later prefix hit reads them back in before
the prefill starts, which should take well under a second.

## 1. What existed, and what it does for GLM today

### Prefix cache (radix tree), as served

- `--enable-prefix-caching` builds a `RadixTree`
  (`crates/spark-runtime/src/radix_tree*`). There is one node per 16-token
  block, and each node's `context_hash` is chained from its parent.
- LRU eviction (`radix_tree/inner/evict.rs`) **deletes** the least recently
  used leaf node that no sequence holds, and returns its block to the free
  list. From then on, that prefix is a cache miss.
- The SSM snapshot index (`radix_tree/snapshot*.rs`) is kept separately from
  the tree nodes. With `ATLAS_MARCONI_PREFILL_ONLY=1`, only block-aligned
  prefill checkpoints (about 74 MB each) are saved. On a hit, prefill restores
  the deepest snapshot at or below the matched length and replays forward from
  there (`prefill_b/prefix_lookup.rs`). For a hybrid model like GLM, a KV hit
  with **no** snapshot is recomputed in full: "Prefix cache hit … but no SSM
  snapshot — recomputing all KV".
- Rank agreement (F83): each rank computes its own `matched_tokens`, and the
  ranks agree on the minimum with `ep_min_u32`. The snapshot restore decision
  was **not** synchronised; it was only correct because both ranks' snapshot
  pools happen to evolve identically.

### `--high-speed-swap` (HSS): not usable for this

`crates/spark-storage/src/high_speed_swap*`, `model/block_mgmt.rs`,
`serve_phases/*`:

- HSS offloads a single sequence's **live** KV during **decode**, using a
  sliding window. `--high-speed-swap-cache-blocks-per-seq` (default 64) blocks
  stay resident in GPU memory, older blocks go to per-layer NVMe files, and a
  low-rank predictor chooses which blocks to stream back for tiled attention
  (`ATLAS_HIGH_SPEED_SWAP_REPLACE`, which its own log line marks as "UNTESTED
  on real models").
- Prefill never reads from disk (Issue #31 in `block_mgmt.rs`).
- Its on-disk layout (`GroupLayout`) hard-codes BF16 GQA K/V, striped per KV
  head, sized from `num_key_value_heads`/`head_dim`. GLM's cache is different:
  - a single 512-wide latent head stored as FP8-G128 (528 B per token per
    layer), with V aliasing K;
  - a separate pooled sparse-index cache;
  - sparse MLA decode kernels that do not go through HSS's tiled path.
- Its radix-tree integration (`disk_block_id` refcounts, `matched_disk_block_ids`)
  only keeps disk IDs alive for nodes that are still in the tree. Eviction
  still deletes the node. **HSS never makes an evicted prefix findable again.**
- `kv_paging` / `rdma_kv_backend` are alternative storage backends for HSS
  (a remote RDMA peer), so the same limits apply.

Verdict: none of HSS is reused. The new tier refuses to run alongside it
(`attach_nvme_spill` errors if `cache_blocks_per_seq` is set).

### SSM snapshot spill tier: reused as-is for the KDA snapshots

`ATLAS_SSM_TIER*`, `model/ssm_tier/`, `model/ssm_snapshot_spill.rs`,
`radix_tree/snapshot_tier.rs`:

- When all 16 Marconi slots are in use, the victim snapshot is spilled
  (spill-not-drop) instead of dropped, provided it is at least
  `ATLAS_SSM_SPILL_MIN_TOKENS` deep (default 1024). The index entry stays, so
  a warm lookup finds it and faults it back into a slot (all three prefill
  paths do this, gated by `ATLAS_SSM_FAULT_MIN_TOKENS`, default 256).
- With `ATLAS_SSM_TIER_UNIFIED=1`, the store is a `Residency` with:
  - a host-RAM hot arena of `ATLAS_SSM_TIER_SLOTS` × blob, **allocated up
    front, default 64**;
  - LRU spill to an O_DIRECT swap file in `ATLAS_SSM_TIER_SWAP_DIR`;
  - a record cap from `ATLAS_SSM_TIER_DISK_GB`.
- For GLM the capability gate passes (the model has recurrent state), and the
  blob is `34 × (h + conv)` per rank. Worked out from
  `test_data/glm5_next_nvidia_nvfp4_config.json`:
  - h = 32 heads × 128 × 128 × 4 B = 2 MiB (FP32, TP2);
  - conv ≈ 3 × 12 288 × 2 B;
  - blob ≈ 73.8 MB, which is a 4 KiB multiple, so the O_DIRECT arm applies.
  **This is unverified.** Check for the startup log line `unified SSM tier
  (marconi-host): O_DIRECT swap file in …`. If you see "not a 4 KiB multiple"
  instead, the swap tier is in **host RAM**.
- GB10 gotchas:
  - Host RAM is the same pool as GPU memory, so the default 64-slot hot arena
    costs **about 4.7 GB**. Set `ATLAS_SSM_TIER_SLOTS=2`.
  - Swap files are named per PID and are **never unlinked**. This is known
    defect #3 in `ssm_tier/unified.rs`. Clear the directory between runs.
- Gap closed in this change: when a spill tier is enabled, the ranks now agree
  on the Marconi restore (see §5). Before this, a fault-in that succeeded on
  one rank and failed on the other made the ranks diverge on `skip_tokens`,
  which leads to mismatched collectives and a deadlock.

### Related finding, not fixed here

`--swap-space-gb` sequence preemption (`trait_impl/sequence/state_io.rs`)
saves and restores K/V with `read_block`/`write_block` only. **It does not
carry the GLM sparse-index cache** (pooled keys, scales, raw tails — with
slotted tails those also need re-lending), so a GLM sequence restored from a
preemption swap would have stale index keys. A TODO in `state_io.rs` records
this; until fixed, do not combine `--swap-space-gb` with GLM. The fix could
reuse the segment layout in `kv_cache/nvme_spill.rs`.

## 2. What changed

| Layer | File | What |
|---|---|---|
| tree | `spark-runtime/src/radix_tree/inner/nvme.rs` | On-disk node state, record-slot allocator under a fixed budget, exact on-disk LRU, spill victim selection, restore plan/pin/promote, subtree drop |
| tree | `radix_tree/inner.rs`, `inner/evict.rs` | `walk` stays resident-only (`block_idx != MAX`, a no-op when the tier is off). `insert` moves an on-disk node onto the recomputed block. `evict` routes to spill mode |
| tree | `radix_tree/nvme.rs`, `prefix_cache/nvme.rs` | `NvmePrefixTier` trait (`PrefixCache::nvme()`), `SpillOrder`/`DiskRef`/`RestorePlan` |
| tree | `radix_tree/snapshot_tier.rs` | `peek_deepest`: read-only deepest usable SSM anchor |
| bytes | `spark-runtime/src/kv_cache/nvme_spill.rs` | Record layout, batched pinned gather/scatter, trailer verify, ranged reads |
| store | `atlas-tier` `SwapStore::read_records` | Reads a run of consecutive records with one O_DIRECT `pread` |
| model | `spark-model/src/model/prefix_blocks.rs` | `apply_evicted_blocks` writes spills **before** returning blocks; a failed write falls back to plain eviction; spill-mode evict batch of 32 |
| model | `model/kv_nvme.rs` | `restore_prefix` cycle; hybrid restore policy; `agree_marconi_restore` |
| model | `prefill_b/prefix_lookup.rs` | Restore before the chunk-0 lookup; agreement before the Marconi restore |
| build | `factory/build/kv_nvme.rs`, `build.rs` | Env parsing, record file, enable, startup rank-config check |
| moves | `kv_cache/debug_impl.rs`, `prefill_b/prefix_reserve.rs`, `prefix_blocks.rs` | Pure moves that bring touched files toward the 500-LoC cap |

### Invariants

- **An on-disk node has only on-disk descendants.**
  - A spill only picks a resident node with no resident children.
  - A restore promotes an on-disk run from the top down, starting directly
    below the resident prefix.
  - `insert` moves nodes from the top down.
- **Every existing consumer of `PrefixMatch` still sees resident blocks
  only.** A restore happens before `lookup`, and turns on-disk nodes back into
  ordinary resident nodes that hold a single cache reference.
- **Only full blocks are spilled.** A spilled node's `partial_suffix` block is
  released, as happens today.

## 3. Flags

Set identical values on **every rank**. On a multi-rank world, startup
all-gathers a fingerprint of these settings and refuses to start if they
differ. The exchange happens after each rank's own setup (env parse, disk
budget, record file), and a rank whose setup failed sends a sentinel, so one
misconfigured rank stops every rank instead of leaving its peers hung.

| Variable | Meaning |
|---|---|
| `ATLAS_KV_NVME_DIR=<dir>` | Enables the KV tier. Must be on the real NVMe filesystem, not tmpfs/overlay (O_DIRECT). |
| `ATLAS_KV_NVME_GB=<GiB>` | **Required** alongside the dir (strict parse, > 0). Per-rank disk budget; the coldest on-disk blocks are dropped beyond it. |
| `ATLAS_SSM_TIER=1` `ATLAS_SSM_TIER_UNIFIED=1` `ATLAS_SSM_TIER_SWAP_DIR=<dir>` `ATLAS_SSM_TIER_DISK_GB=<GiB>` `ATLAS_SSM_TIER_SLOTS=2` | Existing KDA snapshot tier (see §1). |

Setting `ATLAS_KV_NVME_DIR` without `--enable-prefix-caching`, or together
with `--high-speed-swap`, is a startup error.

First hardware test (both nodes):

```sh
# CLI, unchanged: --enable-prefix-caching --ssm-cache-slots=16
export ATLAS_MARCONI_PREFILL_ONLY=1
export ATLAS_KV_NVME_DIR=/<nvme-mount>/atlas-kv
export ATLAS_KV_NVME_GB=200
export ATLAS_SSM_TIER=1
export ATLAS_SSM_TIER_UNIFIED=1
export ATLAS_SSM_TIER_SWAP_DIR=/<nvme-mount>/atlas-ssm
export ATLAS_SSM_TIER_DISK_GB=200
export ATLAS_SSM_TIER_SLOTS=2
export ATLAS_SSM_TIER_TIMING=1      # optional: per-spill timing lines
```

## 4. Disk layout

- **One record file per rank:** `<dir>/atlas-kv-prefix.<pid>.r<rank>.swap`,
  opened O_DIRECT and **unlinked immediately**.
  - Its space is returned when the process exits for any reason, including a
    crash.
  - No other process can ever read a stale record.
  - `ls` does not show it; use `df` or `lsof` instead.
- **Record `slot` is at offset `slot × record_bytes`**, and the file grows
  sparsely.
- **Record contents** (this rank's bytes for one 16-token block):
  1. For each attention layer:
     - `K` (the full per-block stride; for FP8-G128 that includes the scales);
     - `V`, only when it does not alias K (GLM: aliased, so omitted);
     - pooled index values, plus pooled index scales when present.
  2. Zero padding.
  3. A 24-byte trailer: `magic "ATLKVNV1" | tag | checksum`.
- **Raw index tails are not stored.** They only hold the un-pooled keys and
  gates of a pool that is still being filled. `glm_index_kpool_finalize`
  pools each 4-token group when its last token is written, and only full
  blocks (all 4 pools finalised) are spilled.
  - Slotted tails (prefix caching + `ATLAS_MARCONI_PREFILL_ONLY`) change
    nothing in the record. A spilled block's lent tail returns with its last
    KV ref (`return_evicted_block`), and a restored block is published
    tail-less before the prefill reads it.
  - If the index moves to scaled FP8, the scales segment is picked up
    automatically.
- **GLM-5.3 sizes per rank:**
  - payload: 11 layers × (8448 + 1024) = 104 192 B;
  - record: **106 496 B per block** (26 × 4 KiB), which is **6.66 KB per
    token**;
  - 200 GiB ≈ 2.0 M blocks ≈ 32 M tokens.
- **Tag:** a hash of the node's causal `context_hash` and a spill epoch, held
  by the tree.
- **Checksum:** a 4-lane, 64-bit multiply-fold over the payload.
- **Host RAM:** one tree node per on-disk block, roughly 150–250 B each
  (estimated; about 0.4 GB at 2 M blocks), plus 3.4 MB of pinned staging.

## 5. Spill, restore, and rank agreement

### Spill (at eviction)

1. `alloc_block_evicting` asks the tree for 32 blocks. With the tier off it
   asks for 1, as before.
2. The tree converts the LRU victims to on-disk nodes and returns
   `SpillOrder`s.
3. `apply_evicted_blocks` then:
   1. syncs the whole device (`cuCtxSynchronize`) — a victim's last writes
      can still be in flight on the prefill stream or another non-blocking
      stream;
   2. queues 22 async D2H copies per block into pinned staging;
   3. syncs once;
   4. stamps the trailers and `pwrite`s each record;
   5. **only then** returns the blocks to the free list.

A failed write triggers `spill_failed`, which drops that node, so the block
is evicted as it would be without the tier. Warnings are throttled (1, 2, 4,
8, …).

### Restore (prefill_b, chunk 0, before `lookup`)

1. `plan_restore` walks the prompt's verified full blocks, finds the on-disk
   run below the resident prefix, and **pins** the whole path (one extra
   radix ref), so evictions caused by the restore cannot take it.
2. It decides how many blocks to restore:
   - pure attention: all of them;
   - hybrid (GLM): up to the deepest usable Marconi anchor, resident or
     spill-tiered (`snapshot_anchor_depth`, read-only), and nothing if there
     is no anchor at least `marconi_min_tokens` deep. Without an anchor the
     KV would be recomputed anyway, and KV past the anchor is replayed through
     every layer regardless.
3. It allocates blocks (possibly spilling other conversations), then reads
   runs of consecutive slots with one `pread` each. A chain spilled leaf-first
   gets consecutive slots in descending order, and the reader handles either
   direction. It verifies every trailer, then scatters with `copy_h2d_async_retained`
   and one stream sync per 32 records.
4. `complete_restore` promotes the verified prefix, drops the subtree of a
   record that failed, and unpins. Blocks that were not adopted are freed.
5. The ordinary `lookup` then matches the restored blocks. From here on the
   existing flow applies: F83 min-agreement, then Marconi lookup/fault-in and
   replay.

### Expected latency (per rank; both ranks run in parallel on their own NVMe)

Measured input: 9.6 GB/s sequential O_DIRECT read. Everything else below is
an estimate to be checked against the `NVMe prefix restore: … in X ms` log
line.

| Context | Blocks | Bytes/rank | Pure NVMe time at 9.6 GB/s | Estimated restore (contiguous runs) | Fragmented slots |
|---|---|---|---|---|---|
| 16 K | 1 000 | 106 MB | 11 ms | ~0.07 s | ~0.15 s |
| 32 K | 2 000 | 213 MB | 22 ms | ~0.15 s | ~0.3 s |
| 64 K | 4 000 | 426 MB | 44 ms | ~0.3 s | ~0.6 s |
| 128 K | 8 000 | 852 MB | 89 ms | ~0.6 s | ~1.2 s |

How the estimates were built:

- **Contiguous-run estimate:** about 60–80 µs per block, dominated by
  submitting 22 `cuMemcpyAsync` calls at an assumed 2–3 µs each. NVMe time at
  a coalesced ~6–9 GB/s adds about 12–17 µs per block, and the checksum a few
  µs.
- **Fragmented-slots estimate:** once the budget is full, slots are recycled
  and runs break into single records. Reads are then one synchronous 104 KB
  `pread` per block, estimated at 80–100 µs each at QD1.
- **Plus the KDA snapshot:** faulting in a 73.8 MB snapshot (about 10–30 ms
  from disk), then the replay from the anchor, which is existing behaviour.
- **Compared with today:** a full re-prefill of the same context takes tens
  of seconds or more.

**Spill cost:** estimated at about 60–90 µs per evicted block, spread across
the prefill that caused the eviction. For a 100 K-token cold prefill that
evicts about 6 250 blocks, that is roughly 0.4–0.6 s, against tens of seconds
of prefill.

### Rank-agreement guarantees

- **KV:** each rank restores on its own, and the existing F83 `ep_min_u32` on
  `matched_tokens` then caps every rank to the shortest match.
  - A rank that restored more keeps the extra blocks as ordinary cache
    entries. They are released and re-looked-up, exactly as in F83 today.
  - A restore issues no collectives, so its partial failure on one rank cannot
    deadlock.
- **Marconi:** when any spill tier is enabled on a multi-rank world, every
  rank calls `agree_marconi_restore` exactly once per chunk-0 lookup, with
  depth 0 if it has no eligible anchor. This is two `ep_min_u32` rounds (min
  of the value and min of its complement). A restore happens **only if every
  rank is eligible at the same depth**. Otherwise all ranks take the
  no-restore path together: a recompute, never a divergence.
  - With both tiers off this is a no-op, so there are no new collectives on
    the default path.
- **Configuration:** at startup, every multi-rank world all-gathers
  `(KV slots, record size, ATLAS_SSM_TIER set)` (or a failure sentinel) and
  aborts on a mismatch or a failed rank, because the flags gate paired
  collectives. This single 8-byte all-gather at startup is the one addition
  to the default path.

## 6. Failure handling (never serve stale data)

| Event | Behaviour |
|---|---|
| Disk full / `pwrite` error at spill | The node is dropped and the block evicted as without the tier. Throttled warning. Later hits on that prefix recompute. |
| Budget full | The coldest droppable on-disk leaf is dropped (LRU). If the eviction candidate is colder than every on-disk block, the candidate itself is deleted instead. |
| `pread` error / short read at restore | The verified prefix is kept. The failing record's node and its on-disk subtree are dropped. The rest is recomputed. |
| Wrong tag, bad magic, or checksum mismatch | Treated the same as a read error. The bytes are never scattered into a block the tree adopts. |
| No free blocks to restore into | Partial restore. The remainder is recomputed. |
| KDA snapshot fault-in miss or error | Existing behaviour (reap on miss, retain on error), and with rank agreement it is always a symmetric recompute. |
| Process crash | The record file is already unlinked, so its space is reclaimed. No state survives a restart; the cache starts empty. |

## 7. Not done / unverified

- **Nothing has run on a GPU.** These are unverified on hardware:
  - the CUDA pinned-staging copies;
  - O_DIRECT alignment on the Samsung filesystem (the pinned buffer from
    `cuMemAllocHost` is assumed to be page-aligned; otherwise
    `DirectSwapFile` bounces, which is correct but slower);
  - every latency figure above.
- **Restore only runs in `prefill_b`** (chunked prefill), the multi-rank path.
  `prefill_a`/`prefill_c` and the single-rank batched reservation path stay
  resident-only: correct, just no restore.
- **Assumptions to watch on hardware:**
  - all ranks build through `build_model` (as `agree_kv_blocks` already
    requires);
  - GLM sets `uses_local_mla_prefill() == false`, so the prefix skip is taken;
  - tails of full blocks are never read.
- **Worth doing next:**
  - reads at queue depth > 1 (io_uring, or allocating slots in extents) so
    fragmented slots stay fast;
  - a batched scatter (`cuMemcpyBatchAsync` or a gather kernel) to remove the
    per-copy submit cost that dominates;
  - keeping the on-disk copy when a block is restored, so re-evicting it
    needs no write;
  - unlinking the SSM tier's swap file (unified.rs defect #3).
- **Not persisted across restarts, by design.**

## 8. Hardware validation plan

1. **Baseline, cache off:** N = 24 conversations with fixed seeds (distinct
   system prompts; 8–64 K tokens each, total well past GPU KV capacity; see
   the capacity line `KV cache: … blocks`). Run 2 turns each, temperature 0,
   and record each turn-2 answer and TTFT.
2. **Cache on, tiers off (control):** the same run. Revisiting a conversation
   after the others have filled the cache should log `cached_tokens=0` or no
   snapshot, and TTFT should look like a full prefill.
3. **Cache on, flags from §3:**
   - Fill: run turn 1 of all N conversations sequentially. Expect the
     `prefix cache NVMe spill tier ON` line at startup and `SSM tier disk tier
     ENGAGED` once the snapshot hot arena is full. The KV tier's spill count
     shows up in the stats on each restore line.
   - Revisit: run turn 2 of each conversation, oldest first. Expect, per
     request on **both** ranks:
     - `NVMe prefix restore: k/k blocks … in X ms`, with X under 1 s at 128 K;
     - `SSM tier fault-in: restored spilled snapshot at token D`;
     - a Marconi intermediate hit that replays only a few dozen tokens;
     - usage `cached_tokens > 0`;
     - **TTFT ≈ restore time + replay**, not a full prefill;
     - no `spill-tier rank agreement: … differs` lines.
   - Correctness: every turn-2 answer is **byte-identical** to the cache-off
     baseline at temperature 0. Also run the golden BFCL leg with the flags on
     and compare against the flags-off score.
4. **Failure drills:**
   - Tiny budget (`ATLAS_KV_NVME_GB=1`): expect disk drops, and still correct
     answers.
   - Unwritable dir: expect a startup error.
   - Env on one rank only: expect a startup abort naming the mismatch.
   - Fill the filesystem mid-run: expect throttled spill warnings, answers
     still correct, and no hang.
5. **Soak:** 2 h of mixed agentic traffic, checking that no free-block leak
   appears (`num_free_blocks` returns to baseline when idle), that RSS is
   bounded, and that there are no deadlocks.
