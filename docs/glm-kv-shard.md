# GLM-5.3-Flash token-sharded MLA latents (`ATLAS_GLM_KV_SHARD=1`)

Status: implemented behind an opt-in flag, compiled and unit-tested on a
GPU-less host only. **Not yet validated on hardware** (see the last section).
Default off: with the flag unset every allocation, kernel and launch is the
one the engine ran before.

## Why

GLM-5.3-Flash on a TP2 pair (two GB10, 128 GB unified memory each) stores
the full MLA latent cache and the full sparse-index cache for every token on
BOTH ranks. The latent is 90% of the per-token KV bytes, so four concurrent
512K-token contexts do not fit. The latent only has to exist once per pair.

## Layout

- **Ownership by block-id residue, pinned to the logical index.** Block `b`
  is stored by rank `b % 2` at local slot `b / 2` of that rank's per-layer K
  pool (V aliases K). The rule lives in
  `crates/spark-runtime/src/kv_cache/latent_shard.rs` (`LatentShard`) and,
  for the device side, in `kernels/gb10/glm-5.3-flash/nvfp4/glm_kv_shard.cu`.
- **The ranks' physical ids differ; their residues do not.** Each rank runs
  its own allocator and frees in its own order (DFlash verify-row rollback,
  prefix-cache inserts and sequence frees iterate rank-local state), so after
  real traffic the same logical block has different ids on the two ranks
  (seen on hardware: a 128-block table differed from its first entry). A
  sharded pool therefore keeps one free list per id residue and draws the
  block for logical index `l` from list `l % 2`
  (`crates/spark-runtime/src/kv_cache/free_blocks.rs`,
  `PagedKvCache::try_alloc_block_at`). Every table entry then satisfies
  `b % 2 == l % 2`, so both ranks agree that logical block `l` is stored by
  rank `l % 2`, whatever the ids are. The sequence allocators
  (`block_mgmt::ensure_blocks_through_{prefill,decode}`) pass the index;
  index-less `alloc_block` / `try_alloc_block` are refused under the shard.
- **Prefix caching needs no change:** sharing is positional (a cached block
  is only matched at its own logical index), so a shared block keeps its
  residue and its owner.
- **Why residues rather than a per-rank slot map:** slot `b / 2` needs no
  map, no device table and no kernel change, and a rank can never be handed
  more blocks than its `ceil(N / 2)` slots. The cost is that a residue can
  run dry while the other still has blocks (many short sequences all at
  logical 0, or a prefix tree with more even than odd blocks):
  `num_free_blocks()` reports `2 x` the scarcer residue, which the
  scheduler's admission and prefix-cache reclaim loops already act on, and
  `block_mgmt::alloc_block_evicting` evicts from the prefix cache until the
  residue has a block. A slot map with spare slots would absorb such
  imbalance only by paying for the spare latents up front; the same bytes
  buy a larger pool.
- **Everything else is replicated:** block tables (in logical structure),
  refcounts, the prefix cache, eviction, the pooled semantic-index keys, the
  lent index tails and all KDA state.
- **Writes:** `write_kv_cache_bf16_latent` maps each row's global slot to the
  owner's local slot (`glm_kv_shard_map_slots`, `-1` elsewhere; every GLM
  latent writer skips `-1`). Only the owner writes a row's latent.
- **Fail closed:** under the shard, `PagedKvCache::k_pool_ptr / v_pool_ptr /
  k_cache_ptr / v_cache_ptr` panic (naming the caller) and `read_block /
  write_block` (sequence save/restore) return an error, so any latent reader
  that was not ported crashes loudly instead of reading another block.
  `zero_block(s)` and `poison_block` touch only owned local slots.

## Attention

Both ranks hold the replicated index keys, so both compute the identical
top-2048 selection locally (no top-k exchange). Every GLM MLA attention that
reads the latent cache in the DFlash serving lane goes through
`glm_chunk_attention` (prefill chunks, single-owner and owner-batched DFlash
verify, fused prefill+verify), plus the single-row eager decode
(`attention_forward_mla`). In shard mode each owner takes one of two forms
(`crates/spark-model/src/layers/qwen3_attention/prefill/paged_glm_shard.rs`):

### Merge form (owners of at most 64 rows: verify, decode, prefill tails)

1. Exchange the owner's absorbed queries (this rank's 32 heads,
   `rows x 32 x 512` BF16) with the peer.
2. `glm_kv_shard_localize`: rewrite the selected token ids to this rank's
   local token ids (read through an identity block table over the local pool);
   tokens stored by the peer become `-1`, which every GLM sparse kernel skips.
   Dense owners (sequence end <= 2048) generate causal ids instead.
3. Run the existing tensor-core split kernel
   (`glm_sparse_mla_prefill_{fp8g128,bf16}_head32_tc_kv_pad_split`) for the
   PEER's heads over this rank's tokens, merge its partitions to one FP32
   partial + natural LSE (`glm_sparse_decode_split_merge_f32`, new), and
   exchange that partial (`rows x 32 x 513 x 4` bytes) with the peer.
4. Run the split kernel for this rank's own heads over its tokens, append the
   peer's partial as the last partition, and merge everything in one exact LSE
   merge (the existing `glm_sparse_decode_split_merge`) into the head-sharded
   BF16 attention output. `W_uv`, `o_proj` and the TP all-reduce are
   unchanged.

A partition that owns no selected token has `LSE = -inf` and weight 0, so a
rank owning nothing contributes nothing (the unit tests check this and the
equivalence with plain softmax attention on a CPU mirror).

Why this over fetching the remote selected latents: per row the merge form
moves 32 KiB of queries + 64 KiB of FP32 partials each way, with fixed shapes
and no host sync. Fetching latents moves ~1,025 remote tokens x 528 B = 541 KiB
per row (less after de-duplicating across an owner's rows, but then the
payload size is data dependent and needs a host sync or a padded worst case).
At the served shapes (<= 4-8 owners x <= 8 rows) the merge form moves 5-10x
fewer bytes. Attention FLOPs and latent bytes read are unchanged: each rank
reads only its half of the selected tokens, for twice the heads.

### View form (owners of more than 64 rows: prefill chunks)

The chunk's queries select most of the history, so each rank assembles the
sequence's whole latent history `[0, end)` in a scratch view: it reads the
block table to the host (one sync per layer), copies its own blocks into
their logical positions, and exchanges the rest with the peer in 2,048-block
pieces (packed in logical order, so both sides agree on the order). The
unchanged kernels (TC sparse prefill, the BF16 dequantized view, the dense
<= 2048 path) then read the view through an identity block table. The
chunk's new rows are written (owner-only) before the view is assembled, so
the view includes them.

## Memory (per rank, fp8_g128, 11 MLA layers, 16-token blocks)

| per token | before | after |
|---|---|---|
| MLA latent (11 x 528 B) | 5,808 B | 2,904 B |
| pooled index keys (11 x 64 B) | 704 B | 704 B |
| index tail-slot map + identity table | 0.25 B | 0.375 B |
| **total** | **6,512.25 B** | **3,608.4 B** (-44.6%, pool x1.80) |

(The task brief quoted 5,892 B for the latent; the code's layout is
16 x 528 B per block per layer = 5,808 B per token over 11 layers.)

Fixed cost of the shard scratch (allocated before KV sizing and charged to
the budget through `GlmCachePlan::latent_sharded`): at `--max-seq-len 524288`
a 32,772-block view (264 MiB for fp8_g128, 512 MiB for BF16), ~38 MiB of work
space for exchange pieces / merge partials, and small lists — about 0.3 GiB.

At the production budget (852,448 tokens x 6,512 B = 5.55 GB at
`--gpu-memory-utilization 0.90`) the sharded pool is
(5.55 GB - 0.32 GB) / 3,608 B = **~1.45M tokens** (was 852K). Four full
512K contexts need 2,097,152 tokens = 7.9 GB with the shard (13.7 GB
without), so the shard alone closes most, not all, of the gap: another
~2.3 GB of budget is still needed for 4 x 512K.

## Expected speed impact (reasoned, not measured)

- **Verify / decode (merge form):** two pair exchanges per owner per MLA
  layer (4 owners x 11 layers = 88 per step; ~15-40 us each on the RDMA pair,
  ~34 MB per step each way at 4 x 8 rows) plus a localize kernel, one extra
  split launch (same bytes read), one FP32 merge and two small copies. Expect
  roughly +1.5-3.5 ms per verify step (1-3% of today's 110-250 ms steps).
  Follow-up: batch all owners of a layer into one query exchange and one
  partial exchange (22 per step).
- **Prefill (view form):** per layer per chunk one host sync, a local copy of
  the own half of the history, and an exchange of the peer's half
  (at 256K history ~69 MB each way per layer, ~31 ms per 8K chunk over 11
  layers at ~25 GB/s, ~3% of a chunk; ~6% at 512K). Follow-up: prefetch layer
  l+1's view while layer l computes (the history outside the chunk is known
  before the chunk starts), and plan once per chunk instead of per layer.
- **CUDA graphs are suppressed** when the cache is sharded (the exchanges and
  the view's host sync are eager). The DFlash verify lane is already eager;
  plain decode loses graphs.

## Numerics

Mathematically exact (no quantization or approximation is added), but not
bitwise identical to flag-off: the merge form changes the FP32 summation
order of the softmax partitions (as the existing split verify already does),
and dense <= 2048 owners take the tensor-core split kernel over causal ids
instead of the BF16 dense kernel. The view form runs the same kernels on the
same bytes as flag-off.

## Supported / not supported

Supported (implemented): prefill chunks (incl. sequence-parallel prefill,
pieces across the 2048 dense/sparse boundary), DFlash verify (single owner,
owner-batched, fused prefill+verify), single-sequence eager decode, BF16 and
fp8_g128 caches, prefix caching (positional sharing keeps each block's owner).

Not supported (fail closed with the `k_pool_ptr` panic or a boot error):
multi-sequence batched decode (`decode_batch` / multi-seq MLA), the repaired
MTP K3/C2/C4/independent lanes, `ATLAS_GLM_MULTI_SEQ_SPARSE`, the long-verify
serial diagnostic, sequence save/restore, `--high-speed-swap`, and any world
size other than a TP2 pair with 32 heads per rank. In the DFlash lane plain
decode is reached only for sequences whose speculation is suspended; those go
through the supported single-sequence decode, but a batched bootstrap of
several such sequences would hit the fail-closed panic.

## Hardware validation plan

1. Build the engine (the new `glm_kv_shard.cu` module and the
   `glm_sparse_decode_split_merge_f32` entry point must compile). Boot the
   production profile on both Sparks with `ATLAS_GLM_KV_SHARD=1
   ATLAS_GLM_KV_SHARD_CHECK=1`. Check the boot log on both ranks:
   - `KV latent shard: rank r of 2 stores X of N blocks' latents; ~300 MiB
     exchange scratch` (X = ceil(N/2));
   - `KV cache: ... -> N blocks x 16 tok/block = T max KV tokens` with T about
     1.7x the flag-off boot at the same `--gpu-memory-utilization`;
   - `KV cache: N blocks x 11 layers = G GB total (V aliases K)` with G about
     half the flag-off value's latent part.
   `..._CHECK=1` makes every sharded attention check on the host that each
   table entry's residue matches its logical index and exchange the block
   count with the peer (fails if they differ); drop it for performance runs.
   The view form checks the residues on every call regardless.
2. Greedy equality: the same prompts at temperature 0 with the flag off and
   on (short, 16K, 64K prompts; single sequence and 4 concurrent). Expect
   identical output for most prompts; where the text diverges, compare the
   per-token logprobs up to the first divergence (differences should be at the
   FP32-reassociation level, well below the top-1/top-2 margin at the
   divergence point). A systematic difference in the first tokens means a bug.
3. Long-context needle retrieval at 64K, 200K and 400K with the flag on vs
   off, then 4 concurrent 512K contexts (the goal) — the flag-off run cannot
   hold them.
4. DFlash acceptance: mean accepted drafts per step must match flag-off
   within noise (a broken attention collapses acceptance first).
5. Performance: the RigMark / matrix cells vs flag-off; per-step time of
   owner-batched verify at 1 and 4 owners (`ATLAS_GLM_VERIFY_PROFILE=1` gives
   the MLA vs KDA split); prefill throughput at 16K/128K/256K history.
6. Prefix caching on (`--ssm-cache-slots > 0`) with shared prefixes.
7. If a panic names `k_pool_ptr`/`v_pool_ptr`, the backtrace names a latent
   reader that is not shard-aware (see "Not supported").
