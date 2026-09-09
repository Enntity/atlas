# Bounded C4 sparse 24K qualification plan

2026-09-09. Design only; no 24K admission change or qualification is authorized
by this document. First pass the corrected ordinary 4K/8K/16K quality, boundary,
cancellation/reuse and clean-shutdown gates. In particular, the 22b56144 4K OFF
C3 auto-tool failure remains a failure, not a passing prerequisite. Root must
explicitly release any subsequent source or native work. No 32K claim.

## Existing evidence and prospective memory gate

Root reports the actual 22b56144 **4K** allocation: head12606 and worker13112
physical KV blocks, block size16,11 attention layers; approximately10GiB minimum
observed MemAvailable and zero swap. These are only 4K observations, not a 24K
allocation or continuous high-water proof. Preserve the4GiB headroom guard.

For a fresh24K C4 process require, independently on both ranks:

`total_blocks >= 4 * ceil(24576 / 16) + 1 = 6145`

The extra block is the permanent zeroed dummy allocated in
`model/impl_a1.rs`. Distinguish total blocks from post-dummy free blocks
(which must be at least6144). With the actual BF16 compressed512-wide K and V,
11 layers cost360448 bytes per physical block. The BF16 semantic cache adds
101376 bytes per block across11 layers, including uncompressed pooled key/gate
tails (`kv_cache/sparse_index.rs`). At6145 blocks these are2112.34MiB KV and
594.10MiB index. This is not the whole process requirement: retain model,
live KDA/rollback, activation/scratch, runtime and system headroom accounting.
Use the existing sparse-index-aware block planner, not a KV-only allowance.

## Source bounds and smallest prospective change

- `model/glm_c4.rs::SPARSE_CONTEXT_LIMIT` is the shared production16K policy
  used by C4 launch and all sparse C2/C3/C4 drain positions. Server preflight
  delegates to it. The shell duplicate is `scripts/start-glm53-ep2.sh`.
  A future bounded change raises these to24576, not32768; independent-C8 and
  MTP limits remain unchanged. Keep sparse eager-only, capacity4, BF16 KV/index,
  FP32 KDA, no overcommit and configured prefill at most1024.
- `prefill/glm_index.rs::glm_index_prefill_select` uses dynamic history scores:
  `ceil(sequence_end/4)*4` bytes per query. At24576 this is24576B. Crucially,
  the final one-row chunk has logical score capacity `1*8*2048*2=32768B` and
  fits; do not infer a32K pass from its equality boundary. Selection output
  stays2051 signed token IDs per query. Scorer grids and radix top-K iterate
  dynamic history; no16K fixed-array boundary was found in their CUDA bodies.
- `prefill/paged_glm.rs` maintains index history before selecting dense versus
  sparse at sequence_end2048. `multi_seq/mla_glm_sparse.rs` similarly updates
  dense rows' history, writes every owner's KV before reusing selector scratch,
  and waits until all selectors finish before V extraction. Preserve this order.
- `buffers/sizes.rs` derives block-table metadata from context and block size:
 1537 entries/owner at24K. Its96-row metadata envelope is625024B versus133504B
  at4K; use the actual arena sizing/upload paths without invented fixed offsets.
  LM-head row/vocabulary size and fixed-state KDA dimensions do not scale with
  history, while KDA prefill still runs bounded chunks.

## Required focused gates, not a new framework

1. Actual policy RED then GREEN:24576 accepted/24577 refused; positions24575
   accepted and24576 refused at drain widths2/3/4. Preserve dense2048 and graph
   refusals. Update the sparse MLA literal-boundary test and launcher tests.
2. Actual arena/upload controls at24K: final chunk1, all four pool-tail phases,
   high/noncontiguous physical blocks, and unchanged first/later chunk budget.
   Resolve the observed solo arena1028 versus configured1024 chunk issue before
   treating the configured1024 envelope as an issued-chunk guarantee.
3. Reuse `bench_glm_indexer.cu` and `bench_glm_sparse_mla.cu`: histories24573..24576,
   rows1/7/1024, masked2051 IDs, permuted blocks and existing numerical/canary
   checks. Existing indexer source includes32768-history cases; that is not a
   fresh execution receipt or whole-model quality evidence.
4. Root-only guarded native C1..4 near-full chat NIAH early/middle/late, exact
   facts, automatic tools and actual-ID tool-result roundtrips; no peer values.
   Calibrate actual chat usage, cap output128 (initial tool192), retain the
   input+output context check. Exercise2048 crossing, drain4→3→2→1, streamed
   cancellation/reuse and exact over-context rejection. Keep existing overall
   workload<=1800s and request<=600s limits; a timeout is failure, not permission
   to relax leases. Run24K as its own fresh profile, not an added unbounded
   matrix. Record full-wall latency/actual tokens for near-full C4; do not mix
   those timings with the148-input/256-output short-context throughput result.

All allocation, both-rank memory/zero-swap, request and final cleanup receipts
must pass before describing24K as qualified. Health alone does not prove GPU
release, numerical correctness, or successful cancellation reclamation.
