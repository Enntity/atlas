# Opt-in eager C4 semantic indexing to 16K

Status: planned before implementation; policy/eager dispatch implemented;
CPU checks passed; independent source review completed; full-model validation pending.

## vLLM reference and invariant mapping

The implementation reference is vLLM's actual
[GLM5Next NVIDIA attention/indexer at campaign pin 6865e67](https://github.com/vllm-project/vllm/blob/6865e67/vllm/models/glm5next/nvidia/attention.py),
specifically `Glm5NextIndexerCache`, `Glm5NextTailCache`, and `Indexer`.
Its cache metadata counts pooled states separately from raw tokens, and its
per-request tail retains raw keys/gates across prefill/decode boundaries.
Atlas already represents these through `SparseIndexCacheConfig` and physical
block ownership; this extension reuses that machinery for a fourth independent
row, rather than introducing another cache representation. It preserves
maintenance below the sparse-selection threshold and per-row block tables.

Upstream issues [55221](https://github.com/vllm-project/vllm/issues/55221)
(token/pool workspace accounting and FP8 storage dtype) and
[54359](https://github.com/vllm-project/vllm/issues/54359)
(backend/kernel storage-page disagreement) are correctness cautions, not proof
of Atlas defects. Our score arena is pool-count sized; storage stays BF16 and
the existing 16-token block/four-token pool contract is unchanged. Do not copy
vLLM/DeepGEMM's backend-specific page size into Atlas's different CUDA kernels.
The new flag is rollout gating for the existing general row loop; policy is
centralized in `model/glm_c4.rs`, not a new parallel inference implementation.

## Contract and implementation sequence

1. Add failing CPU launcher and model policy cases for a new default-off
   `GLM_C4_SPARSE=1` / `ATLAS_GLM_C4_SPARSE=1` mode. It requires C4 decode,
   base multi-sequence sparse indexing, eager execution, independent non-speculative
   TP2/EP2, BF16 KV/index, FP32 KDA state, active/admitted 4 and context 1..16384.
   Reject missing prerequisites, invalid flags, graph opt-in, excess context,
   and missing/excess host positions. Existing C1/C3/short-C4 defaults stay intact.
   Configured prefill budgets are restricted to 1..1024 in this mode only.
2. Update shared C4 launch/runtime policy and host position guards. C3/C2 drain
   calls while the C4 mode is active must retain the same bounded position check.
   Admit C4 in the eager sparse validator only with this explicit mode; leave
   `GlmDynamicShape`'s C2/C3 graph contract unchanged.
3. Update direct-server preflight and launcher together. Forward the new flag
   to both ranks. Expose `KV_OVERCOMMIT` with its existing default for other
   modes, but require it explicitly disabled for long C4. No CUDA kernels,
   persistent scratch, recurrent slots, speculative policy or admission-width
   expansion is part of this change.
4. Run targeted/full CPU model policy tests, direct-server tests, launcher-prefix
   tests, formatting and whitespace checks. Obtain independent source review,
   then freeze for root-owned builds and serialized hardware tests.

## Memory and operational preconditions

Per node, one 16-token physical block across 11 MLA layers needs:
`11 * (16*512*2*2 + 4*128*2 + 16*128*2*2) = 461824 bytes` (451 KiB).
This includes BF16 latent K/V, pooled semantic keys, and key/gate tail storage.
The compressed cache is replicated across TP ranks: do not divide by TP2.

Four 2K sequences consume 225.5 MiB live block capacity; four 16K sequences need
1804 MiB, an increase of 1578.5 MiB (1.5415 GiB) per node. Require at least 4097
physical blocks including the permanent dummy; operational target 4101 provides
one spare per sequence. The pool is budget-allocated, not automatically grown
by this live-capacity delta: inspect each rank's actual pool size. Disabling
overcommit restores the existing startup capacity refusal, but its generic
division omits the dummy, so root must also check the explicit block minimum.

SSM slots remain four active plus one dummy; state/snapshot count does not grow
with context. Query/attention scratch remains four rows; selectors reuse one
score/ID arena. At 16K the eager score array needs 16384 B and IDs 8204 B, already
within the validated C4 arenas. Context metadata grows by a small amount.
Keep `MAX_PREFILL_TOKENS=1024` explicitly; the launcher's generic default would
otherwise grow to 6144 and increase transient/staging memory.
The new mode refuses that oversized default. Runtime alignment can add small
token padding beyond the configured budget; the existing allocation planner
accounts for this, and this change does not alter its alignment rules.

Keep `OOM_GUARD_MB=4096`, existing GPU utilization and container ceiling. This
flag protects weight loading; the nonspeculative inference reserve's CUDA
headroom is 512 MiB, not a 4 GiB post-load guarantee. Root must measure at least 4 GiB
of host `MemAvailable` on **both** unified-memory nodes after load/warmup and at
campaign checkpoints. Record this as available host memory, not CUDA allocator
free bytes or the kernel's smaller `MemFree` field. Do not
raise utilization, remove guards, enable swap, or admit extra requests to fit.

## Hardware gates and rollback

Root only: validate short sparse C4 first, then mixed 2047/2048/2049 threshold
rows and unequal 8K/12K/15K independent needles, then four near-16K requests.
Use staggered completions and permuted slots for C4→C3→C2→C1 drain, followed
by fresh slot reuse. Count prompt+completion against 16384. Test request-boundary
rejection beyond the cap, finite outputs, block availability and both-rank free
memory. Compare aggregate decode with matched eager long C3, not short graphs.
Sparse index maintenance must run for every decode row below 2048 as well.

Graph support is explicitly deferred. Roll back by disabling the new flag and
returning to the validated short-C4 context/sparse settings, or to long C3.

## CPU receipt (before source freeze)

New launcher/model policy tests first failed against short-only behavior.
After implementation: all 687 CPU model tests, eight GLM server tests, three
new sparse launcher tests and three existing M4 launcher tests passed.
`bash -n` and `git diff --check` passed. Independent review found no state,
index scratch, metadata-offset or drain-cap blocker. GPU validation is pending;
these checks do not establish full-model correctness or speed.
