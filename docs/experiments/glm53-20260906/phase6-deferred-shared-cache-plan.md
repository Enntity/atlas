# Deferred target-shared cache allocation

Root approved this structural fix after the first cache load safely refused
the worker's pre-layer reserve: required 9,547,603,708 bytes, available
8,832,421,888. The unchanged non-cache arena+inference reserve is
8,490,639,100 bytes and fits that point. Inline target transformations later
free replaced checkpoint storage; this is not evidence that the safety floor
should be reduced.

## Allocation boundary and ownership

Remove only the inline cache install from GLM `components::load_moe`. Keep all
ordinary target/MTP loading, unified layout replacement and typed shared-weight
origin capture unchanged. After target layers, MTP weights/body, vision and
LM-head setup plus existing post-load transforms have completed, and before
`BufferArena::new`, run a GLM-only factory cache pass.

Use the actual target layer vector, not a process-global deferred pointer
registry. Existing `TransformerLayer::as_any_mut` supports the two concrete
GLM layer types. Add narrow accessors that verify the global layer ordinal
(`Glm5KdaLayer.layer_idx`, `Qwen3AttentionLayer.block_idx`, not attention index)
and return the main FFN. Validate all 45 concrete types/ordinals and dense
0..2 versus MoE 3..44 in a first pass before allocating any cache. MTP layer45
is outside this vector and is never revisited. A second ordered pass calls
the existing atomic per-layer cache installer with its typed stored origins.
Do not reread a freed BF16 source allocation; the provenance owner handles
native checkpoint versus quantized derived weights.

## Reserve invariant

Let A be the exact configured BufferSizes total, I the unchanged inference
reserve, C the remaining unallocated cache bytes, and F actual free memory.

- Before target loading: retain `F >= A + I`, with no cache allocation/debit.
- Before the deferred pass AND before each layer: require
  `F >= C + max(A + I, 4 GiB)` using checked arithmetic. C includes the next
  layer's 24 MiB. Both the full remaining cache and future arena/inference
  obligations are protected; the 4 GiB physical floor is not lowered.
- After the pass: retain the existing `F >= A + I` pre-arena check.
- After arena allocation: retain normal actual-free/used-so-far KV sizing.
  Cache bytes are already resident; do not subtract them a second time.

Use a typed `SharedFp8Reserve` passed from factory to each installer, rather
than recomputing a floor-only budget inside the layer. No cache can be
installed from the old inline path. External memory pressure may still refuse
the pass safely; this change makes no promise that a future load will fit.

## Ownership and TDD

Index owner: factory/build.rs, new factory/glm_shared_cache.rs plus tests and
module declaration, narrow concrete-layer accessors, cache.rs reserve type
and its tests, removal of only the inline components cache call.
Prefill owner: typed shared origin capture and validation, cache_load.rs
threading the reserve argument and existing transaction tests. Root owns
nodes/builds/rollout. Upstream independently reviews frozen integration.

Tests first: the reported pre-layer values pass A+I but fail a premature C
debit; a later sufficient free value admits the deferred pass. Exact-boundary
and one-byte-short remaining+protected checks, overflow, floor dominance,
and A+I dominance. First-pass layer-vector validation must reject missing,
duplicated/reordered, wrong-type and wrong-FFN entries before install callbacks;
zero callbacks/allocations for off/non-target paths. Test all 42 cache ordinals
and no MTP visit. Preserve per-layer atomic cleanup and no double KV debit.

After CPU GREEN and independent review, root repeats native source checks and
both-rank cache load/oracles. Record free memory before pass, each layer and
after arena; preserve watchdog thresholds and fail closed on any reserve or
origin mismatch. No default feature changes or kernel arithmetic changes.

## Source/CPU receipts

The typed-reserve RED compiled and failed because its old floor-only stub
incorrectly admitted the reported early cache allocation. The actual target-
ordinal planner RED passed its valid45-row control and failed the malformed
layer rejection test. Receipts: `shared-fp8-deferred-reserve-red.log` and
`shared-fp8-deferred-plan-red.log` under the phase6 receipt directory.

Final combined full model CPU suite passed 817 tests, including derived-source
provenance and cache transactions, after the root-requested formatted load-only
free/remaining/protected/required log addition. Independent source
review found no blocker in the deferred boundary, global layer identity or
reserve flow. CPU tests exercise the actual pure ordinal planner and disabled
factory entry, not a mocked complete45-concrete-layer factory integration;
the live deferred pass still requires root's native model gate. The exact
final suite receipt is `shared-fp8-deferred-full-model-green.log` in the same
phase6 directory.
