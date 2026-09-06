# GLM cache contract: geometry and accounting first

Status: implemented after failing CPU tests; all691 model CPU tests pass.
Source review found no blocker. Hardware validation remains a separate gate.

Introduce one pure GLM absorbed-cache shape and checked allocation plan in
`spark-model::model::glm_cache_plan`. The shape validates the already-required
latent rank512/zero-RoPE combination and supplies both the exact BF16 kernel
requirement and the cache's one-head/512-element geometry. The allocation plan
validates the existing runtime cache configuration and computes checked byte
counts, physical capacities and budget-derived block counts.

Use this contract at the main-cache construction in `factory/build.rs`, the
separate MTP cache in `layers/glm5_mtp.rs`, and the existing GLM-only attention
kernel selector. Other model branches are unchanged. Preserve current K/V dtype
and per-layer dtype choices, MTP's BF16 cache and optional index attachment,
the two separately owned K/V pools, full-block raw index tails, dummy-block
policy and existing allocation/attachment functions. No aliasing, new kernel,
rollout flag, context expansion or ownership change is included.

Tests first: exact512 binding/geometry; reject inconsistent rank/RoPE and
per-layer geometry; compare checked accounting with existing runtime formulas
for BF16 and FP8, including mixed layer dtypes; validate index pool/block units;
test zero/overflow capacities and budget boundaries. Exercise small real
`PagedKvCache` allocations through `MockGpuBackend` and assert distinct K/V and
index/tail pointers and unchanged strides. Full model CPU tests and formatting
must pass before source freeze.

Validation: four new contract tests cover the cases above, including the
`u32` physical block boundary, partial per-layer fallback, scaled-FP8 index
accounting and actual mock allocation byte totals. At block16, eleven BF16
MLA layers plus the existing index layout remain461824 bytes per physical
block. No model dtype is changed by the plan; main-cache FP8 and mixed-dtype
accounting still use the runtime's existing formulas. MTP retains its exact
`max_seq_len / 16 + 1` block-count rule and optional-index predicate.

Reference architecture, not copied implementation:
[vLLM's typed MLA cache specification](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/v1/kv_cache_interface.py).
Single-latent ownership and bounded request tails are explicitly deferred.
