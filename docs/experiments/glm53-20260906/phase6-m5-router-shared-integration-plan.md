# GLM-only M5 router BN4 and shared M16: guarded integration plan

Status: implementation approved by root; source/CPU review in progress.
Two independent default-off kernel substitutions, not a B-tile layout change.
Root owns all native builds, model/node operations, launcher edits and rollout.

## Scope and existing arithmetic

- Router: replace only the actual `dense_gemm_bf16_router_m5` arm reached by
  `MoeLayer::router_gate_gemm_dense`, with `glm_router_m5_bn4`. Exact BF16
  A[5,4096], B[288,4096], logits[5,288], sequential FP32 K accumulation under
  the production FMA policy. BN4 grid72/block32 replaces BN16 grid18/block16x5.
  No router quantization, cuBLAS reroute, top-k/bias changes or C3/C4 routing.
- Shared: replace only the actual transposed native-NVFP4 `w4a16_gemm_t`
  projections inside `run_shared_expert_prefill`, when M=5. Gate/up use
  N2048/K4096/LDB2048; down uses N4096/K2048/LDB4096. Preserve original
  BF16-A and dequantized-B to E4M3 conversions, K32 FP8 MMA accumulation,
  BF16 output, SiLU, post-EP blending and optional deferred mHC blend.
  M16/N128 changes warp ownership only. It does not replace the distinct
  exact-M5 native GEMV or C4 shared GEMV arithmetic.

Root reports standalone native numerical gates and memcheck passing, including
both FMA modes for shared. Three paired router speed ratios were
1.1706/1.1626/1.1736. Shared M5 gate/up and down median ratios were1.14306
and1.15464, respectively, with all six per-shape/width comparisons winning.
These are microbenchmarks, not model throughput claims. B-tile scalar fallback
work and the existing routed gate/up M16 feature stay separate and unchanged.

## Proposed controls and typed selection

Strict0/1 flags (missing means disabled), independently resolved at construction:

```text
ATLAS_GLM_M5_ROUTER_BN4
ATLAS_GLM_M5_ROUTER_BN4_VERIFY
ATLAS_GLM_M5_SHARED_M16
ATLAS_GLM_M5_SHARED_M16_VERIFY
```

Each VERIFY requires its corresponding main flag. Neither feature requires
the other. The main flag does not force an oracle or an alternate arithmetic
path. An ineligible call retains its exact original launch. Invalid candidate
buffers/handles after positive selection are errors, never unchecked launches.

Add one MoE-local `M5Projections` field containing `RouterBn4` and `SharedM16`
policy/handles/verification state. Use typed `SharedProjection::{Gate,Up,Down}`
and a pure checked launch plan, not shape-derived guessing of which output
arena is being written. Every new Rust module stays below500 lines; split
tests and diagnostic transaction code from the selectors. Do not refactor or
reuse private implementation from `gate_up_m16.rs`: preserve its behavior.

Eligibility for both requires target GLM geometry H4096/inter2048/shared2048,
experts288/top8/sigmoid, TP=EP2 with comm present, rows exactly5, and no LoRA.
The appended MTP proposer has TP/EP1 and remains ineligible with no new handle
lookup or launch. Non-GLM, widths0/1/2/3/4/6+, prefill tails, and dense FFNs
retain existing behavior.

Router additionally requires: original M5 router arm truly selected (including
`ATLAS_GLM_K5_ROUTER_M5` and its existing handle), plain BF16 router weight,
no `gate_fp8`, `gate_nvfp4`, router pre-norm or hash route, and nonnull correction
bias. Place the hook **inside that original arm**, not before C3/C4 or FP8
dispatch. If the old M5 flag is disabled, leave the old generic router fallback
alone; the new flag must not silently turn it back on.

Shared additionally requires: all three transposed shared weights present with
native NVFP4 group16/E4M3 format, native shared precision tag, no BF16-shared
override, no cached `shared_gate_fp8/shared_up_fp8/shared_down_fp8`, no LoRA,
and exact-M5 shared GEMV not selected. Place projection hooks **inside existing
transposed GEMM arms**, after earlier C3/C4, BF16, exact-M5 and FP8 branches.
Do not invalidate/rebuild any FP8 cache. Reject/select-original when those
caches or different precision paths exist. Original same-stream sequencing
and existing overlap events remain unchanged for ordinary execution.

## Buffers and launch contracts

No GPU allocation, weight conversion, persistent scratch or additional pointer
table in either ordinary fast path. Existing buffers only:

| Projection | Input | Existing output | Active BF16 output bytes |
|---|---|---|---:|
| Router | actual router input,5x4096 | `gate_logits` | 2880 |
| Shared gate | original5x4096 input | `ssm_deinterleaved` | 20480 |
| Shared up | same original input | `ssm_qkvz` | 20480 |
| Shared down | post-SiLU shared gate,5x2048 | `attn_output` | 40960 |

Typed boundary validates exact geometry, checked byte multiplication/address
addition, owner output pointer and capacity, input/output disjoint spans and
alignment. Router vector input/weight pointers require8-byte alignment; shared
cp.async input/packed/scales pointers require16-byte alignment. BF16 outputs
require2-byte alignment. Shared B packed/scales extents are N*K/2 and N*K/16
with explicit LDB=N, finite tensor scale2, no unsupported per-row scale2.
Actual weight allocation dimensions remain loader-proven; the launch contract
must not claim it queried an unavailable allocation-size API.

Original router grid and candidate grid differ: add a checked BN4 ops wrapper,
do not pass the candidate handle through the old BN16 launch wrapper. Shared
M=5 original and candidate use the same ABI/grid ceil(N/128),1,1/block128;
the existing `w4a16_gemm_n128` wrapper may be reused after typed validation.

## Independent eager numerical oracles

Each feature has separate verification bookkeeping, model-owned per MoE layer.
Router records the M5 path only after all1440 BF16 logits compare bit-for-bit.
Shared records Gate/Up/Down separately, all at width5, only after each complete
active output passes. Dense positive-width projections always have useful
work; no local-expert worklist shortcut or empty-rank success applies here.

At the selected projection's natural boundary:

1. Validate actual noncapture state and all spans before any D2H, sync or write.
2. Poison only that output, launch the exact old production kernel, snapshot
   the full active BF16 output on the same stream; reject nonfinite/poison
   reference results so two omitted writes cannot compare equal.
3. Freshly poison that output again, launch candidate, snapshot and compare
   every bit. Leave candidate output for the unchanged downstream operation.
4. Mark that feature/projection successful only after complete comparison;
   errors never mark success or fall back. Diagnostic state is not shared
   between layers, projections or model lifetimes.

No extra GPU buffer is needed. Maximum simultaneous host output snapshots are
2*40960=81920 bytes for shared down; router uses5760 bytes. Existing input and
weights remain unchanged and live across both launches. Gate is verified before
SiLU; down is verified after the normal SiLU with the same post-SiLU input for
both launches. Never rerun an entire shared chain in-place after it consumed
the original gate values. No alias with still-live router/routed output areas.

Verify overlap is conservatively refused before reference/candidate writes; ordinary main
features retain the existing `aux` stream and event behavior. Both
`ctx.graph_capture` and `gpu.stream_is_capturing(actual_projection_stream)`
are checked defensively. Crucially these are not the primary graph guard:
new `validate_m5_projection_graphs(model_type,rows,use_graphs)` resolves the
independent flags and invokes its pure policy helper at
actual M5 graph decisions before graph lookup/warmup/capture. Root wires
`verify_d.rs` and any reachable M5 fused/alternate verifier entry found by the
graph audit. C1/C2/C3/C4 bootstrap graphs need not be disabled merely because
an M5-only oracle is enabled. No unrelated routed-M16 graph guard changes.

## Exact file ownership proposal

Index_tensorcore, after explicit approval:

- New `crates/spark-model/src/layers/moe/m5_projections.rs` (state, strict flags,
  shared policy commonality and pure graph guard), `_tests.rs`.
- New `.../moe/router_bn4.rs` and `_tests.rs`: actual M5-arm selector/launch.
- New `.../moe/shared_m16.rs` and `_tests.rs`: typed three-projection selector.
- New `.../moe/m5_projection_oracle.rs` and `_tests.rs`: bounded reusable
  single-output old/poison/new transaction, independent of routed M16 code.
- `.../moe/mod.rs`, `init.rs`: module declarations, one state field, eager
  feature-specific handle resolution only for applicable GLM target geometry.
- `.../moe/helpers_c.rs`: BN4 hook only in existing M5 router arm.
- `.../moe/forward_prefill_phase.rs`: three small projection hooks only in
  existing transposed shared GEMM arms.
- New `crates/spark-model/src/layers/ops/glm_router_bn4.rs` + ops module export;
  checked grid72/block32 exact ABI. Shared reuses its existing wrapper.
- New `kernels/gb10/deepseek-v4-flash/nvfp4/glm_router_bn4.cu`, exposing
  `glm_router_m5_bn4` in module `glm_router_bn4` (unlisted stems map directly).
  Preserve standalone arithmetic/helpers verbatim, no generic router edits.
- New `.../nvfp4/glm_shared_m16.cuh`, included at the end of existing
  `w4a16_gemm.cu` so all original primitives/LUTs have one definition. Expose
  `glm_shared_w4a16_m16` in existing `w4a16` module, no separate duplicate TU.
- Adapt the two standalone harnesses to include final production exports,
  with the old production kernels retained as numerical oracles. Coordinate
  source freeze with root before touching those already measured files.

Root: launcher forwarding/preflight as needed, M5 actual graph-decision hook
calls, native build/GPU tests and all full-model A/B runs. Upstream reviewer:
independent frozen source review. Prefill reviewer keeps independent FP8
standalone work separate; no precision-cache integration into this change.

## TDD before implementation and acceptance

1. Tests first against actual production pure selectors: valid M5 target must
   select BN4/M16; all other widths/models/topologies, no comm, LoRA, router
   FP8/NVFP4, missing original M5 flag/handle, shared FP8/BF16 cache/precision,
   exact-M5 GEMV, missing transposed weights or unsupported dimensions must
   preserve their original path. Observe RED before implementing selectors.
2. Dispatch tests verify exact modules, symbols, ABI bytes and dimensions.
   Router distinguishes grid18/block16x5 original from grid72/block32 BN4;
   shared validates GU vs down geometry/owner. Negative tests must fail for
   the intended resource/shape reason, not an unrelated invalid fixture field.
3. MockGpu-backed oracle tests drive the real transaction and callbacks:
   old/new success, no candidate write, incomplete write, wrong value,
   nonfinite/poison reference, wrong output owner, address overflow/alias,
   insufficient capacity, capture/overlap refusal before any I/O, callback
   failure and per-projection success bits. Assert no GPU allocations and no
   marking success after errors. Wrong values retain strict bit comparison.
4. Pure graph guard tests cover independent VERIFY flags, dependency errors,
   actual graph true/false, width5 versus bootstrap widths and non-GLM calls.
   Root must bind the helper before capture, not rely on a convenient env flag.
5. Focused model CPU tests then full model suite/check with coordinated target
   access. No new Rust file over500 lines, all source SPDX headers present.
6. Root native reruns standalone exact final production exports for router and
   shared, correctness/memcheck and production FMA flags. No claim that earlier
   script-local binaries validate changed compiled production code.
7. Root model gates independently: control; router-only eager VERIFY; shared-
   only eager VERIFY; each main feature graph-on quality; both combined only
   after independent attribution. Record selected/verified projection counts,
   full outputs/acceptance, memory and safety. Then paired timing with VERIFY
   off, same recipe/precision/graphs/context, no compilation overlap. Keep all
   repetitions and negative results. Neither feature becomes default-on here.

## Implementation receipts (native integration still unvalidated)

Root approved implementation after the standalone timing gates. Actual
selector RED compiled and failed the valid-M5 positive case (3pass/1fail).
Actual oracle RED failed poison and pre-I/O capture/overlap protection
(1pass/2fail). The additional flag/span contract RED failed because its APIs
did not exist. The final scoped GREEN passed10selector/oracle tests,
2router boundary/actual-grid tests, and1shared typed-boundary test. Normal
`cargo check -p spark-model --no-default-features` also passed. A wider MoE
run passed66tests and failed only the separate FP8-cache implementation's
2deliberate RED fixtures; it is not a combined-suite GREEN receipt.
Receipts live in `/tmp/atlas-glm53-phase6-20260907.J5PkkO/` under
`m5-projections-{selector-red,oracle-red,contract-red,green}.log`.
Final scoped receipts are `m5-projections-final-green.log`,
`m5-router-bn4-green.log`, `m5-shared-m16-green.log`, and
`m5-projections-check.log`.

The router fixture now includes the actual production `glm_router_bn4.cu`;
the shared fixture obtains its actual production export through
`w4a16_gemm.cu` (or the same production header for CPU-only ownership tests).
Both adapted g++ host fixtures passed. Earlier GPU receipts do not validate
these newly assembled production translation units: root must repeat native
correctness/memcheck and then the independent model oracles/A/B gates.

A modest kernel speedup is useful evidence, not proof that the30C1/60C4 goal
is met. Neither feature is default-enabled by this implementation.
