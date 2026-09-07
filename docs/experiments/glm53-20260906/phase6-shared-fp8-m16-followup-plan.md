# Optional target-shared FP8 M16 follow-up

Status: planning only. No implementation, native build or deployment is
authorized by this document. First measure the frozen target-cache M64 model.
This is a tile-only follow-up, not another weight/cache layout or conversion.

## Evidence and decision boundary

The existing standalone `scripts/dev/glm_shared_fp8.cuh` candidate preserves
BF16 activation-to-E4M3 conversion, native row-major FP8 weights `[N,K]`, K32
MMA order and BF16 output. It reduces A staging to 16 rows and distributes four
warps along N, rather than along M. Root reports its native full-output,
graph-refresh and both-FMA memcheck gates passed. The large-prefill profile
intentionally excludes this candidate and validates only existing M64.

All three M5 paired timings from
`/tmp/atlas-glm53-phase6-20260907.J5PkkO/shared-fp8-native-timing-pairs.log`
are retained below. Each is an eager CUDA-event median of five interleaved
100-iteration samples; hot single-projection weights, predecode excluded.

| Projection | Run 1 M64 → M16 (µs) | Run 2 | Run 3 |
|---|---:|---:|---:|
| Gate | 49.167 → 43.251 | 49.203 → 40.985 | 49.159 → 40.990 |
| Up | 49.179 → 40.986 | 49.154 → 40.971 | 50.056 → 40.987 |
| Down | 26.572 → 22.541 | 26.593 → 22.538 | 26.600 → 22.555 |

All nine comparisons favor M16. Summing the three separate projection medians
within each run gives 14.5%, 16.4%, and 16.9% less kernel time than FP8 M64.
That arithmetic is **not** a timed shared chain or full-model estimate. The
cache already removes most of the original W4A16 conversion cost; this is a
smaller remaining opportunity, not another approximately 3× cache gain.

Proceed only if the actual cache-M64 model's K5 profile shows shared projection
time remains material and the model is correct/stable. These kernels do not
reduce the cache's approximately 1008 MiB resident bytes or full-model memory
traffic. Streaming 42 different layers may erase hot-fixture gains. Measure
the shared fraction of the target step first: even eliminating that entire
path cannot save more than that measured fraction. No TPS forecast is justified.

## Minimal production surface, if later approved

- Promote the validated helper/kernel to a new private
  `kernels/gb10/deepseek-v4-flash/nvfp4/glm_shared_fp8_m16.cuh`, included at the
  end of the existing `w4a16_gemm.cu`. Keep all existing primitives and exports
  unchanged. Module stays `w4a16`; symbol is `glm_shared_fp8_m16`.
- Adapt the small standalone harness to obtain the actual production export
  through that TU, retaining the original W4A16 and FP8 M64 numerical oracles.
  Leave the separate large-prefill fixture M64-only.
- Add a small MoE child module `shared_fp8_cache_m16.rs` plus tests. Hold its
  model-owned handle, strict flags and independent verification bits within
  `SharedFp8CacheState`; resolve the handle before load-time cache allocations,
  only when explicitly requested on the validated target-cache profile.
  No process-global pointer/handle cache and no new GPU allocation.
- Add a narrow selection hook at `shared_fp8_cache_output.rs`'s exact cached
  launch boundary. Preserve the existing `state.layer.is_none()` generic path,
  the old M64 launch and the base cache byte/ownership/cleanup logic.
- Use a direct checked launch wrapper, **not** `ops::fp8_gemm_n128`: that generic
  wrapper can substitute default-on LDMAB and allocate activation scratch.
  Exact ABI is `(A, B_fp8, C, M, N, K)`, grid `[N/128,1,1]`, block `[128,1,1]`,
  dynamic shared 0. Gate/up N2048/K4096; down N4096/K2048. It is the same grid
  as M64 at M5, but must bind the new handle explicitly.

## Exact selection and separate controls

Proposed strict default-off controls:

```text
ATLAS_GLM_TARGET_SHARED_FP8_M16
ATLAS_GLM_TARGET_SHARED_FP8_M16_VERIFY
```

Main requires `ATLAS_GLM_TARGET_SHARED_FP8=1`; VERIFY requires its M16 main.
Root owns startup validation, launcher propagation and actual graph guards.
Do not couple these to the separate native-NVFP4 shared-M16 or router flags.

Select only the validated target GLM TP2/EP2 cache state with a target layer
ordinal 3..44, rows exactly 5, and exact per-projection dimensions. Match the
weight pointer to that projection's published cache and the output to its
existing owner/capacity. Reject invalid aligned spans/aliases after selection.
Preserve cache preflight exclusions: LoRA, MTP/TP1, alternate shared precision,
and the earlier exact-M5 GEMV branch. No scalar/C2/C3/C4 or wider-prefill
substitution: every other admitted row count 1..1024 stays on current M64.
An exact five-token prefill tail is indistinguishable from temporal M5 at this
projection boundary: `ForwardContext` has no authoritative verify-phase tag.
The minimal shape-based selector therefore also covers that tail, with the
same proven arithmetic. If "all general prefill stays M64" includes M=5 tails,
require an explicit verifier-owned call discriminator before implementation;
do not infer it from capture state, token IDs, or sequence metadata.

## Oracle and acceptance gates

M16 VERIFY is independent of the base cache VERIFY. At each natural Gate/Up/
Down boundary, once per target layer at M5, compare original FP8 M64 against
M16 in the existing output: poison → M64 → full host snapshot → fresh poison
→ M16 → full snapshot. Reject nonfinite/unwritten reference and every bit
difference; mark only after success. Down uses the same actual post-SiLU input
for both launches. Maximum two host output snapshots is 81920 bytes; no device
scratch, conversion or duplicate weight. Leave M16 output for the normal chain.

The existing cache VERIFY may independently compare original T W4A16 with the
finally selected cache projection. If both diagnostics are enabled, require
both independent comparisons/traces; do not silently reuse one success bit
for the other. Equality then covers T W4A16, FP8 M64 and FP8 M16 without a
second device output. Preserve the base load-time every-byte cache oracle.

Before any graph lookup/warmup/capture, the root-owned actual graph-decision
guards must reject M5 use with M16 VERIFY. Also check actual projection-stream
capture and auxiliary overlap before all diagnostic I/O, including after prior
success. Main-only execution remains graph-capable and allocation-free.

Tests first: positive exact M5 selection, all other widths and target/resource
conflicts retaining exact M64; strict flag dependencies; projection owner,
alignment/capacity/overflow; actual direct launch ABI/grid; independent once-
only bits, missing/partial output, last-byte corruption, nonfinite reference,
and capture refusal even after prior PASS. Reuse bounded numerical transactions
where coherent; do not modify the unrelated native-M16 feature's behavior.

After CPU review, root reruns the standalone against the assembled production
TU, then model oracle coverage for all 42 layers/126 projections on both ranks.
Require unchanged cold-prefill behavior, graph-on quality, acceptance/output
accounting, memory health and independent paired M64/M16 model A/B with all
diagnostics off. No default promotion without a reproducible model benefit.
