# Standalone Atlas-native KDA value32 experiment

Scope: only this plan, `glm_kda_value_tiled.cuh`, and opt-in wiring in
`bench_glm_kda_batch.cu`. No production changes, serving flags, commits, node
operations, or GPU execution by the implementing agent. Root runs GPU gates
only during an orderly stopped-model window. No speedup is assumed.

## Hypothesis and exact contract

The already-deployed indexed recurrence batches independent rows. This experiment
adds four independently scheduled value tiles per head, not another row-batching
change. At local H32/D128, launch grid `(32,N,4)`, block `(32,1,1)`, dynamic
shared memory zero. Each lane owns one contiguous value column in a 32-column
tile; FP32 state remains `[slot,head,key,value]`, value contiguous. Explicit
device slot IDs and FP32 element stride retain the indexed ABI and ownership.

Each tile repeats the original Q/K/gate setup over all128 keys, cooperatively
loaded in four lane passes. The shared-memory norm reduction retains exactly
the original strides64,32,16,8,4,2,1 and additions, with two lane passes at
stride64. Normalization uses runtime `dim`, original rsqrtf, epsilon, gate/beta
formulas and four-key accumulation expressions. Do not prove dim128 in CUDA:
that previously changed output rounding. Host fixture requires H32/D128.
Compile with production `--fmad=false`, without fast-math; optionally check
default FMA separately, never mix reference/candidate compilation semantics.

Invalid row/slot/stride/block checks return uniformly before any shared-memory
barrier or paged state read. Distinct live slots are host-validated. Tiles write
disjoint columns, so no cross-CTA barrier or temporary state is required.
Repeated normalization costs and changed scheduling/register allocation may
outweigh any occupancy gain; timing must decide.

The rationale is pinned vLLM's finer value decomposition, not copied code:
[GLM BV8 launch](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py#L49).
Its state layout and reductions differ; this candidate derives only from Atlas.

## Test-first gate and memory

1. Add harness opt-in `--value-tile` first, selecting the new recurrent export
   while keeping production indexed convolution. Default behavior is unchanged.
2. The existing91-case suite compares the candidate directly against the frozen
   independent scalar recurrence and frozen serial/unchanged TP convolution.
   It checks every FP32 H/conv element, BF16 output, all outer/slot canaries,
   inactive and invalid rows, exact-zero Q/K, stronger magnitudes, permutations,
   nonprefix slots, drain/reuse, and changing indices under captured graphs.
   No tolerance or validator changes. Do not compare candidate against itself.
3. Reuse existing buffers: 38,894,224 explicit device-allocation bytes, below
   64MiB. Only normal CUDA context/graph/event overhead is outside this explicit
   fixture accounting. No third state pool, weights, scratch, or allocation in
   a kernel or capture. Baseline and tiled timing reuse the candidate pool.
4. `--timing --value-tile` reports three distinct conv+recurrent paths: frozen
   per-row scalar/TP, current production indexed, and tiled indexed. Five trials
   of100 steps rotate variant order, reset all state outside event intervals,
   and report the indexed/tiled ratio to isolate gains beyond row batching.
   Default `--timing` retains the original two-path experiment.
5. Root compiles, runs both correctness modes and compute-sanitizer on the
   opt-in mode before timing. Any bit mismatch, runtime error or guard failure
   blocks promotion. Capture reuse must pass even if eager passes. No production
   promotion is authorized by this plan or by microbenchmark success alone.

## Commands for root's controlled GPU window

```sh
nvcc -O3 --fmad=false -arch=sm_121a scripts/dev/bench_glm_kda_batch.cu -o /tmp/bench-glm-kda-value32
/tmp/bench-glm-kda-value32
/tmp/bench-glm-kda-value32 --value-tile
compute-sanitizer --tool memcheck --error-exitcode=99 /tmp/bench-glm-kda-value32 --value-tile
/tmp/bench-glm-kda-value32 --timing --value-tile
```

Status: source frozen for root compilation. The plan and91-case harness wiring
were written before the candidate body. No CUDA red/green run was possible:
local `nvcc` is unavailable. Host-only structural checks passed256 FP32 norm
reduction trees and proved each of128 value columns has exactly one tile/lane
owner; `git diff --check` passed. These do not establish CUDA numerical parity.
Root owns compilation and the outstanding CUDA/GPU/memcheck/timing gates.

## GPU outcome — 2026-09-07, not promoted

Root subsequently compiled the frozen sources with production `--fmad=false`
for `sm_121a` and ran the standalone gates during a stopped-model window.
Both the existing indexed control and value32 passed all91 cases against the
frozen independent scalar oracle, including complete FP32 H/conv state, BF16
outputs, inactive slots and guards. Value32 passed the same91 cases under
compute-sanitizer memcheck with zero errors. Explicit device allocations were
38,894,224 bytes. Ptxas reported64 registers, zero stack/spill bytes and2572
bytes static shared memory for `glm_kda_recurrent_value32`.

All three paired timing runs are retained below. Times are microseconds for
convolution plus recurrence using eager CUDA-event submission, median of five
100-step intervals with rotated variant order and resets excluded. The ratio
is current indexed/value32: greater than1 favors value32. Compare variants
within the same run; the scalar column is not evidence of a new tiling gain.

| Run | Rows | Frozen per-row | Current indexed | Value32 | Indexed/value32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial | 2 | 45.432 | 20.518 | 20.671 | 0.993 |
| Initial | 3 | 68.236 | 26.733 | 26.938 | 0.992 |
| Initial | 4 | 92.547 | 48.784 | 43.292 | 1.127 |
| Repeat1 | 2 | 45.461 | 20.630 | 20.733 | 0.995 |
| Repeat1 | 3 | 68.258 | 25.024 | 26.293 | 0.952 |
| Repeat1 | 4 | 94.543 | 41.490 | 42.524 | 0.976 |
| Repeat2 | 2 | 45.485 | 20.692 | 20.713 | 0.999 |
| Repeat2 | 3 | 74.441 | 27.501 | 28.730 | 0.957 |
| Repeat2 | 4 | 92.198 | 41.223 | 41.275 | 0.999 |

Decision: **do not promote**. The initial N4 ratio1.127 disappeared in both
repeats; N2 showed no gain and N3 regressed in both repeats. Correctness passed,
but these measurements do not establish a repeatable performance advantage
over deployed indexed batching. No production kernel, serving flag or model
binary incorporates this candidate. Root restores the same v11 baseline.
These are bounded microkernel measurements, not full-model throughput results.

Receipts inspected locally (root-owned GPU execution):

- `/tmp/atlas-glm53-phase5-20260907.gu145h/kda-value32-gpu.log`
- `/tmp/atlas-glm53-phase5-20260907.gu145h/kda-value32-timing-repeat.log`
- `/tmp/atlas-glm53-phase5-20260907.gu145h/kda-value32-build.log`

Final status: experiment documented and frozen; production promotion rejected
for lack of repeatable measured benefit. No further kernel edits or GPU runs
are part of this experiment.
