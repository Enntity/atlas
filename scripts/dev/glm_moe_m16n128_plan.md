# M16/N128 native-FP4 expert tile, 128 loader threads

Plan before implementation. Standalone `scripts/dev` experiment only; no
production source, serving flags, commits, node operations or agent GPU runs.
Root owns independent review, controlled native build and stopped-model gates.

## Why this differs from the rejected experiments

`docs/glm53-dual-spark.md:575` records an M32 compact gate/up CTA with64
threads losing10.1% versus M64/block128. That experiment retained native FP4,
but reduced loader participation and duplicated B staging. It was not an
M8/N128/K32 experiment. A proposed one-warp M16 tile repeats the loader risk;
root therefore chose the following different hypothesis before coding.

Keep block128 and one N128 tile per CTA. Map the four warps to four disjoint
N32 slices, all sharing one M16 activation tile, instead of four M16 row
slices of an M64 tile. Keep B's coalesced loads, shared transpose, K64 double
buffering and all128 loader threads. Each warp runs four N8 MMAs per K64
step, rather than sixteen, and uses16 FP32 accumulator values per thread
rather than64. Output rows within each native M16 MMA keep the original
lane mapping and K accumulation order. This is an instruction/resource
hypothesis, not a throughput promise; all weight bytes must still be read.

Neither rejected value32 KDA nor compact-down enumeration is revived. Down
retains dense `(ceil(N/128),1,288)` scheduling; existing compact variants remain
independent controls. No builder cost is added to the M16 candidate.

## Exact prototype boundary

- Down: N4096/K2048, expert-sorted packed FP4 activation rows and FP8 E4M3
  per16 scales; transposed B `[K/2,N]` and `[K/16,N]` scales; original per-expert
  scale2 applied at BF16 writeback. FP32 MMA accumulation, production
  `--fmad=false`; no FP4 conversion, reduction or quantization-policy changes.
- C4 mode permits at most4 rows/expert and32 total rows. Compile-time K5 fixture
  alternative permits at most5 rows/expert and40 total rows, representing five
  verifier token rows, not five independent requests. Never route32/40 rows to
  one expert. Host validates before CUDA; no mixed prefill or unsupported shapes.
- Grid and block remain production dense-down geometry. Candidate ABI matches
  the old11-argument prequantized dense export; sorted IDs must be NULL fordown.
  CTA-uniform remote/empty/invalid-tile guards precede barriers. A's rows0..15
  are initialized; inactive rows use predicated zero-fill. Live output rows
  and buffer dimensions remain unchanged. No read/write aliasing introduced.
- B loads/transpose remain128-thread operations. A loads use first32 threads
  for16 rows; other threads still participate in all barriers and cp.async
  commit/wait steps. Scalar and vecscale loaders both have explicit candidates.

## Test-first gates and measurement

Extend the existing frozen-production down-cost harness with opt-in `--m16`;
first wire complete-output equality against production before implementing
the candidate. Before every candidate launch, poison its output independently.
Retain independent host worklist validation, dyadic FP4 CPU arithmetic checks,
complete dense/scalar/vecscale/compact comparisons, weight/input immutability,
all allocation guards and remote/unused output poison. Use actual production
source includes as baseline, not only another new helper as oracle.

Expand maps to cover1,2,3,4 rows/expert (plus5 in the K5 fixture), boundary
expert IDs143/144/287, partial-local and all-remote routes, all-empty maps and
weight-map permutation. Run each eagerly and under captured replay with changed
metadata at fixed pointers. All output elements are compared bitwise; independent
CPU decoding checks selected columns across all live rows. Those are different
coverage claims; no full-CPU-matrix claim is made.

Hard64MiB explicit GPU ceiling. Reuse the already existing output arena and
eight distinct weight fixtures; do not allocate a second set of weights. K5
adds only eight activation/output rows, keeping the fixture below40MiB.
CPU tests reject invalid row/weight/route limits before CUDA. Each timing mode
first executes the full eager+graph gate. Add M16 dense alongside old dense288,
dense8 and prior compact controls; rotate trial order, repeat paired runs, and
report eager timing without claiming whole-model gain. Root runs memcheck before
using timing to decide any next step. Stop on any bit/oracle/map/guard mismatch.

```sh
g++ -std=c++17 -O2 -x c++ -DATLAS_MOE_DOWN_HOST_ONLY scripts/dev/bench_glm_moe_down_cost.cu -o /tmp/moe-m16-host
/tmp/moe-m16-host --host-test
nvcc -O3 --fmad=false -gencode=arch=compute_121a,code=sm_121a scripts/dev/bench_glm_moe_down_cost.cu -o /tmp/moe-m16
/tmp/moe-m16 --m16
compute-sanitizer --tool memcheck --error-exitcode=99 /tmp/moe-m16 --m16
/tmp/moe-m16 --m16 --timing
# K5 fixture: add -DATLAS_MOE_TEST_ROWS=5 to both compiler invocations.
```

## Later gate/up applicability, not part of initial down promotion

Current fused compact gate/up selects projection pointers in grid.y then calls
the same M64 arithmetic helper. N2048/K4096 uses the same packed bytes per
expert as down. A later separately reviewed experiment can reuse the new tile
with unchanged compact worklist/projection multiplexing and original token-row
gather semantics. Gate/up and down fixtures must run sequentially so two weight
sets are never resident together. Validate fused gate AND up independently,
then actual SiLU/quantization/down/EP output before a full-model decision.
Down alone need not provide the campaign's required whole-model improvement;
root's phase profiling determines whether this follow-up is worthwhile.

## Source checkpoint

Candidate and harness are frozen for independent source review. CPU-only g++
tests passed in both C4 and K5 compile-time configurations; checks include exact
one-owner coverage of every M16/N128 output coordinate. Explicit device bytes
are38,590,472 (C4) and38,798,344 (K5), with no added candidate output allocation.
Every candidate output is freshly poisoned before its vecscale/scalar launch.
The candidate derives mechanically from the current production helper; only M
storage, warp N partition, loader participation for A, and accumulator extent
change. B loading/staging, native MMA operands, K64 traversal and scale2
writeback remain the baseline arithmetic. Later gate/up needs separate shape
and gather eligibility; this initial kernel explicitly accepts down only.
Local nvcc is unavailable; CUDA compile, bitwise numerical, graph and memcheck
gates have not been run by the author. No production promotion is implied.
