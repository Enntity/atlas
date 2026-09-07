# Standalone K5 shared-expert M16/N128 prototype

This is a CPU-first, standalone CUDA experiment. No production dispatch, weights,
quantization policy, allocation, or rollout flag changes are authorized.

## Observed path and historical caution

With `GLM_K5_BATCHED_SHARED=0`, `run_shared_expert_prefill` already processes all
five rows in three `w4a16_gemm_t` calls plus one SiLU call. The fused-shared flag
is nested inside the disabled exact-M5 branch and therefore does nothing here.
GLM's shared checkpoint weights are native NVFP4 in transposed `[K/2,N]` form.
The active GEMM converts BF16 activations and dequantized weights to E4M3, then
uses K32 FP8 MMA with FP32 accumulators and BF16 stores. It is not W4A4 activation
quantization, nor a BF16-MMA kernel.

The M64 tile has four M16 warps along M. For M5, three warps compute only padding.
Change only ownership: four warps each compute M16/N32, sharing the same N128 B
tile. Preserve every output's conversion, K32 instruction sequence and store.
Use the actual original kernel included from
`kernels/gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu` as the primary oracle.

Existing exact-M5 GEMV is a different arithmetic path, not this oracle.
`docs/glm53-dual-spark.md` records an older +4.9% endpoint median and a separate
fusion target-step improvement of only 0.29%; those receipts predate the current
profile and cannot predict this prototype or justify a current flag sweep.
The rejected compact routed-down work is unrelated and remains excluded.

## Bounded design

- New `glm_shared_m16.cuh`: exact M0..16, K multiple32 up to4096, N positive,
  padded LDB multiple128 and at least N, up to4096; block128, grid ceil(N/128)
  by1. No new B layout.
- Two A buffers shrink from64x40 to16x40 BF16. Only threads0..63 load A;
  all128 retain the original B/scales loader and per-column E4M3 conversion.
  All threads participate in both existing stage barriers; FP8 B is overwritten
  only after every warp has finished consuming the previous stage.
- Each warp retains four N8 accumulator fragments instead of sixteen.
  No reduction, activation fusion, FP4-A conversion, new weight copy, or new
  shared expert down arithmetic. GU and down are separate projection tests.
- New harness runs full production GU N2048/K4096 and down N4096/K2048,
  two distinct GU matrices, and a padded-LDB N129/K64 tail case. M0/1/4/5/15/16,
  an exact-zero source row, nonzero BF16 rounding cases, refreshed inputs/scales
  and row permutations are exercised with fixed-pointer graph replay.
- Each shape runs sequentially, not concurrently. Device allocations include
  individually guarded A, two B/scales matrices, and four full outputs, under
  a checked32MiB explicit allocation cap (CUDA context overhead excluded).

## Tests before any measurement

1. CPU RED/GREEN: production ownership helper starts with old warp-M mapping;
   prove exact once-only ownership of every M0..16/N128 cell, A copy coverage,
   LDB/tail addresses, shape rejection and overflow-safe allocation accounting.
2. Native root-only: full original-vs-candidate BF16 bit equality for all live
   outputs, untouched output tails and allocation guards; independent CPU double
   dot products at boundary columns after explicit E4M3 rounding. Strict original
   equality is the acceptance criterion; CPU columns allow at most one BF16 ULP
   for different reference accumulation order.
3. Graph capture uses the same stable pointers; second pass mutates activations
   and scales before replay. Check all input/weight/scale bytes after each case.
4. Optional interleaved CUDA-event timing only after all correctness cases pass;
   recheck every output, input and canary after timing. This is a small resident
   weight microbenchmark, not end-to-end TPS evidence or a production promotion.
5. Both default compiler FMA and `--fmad=false`, plus root-owned memcheck. Review
   source independently before any native run. No GPU operations by the author.

## Commands (root owns native execution)

```bash
g++ -std=c++17 -O2 -DATLAS_SHARED_M16_HOST_ONLY -x c++ scripts/dev/bench_glm_shared_m16.cu -o /tmp/bench-glm-shared-m16-host
/tmp/bench-glm-shared-m16-host --host-test
nvcc -std=c++17 -O3 -arch=sm_121 -Xptxas=-v scripts/dev/bench_glm_shared_m16.cu -o /tmp/bench-glm-shared-m16
/tmp/bench-glm-shared-m16
compute-sanitizer --tool memcheck --error-exitcode=9 /tmp/bench-glm-shared-m16
/tmp/bench-glm-shared-m16 --timing
```

## Local CPU evidence

The initial old warp-M column mapping compiled and failed the intended assertion:
`FAIL: four warps partition N exactly once` (exit2). The corrected shared
host/device mapping passes exhaustive M0..16 output ownership, A/B/scales copy
coverage, padded-LDB bounds, unsupported shape/overflow rejection, independent
E4M3 ties/saturation/subnormal rounding and strict one-BF16-ULP comparison tests.

Raw receipts: phase6 `shared-m16-host-red.log` and `shared-m16-host-green.log`.
The same host-only tests also pass local AddressSanitizer/UndefinedBehaviorSanitizer
(`shared-m16-host-sanitize.log`); this does not inspect CUDA or device accesses.
Explicit guarded device totals are9,832,704 bytes for GU,10,029,312 for down,
39,296 for the LDB tail fixture; each shape releases its allocations before the
next. The peak is therefore10,029,312 bytes, below32MiB. CUDA context, events and
one four-node graph at a time are additional driver bookkeeping, not matrices.

Source comparison confirms the original B dequantization body, full K-stage
pipeline and MMA instruction text are unchanged. Author has not compiled CUDA
or run any GPU test; native validation and independent review remain required.
