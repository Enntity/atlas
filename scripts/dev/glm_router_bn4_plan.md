# Exact K5 router: standalone BN4 experiment

Status: tests-first prototype only; no production dispatch, compilation on the
DGX nodes, GPU execution, or promotion is authorized by this source change.

CPU TDD receipt: the initial `owns_output` stub caused the actual harness to
exit2 with `FAIL: each router output has exactly one owner`. Implementing the
shared host/device mapping produced GREEN for exhaustive output ownership,
tile-load coverage, increasing-K traversal, exact-shape rejection and checked
allocation boundaries. This proves indexing contracts, not CUDA arithmetic.
The explicit device allocation sum is2,407,040 bytes (A, B, two outputs and
all guards). CUDA driver/event/graph bookkeeping is not included in that sum;
the harness creates one tiny two-kernel graph and two timing events at a time.

The existing M5 router is 18 CTAs of80threads. This experiment uses72
single-warp CTAs:20 lanes own five rows by four columns, all32 lanes load
shared tiles. It preserves the complete increasing4096-term FP32 accumulation
and final BF16 conversion. Compile with `--fmad=false`; neither split-K nor
tensor cores are permitted. BF16 inputs/weights retain checkpoint row-major
layout. This targets grid occupancy, not quantization or routing policy.

## TDD and numerical gates

1. CPU RED/GREEN: shared host/device mapping helpers must assign all1440
   outputs exactly once, cover every shared A/B element exactly once per
   Ktile, preserve each output's Korder, and reject bad allocation bounds.
2. GPU: include the unchanged production `dense_gemm_bf16_router_m5` as
   oracle; require all1440 BF16 logits bit-identical for signed, mixed-scale,
   cancellation/near-tie, and zero-weight profiles. Independently calculate
   selected columns using ordered FP32 operations and a double sanity oracle.
   Full identical logits preserve downstream routing inputs without changing
   the production top-k implementation. This is not a resident-weight test.
3. Poison both outputs before each variant, inspect allocation canaries and
   input immutability. Capture both kernels at fixed pointers, update BOTH
   input and weight contents, replay, and repeat all numerical checks.
4. Only with explicit `--timing`: seven alternating event-timing rounds,
   50 launches per variant; validate
   both outputs and immutable inputs again after timing. No timing from a
   correctness failure is accepted. All device allocations are metered below
   16MiB, including128-byte guards at either end of every allocation.

## Commands (root owns native/GPU execution)

CPU only:

```bash
g++ -std=c++17 -O2 -ffp-contract=off -DATLAS_ROUTER_HOST_ONLY -x c++ \
  scripts/dev/bench_glm_router_bn4.cu -o /tmp/atlas-router-bn4-host
/tmp/atlas-router-bn4-host
```

Root, during the established stopped-model GPU window:

```bash
nvcc -std=c++17 -O3 --fmad=false -arch=sm_121 -lineinfo \
  scripts/dev/bench_glm_router_bn4.cu -o /tmp/atlas-router-bn4
/tmp/atlas-router-bn4
compute-sanitizer --tool memcheck --error-exitcode=1 /tmp/atlas-router-bn4
# Only after both correctness gates pass, outside sanitizer:
/tmp/atlas-router-bn4 --timing
```

Default invocation never runs event timings. Unknown/extra arguments fail
before CUDA operations. CLI TDD confirmed the previous host harness accepted
`--unknown`; the corrected shared parser rejects it and distinguishes the
default correctness-only mode from explicit `--timing`.

Do not reinterpret synchronized full-model router time as kernel-only time.
The original90.1us microreceipt and current~166us synchronized phase were not
measured in identical conditions. Native bitwise and sanitizer gates must
precede any production proposal. The eager45-layer router phase is only~5.5%
of the measured verifier timeline; this is a bounded secondary optimization.
