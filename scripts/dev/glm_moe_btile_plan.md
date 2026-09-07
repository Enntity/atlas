# Direct-staged B tiles: standalone test-first plan

This is a separate experiment, not a production promotion. Only new files
under scripts/dev are edited. Production M16 (commit502cda12), current harness,
dispatch, loader and persistent weights remain unchanged. Root owns all native
CUDA compilation, containers, nodes and GPU operations; local g++ CPU-only
tests are permitted. No throughput gain is assumed.

## Hypothesis and ceiling

M16 still loads checkpoint-transposed B `[K/2,N]`, then software-transposes
each K64 stage into MMA-ready shared storage. For N2048/K4096 this is64 stages
per projection. A byte permutation into `[N/128,K/64,128,32]` permits direct
coalesced 16-byte cp.async loads into double-buffered shared B. Keep all128
loader threads, exact A gathering/scales, per16 E4M3 B scales, scale2 and K64
MMA accumulation/row ownership. No new activation quantization or MMQ primitive.

The explicit transpose moves4KiB shared reads plus4KiB shared writes per
stage/CTA, in addition to staging/MMA reads; removing it does not reduce global
weight bytes. Direct B double buffers need12288 bytes versus9216+6144 before,
but actual registers/occupancy must be read from native compilation. Remove
the transpose barrier only after the existing async wait + CTA barrier has
made the next B/A/scale buffers ready; buffer reuse must stay CTA-ordered.

The target verifier is around111ms and proposal10.5ms in the saved v15 ledger.
The gate/up share G is not separately measured: even removing all gate/up has
cycle speed ceiling121.5/(121.5-G), not a guessed TPS. Faster staging may not
beat unchanged global traffic/cache/issue limits. The earlier rejected MMQ
experiment also changed activation arithmetic; this candidate does not.

## Fixture and hard memory cap

- Exact gate/up N2048/K4096; C4 or temporal K5, max4/5 rows per expert and
  actual token-major gathered source rows. Two distinct local expert pairs,
  all other expert pointer entries remote/null. No live expert-weight aliases.
- Four original packed matrices plus four repacked matrices32MiB; unchanged
  read-only scales shared between layouts2MiB. Six independently poisoned
  output buffers cover M64, M16 and tiled M16 gate AND up. Inputs are exactly
  C source rows, not route-padded. No extra weight pool exists concurrently.
- Explicit allocations include128-byte head/tail canaries and must be checked
  before cudaMalloc against a hard64MiB cap. Expected C4 total36,474,632 bytes;
  K5 total36,674,600 bytes. Runtime/context overhead is separate.
- Two local pairs are deliberately limited coverage: they cannot establish
  realistic distinct-expert working-set performance or model-wide scalability.

## Test-first gates

1. Write CPU map/gather/alias tests and packing tests before implementing pack.
   Observe RED with an identity-layout stub, then GREEN with exact packing.
   Exhaustively prove a bijection over all4,194,304 packed-byte indices, verify
   inverse mapping and full pack/unpack roundtrip for nontrivial bytes; reject
   unsupported shapes/extents. Packing never changes either FP4 nibble.
2. Only then implement a separate CUDA header derived from current M16.
   Change B addressing/staging only, with distinct symbols; all arithmetic
   remains independently checked against the included production M64 kernel.
3. Correctness runs execute M64, current production M16, then tiled M16 for
   scalar and vector scale loading. Poison ALL six outputs before EACH variant;
   compare complete gate/up outputs, independent CPU columns on every live
   local route, remote untouched rows, every input/weight byte and all canaries.
4. Capture builder+all three kernels at fixed pointers. Refresh expert offsets,
   original/repacked pointer tables, gather IDs and source activations between
   graph replays. Cover boundary/nonprefix IDs, zero inputs, variable local
   counts, empty/all-remote maps and partial all-local fixture; C4 and K5 builds.
5. Only after numerical/native memcheck gates, interleave M64/M16/tiled event
   timing, both direct and builder-inclusive, repeated5x100 medians. Report
   tiled/current-M16 speed ratio to isolate staging from prior tiling. Timing
   mode must itself still run every correctness gate. Packing is offline host
   setup, excluded from decode timing and not a claim of a production loader.

Stop on any byte, output, gather, worklist, guard or sanitizer mismatch. No
production integration without full-model A/B and a typed, equal-memory layout
contract for EVERY scalar/prefill consumer. Never feed repacked pointers into
checkpoint-layout kernels and never retain model-wide duplicate weights.

## Source checkpoint and root-owned native commands

Local CPU RED observed `FAIL: B tile row mapping` with the identity stub.
After implementation, C4 and K5 CPU tests pass, including all4,194,304 index
positions, independent inverse mapping, full byte pack/unpack, gathered maps,
alias rejection and capacity checks. Seven route maps also cover each live
expert row count1..4/5 using an additional partial uneven-local fixture.
Native compilation and GPU numerical/memory/timing gates are still pending.

```sh
g++ -std=c++17 -O2 -x c++ -DATLAS_MOE_DOWN_HOST_ONLY scripts/dev/bench_glm_moe_btile.cu -o /tmp/moe-btile-host
/tmp/moe-btile-host --host-test
g++ -std=c++17 -O2 -x c++ -DATLAS_MOE_DOWN_HOST_ONLY -DATLAS_MOE_TEST_ROWS=5 scripts/dev/bench_glm_moe_btile.cu -o /tmp/moe-btile-k5-host
/tmp/moe-btile-k5-host --host-test
nvcc -O3 --fmad=false -gencode=arch=compute_121a,code=sm_121a scripts/dev/bench_glm_moe_btile.cu -o /tmp/moe-btile
nvcc -O3 --fmad=false -gencode=arch=compute_121a,code=sm_121a -DATLAS_MOE_TEST_ROWS=5 scripts/dev/bench_glm_moe_btile.cu -o /tmp/moe-btile-k5
```

Root runs both native binaries without arguments, then under
`compute-sanitizer --tool memcheck --error-exitcode 1`. Only after all gates
pass, collect at least three paired `--timing` runs without concurrent model,
compiler or packaging work. All allocation canaries are rechecked after
timing, including the final map. The smaller repeated two-pair working set can
favor cache reuse more than real routing; even a large warm micro gain is not
a model-wide estimate. No registration, loader or deployment changes exist.
