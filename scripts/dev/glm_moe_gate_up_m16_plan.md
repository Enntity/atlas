# Bounded fused gate/up M16 follow-up

Plan before implementation. Test-only header extension and standalone fixture;
no production registration, serving flags, commits, GPU/node operations by the
author. Root independently reviews and runs hardware gates when models stop.

Clean M16 down pairs include losses despite positive medians. Down is not
promoted. This follow-up preserves the128-loader-thread, fourN32-warp M16
hypothesis and native K64 MMA order, testing a different existing hot path.
It does not retry the rejected compact-down scheduling or M32/block64 design.

## Exact geometry and bounded ownership

- Gate/up output N2048, reduction K4096. Inputs are C4 or K5 token-major packed
  FP4 rows with per16 E4M3 scales, NOT expert-sorted activation rows. Production
  `sorted_token_ids[route]` gathers source row0..3 or0..4 for each expert route.
- Each expert receives at mostC token rows, with distinct source IDs within
  that expert; IDs repeat across experts, as real top8 routing permits.
- Four distinct local gate/up weight pairs: eight matrices,36MiB packed+scale
  storage. Other routed experts are remote/null. No live weight aliasing.
  Test full32/40-route maps with four local/four remote experts, boundary IDs
  143/144/287, variable per-expert counts, zero-local and all-empty maps. Partial
  all-local maps are unit fixtures, not a claim of complete top8 C4 routing.
- Dedicated count/worklist, offsets, source IDs and four output buffers; hard
  64MiB explicit GPU cap. Host uses one expert-sized weight temporary at a time.
  CUDA runtime/context overhead is outside explicit device accounting.

## Test-first comparison

1. Write host map/gather/capacity rejection tests and wire candidate calls before
   extending the header. CPU-only build must perform no CUDA calls.
2. Include exact current production grouped-GEMM and worklist-builder sources.
   Compare the production fused compact gate/up export against M16 projection-Y
   wrappers calling the same test-only M16 arithmetic body. Compact worklist
   n_tiles16, m_tile64, maxC*8*16. Since every expert has<=5 rows, each has
   m_tile0; do not generalize beyond that contract.
3. Both gate AND up outputs independently poisoned before every vecscale and
   scalar-scale launch. Full output bit equality; independent CPU FP4 products
   for selected columns of every local route; remote/unused rows remain poison.
   Weights/scales/global scale2 are distinct between projections so swapped
   pointers or duplicated output cannot pass. Source-token IDs are permuted and
   repeated across experts to catch accidental expert-sorted A addressing.
4. Host independently builds expected worklist entries; check every entry/count
   and poison tail. Verify all inputs/weights/metadata immutable and allocation
   guards intact. All legal gather IDs checked before GPU; never test bad IDs
   by unsafe device reads. Default source-row buffer is exactlyC rows.
5. Same graph captures builder+production+candidate at fixed pointers. Refresh
   maps, source IDs, pointer tables and inputs between replays; compare eagerly
   and via graphs. Compile/test C4 and K5 separately. Existing down remains
   eligible only at N4096/K2048 with NULL gather; gate/up accepts N2048/K4096
   with actual gather. No other geometry becomes eligible.
6. Only after full gates/memcheck, event-time production versus candidate fused
   calls, plus builder+each call, using interleaved repeated medians. Builder
   policy and projection multiplexing are matched; distinguish launch-only
   fusion from shared arithmetic. No down fixture resident concurrently.

Stop on any bit/oracle/gather/map/guard/memcheck failure. Synthetic four-local
pair coverage does not represent every route histogram/cache working set, real
activation quantization distribution, downstream SiLU/down/EP collectives, or
full-model speed. No promotion follows from a standalone pass.

## Source checkpoint and commands

CPU-only map, gather, alias and allocation tests pass for both C4 and K5.
Calculated explicit device allocation, including all 18 allocation guards, is
38,303,752 bytes for C4 and 38,438,184 bytes for K5. The runtime allocation
counter enforces the 64MiB cap before each allocation; hardware has not yet
confirmed this new fixture. Native compilation, numerical comparisons and
memcheck remain pending independent review and root's stopped-model window.

```sh
g++ -std=c++17 -O2 -x c++ -DATLAS_MOE_DOWN_HOST_ONLY scripts/dev/bench_glm_moe_gate_up_m16.cu -o /tmp/moe-gu-m16-host
/tmp/moe-gu-m16-host --host-test
g++ -std=c++17 -O2 -x c++ -DATLAS_MOE_DOWN_HOST_ONLY -DATLAS_MOE_TEST_ROWS=5 scripts/dev/bench_glm_moe_gate_up_m16.cu -o /tmp/moe-gu-m16-k5-host
/tmp/moe-gu-m16-k5-host --host-test
nvcc -O3 --fmad=false -gencode=arch=compute_121a,code=sm_121a scripts/dev/bench_glm_moe_gate_up_m16.cu -o /tmp/moe-gu-m16
nvcc -O3 --fmad=false -gencode=arch=compute_121a,code=sm_121a -DATLAS_MOE_TEST_ROWS=5 scripts/dev/bench_glm_moe_gate_up_m16.cu -o /tmp/moe-gu-m16-k5
```

GPU execution is root-only. Run each native binary without arguments and with
`compute-sanitizer --tool memcheck --error-exitcode 1`; only then collect paired
`--timing` receipts. Timing mode itself still runs the full correctness gates.
Use explicit architecture/code generation above: the earlier native FP4
translation unit failed compilation with a generic PTX target.

The private M16 body now accepts two narrowly guarded geometries. Its existing
down exports explicitly retain N4096/K2048 with NULL gather, but this source
change can affect generated code: rerun the existing down C4 and K5 numerical
and memcheck regression gates as well. This does not change the decision to
leave down unpromoted.
