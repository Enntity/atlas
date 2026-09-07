# Standalone GLM routed-down cost decomposition

Plan written before test/harness implementation. No production changes,
serving flags, remote operations, GPU execution by the author, or commits.
Root reviews the source before any controlled compile/run. The historical
compact-down stall/regression remains unexplained; this experiment cannot
retroactively identify its cause from a synthetic fixture.

## Scope and test-first contracts

Include the exact current production `moe_w4a16_grouped_gemm.cu` shadow and
`common/moe_permute.cu`; do not copy or modify their arithmetic. Test native
prequantized FP4 down only: K2048, N4096, M_TILE64, output tiles128, 288 global
experts, at most32 expert-sorted activation rows. `sorted_token_ids` is NULL.
The scalar-scale and vecscale dense/compact exports must agree bitwise.

1. Host-only `--host-test` validates fixture offsets, local/null mappings,
   maximum4 rows/expert, at most32 routes, canonical independent worklists and
   checked allocation arithmetic. Reject malformed maps before CUDA. This mode
   exits before CUDA stream creation or allocation.
2. First GPU test uses eight populated experts, four rows each, followed by280
   empty experts. Compare the SAME dense export/grid `(32,1,8)` against
   `(32,1,288)`, block128: identical useful rows, weights, strides and math.
   Only trailing empty experts differ. Never trim through a populated expert.
3. Build the compact worklist using the production builder, but check its count
   and every `(expert,m_tile,n_tile)` against independently computed host entries.
   Down requires32 N tiles, not gate/up's16. Capacity is32*32 entries (8192
   bytes), plus separately allocated count. The builder has no capacity argument:
   validate all counts before submission. No deliberately malformed GPU maps.
4. Compare complete dense/full, dense/trimmed (when eligible), and compact outputs
   bitwise, with sampled independent CPU decoding of FP4/scale products to avoid
   two wrappers being the sole arithmetic oracle. Use bounded dyadic scales so
   the reference sum is exact in FP32; require the correctly rounded BF16 bits.
5. Cover first8 experts, mixed local/remote experts, IDs143/144/287, single-expert
   skew with4 rows (a partial-route unit fixture, not32 rows on one expert),
   all-empty, and remote-only. Real C4 top8 cannot route more than4 rows to one
   expert when each request's top8 is unique. Poison every output before each case; remote
   and unused rows must retain poison. Check all input, weight, pointer-table,
   offset, count/worklist and output canaries. Refresh device offsets/tables under
   one captured full-dense + builder + compact graph without changing pointers.
6. The synthetic immutable weight pool has eight distinct real-shaped down
   matrices. Different maps may reuse these buffers across cases; no two live
   local experts within a case alias weights. No full288-expert weight allocation,
   persistent model, collective, quantization or SiLU staging-copy integration.

## Memory and aliasing boundaries

Hard64MiB ceiling for explicit live GPU allocations, checked before cudaMalloc.
Eight packed weight matrices plus scale matrices use36MiB. Activations, three
outputs and small metadata remain below2MiB extra. Host initialization/validation
uses one expert-sized temporary at a time, not a second36MiB weight pool. CUDA
context, graph and event implementation overhead is outside explicit accounting.

Activation input is already expert-sorted, not original-token-major. Worklist and
count are dedicated test allocations, never aliased with gate/up or down outputs.
Therefore a passing isolated test does not prove production scratch lifetime:
the historical integration would additionally need correct down worklist extent,
same-stream ordering, staged SiLU/FP4 copy ordering, remote-output masking, and
shared-after-EP reduction. Preserve current serving exclusions.

## Timing, only after correctness and memcheck

Separate `--timing` reports five interleaved median-of100 CUDA-event intervals
for dense8, dense288, compact-only, builder-only, and builder+compact. Setup,
output poisoning and graph construction are outside intervals. Report all paths
for first8, all-empty and remote-only fixtures; dense8 is eligible only when all
nonempty experts are in the first8. The no-work cases measure actual early-exit
overhead, not an estimated subtraction from useful kernel runtime. Mark eager
submission explicitly; never infer whole-model speedup from synthetic locality.
No timing acceptance threshold or serving promotion is part of this experiment.
Timing mode first runs the complete eager and captured-graph correctness suite,
then revalidates each fixture on a third pass before any measurement. It does
not bypass correctness; memcheck remains a separate root-owned invocation.

## Root-owned commands and stop gates

```sh
g++ -std=c++17 -O2 -x c++ -DATLAS_MOE_DOWN_HOST_ONLY scripts/dev/bench_glm_moe_down_cost.cu -o /tmp/bench-glm-moe-down-cost-host
/tmp/bench-glm-moe-down-cost-host --host-test
nvcc -O3 --fmad=false -arch=sm_121a scripts/dev/bench_glm_moe_down_cost.cu -o /tmp/bench-glm-moe-down-cost
/tmp/bench-glm-moe-down-cost --host-test
/tmp/bench-glm-moe-down-cost
compute-sanitizer --tool memcheck --error-exitcode=99 /tmp/bench-glm-moe-down-cost
/tmp/bench-glm-moe-down-cost --timing
```

Stop on any oracle/bit/canary/map mismatch, runtime error or memcheck finding.
Model must be stopped before the GPU experiment; root owns node memory checks,
execution, receipts and restoration. Local nvcc availability and source checks
will be reported separately; no CUDA correctness claim before root's receipts.

## Source checkpoint

Implemented only `bench_glm_moe_down_cost.cu` plus this plan. Local CPU-only
g++ compilation and map/offset/worklist/weight-alias/allocation tests passed.
Explicit GPU footprint computes to38,590,472 bytes including allocation guards;
the runtime also checks each allocation against64MiB. `git diff --check` passed.
Local nvcc is unavailable, so CUDA compilation and all native numerical,
graph, memcheck and timing results remain pending. No GPU/node operations or
production modifications were performed. Root must review before compilation
on the Sparks and before any GPU gate.

Review correction: scalar-scale control outputs are freshly poisoned after
the vecscale comparisons and before scalar launches. Reusing correct prior
output could otherwise conceal omitted scalar writes. Output guards are checked
again immediately after scalar synchronization. Apply this independent-output
initialization rule to every future multi-variant correctness harness.
