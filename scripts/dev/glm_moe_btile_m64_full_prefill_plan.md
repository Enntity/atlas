# Standalone B-tile M64 full-prefill compatibility

Scope: script-only kernel/harness extension; root owns commits, native builds,
GPU gates and all node operations. No production dispatch or precision changes.

1. Add a shared host/device M64 geometry predicate; require actual CPU tests
   to fail with a rejecting stub before implementing 1..1024 expert rows and
   M tiles 0..15. Keep N2048/K4096 and prequantized FP4/K64 accumulation.
2. Match all existing production prequant ABIs: separate dense grid, separate
   compact and fused compact, scalar/vector scale loaders, gathered and
   explicitly route-major no-gather inputs. No BF16-input claim.
3. Extend the existing guarded fixture, retaining two distinct gate/up pairs.
   Four original and four packed matrices cost 32 MiB; four scale matrices
   cost 2 MiB. Four BF16 outputs of 1088x2048 cost 17 MiB; A packed/scales
   of 1088x4096 cost 2.390625 MiB. Pointer tables, offsets, worklist and all
   individual 256-byte guards bring the explicit total to **56,015,240 bytes**
   across 21 allocations, below 64 MiB.
   Gathered inputs address only 1024 real token rows. No-gather inputs are
   explicitly expanded route-major inputs, with capacity 1088, not fake
   token-major rows. Output routes include remote and unused poison regions.
4. Cover fallback widths 1/2/3/4/5, preserve tails 15/16/17/63/64/65/127/128/129 and add concentrated
   130/148/255/256/257/1023/1024 local populations with shuffled gathers,
   nonzero offsets, another local pair, remote-only and empty cases.
5. Fresh poison and full output bit equality to old production kernels for
   every ABI/scale/gather choice; finite/local and untouched remote/padding
   checks, independent sampled CPU columns on every local row, graph replay
   with refreshed inputs and metadata, immutable inputs/weights/scales and
   allocation guards. Root runs both FMA policies and memcheck before timing.

The CPU fixture validates the exact work bound/ownership and guarded budget;
CUDA compilation, numerical correctness, sanitizer and speed remain unproven
until root records native receipts. No model-wide performance prediction.

## CPU evidence and exact native interfaces

The shared `glm_btile_m64_work_valid` host/device predicate initially returned
false; its actual CPU test failed with `FAIL: full-prefill M64 work coverage`.
After implementing bounded rows/M/N tiles the test passes, including empty,
negative, 1025-row and oversized tile rejections. The extended fixture passes
exhaustive byte permutation/inverse, unique gathers, full row ownership, alias,
route/source bounds, worklist bounds and the exact guarded allocation sum.

New separate projection exports mirror production arguments in order:
`A_packed, A_scale, B_packed_ptrs, B_scale_ptrs, scale2_vals, C,
expert_offsets, sorted_token_ids, num_experts, N, K`.
`glm_moe_btile_m64_{dense,vecscale_dense}` uses grid `(N/128,16,288)`;
`glm_moe_btile_m64_{compact,vecscale_compact}` additionally takes
`worklist, total_tiles, max_tiles` and grid `(max_tiles,1,1)`.
The existing `glm_moe_gate_up_btile_m64{,_vecscale}` fused ABI is unchanged,
with grid `(max_tiles,2,1)`. All use 128 threads and the same M64 body.
The compact fixture capacity is 512 work items; a work item is the original
`(expert, (m_tile << 6) | n_tile)` pair. Null gather explicitly means route-major.

The fixture captures 12 graphs: every ABI x scalar/vector scales x gather mode.
Every case refreshes pointer tables, offsets, gathers and activations at fixed
addresses, then compares all four outputs after fresh poisoning. A zero source
row is guaranteed to be consumed whenever the case has local work. Both native
FMA policies and sanitizer remain root-owned pending gates. The optional
`--timing` currently reports only fused compact/vector-scale performance (both
gather modes), not dense/separate ABI speed or serving throughput.

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror scripts/dev/test_glm_moe_btile_m64_bounds.cpp -o /tmp/glm-btile-m64-bounds
/tmp/glm-btile-m64-bounds
g++ -std=c++17 -O2 -x c++ -DATLAS_MOE_DOWN_HOST_ONLY scripts/dev/bench_glm_moe_btile_m64.cu -o /tmp/glm-btile-m64-full-host
/tmp/glm-btile-m64-full-host --host-test
```

Native root-only build command (model and other GPU work stopped):
`nvcc -std=c++17 -O3 --fmad=false -gencode=arch=compute_121a,code=sm_121a
scripts/dev/bench_glm_moe_btile_m64.cu -o <frozen-output>`.
Repeat omitting `--fmad=false`, then execute each binary ordinarily and through
`compute-sanitizer --tool memcheck --error-exitcode 1`. No timing before gates.
