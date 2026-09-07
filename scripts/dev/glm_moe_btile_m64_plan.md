# Standalone M64 compatibility for packed B tiles

Scope: new files in `scripts/dev` only. No existing prototype header, loader,
kernel registration, production dispatch, or persistent allocation changes.
Root owns every native CUDA/GPU/node operation; author runs only local g++
CPU tests. This checks a possible future prefill consumer, not promotion.

Keep the original M64 prequant gate/up A gathering, four-warp M ownership,
16 N fragments per warp, FP4/scales, K64 accumulation order and output layout.
Change only B addressing into the validated byte layout
`[N/128,K/64,128,32]`, directly loaded into double-buffered shared B rows.
Reuse the original worklist builder with M tile width 64; unlike the M16
prototype, accept multiple packed M-tile indices. All returns stay CTA-uniform;
the async wait plus CTA barrier protects both next-buffer readiness and old
buffer reuse. Exact N2048/K4096 only, scalar and vector scale loaders.

Write CPU layout/worklist/gather/budget tests first; observe RED with an
identity-map stub, then implement the same byte permutation and observe GREEN.
Check M15/16/17,63/64/65,127/128/129, nonzero expert start offsets,
shuffled/nonprefix gather IDs, empty and remote-only cases. A source allocation
contains 129 token-major rows, never route-padded fake sources. Two distinct
local gate/up pairs remain a deliberately limited working set.

Hard explicit device allocation cap: 64 MiB checked before every cudaMalloc,
including 128-byte guards on each side. Four checkpoint matrices plus their
four tiled copies are 32 MiB; scales are 2 MiB. Four complete output buffers
(original/candidate gate/up) for at most 8*129 routes add ~16.1 MiB. Source,
tables, offsets and worklist stay within the remaining allowance. CUDA runtime
and graph bookkeeping are separate; no model may be loaded concurrently.

Native gates before timing: complete output bit equality to original M64,
independent CPU sampled columns on every local row, finite outputs, untouched
remote/padding rows, full byte immutability for weights and all live inputs,
all allocation guards, and fixed-pointer graph replay after metadata/input
refresh. Timing must run these gates first and repeat all output/immutable/
guard checks afterward. Interleave original/candidate with and without builder;
do not infer whole-model speed from a two-pair warm-cache fixture. Root runs
native and memcheck, then optional timing only after approval.

## CPU checkpoint and root-owned native gates

Identity-layout stub produced the expected `FAIL: B tile row mapping` (exit2).
The real packing implementation passes exhaustive 4,194,304-byte bijection,
inverse and roundtrip tests, all nine requested row geometries, explicit
four-warp row ownership with tails and multiple packed M tiles, empty/remote
maps, invalid gathers/aliases, and allocation overflow/budget checks. Explicit
peak device buffers total **52,884,648 bytes**, including all 21 allocations'
guards. The runtime asserts that actual allocation accounting matches this
CPU-derived total. Native compilation/numerical/sanitizer gates are pending.

```sh
g++ -std=c++17 -O2 -x c++ -DATLAS_MOE_DOWN_HOST_ONLY scripts/dev/bench_glm_moe_btile_m64.cu -o /tmp/moe-btile-m64-host
/tmp/moe-btile-m64-host --host-test
nvcc -O3 --fmad=false -gencode=arch=compute_121a,code=sm_121a scripts/dev/bench_glm_moe_btile_m64.cu -o /tmp/moe-btile-m64
/tmp/moe-btile-m64
compute-sanitizer --tool memcheck --error-exitcode 1 /tmp/moe-btile-m64
```

Only root runs the final three commands with both models stopped. After all
correctness gates pass, `--timing` reruns those gates and adds interleaved event
medians. Report the actual per-expert M separately from the fixed 129-row
source capacity. Check all outputs/immutable metadata again after timing;
check every original/repacked weight and scale byte at the end. No inference
about a production loader or serving performance follows from these tests.
