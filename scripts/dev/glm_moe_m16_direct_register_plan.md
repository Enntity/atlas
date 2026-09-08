# Original-T M16 direct-register B fragments

Scope: new `scripts/dev` files only. Root approves this plan before code, owns
commits/native builds/GPU/node operations and eventual production decisions.
No model, loader, flags, resident layout or existing frozen source changes.

## Hypothesis and exact change

Current production `glm_moe_gate_up_m16.cuh` loads original T packed B
`[K/2,N]` into double-buffered `m16_BpT[2][32][144]`, software-transposes to
`m16_Bp[128][48]`, then reads two32-bit B words per M16 MMA fragment.
Preserve that original global layout and asynchronous B/A/scale loaders.
Instead, each consumer lane forms the same words directly from `BpT`:

`b0 = OR_j(BpT[cur][4*tid+j][nc] << (8*j))`,
`b1 = OR_j(BpT[cur][16+4*tid+j][nc] << (8*j))`, j0..3,
`nc = warp*32 + nt*8 + lane/4`, `tid=lane%4`.

Remove only the intermediate6144-byte shared allocation, software transpose
macro/calls and the second associated CTA barrier. Retain async wait plus the
first CTA barrier before next-buffer consumption/reuse. This removes4096 bytes
of shared stores plus4096 bytes of shared reads and one barrier per K64/CTA;
global traffic, A FP4 quantization, scales and K64 MMA accumulation are unchanged.
Different shared-bank access patterns/register scheduling may erase the gain.
Static source-derived shared storage drops18112→11968 bytes, but native register,
spill and occupancy evidence must be measured; no speed prediction is made.

## Files and test-first execution

1. New `glm_moe_m16_direct_fragment.h`: shared host/device word builder actually
   called by the new CUDA candidate. CPU test first uses an incorrect byte
   order and must fail against an independent materialized transpose/reference
   word, then correct it. Exhaustively check both stage buffers, all128 lanes,
   four N fragments, both words and all bytes (all4096 active B-stage bytes
   exactly once), several patterns plus isolated-byte bit probes. Preserve and
   verify the16-byte padding region per T row. No floating-point math in this
   fragment test.
2. New `glm_moe_m16_direct_register.cuh`: standalone clone of current production
   M16 helper with distinct symbols. Apply only the changes above. Exact
   N2048/K4096, real token-major gather, at most5 rows/expert; no BF16-input or
   oversized-prefill claim. Keep original production M16/M64 as references.
3. New `bench_glm_moe_m16_direct_register.cu`: adapt the current guarded fused
   GU fixture to three independently poisoned output pairs (M64, old M16, new
   direct-register M16). Both scalar/vector scale loaders and graph variants;
   C4/K5 compile profiles and explicit per-expert populations1/2/3 plus existing
   full/varied/remote/empty/boundary/partial-local maps. Full numerical bit
   equality and independent CPU columns on every local row, consumed zero
   source row, untouched remote/padding outputs and full immutable inputs.
4. Complete default-stream initialization before nonblocking work; enqueue
   metadata/input uploads on that same stream and preserve their host lifetime
   through synchronization. Graph addresses remain fixed while all metadata
   and inputs refresh. Check all allocation guards and all weight/scale bytes,
   including after timing; fail closed on any CUDA/oracle error.

## Explicit device fixture budget

Retain four distinct local gate/up pairs: eight packed matrices32MiB and eight
scale matrices4MiB. Add only the third output pair to the existing fixture.
Exactly20 allocations, each with128-byte head/tail guards. Explicit budgets:
**38,566,408 bytes at C4**, **38,766,376 bytes at K5**, both below64MiB.
Derive/assert exact allocation sum in the CPU fixture and actual native counter;
no resident T/Btile duplicate, quantization scratch or extra worklist.
Host weight verification streams one matrix at a time, not all weights at once.

## Root-only gates and timing

CPU fragment RED→GREEN plus map/budget tests precede independent review/freeze.
Root compiles both C4/K5 under `--fmad=false` and default FMA, records ptxas
register/shared/spill data and native/memcheck gates before timing. The fixture
must compare all three kernels, not only M64 versus the candidate. At least
three clean timing invocations report separate old-M16/new-M16 launch-only and
builder-inclusive paired medians, interleaved5x100; keep negative cases. Full
post-timing outputs, input/metadata and guards are rechecked. No whole-model
inference or production promotion follows from a four-local-pair hot fixture.

CPU receipts persist under
`/home/abc/storage/models/atlas-campaigns/20260908/m16-direct-register/`.

## Source-only receipts

`fragment-red.log` records exit 2 from the actual helper with reversed byte
order: `FAIL: actual direct fragment vs materialized transpose bits`.
`fragment-green.log` records the corrected helper passing. The final complete
fixture repeats that same exhaustive helper test in both `c4-host.log` and
`k5-host.log`, followed by map/population/guarded-budget checks. CPU compilation:

```sh
g++ -std=c++17 -O2 -x c++ -DATLAS_MOE_DOWN_HOST_ONLY \
  scripts/dev/bench_glm_moe_m16_direct_register.cu -o <receipt-dir>/c4-host
g++ -std=c++17 -O2 -x c++ -DATLAS_MOE_DOWN_HOST_ONLY -DATLAS_MOE_TEST_ROWS=5 \
  scripts/dev/bench_glm_moe_m16_direct_register.cu -o <receipt-dir>/k5-host
<receipt-dir>/c4-host --host-test
<receipt-dir>/k5-host --host-test
```

These host passes do not compile CUDA, establish native numerical correctness,
prove sanitizer safety, or measure performance. Native gates remain mandatory.
