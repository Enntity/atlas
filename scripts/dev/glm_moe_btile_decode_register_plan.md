# B-tile BF16 decode: split shared loop and register-packed B

Standalone follow-up to the correct but slow direct/shared-stage candidates.
Do not edit their frozen files or production. Root owns native/GPU/node work.

## Evidence and hypothesis

Saved `btile-compat-native-timing-pairs.log` includes rows1/case2 with all
routed experts remote and only original-layout shared up active: production
88.102us, direct188.156us, stage298.261us in the first printed run. This
workload reads no tiled B and takes no staged reload barriers. Therefore
neither tiled coalescing nor staging alone explains the regression.

Root's `btile-decode-native-codegen.log` reports REG40/STACK0/LOCAL0 for
all production and candidate decode exports: no observed register spills.
Stage1 SASS emits two128-bit loads, eight shared stores and two CTA barriers
per K64 reload, followed by per-byte predicated original-global versus
staged-shared read/address paths. The compiler did not isolate the shared
arithmetic into a clean independent loop. Extra instruction scheduling is
implicated; its exact cycle contribution is not measured.

New variant pair:

1. `word`: branch once to an unchanged ordered original-layout shared loop;
   routed B uses one aligned uint2 load per group16, extracting the eight
   packed bytes from two registers. Preserve the existing256-group loop.
2. `vec`: same separate shared loop; routed B loads two aligned uint4 chunks
   per K64, then consumes four explicitly ordered groups, each eight packed
   bytes. More live registers/code size are risks, not assumed wins. No
   register arrays with runtime indexing, reductions, activation quantization,
   or arithmetic reassociation. Exact FP32 expressions/FMA policy unchanged.

Both avoid B shared staging and its barriers. The existing16-entry LUT shared
initialization remains. Shared B stays transposed, routed B stays the exact
previous tiled byte layout, scales/scale2 unchanged. Runtime N/K retained.

## Gates, allocation and independence

Use a new harness copied from the frozen BF16 decode fixture, with the exact
same included production scalar/batch2/batch3 numerical oracles. Replace only
the two candidate launch names/header and timing labels; retain all48
eager/graph cases, CPU projection columns, full12-output comparisons, fresh
poisoning, immutable buffers, width tails, remote zero and post-timing checks.
The old negative candidates remain in their original separate harness; run
those binaries serially as additional controls, not concurrently.

Device footprint remains45,799,520 bytes across25 guarded allocations, hard
64MiB cap. Two routed local pairs plus a distinct original shared pair are a
limited hot working set; no serving gain can be inferred from this alone.

CPU TDD first: deliberately wrong packed-word byte extraction must fail,
then validate every byte of all4,194,304 packed positions through independent
word/vec selection and the existing packing/slice/map/budget tests. Native
correctness and memcheck under both production `--fmad=false` and default FMA
must remain bit-exact. Capture resource usage/SASS: expected word read width64,
vec width128, no local spills, and no routed address calculations inside the
shared loop. Correctness takes precedence if compiler decisions differ.

Three interleaved timed runs compare production, word and vec. Preserve all
cases and repetitions, especially shared-only and shared-present cases; no
selection based only on shared-absent workloads. Full-model integration stays
blocked until scalar compatibility costs are acceptable and all other layout
gates pass. No production promotion or new serving flag in this experiment.

```sh
g++ -std=c++17 -O2 -x c++ -DATLAS_BTILE_DECODE_HOST_ONLY scripts/dev/bench_glm_moe_btile_decode_register.cu -o /tmp/btile-decode-register-host
/tmp/btile-decode-register-host --host-test
nvcc -O3 --fmad=false -gencode=arch=compute_121a,code=sm_121a scripts/dev/bench_glm_moe_btile_decode_register.cu -o /tmp/btile-decode-register
/tmp/btile-decode-register
compute-sanitizer --tool memcheck --error-exitcode 1 /tmp/btile-decode-register
/tmp/btile-decode-register --timing
```

Omit `--fmad=false` for the second required numerical/memcheck build. Only
local g++ CPU tests are agent-run; all other commands above are root-owned.

CPU checkpoint: wrong byte-extraction stub compiled then failed with
`FAIL: packed word byte extraction` (exit2). Corrected helper passes exhaustive
word/vec byte selection plus all inherited fixture tests. Receipts:
`btile-decode-register-host-{red,green}.log` in the phase6 receipt directory.
No native or GPU correctness/performance claim for these new variants yet.
