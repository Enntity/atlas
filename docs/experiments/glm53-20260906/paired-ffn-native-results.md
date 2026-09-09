# Paired routed-FFN native comparison

2026-09-09, source `9aca8d8ff159aa9c7a6db21e819abeb92bff9f14`.
This is a one-GB10 arithmetic microbenchmark, not serving throughput or NCCL.

The two-M5 control uses the current generic-T shared projections and separate
dense-grid native-FP4 routed gate/up calls. The M10 candidate combines routed
work through the existing compact fused gate/up path, preserving two generic-T
shared projections. Both retain the existing native-FP4 dense-grid down kernel.

## Exact comparison and timing

Six input cases cover the expert-rank boundary, all-local/all-remote route
masks, ten rows per selected expert and reversed owner rows. Each runs scalar
and vector scale-load variants. Every expert projection has its own complete
packed/scales tensor; no aliased weight replicas. Actual router/top8/sort,
quantizers, routed intermediates and unpermute results are compared, including
sampled independent projection oracles and an independent mHC arithmetic oracle.

The initial exact-K5 GEMV shared candidate failed: shared relative RMS difference
0.030117359 and maximum absolute difference17.75 on this synthetic fixture;
final mHC relative RMS was approximately0.02725. It was rejected. Preserving
generic-T shared arithmetic produced **zero differing checked routed, shared
and final mHC elements**, with absolute/relative tolerance both zero. Restored
input repeats and allocation canaries passed. Compute Sanitizer reported
`ERROR SUMMARY: 0 errors` with a normal process exit.

One post-commit process, three warmups and20 measured repetitions per order:

| Scale loads | Order | Two-M5 ms | M10 ms | Ratio |
| --- | --- | ---: | ---: | ---: |
| Scalar | Control first | 4.755650 | 2.986783 | 1.592232 |
| Scalar | Candidate first | 4.677725 | 2.899910 | 1.613058 |
| Vector | Control first | 4.222815 | 2.715565 | 1.555041 |
| Vector | Candidate first | 4.152382 | 2.714434 | 1.529742 |

An earlier process gave1.52–1.60x. These timings serialize two ranks' arithmetic
on **one GPU**, including replicated shared work. Rank addition is local BF16
arithmetic, not communication latency. There is no timed D2H and no attention,
Model, scheduler or token generation in this benchmark. Routing normalization
and scale are explicitly1/1; this is not a checkpoint-distribution benchmark.

## Bounds and reproducibility

All52 guarded device allocations total147,131,880 bytes under the explicit
192MiB cap. Plain execution has a120-second deadline and2GiB no-swap container
limit; memcheck has180 seconds and4GiB. Both used two host CPUs with no other
model/build/GPU job active. All observed swap was zero; neither run was OOM-killed.
The production pair workspace introduces no new resident allocation.

Build with CUDA13, `-O3 --fmad=false -std=c++17`, `sm_121a`; source and exact
translation units are listed in `scripts/dev/bench_glm_pair_ffn.cu`.
Run `--atol 0 --rtol 0 --repeat 20`; timing cannot begin before comparison passes.
Native ELF SHA256:
`e593712976366fc80199dfa164ecd65f419ff189fd0b0bf7947383135b006803`.
External retained evidence under
`/home/abc/storage/models/atlas-campaigns/20260909/glm-native-controller/`:
`m10-ffn-native-strict.log`, `m10-ffn-native-genericT.log`,
`m10-ffn-native-genericT-postcommit.log`, `m10-ffn-native-memcheck.log` and
the two source archives/build logs. Full-model state/quality and warmed serving
comparison remain separate requirements; these ratios are not tok/s gains.
