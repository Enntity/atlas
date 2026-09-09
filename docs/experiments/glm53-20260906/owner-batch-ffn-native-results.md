# Three/four-owner FFN arithmetic qualification

2026-09-09, harness source `0caf784a`. One GB10, serial emulation of two ranks'
arithmetic. This is not a model, attention, NCCL, quality or serving benchmark.
The production selected server still admits two owners; M15/M20 dispatch is
not integrated into the model. No new tok/s result is claimed here.

## Comparison

Control uses the qualified JointShared M10 traversal in chunks: M10+M5 for
three owners, M10+M10 for four. The M5 tail retains its literal router, dense
gate/up and fused mHC path. Candidate processes all15/20 rows in one traversal,
with compact gate/up, generic-T shared projections and the existing dense M64
down kernel. Existing kernel arithmetic is unchanged. The harness now sizes
scratch explicitly and launches the generic router with `ceil(rows/16)` tiles.

All six input cases (three EP ownership masks, normal/reversed owner order)
pass scalar and vector scale-load variants. Checked router/top8, quantized
inputs, canonical routed intermediates, complete shared intermediates and
final mHC outputs are bit-exact. Poisoned outputs, canaries and restored-input
repeats pass. Independent full-router/full-post oracles and sampled full-K
projection oracles include every row, specifically rows16..19. These bounded
synthetic tensors do not establish checkpoint-wide numerical equivalence.

## Timing

Three warmups and20 measurements per order, after numerical gates. Both
orders include shared projections on each simulated rank, routing, sorting,
quantization, routed FFN, unpermute, local BF16 rank addition and mHC. No timed
device-to-host copies; local addition does not model communication latency.

| Owners | Scale loads | Order | Chunked ms | Wide ms | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| 3 | scalar | control first | 4.567136 | 2.204571 | 2.071666 |
| 3 | scalar | candidate first | 4.476312 | 2.155933 | 2.076276 |
| 3 | vector | control first | 4.072381 | 1.964467 | 2.073020 |
| 3 | vector | candidate first | 4.033902 | 2.008190 | 2.008725 |
| 4 | scalar | control first | 4.371240 | 2.213856 | 1.974491 |
| 4 | scalar | candidate first | 4.376645 | 2.166304 | 2.020328 |
| 4 | vector | control first | 3.915719 | 2.019848 | 1.938620 |
| 4 | vector | candidate first | 3.913102 | 1.967784 | 1.988583 |

A fresh four-owner process reproduced1.911–2.016x and a fresh three-owner
process2.030–2.116x across both load variants and orders, with all exact-output
gates again passing. This supports pursuing
the wider layer-major traversal after bounded ownership/scheduling integration;
serialized pairs alone do not realize this weight-reuse gain.

## Safety and evidence

Measured guarded device allocations: M15=148,714,200 bytes and
M20=150,296,520 bytes, both below the unchanged192MiB cap. Each variant passed
Compute Sanitizer memcheck with `ERROR SUMMARY: 0 errors` and zero leaked
bytes/allocations. Timing processes have120-second deadlines and2GiB no-swap
container limits; memcheck uses180 seconds and4GiB. Two host CPUs, isolated
network, no other model/build/GPU workload, and zero observed host swap.
Completed runs exited0 with OOMKilled=false. No node reset or driver change.
The same rebuilt harness also passes its original two-owner `joint-shared`
numerical control with the original147,131,880-byte allocation count.

Source archive SHA256:
`bf4877b12e9f3b2cd1c4f95282dff53c12671c19c530be72e5a2efca92e413f1`.
Native ELF SHA256:
`6bda63d02586942a4a6bb09fefe728e4d53c6255969311a1684350bf23b02f69`.
Build flags and translation units are in `scripts/dev/bench_glm_pair_ffn.cu`.
Run `--atol 0 --rtol 0 --repeat 20 --compare owner-batch --owners 3` or4;
memcheck uses repeat0. Exact CUDA builder image and container bounds are
retained in the inspect receipts.

Evidence directory:
`/home/abc/storage/models/atlas-campaigns/20260909/glm-native-controller/`:
`owner-batch-0caf784a-source.tgz`, `owner-batch-native-build.log`, and
`owner-batch-m{15,20}-native-{timing,repeat,memcheck}.log` plus inspect JSON.
Remote source/binary: `/tmp/atlas-owner-batch.jpnSW9G0` on the head Spark.

Large-context qualification, coherence, real tool calling and needle retrieval
remain mandatory for the eventual serving integration. This isolated harness
cannot answer those questions.
