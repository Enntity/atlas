# Packed QKV M5: correct, noisy timing; not promoted

2026-09-09. Standalone GB10 experiment, not an endpoint throughput result.
Production kernels and serving images remain unchanged. The candidate writes
row-packed Q/K/V directly instead of launching the existing packing kernel.

At M5, N4096, K4096, five alternating rounds of 100 repetitions per arm,
after ten warmups per arm, produced:

| Fresh process | Existing fused+pack median, us | Direct median, us | Median paired speed ratio |
| --- | ---: | ---: | ---: |
| Initial | 124.855 | 125.983 | 1.025 |
| Repeat | 133.077 | 128.959 | 1.027 |

The median paired ratio and ratio of arm medians are different statistics:
the latter is 0.991 initially and 1.032 on repeat. Four of ten paired rounds
regressed; paired ratios ranged 0.979–1.085. There is no robust improvement
large enough to prioritize integration over C2 feed-forward batching.
Keep this candidate isolated, not enabled in a serving build.

The v2 fixture passed all 18 numerical cases against both fused+pack and
independent scalar GEMVs, including owner/row permutations, signed impulses,
finite outputs, unused rows, immutable inputs and allocation guards. Memcheck
reported zero errors. Explicit device payload peaked at 30,018,304 bytes;
host payload at 30,572,800 bytes. Both kernel arms use 48 registers, 5,344 bytes
shared memory and zero stack/spills. An earlier fixture ABI mismatch was
corrected before any timing; v1 timings do not exist.

Reproduction: apply `scripts/dev/glm_packed_qkv_store.patch` using
`git apply --unidiff-zero` **only to an isolated source copy**, then compile
`scripts/dev/bench_glm_packed_qkv.cu` with CUDA 13.0:
`nvcc -std=c++17 -O3 --fmad=false -arch=sm_121a -Xptxas=-v`.
Run `--repetitions 0` under memcheck before `--repetitions 100` in two fresh
processes. Numerical/fault modes are described by the fixture and approved plan.

Fixture SHA256:
`c6a1e7913f7240e7b2ecb1a59ca9e1645be604461aff41631bc6254eefa4c6f2`.
Native binary SHA256:
`8a4bca88767001319019356822269530a47ff880a21b901b39504739a8de1fc0`.
Raw logs remain under `/home/abc/storage/models/atlas-campaigns/20260908/`:
`glm-packed-qkv-{compile,numerical,memcheck}-v2.log` and
`glm-packed-qkv-timing-{initial,repeat}-v2.log`. Native containers exited 0
without OOM. Final checks found both nodes idle, no GPU processes and zero
swap use. No production source, clocks, drivers or system settings changed.

## Working priority

Following the user's correction, prioritize measured warmed C1/C2/C3/C4
throughput over expanding test infrastructure. Use focused correctness and
hardware-safety gates during development; run broad suites at deployment
checkpoints. Test counts are not performance progress. The next direct
experiment is the default-off C2 compact feed-forward adapter, with a matched
same-binary scalar control and unchanged C1/C3/C4 paths.
