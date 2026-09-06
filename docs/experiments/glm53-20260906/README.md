# Dual-Spark kernel and concurrency campaign — 2026-09-06

Base: `f44b03235e6ac5e2881f08673b307ad9fb16d85d`.
Raw full-model observations are in [receipts.json](receipts.json).

This page records the first campaign. The follow-up
[graph comparison](phase2-results.md) found only small/mixed throughput
changes; its graph opt-in stays default-off. See the normalized
[peer comparisons](phase2-comparisons.md) and next
[C4 implementation plan](c4-implementation-plan.md).

The [C4 follow-up results](phase3-results.md) include a corrected dense MLA
kernel binding; historical dense-path needle passes did not establish numerical
correctness. The next work follows the pinned
[vLLM infrastructure roadmap](vllm-infrastructure-roadmap.md), beginning with
state-indexed KDA execution and explicit cache ownership.

Measured wins are distinct experiments, not one combined benchmark:

| Experiment | Control | Candidate |
| --- | ---: | ---: |
| 32K C1 prefill, WMMA scorer | 396–403 tok/s | 449 tok/s |
| 1K C3 aggregate decode, grouped MoE + MLA | 22.129 tok/s | 27.366 tok/s |
| 1K MTP prefill, existing FP4 prefill option | 923.049 tok/s | 1049.828 tok/s |

[Deployment profiles and rollback](deployment.md) distinguish short-context
throughput from long-context concurrent service. The latter requires eager
semantic indexing and is slower; its final-profile 3K-prompt C3 test measured
15.288 aggregate tok/s (median session 5.804 tok/s). Do not quote 27.366 tok/s
as its performance. Final-profile mixed-threshold and three-way 10K retrieval
checks passed, without foreign needles. Reported 10K TTFT ranged 22.2–40.1 s.
Chunk size and FP4 precision changed together for this final profile, so that
TTFT difference is not an isolated kernel A/B. Some retrieval responses ended
before their caps through EOS or the content-loop watchdog; they are quality
checks, not fixed-output throughput tests.

## Measured prefill result

Same GLM-5.3-Flash NVFP4 checkpoint, TP2/EP2, native FP4 target head,
BF16 KV, speculative decoding off, one admitted sequence, 32768 context,
6144-token prefill chunks, prefix cache disabled. Control image: `decode-v65`.
Candidate image: `kernel-20260906-v1`, with `GLM_INDEX_WMMA=1`.
Both ranks used the same candidate binary SHA256 recorded in the receipts.

| Prompt | Control prefill tok/s | Candidate prefill tok/s |
| --- | ---: | ---: |
| 1K, warmed | 919.441 | 897.474 |
| 10K, two runs | 628.149–632.654 | 653.369–654.330 |
| 32K, two runs | 396.189–403.393 | 448.686–449.329 |

32K TTFT improved from 79.3–80.8 s to 71.2–71.3 s (about 11–13%
throughput improvement). Short-prefill performance is effectively unchanged
in this small sample; this does **not** establish a gain over the original
~900 tok/s short-prefill number. These nonspeculative decode rates also must
not be compared to the original ~22 tok/s MTP profile.

The new index scorer uses BF16 WMMA with FP32 accumulation, keeping the same
per-head ReLU, weighted reduction, scaling and causal masking. It adds no
persistent allocations. Sixteen GPU validation cases passed, including
shuffled physical pages, row tails, causal tails and top-k set comparisons.
Maximum absolute score difference was 7.15e-7; tested top-512 sets matched.
The isolated scorer is 13–17.5x faster at tested 4K/32K history shapes.
This reduction-order change is not a proof of identical top-k for every input.
Early-needle retrieval tests passed at both 10K and 32K. This is a limited
behavioral check, not a comprehensive model-quality evaluation.

## Concurrency work

Exact C2/C3 batched MLA absorption/extraction kernels are behind
`GLM_MLA_BATCH23=1`. All isolated GPU tests were bit-identical to per-row
control, with about 1.54–1.94x improvement for those projections alone.
Short-context full-model concurrency measurements used v2 with binary SHA256
`4916c7c1d162d9cd42bbfde8ba5c6ddecc447da4f4b7458b1d168ca7ba8cf7b7`.

Short-context control: 2048 context, 1024 prefill chunk, three active and
admitted slots, no MTP, `GLM_MLA_BATCH23=0`, `GLM_C3_GROUPED_MOE=0`,
`GLM_MULTI_SEQ_SPARSE=0`. Each request has 1000 prompt and 96 output tokens;
one warm-up batch then two measured batches at each concurrency.

| Concurrency | Control aggregate decode-window tok/s | Control median session tok/s |
| --- | ---: | ---: |
| C1 | 13.738 | 13.596 |
| C2 | 17.395 | 9.138 |
| C3 | 22.129 | 8.004 |

With only `GLM_MLA_BATCH23=1`, C2 measured 17.456 aggregate tok/s and C3
22.505 (two measured batches each). This is a small observed gain, about
0.4% and 1.7%, not evidence that the isolated kernel speedup transfers to
the whole model. Three independent retrieval requests passed for both
profiles, with no foreign needles. Generated suffixes were not identical;
the bit-exact microbenchmark claim applies to the projections, not the
whole scheduling/attention path or generated completions.

Aggregate window is all completion tokens divided by the interval from the
first session's first text to the last session's finish. It includes staggered
prefill and drain overhead, and differs from the sum of server session rates.

With `GLM_MLA_BATCH23=1` plus `GLM_C3_GROUPED_MOE=1`, C3 delivered
**27.366 aggregate decode-window tok/s** (three measured batches:
27.587, 27.366, 27.341), about **24% above control**. Median session rate
was 10.152 tok/s, versus 8.004 control. Warm-up was excluded; the same
1K/96-token workload and short-context configuration were used. The three
independent retrieval requests passed without foreign needles. Captured
server traces showed C3 slots `[0,1,2]` and C2 drain slots `[0,2]`.

The opt-in C3 compact grouped MoE path retains router behavior,
shared expert projections and dense down layout,
but changes routed activations to FP4, so requires quality as well as speed
checks. Do not infer bit-exactness from the MLA results.

Review found that existing multi-row MLA and MTP verification bypass the
semantic index beyond 2048 tokens. The launcher now rejects unsafe long
multi-row configurations unless independent-sequence sparse decoding is
explicitly enabled. `GLM_MULTI_SEQ_SPARSE=1` now provides
that experimental independent C2/C3 path, including index maintenance below
the threshold and graph suppression while host positions are embedded in
launches. Six independent retrieval checks at mixed 2047/2048/2049 prompt
lengths passed over two runs, including threshold crossings and pool tails.
The traces recorded 60 C3 steps and 61 C2 steps on noncontiguous slots
`[0,2]`. Three simultaneous ~10K early-needle requests also passed with no
foreign needles. Long-context MTP remains unsupported.
`scripts/benchmark_glm53_concurrent_niah.py` checks different
needles and output caps, with foreign-needle detection; server traces are
needed to establish actual co-batched step coverage. Long-context tests use
v3, which differs from v2 only by an equivalent clippy-style divisibility
predicate. Both nodes have SHA256
`224476e0b9081b056574b0fed4a9afa27e71ba74968b007fdb91badbd1a87e16`.

Long 10K requests with 1024-token chunks still incur substantial serialized
prefill stalls: observed TTFT was 26.5–71.0 seconds for three simultaneous
requests. These checks establish behavior, not a 24% long-context throughput
claim. Larger bounded chunks and [graph-safe concurrency](graph-safe-concurrency-next.md)
are separate next steps; the graph design is not implemented.

## Short-prefill configuration A/B

The existing launcher option `FP4_PREFILL=1` was missing from the tested
deployment (default remains zero). Same v3 image, TP2/EP2, MTP4 distributed,
fixed K5 compact verifier, native FP4 head, BF16 KV, context1536/chunk1024,
one active/admitted sequence. Excluding one warm-up request per profile,
five uncached 1K-prompt/96-output runs gave:

| FP4 prefill | Median prefill tok/s | Range |
| --- | ---: | ---: |
| off | 923.049 | 912.203–924.002 |
| on | 1049.828 | 1046.773–1054.448 |

That is a 13.7% prefill improvement from an existing configuration lever.
It quantizes activations for the first three dense FFNs; this is not a new
kernel from this campaign. A 1K needle and three arithmetic checks passed
on both profiles. This small check does not establish general quality parity.

For a matched original-settings sanity check, v3 was also run with context512,
chunk64, 128 prompt and 96 output tokens, all new optimization flags off.
Cold decode was 22.168 tok/s, then 23.598 and 27.437 warmed, versus the initial
v65 observation 23.02. This small, acceptance-sensitive sample is not a
single-session speedup claim.

Bounded 256-output requests completed at 30.626 tok/s (off) and 32.379 (on)
on the repetitive 1K prompt. MTP acceptance varies; do not interpret these
as a controlled gain over the original ~22 tok/s number on another prompt.
An earlier 256-cap attempt stopped at113 tokens because of the existing
content-loop watchdog. `benchmark_glm53.py --allow-repetition` now raises
that watchdog threshold for this request only; EOS and output caps remain.
Service-wide watchdogs are unchanged.

## Rejected experiment

`scripts/dev/glm_sparse_mla_warp.cuh` is a standalone warp-per-head sparse
attention experiment, deliberately outside production kernel discovery.
Although bit-exact, its tested variants were about 20% slower on larger
prefill shapes. It is not included in the runtime image.

## Safety and reproducibility

Builds run without GPU access, offline, with 4 GiB memory and two CPU cores.
Runtime containers are capped at 114 GiB; launch retains the 4 GiB free-memory
guard. GPU tests are serialized, with memory/temperature checks. No node resets,
OOMs, or kernel hangs have occurred. Existing remote dirty source is untouched.
The original stopped containers are preserved on both hosts as
`atlas-glm53-v65-rollback-ep0` and `atlas-glm53-v65-rollback-ep1`.

The selected service profile is 16K per sequence, three active/admitted
sessions, 4096-token prefill chunks, and no MTP. It explicitly enables the
WMMA scorer, batched MLA, C3 grouped MoE, independent sparse indexing and
FP4 prefill. Flags remain default-off in the general launcher; activation
precision options remain experimental. The service is at
`http://192.168.8.181:8890`. The final v4 binary on both ranks is
`d880d3826e2e9a41030443b927a3f6f1be53eb48ad5b17ff790675f93794c2ba`.
The persistent head launcher is under
`/home/mangokid/atlas-glm53-deploy-20260906/`. No host reboot, clock change,
swap-policy change or sudo operation was performed. The receipts are archival
observations from explicitly identified binary hashes built before the first
source commit, not clean-tip CI gate records. No git push was performed.

## External comparison

[PixelML's dual-Spark report](https://github.com/PixelML/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark/blob/main/results/APOLLO-2026-08-27.md)
reports 26.55 tok/s median C1 and 1277–1372 uncached prefill with NVFP4,
TP2, MTP4, and FP8 KV. It is a useful target, not a controlled comparison:
KV precision, serving engine and benchmark details differ. SparkGLM's
reported 24.27 tok/s EXL3/DFlash result is C4 aggregate, not C1 decode.

## Local validation notes

CUDA microbenchmarks: `scripts/dev/bench_glm_indexer.cu`,
`scripts/dev/bench_glm_mla_batch.cu`, and `scripts/dev/bench_glm_sparse_mla.cu`.
Compile on GB10 with `nvcc -O3 -arch=sm_121a`.
GPU-free Rust launch-geometry tests passed (7 tests); cargo check, formatting,
kernel-shadow and license checks passed for v1 and v2. All 223 model-layer
CPU tests and seven GLM server preflight tests pass after the final changes.
Native offline release builds on aarch64 also passed, including v4's direct
server MTP context guard. Full clippy is blocked by
existing warnings in `glm5_kda.rs`, `glm5_mtp.rs` and
`forward_prefill_routed.rs`.
