# GLM concurrency: online reference checkpoint

2026-09-08, after Atlas Gate 1 commit `bd14c289`. The user requested renewed
online investigation if implementation progress was not translating into
throughput. This checkpoint checks the roadmap against primary sources; it
does not select new weights, change precision, or authorize native activation.

## Compare like workloads

SparkGLM now develops a vLLM/EXL3 line and preserves Atlas as an archive.
Research pin: `fef5152ef20159b33c461989284942620d077037`; its vLLM base is
`487ecf187d3dfe74d2cf6119a92881dba403c219`. Its current source configuration
uses EXL3/TR3 and DFlash2, unlike our NVFP4/MTP4 profile.
[Current project scope](https://github.com/Enntity/sparkglm/blob/fef5152ef20159b33c461989284942620d077037/README.md).

The retained headline experiment measured an older engine, `495f27d9`, with
warm counting, 400 forced output tokens and five repetitions. It reports
C1/C2/C4 of 65.16/98.23/138.10 tok/s. Its aggregate sums per-stream decode
rates; the stricter C4 delivery rate is 132.96. Counting accepted 6.843 of
seven drafts per step, whereas its prose C1 accepted 2.655 and ran at 29.29
tok/s. Concurrent acceptance was not captured. These are useful distinct
workloads, not an apples-to-apples measurement of our warm 148/256 benchmark.
[Frozen headline report](https://github.com/Enntity/sparkglm/blob/fef5152ef20159b33c461989284942620d077037/results/legacy/2026-09-03-current-headline-decode/2026-09-03-current-headline-decode.md).

The earlier Mia NVFP4 recipe does combine MTP4 and concurrent requests; that
supports the capability target, although its table does not establish our
exact workload or timing formula.
[Mia NVFP4 recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark#structural-decode-toks).

## Transfer mechanisms, in order

1. **Request-segmented verification.** vLLM's GLM KDA consumes request-specific
   sequence boundaries, state indices and accepted-token counts. Its state
   allocation includes speculative width. It also combines input projections
   and convolutions, but its BF16 projection representation differs from our
   quantized weights. This supports our ownership, serialized two-K5 control,
   then genuinely segmented M10 plan. It does not support passing ten rows to
   an ordinary batch-two or single-history recurrence.
   [GLM KDA source](https://github.com/vllm-project/vllm/blob/487ecf187d3dfe74d2cf6119a92881dba403c219/vllm/models/glm5next/nvidia/kda.py).
2. **Work-conserving request scheduling.** The retained matched long-prompt
   C4 comparison reports a 44.2% aggregate improvement from mixed admission
   versus strict skipping, with two measured repetitions. That includes
   queueing and prefill effects, not a 44.2% decode-kernel acceleration.
   After bounded C2 correctness, measure staggered arrivals, admission stalls
   and per-request progress separately from simultaneous short-prompt decode.
   [Results map](https://github.com/Enntity/sparkglm/blob/fef5152ef20159b33c461989284942620d077037/docs/RESULTS.md).
3. **GPU-side expert work planning.** Study counts/offsets, compact work lists
   and phase-wide launches in the EXL3 implementation when profiling our
   resident B-tile and future M10 paths. Transfer scheduling principles, not
   EXL3 arithmetic or its scratch assumptions into NVFP4.
   [Serving implementation](https://github.com/Enntity/sparkglm/blob/fef5152ef20159b33c461989284942620d077037/overlay/exl3.py).

## Decision

Continue accepted-row ownership and repair, then the native serialized C2
control and segmented verifier. No additional standalone kernel sweep is
justified before those paths can consume the candidates already qualified.
Measure complete verify/propose/collective time and per-request acceptance;
keep optimized C1 and nonspeculative C1/C2/C3/C4 as separate regression profiles.

An instructive rejected candidate was FlashKDA: roughly 3.5x faster in its
isolated operator test, yet slower at the medium-C2 endpoint. This does not
prove FlashKDA is intrinsically unsuitable; it shows why another microbenchmark
win alone would not resolve our current bottleneck.
[Rejected integration](https://github.com/Enntity/sparkglm/blob/fef5152ef20159b33c461989284942620d077037/results/legacy/2026-09-03-flashkda-candidate/2026-09-03-flashkda-candidate.md).
