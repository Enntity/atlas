# September8 reference check and next implementation boundary

This is source/receipt review, not a matched competing-engine run on our nodes.
Keep the existing148/256 full-wall acceptance workload and safety limits.

The [SparkGLM Atlas archive](https://github.com/Enntity/sparkglm/blob/main/research/atlas/README.md)
identifies its Rust/CUDA line as an archive, not its recommended serving engine.
Its linked [August29 prefill analysis](https://github.com/Enntity/sparkglm/blob/main/results/legacy/2026-08-29-prefill-optimization-campaign/improve-prefill.md)
concerns an EXL3 deployment. Its useful architectural lesson is to eliminate
host-synchronized oversized-expert fallbacks with bounded GPU row tiling.
Its proposed gains and chunk limits are not measurements of our NVFP4 Atlas
binary and do not authorize raising our context or memory limits.

The current [SparkRing profile table](https://github.com/FujitsuPolycom/sparkring#benchmark-results)
reports different model/checkpoint, node-count, context and sampling profiles.
Its GLM native-MTP3 row is four Sparks, not our pair. The documented verifier
captures cover request-batch row counts, and its DFlash2 path uses a separate
predictor. These are architectural references, not directly comparable C1/C4
baselines. We are not adopting a new draft checkpoint or transport here.

The dual-Spark [DFlash2 open-problems report](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/blob/main/docs/OPEN-PROBLEMS.md)
records workload-dependent acceptance, memory-pressure lockups, and failures
that escape cheap startup checks. Preserve both-rank memory reserves and
post-load boundary/cancellation gates. No swap, driver, or clock changes follow
from this report.

## Actionable split

1. Finish bounded native hidden tracing to distinguish changed target
   conditioning from divergence inside the proposer or vocabulary projection.
   Identical hidden hashes do not establish equal private KV.
2. Integrate the stronger B-tile experiment through checked native ownership,
   a4MiB scratch transaction and layout-safe readers. First build an unpublished
   capability; do not expose a loader switch while any fallback interprets the
   bytes as native/transposed. Keep activation precision unchanged.
3. Distributed multi-request speculation remains the major concurrency gap in
   the [pinned vLLM comparison](phase5-vllm-gap.md). It requires request-owned
   proposal/verification state, row-count-compatible expert execution and
   both-rank graph/command agreement, not relaxing the existing C1 guard.

The original-T direct-register shortcut did not earn promotion. The B-tile
microbenchmark also cannot establish60 C4: applying its historical1.299x hot
kernel ratio to the old47.319 C4 baseline would require roughly92% of the entire
cycle to benefit to reach60. This is only Amdahl arithmetic, not a measured
phase share. Use real full-model measurements before assigning a TPS gain.
