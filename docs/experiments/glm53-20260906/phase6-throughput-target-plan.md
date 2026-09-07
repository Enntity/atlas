# Phase 6: throughput target and measured implementation selection

User target (2026-09-07): keep working until at least 60 aggregate tok/s at C4
or 30 tok/s at C1. Start from deployed v13, source `2465672f`, and the committed
workspace `e1dc1046`. Do not trade away numerical correctness or host safety.

## Acceptance and safety

- Use the existing LRU completion fixture: 148 prompt tokens, 256 requested
  output tokens, temperature0/seed1, unchanged repetition policy. Require all
  measured outputs to reach cap, not just a faster early stop. Report full-wall
  aggregate and post-first-token rates separately. Target full-wall median,
  one warmup plus at least three measured waves, with a confirming repeat.
- C4 baseline is47.319 full-wall aggregate tok/s in v13. C1 native MTP needs
  a fresh matched baseline; the user's historic22 tok/s is not this exact test.
- Keep context2048, container114GiB, memory guard4096MiB and at least4GiB host
  MemAvailable on each rank after load/warmup. No host resets, swap/clock changes
  or competing model/microbenchmark while the service is resident.
- Preserve v13 containers before profiling/experimental launches. Root owns
  all node operations. Independent agents audit/code locally, never launch
  concurrent GPU work. Standalone kernel fixtures remain below64MiB explicit
  device allocations and require oracle/canary/memcheck gates before timing.

## First measured decision

1. Read existing source and historical failures; compare current primary
   vLLM/SGLang implementations, distinguishing their topology/precision/MTP
   configuration from Atlas. Do not repeat the rejected value32 KDA or compact
   down experiment without a new measured hypothesis.
2. Profile v13 C4 with BOTH ranks `MS_PROFILE=1 KDA_MS_PROFILE=1 PROFILE=0`,
   preserving all other short-profile settings. These switches force eager
   execution and synchronized timers. Two bounded32-output waves on the coding
   fixture provide a coarse whole-block/mixer/FFN/head ranking, NOT a breakdown
   of graph-enabled production latency or a qualified throughput result.
3. Audit graph-on timeline instrumentation and native C1 MTP independently.
   Account for command/control, sampling, draft, verifier, rollback and
   communication costs; do not infer all such time from layer timers.
4. Select one major implementation chunk from measured costs, write its exact
   test-first numerical/state plan, implement a bounded prototype and review it
   independently. The pure rank-binding roadmap is available, but unused
   scaffolding alone cannot satisfy this throughput objective.
5. Only promote after complete output/state gates, safe full-model answer and
   boundary/cancellation checks, and matched unprofiled performance validation.
   Retain failed tests and negative results; do not lower benchmark thresholds.

Raw phase6 receipts: `/tmp/atlas-glm53-phase6-20260907.J5PkkO/`, to be archived
under the head's persistent campaign directory. The goal remains open until a
reproducible target is actually achieved or a genuine external blocker recurs.

## First C4 profile and selected prototypes

The synchronized eager diagnostic yielded60 N4 steps across the two bounded
waves. Last16 head medians:79.612ms summed whole-block/head time,34 KDA
blocks56.432ms,11 MLA blocks21.634ms, final head1.4995ms. KDA mixer/FFN
timers across those34 layers average21.162/35.077ms per step. Worker reports
79.0625ms total and35.002ms KDA FFN. These are coarse instrumented costs.
They support investigating useful expert work, not treating the empty-CTA
cost from phase5 as recoverable model latency.

Kernel prototype selected: retain128 loader threads but repartition four warps
over N32 slices of an M16×N128×K64 tile, sharing the16-row A tile. This avoids
the historical M32/block64 loader-participation regression and keeps K64
accumulation/layout/precision. Standalone down first, then assess fused gate/up
applicability only after complete numerical and speed gates.

An independent MTP source audit checks whether accepted draft KV is refreshed
using verified target hidden rows, as in the reference proposer, or merely
trimmed. It is a hypothesis until call-site and alignment review confirms it.

Fresh C1 control uses the same v13 image/weights and coding workload, existing
distributed four-draft MTP, compact fused K5 gate/up, and existing graph path.
Use one admitted/active request, context2044 plus four lookahead positions
(the existing <=2048 verifier bound), chunk1024 and the same memory ceilings.
The non-synchronizing MTP phase ledger is diagnostic; detailed synchronized
profilers stay off. No unsafe batch/speculative guard is relaxed. This is a
baseline for a substantive MTP correction, not a claimed new flag optimization.

## Measured checkpoints (goal still open)

Fresh v13 C1 baseline, one warmup plus three coding148/256 measurements:
full-wall18.425/18.516/18.729 tok/s, median18.516; median post-first19.104.
All outputs reached256. Measured request acceptance reports first-draft
probability0.827/0.835/0.832 and mean accepted drafts1.318/1.367/1.374.
The non-synchronizing ledger reports roughly110–114ms target verification
and10.6–10.8ms proposal, excluding first/cross-request ledger intervals.

Two independent reviews confirmed GLM scheduler MTP consumed raw target
hidden while prompt priming, the plain generator and upstream use final-norm
target hidden. Commit `a35ad139` fixes only that representation choice;
other model families retain raw input. Four transfer-contract tests and the
full736-test model CPU suite passed. v14 source snapshot is `014dbe29`
(the second commit only adds standalone kernel tests); native binary SHA256
`3c1ddb924b0d76f6438d1231d67bbfbd8d490e3619afaf8c0f9c2ade59eb57fb`
matches both nodes. Live acceptance/throughput validation is pending here.

v14 matched coding measurement subsequently completed:24.813/24.280/21.741
full-wall tok/s, median24.280 (+31.1% versus v13's18.516); median
post-first25.304. All three outputs reached256. First two measured requests
report mean accepted drafts2.173/2.096; the third1.880 and16% serial tokens.
The process-persistent throughput gate periodically probes serial decode;
its default1024-token refresh can cross request boundaries. This is not
a kernel-latency regression and is not silently excluded from the median.
An always-speculative lane needed for initial accepted-history repair will
require a separately matched control; do not attribute gate-only gains to
that repair.

v14 is **not quality-qualified**: both an initial zero-thinking-budget
arithmetic smoke and the established explicit32-budget arithmetic case
returned empty visible content, with the correct calculation only in
reasoning. The MTP emitter's source predates this image and omits the
ordinary decoder's thinking-only EOS suppression. See the separate
`mtp-thinking-eos-plan.md`; a control reproduction and reviewed fix are
pending. The coding throughput result is a diagnostic result, not promotion.

M16 expert-down prototype `014dbe29` compiled natively for C4 and K5.
Each passed full-output bit equality against production, independent CPU
sample columns, eager and changed-metadata graph replay, input/weight
immutability, allocation guards and Compute Sanitizer memcheck (zero errors).
Explicit allocations38,590,472/38,798,344 bytes. Both model services were
stopped; no model and standalone GPU workload overlapped.

Initial three paired timing runs overlapped runtime-image packaging (not
Rust/nvcc compilation), so a further three clean C4/K5 pairs were collected.
Clean useful-map dense/M16 speed ratios span0.9874–1.0756 at C4 and
0.9949–1.0778 at K5: modest median benefit, not a consistent win. Empty and
remote-only maps regress from roughly14.4us to16.4us. Do not promote this
down kernel on these receipts, or claim a whole-model gain. A separate
bounded fused gate/up experiment may test the same arithmetic partition
with the real token gather and projection multiplexing.
