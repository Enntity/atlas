# Phase 6 measured checkpoints — throughput goal remains open

Goal: matched LRU coding148/256 full-wall median >=60 tok/s at C4 or >=30
at C1, with cap-complete outputs, quality gates and a confirming repeat.
Do not substitute a post-first-token metric, change the workload, or count
an unqualified diagnostic as completion.

## Normalized MTP handoff

| C1 image | Source | Full-wall runs | Median | Post-first median |
|---|---|---|---:|---:|
| v13 | `2465672f` |18.425,18.516,18.729|18.516|19.104|
| v14 | `014dbe29` |24.813,24.280,21.741|24.280|25.304|
| v15 | `4a5a45d5` |24.457,23.013,24.016|24.016|25.004|

Both used one warmup plus three measured waves, exact coding148/256,
temperature0/seed1 and normal repetition policy. All measured outputs hit256.
v14 fixes GLM's scheduler handoff to use existing final-normalized hidden,
matching its prompt capture/plain generator and upstream GLM MTP contract.
No extra GPU kernel/allocation/synchronization is introduced by that fix.
The median diagnostic gain is31.1%. The third v14 request includes16%
serial tokens; the persistent MTP gate refreshes its serial baseline across
request boundaries. Do not discard that run or compare a later forced-MTP
repair against a gate-on control without a separately matched baseline.

v14 is not quality-qualified: arithmetic returned the correct calculation
only in reasoning, leaving visible content empty. The preserved v13 native
MTP control reproduced the exact24-token arithmetic failure and also failed
the strict JSON case; sort and Python AST checks passed. These are not
introduced by the normalized handoff. The MTP emitter lacked the ordinary
decoder's thinking-only EOS suppression; commit `4a5a45d5` adds it and keeps
output/context ceilings authoritative even through suppressed-EOS returns.
Regression tests: RED0/2 then GREEN2/2. A first test fixture mistakenly
started already finished and was corrected before the meaningful RED run;
all failed logs remain in the raw archive. Full server CPU suite2338 passed,
12 ignored. Native v15 visible-answer checks now pass all four cases:
arithmetic, sorting, Python AST and strict JSON. Its matched coding run also
completed all256-token caps. The persistent gate includes intervening quality
requests, so its phase differs from v14; the small median difference is not
an isolated EOS-fix performance measurement.

A bounded v15 near-context retrieval check passed at1984 prompt tokens with
48 output tokens permitted (16 actually emitted), needle at90% of the prompt.
TTFT2973.690ms and full wall4.002s. It returned the needle twice; this is a
retrieval pass, not a strict answer-format or throughput-target pass.

v15 is the EOS-only snapshot `4a5a45d5`, excluding the subsequent pair/KV
foundations. Correct binary SHA256 on both nodes:
`c9719ef8e0e8640893897e76abd7761ec747a0f556c22a3f6446f9c33ba8eba5`.
The experimental service is C1/context2044+four-lookahead, not the preserved
normal v13 C4 profile. No stable deployment promotion is implied.

## M16 native FP4 expert work

Down prototype `014dbe29` passed full-output production equality, independent
CPU columns, refreshed graph replay, input/weight/guard checks and memcheck
at C4/K5. Six clean useful-map ratios include losses: no down promotion.

Fused gate/up prototype `4c9b0371` passed both full outputs against production
scalar/vecscale kernels, CPU gathered-row columns, zero-input rows, changed
metadata graph replay, immutable weights/inputs and guards. Memcheck found
zero errors at both C4/K5; updated shared-header down regressions also pass.
Peak explicit device bytes38,303,752/38,438,184. All model services were
stopped for standalone GPU work; no model/microbenchmark overlap.

Three clean interleaved C4/K5 timing pairs: all24 useful builder-inclusive
comparisons improve, median speed ratios1.0619 at C4 and1.0364 at K5.
Kernel-only medians1.0494/1.0421 include one small loss each. Empty/remote
inclusive differences <=0.020us. This selects a guarded full-model A/B,
not default promotion. The fixture contains only four distinct local pairs
(36MiB warm weights), synthetic dyadic inputs, and eager event timings;
larger real expert working sets, downstream operations and graph-on model
throughput remain unmeasured. Keep down and all quantization policies fixed.

Production-source extraction/guarded selection committed as `502cda12`.
Recompiled standalone fixtures against that actual production translation
unit: C4/K5 gate/up and unchanged test-only down passed again, including four
zero-error memchecks. The new three-pair gate/up timing receipt has23/24
useful builder-inclusive wins (C4 median1.031298, K5 median1.043205); the
smallest C4 ratio is0.999296. Kernel-only comparisons24/24 win, medians
1.040765/1.045365. Preserve this newer variability rather than treating the
earlier24/24 result as universal. All services and CPU builders were stopped
for these timings. Resident-model selection/oracle and end-to-end A/B remain
pending; the production switch is default-off.

## Accepted-history foundation and safety receipts

`9c54d24c` adds checked GLM pair planning; `8bd85db1` extracts the existing
BF16 KV-only primer into a checked arbitrary-row writer. They are not in
v15 and do not yet repair accepted history. Combined model CPU suite751/751
passes (nine pair-plan and six KV-writer tests added). Both received
independent source review. Runtime integration is a separate candidate.

Runtime repair plus resident BF16 KV diagnostic committed as `14a47186`,
after independent review. Final CPU receipts: model771/771, server2342 passed
with12 ignored, launcher9/9. The server suite requires the known serial/color
test environment; an initial parallel run retained four unrelated TUI color
failures before the corrected complete run. No UI code was changed. Native
repair quality, accepted-count and throughput attribution remain pending.

The v16 binary (`14a47186`) built successfully in2m11s after a separate
3m56s committed-kernel cache-prime build. Both completed without container
OOM; the final Rust build used4GiB. Both nodes' binary SHA256 is
`412a05f3f479fb0629d895268bb4d39014d96d618820dafeee34072f0f8abac0`.
The matched repair-OFF control uses continuous forced MTP, cold prefill-only
context and Atlas swap-space0; it is not the same policy as v15 above.
Its four visible-answer checks all pass, identical to v15. Coding148/256
full-wall runs23.399,22.867,24.797 give median23.399, post-first24.332;
all outputs reached256. Every timed request reports serial0, depthK5 and
no regime re-probes. Mean accepted drafts1.988,1.920,2.160. M16 is OFF.
This is the baseline for repair attribution, not a performance gain.

Repair-ON with the separate resident KV diagnostic passed215 comparisons on
each rank, with identical row/block-count histograms. All widths1..4 occurred;
12 comparisons per rank touched three blocks (reference plus a destination
crossing a block boundary). No cache mismatch or stale-state error was found.
Four strict answers pass; arithmetic also passes immediately after a cancelled
stream is retired. The1984/48 near-context retrieval passes (16 emitted tokens,
TTFT2984.739ms). A tool request is rejected by the bounded-lane admission guard
(HTTP500 with its explicit error), followed by a successful plain completion.
The cancelled client hit its intentional2-second deadline after receiving SSE;
server logs confirm receiver-drop retirement after29 generated tokens.
Diagnostic coding128/128 completes, but its rate is not the matched256-token
acceptance benchmark.

Repair-ON with both numerical diagnostics OFF passes the same four strict
answers. Matched coding148/256, warmup plus three measured waves, gives
full-wall23.945,25.130,24.751: median24.751, post-first25.781. All outputs
reach256 with the ordinary repetition policy. This is5.8% above the matched
repair-OFF23.399 median, not a comparison against the differently gated v15.
M16 and the proposed B layout remain OFF. The30 C1 /60 C4 goal remains open.

## Direct-staged expert B layout, standalone only

Prototype `3f1e916a` permutes packed bytes without changing FP4 arithmetic,
scales or K64 accumulation order. It removes runtime B transposition using
double-buffered MMA-ready tiles. CPU C4/K5 tests exhaustively check all4,194,304
packed-byte indices and invertibility; independent source review passed.
The reviewer found and fixed a timing-harness gap before native testing:
all six outputs, immutable metadata/inputs and canaries are now checked after
timed loops as well as before them. Production weights/dispatch are unchanged.

Native C4/K5 eager/refreshed-graph full-output equality, independent CPU
columns, immutable weights and guards pass; both memchecks report0 errors.
Explicit device peaks36,474,632/36,674,600 bytes. Three clean paired timing
runs win all15 useful cases per width against existing M16. C4 direct and
builder-inclusive median speed ratios1.299195/1.303584; K5 ratios
1.727232/1.414485. The fixture has only two local expert pairs and a warm,
small working set. This is not an end-to-end rate or production-layout
approval. Scalar bootstrap, small-batch fallback and larger-prefill consumers
would all need typed, equal-memory layout support before replacing weights.

The initial v15 CPU build hit its own4GiB container limit while an archive
timestamp change unnecessarily triggered CUTLASS recompilation. Builder
status was `exited101/OOMKilled=true`; model services were stopped and both
hosts retained about118GiB MemAvailable. The unchanged extracted SHA exposed
a stale artifact before any launch. The first v15 tag briefly held that
old binary, was never served, and was replaced after a successful8GiB-bound
retry (`exited0/OOMKilled=false`,4m58s). Builder limit returned to4GiB.
Future source syncs must copy only changed frozen files, preserving unrelated
CUDA/runtime timestamps. Artifact packaging now requires successful builder
status and rejects the old binary hash before creating a candidate image.
No host reset/reboot, host OOM, model-container OOM, swap/clock change or
privilege escalation occurred. This CPU-container failure is retained, not
reported as an entirely failure-free campaign.

Raw receipts: local `/tmp/atlas-glm53-phase6-20260907.J5PkkO/`, periodically
archived under head `/home/mangokid/atlas-glm53-deploy-20260906/phase6/`.
Archive completeness/hash confirmation must be repeated after the last run.
