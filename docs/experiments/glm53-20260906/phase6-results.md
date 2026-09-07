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

Subsequent v16 eager resident-model oracle passes all42 MoE layers on each
rank at actual K5, each with positive useful work and both full gate/up
outputs bit-identical. Both numerical diagnostics are separate from timings.
The four strict C1 answers and bounded coding32 check pass. This validates
K5 selection on real resident weights, not C4 selection or a full-model gain;
the graph-on M16 A/B is recorded below.

Graph-on M16/oracles-OFF passes all four strict answers. Matched coding256
full-wall runs24.693,25.160,25.503 yield median25.160, post-first26.277,
all cap-complete. The preceding M16-OFF repaired control was24.751; this small
1.65% endpoint difference is not yet a demonstrated kernel gain. Accepted
draft means2.160,2.266,2.308 differ from that control, while target-forward
windows do not show a clear improvement. Keep the default OFF and separate
numerical validation from performance attribution.

## Predictor EH precision contract

The repaired cache writer and prompt primer use retained BF16 EH, while the
proposal path still used an optional requantized NVFP4 EH copy. Repair keeps
the true-conditioned seed, leaving mixed precision in private history. This
is a numerical approximation, not a demonstrated new token-alignment defect.
Pinned vLLM uses an unquantized EH linear; WO and vocabulary are distinct
precision choices. Isolate EH only, keeping WO/head/M16/repair unchanged.

`GLM_MTP_NVFP4_EH=0` saves the optional18MiB/rank copy (BF16 was already
resident). It passes all four strict answers. Coding148/256 full-wall runs
26.179,27.688,27.258 give median27.258, post-first28.517; all caps complete.
This is8.34% above the preceding matched M16-ON/EH-NVFP4 control25.160.
Mean accepted drafts2.395,2.583,2.534 versus2.160,2.266,2.308. Client hash
instrumentation `c45bf998` now records text identity after the timed window:
all three BF16-EH outputs have SHA256
`12046a6857a2c4411efb58c919331ecc16c7243ae22a1d12d9281857c0903fc2`.
This is text equality, not a token-ID or broad semantic quality claim.
The same target output despite varying acceptance motivates a bounded
proposal-path determinism audit. Goal still open; confirming repeat pending.

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
Measured request mean accepted drafts OFF1.988,1.920,2.160 versus
ON2.071,2.225,2.185. Clean25-step timing windows (excluding the first after
each request boundary) give median target forward111.360/110.765ms and
proposal10.400/10.920ms, OFF/ON respectively. The seven/six retained windows
support acceptance improvement with a small repair cost, not a faster target.

## Actual K5 phase diagnostic

Same v16 repaired policy, `VERIFY_PROFILE=1`, actual verify graphs OFF, both
numerical oracles OFF. Coding32 warmup plus one measured request yields20
complete K5 intervals per rank (11 warmup,9 measured), each42 MoE/EP sets,
34 KDA sets and45 total layers. No partial intervals. Discarding each request's
first interval leaves18 per rank. Median summed phase milliseconds:

| Phase | Rank0 | Rank1 |
|---|---:|---:|
| Routed gate/up |27.635|28.804|
| Routed SiLU/down |13.513|14.475|
| Shared expert |18.254|18.505|
| Router projection |6.984|7.073|
| MoE EP communication |16.116|13.307|
| KDA TP communication |7.271|6.900|
| All45 layers |127.830|127.735|

The layer total excludes final norm/vocabulary/argmax. Do not sum ranks or
independently calculated medians. Communication includes peer waiting, not
pure transport cost. Gate/up includes setup/activation quantization and is
about22% of this eager timeline. Halving the entire phase would improve
layer-only throughput about12–13%; actual B-layout benefit has a lower ceiling.
Synchronization/logging and disabled graphs perturb these numbers: use them
for prioritization, not as graph-on performance or the acceptance workload.

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

Compatibility prototypes `91bc1cda` plus final decode harness `5867d7de`
pass native full-output equality, CPU columns, refreshed graphs, immutable
inputs/weights and guards. M64 covers15/16/17,63/64/65,127/128/129 rows per
expert; BF16-input decode covers widths1/2/3 and shared/remote cases. All three
memchecks (M64, decode `--fmad=false`, decode default FMA) report0 errors.
Explicit device budgets52,884,648 and45,799,520 bytes; models/builders stopped
for GPU gates and timings. Three clean timing pairs show M64 useful-case
median speed ratios1.253–1.339 across the nine geometries. The decode
compatibility is a negative result: direct addressing regresses every case,
and staged addressing regresses most. An all-remote/shared-only case also
regresses despite no tiled reads, motivating compiler/loop-path isolation.
Do not integrate the model-wide weight layout on these mixed results.

The revised split-loop/register-byte decode prototype `cac9764e` resolves
the large regression without changing BF16 inputs, scale arithmetic or
accumulation order. Native full-output/CPU/graph/immutable/guard checks pass
under both FMA policies; both memchecks report0 errors. Three clean timing
pairs: word-load median speed ratios C1=1.04588, C2=1.30100, C3=1.17950;
wide-vector medians1.04477/1.28859/1.20783. C2 and C3 win all24 comparisons
per variant; C1 includes losses (word minimum0.95321, vector0.93010).
Preserve those losses and measure model bootstrap/drains before selection.

## Small exact router and shared tiles

Router `574322d3` plus reference-output hardening `722a64f6`, and shared
M16 `23856fb5`, pass full production-bit equality, independent CPU columns,
refreshed fixed-pointer graphs, immutable inputs/weights and guards. Router
uses2,407,040 explicit device bytes; shared shapes peak10,029,312 bytes.
Router memcheck and both FMA-policy shared memchecks report0 errors. Together
with register decode above, this native gate window has five clean memchecks.

Three clean timing pairs, no model or CPU compiler running: router baseline
90.263/90.235/90.224us versus BN4 77.111/77.613/76.877us, speed ratios
1.1706/1.1626/1.1736. Shared M16 GU median ratios1.14763 at M4 and1.14306
at M5; down1.15517/1.15464. All six projection/pair comparisons improve for
each shape/width. These are warm standalone weights, not endpoint gains.
The shared kernel preserves the original BF16-A/E4M3-B conversion and K32
MMA order; it is not a routed W4A4 replacement. Production integration and
resident-model oracles remain pending, with explicit default-off selection.

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
