# Phase 5: measure the vLLM gap and build execution-plan contracts

2026-09-06–07. v13 lifecycle fixes pass live correctness gates with identical
resource/kernel settings; earlier baselines are preserved. No competing engine
is installed or loaded alongside it. This is a continuation of the
[EP execution-plan design](ep-mixed-execution-plan.md).

## Work plan

1. Recheck published primary receipts and their exact benchmark request. The
   public C4 native-MTP medians are 56.00 and 58.36 full-wall tokens/s, but our
   v11 value 47.309 uses a different LRU completion prompt. Those values are
   not a controlled engine comparison.
2. Implement a bounded chat reference client, tests first. Use the public
   topological-order prompt, temperature0, reasoning effort low, 256-output
   cap, and first content/reasoning/tool event timing. Reuse Atlas's tested
   aggregate metric code; separately include thread-pool overhead in the
   reference-style wave wall. Record actual token counts, early stops,
   prompt-message hash, payload and per-request receipts. Do not execute
   generated code or change service-wide watchdogs.
3. Atlas has no `ignore_eos` chat request field in this checkout. Do not send
   that unknown field and assume it works. Record the EOS-policy mismatch;
   only cap-complete runs qualify as a fixed-output comparison. Keep memory
   safeguards and C1/C2/C4 only. Check both nodes before live requests.
4. In parallel implement the first pure typed intent/validated sequential
   plan slice, with no scheduler route, protocol bytes, or GPU changes.
   Cover budgets, full-prompt versus scheduled-token spans, unique logical
   request slots, ordering, context checks and result-consumption obligations.
   Later rank-local binding must verify actual state/generations separately.
5. Independently review tests/contracts, run CPU gates, commit the bounded
   slice and record findings. Do not claim a throughput gain from an unused
   planning contract. GPU rollout requires the later rank-agreement gates.

## Pinned benchmark reference

PixelML repository HEAD resolved on this date:
`3407023e0b8109a1dd12e8a5544e106ca6912afe`.
[Benchmark request and client metrics](https://github.com/PixelML/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark/blob/3407023e0b8109a1dd12e8a5544e106ca6912afe/benchmark.py),
[original receipt](https://github.com/PixelML/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark/blob/3407023e0b8109a1dd12e8a5544e106ca6912afe/results/APOLLO-2026-08-27.md),
[fresh revalidation](https://github.com/PixelML/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark/blob/3407023e0b8109a1dd12e8a5544e106ca6912afe/results/APOLLO-2026-08-27-REVALIDATION.md).

That deployment uses vLLM `487ecf187`, a patched SM121 runtime, TP2/Ray,
Marlin weight-only routed experts, FP8 KV and native MTP4. Official vLLM
`6865e67f0be02d53694517f6f71d7fb96492792d` remains the architectural reference,
not the binary behind those public measurements. No external implementation
source is copied; only the short benchmark prompt is reused with attribution.

## What remains different

- **Distributed multi-request speculation:** the public recipe enables native
  MTP4 with eight admitted slots. Atlas's GLM-specific launcher, preflight,
  proposer and worker enforce C1. Generic Atlas batched verification exists,
  but requires no communicator and K2–K4; this does not cover GLM EP2/K5.
  With four draft lookahead tokens, the present exact-attention configuration
  must satisfy context+4 <=2048. The public receipts do not prove the target
  executes C7×K5 rows simultaneously; configuration is not a verifier trace.
- **EP mixed/prefill execution:** Atlas's grouped/mixed scheduler excludes EP,
  and its fused model path also excludes distributed communication or MLA.
  Each prefill chunk currently sends the entire prompt again: uncached16K
  with1K chunks transfers1MiB of token payload rather than64KiB once, using
  96 synchronous v2 broadcasts. That is not evidence of network bandwidth
  saturation; repeated staging/synchronization must be profiled separately.
- **Expert topology and kernels:** Atlas EP2 owns144 complete experts/rank.
  The public Marlin TP2 recipe partitions every expert's intermediate width.
  These have different shapes, load balancing and backend eligibility. Atlas
  already fuses compact gate/up at C3/C4; its down projection still launches
  a dense expert grid (9216 CTAs at the reviewed C4 shape, many exiting empty
  or remote) and uses a staged activation copy to avoid real aliasing. An
  empty CTA count is not a timing measurement. The previously rejected
  compact-down path must not be re-enabled without explaining its failure.
- **Cache and admission:** FP8 versus BF16 KV, single latent ownership versus
  duplicate K/V, and bounded versus full raw index tails differ. Typed cache
  geometry is now implemented; storage reduction is not. C6/C7, long-context
  admission, and concurrent speculative state need their own capacity audit.

[Pinned vLLM expert topology](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/model_executor/layers/fused_moe/config.py),
[NVFP4 backend eligibility](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py).
Some attractive SM12x backends reject EP or are excluded from automatic SM121
selection at this pin. They are not drop-in explanations of the public result.

For steady short-request throughput, concurrent speculation and generalized
row-count expert execution remain major capability targets. For long/mixed
traffic, the EP work/state contract and packed prefill are separate targets.
The sequential plan alone neither shares weights nor promises higher TPS.

## Exact public chat prompt exposes a lifecycle defect

New Atlas-owned client `scripts/benchmark_glm53_reference_chat.py` reuses the
short public chat prompt, low reasoning, temperature0, max256 and reference-style
wave wall, while explicitly recording unsupported ignore-EOS behavior. Six
client CPU tests and independent review passed. Host memory before this run
was10542/10601MiB available on head/worker; runtime configuration stayed v11.

One warmup and three measured batches per width, no overlapping GPU workload:

| Width | v11 median reference-style full-wall tokens/s | Actual completion counts per stream |
| --- | ---: | --- |
| C1 | 13.255 | 125 |
| C2 | 18.572 | 125 |
| C4 | 46.073 | 181–255 |

All requests have28 prompt tokens and API `finish_reason=length`. **None of
these measured waves reaches256 tokens in every stream**, so these numbers
cannot replace the cap-complete LRU benchmark or establish a controlled
comparison with public56.00–58.36 C4. The length label is not proof of budget
exhaustion: Atlas deliberately uses it for named stream-guard stops too.

Logs show the chat SimHash guard firing. A fresh raw SSE diagnostic with
`return_token_ids=true` captures first token154842 (`</think>`), two generated
code blocks separated by a second closing marker, and125 reported completion
tokens with62 incorrectly attributed to reasoning despite empty reasoning
text. Detokenizing the124 returned IDs confirms generated duplication rather
than replay of a complete parser buffer. The wire also drops punctuation from
the guard-triggering token but continues subsequent content.

Root cause is host lifecycle, not evidence of a numerical KDA failure:

1. Prefill constructs `inside_thinking` from the request flag without consuming
   the first closing marker. This can suppress later EOS and misclassify code.
2. Non-speculative token commit lacks the cancellation check present in the
   speculative emit path; a stream guard does not promptly retire the request.
3. Terminal stream handling still emits subsequent sanitized token text.

See [first-token correction](first-token-thinking-fix.md) and
[cancellation correction](non-spec-cancel-fix.md). These are scoped fixes with
tests, not watchdog disabling. The saved-stream regression validator parses
Python syntax but never executes or grades generated code; it correctly fails
the v11 baseline. The rebuilt v12 passes this same saved-stream gate: one
closing marker, one syntactically valid function, normal stop at63 reported
tokens, zero reasoning tokens. This structural gate does not grade the
function's algorithm. Shorter output is not a kernel-throughput improvement.

Receipts: `/tmp/atlas-glm53-phase5-20260907.gu145h/v11-reference-*` and
`v11-first-think-regression.json`. The long-idle NCCL warning recurred after
10749.8 seconds of successful command wait; its separate known classification
issue remains documented and is not cleared through reconnect.

## Infrastructure and kernel work completed this stage

Commit `2463786b`: pure `ScheduledIntent` → immutable `ValidatedStepPlan`,
checked context/payload/scheduled/arena budgets, distinct full-prompt and chunk
spans, unique logical slots, and ordered result-consumption/normalization
obligations. Nine focused tests and all721 model CPU tests pass; independent
reviews found no blocker. No scheduler route, wire codec, rank-state binding
or GPU execution uses it yet. That next stage must validate live identities,
generations, local SSM/KV ownership and rank agreement before any dispatch.

The Atlas-native value32 KDA experiment passes all91 full-state/output cases
and memcheck, but repeated timing does not improve on current indexed KDA.
It is **not promoted**. All three paired runs and compiler/resource details
are retained in [the experiment](../../../scripts/dev/glm_kda_value_tile_plan.md).
Both models were stopped for the38.9MiB GPU fixture, then the unchanged v11
containers were restarted. No GPU faults, OOM kills or node resets occurred.

## v12 lifecycle rollout and test provenance

Source `0c625fad`, image `atlas-glm53-flash:kernel-20260907-v12`, binary SHA256
on both ranks:
`d06eff5e8dadf20c84b5bb14eb5ea7c75cf201bfddf3c221235306e6ec2d0054`.
The native CPU-only build completed in2m11s with the existing4GiB/two-CPU
builder. Both v11 containers were gracefully stopped and preserved as
`atlas-glm53-v11-short-control-ep0/1`; neither was OOM-killed. v12 uses the
same context2048, chunk1024, C4 graphs, nonspeculative profile. No kernel,
watchdog threshold, memory budget or host setting changes in this rollout.

CPU gates: formatting, kernel shadow structure and license headers pass;
all721 model tests,250 scheduler tests,50 chat-stream tests and15 Python
client/validator tests pass. The full server run initially yielded2327 pass,
four TUI failures and12 ignored. Three color assertions fail with inherited
`NO_COLOR=1`; the fourth compares render snapshots contaminated by concurrent
global log-ring test fixtures. The same server binary with `NO_COLOR` unset,
`COLORTERM=truecolor` and `--test-threads=1` passes2331 tests, with12 ignored.
Both original and controlled logs are retained. This is not a full GPU/model
serve matrix, clippy qualification or multi-request speculative validation.

Live first-token regression and four strict concurrent answer checks pass.
Both mixed-cap, four-request needle batches pass at prompt lengths
1900/1920/1950/1984 and output caps96/64/48/64, with no foreign needles.
Decode traces include N3 slots `[0,1,3]` and N2 `[0,3]`, with graph captures.

Matched fixed-output results (one warmup plus two measured batches per width):

| Workload | v11 full-wall / post-first tok/s | v12 full-wall / post-first tok/s |
| --- | ---: | ---: |
| Coding148/256, C3 | 34.890 / 35.472 | 34.936 / 35.531 |
| Coding148/256, C4 | 47.309 / 48.123 | 47.325 / 48.159 |
| Synthetic1024/64, C4 | 28.886 / 31.989 | 28.822 / 31.883 |

All measured requests reach their output caps. Same workload hashes and
per-request repetition policy as phase4; no concurrent node build or GPU job.
These small differences show no observed throughput regression, not a new
speedup. No change to the vLLM capability gap follows from these results.

Explicit-stop probing preserves the code prefix and omits `return` and all
following text, finishing with `stop`. A separate repeated-sentence probe
does trigger SimHash. Scheduler completion follows its warning by72.896ms,
but Done finalization still emits a buffered `The lantern` fragment. Thus v12
fixes prompt scheduler cancellation but does **not** pass the complete
terminal-stream contract. The new [Done correction](terminal-done-fix.md)
must preserve safe explicit-stop prefixes while discarding guard-rejected
buffers; the v13 validation below covers that correction. Guard finish
reason `length` remains intentional and is not a claim of cap completion.

The next CPU codec slice follows
[the broadcast-boundary and rank-binding plan](ep-execution-codec-plan.md).
Legacy v2 currently couples wire IDs to equal local SSM slots and carries no
generation/session identity. A pure codec must not claim new wire protection
or install a unilateral validation collective into that existing protocol.

## Reference-prompt outcome after the first-token fix

The v12 reference rerun uses the identical request/prompt and one warmup plus
three measured waves at each width. All 28 streams, including warmups, stop
normally and contain one syntactically valid function; none of the generated
code is executed or algorithmically graded. C1/C2 outputs are 63 tokens;
C4 outputs are 91 or 105 tokens. Reasoning/response text and usage are retained.

| Width | v11 median wave duration | v12 median wave duration | v12 median full-wall tok/s |
| --- | ---: | ---: | ---: |
| C1 | 9.430 s | 4.841 s | 13.015 |
| C2 | 13.461 s | 7.047 s | 17.880 |
| C4 | 20.529 s | 9.784 s | 40.067 |

The shorter completed responses avoid v11's duplicated code and guard stops;
this is useful response-latency improvement, not faster token generation.
Natural EOS still differs from the public forced-256-token benchmark. Do not
present these variable-output rates as an apples-to-apples vLLM comparison.

Commit `79c01f03` now implements the bounded CPU legacy codec: borrowed payloads,
exact scalar/bulk boundaries, staged header bounds before payload parsing,
reserved-token rejection and ordered compute/normalize/consume obligations.
All 11 focused tests and 732 model CPU tests pass, with independent review.
No production call site consumes it, and it is not in the v13 binary. Live
rank-state binding, generation authority, agreement and scheduler integration
remain separate required stages.

## v13 finalization correction

Source `2465672f`, image `atlas-glm53-flash:kernel-20260907-v13`, binary SHA256:
`06a77b12a0f920d6823ce5a7466b2fb4a4ebf1ddcd672b80dc370461c1d00ac1`.
Only server source changed for this rebuild (native build 1m20s); the later
codec commit is not included. Both v12 containers were stopped cleanly and
preserved as `atlas-glm53-v12-terminal-control-ep0/1` before v13 startup.

The actual Done-finalization core now gates pending-output flushing before and
after the existing callback. Guard-rejected content/tool/refusal deltas and
pending IDs cannot reappear during Finish. Genuine explicit-stop prefixes and
normal EOS tails remain accepted. Scheduler usage and finish-reason precedence
are unchanged. Five new regressions reproduce four failures before the fix,
then pass; all 55 chat-stream tests and seven stream-guard tests pass.
The full server binary passes 2,336 tests with 12 ignored using the repository
working directory, color-enabled environment and one test thread. A mistakenly
launched outside-repository run is also retained, not counted as a passing gate.

Live v13 gates:

- First-token saved-stream regression passes at 63 tokens, normal stop, zero
  false reasoning tokens, one structurally valid function.
- Exact v12 explicit-stop payload returns the same safe code prefix and `stop`.
- Exact v12 semantic-repeat payload triggers SimHash at00:51:20.253154 and
  scheduler Done at00:51:20.326820, a73.666ms interval. The previous extra
  `The lantern` finalization fragment is absent; only usage/finish follows the
  prior accepted content. Wire `length` remains the named-guard policy.
- Four strict concurrent answers and both mixed-cap four-request boundary
  batches pass, with no foreign needles. These remain bounded checks, not a
  comprehensive quality or tool-use evaluation.

Final v13 C4 coding148/256: one warmup plus two measured batches, all eight
measured responses reach256 tokens. Full-wall aggregate is47.319 tok/s
(batches47.278/47.361), post-first-token48.130 and median session12.447.
The v11 control is47.309 full-wall: effectively unchanged, not a speedup.
Receipt: `v13-coding-256.json`. No build or other GPU workload overlaps it.
All four post-benchmark answer checks also pass. Final readiness and matching
binary hashes were checked on both ranks; host available memory was11,607MiB
on head and9,830MiB on worker, with unchanged114GiB container ceilings and
worker's preexisting98MiB swap usage. Temperatures were49/54°C.

Worker logs again classify a52.6-second successful idle command wait as an
unhealthy NCCL broadcast at00:54:32.771588; subsequent answer checks pass.
This is the previously documented idle classifier issue, not a clean health
log. No unilateral reconnect/reset is used to clear it. Independent CUDA or
transport errors would still require investigation.

## Measured MoE-down cost, not another flag change

The [standalone down-cost harness](../../../scripts/dev/glm_moe_down_cost_plan.md)
now isolates empty-expert grid overhead, useful FP4 MMA, and compact worklist
construction. The 38,590,472-byte fixture passed full-output comparisons,
sampled independent CPU arithmetic, graph metadata refresh, remote poison,
immutability/canaries and memcheck (zero errors), while both model containers
were stopped. No production path, precision or memory reserve changed.

Across all three paired useful-work runs, dense288 measured173.605–184.695µs,
while builder+compact measured218.970–223.914µs. Trimming the trailing empty
experts did not beat dense288, and the builder alone costs about31µs. The
empty-only grid cost is real, but is not an additive estimate of full-model
savings. Compact down is **not promoted**, and this isolated experiment does
not establish the cause of the historical integration stall.

## Next gates and retained evidence

1. Bind validated intents to immutable live rank snapshots: actual positions,
   request lifetimes, legacy SSM slot identity, KV/state capacity and prompt
   revision. Model both-rank accept/reject and logits-consumption obligations
   in CPU tests before any scheduler or communication adapter.
2. Profile representative expert distributions and end-to-end operation time
   before another down-path rewrite. The synthetic eight-expert result does
   not justify production compact dispatch or a projected token/s gain.
3. Distributed multi-request MTP, generalized expert row counts, and reduced
   KV ownership remain the substantial vLLM capability gaps. None is unlocked
   by relaxing admission or speculative guards; each needs explicit capacity,
   temporal-state and collective-order validation.

Raw phase5 receipts are in `/tmp/atlas-glm53-phase5-20260907.gu145h/`, with a
persistent copy under `/home/mangokid/atlas-glm53-deploy-20260906/phase5/` on
the head after archival. Historical phase3 receipts retain the v10/v11 matched
controls. No GPU fault, OOM kill, host reset, clock/swap-policy change or sudo
operation was observed/performed during these bounded experiments. The known
command-idle health classifier remains separate and unfixed.
