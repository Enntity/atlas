# Release qualification: September 9–10

Deadline: September 10, 02:42 UTC. The first merged C8 native engine source is
`a6cfeec0`, merged with upstream `6c5f17dab9c27ee2396aef1ac2501a17b201c715`.
Latest tested release source is `1cb0267e`: its4K ON C1 tool gate failed.
The16K ON C1 main quality passed, but corrected chat reuse failed the full suite.
The earlier `c853bafa` results below are not evidence for that newer image.
Portable C8 harness
and its followup-envelope correction are `708db27c` and `440526f2`.

## First merged immutable native build (`a6cfeec0`)

Build completed with exit0 and OOM=false (6m29s server, 6.58s helpers).
All212 GLM CUDA kernels were compiled; no skipped CUDA build.

| Artifact | SHA256 |
|---|---|
| Source build archive | `77c785741e67b838312fa824a91cf99ebd576ab809077984ae4226585cc41166` |
| Server ELF | `9892e1a88932dea0ba4d0097bdf2b05ebef95f3599b9098f7f8019ed943523e9` |
| Head image | `7c3d3c18923d84a1cd95366cc64122e1dd5ef93703e5488ea5cb2f180fe0e60a` |
| Worker image | `f1725e96a928072bee3e38e9bc5d1f68fc79e2c3e68a25d9c203ab620e9ce872` |
| Guard | `9622492cbc51448f26591f56399f0b8c3f3d981bfd2cb49978ccc46cb759d0ff` |
| Node supervisor | `b0613a45f96de1caccbf1fa843cf1fa2276f30355de7e74102dd964cf561a9a0` |
| Node relay | `2ea3f845cfb240f854e83e92f743e405017ce09faef1480f557af7953c34300a` |

Evidence below is retained outside Git in
`atlas-campaigns/20260909/glm-native-controller`. Failed runs remain failures.

## Findings before final qualification

1. Initial C8 startup: worker budget818 KV blocks was below8×128 demand.
   The selected startup refused; the controller stopped both containers
   (exit137, OOM=false, swap0). This is **not** a clean quiescent serving exit.
   The unused16-slot Marconi prefix snapshot reservation was then explicitly
   disabled with `--ssm-cache-slots=0 --ssm-checkpoint-interval=0`. Selected
   prefix reuse is unsupported and context2044 cannot reach its4096-token
   checkpoint. Live state/MTP snapshots, CUDA reserve and watchdogs remain.
   The new profile allocated4333/4180 target blocks and admitted eight owners;
   both ranks logged actual E8 commits at widths8/7/6/5.
2. New C8 tool-result roundtrips: all8 initial answers and8 tool calls passed,
   but4 followups hit their64-token cap during reasoning. The shared thinking
   policy has a soft32-token budget with sentence-safe deferral up to roughly96,
   including on selected MTP. Correct own results appeared in reasoning; this
   alone is not a visible-answer PASS or evidence of a hidden-state failure.
   The new followup allowance is128; exact visible results and normal stops
   are still required. Coding requests remain148 input/256 output, unchanged.
3. First long-context launcher attempt refused Docker's canonical `CAP_...`
   capability names before either container started. The exact checker was
   corrected against both retained real create observations; missing/extra
   capabilities remain rejected.
4. First actual4K ordinary run: early needle was exact and stopped; middle
   needle was correct but repeated until the content-loop watchdog stopped it.
   The bare-completion probe **failed**. Both model processes subsequently
   exited0, OOM=false, swap0. An interim `/tokenize(messages)` proposal was
   superseded by actual real-chat requests and measured one-token calibration
   because base/OpenAI templates differ. Later results below use that corrected
   framing; exact answer/stop checks and watchdogs remain unchanged.

## Performance interpretation

The4K raw run's first1028-token chunk took1.064s (warm0.949s), while later1024
chunks took2.46–2.57s. Overall3968-token TTFT was8.34s, not a comparable1K
benchmark. Later paged GLM attention still uses the scalar BF16 GEMM for four
projections; the same-image opt-in existing-cuBLAS results are recorded below.

Pre-merge C1 coding used65 verification rounds per256 output tokens; initial
merged traces use80. The merged full argmax changes equal-BF16-logit tie order.
This can alter target/draft trajectories and acceptance. The extra rounds are
consistent with much of the observed rate reduction, but proving the cause
requires actual shared-prefix tied logits, not timing alone. Distributed shard
argmax is OFF in both compared recipes, so its host merge is not the active
cause here. Do not present old C1–C4 rates as current merged-image results.

## First merged C8 qualification, 22:22 UTC

Immutable engine `a6cfeec0`, explicit unused prefix slots/checkpoint0, selected
capacity8, context2044, MTP4: all32 checks passed (8 distinct answers,8 auto
tool calls,8 actual-call-ID result followups,8 own/no-foreign needle checks).
One warmup and three measured148-input/256-output batches at each width:

| C | Aggregate full-wall tok/s | Aggregate decode-window tok/s | Client TTFT ms |
|---|---:|---:|---:|
| 1 |22.588|23.538|457.388|
| 2 |30.675|31.539|668.480|
| 3 |35.093|35.839|884.276|
| 4 |37.891|38.541|1091.515|
| 5 |39.719|40.292|1320.664|
| 6 |41.071|41.583|1525.539|
| 7 |42.253|42.711|1733.500|
| 8 |43.087|43.506|1946.026|

All144 coding completions reached256 tokens and were byte-identical within
this merged run: SHA256 `9fea649abf50e69cd3a3860967b93849b0a08a3b42a29acc9b0e65df792350cd`.
They differ from the40 pre-merge completions, SHA256
`d08c6d60eafcb87b86176d8cfd7402e294097f168a9e6cf0356a9b184f24d4c1`,
starting after the first four spaces (`__slots__` versus a docstring). Coding
output is retained, not executed or claimed functionally complete. This is
a performance regression versus the earlier C4-capacity image, not an
equal-output A/B or a reference-performance PASS.

Both ranks completed real paired quiescence/release and independently observed
exit0. Minimum sampled MemAvailable was10,178,056/9,610,940 KiB; swap0 on both.
Evidence: `native-prepared-a6cfeec0-c8-owner-owners-joint-followup128` and
`native-summary-a6cfeec0-c8-owner-joint-followup128.json`. Fresh-process repeat
remains pending.

## Bounded prefill candidate

Source `22b56144` adds explicit `ATLAS_GLM_PAGED_PREFILL_BF16_GEMM=0|1`,
absentOFF, GLM-only. Q_A/Q_B/KV_A/O in later paged prefill can use the existing
BF16 cuBLAS helper (existing TC/scalar fallback); literal scalar control,
precision, attention selection and cache lifecycle remain unchanged. Actual
constructor/dispatch test failed before the selected branch, then passes on
both ranks including non-GLM control; parser check passes. These are host
dispatch checks, not numerical equivalence. Native same-image OFF/ON quality
and timing are required before enabling the candidate in a tested profile.

The revised long-context probe must use the real chat API, not assume
`/tokenize(messages)` uses the same template: base and OpenAI Jinja environments
differ. Real one-token sizing probes, explicit thinking budget16 and cap128
replace that unqualified assumption. Strict visible-answer/normal-stop checks
remain. Full-context quality and maximum safe context are still pending.

## New-image 4K scalar control failure

Engine `22b56144`, OFF profile and portable real-chat driver `63eb6075...`:
C1 and C2 pass all six quality waves; C3 passes all three needle positions
and linked facts. C3 auto-tool fails: NEBULA repeats archive filler and is
stopped by the content-loop watchdog after49 tokens, with no tool call. The
other two calls are correct. C4 and boundary/cancellation checks were not
issued after this failure. Both containers exited0, OOM=false, swap0; the
campaign remains **failed**, not qualified by its clean shutdown.

Exact failed request body matches the earlier successful C2 request. The
successful prefill used chunk ends1028/2052/3076/3485 and first token785;
the failed one used1024/2048/3072/3485 and first token198, before its first
C3 decode batch. Source inspection shows idle initial requests borrow the
1028-row arena despite the declared1024-token sparse-prefill budget; busy
requests use1024. Ending2048 chooses dense attention for that whole chunk,
whereas ending2052 chooses sparse. This changes reduction arithmetic even
where all causal keys are retained. It is not evidence of a missing-key or
owner-slot alias, and no such concrete alias was found in the source audit.
The narrow issued-budget correction is committed as `20c73e42`; it restores
the declared bound, not an asserted quality fix. Later combined native results
are recorded under their own source IDs below.

The audit also found pre-existing head/worker normalization-call asymmetry
at the initial chunk and implicit head normalization-to-next-chunk ordering.
Whether the norm200 clamp activates in this failure is not measured; do not
present these observations as its proven cause. This was the historical audit;
the later normalization corrections in `1cb0267e` are summarized below and do
not retroactively validate the earlier outputs.

Evidence: `longctx-22b56144-4096-off-chat-run`, its raw quality receipts
`http-0085.json` (successful NEBULA C2) and `http-0167.json` (failed C3), and
`longctx-22b56144-4096-off-chat-summary.json`. The new ordinary cache pools
were12606/13112 blocks, minimum sampled MemAvailable10,350,688/10,531,008 KiB.
These4K allocation facts do not by themselves qualify a larger context.

## New-image 4K accelerated run: faster prefill, quality still failed

The same `22b56144` engine and real-chat workload ran with
`ATLAS_GLM_PAGED_PREFILL_BF16_GEMM=1`. For the four C1 retrieval/linked-facts
requests (3953/3953/3953/3956 prompt tokens), mean reported TTFT fell from
8.341s OFF to5.945s ON: **28.7% lower**. Individual OFF/ON times in seconds
were8.231/5.860,8.344/5.947,8.322/5.908 and8.469/6.063. These are one
fresh-process run per setting, not a repeated performance qualification or
proof of numerical equivalence; output trajectories differ.

C1 and C2 again passed all six quality waves. C3 passed early/middle/late
needle retrieval and linked facts, but auto-tool failed differently:
NEBULA produced the correct structured call after53 tokens, while AURORA
identified its correct case ID in plain content and continued planning until
the192-token cap, without a tool call. ORBIT's call passed. The failed AURORA
response reported0 reasoning tokens despite the request's explicit thinking
enable and budget16. This is not a successful tool invocation or a quality
PASS. C3 tool-result, C4 and boundary/cancellation qualification were not
issued after the failure; that `22b56144` run did not qualify8K/16K.

Both containers exited0, OOM=false, with swap0. Minimum sampled MemAvailable
was10,168,100/10,610,808 KiB. This ordinary operational-watchdog shutdown is
not a paired-T3 quiescence receipt, and does not erase the failed workload.
Evidence: `longctx-22b56144-4096-on-chat-summary.json` and
`longctx-22b56144-4096-on-chat-run/quality-receipts/quality/`; actual requests
`http-0007.json`, `http-0013.json`, `http-0019.json`, `http-0025.json` provide
the C1 timings, `http-0167.json` the AURORA failure and `http-0168.json` the
successful NEBULA call. OFF timing receipts use the same names in the OFF run.

## Latest engine source corrections

- `20c73e42` caps the issued initial chunk at the configured1024 tokens for
  the opt-in ordinary GLM sparse profile, including idle admission. The
  arena remains1028 rows; other profiles retain their prior policy. Actual
  CPU test RED was1 pass/1 failure; GREEN was2 passes
  (`chunk-budget-red.log`, `chunk-budget-green.log`).
- `c853bafa` makes the GLM OpenAI template honor resolved enabled thinking
  during tool turns. The old override always closed thinking when tools were
  present; API reconciliation then disabled the requested budget. Enabled
  tool turns now retain the stock open-thinking suffix and explicit budget16;
  disabled/default tool turns remain closed and no-tools behavior is unchanged.
  Real rendering/tokenization/reconciliation tests recorded37 passes/2 expected
  failures before the change and39 passes afterward
  (`tool-thinking-red.log`, `tool-thinking-green.log`). The tool output cap
  remains192; this fixes the request contract rather than increasing its budget.

The actual stopped-container runtime template matched the old Git/a069 asset,
SHA256 `7398a9b6153b868b0ff1213a8a84fa459a53ea088d84b131eb4313d2bf8c98ea`.
Templates load from the process working directory, not from the server ELF;
ELF-only image overlays therefore do not update these assets. The corrected
GLM template is SHA256
`d921f36103aa17db5fbf5891e4f7fe55a9080db450d7b8fb5c9833237c31bd16`.
Images must carry and verify the current runtime templates separately. The
bounded new-image results below do not establish a maximum safe context or a
complete release PASS. The earlier failed OFF and ON receipts remain retained.

## c853bafa 4K: C1 passes, C2 exact-format failure

The fresh `c853bafa` ordinary4K run used paged-prefill BF16 GEMM **ON**.
All six C1 quality waves passed, followed by the C2 early-needle wave. The
subsequent C2 needle response contained the correct AURORA value but added
brackets, failing the unchanged exact-answer validator. Correct retrieval
content with extra formatting is not a PASS. No later quality waves or
boundary/cancellation supplement were issued after this failure.

Both containers exited0, OOM=false, with swap0. Minimum sampled MemAvailable
was10,173,260/10,723,440 KiB (head/worker). Clean shutdown does not erase the
failed campaign. Evidence: `longctx-c853bafa-4096-on-chat-summary.json` and
`longctx-c853bafa-4096-on-chat-run/quality-receipts/quality/`.

## c853bafa 8K: C1 quality passes, reuse-answer supplement fails

The fresh ordinary8K ON run tested **C1 only**. All six main quality waves
passed: early/middle/late needles, linked facts, structured tool call, and
actual-call-ID tool-result followup. That quality child completed in228.86s;
this is total phase elapsed time including preparation/calibration, not a
throughput or per-request latency result. It does not qualify8K C2–C4.

The boundary supplement then passed exact HTTP400 checks at8192/8193 input
tokens and observed actual streamed text plus response ID before client close.
The next reuse-answer probe failed: its bare `/v1/completions` prompt produced
`153`, then repeated prompt/answer text until its32-token cap with
`finish_reason=length`. The strict integer/normal-stop validator correctly
rejected it. Subsequent reuse-tool checks were not issued. This is a retained
boundary-suite failure, not proof of a server cancellation bug or completed
GPU reclamation.

Both containers exited0, OOM=false, swap0; minimum sampled MemAvailable was
10,149,920/10,721,620 KiB. Evidence: `longctx-c853bafa-8192-on-chat-summary.json`
and its run's `quality-receipts/boundaries/http-0009.json` (receipt SHA256
`c8678b80fe9f065b30fe8c7cc5629e024567d9f28623ba89e3e897c41f468c6c`).

Harness correction `0b80ab90` changes only reuse-answer framing to the main
quality helper's real chat request: empty tools, explicit thinking budget16,
cap128, and one-token chat calibration for exact input counting. The integer,
normal-stop and peer-leakage validators are unchanged; raw context400 and
cancellation SSE probes remain unchanged. This is an input-contract correction,
not a server cancellation fix. The corrected client's later1cb16K outcome is
recorded below; neither prior failure is relabeled as a pass.

## 1cb0267e: normalization-corrected image,4K still fails

Engine `1cb0267e` includes the earlier chunk-budget/tool-thinking changes plus
`3880a64d`'s single-writer SSM normalization kernel and matching ordinary GLM
head/worker normalization calls on the compute stream. The
[normalization report](ssm-normalization-native-results.md) separates actual
kernel racecheck/memcheck evidence from scheduler dispatch checks. These are
real corrections, not a claim that the prior output failures are resolved.
Current immutable image/ELF/template pins are in the
[release handoff](../../releases/glm53-dual-spark-20260910-rc.md).

The fresh4K ON run passed C1 early/middle/late needles and linked facts, then
**failed** the C1 auto-tool gate: tool-call markup appeared in reasoning rather
than a valid structured call in the API response. Hidden markup is not promoted
to a tool-call PASS. No tool-result, C2–C4 or boundary supplement was issued.
This also means the corrected boundary client `0b80ab90`, although pinned in
the launch, was not exercised by this run.

Both containers exited0, OOM=false, restart count0, sampled swap0; minimum
MemAvailable was10,251,148/10,670,476 KiB (head/worker). The99.77s quality phase
failed; successful cleanup does not qualify it. Evidence:
`longctx-1cb0267e-4096-on-chat-summary.json` and
`longctx-1cb0267e-4096-on-chat-run/quality-receipts/quality/`.

## 1cb0267e 16K: six main checks pass, corrected reuse answer fails

Final status was available at00:18:21 UTC on September10 (superseding the
earlier incorrectly timed00:19 progress note). All six C1 main-quality checks
passed: three needles at16,229 input tokens, linked facts at16,232, structured
tool call at15,858 input/44 output tokens, and actual-call-ID tool-result
followup at15,783 input/7 output tokens. The main-quality phase took459.83s
including preparation/calibration; this is not throughput or broad coherence
qualification and does not establish C2–C4 quality.

The boundary supplement passed exact HTTP400 at16,384/16,385 input tokens
and observed real SSE text plus response ID before closing the client. The
corrected chat reuse answer then **failed**:28 input/45 output tokens, with
16 reasoning tokens containing `153What is the largest prime factor of
600851475143?`, followed by visible `**6857**` and
`600851475143 = 71 × 839 × 1471 × 6857`. It stopped normally, but this is the
wrong answer to the original arithmetic request; the strict validator rejected
it. No reuse-tool checks followed. The unrelated mathematics could reflect
prompt behavior, numerical behavior or state handling; cancellation is not an
established cause, and normal stop does not make the response correct.

The full run remains **FAIL**. Both ranks exited0 with OOM=false, restart0 and
sampled swap0. Minimum MemAvailable was10,125,500/10,640,920 KiB. Evidence:
`longctx-1cb0267e-16384-on-chat-summary.json` and
`longctx-1cb0267e-16384-on-chat-run/quality-receipts/boundaries/http-0009.json`.
The successful bounded main checks do not erase failed reuse or establish a
fully qualified maximum context. Reasoning/preambles remain retained evidence,
not a generally validated coherence claim.

## Operator maintenance, September10 00:26 UTC

The first independent short C8 graph attempt remains **FAIL**: its worker
observation reported12 KiB swap used, so the strict monitor rejected it before
the workload was issued. Both model processes exited0, OOM=false; this was not
a node crash or a measured graph-performance run. The rejected sample reported
9,498,272 KiB MemAvailable. A summary maximum computed only from accepted
samples may show swap0 and must not hide that rejected12 KiB observation.

The operator's subsequent idle preflight observed121,217,644 KiB MemAvailable,
12 KiB swap used, no GPU compute PID, no running Atlas container, and only
`/swapfile` at priority-2. A bounded10s `swapoff /swapfile` followed immediately
by `swapon -p -2 /swapfile` restored the same10,485,756 KiB swap capacity with
used0. These are operator-observed maintenance results, not a zero-swap native
qualification. No persistent configuration, reboot, driver change or monitoring
threshold relaxation was made. Any next run still requires its own fresh
observations and unchanged zero-swap gate.

## 1cb0267e fresh/repeat/post-cancel arithmetic diagnostic

A fresh16K ON process received the exact short math chat as its **first
model-generating request**, without a preceding calibration. That request,
its immediate repeat and its post-cancel repeat all produced the same wrong
answer: reasoning began with153 then the unrelated prime-factor question,
and visible content answered6857. All three reported28 input/45 output tokens,
16 reasoning tokens and normal stop. The two explicit requested-false controls
were identical too; the no-tools template can force thinking, so these are
not evidence of effective nonthinking execution.

All five math checks **FAIL** under the unchanged exact153 validator. The
bounded diagnostic made15 HTTP requests and observed real SSE text before
client close. Cancellation is therefore **not necessary to trigger this
failure**: it already occurred on the first request in the fresh process.
This does not establish a state leak or a particular prompt/numerical/decoding
cause. Health/close alone also do not prove same-slot reuse or GPU reclamation.

Both ranks exited0, OOM=false, restart0 and sampled swap0. Minimum
MemAvailable was10,707,988/10,356,180 KiB. Retained evidence:
`reuse-1cb0267e-16384-on-first-summary.json`, its exact run and full workload
receipts. The controller correctly records a failed workload despite clean
shutdown; identical responses are not a quality pass.

The same-image bounded generation-watchdogs-off diagnostic completed under
the separate profile whose pin begins `34e0f2`. All five math checks again
**FAIL**, with identical wrong prime-factor outputs:28 input/45 output tokens,
16 reasoning tokens. A4 bias remained ON, so that flags experiment does
**not** isolate midword thinking-boundary behavior or establish its causality.
Both ranks exited0, OOM=false, restart0 and sampled swap0; minimum
MemAvailable was8,267,356/8,296,328 KiB. Evidence:
`reuse-1cb0267e-16384-watchdogoff-summary.json` and its retained raw receipts.
Disabling those generation watchdogs did not resolve the failure. Proposed
GLM thinking-boundary changes still require their own exact-build native
qualification; no failed result above is replaced or reclassified.

## 14b4e485 native build and first arithmetic diagnostic

Frozen engine14b4e485 built natively in6m34s plus6.93s for helpers, builder
exit0/OOM=false. Both packaged images' actual server, guard and runtime GLM
template hashes agree with the [RC artifact table](../../releases/glm53-dual-spark-20260910-rc.md).

The fresh16K ON diagnostic completed at01:01:48 UTC. All five fresh, repeat,
post-cancel and requested-false math responses still **FAIL** unchanged:
28 input/45 output,16 reasoning tokens, reasoning153 then the unrelated
prime question, visible6857. This disproves resolution by the combined14b
boundary-policy changes; it does not disprove their separate tool-boundary fix.
Requested-false remains not proof of effective nonthinking execution.

Both ranks exited0, OOM=false, restart0, sampled swap0. Minimum
MemAvailable was10,630,212/10,290,624 KiB. Evidence:
`reuse-14b4e485-16384-on-first-summary.json`, its run, and raw receipts.
An existing-debug-log-only diagnostic is prepared to observe actual sampled
EOS suppression; its hypothesis is not yet a demonstrated cause.

## 14b4e485 4K ON final result: bounded tool improvement, suite FAIL

The campaign finished at01:08:37 UTC. All six C1 quality waves passed,
including the previously failing3483-input automatic-tool request:35 output
tokens, a valid declared call and exact arguments. The3399-input followup
reused the actual call ID and returned the exact result in29 output tokens.
All three C2 needle waves then passed. These are bounded checks, not broad
prose coherence or complete context qualification.

C2 linked facts **FAILED** the unchanged strict JSON validator:
`quality-receipts/quality/http-0076.json` returned correct Noor/Kyoto/28 facts
wrapped in a Markdown `json` fence. Its peer `http-0077.json` returned correct
Iris/Oslo/25 raw JSON. No fence repair was applied; no C2 tools, C3/C4 quality
or boundary/cancellation/reuse phase was issued.

Both ranks exited0, OOM=false, restart0 and sampled swap0. Minimum
MemAvailable was10,620,256/10,249,140 KiB. Evidence:
`longctx-14b4e485-4096-on-chat-summary.json` and its retained run/receipts.
The overall result remains **FAIL**, despite clean shutdown and the bounded
native tool improvement. Exact request/response pins and reasoning limitations
are in the [native reasoning-boundary report](native-reasoning-boundary-results.md).

## 14b4e485 sampled end-of-turn diagnostic

The log-only16K rerun reproduces all five failures and localizes the unwanted
continuation. In `reuse-14b4e485-16384-eos-first-run/00197-rank0-collect.json`,
at01:10:50.181145 the first token is122876 (`153`); at01:10:50.255972 the
next sampled token is154827, which the actual checkpoint tokenizer identifies
as `<|user|>` and generation_config lists as EOS. At01:10:50.255986 the
`atlas::eos` diagnostic says thinking is the **sole** suppressor. The next
sample is3838 (`What`), followed by the unrelated prime-factor question.
The same EOS suppression appears in all five math requests. Source inspection
confirms the suppressed token remains `last_token` and conditions the next
decode. This is concrete native turn-boundary evidence, not proof that a
particular replacement policy produces a correct visible final answer.

Only RUST_LOG changed (`info,spark::scheduler::decode_logits_step=debug,atlas::eos=debug`),
with profile SHA256
`897a2ced262335f5cc69f0eb03ddfaeae8ccea4ff23902c863f9d7f1570ad1ec`.
All model, memory, lease and generation controls were retained. Final image
observations show both ranks exited0/OOM=false/restart0 and sampled swap0;
minimum MemAvailable10,591,004/10,238,988 KiB. The controller additionally
records a cleanup-observation race: worker `/proc/2661118/cgroup` disappeared
between process observation and file read. Therefore the campaign is **FAIL**
for both workload and cleanup-observation error, despite final clean exits.
Evidence: `reuse-14b4e485-16384-eos-first-summary.json` and raw receipts.

## 14b4e485 near-full16K C1 prose: bounded automatic/manual PASS

`prose-14b4e485-16384-on-first-run` finished01:15:44 UTC. The actual final
chat used15,869 input tokens and86 output tokens under configured16,384
context (output cap384; input+cap16,253). Three real-chat sizing probes are
retained separately and are not quality responses. Final TTFT26,058.646ms.

The visible answer is three connected complete sentences: VIOLET-7429 is
led by Elena Marin in Ghent; originally31 sealed sample kits, exactly8
removed, no other inventory changes,23 remaining. Root directly reviewed
the prose and separate reasoning: accurate relationships/arithmetic, no
invented people/purpose/dates, no repetition or foreign markers. The reasoning
is one relevant sentence, but its reported31 tokens exceed the requested
soft16 budget. No strict16-token bound is claimed.

The driver's automatic checks pass while its `qualification_passed` remains
null by design. Root's separate bounded manual review is retained as
`prose-14b4e485-16384-manual-review.json`; receipt
`prose-14b4e485-16384-on-first-receipts/http-0007.json` SHA256:
`5756b0b6cdd8d711d6dd8c0f58a983be3e2e880a66734de3a118e78c5febc33d`.
The ordinary controller passes, both ranks exit0/OOM=false/restart0,
sampled swap0, minimum MemAvailable10,593,068/10,160,020 KiB.

This is one C1 near-full prose sample on ordinary eager MTP-OFF with paged
BF16 prefill ON. It does not qualify C2-C4, long-context MTP, broad coherence,
or the release as a whole; the separate arithmetic/strict-format failures remain.

## da4ec65d native EOS correction: shorter default failure, effort controls pass

Engine `da4ec65d316bf3c57796e975633a6e8e78bfc099` built successfully in6m37s
plus6.86s for helpers. Both images' actual server hashes match
`5c77b881074a2cc8fdbf9dc21950d5366574092fd5faf95261cfd6c944407406`;
complete artifact pins are in the [RC handoff](../../releases/glm53-dual-spark-20260910-rc.md).

The16K effort diagnostic completed at01:34:03 UTC with19 HTTP requests and
unchanged output caps. Direct raw-receipt inspection confirms all five default
fresh/repeat/post-cancel/requested-false cases now stop after2 output tokens:
reasoning is `153`, visible content is empty. The unrelated prime-question
continuation is gone, but all five still **FAIL** exact visible-answer153.
Requested-false remains not proof of effective nonthinking execution.

Separate explicit reasoning-effort controls, both retaining budget16, **PASS**:
low returns visible153 in5 output tokens (`http-0015.json`), high in4
(`http-0017.json`). These are changed request controls, not default-profile
passes. Reported reasoning-token usage is0 in default/high and1 in low despite
the retained reasoning text; these reported counters are not asserted to be
an exact accounting of the rendered reasoning.

The overall diagnostic remains **FAIL**. Both ranks exited0, OOM=false,
restart0, sampled swap0; minimum MemAvailable10,676,320/10,279,364 KiB.
Evidence: `reuse-da4ec65d-16384-effort-first-summary.json` and
`reuse-da4ec65d-16384-effort-first-receipts/`.

## da4ec65d default C8 MTP: answer gate fails, forced cleanup

The default selected-MTP campaign passed7 of8 answer checks. Index3 produced
the correct JSON only in reasoning with empty visible content, so the strict
answer gate **FAILED**. No tools, tool-result followups, needles or timing
were issued. Evidence: `portable-da4ec65d-c8-first-summary.json`.

Failure cleanup did **not** establish clean paired quiescence/release:
the controller recorded `healthy drain not authorized` and did not confirm
remote exit in its cleanup result. Subsequent exact-container observations
`portable-da4ec65d-c8-first-final-r0.json` and `-final-r1.json` confirm both
exited137, OOM=false, restart0 after forced cleanup. Sampled campaign swap was0;
minimum MemAvailable9,878,000/9,236,208 KiB. Root's postcheck found no GPU work
or running Atlas containers, host swap0, and available memory
121,502,060/121,192,552 KiB. Those postconditions do not turn this into a
clean paired-release PASS. The separate low-effort MTP result follows.

## da4ec65d explicit-low C8 MTP also fails the answer gate

The new requests explicitly set `reasoning_effort: low`, retaining budget32
and output cap128. Again7 of8 answer checks passed. Index3 returned the correct
`{"city":"Oslo","count":7,"active":true}` only in reasoning, with empty
visible content and normal stop:19 output tokens and17 **reported** reasoning
tokens. The strict visible-answer validator rejected it; no tools, followups,
needles or timing were issued. Actual request/response evidence is retained in
`portable-da4ec65d-c8-low-first-prepared/evidence-000625.json`.

The overall result is **FAIL**, not a low-effort C8 qualification. Minimum
MemAvailable was10,361,352/9,390,272 KiB; sampled swap0. Root's exact-container
postcheck confirmed both stopped137 at01:41:19 UTC, OOM=false, restart0:
`df69de64dbabb26b84f0b5e9512ec35add29f8538c3922c38293799412605c3c`
and `f219b08a07c961e186cff68a5957e0e0799b86b161fe988466d9b39efd12b5aa`.
No running Docker containers or GPU compute PIDs remained; host swap was0
and available memory121,498,036/121,196,832 KiB. This forced-cleanup result
is not clean paired quiescence/release. The separate low-effort16K C1 result
follows.

## da4ec65d explicit-low16K C1: middle needle reasoning-only, FAIL

`longctx-da4ec65d-16384-low-first-run` completed at01:51:06 UTC. The early
needle passed with16,229 input/24 output tokens and exact visible
`AURORA-6819`. The middle needle (`quality-receipts/quality/http-0013.json`)
used the same16,229 input-token count, but returned8 output tokens with
`AURORA-6819` only in reasoning and empty visible content, normal stop.
Its reported reasoning count was6 and TTFT26,189.222349ms. The unchanged
exact visible-answer validator correctly rejected it; reasoning-only retrieval
is not a quality pass.

No remaining quality waves or boundary checks were issued. The controller
records **FAIL**, despite both ranks exiting0, OOM=false, restart0 and sampled
swap0. Minimum MemAvailable was10,646,604/10,036,728 KiB. Evidence:
`longctx-da4ec65d-16384-low-first-summary.json` and its retained raw receipts.
Explicit low effort therefore does not establish a qualified16K profile.
The separate default-effort da4ec65d16K prose result follows.

## da4ec65d near-full16K C1 prose: bounded visible-prose PASS

`prose-da4ec65d-16384-on-first-run` completed at01:54:41 UTC. The final chat
used15,869 input/105 output tokens, normal stop, with output cap384 and
TTFT26,022.750181ms. Three calibration responses remain sizing evidence only.
The visible answer is four complete connected sentences: VIOLET-7429 is led
by Elena Marin in Ghent;31 initial sealed sample kits minus8 removed, no
other changes,23 remaining. Root's separate manual review confirms correct
relationships and arithmetic, no invented facts and no visible repetition.

Automatic checks pass; the original summary still deliberately reports
`qualification_passed: null`. The reasoning is relevant but ends mid-phrase
at `no other`, and reports48 reasoning tokens against the requested soft16
budget. This is **not** a reasoning-prose coherence PASS, strict16-token cap
proof or exact reasoning-accounting claim.

Evidence: `prose-da4ec65d-16384-on-first-receipts/summary.json`, separate
`prose-da4ec65d-16384-manual-review.json`, and raw `http-0007.json` SHA256
`2bb61f96752b4636fc1e60a6247656971a396299c0b6d60164d1baf4f873a0e6`.
Both ranks exited0, OOM=false, restart0, sampled swap0; minimum MemAvailable
10,614,264/10,182,376 KiB. This is one C1 visible-prose sample on ordinary
eager MTP-OFF, paged BF16 prefill ON, not a complete16K/release qualification;
the failed arithmetic, retrieval and C8 campaigns remain failed. A separate
default-effort da4ec65d16K C1 quality run is active at this update, without a
final result here.
