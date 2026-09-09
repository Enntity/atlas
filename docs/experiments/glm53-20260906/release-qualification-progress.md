# Release qualification: September 9–10

Deadline: September 10, 02:42 UTC. The first merged C8 native engine source is
`a6cfeec0`, merged with upstream `6c5f17dab9c27ee2396aef1ac2501a17b201c715`.
Latest release source freeze is `c853bafa`; its native qualification is pending.
Portable C8 harness
and its followup-envelope correction are `708db27c` and `440526f2`.

## Immutable native build

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
   exited0, OOM=false, swap0. A fresh probe now uses actual `/tokenize(messages)`
   no-thinking chat-template tokens, not guessed framing. Exact answer/stop
   checks and watchdogs are unchanged; new quality remains to be measured.

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
the declared bound, not an asserted quality fix. Native validation is pending.

The audit also found pre-existing head/worker normalization-call asymmetry
at the initial chunk and implicit head normalization-to-next-chunk ordering.
Whether the norm200 clamp activates in this failure is not measured; do not
present these observations as its proven cause. Exact replay is prepared.

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
issued after the failure;8K/16K remain pending.

Both containers exited0, OOM=false, with swap0. Minimum sampled MemAvailable
was10,168,100/10,610,808 KiB. This ordinary operational-watchdog shutdown is
not a paired-T3 quiescence receipt, and does not erase the failed workload.
Evidence: `longctx-22b56144-4096-on-chat-summary.json` and
`longctx-22b56144-4096-on-chat-run/quality-receipts/quality/`; actual requests
`http-0007.json`, `http-0013.json`, `http-0019.json`, `http-0025.json` provide
the C1 timings, `http-0167.json` the AURORA failure and `http-0168.json` the
successful NEBULA call. OFF timing receipts use the same names in the OFF run.

## Latest source corrections awaiting native validation

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
The next image must carry and verify the current runtime templates separately.
Neither CPU fix has yet established a native quality PASS or a maximum safe
context. The failed OFF and ON receipts remain retained.
