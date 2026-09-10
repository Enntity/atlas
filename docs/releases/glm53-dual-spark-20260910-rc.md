# GLM-5.3-Flash dual-Spark fork: release-candidate handoff

**Draft test handoff, not a final qualification or model card.** Engine source
`14b4e485a8719d763b1b043e1c7a8844994ae4c2` includes the upstream merge at
`6c5f17dab9c27ee2396aef1ac2501a17b201c715`, an OFF-by-default later-chunk
BF16 projection experiment, bounded sparse chunk admission, and explicit
tool-thinking support, matching ordinary GLM EP normalization calls/streams,
and a single-writer state-normalization kernel. It additionally preserves
native GLM reasoning/tool boundaries and stops selected-MTP acceptance at a
reasoning phase change. The native binary and both images are built and
hash-verified. Its fresh/repeated/post-cancel arithmetic diagnostic still
**FAILS** with the same unrelated prime-factor answer; final tool/context and
selected-MTP validation are **PENDING**. The prior `1cb0267e`
4K ON probe failed its C1 structured-tool gate; its separate16K ON C1 probe
passed all six main-quality waves but failed corrected chat reuse. Neither
complete campaign passed, and neither qualifies the new build.

## Reproduce the bounded profiles

- [Portable C8 MTP bundle](../../scripts/dev/glm_release/README.md): actual
  guard/relay/controller, strict recipe materialization and workload procedure.
  TP2/EP2-v2, up to eight independent owners, MTP4, context2044, cold single-chunk
  prefill at most1024, BF16 KV and snapshot rollback. This is not long-context MTP.
- [Portable long-context bundle](../../scripts/dev/glm_release/long_context/README.md):
  separate ordinary **MTP-OFF**, eager sparse C1–C4 profile, sequential4K/8K/16K
  exploration with1024-token prefill chunks. Full-context quality and boundary/
  cancellation/reuse qualification are **PENDING**. Actual qualified maximum
  context: **PENDING**;16384 is the current C4 policy ceiling, not a fresh PASS.

Use the bundles' explicit operator inputs, immutable images/ELFs and local
dry-run checks. They do not provision credentials, build images or grant access.
Run only in an exclusive authorized two-node window; do not overlap profiles.
Keep generated recipes, session identities and private configuration outside Git.

Both profiles retain114GiB cgroup limits with no swap, at least4GiB monitored
host headroom and bounded cleanup. The MTP runner requires real both-rank
quiescence/release plus independently observed clean exits; HTTP success alone
is insufficient. The ordinary long-context watchdog is a different operational
safety mechanism, not a paired-session certificate. Neither promises recovery
from an unresponsive GPU, driver, kernel or Docker daemon.

## New build pins — not release-qualified

| Artifact | Full SHA256 |
| --- | --- |
| Source archive (`14b4e485`) | `2506dce744e767569683079c837e6a02a88240992db648073e99c94c472e4fa8` |
| Server ELF | `1b4b9a0e36046fd6be51773126e49189ee318e7b28719efcac5c4662f0284ee6` |
| Head image | `f50c86ffdd29200627db62917ae955d485878d64aa04a833d50fe0468e414f35` |
| Worker image | `aa31bf3a0b69104122df94c7255c949d90b424e56ab50dd5fef3aa6e5714f23c` |
| GLM OpenAI runtime template | `d921f36103aa17db5fbf5891e4f7fe55a9080db450d7b8fb5c9833237c31bd16` |

The bounded CPU-only native builder completed successfully (reported build
stages6m34s and6.93s), exit0, OOM=false. Helper binaries are unchanged from the
retained `a6` helper set; actual inspection of both images verified the exact
guard, server and runtime-template hashes, with working directory `/`. A successful build is not a
model-quality result. Do not combine this server pin with the old images below.

The [native reasoning-boundary report](../experiments/glm53-20260906/native-reasoning-boundary-results.md)
records the actual native opener observation, focused RED/GREEN corrections,
and44 committed-source CPU tests passing. Fresh math, tools, context and
selected-MTP validation remain required; the earlier math failure's cause is
not established by these tests.

### Retained prior `1cb0267e` pins — failed qualification

| Artifact | Full SHA256 |
| --- | --- |
| Head image | `f65d53b311f6ca2c0af1b64538b342d15807fe616f550be714b7a2c2532e867d` |
| Worker image | `0b7e4677f9d31ccc2241e4af3eecab9593469729de703d4db0f666f8e44e0f47` |
| Server ELF | `bdda966a70a22821d257feb8d39814833d4d2abfb48cebb7ecd2049dcb64fef0` |
| GLM OpenAI runtime template | `d921f36103aa17db5fbf5891e4f7fe55a9080db450d7b8fb5c9833237c31bd16` |

That prior server and all212 target kernels built without the stub gate; the bounded
CPU-only native builder exited0, OOM=false. Both images were inspected by full
ID; actual in-image server, guard and runtime GLM template hashes matched,
and the working directory was `/`. Source archive SHA256 is
`961436bbd49dc938deaa7f372c5df8d3262c546be5a99af147c572137b63a7e5`.
That source archive includes the complete runtime template tree separately from
the ELF. These packaging checks do not establish model-quality success.
Actual prior serving logs report NCCL2.31.2 with CUDA13.3; the build compiler
was nvcc13.0. These are distinct observations, not a claim that the pending
new image has already served or that its runtime is NCCL2.27.7/CUDA13.0.

The [normalization correction report](../experiments/glm53-20260906/ssm-normalization-native-results.md)
records actual CUDA racecheck32→0 and corrected memcheck0, with numerical
checks passing both before and after. It also records real scheduler/worker
dispatch checks for missing first-chunk normalization and incorrect continuation
stream selection. These fixes are not yet established causes or resolutions of
the earlier long-context output failures; the new combined image must pass.

### Current versus historical native results

| Engine and profile | Retained outcome |
| --- | --- |
| `14b4e485` | Native build and image verification **PASS**. Fresh/repeat/post-cancel arithmetic **FAIL**, all five outputs unchanged from1cb. Both ranks exited0, OOM=false, restart0 and sampled swap0. Tool/context and selected-MTP qualification pending. |
| `1cb0267e`,4K ON | **FAIL**: C1 early/middle/late needles and linked facts passed; auto-tool markup appeared inside reasoning, not a valid structured tool call. No tool-result, C2–C4 or boundary checks followed. |
| `1cb0267e`,16K ON,C1 | Six main quality waves **PASS**; full suite **FAIL**. Exact16384/16385 HTTP400 and actual-text cancellation probes passed, then corrected chat reuse answered unrelated prime-factor mathematics instead of153. No reuse-tool check followed. |
| `c853bafa`,4K ON | **FAIL**: all six C1 waves and C2 early needle passed; subsequent C2 AURORA needle added brackets and failed strict format. No later quality/boundaries. |
| `c853bafa`,8K ON,C1 | Six main quality waves **PASS** in228.86s including calibration; full suite **FAIL** after passed exact-context400 and actual-text cancellation probes, because the raw-completion reuse answer repeated after153 and finished with `length`. |

The c8538K elapsed time is not throughput or broad coherence evidence. Its
boundary framing correction `0b80ab90` uses counted real chat, thinking16 and
cap128, without weakening the exact-answer/stop validator. The1cb4K input pins
that corrected client but never reached it; the1cb16K run did reach it and failed
the exact answer despite a normal stop. No cancellation-cause inference follows.
Both completed c853 runs and both1cb runs exited0 on both ranks with OOM=false
and sampled swap0; their workload failures remain failures. Full paths and
memory minima are in [qualification progress](../experiments/glm53-20260906/release-qualification-progress.md).

The new `ATLAS_GLM_PAGED_PREFILL_BF16_GEMM=0|1` experiment affects only four
later-chunk GLM BF16 projections; absent/OFF preserves scalar control. Its CPU
dispatch tests are not numerical evidence. Same-image OFF/ON multi-chunk native
quality and timing remain required before selecting ON in a qualified profile.

## What has actually been measured

The following belongs **only to the earlier merged `a6cfeec0` engine**, whose
image/ELF pins and raw-receipt names are in
[release qualification progress](../experiments/glm53-20260906/release-qualification-progress.md).
It used selected capacity8/context2044/MTP4, unused prefix-cache slots and
checkpoint interval explicitly0, one warmup and three measured148-input/
256-output batches per width:

| Concurrency | Aggregate full-wall tok/s | Aggregate decode-window tok/s |
| --- | ---: | ---: |
| 1 |22.588|23.538|
| 2 |30.675|31.539|
| 3 |35.093|35.839|
| 4 |37.891|38.541|
| 5 |39.719|40.292|
| 6 |41.071|41.583|
| 7 |42.253|42.711|
| 8 |43.087|43.506|

That run passed32 bounded checks: eight distinct answers, eight automatic tool
calls, eight actual-call-ID result followups and eight own/no-foreign needles.
Both ranks completed paired release and exited0; sampled swap remained0.
All144 coding completions reached256 tokens and matched one another, but
**the generated code was not executed or functionally qualified**. These checks
are not a broad coding benchmark or comprehensive model-quality evaluation.

The merged outputs differ from the pre-merge control, and rates regressed versus
the earlier C4 image. This is not an equal-output A/B. The requested performance
goal remains **unmet**; acceptance counts/timing alone do not establish the
cause. See the [bounded argmax diagnostic plan](../experiments/glm53-20260906/merged-argmax-diagnostic.md).
Fresh-process repeat and new-build C8 qualification remain **PENDING**.

The first earlier-image bare-completion4K probe failed its repetition/normal-stop
quality check despite finding the needle. Correct real-chat framing is being
qualified without weakening the answer validators or watchdogs. Do not quote
the historical16K run as a PASS for this release candidate.

### Retained 4K full-response audit and quality limitations

A local, read-only audit examined the `22b56144` real-chat OFF/ON receipts under
`longctx-22b56144-4096-{off,on}-chat-run/quality-receipts/quality/` in the
20260909 `glm-native-controller` campaign. Each run contains33 final quality
responses and87 one-token sizing responses. A request-owner-aware scan of the
full returned choices, including content, reasoning and tool fields, found no
other-owner case names, result codes, function names, project names, leaders or
cities. This is a bounded known-marker check, not proof of arbitrary data
isolation; calibration responses are sizing evidence, not quality passes.

The raw fields expose weaknesses that exact-answer checks alone do not reject:

- ON `http-0085.json` (C2, NEBULA) makes the correct tool call, but its visible
  preamble repeats archive filler and breaks a sentence before announcing the
  correct case. ON `http-0090.json` returns the exact tool result, while its
  reasoning repeats that result and incorrectly describes three user sends and
  a garbled message.
- OFF `http-0154.json` returns the correct linked-fact JSON, but its reasoning
  introduces unrelated `question, answer` keys. Other successful responses have
  filler fragments or repeated own answers in reasoning. These observations do
  not by themselves identify a model, template or engine cause.
- Both complete runs remain **FAIL**, not qualified4K results: OFF
  `http-0167.json` (C3, NEBULA) produces a49-token filler loop without a tool call;
  ON `http-0167.json` (C3, AURORA) spends192 tokens describing the intended call
  without emitting one. Both finish with `length`; the strict tool gate rejects
  them. C3 tool-result followups and C4 quality were consequently not reached.

The harness checks exact retrieval at approximate early/middle/late positions,
linked facts/arithmetic, and actual assistant-call-ID/tool-result roundtrips;
it does **not** establish broad long-form coherence. Tool-call preambles and
reasoning are retained but not generally coherence-validated. Same-owner values
repeat across waves, so stale same-owner answers are not independently ruled
out. This audit describes the earlier `22b56144` receipts, before the later
chunk-budget, explicit tool-thinking and normalization corrections. `c853bafa`
has since produced the bounded native results above; `1cb0267e` additionally
contains the matching normalization calls/streams and single-writer kernel.
Neither those corrections nor later kernel checks retroactively validate these
historical responses. The retained1cb4K structured-tool gate remains a failure;
the new14b4e485 boundary changes have not yet passed native qualification.

## Human-operated first smoke test

Use a fresh private copy of the long-context bundle's `input.example.json`,
populate every required path/fabric/credential and a matching verified set of
image/ELF pins, and leave `workload` as `GENERATED_LOCAL_WORKLOAD`. The verified
14b4e485 pins above are test candidates, not a quality-qualified release; the retained1cb images are only
for explicitly labeled historical reproduction. Start with
`context: 4096`, `paged_prefill_bf16_gemm: false`, and C1: ordinary TP2/EP2-v2,
MTP-OFF, eager, BF16 KV, prefill1024, capacity4,114GiB/no-swap and existing
watchdogs. This conservative numerical-OFF smoke profile is **not** a fresh
14b4e485 quality PASS. To reproduce the known1cb4K ON failure, use its exact
retained artifact pins and a separate new input with the boolean `true`;
never mutate an already pinned launch.

```bash
/absolute/pinned/python3 -B /absolute/checkout/scripts/dev/glm_release/long_context/configure.py \
  --input /private/operator/4k-input.json --output /private/operator/4k-new-launch.json \
  --python /absolute/pinned/python3 --python-sha256 FULL_PYTHON_ELF_SHA256 --concurrency 1
/absolute/pinned/python3 -B /absolute/checkout/scripts/dev/glm_release/long_context/long-context-runner.py \
  --dry-run /private/operator/4k-new-launch.json
# Only after the exclusive two-node window is authorized and idle:
/absolute/pinned/python3 -B /absolute/checkout/scripts/dev/glm_release/long_context/long-context-runner.py \
  --run /private/operator/4k-new-launch.json
```

Use the authorized local-root account when private operator paths require it;
these commands provision no access. Preserve all raw content/reasoning/tool
fields and stopped-container receipts. Inspect coherence even when exact values
pass, and do not advance concurrency/context on a failing profile. For MTP use
the separate portable C8 recipe/controller procedure linked above, not these
ordinary-launch commands; current-image C8 still needs its own qualification.

## Rollback and acceptance checklist

Retain the qualified **capacity4** `a069efc3` rollback and its original recipe;
do not run the new capacity8 configuration against it. The
[owner-batch native report](../experiments/glm53-20260906/owner-batch-live-integration.md)
records its limits, controls, results and provenance:

- Head image: `e8afb24e836db7213ecad092e487d86b50dbf6ab90a562a8041d392847c18167`.
- Worker image: `8c2fa15cc174e48877cd59244077af494de566eb777848d37674b90f11d7be39`.
- Server ELF: `4247393f38723cffd9e6133f855ed8667de322099ef22533e0a9eaedcf5916c6`.

Before declaring this new build qualified, retain exact source/image/helper/
workload pins, pass the chosen profile's quality and boundary checks, obtain
fresh measured rates and final both-node clean-exit/no-OOM/no-swap receipts,
and update each PENDING item with evidence for that exact artifact. Preserve
failed attempts and stopped containers; no broad cleanup or mutable-tag rollback.
See [merge evidence](../experiments/glm53-20260906/upstream-release-integration.md)
and the [release checklist](../experiments/glm53-20260906/release-candidate-20260910.md).
