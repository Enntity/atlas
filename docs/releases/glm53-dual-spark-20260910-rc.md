# GLM-5.3-Flash dual-Spark fork: release-candidate handoff

**Draft test handoff, not a final qualification or model card.** Engine source
`22b56144` includes the upstream merge at
`6c5f17dab9c27ee2396aef1ac2501a17b201c715` and an OFF-by-default later-chunk
BF16 projection experiment. The new build's native qualification is **PENDING**;
its4K probe is in progress. Older-image results below do not qualify this build.

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

## New build pins — runtime qualification pending

| Artifact | Full SHA256 |
| --- | --- |
| Head image | `7029712412b5547a3c4eacf9162f1520cb6ca034ed78074f60223de721fe30bc` |
| Worker image | `70ffa1aee2ab7c912c3e73985eb10cabc4e1aad6bf0397543bc8fcdfdae89996` |
| Server ELF | `35afceb060ef2bca140f91891414ba3d525b7908797702ca5fc9766967cc19a8` |

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
out. Native validation of the later `c853bafa` source, including its chunk-budget
and explicit tool-thinking changes, is **PENDING**; no normalization change has
been made. The earlier failures and this audit
must not be relabeled as passes for that source.

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
