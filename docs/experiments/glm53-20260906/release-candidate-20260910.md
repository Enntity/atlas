# Six-hour release-test checkpoint

User request received2026-09-09 20:42UTC. Target handoff:
**2026-09-10 02:42UTC**. This is a testable fork release candidate deadline,
not a revision of the full reference-performance/serving goal.

## Execution order

1. Finish the in-flight eight-owner MTP implementation, including admission,
   memory reserve and retained retirement; commit a recoverable checkpoint.
2. Fetch and merge current upstream `Atlas-Inf/atlas` main, preserving our
   feature commits. Record exact upstream and merged source revisions.
3. Build immutable native artifacts from the merged committed source. Qualify
   short-context C1–C8 MTP with actual coherence, auto-tool calls, per-owner
   needle isolation, warmed full-wall/decode-window timing, TTFT and repeats.
4. Separately qualify the existing eager nonspeculative sparse path at4K,
   8K and16K as safety permits. Exercise near-cap prompt+output totals,
   early/middle/late needles, no foreign answers, real tool round trips,
   coherence, cancellation/drain/reuse and over-cap refusal. Higher limits
   require their own fresh evidence; historical results do not qualify this tip.
5. Freeze a candidate with time for recovery and handoff. Publish exact tested
   profiles, source/artifact hashes, launch commands, limitations and rollback.
   Keep the already qualified `a069efc3` images available throughout.

No new optimization should displace release qualification late in this window.
The release checkpoint may precede the complete performance goal; report any
remaining target gap explicitly and continue work afterward.

## Context correctness boundary

Concurrent MTP currently has a2044-token context cap: four lookahead rows must
remain inside the checkpoint's2048-entry dense/index threshold. Temporal K5
sparse selection and semantic-index repair are not implemented merely by
enlarging pool/metadata bounds. Do not alter checkpoint `index_topk` to bypass
this restriction. Selected cold prompt ingress also remains bounded at1024.

The available longer-context lane is independent eager sparse C1–C4 with
`ATLAS_GLM_C4_SPARSE=1`, not the short-context independent-C8 mode and not
paired MTP. Historical16K evidence exists in `phase4-infrastructure-results.md`;
fresh merged-build testing is still required. Label these separate profiles
clearly, without implying long-context concurrent MTP or long-context C8.

## Safety and evidence

Serialize model/native/GPU jobs. Use bounded immutable containers and the
existing matched-rank health/lease/quiescence safeguards. Preserve at least
4GiB actual headroom, zero swap and no OOMs/resets; stop before an unsafe cap.
Do not infer fit from weights alone or divide replicated target cache by TP2.

At the short C8 recipe,1024 prefill plus8 decode slots gives1032 arena rows.
The checked metadata region ends84224 within111464 quoted scratch bytes.
The existing34-KDA-layer accounting oracle adds1,920,304,128 bytes per rank
from C4 to C8 before the arena delta. Actual resolved allocation and measured
per-rank headroom, not that estimate alone, govern native admission.

Status at plan creation: model/E8 commit `e4eac89e` and scheduler commit
`6d18ec0c` pass actual byte-backed ownership/continuation checks. Native wider
FFN arithmetic/memcheck/repeat evidence is in `owner-eight-ffn-native-results.md`.
C6/C8 native serving, the upstream merge and fresh large-context qualification
are not yet complete.

## Portable operator handoff

The [portable C8 bundle](../../../scripts/dev/glm_release/README.md) packages
the retained standalone workload and canonical recipe generator with explicit
operator templates, pinned local inputs and the existing production supervisor.
It removes campaign-directory/historical-template dependencies; it does not
replace guard/lease/identity checks or publish a new native PASS. Generated
sessions, credentials and raw receipts stay outside Git. The bundle documents
the separate long-context lane and the selected profile's known API limits.

The upstream merge is committed as `a6cfeec0`; qualification status and exact
evidence are tracked in [upstream integration](upstream-release-integration.md).
The status paragraph above remains the plan-creation snapshot, not a claim
that this handoff qualifies the merged native image.
