# v25 consumption-time source and immediate-KV diagnostic

Root-owned execution plan, 2026-09-08. Implementation must first complete the
approved prompt-source plan, behavioral RED/GREEN, full CPU suite, non-test
check, strict v4 analyzer and exact v22/v23/v24 legacy replay, independent
review, source freeze and commit. Never overlay the active reader-integration
worktree. Before native execution, also complete the separately reviewed
[eager prefill stream fix](glm_eager_prefill_stream_fix_plan.md). Source audit
found the real head target/primer stream handoff unordered; do not deploy that
known unsafe handoff merely to collect another reproduction. Record both
independent commits. No performance promotion is authorized by this probe.

## Frozen build and unchanged serving profile

Archive only the committed spark-model crate for the existing stopped offline
native Rust builder (8GiB, CPUs0/1, Cargo jobs2). Keep its enabled CUDA189db87e
unchanged. The separately promoted B-tile CUDA and unfinished resident readers
are not part of this diagnostic. The committed private checked family has no
serving caller and must not cause new runtime kernel lookups.

Verify both nodes idle before any build/package operation. Package the resulting
executable over v24 in bounded, executable-only image contexts; record complete
Rust commit, CUDA revision, binary SHA256 and both inside-image hashes. Retain
v20 as a historical timing artifact, not a safe C1 restart recommendation after
the stream audit, and v24 as the directly preceding diagnostic.

Use the v24 overlap-on recipe with only image changed to v25. The binary also
contains the separately committed stream fix; this is not a probe-only A/B.
Explicit cache1,
verify0, trace1; C1, TP2/EP2, four drafts, accepted-pair repair, BF16 KV, existing
target verification graphs and numerical policy. Keep context2044, prefill1024,
114GiB memory limits, 4096MiB guard, KV overcommit0 and swapspace0. Refuse existing
standard container names; preserve containers instead of removing them. Require
at least4GiB host available memory on both nodes and no competing GPU workload.

## Complete observation window

Run the unchanged literal148/64 workload: one warmup and five sequential repeats.
Freeze complete both-rank logs before any quality requests. The committed v4
analyzer must account for every record through the explicit generation1..6
manifest: 192 records/rank, eight attempts/request, four steps/attempt.

Require the seven source/token/immediate-KV fields only at attempt1/step0,
alongside the unchanged v3 prefix/appended/map fields. At P148 this is twelve
source observation sets total, not384. The checked transfer envelope is
1,820,672 diagnostic D2H bytes/request/rank; this is an implementation-derived
budget, not an independent native traffic measurement. No timing from this
trace-enabled run qualifies the30 C1 or60 aggregate C4 target.

Report separately, both cross-rank and across repeated requests:

- Exact shifted primer and bootstrap token hashes.
- Consumption-time primer source and bootstrap source hashes.
- Immediate primer KV and bootstrap KV hashes.
- Composed written-prefix hash versus later same-rank prefix hash.
- Existing first input, post-EH, appended row, final and physical map hashes.
- Matching trajectory/metadata coverage, chosen drafts and gathered pairs.

Do not interpret unequal token inputs as writer divergence. The composed prefix
contains observations at two completion times, not an atomic full-cache
snapshot. A difference from the later prefix demonstrates change after at least
one writer observation, including possible bootstrap mutation of prior rows;
it does not identify a kernel. Equal source/token hashes with unequal immediate
KV narrows the differing boundary but does not independently prove correctness
or equality of unobserved weights/intermediates.

## Health, shutdown and subsequent decision

After freezing the six-request logs, run four answer checks, the1984/16 needle
case, a deliberate two-second streaming cancellation, and four recovery checks,
sequentially. P1984 must have source evidence unavailable without source I/O;
the existing v3 KV probe remains eligible. Preserve full later logs separately.
Any eligibility, ownership, budget, schema or health failure stops this probe;
do not relax the guard, omit a failed request or add an unreviewed fallback.

Gracefully stop both services with a30-second timeout and preserve renamed
containers. Record exit status/OOM state and host memory. Archive verified
receipts on controller and head. Select any subsequent fix or discriminator
from the actual observations, with its own TDD and bounded execution plan.
Continue resident B-tile reader integration independently on the controller;
do not mix its activation or arithmetic with this diagnostic.

## Execution addendum: idle receive health latch

The initial v25 diagnostic has complete equality but its first idle command
receive lasted376.9s and set the worker's communicator-health latch. A fresh
six-request window also agrees completely, with no warning in that window,
but manual gaps before later quality requests caused50.5s/32.9s idle warnings.
Both full receipts remain preserved and are not clean whole-run health passes.
No timeout is relaxed. Source audit identifies ordinary worker command waiting
inside the elapsed broadcast timer; a separate explicit idle-receive fix is
being planned, without exempting timed data/payload collectives.

Root's fixed campaign runner `run-v25-gated-native.sh` automates readiness,
unchanged requests, complete-window log freeze, quality, cancellation, recovery,
and graceful stop. Run a third diagnostic under tag `v25-gated-hidden`; require
the same strict384-record/equality analysis and no health errors in complete
post-stop logs. Only then run separate trace-OFF `v25-timing-initial` and fresh
`v25-timing-repeat`, each148/256, one warmup plus three measured runs with all
outputs capped, plus the same health/quality sequence. Automated short idle
gaps are a validation control, not a repair or endorsement of idle serving.
All use the exact same frozen v25 executable and safety limits; no build or
standalone GPU workload overlaps. Preserve every attempt, not just clean ones.
