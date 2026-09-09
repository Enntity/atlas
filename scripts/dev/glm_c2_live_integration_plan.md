# Connect supervised paired MTP4 serving

2026-09-09. Implementation order following native independent C8 qualification
(`ec300d06`), not a replacement for the full reference-parity goal. Root owns
native operations. Controller-only development may proceed while the unchanged
v29 binary runs its fresh-process benchmark repeat.

The existing terminal-session plan's approved quiescent-exit amendment is the
contract. Read it, the supervision plan and serial-scheduler audit together;
their historical design-only descriptions do not prove missing work is done.
The actual cold F0, bootstrap, serial K5/verdict/proposal and checked retirement
already exist. Reuse them; do not implement another parallel transaction model.

## Connected implementation order

1. Actual Model completion/quiescence: extend only its sealed paired capability
   with an actual communicator-health check and strict quiescence operation.
   Require the selected actual Model/communicator and use its real default and
   distinct secondary streams. Check communication health before/after each join;
   stop at first failed check/join. Do not mistake an event wait for a host join,
   retire/free/sweep owners, send protocol work, reconnect, or claim that this
   local result is the two-rank release certificate. Prove the real capability
   entry against the existing owned Model fixture, including both ranks,
   unhealthy communication, join errors and stream aliasing; preserve OFF.
   Approved API: sealed `check_communication_health()->Result<()>` and
   `quiesce()->Result<()>`, plus narrowly read-only actual paired-pool
   `validate_session(gpu)` (not allocation availability). Reject previously
   failed/closed sessions and active capture before joining. Observed health/
   synchronization errors latch the existing irreversible paired failure state;
   fully occupied healthy owners remain valid for quiescence. Neither operation
   can manufacture selected-server or pair-release authority.
2. Extend the existing local guard (not another guard framework) with its private
   inherited Spark channel, actual gated child-instance receipt, one-shot paired
   ticket and matching two-rank QUIESCENT release. Freeze a bounded byte format
   shared by its actual server consumer before implementing either side. Keep
   current pidfd/parent-death/lease protections and exact-instance targeting.
   The local subprocess proof must distinguish quiescent release from arbitrary
   exit0. Root reviews the exact protocol/API and two-node adapter before native
   deployment; controller/daemon loss must not depend on that failed link.
3. Server preflight validates the supervised channel/resolved fixed profile
   before selected GPU work. Actual owned-Model registration completes before
   worker/scheduler handoff. Select `Glm5MtpHead::new_paired` explicitly in the
   real factory; missing capability/authority must not fall back to ordinary MTP.
   First connected profile: TP2/EP2v2, two owners, MTP4, eager, BF16 KV,
   cold text2..1024, greedy grammarless, no prefix/logprob/adapter/swap routes.
   Independent nonspeculative C2..8 mode remains OFF on this separate profile.
4. Selected head admission precedes generic prefill/MTP arbitration. Arm the
   whole cold-prefill/selection/publication, bootstrap and F5/verdict/commit/E1
   operations, then reuse the existing checked retirement. Keep physical slot1
   when its peer retires. Worker arms before slot allocation/bind and each whole
   `ep_worker_step`, including preamble and follow-on accepted-count receive.
   Actual communicator health must pass before an operation completes. Issued
   Err/panic enters the terminal path before ordinary failure cleanup/unwind.
5. Selected shutdown stops admission, completes issued work, sends matched
   shutdown and retains all owners. Both actual Model quiescence receipts and
   the current-session pair certificate are required before non-returning
   `_exit(0)`. No generic scheduler drain/Model teardown/backend Drop follows.
   Failed or incomplete release remains supervised uncertainty, not success.

## Ownership and promotion

First Model slice is implemented and independently reviewed: sealed health and
quiescence methods, with real open/unfailed pool validation and the existing
irreversible failure latch. The actual capability-entry scaffold failed two
runtime checks before implementation; final focused4/4 and legacy3/3 pass,
including health becoming false after a successful join. Non-test and existing
test-support-feature checks pass. Scoped formatting/SPDX/caps pass; clippy is
blocked by four unchanged runtime stub argument-count errors, not claimed green.
Ten-file manifest SHA256
`bd50fc3506dbfed8e77e7132f8567f09c6acd9ad5be6a6984c9bbdf698c9681c` and raw logs:
controller `20260909/glm-c2-quiescence/closure.md`. These are local Model-fixture
checks, not GPU numerics, host drain, live admission or two-rank release proof.
Committed as `bdaa2558`; root's post-commit focused run again passed4/4 in0.22s
(`postcommit-focused.log` in the same campaign). The unchanged v29 native binary
was independently reproduced while this controller-only development occurred.
The shared guard/server record and wire contract is detailed in
`glm_c2_live_wire_plan.md`. The shared codecs, authenticated socket transport,
existing guard child extension and actual inherited server consumer are now
implemented. A controller-only fixture connects two actual gated children under
separate PID1/proc namespaces to the production consumer, with valid and rejected
ticket cases. See `docs/experiments/glm53-20260906/live-pair-handshake-results.md`.
The next slice now implements the production `--live` guard main, canonical
recipe codec, pinned startup-file ingress and two-rank quiescent release. The
actual guard ELF runs as private PID1 and execs the exact server ingress/release
source in the GPU-free fixture. Valid and delayed-peer release, bad/replayed
release, unreleased exit0, environment mismatch, consumed-session reuse and
hardlinked-input refusal pass. See `live-pair-release-results.md` in the same
experiment directory. This is not an actual Model quiescence, Docker resource,
native recipe or serving proof. Actual Model registration, selected factory/
scheduler/worker activation and the two-node Docker/controller adapter remain
the next connected integration step. Do not rerun the older PID1-death
qualification as if that prerequisite were absent.

First model slice: model author owns the sealed capability, its actual impl,
small quiescence child and existing Model-fixture extensions. No factory flag,
server/scheduler selection or native deployment is authorized by that slice.
Root owns the guard/server wire boundary and live admission integration; source
reviewer independently checks exact failure ownership and scope. Serialize
controller Cargo; keep benchmark clients and v29 artifacts unchanged during
the ongoing repeat. Every new Rust file<=500 lines and SPDX required.

Capture actual runtime RED at the production entry, then focused GREEN and
independent source review. Avoid broad redundant suites; evidence must cover the
connected behavior, not a fake capability or mocked completion certificate.
Subsequent qualification must exercise the real factory/admission/worker/driver
path plus actual local guarded children before a healthy bounded native C2 run.

Next connected implementation boundary (source rechecked after LIVE release):
move selected ingress ahead of runtime construction (`main` currently uses
`#[tokio::main]`, which violates ingress's single-threaded precondition if called
inside the async body). Carry the validated session through the actual resolved
startup into a private owner containing the boxed Model and terminal key before
`serve_load` hands ownership to either worker or scheduler. Do not activate from
an environment flag alone, or add another record-only admission validator.
Explicitly select the paired factory constructor without an ordinary fallback.
The selected worker branch must precede its current bind/alloc/log-break-free
loop; the selected head branch must precede generic scheduler setup/admission.
Use existing cold F0/serial/retirement paths, with actual communicator-health
checks before token emission/next E1 and before retirement response/removal:
an outer check after either helper returns is too late. Reuse the actual Model
fixture and subprocess terminal boundary for this dispatcher, not a mock sealed
capability. Cold-construction inner error cleanup still requires its own explicit
ownership treatment; later Model registration cannot retroactively contain it.

No fault injection, forced-kill experiment, reset or driver changes on Sparks.
Only after native quality/memory/clean-exit control should batching and higher
MTP concurrency be optimized. Serialized C2 alone does not satisfy C1>=30,
C2>=37,C4>=43,C6>=64,C8>=72 or the remaining serving/quality/parity requirements.
