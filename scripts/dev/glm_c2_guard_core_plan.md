# T3.1: local Linux guard core, before deployment authority

Status: T3.1 is committed (`caf585ed`) and standalone CPU-gated after actual
behavioral RED. Root's actual private-PID1 gate is also committed (`06561c10`)
and passed, including a post-commit run with unprotected descendant death and
outside-decoy survival. Do not repeat that work as a missing prerequisite.
Authoritative closure: controller campaign `20260908/glm-pair-guard-core/`
`postcommit-closure.md`, including exact source/executable hashes and raw receipts.
The live child ticket/channel, pair-disarm/controller adapter and actual T2
registration remain absent; no selected serving/deployment qualification follows.
This narrows [T3](glm_c2_supervision_plan.md), not an alternative supervisor.
No Spark/model/server edits, Cargo workspace registration, Docker/SSH, privilege,
node access, image packaging, GPU imports or native fault injection in this slice.

## Exact boundary and files

Build a standalone Linux helper under `scripts/dev/glm_pair_guard/`, using its
own `Cargo.toml`/`Cargo.lock` and `[workspace]` boundary. Only dependency proposed:
exact `libc=0.2.189`, already in the repository lock; verify offline availability
before building, never silently download/upgrade. All Rust files <=500 lines,
SPDX AGPL-3.0-only; no engine dependency or dependency feature changes.

Owned source: `src/{main,core,frame,child,linux}.rs`,
`tests/{lifecycle,lease}.rs`, and `src/bin/probe_child.rs` (explicit harmless CPU
test executable, never the serving entrypoint). Tests launch the actual helper;
the pure state tests additionally cover exact time boundaries without sleeps.
Root later owns any privileged private-PID-namespace harness, outside this slice.

`core::Policy` requires explicit startup/lease/challenge/frame/campaign/poll
durations and validated maxima; no implicit production defaults. `core::State`
accepts actual BOOTTIME samples and validated frames, emits finite actions only.
`linux::Io` is the narrow syscall boundary for clock, randomness, descriptors,
poll, read/write, signal and reap; `main::run(control_fd, child_spec, policy)`
routes those actions. Tests use the same live runner, not a side-model runner.
`child::prepare` returns a non-cloneable gated owner with actual pidfd; `release`
can occur once; `terminate` signals that owner only. No public raw-PID kill API.

The inherited AF_UNIX control socket is a trusted local harness/adapter boundary;
fresh OS-random 32-byte session and guard-instance identities bind every frame.
They are NOT Docker verification, a paired capability, T2 authority or proof that
both nodes exist. No environment/ticket file can activate selected model code.
Later T3.2 must verify full immutable container/image/recipe identities and both
actual gated children before issuing its pair ticket; T2 consumes that live
authority separately. Do not pre-implement either interface with fabricated IDs.

## Child lifetime, before any release

The actual helper starts single-threaded and remains so; its creating thread
lives for the guard lifetime. Prebuild argv/env, descriptor lists and executable
validation before `fork`. Use direct single-thread fork, not blocking
`Command::spawn` pre_exec: spawn waits its exec-error channel and would deadlock
if the child awaited a release sent only after spawn returned.
Child sets `PR_SET_PDEATHSIG(SIGKILL)`, checks its real parent against the
pre-fork parent, reports readiness and blocks on a pre-exec gate. After fork use
only reviewed syscall/async-signal-safe operations, no Rust allocation/log/Drop.
Parent obtains pidfd before gate release, retaining normal SIGCHLD ownership
(no SIG_IGN/SA_NOCLDWAIT/other reaper). Readiness and setup have finite deadlines.
If setup fails, close the unreleased gate: EOF makes the child `_exit` without
exec; no numeric-PID signal fallback. Parent death before setup is caught by
the parent check, afterward by PDEATHSIG. Test both, plus exec failure.
Reject setuid/setgid/file-capability executables and missing syscall support;
no credential changes or shell wrapper. Pin the validated executable inode via
an owned FD for exec, not a mutable-path second lookup. Preserve PDEATHSIG across
ordinary exec; keep only intended FDs, restore the child's intended signal mask.

## Bounded frames, lease and terminal behavior

One inherited nonblocking control connection; versioned fixed-width binary
frames with length <=4096, exact schema/identity/type checks, no serde/unbounded
collections. One partial frame, one outbound frame, no queued renewal backlog.
Check BOOTTIME against every deadline BEFORE reading/parsing/accepting each
frame, including equality; terminal never rearms. Bound each poll turn to one
frame/4096 bytes and recheck time, so flood/trickle cannot starve expiration.
Partial-frame and blocked-output deadlines do not reset when bytes arrive.
Log/receipt output is bounded/nonblocking and cannot postpone kill/expiry.

Only one outstanding random challenge and strictly increasing nonwrapping
ordinal. Renewal consumes that challenge once and anchors its new deadline to
challenge ISSUE time, capped by the immutable campaign deadline; reject expired,
replayed, wrong-instance, duplicate and overflow messages. Check policy arithmetic
and time regressions fail closed. No delayed packet may resurrect an old lease.
Use explicit SIGTERM/INT/HUP ingress (blocked signals plus signalfd), including
when later PID1; signalfd/pidfd/control share the bounded loop. No silent default
PID1 signal assumption. On any terminal condition, latch first, send SIGKILL by
held pidfd, and observe exit only with bounded polling/nonblocking wait. Kill or
reap failure is failed evidence, never permission to drop identity and kill by PID.
Guard panic exits without unwinding; PDEATHSIG remains the death backstop.

There is intentionally NO successful-disarm/normal-child-release command here:
even child exit0 is terminal. Later pair-drain requires both actual receipts,
finite normal-exit deadline and retained pidfd until actual exit0. Core test
success means the expected containment behavior, not a clean serving shutdown.
Lease timing assumes the guard is scheduled/runnable and the kernel responds;
SIGSTOP/frozen guard, uninterruptible driver calls and host failure have no
promised recovery/stop bound. This slice neither detects GPU progress nor proves
private-PID1 death semantics or remote peer supervision.

## Actual TDD and handoff gates

Capture RED before fixing each boundary: actual pre-exec gate cannot run child
early; parent death before/after PDEATHSIG, guard death after exec; lease expiry
terminates only retained child while an unrelated decoy survives; Drop/atexit
witnesses absent after forced terminal path. Actual harmless child exit0/nonzero,
exec error and control EOF must fail closed. No fabricated pidfd success.
State tests cover exact deadline equality, issue-time anchoring, ordinal overflow,
campaign cap and terminal replay. Actual process tests cover partial/truncated/
oversized/invalid frames, trickle/flood, blocked output, signals, renew then loss,
and delayed renewal after expiry, with bounded outer harness timeouts/cleanup.
Fork/parent-race rendezvous belongs to a local test adapter of the real child
primitive, never a runtime fault flag or test-only alternate lifetime algorithm.
Record actual syscall absence as a failed/missing prerequisite, not an ignored
test promoted to PASS. Root separately runs a disposable CPU-only PID1 namespace
death gate: successful namespace creation already observed is not that proof.
Freeze source/lock hashes, raw RED/GREEN, fmt/clippy/SPDX/caps and independent
review before root decides any T3.2 controller work or T2 live registration.

Receipts: `/home/abc/storage/models/atlas-campaigns/20260908/glm-pair-guard-core/`.
`RED_RESULTS.md` records the actual deadline/SIGTERM witness failures;
`GREEN_RESULTS.md` gives exact commands and coverage (including duplicate state
test invocations), limitations and source manifest. No engine code was changed.

References: [Rust spawn implementation](https://doc.rust-lang.org/src/std/sys/process/unix/unix.rs.html),
[pidfd](https://man7.org/linux/man-pages/man2/pidfd_open.2.html),
[pidfd signal](https://man7.org/linux/man-pages/man2/pidfd_send_signal.2.html),
[parent death](https://man7.org/linux/man-pages/man2/PR_SET_PDEATHSIG.2const.html),
[signalfd](https://man7.org/linux/man-pages/man2/signalfd.2.html).
