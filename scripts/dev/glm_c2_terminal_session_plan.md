# GLM paired session: terminal containment plan

Status: T1 implemented and committed; T2/T3 live integration remains absent.
The 2026-09-09 quiescent-exit amendment below is root-reviewed and approved
for implementation, not qualified for native activation.
This is local process containment, not a GPU recovery mechanism or peer watchdog.

## 2026-09-09 healthy-exit amendment

For the first supervised, fixed-profile serial C2 implementation, replace the
healthy-shutdown requirement for explicit allocator teardown with both-rank
quiescence followed by non-returning `_exit(0)`. Other operation boundaries and
the Result-owner audit below are unchanged. This is an intermediate deployment
profile, not a reduction of the reference-parity goal or its serving requirements.

Stop admission, finish every issued transaction, and send the existing matched
shutdown command. Retain the actual Model and sequence owners. Both adapters
then require the actual communicator's `is_healthy()` probe before and after
directly synchronizing the actual default and distinct secondary streams,
stopping on the first error. The live selected-operation completion boundary
also checks communicator health before success can lead to retirement. No free,
F1, teardown, backend Drop or sweep follows an error. Keep T1
armed while waiting for the current session's two-rank QUIESCENT release; only
that release permits `_exit(0)`. No ordinary main cleanup follows success either.

Source audit: `sync_secondary_dispatch` only enqueues an event wait and is not
a host join. NCCL also owns a communication stream even with overlap disabled;
successful `all_reduce_async` and `peer_exchange_async` enqueue their completion
event back onto the caller's compute stream. The fixed profile uses the Model's
default stream, which topology passes as NCCL's legacy stream. Successful final
shutdown/default-stream synchronization therefore joins those successful
communication operations transitively. This argument does not cover failed
event recording/waits: those must already enter the armed terminal boundary.
The active health probe is necessary because collective APIs can latch an async
NCCL error yet return Ok. No reconnect is permitted.

Implement this as one usable fixed-profile vertical path: live guard channel and
actual Model registration, selected head admission/prefill/serial driver/retirement,
selected whole-worker loop, explicit paired factory construction, and two-node
supervisor startup/lease/quiescent release. Do not add another generic ticket or
per-command acknowledgment framework. Native admission still requires review
of all actual error-owning callees, focused adapter/no-Drop and real guard-pair
CPU evidence, and a pinned bounded-memory recipe. Quiescent process exit does
not by itself prove driver resource reclamation or hardware recovery.

## Why the existing paths cannot own selected failure

Read together with `glm_c2_serial_scheduler_audit.md` and the Partition B plan.
Current concrete sites are:

- `scheduler/verify_dflash_step.rs::step_verify_dflash_inner` marks a request
  finished and returns on header/payload/target/accepted-count/record errors.
  Its later emission can return before trim/commit. A fence around only the
  target verifier therefore misses essential parts of the issued transaction.
- `scheduler/lifecycle.rs::finish_sequence` caches a request, logs a failed free,
  and then sends F1 anyway. `send_error` also continues after failed cleanup.
- `scheduler/mod.rs::run` drains requests, sends F1/shutdown, synchronizes, and
  explicitly tears down even after a failed synchronization.
- `serve_phases/build.rs::maybe_run_ep_worker` owns the model in a worker thread;
  a command error breaks its loop, frees all slots, then returns success. A panic
  unwinds that thread before its parent observes `join()` failure.
- `model/impl_a2.rs::ep_worker_step_impl` receives the whole v2 preamble before
  routing. F1 takes the old sequence into a local owner. F5 dispatch continues
  through the accepted-count receive, rollback, record, trim, and commit.
- `main.rs`'s startup escape and tail restore the TUI/flush logs before exiting.
  The existing fatal-GPU latch classifies context loss and exits with 70; it must
  not be reused to label an unproven transport/completion failure as context loss.
- `tui/events.rs::run` installs the TUI panic hook late. That hook restores the
  terminal and locks/dumps the log ring before chaining to its predecessor.
- `AtlasCudaBackend::drop` calls `sweep_unreleased`, independently of Model-level
  quarantine. Ordinary unwinding can consequently attempt native frees.

`catch_unwind` around the scheduler or worker is **not** pre-unwind containment:
inner destructors execute before the catcher receives the panic. A panic hook
must intercept before those destructors and before potentially blocking prior
hooks. A Result-returning adapter has a different limitation, discussed below.

## Scope and activation prerequisite

The ordinary model/server paths remain byte-for-byte in behavior: no new fatal
policy for C1, nonspeculative C1..C4, other models, or unselected GLM. The optional
actual `Model::glm_paired_execution()` capability is necessary but not by itself
sufficient to admit serving. No environment-only flag may manufacture authority.

First native activation also requires an independently reviewed out-of-band
supervisor with an exact two-process launch identity. It is absent today. Missing
supervision must be rejected before selected model work/worker ready/public
admission; this slice may CPU-test containment without enabling that admission.
Local child termination cannot bound a peer blocked inside a synchronous driver
call. Do not deploy a selected session merely because its local exit tests pass.
The current detached named-container launcher is not such a supervisor. Future
registration must pin full immutable container/process instance and image IDs,
disable automatic same-name restart, and reject stale-name/PID reuse. Monitoring
failure is itself terminal for the exact pair. If controller/network-loss
containment is claimed, controller-only SSH actions are insufficient: a per-node
expiring watchdog lease (or equivalent independently reviewed local mechanism)
must enforce it without relying on the failed controller link.

Selected processes initially disallow hot-swap, multiple model sessions, TUI,
and in-process reconnect/re-registration. These are resolved launch restrictions,
not changes to the ordinary defaults. Cold load/construction failure before
selected session registration is not covered by a later serving guard; admission
must not imply otherwise.

## Exact live session authority (T2, not T1)

Use one small server-private `SelectedSession`, not a generic transaction system.
Its private registration occurs at the actual owned-model handoff in
`serve_load.rs` / `maybe_run_ep_worker`, before spawning the GPU-driving thread,
binding it, allocating selected worker slots, or publishing ready.

Registration receives:

- The actual live Model reference and its non-None sealed paired capability.
  Record their identity together, rather than trusting the model name or an HTTP
  request field. Operation entry must compare against that same live model.
- The already resolved rank/world/profile from the actual build handoff, not a
  second environment parse or fabricated model defaults.
- An explicit supervisor launch ticket: a fresh shared launch nonce, resolved
  image/recipe identity, and exact local/peer process-instance registrations.
  The nonce originates at the supervising launch, never PID/time/device address.
  No ticket parser/admission flag is added in the initial containment-core slice.

A private, lifetime-branded local handle ties operations to this registration.
Process-local registration is one-shot; closing it does not make its identity
reusable. A different model, old ticket, different rank, stale handle, or second
registration cannot rearm it. Addresses alone are not lifetime identity: the
one-shot registration and private handle are essential even if an allocator
later reuses a Model address. Request slot/target position are diagnostics and
remain checked by actual Model preflight; they are not replacements for its
private lease generation or permission to mint a Ready object.

The panic ingress reads only immutable prepublished session data plus atomics.
Registration and hook installation finish before the model is handed to a
thread that can issue selected work. No mutex, allocation, formatting, device
read, communicator query, or logging lock is permitted in the fatal ingress.
An intentionally process-lived registry avoids dereferencing a freed record
from a concurrently executing panic hook. Do not add reset-for-tests to it;
global-path tests run in separate processes.

## Minimal terminal API and state

T1 provides only a server-private production core, no actual Model/launch/ticket
registration and no activated selected branch. Proposed module
`glm_terminal_session.rs` with small private children. Concrete T1 interfaces:

```text
TerminalCore::new() -> TerminalCore
TerminalCore::activate() -> Result<SessionKey<'_>>
SessionKey::begin() -> Result<InFlight<'_>>
InFlight::require(Result<T>) -> T
InFlight::complete(self)
TerminalCore::fatal() -> !
TerminalCore::panic_if_live()             // returns only when not selected-live
panic::install(&'static TerminalCore)     // private implementation
install_panic_ingress()                  // installs the actual inert static core
```

The production static core starts inert and T1 has no serving activation caller.
`activate` is private to the module/its later registration sibling; tests exercise
that real production core directly, not a counterfeit Model capability or Ready.
The key borrows its originating core and activation is one-shot. An operation
is bound to that key; completion cannot affect another core or a completed
operation. Core states are never-selected, selected-idle, in-flight, and closed;
closed is not eligible for reactivation. Drop never disarms. T1 tests prove
these core transitions and actual no-unwind exit only, not live-model identity,
rank agreement, protocol coverage or supervisor admission.

T2 adds `SelectedSession::register(actual_model, resolved_launch,
supervised_ticket)` and binds the private core key to the exact authority above.
It owns actual-model/launch registration tests and operation metadata (slot,
target position). It cannot make the T1 tests retroactive proof of that binding.

Names can change, but the ownership contract cannot. In T2, `begin` precedes the first
potentially side-effecting GPU operation or header/idle-receive attempt, including
a header call that ultimately fails. Immutable request rejection happens before
`begin`. There is one local operation at a time; nested/foreign/stale completion
cannot clear it. `complete` is explicit and operation-specific; a guard's Drop
must never clear uncertainty. An abandoned live guard is itself terminal, but
that is only a backstop: panic ingress must run first, before guard Drop.

`require` exits on Err before returning control to any legacy request failure,
cleanup, ordinary drain, or owning caller destructor. It must neither stringify
the error nor call an injected sink that returns/panics in production. Tests of
the actual non-returning branch use a subprocess, not catch_unwind as a fake
terminal sink.

Use a distinct selected-session-uncertainty exit status (proposed 74, not the
existing GPU fault status 70). The Linux implementation calls `libc::_exit`,
using the existing dependency: no Rust unwinding, C atexit, TUI restoration,
tee flush, reconnection, `cudaDeviceReset`, extra synchronization, or cleanup.
Do not use `abort` and risk an expensive core dump. Terminal diagnostics must
not delay exit: the supervisor already knows the ticket/process identity and
can observe the exit status. Any future best-effort event uses a pre-established
nonblocking channel, fixed bounded data, and must not become a completion gate.

While a selected session is live, a panic on **any** local thread is terminal,
including idle time and a pre-issue validation panic. This is deliberately more
conservative than the recoverable ordinary Result rejection before issuance.
An unrelated HTTP request's normal error is not a panic and remains ordinary.

## Panic/main integration

T1 installs the inert-core-aware dispatcher early in `main`, before selected
loading or thread creation. In the unselected case it chains to the existing hook with
unchanged behavior. Prevent the later TUI hook from doing work first: add the
same selected check as its literal first action, or centralize its installation
under the dispatcher after source review. Prefer the first-action check as the
smallest change. The selected launch rejects TUI nevertheless; hook order must
not rely solely on an advertised flag.

Selected panic dispatch skips all previous hooks. Subprocess tests install a
prior hook that would write a sentinel and prove it was not invoked. Test both
plain and late-TUI-hook installation orders. A panic caught by an inner caller
still invokes the global hook first and therefore cannot resume a selected
session. Hook changes themselves are startup-only; never take/set a panic hook
from the panic handler.

The existing main exits retain their ordinary status mapping. A selected
startup/serve cancellation that encounters an in-flight operation cannot take
the generic TUI/normal-drain path. It invokes the terminal sink instead. A clean
selected stop is allowed only through an explicit matched drain/completion path;
failure at its wait/free/teardown boundary is terminal. Do not relabel a timeout
or cancelled request as proof the device became quiescent.
Healthy disarming requires an explicit both-rank post-drain receipt; observing
an exit status of zero alone does not prove that receipt existed.

## Complete operation boundaries

| Operation | Arm before | Complete only after |
|---|---|---|
| Initial selected slot/setup | First potentially mutating selected alloc/bind/setup call | Actual setup succeeds; no ready publication on failure |
| Head cold prefill | First F0/header attempt or selected local GPU preparation, whichever is earlier | Whole single chunk, normalized capture, synchronous P-1 primer, owned tail publication and initial selection |
| Worker command | First idle-command receive attempt | Actual addressed command and all follow-on receives/state completion return successfully |
| Head bootstrap | Header/token dispatch or first target write | Actual target decode and immediate owned H[P] copy/completion |
| Head E1 | First selected E1 header attempt after B1 preflight | All payloads, real private repair/proposal, returned IDs and required completion checks |
| Head F5 | First F5 header attempt after B1 preflight | K/tokens, actual K5, checked selection, accepted-count exchange, target token rollback, owned record, target commit and proposer trim |
| Selected retirement/F1 | First local retirement operation or F1 header attempt | Local retirement plus complete matched worker command; no replacement if cleanup fails |
| Selected shutdown | First matched drain/wait/free operation | Both operation completion and explicit successful teardown; no free-anyway branch |

The worker must stay armed across all of `ep_worker_step`, not merely its target
Model call. It cannot disarm after receiving K5 or after target logits become
available; accepted-count receive/read errors and invalid count are fatal too.
Its existing idle classification still exempts only normal command waiting from
duration-health poisoning. Terminal containment does not add a command timeout.

Head visible token emission occurs after the complete F5 commit boundary. EOS,
output cap, cancellation or disconnected response sink then suppresses E1 as
appropriate; none may skip completion of an already issued F5. The selected
driver does not call generic `step_verify_dflash` and hope its swallowed errors
will reach `require`. Likewise, selected prefill/retirement need actual Result
boundaries rather than wrappers around void helpers that already free/continue.

Do not add a new per-command acknowledgment simply because the head cannot
observe the worker's host return. The existing serial worker loop cannot receive
the next command until the previous command's trim/commit returns, and actual
stream/event fences may already protect its resources before subsequent use.
T2 must audit the exact local allocation lifetimes, worker receive order, stream
fences, and next-command dependencies. The ordinary local operation can complete
when those actual ordering guarantees establish the required safety; it need not
wait for an otherwise redundant remote host acknowledgment. Require a new ACK
only for a demonstrated unmet dependency/lifetime boundary, with its exact proof
and cost reviewed separately. Clean-shutdown/supervisor disarming is distinct:
there is no subsequent command whose serial order can stand in for a post-drain
both-rank receipt. The out-of-band supervisor detects peer termination/stall; it
does not fabricate successful completion evidence.

## Result-return limitations: a mandatory closure gate

An outer `require(callee())` cannot retroactively stop destructors or explicit
cleanup that ran **inside** `callee` while it returned Err. This differs from
panic interception and must not be concealed by a subprocess test that only
places sentinels in the outer caller.

Before claiming selected post-issue Err has no GPU-owning destructor path,
enumerate the actual F0/bootstrap/E1/F5/F1 callees and their local owners. Keep
Model, SequenceState/SlotGuard, private leases and device-owning workspaces in
outer live owners, borrowed by the operation. For each resource-owning local
or explicit failure cleanup, either prove its selected error branch already
quarantines without free/reuse, or place the terminal check before that cleanup.
F1's `slots[slot].take()` local owner is a concrete audit target, as are selected
allocation failure and teardown. CPU tests must cover these actual sites.

If this requires an inner model failure ingress, propose its exact selected-only
site/API as a reviewed addendum with the Model author; do not silently introduce
a global error policy to GpuBackend/CommBackend or claim the server wrapper alone
solved it. Host-only Vec/error/lock-guard destruction before an Err reaches the
server is not a GPU-owner-free proof and is not itself a native failure.

## Bounded implementation partitions and file ownership

T1, after approval: server-local terminal core/sink and early panic ingress,
plus real subprocess Drop tests. Exact files:

- `crates/spark-server/src/glm_terminal_session.rs`: inert static/core access and
  the narrowly owned production terminal sink.
- `crates/spark-server/src/glm_terminal_session/core.rs`: one-shot core, borrowed
  keys and operation state; no Model/launch/ticket dependencies.
- `crates/spark-server/src/glm_terminal_session/panic.rs`: core-aware pre-unwind
  ingress and previous-hook chaining while unselected.
- `crates/spark-server/src/glm_terminal_session/tests.rs` and, if needed for the
  500-line cap, a private subprocess-test child.
- `crates/spark-server/src/main.rs`: module declaration and early inert ingress.
- `crates/spark-server/src/tui/terminal_guard.rs`: literal first-action ingress
  check before terminal/log restoration in its later hook.

The normal main/fault/TUI behavior is characterized first. T1 has no live-model
registration, launch-ticket integration, selected factory/driver caller, or
admission flag. No cross-crate actual-model fixture is added merely to test T1.

T2, reviewed after B1 transport and the Result-owner audit: exact actual worker
loop integration in `serve_phases/build.rs` (extract a small worker child if
needed), registration at `serve_load.rs`, and the selected serial driver's
prefill/bootstrap/F5/E1/retirement/drain callsites. These are not a separate
generic transaction framework. Tests drive the real adapters and the actual
paired Model capability; they may not mint a mock Ready/capability. Do not widen
existing ordinary helpers or reopen the public C2 E1 gate as a shortcut.

T3, required before native admission: independently implemented/reviewed exact
session supervisor and completion protocol. Its files and process-launch policy
need their own plan. T1/T2 cannot substitute for it or claim hardware safety.

Keep every new Rust file <=500 lines; preserve SPDX and the source/Cargo windows.
Root alone commits, builds/deploys native images, and operates either Spark.

## Behavioral test gates

1. T1 core tests directly exercise one-shot activation, key/operation identity,
   rejected duplicate/stale completion, inert behavior, and no Drop-disarm.
   They do not claim actual Model or supervised launch registration. T2 actual
   selected registration rejects absent capability, mismatched/stale
   ticket/model/rank, duplicate registration and invalid operation completion
   before GPU/command work. Unselected calls preserve ordinary behavior.
2. T1 child-process positive controls complete a core operation and permit its
   explicit healthy continuation; direct Err controls exercise the real sink.
   T2 independent fault children inject actual
   adapter Err at the first header and each meaningful payload/target/selection/
   accepted-count/record/trim/commit/free boundary. No F1, shutdown, further
   collective, pool recycling, backend free/sweep or caller Drop marker follows.
3. T1 panic children place Drop sentinels **inside** the guarded nested
   core operation as well as outside it. Deliberately catch the panic inside another
   closure: selected hook must still terminate before any sentinel/prior hook.
   Unselected controls retain ordinary catcher/Drop behavior.
4. T2 covers actual F5 accepted-count receive/copy/invalid-count on worker and head,
   not just errors from `verify()`. Existing owned-byte fixtures supply exact
   command traces; no CUDA arithmetic claim comes from the test backend.
5. T2 covers selected cold prefill target/capture/primer, initial bootstrap owned
   copy/completion, early retirement failure, teardown wait/free failure and
   repeated attempts. Test uncertainty never becomes clean merely because a
   later synchronization succeeds.
6. T2 separates pre-issue validated rejection (normal error, zero commands) from
   post-issue fatality. Test cancellation before issuance, during F5, after
   commit, and during shutdown. Match both-rank operation completion semantics.
7. Future supervisor subprocess tests bind exact launch instances, reject stale
   PID/nonce reuse, stop admission for the whole pair, and bound the policy for
   one terminated/stalled peer. Do not simulate success by reconnecting NCCL.

Capture actual runtime RED before implementation and exact-source GREEN,
independent review, and separate postcommit CPU qualification. Never fault-inject
GPU, NCCL, memory exhaustion or process death on either Spark for these tests.
Healthy native correctness/memory/clean-stop gates come first after all admission
prerequisites; local `_exit` cannot guarantee CUDA-driver or hardware recovery.

## T1 implementation and development receipts

Persistent controller receipts live in
`/home/abc/storage/models/atlas-campaigns/20260908/glm-terminal-session/`.
The copied `cpu-command.sh` records the exact server CUDA-feature link command,
offline shared CPU target and official NVIDIA driver link-stub path. The stub
is not a GPU backend or numerical proof; these child tests call no GPU API.

- `red.sha256` and `behavioral-red.log`: actual compiled production core with
  `std::process::exit(74)` and a prior-hook-only panic ingress. Two tests passed,
  two failed at runtime: the error path ran its registered C `atexit` handler;
  selected nested panic reached the prior hook, inner destructor, catch resume,
  caller destructor and ordinary exit. These are behavioral RED, not compile
  failures or injected fake sinks.
- `focused-green.log`: all four tests passed after the real `_exit` sink,
  first-action panic ingress and actual TUI-hook first-action check.
- `focused-thread-green.log`: all four tests passed after adding a real spawned
  thread's selected-idle panic and actual inert-TUI positive control. The tests
  cover eleven subprocess modes: three ordinary/closed controls, four terminal
  error/abandonment modes and four panic modes. Selected children exit 74 without
  registered C, Rust, prior-hook, catch/join-resume or TUI-log witnesses.
- A second actual core cannot complete the first core's operation; duplicate
  activation/begin, forgotten operation and reactivation after close are covered.
  Consuming Rust handles prevent safe-code stale operation completion; no
  unsafe fabricated handle or test reset is added to claim that property.

These are development-tree server receipts with model production at `67640368`;
parallel model transport tests/fixture edits are `cfg(test)` and not linked into
this server test binary. Broad exact-source and postcommit gates are still
separate requirements. New Rust files are 28/130/11/219 lines. Existing main's
dead-code allowance is unchanged; there is no new lint suppression to hide the
deliberately absent serving registration. Unix/macOS compilation is preserved by
using the existing libc `_exit` API; no macOS execution gate has been run here.

The panic hook is installed at the start of the async main body, after Tokio's
runtime construction but before server/model loading and selected registration.
T1 does not claim pre-runtime startup failure containment. No ordinary Result
error is routed into the core and there is no admission flag, live-model binding,
peer watchdog, remote acknowledgment, CUDA operation or native deployment.
