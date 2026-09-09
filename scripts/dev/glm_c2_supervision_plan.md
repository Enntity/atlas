# T3: exact-session supervision for the first paired control

Status: DESIGN ONLY, awaiting root review. No implementation, dependency,
deployment, node access or fault injection is authorized. Read the terminal
session plan and serial scheduler audit; T2 registration is still a prerequisite.
Implementation must be split: the proposed first slice is
[the local guard core](glm_c2_guard_core_plan.md), not this entire design.
Docker/SSH identity and controller logic, pair tickets, clean-drain certificates
and actual T2 registration remain separate, unapproved later slices.

## Small deployment-specific shape

Use one controller and two identical Linux-only, GPU-free guard processes.
Each guard is the exec-form PID1 of its own private-PID-namespace container;
it starts exactly one Spark child, never restarts it, and owns its pidfd.
This is node-local even if Docker/SSH/controller later becomes unavailable.
The guard has no CUDA/NCCL imports, engine dependency or generic job API.
Unlike `start-glm53-ep2.sh:731,745,835`, do not delete/reuse standard names or
start Spark immediately with `docker run -d`. Preserve that legacy launcher.
The v26 controller trap's named `docker stop --timeout 30` is also not this
selected failure policy; it depends on a surviving controller and graceful drain.

## Identity and startup gate

1. Controller generates256 random bits with the OS RNG; creates fresh0700
   session directories on both nodes with exclusive creation, refusing reuse.
   Freeze both image IDs, Spark/guard executable hashes and canonical argv/env/
   mounts/resource-limit recipe hash. Resolve tags once; never launch by tag.
2. `docker create --restart=no` both containers, nonce-labelled unique names,
   guard entrypoint, private PID namespace, no shell wrapper or `--init`/hostPID.
   Capture full64-hex `.Id`, full `.Image` and inspect exact config on each node.
   Neither container runs yet. Reject ID prefixes, unexpected restart policy,
   image mismatch, existing session records or altered command/mount limits.
3. Start those exact IDs. Guards wait without starting Spark, with a finite
   startup deadline. Controller verifies both guard instances and their fresh
   random instance challenges, host boot IDs and actual container init identity.
   A durable consumed marker prevents restarting the same container/session
   with old credentials; no automatic recovery or manual same-session rearm.
4. Only after both guards are armed does the controller issue the pair manifest.
   Guard launches Spark as its direct child, establishes parent-death SIGKILL
   before exec, checks the parent race, and obtains pidfd before releasing the
   child startup gate. No SIGCHLD auto-reap or separate reaper before pidfd_open.
   Refuse missing pidfd support, setuid/file-capability executable or credential
   transitions that would clear parent-death protection. No PID-number fallback.
   Both actual child instances must be observed while still gated before the
   final pair ticket is issued; supplied future child IDs are not evidence.
5. A private inherited socketpair carries the one-shot T2 ticket and child
   receipts. Keep only the intended endpoint in each process; close unrelated
   FDs. Ticket binds schema, session+guard challenges, both full container/image
   IDs, boot IDs, rank, recipe/binary hashes and actual child instance identity.
   T2 checks the actual parent/child instance and consumes this live channel with
   the actual paired capability, not JSON/environment fields alone. No public
   HTTP request, CLI flag or copied ticket file can mint supervised authority.

## Lease and failure policy

Controller-to-guard messages use bounded framed local Unix sockets via existing
authenticated SSH, not a new public TCP service. Restrict endpoint permissions,
validate session/instance/schema/rank on every frame and cap messages at4KiB.
Treat root/Docker administrators and the pinned executables as trusted; this is
not containment of a malicious privileged model or compromised host.
Guarding process lifetime also does not prevent cleanup inside a failed cold
Model constructor before T2 registration; that existing limitation remains.

Initial proposed heartbeat period5s, node lease30s, guard poll<=250ms, each
controller I/O deadline<=3s; freeze these explicitly in the reviewed native
recipe, never infer them from request timeout. Each renewal needs fresh status
from BOTH guards and one outstanding per-guard random challenge/strict ordinal.
Anchor the renewed deadline to local challenge-issue BOOTTIME, not packet arrival:
late/replayed buffered responses cannot prolong a dead controller's lease.
Check expiry before processing every frame; an expired lease cannot be revived.
One failed/missing peer check stops both renewals permanently; direct revocation
is best effort, local expiry is independent of the failed network/daemon.
Do not renew from ready-log text, a detached heartbeat thread without peer checks,
or HTTP success alone. Keep an explicit nonrenewable campaign deadline and
controller request/load deadlines: guard heartbeats alone do not prove GPU work
progress. All waits/log writes are bounded/nonblocking in the guard event loop.

On child exit before clean release (including exit0), malformed channel, revoked/
expired lease, guard shutdown signal or deadline: latch terminal, refuse all
renew/rearm, signal ONLY the retained Spark pidfd with SIGKILL, and report failure
without attempting Spark SIGTERM, F1, NCCL shutdown, allocator cleanup or reset.
No successful reap is assumed if a driver syscall is uninterruptible. Guard
panic/fatal exit must not drop/restart Spark; parent-death protection and private
PID1 lifetime are backstops. Controller also requests `docker kill --signal=KILL`
for each pinned full ID when reachable; it never broadens to a name/PID search.
Failed inspect/kill/reap is a failed receipt, not permission to force-remove,
rename a running container, rebuild or try again on those nodes.
Explicit signal ingress is required for PID1. Timer bounds assume a runnable,
scheduled guard and functioning kernel: SIGSTOP/frozen processes are not death,
and neither parent-death protection nor namespace lifetime enforces that timer.

## Healthy shutdown is a different protocol

Root-approved amendment, 2026-09-09: the first fixed-profile implementation uses
quiescent process exit rather than explicit allocator teardown; see the exact
stream/communicator and failure contract in `glm_c2_terminal_session_plan.md`.
T2 stops admission, completes every issued command, sends the matched shutdown
command, retains Model/sequence owners, and performs strict actual-stream joins
and the actual communicator health probe. Each still-live Spark sends its guard
a `QUIESCENT(session, drain_epoch, last_command, rank, child_instance)` receipt
only AFTER that succeeds, and waits without more GPU work while T1 remains armed.
Controller validates BOTH matching receipts and sends a pair-disarm certificate
containing both receipt digests. Each guard validates it before allowing its
child's non-returning `_exit(0)`; a one-rank receipt or arbitrary exit0 is insufficient.
No Model teardown, backend Drop/sweep or normal main cleanup runs on this path.
Keep leases running through this handshake; lost certificate/peer/controller is
terminal, not inferred clean. Partial delivery after both drained is safe to
classify as FAILED, never a both-rank clean pass. Final success needs both actual
exit0, OOMfalse, no outstanding issued operation or unresolved error evidence,
and preserved full logs. Retained Model/sequence allocations are expected.
Disarm permits bounded process exit, not immediate release of the guard's pidfd;
retain child supervision until actual exit0 or a finite exit deadline fails.
This adds a shutdown-only receipt, NOT a per-E1/F5 acknowledgment framework.
Process exit alone does not prove successful driver reclamation or hardware
recovery; native post-exit health and memory checks remain required.

## Files and CPU gates after approval

- `scripts/dev/glm_pair_supervisor.py`: exact two-node Docker/SSH adapter and
  bounded controller loop; injected transport boundary for CPU tests only.
- `scripts/dev/glm_pair_guard/Cargo.toml` plus `src/{main,protocol,child,tests}.rs`:
  standalone deployment helper, outside engine/workspace membership; existing
  libc/serde dependencies only if confirmed cached, pinned with its own lockfile.
  Every Rust file<=500 lines; no Python dependency in the serving image.
- `scripts/dev/test_glm_pair_supervisor.py`: actual local guard/child subprocesses
  plus recording Docker/SSH endpoint; no Docker daemon or Spark model required.
- T2 later owns its real ticket/drained-channel consumer and checked Model binding;
  root later owns recipe/image packaging. No changes to legacy launcher now.

TDD: positive gated startup/renew/two-rank drain; every stale nonce/instance/
full-ID/image/config/restart mismatch; same-name and PID-reuse decoys untouched;
one rank exit0/74, lost controller/one link, expired/replayed/delayed renewal,
oversized/truncated frame, blocked log/transport, guard death and parent-race
failure. Use real pidfd/parent-death child tests and Drop/atexit witnesses;
test unknown completion never invokes graceful child cleanup. Injected transport
tests do not prove Docker identity behavior. A local disposable private PID
namespace gate must independently check PID1 death semantics; unavailable
namespace support is a missing gate, never mocked into PASS. No native fault
injection, forced-kill qualification, or GPU/driver/hardware recovery claim.

## Primary references (read2026-09-08)

Docker documents create-before-start and full configuration preparation:
[container create](https://docs.docker.com/reference/cli/docker/container/create/).
Explicit no-restart is essential; Docker also warns against competing managers:
[restart policy](https://docs.docker.com/engine/containers/start-containers-automatically/).
Signal target/exec-form caveats support using exact IDs and avoiding shells:
[container kill](https://docs.docker.com/reference/cli/docker/container/kill/).
Linux documents stable process descriptors, parent-death races/exec restrictions
and private namespace init lifetime:
[pidfd_open](https://man7.org/linux/man-pages/man2/pidfd_open.2.html),
[parent-death signal](https://man7.org/linux/man-pages/man2/PR_SET_PDEATHSIG.2const.html),
[PID namespaces](https://man7.org/linux/man-pages/man7/pid_namespaces.7.html).
These justify a design; they do not qualify this undeveloped adapter on GB10.
