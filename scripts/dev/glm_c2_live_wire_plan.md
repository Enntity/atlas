# LIVE paired guard wire and quiescent release

2026-09-09. Root has authorized the bounded wire, inherited-channel transport,
actual child primitive and server consumer implementation below. Native admission
remains disabled until the connected path and literal recipe are qualified.
Read with [live integration](glm_c2_live_integration_plan.md) and
[terminal session](glm_c2_terminal_session_plan.md). Root owns live integration.

## Current facts and ownership

The GPU-free guard already has a single gated ELF child, pinned executable FD,
PDEATHSIG/parent-race checks, held pidfd, finite BOOTTIME lease, replay-resistant
renewals, and private PID1 qualification. Core `caf585ed` and PID1 harness
`06561c10` passed their actual postcommit CPU gates: see campaign
`20260908/glm-pair-guard-core/{postcommit-closure.md,pid1-postcommit-run.log}`.
Private PID1 death killed its unprotected descendant while an outside decoy
survived. Do not reopen that completed prerequisite. It does not prove live
Docker pairing, server registration, GPU recovery, or this release protocol.

This slice now implements the explicit LIVE child environment/channel, retained
child identity/status, shared wire/I/O and actual server consumer; connected
controller-only evidence is in
`docs/experiments/glm53-20260906/live-pair-handshake-results.md`.
The production guard main still always exits failure and has no LIVE loop.
The server must still call its consumer before selected GPU initialization,
then bind the actual Model capability. Startup files/recipe and pair release
also remain unconnected; handshake evidence is not native admission.

Proposed source split, all Rust children <=500 lines:

- One owner freezes `crates/atlas-glm-pair-wire`: bounded codecs, canonical
  records and digest functions only. No I/O, GPU/engine dependency, launch,
  selected-session constructor, or Ready authority. Use the existing pinned
  SHA-256 dependency. Guard remains a standalone workspace with a path dependency;
  server consumes the same crate, not a copied parser. Flag this shared boundary.
- Guard owner extends existing `src/{main,child,linux}.rs` plus a small LIVE
  phase/codec adapter. Reuse existing `core::State` renewal semantics and child
  primitive; no second supervisor or execution abstraction.
- Root owns server inherited-channel consumer, pre-GPU profile validation,
  actual Model registration, nonreturning release, and the two-node controller
  adapter. Model author owns only actual health checks/stream joins.

Implementation ownership for the connected boundary: protocol author owns the
wire crate; root owns a shared Linux-only, GPU-free `atlas-glm-pair-io` transport;
the Model author implements its local identity I/O and the real server consumer;
the guard author extends the existing child primitive. The I/O crate centralizes
the unsafe ancillary parser for both endpoints, not a second supervisor or an
execution abstraction. Its credentials are actual per-message kernel facts;
neither a valid packet nor an identity object grants Model capability. Serialize
controller Cargo and retain runtime RED before each behavioral implementation.
Use real socketpair/fork/exec checks and a connected private-PID1 consumer gate,
then independently review before wiring native admission. OFF remains untouched.

OFF retains the existing CLI, 112-byte parser and failure behavior. No per-command
ACK, new E1/F5 counter, K5/F0 redesign, or generic resource-cleanup framework.

## Trust and Docker-reachable endpoint

Controller, node root/Docker administrators, kernel, pinned guard/server and
reviewed image/recipe are trusted. Digests detect mismatch/replay; they are not
signatures or protection against a malicious administrator or privileged child.

Docker does not inherit an arbitrary controller FD. For each rank, controller
exclusively creates a fresh root-owned 0700 directory under
`/run/atlas-glm-pairs/<64-hex-session>/rank<0|1>` on that node. Bind-mount that
directory at `/run/atlas-pair` in its container; do not mount the Docker socket.
Use immutable image ID, unique full container ID, private PID namespace,
exec-form guard entrypoint, no shell/init wrapper, `restart=no`, and a durable
exclusive consumed marker. Existing paths/markers/sockets cause refusal, not
unlink/reuse. Create and inspect BOTH containers before starting either.

Before `docker create`, root exclusively writes each canonical `recipe.bin`
(0600) in that rank directory and freezes its digest; recipe includes these
paths but excludes container ID and startup-record contents. After BOTH creates
and exact config/image inspections succeed, and BEFORE either `docker start`,
root exclusively creates `startup.bin` there with O_CREAT|O_EXCL|O_NOFOLLOW,
mode0600, writes exactly the record below, fsyncs file and directory, closes it,
and verifies both records against those inspections. No replacement/rename-over,
partial-record acceptance, or start on one successfully written rank alone.
The guard reads `/run/atlas-pair/{startup.bin,recipe.bin}` through pinned FDs,
requiring regular root-owned0600 files, one hard link, exact bounded sizes and
the checked root-owned directory. Missing/malformed records stop before fork.

`StartupRecord` is exactly288 bytes: the new16-byte header defined below with
kind0x01, local rank, prefix284; body pair_session[32], container_id[32],
image_digest[32], recipe_digest[32], server_elf_digest[32], guard_elf_digest[32],
Policy[80], in that order (body272). It is a file record ONLY, never accepted
as a control/child frame. Policy digest is derived from those exact80 bytes.
After validating the records, guard durably creates its exclusive consumed
marker before exposing the listener or preparing a child; restart cannot reuse
the same record. The guard's fresh local session/instance/startup challenge and
actual postfork child identity then complete GATED_REPORT. Thus the report's
pair/container/image/policy fields exist before PAIRED_START without inventing
child IDs or a container-ID/recipe-hash cycle.

Guard independently checks recipe bytes/digest, supplied policy against its
resolved policy, argv/env against the recipe, pinned server ELF and its own ELF
hashes, and actual local process/namespace facts. Full container/image IDs and
Docker mount/resource configuration are trusted controller assertions from
the recorded inspections, NOT facts the guard magically recovers through proc.
Root must match the resulting GATED_REPORT to the same record and live Docker
instance; no response can overwrite the startup record's asserted identities.

LIVE guard PID1 binds `/run/atlas-pair/control.sock` (0600), accepts exactly one
AF_UNIX/SOCK_STREAM connection before its finite startup deadline, then closes
the listener. A bounded node-local relay, invoked over the controller's existing
authenticated non-PTY SSH connection, connects to that exact host pathname and
pumps only framed bytes over stdin/stdout; stderr is diagnostics, never protocol.
It grants no tickets or renewals independently. No reconnect, public TCP service,
or Docker exec helper inside the container. Root-owned path plus authenticated
SSH is the trust boundary; peer UID is checked, but an outside-namespace peer PID
may appear as zero and is not claimed as the controller's process identity.
SSH/relay EOF is terminal; an orphan relay cannot renew without fresh controller
responses. A blocked link expires the guard's own lease without Docker/SSH help.
Because the host directory is root-owned0700, that node-local relay must run
through the existing authorized sudo path; do not broaden directory permissions
to make an unprivileged SSH account connect.

Root correlates each full container's inspected host init PID/start ticks,
host boot ID and `/proc/<host-init>/ns/pid` identity with the guard's local PID1
report. Never compare the host PID numerically with namespace PID1 or use a PID
number as a signal target. Both actual children must already be gated and held
by pidfd before the controller constructs PAIRED_START.

## Canonical recipe and environment

Each rank's frozen recipe is a bounded binary record, not shell text or ambiguous
JSON: version u16; ordered argv vector; environment vector sorted by raw key;
mount vector sorted by container destination; image/guard/server digests;
rank/world/profile fields; and explicit Docker resource/security/network fields.
Integers are big-endian, booleans u8 exactly 0/1, vectors start with u16 count,
strings with u32 byte length and UTF-8 bytes, no NUL. Maximum record 64 KiB,
64 argv entries, 128 environment entries, 32 mounts, 4096 bytes/string. Reject
duplicates, unknown schema fields, missing fields and noncanonical ordering.
Mount entries encode source, destination, read-only and propagation; resource
entries explicitly encode memory, swap, CPU set, shared-memory, device requests,
ulimits, capabilities, UID/GID, network and PID modes, restart and init policy.
Container IDs are not recipe inputs (avoids create-time circularity). Session
paths/FD locator and actual requested resource values are recipe inputs.

`Spec` constructs the child's entire envp from that validated record before fork;
it must not copy ambient `environ`, accept wildcard `ATLAS_*`/`NCCL_*`, or retain
the current PATH-only behavior in LIVE. Root materializes a literal reviewed
key/value table for the selected image/recipe, shared by guard and server checks:

- `PATH=/usr/bin:/bin`, `ATLAS_GLM_PAIR_FD=3`, and explicit logging values.
- Explicit NCCL settings already used by the launcher: SOCKET_IFNAME, NET,
  IB_DISABLE, IB_HCA, IB_GID_INDEX, IB_ROCE_VERSION_NUM, IB_ADDR_FAMILY, IB_TIMEOUT,
  IB_RETRY_CNT, NET_GDR_LEVEL, NET_GDR_C2C, DMABUF_ENABLE, NVLS_ENABLE,
  CUMEM_HOST_ENABLE, CUMEM_ENABLE, PROTO, ALGO, BUFFSIZE, MIN_NCHANNELS,
  MAX_NCHANNELS and DEBUG, each with its `NCCL_` prefix and exact value.
- Every selected Atlas knob is an explicit table entry, including EP_PROTOCOL=v2,
  independent decode OFF, the resolved eager/MTP4 profile and its actual kernel
  policy. Required absence is checked too; no implicit image/env defaults.
- Loader injection (`LD_PRELOAD`, `LD_AUDIT`, etc.) is rejected. If the frozen
  image needs `LD_LIBRARY_PATH`, it requires one explicitly reviewed literal
  value with no empty/current-directory/writable entries; no ambient forwarding.

The record contains the exact table, not a rule to forward arbitrary prefixes.
Its bytes/hash must agree with Docker inspect, guard envp, and server resolved
settings. This plan does not invent the not-yet-registered selected admission
flag or claim an executable native recipe exists already. Missing materialized
settings block launch. First profile remains TP2/EP2v2, two owners, MTP4, eager,
BF16 KV, cold text2..1024, greedy grammarless and excluded routes as in integration.
The first paired served context is2044, not2048: the actual paired-head reserve
requires context+4<=2048. This is an interim bounded gate, not long-context parity.

## Exact byte records and frames

All integers below are unsigned big-endian. IDs/digests/nonces are raw bytes,
not hex strings; Docker IDs are decoded from exact full64 hex. No optional fields.

`ProcessIdentity` is 88 bytes: boot_id[16], pid_namespace_device u64,
pid_namespace_inode u64, guard_pid u32, child_pid u32, guard_start_ticks u64,
child_start_ticks u64, **guard-minted postfork child_instance[32]**.
The child_instance is generated only after real fork/pidfd/READY; it is not the
server's later challenge and is not supplied as a future PID reservation.

`RankRecord` is 344 bytes: container_id[32], image_digest[32], recipe_digest[32],
server_elf_digest[32], guard_elf_digest[32], local_control_session[32],
guard_instance[32], original_startup_challenge[32], ProcessIdentity[88].
Rank is its position in the ordered pair, not an independently mutable field.

`Manifest` is 752 bytes: pair_session[32], policy_digest[32], RankRecord0[344],
RankRecord1[344]. Both rank records must be distinct actual instances, agree on
the selected fixed profile, and match controller inspect plus local guard facts.

New header is 16 bytes: length_excluding_prefix u32, magic[4]=`GLP2`, version
u16=1, kind u8, rank u8 (exactly0/1), reserved u32=0. Rank means reporting/local
recipient rank; embedded receipts carry their own rank. The new version namespace
does not reinterpret the old version1 bytes. Sizes include no implicit padding:

| Kind | Direction | Exact body in order | Body | Full frame / prefix value |
| --- | --- | --- | ---: | ---: |
| 0x01 STARTUP_RECORD | root-written file only | pair_session[32], container_id[32], image_digest[32], recipe_digest[32], server_elf_digest[32], guard_elf_digest[32], Policy[80] | 272 | 288 / 284 |
| 0x10 GATED_REPORT | guard to controller | pair_session[32], RankRecord[344] | 376 | 392 / 388 |
| 0x11 PAIRED_START | controller to guard | Manifest[752] | 752 | 768 / 764 |
| 0x12 CHILD_HELLO | server to guard | observed boot_id[16], namespace_device u64, namespace_inode u64, guard_pid u32, child_pid u32, guard_start_ticks u64, child_start_ticks u64, **server postexec challenge[32]** | 88 | 104 / 100 |
| 0x13 CHILD_TICKET | guard to server | Manifest[752], echoed_server_challenge[32], fresh_guard_ticket_challenge[32] | 816 | 832 / 828 |
| 0x14 QUIESCENT | server to guard; same bytes forwarded to controller | pair_digest[32], child_instance[32], epoch u64, last_command u32, reserved u32=0, receipt_nonce[32] | 112 | 128 / 124 |
| 0x15 PAIR_RELEASE | controller to guard to server | pair_digest[32], epoch u64, complete_QUIESCENT0[128], complete_QUIESCENT1[128], receipt_digest0[32], receipt_digest1[32] | 360 | 376 / 372 |

The largest frame is 832 bytes; allocated receive bound remains 4096. Reject
wrong direction, exact-size mismatch, unknown kind/rank, trailing data, reserved
bits, truncated packet/ancillary data and phase-inappropriate input. CHILD_HELLO
has no child_instance field: the server cannot know/mint it before the ticket.

Digest rule: `D(label, bytes) = SHA256(ASCII(label) || 0x00 || u32be(len(bytes))
|| bytes)`. Exact labels: `atlas.glm.pair.recipe.v1` for canonical recipe;
`atlas.glm.pair.policy.v1` for policy below; `atlas.glm.pair.manifest.v1` for
Manifest; `atlas.glm.pair.ticket.v1` for complete CHILD_TICKET;
`atlas.glm.pair.quiescent.v1` for complete QUIESCENT; and
`atlas.glm.pair.release.v1` for the PAIR_RELEASE **body** (recipient headers
differ). Pair digest means manifest digest. ELF/image digest fields retain their
ordinary SHA-256 definition, not this domain rule. Guard/controller/server
compare both receipt digests against the embedded bytes; local receipt must
also match its exact stored pending receipt. Digests are never bearer authority.

## LIVE multiplexing and deadlines

The control stream accepts either the unchanged prefix108/full112 old frame or
one of its direction-allowed new exact lengths above. Dispatch by prefix then
validate the respective entire header; no scanning/resynchronization after error.
Legacy mode accepts only112 and uses its existing parser. LIVE consumes a valid
PAIRED_START by deriving the original local START response (session, instance,
ordinal0, original challenge) and calling existing `State::accept` once, only
after complete manifest validation. Direct old START in LIVE is refused. HELLO,
CHALLENGE, RENEW and REVOKE remain their exact old112 bytes and semantics.

One nonblocking event loop polls control, child channel, signals and pidfd. Check
clocks/failure before and after I/O and before acceptance; at most one bounded
input frame per channel per turn, control first. Use one partial stream frame
and bounded pending output slots (one lease frame, one one-shot LIVE frame).
Never interleave a partially sent frame. Lease output takes priority before any
new LIVE frame; partial output must finish inside its original I/O deadline or
fail, never defer the challenge indefinitely. No unbounded queues/drain loops,
child callbacks, blocking logs or renewal threads. Unexpected extra phase frames
are terminal, not queue growth. Controller applies the same priority and checks
both guards before each renewal; a failed peer check permanently stops both.

Policy is exactly ten u64 milliseconds (80 bytes), ordered startup, lease,
challenge, frame, campaign, poll, reap, child_handshake, quiescent_wait, exit.
Proposed explicit recipe: startup30000, lease30000, challenge5000, frame3000,
poll250, reap10000, child_handshake30000, quiescent_wait10000, exit5000;
campaign is an explicitly supplied immutable positive duration <=86400000.
Retain existing validation and require frame<challenge<lease, added bounds
positive <=lease, and every duration <=campaign. No fallback values. Existing
non-LIVE Policy is unchanged. Child handshake starts at gate release; quiescent
wait starts at accepted local receipt; exit starts when guard accepts release,
not when delivery finally completes. Each wait is capped by lease/campaign
deadlines and frame I/O remains capped at3000ms. BOOTTIME origins are node-local,
never compared across hosts. Equality/overflow/clock regression fails.

## Inherited authority and successful release

Guard creates a private nonblocking SOCK_SEQPACKET socketpair before fork;
only the intended child endpoint survives exec as FD3. Relocate pinned ELF FD
away from3 before fork, apply existing close_range(CLOEXEC), then clear CLOEXEC
only for child FD3. Do not allocate or unwind after fork. Parent drops the child
endpoint; child closes the parent endpoint. No SCM_RIGHTS is admitted.

Set SO_PASSCRED on BOTH socketpair endpoints before any sends (and confirm it
on the server-owned descriptor). Each recvmsg uses MSG_CMSG_CLOEXEC and a bounded
ancillary buffer with room for one ucred and16 descriptor integers. Require
exactly one SOL_SOCKET/SCM_CREDENTIALS with exact ucred length; reject missing,
duplicate or unexpected ancillary records, MSG_TRUNC and MSG_CTRUNC. Perform
bounded ancillary validation on every packet, including one rejected for its
data/header. If SCM_RIGHTS is present, first close EVERY descriptor actually
delivered in all returned rights records, then enter terminal failure; no early
return after the first bad record and no leaked FDs. Kernel-discarded excess
descriptors on control-buffer truncation are closed by the kernel; still close
all delivered descriptors and fail. Do not enable other FD-delivering socket
options. SO_PEERCRED reflects credentials at
socket creation/connect time; a pre-fork socketpair alone does not authenticate
the later child. Validate per-message SCM_CREDENTIALS against actual sender
PID/UID/GID, rather than trusting the reported body identity.
[Linux unix(7)](https://man7.org/linux/man-pages/man7/unix.7.html).
This assumes the trusted child does not use privileges to forge credentials.

The server early selected ingress consumes FD3 exactly once (owned CLOEXEC dup,
close original; no reopening by pathname), validates AF_UNIX/SEQPACKET, obtains
its own PID/getppid, requires actual parent PID1, reads both `/proc/*/stat` start
ticks and current PID namespace/boot identity, and checks PR_GET_PDEATHSIG is
SIGKILL. Guard matches the received credentials to its retained child record.
After exec, server sends its independently generated HELLO challenge and checks
the echoed challenge in the ticket plus fresh guard challenge and all local
manifest fields. Guard checks process start/namespace again while held pidfd is
live; server checks `/proc/self/exe` and `/proc/<parent>/exe` against pinned ELF
digests, with local descriptor stat identity retained where available. Guard
hashes its opened executable FD, not a path reopened after validation. Kernel
process identity is local; remote identity comes from the trusted controller.

Missing/malformed/wrong-session/channel/recipe/parent authority is rejected
before CUDA/NCCL initialization, weight loading or selected factory work, with
no ordinary fallback. Parsing a ticket cannot create selected Model authority;
actual capability registration completes separately before worker/scheduler
handoff. Preserve the known cold-constructor inner-error ownership limitation,
not a claim that ticket validation fixes all later cleanup.

For this one-shot process, epoch is exactly1. last_command is the existing actual
matched shutdown word `0xffffffff`, not a fabricated transaction counter.
Source checked2026-09-09: `scheduler/mod.rs:1110` calls
`model.ep_broadcast_cmd_for_seq(0, 0xFFFFFFFF)` and
`model/impl_a2.rs:430-435` receives the actual preamble and returns `Ok(false)`
for that word before slot lookup. The old scheduler ignores the send Result;
selected root integration must check it, not inherit that ignored failure or
treat the literal alone as completion evidence. After
issued work completes and actual Model quiescence succeeds, server sends its
one QUIESCENT and waits, retaining owners with T1 armed and no more GPU work.
Controller needs both current receipts; neither one rank nor successful HTTP
authorizes release. Guard validates local stored receipt, both manifest
identities/digests and epoch, then forwards the certificate once. Server validates
its exact pending receipt and both bound identities and calls `_exit(0)` directly.
There is no disarmed-return gap, Model teardown, allocator sweep or backend Drop.

Guard retains pidfd through waitid's actual `CLD_EXITED/status=0` within the
exit bound; only then can guard `_exit(0)`. Arbitrary exit0 before release,
nonzero/signal after release, expired/replayed receipt/certificate, controller
loss or incomplete delivery remains terminal. Child EOF after a delivered
release may precede pidfd readiness: wait boundedly for that exact exit, never
infer success from EOF. Partial pair delivery can be FAILED, not pair success.
Retain existing kill-via-held-pidfd and PID1 backstops; never PID-search/restart.
Final pair success additionally needs both actual guard/child exit receipts and
native post-exit health checks; process exit does not prove driver recovery.

## Implementation gate after protocol approval

Freeze the shared codec/API first; root supplies the literal selected recipe.
Then exercise actual gated CPU children and the real server consumer: both
identities before START, postexec HELLO, one-shot FD, renewal while handshaking,
wrong recipe/credentials/start ticks/challenge, partial/oversized/replayed frames,
both receipt digests, arbitrary exit0 versus released exit0, and release-timeout/
nonzero-exit failure. The ACTUAL server consumer requires its direct parent to
be PID1: its CPU integration fixture must therefore execute the real guard as
PID1 in a private PID namespace and exec the consumer as its actual child.
Reuse the setup pattern in `glm_pair_guard/tests/pid1_cpu.cpp` (unshare PID
namespace then fork the namespace init), with a private mount namespace and
proc mount appropriate to that PID namespace for the consumer's `/proc` checks.
Do not relax the parent check, add a test flag, or replace actual identities
with supplied numbers. This exercises the NEW ticket/consumer path using the
existing namespace prerequisite; it is not a request to redo the old PID1 death
qualification. Unavailable namespace/proc setup is a blocked gate, not a mocked
PASS. Reuse existing lifetime controls; no fabricated future child IDs or fake
Model capability. No native fault injection. Source/CPU review precedes root's
bounded healthy live deployment.
