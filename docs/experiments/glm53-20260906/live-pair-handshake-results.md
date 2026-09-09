# Connected inherited paired-channel boundary

2026-09-09. Implementation progress toward concurrent MTP4, not a new throughput
result or native serving qualification. The v29 full-wall benchmark results and
full reference-parity goal are unchanged. No SSH, Docker, GPU workload, reset or
node operation was performed for this slice.

## Implemented production boundary

- `atlas-glm-pair-wire`: GPU-free exact typed startup/report/start/HELLO/ticket/
  quiescent/release records, direction/size checks, explicit policy and domain
  digests. It does not validate the pending canonical recipe schema, authenticate
  processes, enforce replay phases or grant Model capability.
- `atlas-glm-pair-io`: shared nonblocking SEQPACKET transport. Every packet uses
  actual SCM_CREDENTIALS; unexpected received FDs are disposed before rejection,
  including oversized/truncated/wrong-sender packets. FD3 is checked open before
  creating a Rust owner, consumed into a CLOEXEC duplicate, and never reopened.
  Proc/PDEATHSIG and retained ELF observations are shared I/O, not authority.
- The existing guard `Child` retains its one fork/gate implementation. LIVE adds
  an explicit bounded environment, safe ELF/socket relocation away from FD3,
  held child credentials and nonconsuming pidfd exit-status observation. Legacy
  CLI/parser/main behavior is unchanged; its default remains failure exit74.
- The actual server `glm_terminal_session::inherited` consumer binds the ticket
  to its fresh postexec challenge, actual local processes and namespace, pinned
  server/guard ELF hashes and explicit expected launch record. It requires direct
  parent PID1 and PDEATHSIG SIGKILL. Each send/receive has the explicit frame
  timeout, capped by the overall handshake timeout. There is no Model capability,
  serving activation, channel escape or release API.

The shared dependency additions are intentional cross-cutting boundary changes;
existing SHA256/libc versions are reused without unrelated dependency upgrades.
Linux-only server imports preserve the ordinary non-Linux module boundary.

## Evidence and important distinctions

Raw controller receipts:
`/home/abc/storage/models/atlas-campaigns/20260909/glm-c2-live-boundary/`.
Guard-specific child receipts:
`/home/abc/storage/models/atlas-campaigns/20260909/glm-live-child/`.

Actual runtime REDs were captured before implementation:

- Wire decoder rejected an independently assembled valid HELLO.
- Transport could not receive authenticated data or dispose queued rights.
- Existing close-on-exec child behavior lost the required channel; the probe
  exercised a real ELF occupying FD3 before relocation.
- Both actual server children rejected the ticket-binding scaffold in the
  connected PID1/proc fixture (`connected-red.log`).
- Review found a missing-FD ownership violation and an overlong frame wait.
  A real isolated missing-FD child aborted before its fix (`missing-fd-red.log`);
  two actual consumers exceeded the 3s ticket frame bound before its fix
  (`frame-deadline-red.log`). Both fixes have runtime GREEN evidence.

Focused results: wire7/7, transport/local-identity7/7, inherited-record binding3/3,
guard child3/3 and existing legacy lifecycle10/10 pass. Shared-crate clippy with
warnings denied passes. The real server's non-test CUDA-feature/stub-build check
passes. This is not a workspace-wide lint or GPU numerical pass.

`scripts/dev/glm_pair_live_cpu` includes the exact production child and server
consumer sources. Root launches two actual private PID/proc namespaces, obtains
both real gated child identities before paired START, and executes the real
server HELLO/ticket consumer with kernel sender credentials. Four separate
two-rank runs pass: valid handshake, wrong echoed challenge, wrong local recipe
digest, and a withheld ticket that must fail inside the 3s frame bound. See
`connected-final-{valid,bad-echo,bad-recipe,stalled-ticket}.log`. Expected rejection
cases contain child exit74 diagnostics; the fixture verifies those exact exits.

These are CPU-fixture launch IDs, not asserted Docker containers. The fixture's
guard orchestration is **not** the production LIVE guard main loop. Its positive
child exit0 is only a completed-handshake witness: production supervision must
still reject an exit0 without a paired quiescent release. There are no fake
Model capabilities or supplied future child PIDs, and the real parent-PID1 check
is never relaxed. Closing FD3 permits later pinned-ELF opens to reuse that number;
the check requires that it is no longer an inherited socket, not permanently
unallocated. Legacy child tests separately verify consumed-channel CLOEXEC across
a descendant exec.

## Next integration (still required)

Implement the production LIVE guard loop and canonical recipe/startup-file
validation using these actual primitives; retain lease priority and exact held
instances. Materialize explicit Atlas/NCCL environment and Docker resource
records. Connect the server consumer before GPU startup, then register its actual
paired Model, select the paired factory/head/worker/scheduler path, and implement
the current two-rank quiescent-release/nonreturning exit boundary. Native C2
admission remains disabled until that connected path is reviewed and qualified.
The first paired context is2044 (actual constructor requires context+4<=2048).
Coherence, tool calling, retrieval, memory headroom, warm throughput/TTFT and clean
both-node exit must be checked on the healthy native run; none are inferred from
the controller-only work here.
