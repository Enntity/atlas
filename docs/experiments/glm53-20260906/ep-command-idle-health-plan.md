# EP command-idle health: test-first follow-up

Status: plan only; no serving-code change. The inspected v11 eager binary is
`e8771f30`; subsequent style-only commit `d3b0ac9f` does not fix this issue.
No GPU/node operations, recovery, or log rewriting are part of this plan.

## Saved-log evidence

Read-only inspection of `/tmp/atlas-glm53-phase3-20260906/` snapshots:

- `v11-eager-rank0.log`: 855 lines, through 20:51:59.928879 UTC;
  SHA256 `882bda65beeb0ce4beefa49e82e709095bad14bd731bd9803ae701c432dfedc7`.
- `v11-eager-rank1.log`: 733 lines, through 20:51:58.440557 UTC;
  SHA256 `17997fcb68df51f05dfb6bcd587c6cb601ea0b0fbfc7dbf6b9d1bfcf5946e5c1`.

On 2026-09-06 the worker reports ready/waiting at 20:49:41.586099 (line518).
Its next line reports an 83.6-second broadcast at 20:51:05.186503: the
83.600404-second interval matches the idle command wait. Rank0 becomes ready
at 20:49:43.212694 (line541) and starts the first prefill at 20:51:05.171458
(line543), immediately before the worker warning. Both ranks select indexed
C3 KDA at approximately 20:51:10.162 and continue processing.

Search of both complete snapshots found no independent CUDA failure,
NCCL asynchronous-error report, communicator abort, or worker-stop error.
There are four rank1 bootstrap connection-refused retries before successful
initialization, not post-readiness failures. Rank0 also contains three
content-loop watchdog warnings (lines668/763/766): these are separate output
quality observations, not transport errors, and must not be hidden by calling
the logs simply "clean." Absence of logged faults is not a hardware-health
proof. Preserve these raw snapshots and their warning.

## Cause and exact boundary

`spark-comm/src/nccl_backend/comm_impl.rs::broadcast` starts its wall clock
before `ncclBroadcast`, then synchronizes the legacy stream. The threshold
check runs only after successful synchronization; it includes time waiting
for the root to participate. A long successful call latches `unhealthy` but
returns `Ok`. This logic already exists at historical boundary `e5d088f6`,
well before this campaign. It is not an interrupting timeout watchdog.

`spark-server/src/main_modules/serve_phases/build.rs` loops immediately into
`ep_worker_step`; `spark-model/src/model/impl_a2.rs::ep_worker_step_impl`
begins with `ep_recv_seq_and_cmd`. That first receive intentionally waits
indefinitely when there is no request:

- Protocol v2: only the first `seq_id` u32 receive is command-idle intent;
  the following command u32 remains an ordinary broadcast.
- Protocol v1: the command u32 is the first receive and gets idle intent.
- Rank0 sends, subsequent lengths/IDs/token payloads, synchronization votes,
  and every other collective retain ordinary behavior. Do not exempt all
  calls to the shared `ep_broadcast_u32` helper or all non-root broadcasts.

## Minimal implementation, tests first

1. Add a typed broadcast intent, e.g. `Collective` / `AwaitCommand`, through
   a backward-compatible `CommBackend` extension. Existing `broadcast` calls
   keep their current behavior; non-NCCL backends may delegate the extension
   to ordinary broadcast. Factor NCCL submission into one implementation so
   the new intent changes only elapsed-time classification, not wire order.
2. Before implementing classification, add pure failing tests: ordinary
   durations below30s do not latch; exactly30s/83.6s do; successful idle waits
   do not latch at either duration. Neither intent clears an already-latched
   fault. Keep an actual NCCL async fault unhealthy regardless of intent.
3. Preserve `check_nccl` and stream-sync error propagation, and the existing
   async-error check/latch after successful synchronization. Do not downgrade
   real errors, remove synchronization, alter streams, or claim idle elapsed
   time is network latency. An optional low-volume idle-wait diagnostic must
   identify its intent, rank, bytes, and root without declaring a fault.
4. Add CPU recording-backend tests for the worker header receiver: v1 issues
   one idle-intent call; v2 issues idle then ordinary; both preserve values,
   root0, four-byte size, and operation order. Test shutdown, repeated worker
   steps, and ordinary payload/header failures. The rank0 send path must
   never gain idle intent. Mock enqueue/sync/async-error paths so error
   preservation does not require a GPU or a real30-second sleep.
5. Implement only this boundary and run focused `spark-comm`/model tests,
   formatting, and the full CPU gate. A separately authorized root-controlled
   deployment may later test an idle interval followed by C1 and C3/C4;
   this document authorizes no such run or transport fault injection.

## Recovery cautions and exclusions

At review, repository Rust call-site search found no production consumers
of `is_healthy()` or `attempt_reconnect()`; the false latch presently does
not trigger automatic recovery. Do not invoke reconnect just to clear it.
`reconnect_inner` aborts the old communicator, requires both ranks to enter
bootstrap together, and clears old registration handles. Safe recovery also
needs explicit treatment of outstanding work, registrations, captured graphs,
and request/recurrent-state consistency; it is a separate infrastructure task.

This narrow fix must not silently clear genuine faults or introduce unilateral
abort/reconnect. Real peer-loss detection during an idle wait needs a separate
liveness/heartbeat design; a post-completion elapsed-time check cannot detect
a synchronization that never completes.
