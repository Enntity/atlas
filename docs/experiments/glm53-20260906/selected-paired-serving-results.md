# Connected selected paired MTP serving

2026-09-09 development evidence; native admission is not yet qualified. This
implements the actual serving path after the LIVE guard/release work in
`2c31bf37`. It is a serialized two-owner control, not batched `[5,5]` verification
and not completion of the reference-parity goal.

## Implemented path

Synchronous main consumes the inherited guard session before Tokio starts.
The resolved CLI/model profile is checked before GPU initialization. The actual
factory explicitly selects the paired head; it cannot silently fall back to
ordinary MTP. Selected cold-construction failures retain successful GPU,
communicator and assembled Model owners. Ordinary construction is unchanged.

The actual boxed Model is registered immediately after construction, before
fallible metadata work or worker/scheduler handoff. Registration binds the
inherited rank to the real sealed capability and checks communicator health.
The selected head handles cold admission, first selection, serial bootstrap/
verification/proposal and checked retirement without entering ordinary MTP
arbitration or cleanup. The worker retains both slots and covers the entire
worker command, including its follow-on receives.

Health checks precede visible token emission, new proposals and retirement
completion. Shutdown stops admission, finishes issued work, retains owners and
uses actual Model quiescence plus the existing paired release before `_exit(0)`.
Unexpected selected returns and the existing OOM watchdog use `_exit(74)`;
ordinary watchdog behavior, thresholds and timing are preserved.

The selected reserve uses the actual TP-local configuration, resolved arena
rows, three live state slots (two owners plus dummy), both MTP verification
state sets and speculative headroom. The independent C2..8 and ordinary reserve
paths retain their previous checks/behavior.

## Focused development checks

Raw logs are on the controller under
`/home/abc/storage/models/atlas-campaigns/20260909/glm-selected-serving/`:

- Actual CLI profile checks pass, including explicit swap=0, snapshot rollback,
  prefill-only drafter context and absence of the presence-style eager kill
  switch. The original fixture accidentally inherited the CLI's 3 GB swap
  default; that fixture failure was corrected without relaxing launch policy.
- Both-rank selected reserve checks pass against the actual topology/resolver:
  exact pool/arena terms and the accepted/rejected free-memory boundary.
  Existing independent C2..8 and OFF accounting checks pass.
- Actual Model completion-health, cold F0/first selection, serial continuation
  and checked-retirement checks pass. Genuine runtime failures were recorded
  before adding the missing health checks and cold producer.
- A new actual cold-selection/promotion regression reproduced the first-token
  tool-opener bug. The selected path now updates tool bookkeeping without
  emitting/charging that token twice; thinking and zero-token controls pass.
- The watchdog terminal-dispatch check reproduced legacy C exit-handler
  execution on the selected path, then passed with the supplied terminal sink.
  This does not simulate GPU memory pressure.
- The production server's CUDA-feature check/build pass using the controller's
  link stubs. With FD3 absent, that actual executable exits74 before any
  clone/clone3 syscall (`missing-fd-stub.{log,trace}`). An earlier attempt lacked
  the driver link stub at process load and exited127; it is not ingress evidence.

Factory construction evidence is in sibling campaign `glm-live-factory/`:
actual paired selection, rank binding, missing-module refusal, selected
registration failure and GPU/communicator/assembled-owner retention. The frozen
factory source was independently reviewed; legacy handoff regressions pass.

The new two-guard registered-Model fixture passes all four modes: valid paired
registration/release and wrong actual Model rank, missing paired capability,
and unhealthy communicator refusal. Each negative requires exit74 and the
before-registration witness without a return/Drop witness. The positive uses
the actual head shutdown, worker loop and Model quiescence paths before both
guard receipts/release and exit0. Its Model command transport is local scripted
replay, not cross-process NCCL; join-order assertions remain in the existing
actual Model tests. Raw logs: sibling `glm-live-registration/registered-*.log`.
The pinned guard is the already-qualified `2c31bf37` ELF; no Docker or GPU runs
are hidden in this fixture.

The node-local `glm-pair-relay` is also implemented. It derives a single
root-owned session/rank socket path, checks directory/socket/peer ownership,
and forwards bounded directional frames over nonblocking pipes/socket. It
cannot issue tickets, generate renewals or reconnect. Original partial-frame
deadlines and lease priority are preserved; EOF is always transport failure,
never a clean-pair certificate. Actual local pipe/socket and path checks pass,
as do the existing guard I/O checks and scoped strict relay lint. These are
not an SSH deployment or Docker identity proof. Raw logs: sibling
`glm-pair-relay/`.

## Remaining work and limits

No new native throughput, coherence, tool-calling or needle-in-haystack result is
attributed to these controller checks. The reproducible nonspeculative v29
baseline in `independent-c2-c8-results.md` remains unchanged.

Before a healthy native paired run: connect the relay to the exact two-node
Docker/controller adapter;
review a literal memory-bounded recipe; package/pin both executables; inspect
the real containers and qualify load, requests and paired clean exit. The
recipe inspection must explicitly cover RDMA device mappings and security
options, not only GPU DeviceRequests. No native fault injection is authorized.

After that control, measure warmed C1/C2 with quality/TTFT and implement shared
multi-owner verification/proposal where profiling warrants it. Higher concurrent
MTP widths, comparable serving capabilities/context and all reference targets
remain required. CPU checks and inactive infrastructure are not performance.
