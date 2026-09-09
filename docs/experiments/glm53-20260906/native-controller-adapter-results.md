# Native paired-MTP controller adapters and healthy drain

This slice follows `0761a42f` selected serving/relay integration. It does not
claim a native deployment or a throughput improvement. The full MiaAI C1/C2/C4/
C6/C8 goal, concurrent MTP and quality requirements remain unchanged.

## Implemented

- Canonical recipe **v2** now records RDMA device mappings and security options
  separately from GPU DeviceRequests. Version 1 is rejected, not silently
  reinterpreted. All consumers must be rebuilt together. No-new-privileges still
  has one boolean authority; free-form duplicates are refused.
- One-shot controller-to-rank0 DrainRequest binds the current manifest and
  epoch. The actual Running guard reobserves its held child and sends SIGINT via
  its retained pidfd. It cannot signal a gated/exited child, repeat a request or
  extend the original drain deadline when quiescence arrives.
- The Docker adapter emits an explicit create request and checks actual inspect
  data against recipe resources, executable/image identity and lifecycle phase.
  It rejects extra mounts/devices, privilege, restart, init, swap allowance,
  unhealthy states and nonzero final exits. Runtime is fixed to `runc` as a
  deployment policy. Guard path remains an explicit input, with the existing
  startup ingress separately pinning the actual guard ELF digest.
- The controller transition layer binds HELLO, independently supplied process
  observations, gated reports, readiness/workload completion, receipts and
  release delivery. It handles in-flight release-phase challenges but stops
  renewing after either relay EOF or container exit. Only both exact observed
  normal exits can finish successfully.
- The finite subprocess adapter uses explicit environment/limits, nonblocking
  pipes, cumulative output caps and original deadlines. It does not wait inside
  the lease loop. Abort targets only its retained direct local child; it does
  not certify descendant cleanup. It is not a persistent relay transport.

`glm-pair-supervisor` currently exposes `create-request` and `inspect` data
commands. The transition and subprocess modules are compiled and tested, but the
production SSH preparation/launch/renewal driver is **not connected yet**. Do not
use the data commands alone as a serving launcher or regard them as deployment
qualification.

## Evidence and interpretation

Raw runtime RED/GREEN logs are retained outside the repository at
`/home/abc/storage/models/atlas-campaigns/20260909/glm-native-controller/`.
The closure there identifies the committed source and exact-tip reruns; tests
run before commit are development evidence, not release gate records.

Wire RED exercised four real missing behaviors. Controller review added real
regressions for Docker omitting writable-bind `ReadOnly=false`, valid challenges
crossing release, and attempts to renew after EOF. Process deadline/clock tests
use actual local children. No synthetic Docker/process records in pure state
tests are presented as observations of native containers.

The connected fixture uses two actual private PID1 guards, the real inherited
startup consumer and registered Model owner. Its rank0 SIGINT handler is a CPU
fixture handler, not the HTTP server's Tokio listener. Model command transport
remains local scripted replay, not NCCL. Positive drain requires the actual
signal witness, Model shutdown/quiescence, both releases and both guard exits.
Foreign/replayed requests must fail promptly without either release. An initial
fixture timeout before registration is excluded as drain evidence; the wait was
restored to the existing fixture envelope for hashing the debug consumer ELF.
A signal-probe harness mask bug was also corrected without changing production
signal behavior.

Read-only node checks found Docker29.2.1/API1.53 on both Sparks, about116GiB
available host memory, zero used swap and the previous benchmark containers
exited0. GPU memory counters report N/A on these systems; they are not a zero
GPU allocation measurement. The exact previous head inspection is retained as
`observed-v29-head.json` for adapter comparison.

## Native integration still required

Implement the prepared-session and bounded SSH driver in
`scripts/dev/glm_pair_native_controller_plan.md`. Refresh actual host init PID,
boot/startticks/PID namespace before marking status healthy; retain first-byte
frame clocks through queued writes. Pin both full IDs before starting either
container, install both startup records, and never infer success from relay74.
Then build an immutable candidate and run the short-context healthy C2 MTP gate,
followed by warmed C1/C2/C3/C4 timing, coherence, tools and retrieval. Scheduling
width expansion and batched verification remain subsequent performance work.

The proposed launch intentionally differs from v29's host IPC/no-NNP profile:
private IPC with1GiB shared memory and no-new-privileges enabled. These choices
need native qualification; they are not asserted equivalent to the prior launch.
Literal recipes must include the pinned image environment Docker merges during
creation, or strict inspection will refuse it. Docker representations were
checked against the [Engine API schema](https://raw.githubusercontent.com/moby/moby/v28.0.0/api/swagger.yaml)
and [mount struct](https://raw.githubusercontent.com/moby/moby/v28.0.0/api/types/mount/mount.go),
as well as the saved actual node inspection.
