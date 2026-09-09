# Local paired-process guard: CPU closure

2026-09-09. Core commit `caf585edbc28ee93e4c786d910d1c3585f3ff576`;
private-PID1 qualification commit `06561c10fb47a34d035ef3608868cfcf5f8ba1ba`.
This closes the bounded **local CPU guard core**, not selected serving activation.

The standalone Linux guard owns one gated child through a pidfd, establishes
parent-death SIGKILL before exec, validates bounded local frames and expiring
leases, and fails terminally without graceful child cleanup. No successful
pair-disarm command exists yet. Its local session fields do not grant T2 or
Docker identity authority.

## Recorded gates

| Gate | Evidence |
|---|---|
| Actual initial RED | Deadline equality wrongly accepted; retained-pidfd SIGTERM ran forbidden child Drop witnesses. |
| Final standalone CPU suite | 18 invocations /15 unique tests: unit3, lease5 (including3 repeated core tests), lifecycle10. |
| Source hygiene | Offline check, strict all-target clippy, fmt, SPDX and <=500-line source gates passed. |
| Post-commit confirmation | Root directly reran the existing frozen unit/lease/lifecycle executables: all18 passed; lifecycle8.18s. This was **not a Cargo rebuild**. |
| Actual private-PID1 death | Two root-owned privileged **local CPU** runs passed, including one post-commit repeat. |

The initial unprivileged namespace-creation attempt failed at uid_map; the later
privileged creation check only established the prerequisite. The two subsequent
harness runs actually execed the frozen guard as namespace PID1, retained its
pidfd and those of the guarded child and a descendant, then killed only PID1.
All three died while an outside-namespace CPU decoy remained alive.

The descendant explicitly reported `pdeath=0`, unlike the guard's direct child
(`pdeath=9`), so its death is evidence for namespace-init lifetime rather than
merely inherited parent-death protection. A separate healthy CPU control
actually produced destructor and atexit witnesses; neither killed namespace
process produced those cleanup witnesses. Original witness files are archived.

The C++ harness's first strict compile failed on two fortified unchecked-write
warnings. Both writes were changed to require their exact byte count or `_exit(87)`;
compile attempt2 passed. This was a compile correction, not a behavioral RED.
The earlier Rust post-I/O deadline review refinement also remains accurately
classified: exact-clock codec/predicate tests plus call-site review, not a
claimed reproduction of runtime descheduling.

## Provenance and remaining boundary

Raw receipts and full source/binary hashes:
`/home/abc/storage/models/atlas-campaigns/20260908/glm-pair-guard-core/`, especially
`postcommit-closure.md`, `postcommit-{unit,lease,lifecycle}.log`,
`pid1-{run,postcommit-run}.log`, and `pid1-witnesses.tar`.
Root receipt archive: `glm-pair-guard-core-06561c10-receipts.tar`, identified by
its detached SHA256 sidecar. It contains source and receipts, not executables.

No Docker/SSH adapter, peer lease controller, clean two-rank drain, T2 live
registration, model deployment or GPU test was performed by this slice.
These CPU results do not establish CUDA/driver/hardware recovery or an
unconditional kill deadline if the guard/kernel cannot run. No throughput gain
is attributed to containment infrastructure. Paired serving remains unactivated.
