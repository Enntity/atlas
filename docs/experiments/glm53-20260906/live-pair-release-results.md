# Actual LIVE guard startup and paired release — CPU qualification

2026-09-09. Development source follows `f1a8dd98`; exact source manifest and
postcommit receipts are retained in controller campaign
`/home/abc/storage/models/atlas-campaigns/20260909/glm-live-release/`.
No Spark operation, native build or throughput measurement belongs to this slice.

## Connected behavior

The real `glm-pair-guard --live` ELF runs as root PID1 in each of two distinct
private PID/mount/proc namespaces. It reads exclusive root-owned startup files
and a bounded canonical recipe, verifies pinned executable hashes, durably
consumes the launch and accepts one private root controller connection. It
execs the intended gated child with exactly the recipe's argv/environment and
inherited credentialed channel. The child fixture compiles the exact production
server startup, handshake and nonreturning release modules, without GPU or Model
dependencies. The existing lease state machine remains the authority.

The LIVE loop requires both actual gated process records before start, validates
the child's postexec identity/challenge, forwards its one current QUIESCENT
receipt, validates the two-rank release and only reports clean after the released
held child actually exits0 and is reaped. Unexpected exit0 is a failure.
Successful server release calls `_exit(0)` while retaining its channel and pinned
owners; failure calls `_exit(74)`. The caller must still provide actual Model
shutdown/quiescence and retain Model/sequence owners with T1 armed.

The shared recipe is a data codec, not permission to launch. The guard checks
the local fixed-path/file/ELF/environment boundary. Full Docker resource, mount,
image and literal native setting equality still belongs to a future controller
adapter. CPU image/container/resource fields are explicitly synthetic assertions.

## Focused evidence

Actual runtime RED receipts precede implementation: `loop-red.log`,
`release-red.log`, `startup-red.log`, and `server-startup-red.log` fail at their
respective real entry points. `eof-red.log` exposes untyped channel EOF;
`fifo-red.log` exposes an executable FIFO blocking before its regular-file check.
The implementation adds typed credential-free EOF and opens the candidate ELF
nonblocking before checking its type. The I/O test's fork/EOF witnesses are
serialized to avoid an unrelated concurrent subprocess transiently inheriting
the pipe writer before exec; production I/O gains no lock.

All eight actual `--run-main` cases pass in `main-case-<mode>.log`:

| Case | Actual observation |
| --- | --- |
| valid | Both actual guards and released children exit0 |
| delayed | Rank1 waits6s before receipt; both guards renew once, then exit0 |
| bad-release | Corrupted receipt digest rejected; guards exit74 |
| replayed-release | Queued duplicate release rejected, not converted into success |
| unreleased-zero | Child exits0 without receipt/release; guards reject it |
| bad-environment | Actual server ingress rejects an unrecorded environment key |
| reused-session | Successful consumed records cannot start a second guard session |
| hardlinked-record | Guard refuses multiply linked recipe before listener/child |

The delayed case verifies that absent atomic release packets wait under QWAIT,
while actual frame receive/validation retains its3s processing bound. Control
stream partial-frame and handshake clocks retain their original origins.

Final review also found two original frame-window gaps. The guard now retains
the pre-recv origin through child decode/identity validation and checks before
queuing a ticket/receipt. Server ticket acceptance checks both original clocks
after final identity validation. `packet-window-{red,green}.log` uses a real
socket packet and expired processing window; `handshake-window-{red,green}.log`
checks the production final-acceptance helper with expired frame/live overall
deadline. Each fails before its correction and passes after. These are focused
helper timing proofs, not an induced full-process deschedule experiment.

Scoped checks: shared wire/recipe12 and I/O7; guard binary12 and legacy
lifecycle10 pass. Strict shared/guard clippy passes. The actual non-test server
CUDA-feature check passes with the retained CPU link/build-stub recipe; this
does not execute CUDA. Guard lint exceptions are limited to the explicit live
ownership argument boundary and its bounded inline frame enum.

## Remaining work and interpretation

This completes the implemented startup/release slice, not the entire live
integration plan or hardware qualification. No native Model is involved in the
fixture's QUIESCENT receipt. Actual resolved CLI/profile validation, Model-bound
registration, paired factory selection, whole selected head/worker operations,
checked shutdown and the two-node Docker/relay adapter remain to connect.
Additional protocol failure gates remain part of that integration; these eight
cases are not advertised as exhaustive fault qualification.

The next native experiment is a healthy bounded C2 MTP4 run only after that
vertical path and its exact memory recipe are reviewed. It must check coherence,
tool calling, needle retrieval, per-rank headroom/zero swap and both-rank clean
exit. Then optimize actual batching/higher concurrency. Existing v29 nonspec
fresh-process C1..8 results remain the performance evidence; CPU checks do not
raise those throughput numbers or establish reference parity.
