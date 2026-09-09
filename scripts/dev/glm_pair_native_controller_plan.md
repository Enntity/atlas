# Paired MTP native controller implementation

Base: `0761a42f`. The previous connected serving/relay CPU qualification is
retained; this work does not repeat it as an unmet prerequisite. The next native
experiment must measure concurrent MTP, not count infrastructure checks as gains.

## Implementation order and ownership

1. Wire author: extend the canonical recipe explicitly to version 2 with Docker
   device mappings and security options. Preserve GPU DeviceRequests separately.
   Bound and canonicalize the new lists; no-new-privileges has exactly one
   authority, its existing boolean. Update actual recipe constructors. Add a
   controller-to-rank0-only `DrainRequest` (kind 0x16, pair digest, epoch 1).
2. Guard author: accept that one-shot request only in Running with the current
   manifest, no pending quiescence/release, and fresh held-child observation.
   Consume it before signaling the existing pidfd with SIGINT. Gated, exited,
   wrong-phase, wrong-rank, foreign and replayed requests fail terminally.
   Keep existing child-initiated quiescence valid. Never signal a guessed PID.
3. Root: implement the deployment-specific Docker observation/create adapter
   using recipe v2 as the sole configuration authority. Exact images, arguments,
   environment, mounts, resources and security settings must agree. Reject
   privileged, automatic removal, extra devices/rules/groups, restart and init.
   Normalize only documented Docker representations (empty PID mode to private,
   for example). Do not reinterpret host IPC as a private shared-memory cap.
4. Controller author/root integration: one bounded Rust controller reusing the
   existing directional framing and queues. Prepare explicit immutable inputs;
   create and inspect both containers before starting either; write both startup
   records; correlate actual host PID/boot/startticks/PID namespace with gated
   reports before forming the manifest. No pulls/builds/tag lookup/reconnection
   or ambient configuration in the run path.
5. Connect readiness and the selected quality/benchmark workload while renewing
   both ranks. Status subprocesses must be bounded and cannot block lease I/O.
   Send DrainRequest only after actual HTTP readiness and completed workload.
   Validate both quiescent receipts, deliver both releases, then independently
   observe both pinned containers exited 0 and OOMKilled=false. Relay exit 74
   after EOF is not success evidence.

## Failure and validation boundaries

Stop renewals on any latched failure; optional REVOKE and bounded SIGKILL apply
only to recorded full container IDs. Preserve containers, records and receipts.
No name-based kills, docker stop, force-removal, native fault injection, reboot,
driver or swap changes. Keep short-context memory reserves and serialize native
build/model activity; root alone operates the Sparks.

Record runtime RED before new behavior, then focused production-code tests.
Wire negatives cover direction/rank/length/digest/epoch and canonical v2 fields.
Guard proof uses the actual held child and existing registered-model CPU fixture;
controller fixtures cover ordering, identity mismatch, bounded stalls and final
exit/OOM refusal. These are CPU safety evidence only. After review, build an
immutable native candidate and run one healthy guarded C2 MTP qualification,
including coherence, tools and retrieval. Measure warmed C1/C2/C3/C4 before
expanding scheduling widths. Do not claim the full goal from a C2-only result.

## Connected driver slice (base d1be1216)

The adapters/drain above are committed and exact-tip CPU-qualified. Implement
the runnable path now, not another replacement state machine:

- Prepared-input author owns `supervisor/prepared*.rs`: strict versioned serde
  structs (unknown/duplicate keys refused), canonical Recipe-v2 files, fresh
  session substitution only in the writable session bind, exclusively created
  local directory/consumed marker, digest over all literal settings and bytes.
  Public result contains launch settings, session digest, two recipes and path.
- Relay author owns `supervisor/relay*.rs`: retained persistent SSH child with
  nonblocking stdin/stdout/stderr, existing directional Reader/Outputs, original
  frame clocks, exact local completion notifications and bounded cleanup. The
  existing finite Process remains for node commands and workload operations.
- Node author owns `supervisor/node*.rs`: fixed root-only node verbs for exclusive
  session writes, create/seal/start/observe/relay/kill of exact recorded IDs.
  Reuse Docker adapter and bounded Process for Engine API calls. Actual process
  observations begin at Docker's host init PID and verify namespace PID1, direct
  child, start ticks, boot and namespace identity; never call the existing
  caller-parent observation method with an unrelated process.
- Root owns CLI registration, shared dependency changes and `supervisor/run*.rs`:
  nonblocking orchestration, both-created-and-inspected before either start,
  both startup records written, bounded socket readiness before one relay per
  rank, report buffering followed by real host process observation, then State
  construction preserving original startup and frame clocks. Status renewal,
  HTTP readiness and workload proceed in the same bounded loop. Retain every
  receipt and exact exit; no relay EOF success shortcut.

The launch JSON explicitly supplies policy, controller bounds, SSH executable/
key/known_hosts/environment, exactly two ranked node destinations and pinned
supervisor/relay paths+hashes, guard container path, canonical recipe file+hash,
readiness curl/URL/expected model and finite workload program/argv/environment/
limits. No shell command supplied by the launch file. Fixed remote verbs carry
bounded stdin records rather than interpolating arbitrary input into shell text.
No implicit build/pull/tag lookup or runtime alternative is added.

Record focused runtime RED for the new real I/O entry points; then exercise the
connected loop against the existing CPU guard fixture where possible. Native
fault injection remains forbidden. Only root builds/deploys/runs on the Sparks;
serialize Cargo source windows and native work. Root will inspect/freeze exact
interfaces before integrating independently owned source files.

## Native handoff after connected CPU qualification

Use the existing stopped head-side builder as a cache snapshot; its inspected
writable layer is about456MB and neither Spark has a running container or swap
use. Create a separate CPU-only, network-none, runc builder with memory=swap8GiB
and CPU0/1 so the original builder/rollback stay recoverable. Refresh the frozen
tracked Rust/manifests/helpers in the validated source tree, preserve unchanged
kernel files/timestamps, and verify the transferred source manifest. Normal
Cargo may reuse native kernel artifacts; ATLAS_SKIP_BUILD must remain absent.
Build spark followed by guard/relay/supervisor serially. Package new versioned
native images and root-owned helper paths with exact hashes; never invoke the
old nonspeculative C8 launcher for selected paired MTP. Recheck memory, disk,
fabric and both stopped states before beginning any native Model work.
