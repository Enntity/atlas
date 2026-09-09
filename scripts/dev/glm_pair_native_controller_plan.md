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
