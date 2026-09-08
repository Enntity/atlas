# GLM paired serial scheduler: B2 integration audit

2026-09-08, after Partition A commit `f8a0bdb9` / closure `aad8679e`.
B1 pre-dispatch and checked-selection implementations are in development.
This is a source-backed handoff for the next integration slice, not admission
approval, native fault containment, or measured throughput.

Checkpoint 1 preflight is now committed as `335c9956`, and checked selection as
`67640368`. Their optional transport/caller remains disabled. Keep the first
native experiment a serialized two-owner control; neither commit is a [5,5]
batched target implementation.

## Selected dispatch must precede the generic MTP ladder

`scheduler/mod.rs` applies ordinary width/eligibility arbitration and can clear
drafts before `step_decode_only`. `mtp_step.rs` independently applies depth
and capacity clamping. Those transitions cannot own the new paired regime:
the surviving slot-1 request must retain its own handoff at occupancy one,
not resume legacy C1 global-hidden behavior.

Use one explicitly selected serial driver before that arbitration. Process
whole per-owner bootstrap/K5/proposal transactions in deterministic slot
order. Keep `ActiveSeq`'s existing `last_token`, `pending_drafts`, sampling
history and output budget as authorities; no scheduler hidden-pointer stash.
Do not route to generic batched bootstrap/verify or silently truncate MTP4.
Missing drafts outside the defined bootstrap state are an error, not a
permission to decode normally and later resume private repair.

## Complete target state before terminal emission can return

`verify_dflash_step.rs` currently detaches a verdict, emits accepted/bonus
tokens, and only then commits target state and trims the proposer. An early
EOS/output-cap/cancellation return during emission skips those latter steps.
Worker F5 instead completes rollback -> record -> trim -> commit before
returning to its command loop.

The selected transaction needs this order:

1. Immutable local validation before the first command attempt.
2. Actual K5 target operation and checked penalty-aware selection.
3. Accepted-count exchange, token rollback, owned detachment, target commit
   and proposer trim on both ranks.
4. Existing visible-token emission semantics, respecting EOS/output cap.
5. If still live, owned next proposal using the actual selected bonus.

A request that finishes in step 4 needs no E1. It still needed step 3. A
cancellation observed before step 1 must produce no command. A cancellation
observed after issuance cannot skip completion of an issued command.

`verify_pipeline_helper`'s old full-read fallback to raw IDs is not a checked
selection. The new checked grammarless sibling preserves existing sampling
math while returning copy failures. The driver must also check returned
cardinality/token bounds and bind the selected seed to the same active owner
and exact target position before invoking B1's selected proposer transport.

## Admission and retirement are part of the same contract

`phase_start_prefills.rs` can reject a request before `prefill_a_step.rs`
allocates a sequence. Resolve and enforce the cold single-chunk 2..1024-token
text-only limit there, plus fixed-four-draft and no prefix/grammar/adaptive/
carry/catchup/HSS requirements. A model arena's larger extent is not request
admission. Native active/admitted capacity remains exactly two.

Selected bootstrap and post-verdict paths must skip the legacy global
`save_hidden_for_mtp`; the actual owned rows already exist. Likewise, the
ordinary post-prefill `normalize_ssm_states` call is a refused legacy mutation
on the selected model, not a warning to ignore in a supported driver.

EP-v2 retirement in `mod_helpers.rs` already avoids survivor compaction;
preserve it. `lifecycle.rs::finish_sequence` currently caches a sequence,
logs/ignores a free error, then sends F1 free-and-reallocate. Selected cleanup
must neither prefix-cache the request nor continue F1 after failed cleanup.
Immediate-finish prefill, slot reuse and C2 -> lone slot 1 -> C2 require actual
entrypoint tests as well as ordinary decode-loop cases.

## Issued-command failure is not ordinary scheduler shutdown

Current generic failure/cleanup is unsuitable for uncertain paired completion:

- `verify_dflash_step.rs` marks only one request finished on wire/target errors.
- `lifecycle.rs` may then free state and send F1.
- `scheduler/mod.rs`'s normal drain sends further F1 and shutdown commands.
- `serve_phases/build.rs::maybe_run_ep_worker` breaks on command error, frees
  all slots and returns success.
- Native CUDA backend Drop independently sweeps allocations. Model-only
  quarantine does not prevent those destructor-side frees.

A selected fatal path must be armed before the first F5/E1 header attempt,
not just before Model begin-verify. It must prevent ordinary unwind/drain,
further collectives, request recycling and backend destruction when completion
is unknown. A distinct nonzero process exit without Rust owner destructors is
the minimal local containment direction; do not mislabel protocol uncertainty
as a proved CUDA context-loss fault. This is not yet implemented or qualified.

Local exit alone does not bound a peer parked in a collective. An exact-session,
out-of-band peer supervisor must be designed and CPU-tested before admission.
The existing communicator exposes health/reconnect, not an irreversible
terminal abort interface; reconnect is outside this scope. A normal graceful
shutdown cannot be assumed to interrupt a blocked synchronous driver call.
Do not enable serving until this remaining requirement is resolved.

Test failure policy through actual command adapters and an injected terminal
sink, then subprocess-test the real exit path with Drop sentinels. Require no
later F1/shutdown/collective/free/sweep on a post-issue failure. These are CPU
tests, not permission to inject faults into either Spark. Native testing starts
with healthy eager correctness, bounded memory, both-rank receipts and clean
stops; graph replay follows only after that control is correct.

Independent review adds two important limits. An outer Drop guard or
`catch_unwind` runs too late to stop inner owners being dropped during panic;
an armed selected-session panic hook must exit before ordinary hooks/unwinding.
Likewise a Result wrapper cannot retroactively prevent a callee's temporary
destructors before it returns Err. Audit the actual inner ownership and cleanup
seams; do not claim arbitrary no-Drop containment from the outer adapter alone.
The head's F5 scope extends through accepted-count exchange and state completion,
not merely the target return. Worker scope starts before the first EP-v2
preamble receive, which can fail before there is a classified command or slot.

A dedicated supervisor should pin a fresh session nonce, both immutable full
container IDs and image identities, with restart disabled. It must not target
reusable container names. Controller-only SSH monitoring cannot cover controller
or network loss; that stronger fail-closed contract requires node-local expiring
leases. Healthy shutdown needs a post-drain disarm handshake before the first
normal rank exit. No watchdog or forced native fault experiment is implemented
or authorized by this audit.

## Bootstrap and test integration details

The real non-DFlash bootstrap in `mtp_step.rs` sends the already-emitted first
token through scalar target decode, samples and emits the second token, then
proposes. Selected bootstrap must preserve that sequence using the actual
per-request sampler and the model's retained prompt-tail/decoded-bonus pair,
without the generic global hidden save or adaptive rebootstrap. A request
finishing at either emission must be retired without an unnecessary E1.

F0 prefill, scalar bootstrap and F1 retirement also issue rank-wide work. A
fatal policy limited to E1/F5 cannot make selected admission safe. The existing
prefill error arms free local state and send F1; event-record/wait failures are
logged and ignored. Selected integration must intercept those real branches.
For the single-chunk profile, worker F0 still reaches the legacy normalization
call in `model/impl_a2.rs`; excluding the multi-chunk scheduler path alone does
not remove that worker mutation.

At the served context/output boundary, complete the full issued K5 state
transaction before emission, then honor the existing `emit_token` ceiling.
Do not issue a new fixed-four proposal after finish, truncate width to fit, or
silently enter ordinary decode. Include accepted prefixes crossing the served
ceiling, output caps 1..5, EOS at every accepted/bonus position, cancellation
before and after issuance, and a surviving slot 1 in the driver tests.

The existing server repair fixture fabricates target predictions through a
plain Model mock; it is not proof of the sealed actual paired capability.
Driver tests must also exercise the real constructed TransformerModel/head.
If a cross-crate fixture is needed, use an explicit test-only feature and reuse
the existing owned-byte model fixture; do not unseal the production capability
or duplicate private lease constructors to make a server mock implement it.

## Graph/control distinction

The current K5 graph cache in `model/trait_impl/verify_d.rs` is already keyed
by `(seq.slot_idx, k)`, not merely K. This avoids one obvious cross-slot cache
key collision but does not prove native paired replay. Actual state pointers,
metadata uploads, slot reuse and secondary-stream ordering must still match.
Initial serialized native control should not be described as a ten-row batch.
Shared [5,5] target work and batched proposer work remain subsequent, separately
measured optimizations against that control.
