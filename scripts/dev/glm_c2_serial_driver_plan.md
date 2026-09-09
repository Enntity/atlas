# B2: selected paired SERIAL scheduler transaction

Status: root/reviewer approved bounded design; unreferenced driver/test scaffolds
are being drafted. Compiled implementation awaits prerequisite qualification.
No admission authority.

September 9 implementation staging: prerequisites are committed and root granted
the bounded engine window. Initial evidence is only actual two-owner bootstrap,
multiple steady-state rounds with worker replay in both vector orders, completed
early finish without E1, and checked-selection failure stopping later work. Reuse
the existing fixture and model-level all25 matrix; no new scripted-logit control,
expanded driver matrix, full regression suite or archive in this slice. The
larger qualification matrix below remains a later deployment prerequisite.
This is two complete slot-addressed K5/E1 transactions, not a [5,5] target batch,
not a throughput claim, and not permission to activate the paired constructor.
Read alongside `glm_c2_serial_scheduler_audit.md`, the selected ownership plan,
and the terminal/supervision plans. T2/T3 remain separate live prerequisites.

## Source boundaries and the smallest implementable slice

The real sealed `GlmPairedExecution` already supplies immutable
`validate_verify`/`validate_propose` and transport-owning `verify`/`propose`
(`spark-model/src/speculative/glm_paired_execution.rs`,
`model/glm_c2_predispatch.rs`, `model/glm_c2_transport.rs`). Their E1 packet binds
actual slot/generation/attempt/position; F5 owns header/width/tokens/target, but
returns BEFORE the worker's accepted-count receive. Never duplicate those packets.

First implementation slice: a private server `step_selected_serial(model,
active: &mut [ActiveSeq], sched: &SchedCtx, verify_ctx: &LogitsContext)
-> anyhow::Result<()>`, plus actual dependency-fixture integration tests.
Retrieve the sealed capability from that exact Model; no separately supplied or
mock-implemented capability, no public Ready, no selected boolean as authority.
Register the module for compilation/tests only; no scheduler live call, factory,
CLI/env flag, request admission, launch ticket or terminal-core registration yet.
Document any staged unused allowance narrowly. This is executable transaction
infrastructure with no serving caller, not a pure eligibility-policy scaffold.

Three prerequisite extractions need explicit review before the driver slice:

1. Add `validate_bootstrap(seq, token)` and `bootstrap(seq, token)->Result<DevicePtr>`
   to the existing sealed capability, not the generic Model trait. The current
   `paired_before_decode` calls mutating `begin_decode`; scheduler `mtp_step.rs`
   broadcasts BEFORE entering Model::decode. Factor the actual Pool bootstrap
   owner/position/tail checks (`layers/glm5_mtp/paired_target.rs`) into one
   immutable plan used both before the scalar header and by the real begin.
   Actual selected bootstrap transport validates profile/rank/target storage,
   sends the existing slot-addressed scalar token, then calls the real Model
   decode on its actual default stream. Header-or-later failure uses the existing
   paired transport terminal latch. Worker uses the narrow selected scalar helper;
   the ordinary nonpaired scalar dispatch remains unchanged.
   No reservation is claimed by immutable validation; begin still revalidates.
2. Share the existing emit body with an explicit logical emission position:
   legacy `emit_token` passes `a.seq.seq_len`; selected emission passes the row's
   `base+i+1`. `emit_step.rs:438,475` currently checks final canonical seq_len;
   after committing all accepted state it could stop at the FIRST emitted token
   near the served ceiling. Preserve the complete ordinary body and sampler/
   cancellation/stop accounting; change only its two ceiling operands to the
   parameter. NEVER temporarily rewrite canonical SequenceState for emission.
   Reuse `helpers::seqlen_force_stop` (position+1>=limit), not new ceiling math.
3. Bootstrap's `sample_token_with_grammar` returns Result but its reduce-only
   fast path still uses legacy bool `logit_is_positive` (sample_step.rs:400),
   which can swallow a copy failure before another read. Add a grammarless
   checked sibling over the SAME sampler body with explicit probe failure policy;
   reuse `logit_is_positive_checked`, keep legacy false->full-path semantics,
   and preserve existing sampler/penalty math. The scalar slow body is BF16,
   passes seed=None and top_n_sigma=0; sharing it does not establish seeded
   stochastic output parity or introduce request-seed behavior. No disabling fast-path flags as a
   correctness workaround. Existing checked VERIFY selection is not this scalar
   sampler. This prerequisite may be separately committed if it keeps review
   bounded; a Result signature alone is not strict bootstrap-copy evidence.

## Exact per-owner execution and state authorities

Validate the whole input slice first: live occupancy1..2, actual distinct slots
0/1, selected capability, grammarless/fixed4 profile and consistent host token
lengths. Require every live owner's actual temperature==0 BEFORE any wire work;
the first paired control is greedy with the existing penalty math, not seeded
stochastic sampling. Reject a mixed two-owner slice if either temperature is
nonzero (or non-finite); do not process the eligible peer first or silently
coerce temperature. Future admission must preserve this restriction. The actual
temperature-zero benchmark profile is unchanged.
Reject any requested logprobs/top_logprobs profile BEFORE any wire work
(including ActiveSeq.top_logprobs.is_some()); future admission must retain that
request fact. The ordinary bootstrap logprob extractor returns None on copy
error and is not eligible here. Do not silently omit requested logprob output
or expand checked extraction in this first control. Process actual slot order,
irrespective of vector order; never compact
slot1 into slot0. Presence of the model capability keeps the regime sticky at
occupancy1; generic `mtp_gate`, adaptive rung, capacity clamp, thinking fallback,
catchup/carry/global hidden save and draft-clearing ladder do not run here.
ActiveSeq owns last_token, pending_drafts, penalties/history/output state; Model
owns all hidden/KV/SSM authority. No scheduler hidden-pointer cache or second
per-request token ledger. Empty pending drafts means bootstrap ONLY at the
actual post-prefill position `seq_len==prompt_len`; later emptiness is Err.
The actual bootstrap validator additionally proves the live owned phase.

Before each transaction, honor existing cancellation/deadline/finished/output
ceiling checks. Cancellation before issuance causes no command for that owner;
it does not authorize cleanup within this step. Invalid live width/owner/input
is Err, not local finish-and-decode fallback. No speculative width truncation.

Bootstrap: F0 and the mandatory synchronous P-1 eager primer have already
completed, and the existing first-token path selected/emitted x[P]. If that
emission finished (including max_tokens1/EOS), no scalar decode or E1 is needed.
Otherwise capability.bootstrap consumes x[P] at target positionP and owns
H[P] immediately. Reuse `sample_step::penalty_params_for(PositionKind::Verify)`,
`penalty_history_scope` and the checked shared scalar sampler above to choose
x[P+1] from those actual logits. Honor the eligible temperature-zero request's
existing penalty/history semantics; no new seed behavior and no
raw-argmax substitute. Emit that token at logical positionP+1, then if still
live validate/propose exactly4 using this selected seed at actual seq_lenP+1.
Owned bootstrap repair pairs are (x[P],H[P-1]) and then (x[P+1],H[P]); never
feed the prompt tail as H[P]. If emitted token or concurrent cancellation ends
the request, no unnecessary proposal/checkpoint/global-hidden work follows.

Steady state, one indivisible issued-F5 scope per owner:

1. Copy `[last_token, pending_drafts[0..4]]` into a fixed five-token array and
   retain checked `base=seq_len`; validate issued IDs/owner via capability.
2. Call capability.verify (no second hand-written F5). Require exactly5 valid
   raw IDs and exact actual append: seq_len/base+5, token tail equals inputs.
3. Call `verify_pick_all_with_pipeline_checked(model,raw,a,verify_ctx,0)?`.
   Require exactly5 valid selected IDs. Its existing probe/full-copy errors are
   terminal here; no raw fallback, later owner step, emission or accepted send.
4. Compute a in0..4 from the first draft/selected mismatch; selected[a] is the
   externally selected bonus. It MAY differ from raw[a] because penalties apply.
   Send accepted count using existing `ep_broadcast_cmd(a as u32)?` exactly once.
5. Checked rollback keeps precisely inputs[0..a+1], with seq_len=base+a+1.
   The sampled bonus is NOT appended to seq.tokens yet: it is the next input.
   Do not use saturating arithmetic/min-pop to conceal malformed model output.
6. `record_glm_mtp_verified(base,inputs,a)?`, `trim_proposer_state(a,0)?`, then
   `commit_accepted_prefix(a+1,5)?`, matching worker F5 at impl_a2.rs:681–706.
   All must succeed before any visible accepted/bonus emission or early return.
7. Clear the now-consumed pending drafts; emit drafts[0..a] then selected[a]
   using the shared emitter with logical positions base+1 through base+a+1.
   Stop visible emission exactly as ordinary EOS/output/cancel rules require;
   target state remains fully completed, not partially rolled back again.
8. Only if still live, set last_token to the actual selected bonus and issue the
   next capability.propose at actual seq_len. Require exactly4 valid drafts and
   publish them into ActiveSeq only on success. No later peer starts before this
   whole owner transaction completes. No extra secondary wait/checkpoint unless
   required by the actual capability (which already orders its owned consumers).

Fixed K5 must fit actual target/private storage even when only1 visible token
remains. Served-context ceiling bounds visible emission, not a license to resize
K5 or enlarge owner storage. Validate the approved profile's speculative
headroom before activation; actual capacity refusal stays Err, not C1 fallback.

## Cancellation, retirement and terminal errors: explicit exclusions

Cancellation after F5 issuance cannot bypass accepted exchange/rollback/record/
trim/commit. After successful completion it can suppress all emission and E1.
Bootstrap likewise completes its issued scalar target before cancellation can
retire it. Recheck cancellation immediately before a subsequent proposal; an
E1 already issued must finish. Partial client-send/backpressure effects retain
existing emission behavior but cannot reopen an unfinished model transaction.

This driver returns errors; it never sets only the failing request finished,
calls send_error/finish_sequence, drops a request from active, sends F1/shutdown,
or retries after transport/model/selection failure. Future T2 must remain armed
over scalar bootstrap and the WHOLE F5-through-verdict/E1 interval and terminate
before ordinary scheduler cleanup. No inside-callee no-Drop claim follows from
Result propagation alone; existing selected ownership audits remain required.

Current lifecycle.rs:209–224 caches, swallows free failure, then issues F1;
mod_helpers.rs:309–320 removes the owner before invoking that cleanup. Neither
is selected retirement. A later checked retirement adapter must keep the owner
in active until actual selected free succeeds, then send the existing slot F1
under T2; no F1 after failed local free, no prefix-cache insertion, no survivor
compaction. Worker F1 free+alloc and replacement failure stay terminal. Only
after completed retirement may host removal/terminal Done occur. Extract the
host response/log body if needed; do not copy finish-reason math. Ordinary
shutdown drain (`scheduler/mod.rs:1088`) and teardown's continue-after-sync-error
policy are also excluded until selected T2 drain/peer supervision is integrated.

F0 prefill/error paths in prefill_a_step.rs/prefill_b_step.rs issue real commands
and currently perform best-effort error cleanup; they are NOT made safe by this
post-prefill driver. Cold single-chunk admission, first-token/immediate-finish,
checked F0 and strict retirement adapters are separate required wiring work.
Likewise a live branch must precede scheduler/mod.rs's generic arbitration
(before draft-clearing sites937/964), not merely wrap step_mtp. No serving
connection occurs in this first slice while these or T2/T3 remain absent.

## Test seam, genuine RED choices and file ownership

Use the pending actual `spark_model::model::glm_c2_test_support` fixture once
its author freezes it: Fixture::paired(rank), deterministic_logits, install_wire,
consuming into_parts -> actual Model/two genuine SequenceStates/Observer.
`test_support::test_owned_seq` consumes each actual state into the existing
ActiveSeq initializer. Observer provides bounded events/fault ordinals/private
cursor/free-block counts/snapshots; Wire records/queues actual command packets.
Do not duplicate the fixture or instantiate sealed leases. It is CPU sentinel
ownership/protocol evidence, not GLM numerical fidelity or real NCCL progress.

Tests call the actual new driver with actual head capability, and replay its
packets through real worker `ep_worker_step`; validate both streams/slot payloads,
worker state and complete queue consumption. Head fixture ranks are not confused
with actual worker rank1. For all25 acceptance pairs, if existing numerical
controls cannot force each prefix, request one test-only target-logit boundary
control from the fixture author, derived from the REAL issued drafts. Never
replace the verifier/capability or manually record a positive verdict.

Initial REDs: staged driver returns Err for a valid actual two-owner bootstrap;
actual ordinary verify characterization with EOS/output-cap shows commit/trim
skipped; forced target/probe-copy/accepted-send/record/trim/commit/E1 faults must
return Err with no following emission/peer command/retirement (not compile RED).
Bootstrap predispatch follows `glm_c2_bootstrap_transport_plan.md`: existing
header-before-decode order is source evidence, while the new non-sending API
scaffold supplies new-entry RED. Do not stage a bad sender to invent a prior bug;
the implemented validator must refuse invalid real owners before any header.
Bootstrap fast-probe-copy fault must reject without a subsequent full copy;
legacy sampler characterization must still recover through its original path.
Explicit-position emitter tests catch first-token premature ceiling retirement
without mutating canonical seq_len. Preserve the ordinary wrapper regression.

Positive matrix: bootstrap and multiple verdict/repair rounds; all25 a0..4
pairs, reversed active order, unequal histories, C2 -> slot1 survivor -> slot0
reuse, actual penalty-selected nonraw bonus feeding next E1. Caps1..5 and EOS
at every accepted/bonus position, cancellation before issuance/after target/
between emissions/before E1, dropped receiver, exact served-ceiling edges,
physical headroom failure and malformed0/3/5 draft sets.
Logprobs/top_logprobs requests must fail whole-slice validation with zero wire,
including when the other slot is otherwise eligible. Likewise test a nonzero
temperature in either owner of a mixed two-owner slice, in both vector orders:
the driver returns Err before any wire/target work for either owner. Include
non-finite temperature refusal and an actual temperature-zero positive control;
do not claim stochastic or seeded parity from legacy sampler regression.
Assert completed owned
KV/SSM/host prefix BEFORE first emitted token and peer scratch reuse. Preserve
default None-capability and existing sampler/emit/legacy-model tests unchanged.

Proposed ownership after approval: new server scheduler/glm_c2_serial.rs plus
small bootstrap/verdict/test children (each<=500), scheduler/mod.rs registration
only, emit_step.rs literal helper extraction and sample_step.rs checked-policy
extraction/test child (split if needed for the500-line cap). Model-author-scoped changes only
to sealed capability/predispatch/transport and paired_target immutable bootstrap
plan plus real tests; no generic Model trait/factory/build changes. Reuse sampler,
checked selection, emit semantics and actual model retirement; no generic
transaction framework. Root reviews this plan before assigning code/Cargo.

Test-only process-helper amendment (root/reviewer approved September9): the
existing glm_c2_fixture_test_process.rs helper takes the full exact test path,
instead of hardcoding the fixture child prefix. Existing fixture tests construct
their unchanged path; driver tests include the same helper and pass their own
path. Child environment and marker remain one source of truth. Confirm that
every subprocess actually runs one test: test-binary success with zero matches
is not evidence. No duplicated process environment or serving change.
