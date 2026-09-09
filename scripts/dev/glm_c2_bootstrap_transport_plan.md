# GLM C2 immutable bootstrap preflight and scalar transport

Status: implemented, awaiting final source freeze/review and root commit. Root
and reviewer approved the original plan SHA569b9109a263892907eb13fd1bd873ff5cbee1da3301c5e0b10c3807f2debf22.
Implementation starts after emitter `96f83f7f`, with fixture seam `d08daf41`.
No serving activation, native or throughput qualification is claimed.

## Scope and existing evidence

Implement the first prerequisite of `glm_c2_serial_driver_plan.md`, not its
driver or admission. A genuine paired Model owns validation, scalar wire send,
actual target decode, and immediate owned H[P] publication. No factory caller,
flag, generic Model method, default capability, sampler, F0 change, scheduler
activation, process-fatal registration, supervisor ticket or native deployment.

The current `paired_before_decode` reaches mutating handoff `begin_decode`;
`paired_target.rs` checks the actual primer/tail before setting pending_target
and writing. Existing legacy scheduler call order broadcasts before entering
Model decode. That source observation is not a runtime RED for a new API.
New-entry TDD will start with explicit unimplemented, non-sending scaffolds;
never install a deliberately bad live sender just to manufacture historical RED.

## Exact ownership and files

Production changes are confined to these existing files under spark-model/src:

- `speculative/glm_paired_execution.rs`: required sealed execution methods
  `validate_bootstrap(&SequenceState, u32) -> Result<()>` and
  `bootstrap(&mut SequenceState, u32) -> Result<DevicePtr>`.
- `speculative/glm_repair.rs`: required GLM-only handoff `validate_decode`
  with immutable actual proposer state, alongside existing begin_decode.
- `layers/glm5_mtp/paired_target.rs`: one private immutable bootstrap plan,
  actual-head validator, and begin's reuse of that same plan.
- `layers/glm5_mtp/paired_prime.rs`: actual Glm5MtpHead delegation; no default.
- `model/glm_c2_predispatch.rs`: execution delegation and narrowly shared
  target arena/profile/map/budget validation.
- `model/glm_c2_handoff_decode.rs`: rank-independent actual Model bootstrap
  validation and reuse before begin, retaining take/restore ownership ordering.
- `model/glm_c2_transport.rs`: head scalar sender and selected worker helper.
- `model/impl_a2.rs`: only selected scalar wildcard delegation; legacy branch
  remains literal. Do not alter E1/F5, first-word receive, F0 or slot replacement.
- `model/glm_c2_handoff_tests.rs`: registration only after root releases graph.
- `model/glm_c2_handoff_cleanup_tests.rs`: root-approved test-only amendment:
  healthy peer bootstrap occurs before installing its intentionally incompatible
  cleanup-only SSM pool. Fault setups and resource assertions are unchanged.

New unit children under model, each <=500 lines, with SPDX AGPL headers:
`glm_c2_bootstrap_transport_tests.rs` (controls, owner/profile/budget checks),
`glm_c2_bootstrap_transport_fault_tests.rs` (issued boundary failures).
If splitting is required, request a named helper child before widening scope.
No fixture numerical edits, test-only Ready constructors, manifests or features.

## Private plan and state transition

`Pool::bootstrap_target_plan(&self, actual_state, input, token, ctx)` returns
only a private slot index, not a transferable authorization or new public view.
It retains genuine backend/pool Arc/slab/generation and canonical private block
identity validation, active/nonfailed/nonretiring/nonwriting ownership, global
scratch-idle/producer-failed refusal, exact primer binding and owned tail.
Binding includes actual sequence slot, capture generation, prompt and original
capture/normalization allocation identities. The tail is the owned row0 H[P-1],
not a reread of whichever request last overwrote global capture.

Require actual target position P, private cursor P-1, valid vocabulary token,
no published bonus and no pending target. Use checked P-1 arithmetic at this
boundary. The immutable actual-head entry downcasts real state and shares all
checks. Begin obtains the lock and reuses the plan immediately before setting
pending_target and writing. No allocation, copy, lease claim or phase mutation
in validation. Existing stream-capture queries are allowed; do not call this
zero backend interaction. Repeated validation is observational, not a lock held
across transport; Model begin revalidates after issue.

Existing publish remains authoritative: actual target append to P+1, actual
token tail, private cursor still P-1, matching pending token and live lease;
copy actual normalized H[P] to owned row5, then synchronize the default stream
before publication. No late global hidden snapshot or fake success receipt.

## Actual model storage and scalar budget

Reuse paired_profile, paired_ssm_bindings, paired_input and existing checked
arena spans, dense map and inactive HSS/prefix facts. Preserve actual TP2/EP2
rank agreement, BF16 target policy, real slot/capture/adapter identity and
stream-not-capturing checks. Preserve selected scalar eager behavior already
committed; do not introduce graph support or a generic graph-policy knob.

Factor the current target preflight so both bootstrap and existing K5 require
the established paired arena envelope (five rows, actual draft/metadata/scratch
capacities and disjoint owners). Parameterize only a private additional-row
budget with explicit callers 1 and 5. Existing K5 checks and callers retain 5;
bootstrap checks one actual scalar row. Both validate the current dense16 map,
checked end, actual max table length and additional <= actual free blocks.

For bootstrap end=P+1, needed=ceil((P+1)/16), matching scalar decode's
floor(P/16)+1. For K5 end=position+5. Do not mislabel P+5 as proof of the first
post-bootstrap K5 at P+6: subsequent actual proposal/verify preflight checks
that later position independently. This is not a context-limit expansion;
first native admission remains cold complete <=resolved1024 elsewhere.
No new target allocation or eviction occurs during preflight.

## Head and worker ordering / failure boundary

Public validate_bootstrap is head-only: actual wire profile rank0, EP-v2,
distributed enabled, actual two slots and live command allocation, then shared
local owner/storage checks and checked slot conversion before any command.
bootstrap repeats these checks itself; a prior caller check is not trusted.

Within the issued scope, send the existing two scalar words [slot], [token]
with `ep_broadcast_seq_and_cmd(..., true)`, then actual Model::decode using
gpu.default_stream(). Return its real logits pointer only after normal owned
bonus publication succeeds. Every header-attempt-or-later Err maps through
existing paired_transport_error, including command upload/broadcast failures.
No extra packet, ACK, synchronization or speculative hidden-save operation.

Worker's selected scalar branch uses a narrow helper: actual wire profile rank1,
same rank-independent local bootstrap preflight, real default-stream Model
decode, and any helper Err mapped to paired_transport_error. Existing ordinary
scalar dispatch stays literal when no paired capability exists. This closes the
current selected scalar owner-only error latch; E1/F5 framing is unchanged.

First v2 preamble receive, malformed slot resolution and other earlier worker
errors occur outside this scalar helper. They remain the explicit future T2
whole-ep_worker_step no-unwind scope. Pool terminal latching alone neither stops
a peer blocked in NCCL nor prevents callee-local cleanup before Err. No claim
of process containment, native health or driver/hardware safety in this slice.

## Actual tests and RED/GREEN sequence

Use the shared genuine Fixture and Wire already committed, through existing
internal test support. Actual Model::prefill creates distinct prompt owners;
do not use transport_test_fixture::bootstrapped, which already decodes them.
Use existing isolated child-process env setup, never process-global set_var.
Numerical kernels remain explicitly sentinel boundaries, not GLM numerics.

1. Characterize real prefill-only tail/private P-1, existing scalar decode and
   ordinary worker framing before changing production. Distinct owners and
   both prefill/dispatch orders prevent a global-capture substitute passing.
2. Register new API scaffolds with explicit unimplemented Err and no wire.
   Run actual valid repeated-validation and successful transport assertions to
   obtain behavioral new-entry RED. Compile errors are separate receipts.
3. Implement immutable shared checks. Validate repeatedly without copy/kernel,
   wire, state/token/map/free-list/slab changes; then actual bootstrap succeeds.
   Missing/unprimed/retired owner, wrong slot/generation/phase/token, duplicate
   bootstrap, peer Produced and actual profile/storage/map/capacity negatives
   refuse before any header. Safe restored metadata permits later valid work;
   tests do not forge a Ready or modify private ownership to manufacture it.
4. Scalar budget controls include P15/P16/P17 and actual exhausted/free-block
   boundaries. Existing K5 budget tests retain their semantics and cover that
   later position has its own five-row requirement. Actual SSM binding negatives
   reuse established real-pool fixtures where applicable, never fabricated guards.
5. Successful head capability sends exactly two words per owner. Replay these
   into actual rank1 ep_worker_step, both owner orders. Require actual target
   position/token append, unchanged P-1 private cursor until proposal, correct
   owned tail/bonus, untouched peer bytes and default stream7. Rank1 must refuse
   the head-only execution entry without wire; its actual receiver succeeds.
6. Derive fault ordinals from a completed control and identify exact command
   uploads/transfers, target event, owned bonus D2D destination and immediately
   following sync. Inject each head header and those writer boundaries; test
   both addressed owners and head/worker writer paths. After helper failure a
   later good Sync cannot allow either owner to retry/progress or allocate; peer
   owned KV/slab remains unchanged. Do not assert rollback of victim writes.
   Earlier receiver-preamble failures are tested/reported as outside this latch,
   not relabeled covered terminal behavior. Preserve legacy None/scalar controls.

Each control must establish it reached the intended real boundary before an
injection can count. No staged broken sender for historical RED; source-order
evidence and newly scaffolded API RED remain separately labeled in receipts.

## Gates and handoff

Root grants the later exclusive model/source/Cargo window after emitter and
exact-tip gates. Use established controller CPU target/unit libraries, offline
model no-default-features tests and -j4; capture exact environment/commands,
source hashes and terminal statuses in campaign `glm-c2-bootstrap-transport`.
Run focused children, complete handoff regression, non-test model check,
fmt/scoped lint and new-file SPDX/caps/diff checks. Report inherited failures
without broad fixes. Independent source review and frozen manifest precede
handback; root owns commit, exact-tip broad model/server qualification and all
future native work. No serving or throughput claim follows from these gates.

## CPU receipt scope

Campaign: `/home/abc/storage/models/atlas-campaigns/20260908/glm-c2-bootstrap-transport`.
`scaffold-red.log` is a missing-import compile failure, not behavioral RED.
`scaffold-red-runtime.log` is actual new-entry RED: all6 executed tests fail at
the explicit nonsending unimplemented entries; frozen five-source hashes are
in `scaffold-red.sha256`. `first-green.log` executes those6 PASS. Expanded
coverage adds P15/16/17 actual free-block budget, unprimed/missing/retired owner
and peer Produced refusals; both rank1 head entry methods refuse zero-wire.

`expanded-focused.log` is a test trait-import compile failure, not RED.
`expanded-focused-runtime.log` has7 PASS/1 failed positive-control setup: the
test used the handoff subprocess environment with a Wire that requires the
existing verdict all-gather environment for E1. Reusing the established verdict
isolator corrected only that control; no production or numerical change.
`handoff-green.log` has126 PASS/2 failed old cleanup-only healthy controls:
decode after installing incompatible SSM geometry is now correctly refused.
Root approved the ordering amendment above. Its later post-failure decode Err
may also be geometry refusal, so isolated session-latch evidence comes from
the genuine valid-pool transport tests, not that particular cleanup assertion.
Existing real SSM pointer/SlotGuard negative tests remain regression evidence
for the identical shared preflight; no duplicate SSM fixture was added.

Commands use unchanged `glm-c2-eager-bootstrap/cpu-command.sh`, SHA
`b1eab741ff257615253831844eff9e3607b868c416f0e13a88a464e7b09c7659`.
It exports ATLAS_SKIP_BUILD=1, CUDARC_CUDA_VERSION=13000 and the campaign
cpu-target; LIBRARY_PATH is campaign unit-libs, LD_LIBRARY_PATH adds the
existing cu13 and nccl vendor library directories. It runs `cargo "$@"
--no-default-features --offline -p spark-model --lib -j4`. This is CPU recording
backend evidence, not native kernel execution. Arguments/logs and exit statuses
are retained; final focused/handoff/non-test/style receipts complete the freeze.
Scoped Clippy exits101 in four unchanged spark-runtime too-many-arguments stub
errors before checking this model slice; do not claim a clean lint pass.
