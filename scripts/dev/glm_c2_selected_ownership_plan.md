# Selected GLM allocation and retirement: inner Result ownership

2026-09-09. Root and independent reviewer approved the revised plan at
`49aef05dff5d8bbbc65d1daa55cc3a8e87c7448ccac929a75aadea05baac0308`.
Implementation followed the explicit source/Cargo handoff after eager closure
`b9dbbef7` / `4928505c`. Factory/admission/native remain OFF.

## Scope and exact source findings

The slice closes Result-returning allocation/F1/retirement cleanup *inside*
Model, before a later server fatal guard receives the error. It does not add a
global callback, generic error-policy framework, new GPU owner, activation flag
or catch-unwind. T1's armed panic hook remains the separate pre-unwind boundary.

- `trait_impl/meta.rs:229` claims a local `SlotGuard`; zero/sync, layer-state
  construction, reset/sync and proposer allocation use `?` before the guard is
  moved into `SequenceState`. Its Drop returns the physical target slot.
- `ssm_pool.rs:790` already provides `take()`: neutralize the guard without
  releasing the index. No SlotGuard allocator or Drop semantics need changing.
- Actual selected GLM state construction is shallow: target linear layers use
  pool pointer views; GLM MLA uses `Qwen3AttentionLayer::alloc_state`, returning
  `EmptyLayerState`. The actual MTP body is also MLA. `SsmLayerState` and
  `EmptyLayerState` have no device-free Drop. The host vectors are not GPU owners.
  This is not a claim about arbitrary third-party TransformerLayer states.
- F1 currently takes the old sequence out of the caller's slots, then owns a
  local replacement across the fallible slot-identity check. A guard in a
  higher server frame cannot protect these local owners.
- Selected free currently neutralizes the guard before wait/zero/sync, but a
  zero error still permits a later synchronize, and successful zero/sync
  returns the target slot before fallible graph/proposer/metadata cleanup.
  Kgamma/fused graph errors are logged and ignored.
- `destroy_graph_cu` makes one `cuGraphExecDestroy` call and returns its error;
  it does not itself continue another graph destroy. `zero_slot`, `reset_slot`
  and `free_chunked_prefill_meta` already stop at their first backend error.
  The latter takes raw-pointer metadata before freeing; on error, remaining
  allocations remain backend-ledger-owned, not transactionally rolled back.
- Actual paired `free_state` synchronizes before returning its canonical
  private block reserve. Its subsequent block-list/view changes are host-only.

## A. Selected allocation with one shared construction body

Keep the existing public `Model::alloc_sequence` signature and legacy GPU
event order. Extract its current slot-dependent initialization body once into
a private helper returning the layer/proposer components. The caller retains
the actual guard until all Result-returning initialization succeeds; only then
does it construct/publish the SequenceState with that guard.

For selected construction only:

1. Before claiming a target slot, use a read-only GLM-specific allocation
   check on the actual paired Pool. Extract/reuse the existing `begin_claim`
   candidate checks (backend/open state, terminal latch, available healthy
   private slot, empty old reserve, checked next generation). Return the actual
   candidate **index** for comparison only; do not create a lease, advance
   generation or return a reservation. This prevents a later
   allocation from zeroing another target slot after a terminal private error.
   Actual `begin_claim` repeats the same candidate checks before transition.
2. Claim the existing target SlotGuard. Before **any** GPU initialization,
   require its authoritative physical index to equal the actual private Pool
   candidate index. This applies to initial ordinary selected alloc_sequence,
   not only F1. F1 additionally supplies a private expected addressed index;
   require all three indices to match. Ordinary alloc_sequence supplies no F1
   index but still requires private-candidate==target-guard. A mismatch
   neutralizes the claimed guard and latches terminal without zero/reset or
   constructing a replacement. No slot-claim API/free-list reordering is added.
3. Execute the existing zero/sync/layer/reset/sync/proposer steps through one
   Result boundary while the guard stays outside it. On selected Err, call
   `guard.take()` without release, latch the existing paired session terminal
   transition, then return the primary error with context. Do not synchronize,
   free, retry or fabricate a partially initialized SequenceState.
4. On success, move the guard and actual components into SequenceState once.
   A no-free-slot preclaim refusal remains non-mutating. No selected caller is
   permitted to treat a postclaim failure as a usable partial allocation.

The terminal transition is the same actual-backend-bound Pool latch introduced
by B1, not another failure owner. A narrow ownership-named Model helper can use
it; if the existing `fail_transport` name is misleading, rename that GLM-only
method to `fail_session` and update its existing delegates, with no new policy.

## B. F1 keeps the old owner in the caller's slot

Add a selected branch only; leave the ordinary F1 body unchanged.

- Borrow `slots[addressed].as_mut()` for old-owner free; do not take it into a
  local before the fallible call. On error it remains in the caller's slot,
  with its actual SSM guard already neutralized by selected free.
- Allocate through the shared private helper with `expected_slot=addressed`.
  Require private candidate==actual target guard==addressed before any GPU
  initialization. On any
  allocation error, the old retired entry (or original None) stays in place.
- Only a fully constructed, index-matched replacement is assigned into the
  slot. There is no fallible operation between returning that local success
  value and moving it into the caller's slot.
- F1 is already issued; predictable mismatch/refusal here is terminal under
  the eventual worker guard. This slice additionally latches the actual paired
  Pool on the selected F1 error path; it does not claim remote recovery.

This avoids publishing a mismatched replacement merely to protect its local
Drop. Fixed paired slot ownership remains mandatory; compaction stays refused.

## C. Selected retirement: finish fallible work before recycling

Route actual paired owners to a small private retirement helper at the start
of `free_sequence_dispatch`. The existing unpaired body remains unchanged.
Reuse existing ownership validators and real low-level cleanup operations.

1. Propagate the existing `Pool::begin_retire -> Result<Option<usize>>` through
   the GLM-only `retire` method instead of discarding it. `Some(index)` comes
   from actual backend/Arc/generation/private-reserve validation and marks the
   real lease retiring. Before taking the target guard or doing GPU work,
   require a guard is present, belongs_to(self.ssm_pool), and its live index
   equals both seq.slot_idx and this authoritative private index. Missing,
   foreign-pool or swapped-peer guards refuse; neutralize any held guard on
   that error, latch terminal and do no zero/release. Do not demand K5 numerical
   geometry merely to free a valid newly allocated/empty owner.
   After a successful exact check, take the guard without releasing and retain
   its index locally for the final successful release.
   
   `None` means **no live private cleanup authority**, not a newly minted
   retired lease or permission to use seq.slot_idx. The real paired free_state
   clears its private lease after returning the reserve. For a repeated old
   Model free, require its target guard already neutralized, target/disk maps
   empty, chunk metadata absent, no acquired adapter/prefix references and
   cleared SSM pointer views. Then return Ok immediately, before any slot-keyed
   graph lookup, wait, zero, proposer free or pool release. A no-lease object
   with remaining owners is an error, not authority to clean a replacement.
   Missing proposer state is also an error for selected cleanup. This uses the
   actual private transition plus absence of remaining owned resources; no
   public integer, fake lease, retirement registry or new receipt is created.
2. Validate the selected cleanup profile: actual base adapter, prefix cache
   disabled, no disk/HSS/compaction state. Ineligible cleanup returns terminal
   after neutralization, not through the generic cleanup branch.
3. Wait on the actual secondary event, zero the target slot, then synchronize
   the default stream. Each `?` stops immediately. In particular, a zero error
   must not be followed by the current unconditional synchronize.
4. Destroy the addressed slot's actual kgamma/fused cache entries now, before
   target/private reserve recycling. Remove one entry before its destroy
   attempt; on error stop and latch terminal, leaving unattempted entries in
   their real maps. Never reinsert/retry a handle with an unknown destroy result.
5. Free actual per-sequence chunk metadata using the existing stop-first-error
   helper, then call the real paired proposer `free_state`. The latter's own
   completion failure still stops before its canonical private reserve release.
6. Only after all fallible per-sequence operations succeed, release target KV
   block references, clear the host SSM pointer views, and return the target
   SlotGuard index exactly once. Selected prefix/adapters/HSS were refused,
   so their alternate release paths are not silently skipped as supported.

On any selected failure after retirement begins: no subsequent GPU/graph/free
operation; no target-slot or target/private block recycling after that failure;
the guard's later Drop cannot return the quarantined target index. Already
successful earlier operations are not rolled back. A terminal latch prevents
another free call from retrying partially destroyed resources. This is a
Model-path Result guarantee, not safety of eventual CUDA/Comm backend Drop.

## Actual behavioral RED controls before implementation

Use existing actual new_paired/new_legacy fixtures and bounded new test children,
not fake Ready values or policy-only machines. Obtain every injected ordinal
from a successful actual operation on a fresh equivalent fixture.

- **Allocation:** real tiny SSM pool, actual Model::alloc_sequence, both ranks
  and either available slot. Inject first H/conv zero, first completion, every
  relevant reset memset, final completion; a layer allocator failure uses a
  test layer at the actual alloc_state callback, not a detached Result stub.
  Capture the chosen physical index from successful control. After Err and
  all locals drop: that slot cannot be claimed by index/guard, its peer bytes
  are unchanged, no returned SequenceState exists, later good sync does not
  revive it, and a second alloc refuses before backend work. Current source
  should RED because local SlotGuard Drop releases the failed slot.
  Independently exercise actual mismatched initial allocation: arrange the real
  free-list/private-lease availability through real claims/retirements so the
  first healthy private candidate differs from the claimed target index. With
  no F1 expected index, allocation must still refuse before zero/reset, publish
  no sequence and leave the mismatched physical slot quarantined after Drop.
- **Proposer allocation:** retain actual paired head and exhaust its two real
  leases while target slots are initially free, so precheck refuses with no
  target zero. Exercise the actual private-body allocation callback failure
  separately where a fixture fault hook can preserve the real paired head;
  never replace it with an always-failing fake proposer capability.
- **F1:** actual EP-v2 worker dispatch with caller-owned slots. Free error
  leaves the old object in place; allocation error publishes no replacement;
  an expected-index mismatch with the real free-list order fails before target
  initialization and does not publish/release the mismatched claimed slot.
  Successful matched F1 replaces once. No allocation order or lease is forged.
- **Retirement:** derive wait, zero, sync, graph destroy, metadata frees and
  final proposer completion from actual successful free. Assert exact failure
  event, no later event/graph/free, unchanged target free counts and canonical
  private reserve until successful retirement, and quarantine after dropping
  the failed SequenceState. Current zero-error control should RED on its extra
  synchronize; graph error should RED on swallowed error plus early recycling.
  Add actual same-index foreign-pool guard, swapped-peer guard, and missing
  live guard negatives. All refuse before zero/graph/pool work; guard take must
  never reinterpret a foreign index as authority in this model's pool.
- **Graph ownership:** extend the fixture with opt-in real mock capture/end
  handle issuance and failable destroy recording; populate kgamma through the
  actual selected K5 capture path, not an arbitrary integer handle. Preserve
  default fixture event ordinals. This proves handle/cleanup ordering, not CUDA
  capture/replay numerics. Other slot graph entries remain untouched.
- **Success/legacy:** selected allocation→prefill/owned proposal→verdict→free→
  exact physical/private reserve reuse; repeated old free cannot recycle the
  replacement **or touch its graphs**. Keep the actually freed old object,
  allocate a replacement at the same target/private index, populate its real
  kgamma graph, then call old free: zero backend events, zero graph destroys,
  graph-cache identity unchanged and replacement remains executable. This
  specifically exercises the private no-lease outcome, not only free counts.
  Actual new_legacy allocation/free event sequences stay unchanged,
  including legacy best-effort cleanup semantics on injected errors. Run all
  existing ownership/transport tests, non-test check and full model CPU suite.

## File ownership and explicit exclusions

Expected production: private new `model/glm_c2_sequence_ownership.rs`; narrow
delegates/shared initialization extraction in `trait_impl/meta.rs` and
`trait_impl/sequence.rs`; selected F1 delegate in `impl_a2.rs`; existing GLM-only
handoff/private Pool allocation-check seam. A separate bounded shared allocation
child may hold literal extracted initialization if needed to keep files <=500.
No SsmStatePool free-list/SlotGuard/Drop changes, runtime allocator changes,
server driver, factory, command format or kernel/math changes.

Expected tests: new allocation, retirement and F1 children; only opt-in fixture
fault/capture controls, with actual legacy constructor retained. Upgrade the
paired common fixture's initial states from host_only + head.alloc_state to
actual Model::alloc_sequence so live selected owners have genuine target guards
from construction. Adapt the few old test helpers that manually re-claim those
guards; custom tiny pools receive real newly claimed guards as before. Do not
relax production guard validation to accommodate host-only test owners. Root owns final
commit/archive and assigns independent review before implementation handoff.

This is **not full T2 closure**. `types.rs::release_pools` and each underlying
ModelResource release implementation still need their separate exact-owner/
stop-first-error inventory. Selected eager decode's unconditional abort-capture
after an error is a separate logical slice. F0/local workspaces are under the
other author's read-only audit. Worker loop drain, scheduler locals/terminal
emit order, armed no-unwind fatal ingress and exact-peer external supervision
remain mandatory before native admission; backend Drop behavior is unchanged.
The target block allocator's fill-before-append failure can leave a newly
claimed block outside the sequence's map. It does not return/free that block
on the error path; do not add cleanup here or call that ownership transactional.
Identity retention for such uncompleted target allocations belongs in the
separate producer allocation inventory before a complete T2 owner claim.

## Implementation evidence and remaining allocation-order limitation

The common paired fixture now constructs both states through actual
Model::alloc_sequence. Existing manual guard claims became ownership assertions;
whole-model teardown controls explicitly drop neutralized external guard Arcs,
retaining private leases for the close-failure checks. This fixture change first
produced 93 PASS / 3 teardown-control failures, then 96/96 characterization PASS.
Those failures are not production RED evidence.

Actual ownership RED then produced 97 PASS / 9 expected failures, plus two
separate profile/no-authority failures and a retirement matrix with 2 PASS /
6 expected failures. Controls executed real allocation, F1, K5-created graphs,
metadata allocations and private completion; graph bytes are not replay math.
Additional allocation callback and legacy best-effort controls are later GREEN
characterization, not separately claimed REDs. Raw receipts remain under
`atlas-campaigns/20260908/glm-c2-sequence-ownership/` outside the repository.

A selected Model cleanup error now latches the entire paired session. Earlier
owner-producer failures remain owner-local where previously specified; their
healthy-peer continuation/reuse controls now run before the distinct failing
Model cleanup call. Post-cleanup tests retain raw canonical addresses captured
while valid and prove unchanged peer bytes without requesting revoked views.

This checkpoint deliberately refuses, rather than supports, arbitrary two-idle
allocation churn: target free slots are LIFO while the private Pool chooses its
lowest available index. Real frees [0,1] followed by allocation consequently
refuse target1/private0 before GPU work. The same mismatch can arise across
separate ticks; descending order within one tick is not a general solution.
The executed positive lifecycle attempt caught this at its final two-idle
allocation (`ownership-green-attempt.log`, lifecycle line231). Its bounded
success control now frees [1,0]; the exact original refusal evidence is retained
for a separately reviewed candidate-aligned guarded-claim follow-up. This slice
does not change SsmStatePool allocation/Drop semantics or claim general churn
support. Reverse priming uses actual aligned allocations then execution [1,0],
not host-only target states paired with direct head-only allocations.
