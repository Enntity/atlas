# Selected paired allocation through actual slot identity

2026-09-09. Root and independent reviewer approved the plan at
`3bf6f2c8c9004da9e64d9494bf7282abf29cb76730b67f55464e2445de6b71c5`.
Implementation begins after exact-tip/archive closure of selected ownership
`07084851` and explicit engine source/Cargo handoff. This is a bounded prerequisite for B2 churn,
not factory/admission, scheduler, supervision or native activation.

## Actual gap

The private paired Pool chooses its lowest healthy inactive slot. Target
`SsmStatePool::claim_guarded` pops a LIFO free list. Freeing head slots 0 then 1,
even in separate scheduler ticks, leaves private candidate 0 but target pop 1.
The new ownership check correctly refuses this mismatch before GPU work.
Descending retirement within one tick cannot solve arbitrary across-tick churn.
Worker F1 immediately reallocates its addressed slot; head retirement and later
admission are separate. Do not solve this by guessing an allocation order.

## Minimal proposed change

Add a crate-private `SsmStatePool::claim_specific_guarded` beside the existing
guarded claim, reusing the existing locked `claim_specific` operation. A success
returns the actual pool's non-cloneable SlotGuard for precisely that removed
index; an unavailable/out-of-range index returns Err without mutating any slot.
No arbitrary caller may fabricate the guard. Guard Drop/take/migrate/release
semantics and the generic LIFO claim remain unchanged.

Only selected `alloc_sequence_owned` supplies the index from its actual paired
Pool's read-only `validate_allocation`, never a public request slot or configured
capacity. Failure to claim that supposedly available matching target means the
two actual owners disagree: latch the selected session and return before GPU
work, without claiming/quarantining an unrelated available target. The expected
F1 addressed-index check remains before any target initialization. The private
claim still repeats its existing checks; successful publication still occurs
only after all initialization. No allocator free-list policy change, new private
lease API, reservation registry or head idle-sequence stash.

## TDD and bounded qualification

First reproduce actual Model free 0->1 followed by allocation, and the reverse
order, using genuine original owners on both ranks. Retain tests for one-owner
replacement, both-free churn across separate calls, alternating reuse and
reversed prefill/verification order. Require actual private index == target
guard index, disjoint reserves, exact reserve return and unchanged peer state.
The current positive lifecycle test's unrestricted retirement tail is useful
RED evidence; do not replace it with a policy-only simulation.

Unit-test actual specific guarded claim for both free-list orders, occupied and
out-of-range indices, no accidental peer claim, Drop returns exactly once, and
take prevents later release. Existing generic LIFO/RAII controls must remain.
The artificial held-target0/private-free0 mismatch must now refuse without
claiming the unrelated free target1; preserve terminal latch/no-GPU assertions.
Re-run allocation/F1/retirement fault matrices: no post-error recycling or later
backend operation, no foreign guard authority. Require independent source review,
model/server CPU gates and exact-source receipts. No performance claim.

## Implementation evidence

The first churn run stopped at a test-control assumption (target free255 rather
than256): the actual Model retains its padding block outside either sequence.
That is not the allocation RED. The corrected control derives target reusable
budget from actual free blocks plus the two disjoint sequence maps.
`churn-behavior-red.log` then reproduced the intended actual Model allocation
refusal after free0->1; single-owner alternation passed. `held-target-red.log`
separately showed that the prior mismatch path consumed unrelated free target1.

The selected allocation helper now uses only the actual private candidate for
the new guarded-specific claim. The guard wrapper reuses existing claim_specific;
generic claim_guarded, Drop, take, migrate and release remain unchanged. Matching
target refusal latches the actual selected session before GPU work and leaves
the unrelated target available. Unit coverage reuses the existing CPU-only bare
pool fixture to inspect actual free-list and non-cloneable guard transitions;
it introduces no Model/lease constructor or device-default policy.

After the first aligned GREEN, the original lifecycle [victim,peer] retirement
tail was restored. New actual both-order churn repeats three full allocation,
reverse priming/K5/repair rounds on both ranks; alternating single-owner reuse
retains peer token, KV, slab and reserve checks. Private reserves can exchange
their physical blocks after both owners retire: tests require exact union and
disjoint ownership, not an invented permanent block-set identity per slot.
The prior checkpoint's safe-refusal churn limitation is resolved by this narrow
follow-up; server dispatch, factory/admission, supervision and native qualification
remain separate. Raw receipts live in the external campaign directory
`glm-c2-aligned-allocation/`.
