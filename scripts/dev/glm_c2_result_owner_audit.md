# Paired GLM: inner Result-owner follow-up

Source audit, 2026-09-08, while B1 transport and T1 process containment finish
CPU qualification. No activation, native safety qualification or new error
policy is implemented by this note. Read with the serial scheduler audit and
terminal-session plan; this narrows their mandatory inner-callee cleanup audit.

## Concrete boundaries still needing implementation

- `model/trait_impl/meta.rs::alloc_sequence_dispatch` owns a local `SlotGuard`
  before its first SSM zero. Errors from zero/synchronize, layer construction,
  reset/synchronize and proposer allocation return through that guard's Drop.
  `ssm_pool.rs::SlotGuard::drop` releases the target slot to the free list.
  An outer server `require(alloc_sequence())` cannot prevent that release.
  The selected allocator must retain or neutralize the guard on uncertain
  failure before returning; ordinary allocation semantics must remain intact.
- `model/impl_a2.rs` F1 takes the old sequence out of its slot, frees it, and
  constructs a local replacement. Existing paired retirement neutralizes the
  old SSM guard on early retirement failure, but replacement allocation and
  the post-allocation slot-identity error still require explicit ownership
  treatment. Prefer keeping selected sequence owners in the caller's slot
  throughout fallible transitions. Do not claim a borrowed outer worker guard
  solves locals inside this function.
- `model/trait_impl/sequence.rs::free_sequence_dispatch` waits/zeros/synchronizes
  selected SSM state and refuses early errors, but later graph destruction
  errors are logged and ignored. Target-slot and KV-block release already
  precede that graph loop. A selected path needs an explicit completion order
  and error propagation before further recycling or teardown. No new graph
  acknowledgment is implied; audit real graph/resource lifetime dependencies.
- `model/types.rs::release_pools` closes the paired capability first, but after
  successful close its generic release accumulator continues other releases
  and `sweep_unreleased` following the first resource-release error. A server
  wrapper sees that error too late to stop those later calls. Selected teardown
  requires an inner stop-on-first-error boundary, including the resource
  implementations themselves; changing only the outer accumulator is not a
  proof that those callees stop internally.
- `main_modules/serve_phases/build.rs::maybe_run_ep_worker` currently logs a
  command error, breaks, frees every slot and returns success. Selected worker
  integration must intercept before this normal drain, starting before the
  first idle-command receive and continuing through command completion.

## Existing useful ownership properties

`glm_c2_handoff.rs::try_glm_paired_propose` and `try_glm_paired_eager` temporarily
take proposer state, capture the operation Result, restore state into the
borrowed sequence, and only then propagate the Result. Their ordinary Err
paths do not drop that proposer owner. Panic remains a separate T1 ingress
requirement. Host Vec/error/lock-guard destruction is not device deallocation.

`free_sequence_dispatch` already neutralizes the sequence SSM guard on failed
paired retirement, and takes it before the selected secondary-event wait and
zero/sync. Preserve those guarantees. They do not cover every later cleanup
error, every allocation failure, or the native backend's independent Drop.

## Bounded next code slice and tests

After current code freezes, have the model author propose the smallest
selected-only allocation/retirement changes before T2 worker integration.
Use the actual paired construction and owned-byte backend. Derive failure
ordinals from successful allocation/free traces; assert no failed target slot
is returned by inner Drop, no replacement is published, and no later release
or graph operation follows the selected failure boundary. Include successful
retirement/reuse and unpaired allocation/free controls. Keep real subprocess
terminal integration separate from these Model ownership tests.

No global GpuBackend or CommBackend fatal callback is justified by this audit.
If a selected inner non-returning ingress is necessary, specify its exact
authority and callsites for review rather than spreading an error policy into
ordinary models. F0/bootstrap compute workspaces and per-resource teardown
implementations still need their own concrete owner inventory before the full
T2 closure claim; this note is not that full inventory.
