# Wider MTP producer and model integration

2026-09-09, source `77a58abb`, following layer source `bc3637fd`.
This is composed host-side implementation, not live serving activation or a
new native throughput/quality result.

## Implemented

- A distinct three/four-owner producer binds every actual issued proposal to
  its physical slot, generation, attempt, tokens and original normalized arena.
  It retains the whole group through every strict target commit. All verdicts
  validate before any detachment; failures during detachment retain the receipt
  and terminally fail participating owners.
- The actual model now performs one layer-major traversal for N=3/4 temporal
  K5 owners, using the wider KDA/MLA/FFN entries. Final normalization and vocabulary
  projection remain separate K5 operations. Noncontiguous physical selections
  such as `[0,2,3]` use packed row ordinals without changing private slot identity.
- Target uploads use fixed80-byte token and N×3328-byte metadata arrays;
  metadata ends at most46080 within the49152-byte window. Cold paired sequence
  allocation reserves2048 host token entries and the actual maximum block-map
  entries. Compute refuses insufficient host capacity instead of growing those
  vectors after issue. This does not claim that every existing dispatch/backend
  internal is free of host allocations.
- Local wider verdict completion detaches accepted rows/bonus into physical
  private slabs, performs every trim/strict commit, then permits new proposals.
- A separate fixed48-word E7 codec and six-word verdict codec require explicit
  shape/mode/bounds, canonical physical slots, full64-bit identities and zero
  inactive C3 padding. Existing E6 remains unchanged. The codec is not yet a
  connected command exchange.

## Verification

Logs live under
`/home/abc/storage/models/atlas-campaigns/20260909/glm-native-controller/`:

- `owner-producer-red.log`, `owner-wire-red.log`, `owner-compute-red.log`:
  actual runtime refusals before implementation, not compilation failures.
- `owner-finish-red.log`: the actual wider model produced the expected C3
  rows before reaching the intentionally unsupported verdict entry.
- `owner-{producer,wire,compute}-green.log`: actual issued-owner validation,
  canonical codec checks, and composed model verification/commit/continuation.
  The composed fixture runs C3 `[0,2,3]` and C4 on both ranks, checks every saved
  normalized row and physical slab, mixed acceptance `[0,1,4,2]`, and real next
  proposals in reverse order. Unselected-owner storage and host vector capacity
  remain unchanged. A malformed last verdict detaches nothing and prevents any
  owner from proposing again.
- `owner-composed-handoff-control.log`:148 existing/new handoff tests pass.
- `owner-composed-postcommit.log`:393 model tests pass on clean source77a58abb,
  including the codec and composed ownership controls; elapsed6.51seconds.
- Workspace formatting and whitespace checks pass. Non-test Clippy still
  reports the same ten pre-existing deny diagnostics in other code; no new
  diagnostic is reported, but this is not a Clippy-clean claim.

The model fixture executes actual ownership/model methods with byte-sentinel
target layers and a recorder backend. It proves row routing and continuation,
not KDA/MLA/FFN CUDA arithmetic, NCCL execution, generated coherence, tool calls,
needle retrieval or a serving speedup. Those native gates remain required.

## Immediate next chunk

1. Add a separate explicit cold wider-compute selection and sealed execution
   methods. Factor complete model preflight for use before a head command;
   do not infer wider mode from JointSharedM10 or enable it by capacity alone.
2. Connect the E7 codec to bounded in-place exchange (no `ep_broadcast_tokens`
   Vec allocation), actual head dispatch and worker reception before scalar-slot
   lookup. Require sentinel preamble0, real registry/capacity agreement and
   whole regenerated packet equality before target writes. Every post-header
   error, including internal model compute errors, must latch the session failed.
   The private compute helper intentionally relies on that outer terminal policy.
3. Connect scheduler selection: choose all3/4 owners before issuing, retain them
   through all verdicts/commits, then emit/propose. Keep existing pair/scalar drain
   paths and prohibit mid-transaction cancellation from dropping a member.
4. Rebuild native candidate from committed source and run bounded numerical and
   actual coherence/tool-call/needle gates before warmed C1/C2/C3/C4 timing,
   fresh-process repetition and normal paired release. Retain memory/no-swap
   guards. Do not run another standalone FFN benchmark in place of this test.

Latest actual C4 MTP serving remains source6e1e37f4 at about36.9 aggregate
full-wall tok/s. Full C1/C2/C4/C6/C8 targets, concurrent C6/C8 MTP, reference
workload comparability and staged large-context validation remain open.
