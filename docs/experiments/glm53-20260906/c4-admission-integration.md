# Bounded concurrent-MTP serving admission

2026-09-09. Host-side integration, not native throughput qualification.

The authenticated recipe now admits two through four owners while retaining
exactly two ranks. Factory construction takes the explicit capacity; armed
registration requires actual Model capacity to equal the recipe. Head admission
and worker slot retention use that immutable registered value. Retirement visits
physical slots without compaction and preserves free/F1/health/Done ordering and
terminal retention on failure. Compute still serializes physical pairs; this
does not implement the wider M15/M20 layer-major traversal.

Private storage planning is shared between actual construction and preflight.
At context2044, each indexed owner needs5,423,104 payload bytes, including private
K/V, pooled index, raw tails and hidden slab. Reserves are10,846,208 at C2,
16,269,312 at C3 and21,692,416 at C4. Actual constructor allocation receipts,
excluding shared weights, agree for capacities2..4 with and without indexing.
The allowance is added once to inference reserve; existing target-state,
snapshot, arena and4GiB headroom terms remain separate. Invalid capacity,
admission mismatch and unqualified context refuse before backend initialization.

## Evidence

Campaign directory:
`/home/abc/storage/models/atlas-campaigns/20260909/glm-native-controller`.

Before implementation, actual runtime RED checks reproduced factory/recipe
capacity3 refusal, four-owner retirement refusal, missing10,846,208 private
bytes in C2 preflight, and false registration of Model4 under recipe2. The
registration RED reached both actual `registered` witnesses; it was not a
compilation or startup failure. Retained logs use `c4-*-red` names.

Focused GREEN logs:

- `c4-private-storage-green.log`:23 actual construction/ownership checks,
  including reserve-versus-allocation receipts.
- `c4-factory-green.log`:10 factory controls, including exact-capacity
  construction and full-pool refusal/reuse.
- `c4-preflight-green.log`:13 accounting/admission controls.
- `c4-retirement-green.log`:6 retirement controls, including physical2/3 reuse
  and failure retention without proceeding to the next F1.
- `c4-scheduler-integration-green.log`:14 existing selected scheduler checks.
- `c4-startup-green.log`:2 startup controls; `c4-recipe-green.log`:18 wire checks.
- `c4-registration-{mismatch,valid2,valid3,valid4}-green.log`: real inherited
  startup, actual Model registration, and actual head/worker shutdown through
  two guard processes and the unchanged paired release protocol. The mismatch
  must terminate before either registered witness.

The connected fixture uses host-backed actual Model ownership and scripted
local command transport. It does not execute CUDA/NCCL or prove numerical
quality. Formatting and scoped diff checks pass. Full workspace Clippy remains
subject to the previously recorded unrelated baseline errors; no clean full
workspace or native qualification claim is made.

## Next execution gate

Build the committed source and refreshed recipe-validator helpers on ARM64,
pin all artifact digests, and prepare bounded capacity4 recipes at context2044.
Run guarded warmed C1/C2/C3/C4, preserving coherence, real tool calls and
per-request needle checks, memory receipts, and actual two-rank release/exit.
Compare against the qualified C2 image before claiming gains. Then implement
wider layer-major weight reuse; the standalone M15/M20 arithmetic win is not
yet integrated. Large-context qualification remains explicitly required in
`mtp-c3-c4-next-plan.md` after semantic-index and memory prerequisites.
