# Checked B-tile kernel family and private launch leases

Status: bounded implementation complete and source frozen, including root's
approved shared-T/arena addendum. Final independent source review is pending.
No CUDA, nodes or serving changes are authorized in this partition.
CUDA source promotion is committed3332c36e; its native gates belong to root.

## Outcome and partition boundary

Implement one completely resolved16-handle family and its actual checked typed
launches, behind the existing private `gate_up_repack` construction boundary.
Neither kernel handles nor raw routed tables become a new MoE sibling API.
No layout flag, loader invocation, forward selection, Ready capability or model
conversion caller. This does not activate the optimized layout.

The meaningful tests will call real source validation, real repack, real table
construction and real launch methods against a recording backend. They will not
test a disconnected selector or manufacture a successful unpublished descriptor.

## File ownership and sequencing

Extraction was committed as d6f4e554 before integrated implementation. Controller
Cargo transferred to this author, with one explicit frozen window for the KV
probe author, now committed as 2da770dc. No concurrent Rust/module-graph edits.
The approved addendum narrowly extends helpers_a and its shared-only phase;
the extracted legacy phase tests remain unchanged. Every new Rust file is <=500
lines.

Proposed private children of `gate_up_repack.rs` (use explicit `#[path]`):

- `gate_up_btile_kernels.rs`: complete family resolution and explicit policies.
- `gate_up_btile_binding.rs`: construction-only table lease and owner validation.
- `gate_up_btile_decode.rs`: checked BF16 rows1/2/3 launches.
- `gate_up_btile_grouped.rs`: checked prequant dense/separate/fused launches.
- Split their actual-backend tests into correspondingly named `_tests.rs`
  children and one byte-storing typed-ABI recording backend as needed.

`gate_up_repack.rs` gets private declarations and changes its private transaction
to consume the resolved family rather than two caller-supplied handles.
Existing transaction tests adapt to the real resolver. Do not widen the existing
native-source module or add public raw launch functions to `layers/ops.rs`.
Reuse the existing checked repack op and transpose op behind the family.

One prerequisite is a narrowly scoped table-construction receipt, described
below. It requires `ptr_table_build.rs`, the `ExpertPtrTable` declaration in
`moe/mod.rs`, and the manually assembled borrowed table literal in `helpers_b.rs`.
It must preserve every existing allocation/upload/free event and caller result.
The approved shared receipt addendum adds only its None initializer to init.rs.
No forwarding files, WeightStore or runtime GPU traits change.

## Complete family: exact module/export/ABI mapping

Resolution occurs outside actual stream capture, against the same live backend
as the source/transaction. Validate the target profile first, then resolve into
locals and return a family only when all16 handles are nonzero. Any lookup error
or zero handle returns no family; never substitute a T-layout handle.
No constructor is called by default serving in this partition.

| Module | Required exports | Ordered ABI |
|---|---|---|
| `glm_moe_btile_native_repack` | `glm_native_to_btile_u8` | 2ptr,2u32 |
| `transpose_u8` | `transpose_u8` | 2ptr,2u32 |
| `glm_moe_btile_decode` | `glm_btile_decode_word1/2/3`, `glm_btile_decode_vec1/2/3` | 21 parameters: input; gate4; up4; IDs; shared-gate(ptr,ptr,f32,ptr); shared-up(ptr,ptr,f32,ptr); N,K,top-k |
| `moe_w4a16` | `glm_moe_gate_up_btile`, `glm_moe_gate_up_btile_vecscale` | fused18 |
| `moe_w4a16` | `glm_moe_gate_up_btile_m64`, `glm_moe_gate_up_btile_m64_vecscale` | fused18 |
| `moe_w4a16` | `glm_moe_btile_m64_dense`, `glm_moe_btile_m64_vecscale_dense` | separate11 |
| `moe_w4a16` | `glm_moe_btile_m64_compact`, `glm_moe_btile_m64_vecscale_compact` | separate14 |

Separate11 is A-packed,A-scale,B-packed-table,B-scale-table,scale2-table,C,
offsets,sorted-token-IDs,experts,N,K. Separate14 appends worklist,total-tiles,
max-tiles. Fused18 is A-packed,A-scale; gate(packed-table,scale-table,scale2,C);
up(the same4); offsets,sorted-token-IDs; experts,N,K; worklist,total-tiles,
max-tiles. All launches use the caller's validated stream and zero shared bytes.

Word/vector decode choice and scalar/vector scale choice are explicit enums;
registration does not choose a winner. An explicit prefer-small fused request
may use M16 only for gathered rows<=5; otherwise select matching B-tile M64.
Missing handles cannot trigger fallback because complete resolution is required.

## Actual table capacities and provenance: proposed prerequisite

`ExpertPtrTable` currently contains only three public-crate DevicePtrs. A
caller-provided288 or pointer-span check cannot prove that a D2H read is in bounds.
Do not accept that raw struct alone as allocation authority.

The existing two builders in `ptr_table_build.rs` already allocate exactly
`n*8,n*8,n*4` bytes. Retain a private allocation receipt minted there only after
the actual allocations/uploads succeed: slot count, the three allocation base
pointers/extents and backend identity. Use checked arithmetic. Receipt fields
and construction stay private to the builder; inspection yields only a checked
validation result to the binding layer, not a forgeable public record.

Store this receipt with the actual table object. The manually assembled
shared/down scratch table in `helpers_b.rs` has no owning receipt and cannot be
bound as routed gate/up. Preserve its legacy behavior. Existing mutable pointer
fields must still match all stamped allocation addresses before any readback;
mutation invalidates the receipt rather than letting stale capacity authorize
another pointer. A receipt proves construction extents, not transformed bytes.
No new GPU allocation, table twin, re-registration or ownership transfer is
introduced by retaining these host facts.

The private table lease borrows all of:

- the successfully returned `UnpublishedBTileLayer` and its sealed source;
- mutable access to the actual existing gate and up table objects/receipts;
- the complete family and its live backend identity.

It rejects capture, foreign backend, non288 slots, missing/stale receipts,
overflow/alignment/capacity failures, table-table/table-weight aliases, and
aliased mutable owners before I/O. Then read each actual table once on the
explicit construction stream: four2304-byte pointer arrays and two1152-byte
scale2 arrays, exactly11,520 bytes. Read only after capacity authority passes.
All288 entries must match sealed source provenance: local packed/scales retain
their original allocation addresses and exact scalar bits; remote packed/scales
and scale2 match the actual null-placeholder contract (zero pointers,+0 bits).
Readback failures return no lease. No matrix/scales payload or full-vocabulary
read is involved. Do not allocate GPU buffers or retain model-sized host data.

The lease has no raw getter, Clone/Copy, conversion to ExpertPtrTable, or Ready
conversion. Launch methods require this lease; no sibling can construct one
from raw pointers. Its lifetime cannot exceed any table, source, or backend
borrow. Raw aliases cannot be magically revoked; exclusive construction and
future removal of legacy pointer views remain mandatory activation work.

Crucially, this is NOT a proposed serving owner that borrows a load-local
WeightStore forever. The actual factory already adopts the same WeightStore
into the model; the resident activation outline records that ownership seam.
Future owning Ready publication and legacy alias invalidation remain separate
loader work, not a new resident borrowed-store design.
Only that future owning Ready may publish tables to serving consumers.

## Checked arenas, dimensions and routing metadata

Derive available capacities from the actual BufferArena/BufferSizes and checked
offsets within those owners, not arbitrary caller-supplied capacities. Bind the
same ForwardContext/backend and source geometry. Host eligibility is exact
GLM N2048,K4096,288 experts,top-k8,TP=EP=2 matching ranks, Standard NVFP4,
no adapters/incompatible layouts. Logical rows and actual arena bounds must be
positive and <=1088 before conversion can ever be activated.

For rows R, expanded routes E=R*8, checked byte requirements are:

| Buffer | Required bytes |
|---|---:|
| BF16 token-major input | R*4096*2 |
| Gathered prequant A packed/scales | R*2048 / R*256 |
| Explicit route-major A packed/scales | E*2048 / E*256 |
| Each routed gate/up BF16 output | E*2048*2 |
| Expert offsets | (288+1)*4 |
| Gathered sorted-token IDs | E*4 |
| Decode route IDs | R*8*4 |
| Compact worklist | max_tiles*8 |
| Compact counter | 4 (retain current16-byte reserved prefix) |
| Each shared BF16 output | R*2048*2 |

Accept only known arena subranges; check nonnull/alignment, checked endpoint,
owner capacity and pairwise overlap among all simultaneously live writes and
reads, including tables/weights/shared inputs and metadata. Distinct slices of
one arena are allowed when their actual live extents are disjoint. Preserve the
existing dead-buffer staging reuse; do not forbid reuse across separate phases.

For gathered compact work, max_tiles must cover the actual builder's conservative
`R*8*16` items without truncation, and the owning metadata arena must fit the
counter plus worklist. Large prefill normally takes the dense ABI; do not allocate
larger worklists just to force compact eligibility. Null sorted IDs is permitted
only for explicit route-major M64, never M16. Decode IDs and group metadata must
be tied to the current stream's existing router/sort/builder output ownership.

Host pointer/capacity checks do not validate the contents of device-produced
IDs/offsets/work items. Do not claim otherwise. This partition issues no routing
D2H, sort or worklist rebuild on a launch; future integration must connect the
real producers, unique top-k row contract and same-stream graph refresh. Tests
check host metadata failures and the actual launch ABI, not fake device-content
validation. Existing native fixtures supply separate dynamic metadata coverage.

## Launch behavior and shared-T contract

Decode grids are `(64,R*9,2)`, block32, R1/2/3. Preserve explicit zero writes for
remote routed experts. Shared gate/up remain original transposed layout and
their existing ordered BF16 arithmetic; bind actual private shared-T layer
state created by the successful transpose phase, not arbitrary QuantizedWeight
inputs asserting a layout. Validate the known exact transform geometry,
finite per-tensor scale bits, no per-row scale vector, and checked live spans.
No layout conversion or shared allocation occurs during launch.

Explicit routed-only mode supplies null shared weights, but still requires
valid bounded shared output scratch because the kernel writes zeros there.
This is needed by K4/K5 K2/K3 decomposition. Do not treat null weights as license
to pass null output pointers or overwrite the precomputed shared contribution.

Grouped blocks are128. Dense grid is `(16,ceil(R/64),288)`:17 M tiles at1088,
covering every possible expert under the real unique-top-k contract. Compact
separate uses `(max_tiles,1,1)`; fused uses `(max_tiles,2,1)`. M64 handles the
full1..1088 range, both scale policies and both input layouts. M16 is gathered
only with <=5 rows and first-M-tile work. Keep grouped remote outputs untouched;
existing sparse EP reduction/output preparation is a separate caller contract.

Launch methods only submit checked typed kernels: no GPU allocations, table
readbacks, pointer-table mutation, synchronization, quantization or fallback
repacking. Existing repack remains copy->launch->sync->scale-copy->transpose->
sync with one4MiB workspace; its complete-family integration preserves that
ordering and every error/cleanup invariant.

The grouped live-span list is a fixed eight-element array. This is not a claim
of zero host allocations: the existing KernelLaunch argument builder uses Vec.
No runtime argument-builder rewrite is part of this slice.

## Behavioral RED/GREEN and review gates

Write failing tests against actual missing/incorrect implementations first.
Use real WeightStore input validation and successful repack, actual existing
table builders with their real allocation/upload receipt, actual BufferArena
construction, and byte-storing backend memory for table readback. Do not use
test-only successful leases or fabricated allocation receipts. Keep table
fixtures tiny; backend records weight addresses without materializing matrices.

Cover each16th handle lookup failure/zero, profile/capture refusal before work,
all15 promoted entry ABIs and the existing transpose ABI, backend mismatch,
real builder receipt mutation/no-receipt/slot/alias errors before D2H, all six
table-read failures and pointer/scalar/remote entry mismatches. Assert exact
11,520-byte read total and zero launch-time readbacks/GPU allocations.

Cover scalar word/vector R1/2/3 with active and disabled shared-T, grouped
scalar/vector dense/separate/fused paths, M16 eligibility/fallback, all byte
capacity/overlap boundaries, rows0/1089 rejection and dense17tiles at1088.
Record launch failures without a second kernel/fallback. Test the real paired
launch path rather than a detached policy function. Keep existing transaction
all576 scalar/all1728 operation fault tests and actual teardown tests passing.
Owner lifetime/visibility constraints must be compiler-enforced, not a runtime
boolean named Ready. Add source/compile-fail coverage where practical without
inventing public testing hooks.

Run coordinated focused/full CPU suites, fmt, scoped SPDX/whitespace and
independent source review; freeze exact hashes. CPU recording proves control
flow/ABI/ownership only. Root's actual compiled-PTX presence/arity and native
byte/output/memory gates on3332c36e remain required separately.

Future activation must additionally reject unsupported BF16-input grouped
prefill BEFORE conversion, complete all scalar/bootstrap/K2/K3/C4/K5/drain and
padded-prefill/graph readers, and solve owning publication/legacy invalidation.
No serving/TPS claim or source mutation beyond approved partition is implied.

## Review addendum: shared-T and arena construction authority

Root approved proposing this bounded extension after implementation review found
that private `shared_gate_t` / `shared_up_t` values alone do not retain dimensions,
allocation extents, or backend provenance. Do not infer those facts from a QW.

Add a private optional shared gate/up receipt to `MoeLayer`, initialized to None
in its actual constructor. Mint it only in the extracted
`transpose_unified_shared_gate_up` helper after BOTH existing real transforms
return successfully. Clear it before an attempted repeat transformation. The
receipt records the actual backend, N/K, the four allocated transformed pointer
extents, and the exact returned per-tensor scalar bits. This adds only host
metadata; preserve every legacy allocation/upload/launch/free event and ordering.
Do not rewrite the helper, change transform math, or enable any new caller.
Other transpose paths remain ineligible for active shared B-tile decode.

Receipt minting stays private under the existing shared-phase helper; its opaque
type is visible only as needed for the MoeLayer field. Binding validates the
current actual fields against the receipt, rejects stale pointers/scalars,
wrong backend/dimensions, per-row scales, nonfinite scalars and aliases before
publishing the private construction lease. Active mode without a receipt fails
closed. Tests call the actual shared-only helper (no duplicate routed-T fixture
allocation needed), including successful transforms, failed second transform,
repeat invalidation, wrong geometry/backend, and stale actual-field mutations.

Move the model-wide source-owner scan out of per-launch validation. A private
CheckedArena borrows the actual BufferArena, lease and ForwardContext backend/
configuration authority. At construction, outside actual capture, scan each
relevant actual arena owner against all borrowed WeightStore owners, six routed
table allocations, and any admitted shared-T allocation. Retain only these
fixed-count arena owner spans, not model-wide metadata or GPU payload. Reject
owner aliases before returning it. Per launch, derive subranges within those
already admitted owners and check only the small simultaneously-live span set.
Launches perform no full-model scan, D2H, synchronization or GPU allocation;
actual routing producer and graph-refresh wiring remain activation work.

The existing BufferArena has no backend ownership receipt. The boundary uses
the actual ForwardContext backend identity and its actual BufferArena (the
same invariant used by existing production dispatch), and does not claim that
numeric pointer bounds can independently discover which CUDA context allocated
an arbitrary forged arena. No runtime trait/arena allocation changes are added.

Additional touched files are limited to the MoeLayer receipt field, its None
constructor initializer, the extracted shared-only helper's invalidate/stamp
lines, and a private receipt child/tests. CPU behavioral RED must precede
implementation; independent review and final source freeze still apply.

## Completed behavior and CPU evidence

Decode uses independent checked input, route-ID and output row origins, covering
both K4 shifted-input/base-output and contiguous output decomposition. Routed-only
shared zero writes use two disjoint, phase-dead expert_down_out subranges, not
the precomputed shared outputs; later down work must overwrite before reading.
Binding rejects hybrid state and any existing routed GU T table before D2H;
the separate down/shared T construction remains admissible.

Receipts live under atlas-campaigns/20260908/btile-kernel-family (outside the
repository). Actual behavioral REDs: authority-behavior-red.log (two missing
family/table-authority behaviors), lease-red.log, launch-red.log, grouped-red.log,
shared-arena-behavior-red.log (two missing actual receipt/arena behaviors),
active-shared-red.log, decode-origins-red.log, and mixed-layout-red.log.
Earlier compile/privacy failures and zero-selected filters are not counted as
behavioral REDs. extended-focused-green.log passed31 tests; full-cpu-final.log
passed895 tests before the final review's extra negative coverage.

The final tests additionally cover both positions of separate paired launch
failure, fused/decode backend failure, stale/incomplete actual shared layer
binding before all six readbacks, and hybrid/existing routed T owners. The exact
final receipts are review-full-green.log (897/897 PASS,34.09s),
lib-check-final.log (non-test library check PASS,7.61s), and fmt-final.log (PASS).
Scoped SPDX, <=500-line cap and git diff --check pass. clippy-final.log FAILED
at four unchanged spark-runtime metal-stub too-many-arguments errors before
checking model code; no lint suppression or new-code clippy PASS is claimed.
The final24-file manifest includes22 Rust files and both plans; the resident
activation outline is next-partition planning, not activation implementation.
CPU tests prove
host control flow, allocation authority, byte metadata and typed ABI only; no
numerical GPU validation, serving activation or throughput improvement is claimed.
