# GLM checked checkpoint retirement for resident B-tile ownership

Status: root read the complete plan and approved bounded implementation. The
stream fix is committed as d3f989c7 and the retirement author now owns Cargo.
Use existing WeightStore names/get/from_map APIs, not a runtime allocator/API
change. Tracking is an explicit GLM construction context; the default legacy
loader remains untracked with identical GPU events. No activation.

## Confirmed defect in the ownership assumption

WeightStore is adopted by the model, but its map is not necessarily a map of
live allocations after the current GLM loader. `helpers_unified_phases.rs`
frees native routed down packed/scales after the transformed table is built.
The original entries remain in WeightStore. A new valid allocation can reuse
one of those addresses, making an unfiltered whole-store alias scan falsely
reject it. `WeightStore::release` later drains all entries and calls gpu.free.
CudaBackend::free removes ledger state and always attempts cuMemFree; absence
from the ledger does not suppress the driver call. Do not rely on such a skip.

WeightStore and WeightTensor have no implicit GPU-freeing Drop implementation.
GPU release is explicit through ModelResource. Replacing only their host map
using existing from_map is therefore possible without a second device owner,
but retirement must be identified at the actual free, not inferred afterward
from a recycled numerical address or a tensor name pattern.

## Exact GLM source audit

Only the actual returned source matching a still-live sealed checkpoint entry
is eligible. `dense_auto` returns the store pointer for BF16, but returns a new
BF16 allocation for FP32/FP8/UInt8 conversions; freeing the latter MUST NOT
retire the original checkpoint key. Capture source origin before conversion.

| Actual site | Potential checkpoint retirement |
|---|---|
| glm5/layers.rs load_mla_layer load_tp, near122 | q_b_proj, kv_b_proj, o_proj original source when TP sharding changes its address |
| glm5/components.rs load_kda_weights load_tp_dense, near289 | Original source of every sharded dense projection/conv below, only when it was the checkpoint allocation |
| components load_hot, near322 | q/k/v/o local BF16 after quantization; this is normally a derived TP shard at TP2, never blindly retire the original name |
| components convolution pack, near346 | Local q_conv1d/k_conv1d/v_conv1d after copy; normally a derived shard, not the already retired full checkpoint allocation |
| components A_log/dt_bias, near351/356 | Full vector when actual TP sharding changes address and dense_keep_f32 returned the checkpoint allocation |
| helpers_unified_down_phase | Local routed down_proj.weight and weight_scale, exact Standard NVFP4 shapes [4096,1024] UInt8 / [4096,128] FP8E4M3; scalar/input-scale checkpoint entries remain live |

The KDA source list is q/k/v/o, b_proj, f_b_proj, g_b_proj and q/k/v_conv1d;
f_a_proj/g_a_proj/o_norm are not freed at these sites. MLA q_a, kv_a and indexer
sources are retained. The appended MTP calls the same MLA helper with replicated
TP1 geometry, so no TP-shard retirement should be invented for it.

Audited non-retirements: dense_ffn quantization retains its BF16 inputs;
quantize_to_nvfp4/fp8 do not free their input; dense_auto conversions retain the
checkpoint source; HC widening retains its input; GLM's BF16 target head and
optional draft-only head quantization retain the target head; shared FP8 cache
retains original shared NVFP4/T weights and frees only new failed cache outputs.
QuantizedWeight's concatenation transpose frees its temporary transpose buffers,
not checkpoint matrices. No temporary or derived allocation gets a checkpoint
retirement record just because its address later equals an old source address.

The B-tile branch never frees native GU. It retains native shared gate/up/down
for exact-M readers. Existing legacy unified GU frees are bypassed, not guessed
or silently included. MMQ/CUTLASS/hybrid/unsupported quantization remain rejected
before conversion. Other-model retirement is explicitly out of scope.

## Proposed GLM-private authority and API

Use one explicitly threaded GLM construction retirement log, not a global,
thread-local, raw-address blacklist or new GpuBackend hook. Fields/constructors
of source-origin and retirement receipts are private to the construction code.
An origin records exact key, pointer, dtype, complete shape/checked extent and
live generation. Validate no duplicate/overlapping store owner before freeing
that checkpoint allocation. Returned derived allocations carry no checkpoint
origin even if their address equals a previously retired source.

At each audited actual free, preserve event ordering. Mark the attempt before
the backend call; on success record Retired, on error record AttemptedUnknown
and fail construction. Neither status authorizes a second free. Packed success
followed by scale failure is Failed construction, never published Ready. The
error retains the exact name/address and reports any required CUDA context
teardown; do not claim a failed free is recoverable by a ledger sweep.

The existing public legacy shared/down converter keeps its event sequence.
The B-tile internal shared/down phase is accessible only with the session's
unforgeable construction token, bound to the actual layer/backend. Thread an
explicit free observer/owner through the narrow internal release step so every
attempt is recorded at the call, not reconstructed after the phase returns.
Do not add a generic GPU proxy or replicate the transpose mathematics.

Until map replacement, source validation uses a private live-checkpoint view:
actual WeightStore plus the immutable successful-retirement receipts already
produced by this same construction. It excludes by exact key+sealed identity,
never address alone. The ordinary from_store path has no exclusions. Loading
an already retired source is an error, not a use of dangling metadata.

At the owned factory seam after target/MTP/head construction, validate every
receipt against its exact current entry, create the replacement map from the
remaining exact tensors, then replace/drop the old host metadata. Only this
single replacement store is adopted by the model. No GPU free/copy/allocation
occurs during replacement. Do not retain both maps or a borrowed serving store.
Failed construction must not adopt a model; cleanup removes ownership for exact
attempted frees before any explicit store release, never retries unknown frees,
and releases only still-owned entries once. Failure details remain visible.

## Borrow and publication order

1. Validate target native GU/down origins, complete family, flags and local map
   before mutation. Future partition2 still prevalidates all42 target plans.
2. Enter Constructing (all public forward/native converters refuse). Repack GU
   using the existing4MiB workspace and immutable live-checkpoint view.
3. Run existing shared-GU phase; validate actual six GU tables/shared receipt,
   seal owned GU metadata/table authority, revoke native GU aliases, and drop
   every source/lease/store/log borrow. Do not publish a dispatchable Ready yet.
4. Run actual down transform/release with explicit construction-token ownership
   and per-allocation retirement recording; validate actual down table/shared
   state and all required down/shared handles. Any error leaves Failed.
5. Complete publication of a non-borrowing, arena-unbound resident owner once.
   Next layer's live view sees the actual recorded retirements.
6. At factory seam rebuild the single store, then bind each actual arena owner
   against the live store and resident allocations. Missing arena binding
   rejects all readers. Move store into model; all forwards use caller stream.
7. Actual model release first nulls arena owners, then releases live store
   entries, then existing backend sweep handles transformed/table allocations.
   Post-release reader validation refuses before any GPU operation.

Owned files would be GLM-local retirement children and the exact GLM loader
free sites above, private B-tile session/source-view/release plumbing, plus the
minimal owned factory-map handoff. No runtime allocator/free/trait rewrite.
Because the generic loader returns through an immutable WeightStore borrow,
GLM's tracked construction must return its private log explicitly to the
factory (or an equivalent GLM-local result), not mutate a shared global map.
This is a separate approval/commit boundary from mathematical reader promotion.

## Actual behavioral TDD and gates

- RED actual down packed/scales free followed by actual store release attempts
  the same original again; GREEN retires exact entries, GU remains registered
  and frees once, repeat release is idempotent.
- RED actual alias scan rejects an arena allocation deliberately reusing a
  successfully retired down address; GREEN accepts that owner but rejects a
  reused address under a different still-live key or changed receipt identity.
- Actual BF16 pass-through/sharding retires its exact source; FP32/FP8/UInt8
  derived dense paths keep original checkpoint entries. Derived shard/conv
  frees never retire a checkpoint twice. Include rank0/rank1 and TP1 MTP.
- Every free fault: success-packed/fail-scale, fail-first, and later cleanup
  errors leave Failed with no forward work, no double-free and exact diagnostics.
- Changed key/pointer/dtype/shape, duplicate receipt, foreign backend, overlap
  with GU/shared/scalar/other checkpoint owners refuse before retirement/free.
- Record actual legacy allocation/upload/launch/free traces unchanged; no extra
  GPU down copy, no model-sized GPU allocation, same4MiB repack workspace.
- Actual factory adoption/model release fixture proves the single live store
  handoff and post-release refusal. No claim that a virtual pointer-only fixture
  proves native CUDA lifetime correctness.

Independent review, focused/full CPU tests, no-default check, format/SPDX/caps,
exact source freeze, then root native gates. No activation until the actual
retirement audit and teardown proof are complete; no throughput claim here.

## Reviewed implementation refinements

Root approved one sorted immutable interval index, bound by an actual immutable
WeightStore borrow throughout construction. Validate all initial extents and
overlaps once; an exact-key retirement only removes live owners. Derived-span
checks use binary interval lookup rather than rescanning and cloning every
checkpoint entry for each of the roughly12,096 routed-down frees per rank.
Consuming `finish` ends the borrow and yields a sealed ownership receipt. Before
map replacement the receipt revalidates the complete original key/identity set.

The replacement must precede TransformerModel::new, which consumes the backend
Box. If that constructor fails, the existing backend Drop/context teardown is
the available failure closure; a caller cannot invoke explicit cleanup through
the already-moved backend. Before replacement, explicit cleanup uses the sealed
attempt list and preserves unknown-free errors. After replacement, cleanup uses
the already-filtered store directly (or actual model teardown after adoption),
never retries the consumed construction receipt.

The existing full legacy load_all entry remains explicitly untracked; it cannot
accept a retirement context while its legacy FFN branch may free native GU.
The actual MLA/KDA source helpers and construction-only down phase accept the
sealed context and are exercised directly by CPU fixtures. The next loader
partition must thread one context through those actual helpers AND its approved
B-tile FFN construction before enabling a tracked whole-model load. No new
factory selection, broad ModelWeightLoader trait or fake None-only factory hook
is added here. Private unused construction entry points carry the same explicit
staging-only dead-code treatment as the already committed unpublished repack
family; test builds retain normal dead-code checking.

## Frozen CPU evidence

Receipts: `atlas-campaigns/20260908/btile-checkpoint-retirement/`.

- `core-red.log`: two actual executed failures at the deliberately absent
  checkpoint retirement implementation.
- `dense-red-behavior.log`: three core tests pass, actual converted/native
  source release fails at its deliberate stub. Earlier compiler-only logs
  are not behavioral RED evidence.
- `down-red.log`: actual down transforms/frees succeed, but the store wrongly
  retains12 instead of8 checkpoint owners. The fixed test distinguishes the
  original checkpoint frees from six legitimate temporary frees.
- `frozen-full-green.log`:935/935 model CPU tests pass in34.63s. The new
  cases exercise real MLA/KDA source sites, TP1 and both TP2 ranks, all19 KDA
  native/derived free faults, every local down free fault, secondary cleanup
  failure, exact identity/alias refusal, actual recycled allocations, live
  NativeGateUpLayer validation, and actual TransformerModel adoption/teardown.
- `frozen-lib-check.log`: non-test library check passes in4.00s.
- `fmt-final.log`, owned SPDX/caps and `git diff --check`: pass. Every new
  Rust file is below500 lines. A test-only duplicate mutex lock was corrected
  before the terminal full pass; its interrupted earlier run is not GREEN.
- `clippy-final.log`: fails at the same four existing spark-runtime Metal
  stub argument-count errors before model linting. No Clippy PASS is claimed.

The actual default KDA free trace is unchanged. No GPU allocation, free or
copy is added by ownership receipts or map replacement; existing transform
allocations/free order remain unchanged. These are CPU/source ownership
proofs, not numerical CUDA validation or a resident activation claim. Native
rollout still requires the next reader/loader closure and root-only gates.
