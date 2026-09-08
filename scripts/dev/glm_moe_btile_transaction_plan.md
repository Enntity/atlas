# B-tile native provenance and unpublished transaction: bounded production seam

Status: root approved the plan; bounded implementation and CPU gates completed.
No model loader activation or native byte-kernel registration. Root owns commits,
all native builds/nodes/GPU operations and eventual deployment. Controller-only
Cargo may run only after coordination with the current CPU-test owner.

## Outcome and hard boundary

Implement actual production Rust byte-copy/launch/cleanup operations with a
checked native-checkpoint input and a sealed **Unpublished** result. This is a
construction prerequisite, not an enabled model layout or serving speedup.
No flag, loader call, Ready capability, forward selector, pointer-table
publication, shared/down extraction, or existing legacy dispatch change.
No model weights may be repacked by this partition in a serving workflow.

The existing integration plan remains the cross-reader contract. Its historical
129/1024-row and native-permutation warnings have since been addressed by the
1088-row and native-source standalone gates; this does not port production
readers. Scalar BF161/2/3, all prequant M64 ABIs, decompositions/drains and
incompatible-path rejection still block eventual activation.

The original-T register attempt is rejected for promotion; see
`glm_moe_m16_direct_register_results.md`. This partition prepares the existing
B-tile approach without inferring a model TPS benefit from its microbenchmarks.

## Exact source ownership

New production modules under `crates/spark-model/src/layers/moe/`:

- `gate_up_native_source.rs`: sealed store-origin validation and checked spans.
- `gate_up_native_source_tests.rs`: actual WeightStore fixtures and rejection.
- `gate_up_repack.rs`: private input/result/workspace and actual transaction.
- `gate_up_repack_tests.rs`: actual transaction/ordering/fault tests.
- `gate_up_repack_test_gpu.rs`: typed-ABI/operation recording and fault backend.

New `crates/spark-model/src/layers/ops/moe_gate_up_repack.rs`: checked packed
permutation launch; use the existing `ops::transpose_u8` for native scales.
Only declarations in `layers/moe/mod.rs` and `layers/ops.rs` accompany these.
Split a new test module further if needed; every Rust file remains <=500 lines.
`gate_up_native_source.rs` is a private child of the transaction module, not a
MoE sibling. Its raw projection accessor is visible only inside that transaction.
The input and Unpublished descriptors borrow the actual WeightStore/backend;
the store cannot be mutably released while those descriptors are live.

No `MoeLayer` storage field or constructor changes in this slice. No edits to
`components.rs`, `helpers_a/b.rs`, `QuantizedWeight`, WeightStore or its release
implementation. Shared/down extraction is a separate later reviewed partition.
No CUDA registration in this slice: the transaction receives checked handles
for the already native-tested byte permutation ABI and existing transpose ABI.
It has no serving caller. Later registration must gate the real production CUDA
helper against the standalone full-byte oracle; these CPU tests cannot supply
that numerical proof or pretend a missing export is usable.

## Sealed provenance before any quantized loader

The input constructor reads actual `WeightStore` metadata, not a caller-provided
`QuantizedWeight` or an assertion that quantization already happened. Its only
admitted profile is target GLM geometry N2048/K4096, group16, 288 experts,
TP=EP=world2, rank0/1, target layer (not MTP/shared/down), with an explicit
rank-local expert map. Require both gate/up sources for every local expert;
do not infer locality from whichever projection happens to be non-null.
The initial source-format contract is Standard NVFP4 only, with the target
`model.layers` prefix (empty/default or `model` weight-prefix configuration),
no adapter, and no compressed/global-scale or E8M0 marker coexistence.

For each local gate/up projection require exact names under the selected target
layer's routed expert prefix, UInt8 packed `[2048,2048]`, FP8E4M3 scales
`[2048,256]`, scalar FP32 `weight_scale_2` and, if present, scalar FP32
`input_scale`. Preserve absent input-scale as absent. Reject FP8/BF16 weights,
E8M0/alternate scale markers, per-row scalars, malformed shape/overflow, missing
local mates, invalid rank/locality, or an MTP/shared projection prefix.

All packed/scales spans must be non-null, 16-byte aligned, exact-sized and
non-overflowing, disjoint across all local gate/up allocations. Scalar/input
metadata spans are checked and excluded from every destination/scratch span.
Validate the whole layer's metadata and spans before scalar D2H reads, then
read only those scalar bytes on the explicit stream and require finite values.
Preserve their exact bit patterns; do not normalize signs or rescale anything.
No full weight D2H, quantization, freeing or permutation during preflight.
Reject active capture before any allocation/copy/synchronization.

Private fields prevent construction from arbitrary raw pointers outside the
module. Input and result are not Clone/Copy and expose no raw native/transposed
table getter. Tests call the real constructor, never a test-only bypass.

## Transaction and ownership

The transaction is construction-only and receives the checked input, explicit
stream, two nonzero kernel handles, and a reusable workspace. It validates
handles and the full workspace/source disjointness before its first write.
The workspace allocates **exactly 4,194,304 device bytes** once; it is reusable
across sequential layer transactions, not one allocation per projection.
No device pointer tables, scale twins, or other device buffers are allocated.
Host bookkeeping is O(local projections), with explicit checked capacities;
no host copy of a complete matrix/layer/model. Scalar reads are four bytes.
Workspace disjointness scans every actual owner in the borrowed WeightStore,
including foreign/shared/down/MTP tensors, before its first copy. A faulty
allocator returning an existing owner is rejected without freeing that owner;
the workspace is poisoned and construction must be abandoned. This is not a
successful scratch cleanup or a numerical simulation of the broken allocator.

For each local expert in ascending order, gate then up:

1. `copy_d2d_async(original_packed, scratch, 4194304, stream)`.
2. Actual checked packed op launches `(scratch, original_packed, 2048, 4096)`
   with grid `[16384,1,1]`, block `[256,1,1]`, zero shared bytes, same stream.
   Destination tile `(nt,kt,n,b)` reads source
   `(nt*128+n)*2048 + kt*32+b`; no nibble/float arithmetic.
3. Synchronize before scratch reuse.
4. `copy_d2d_async(original_scales, scratch, 524288, stream)`.
5. Actual `ops::transpose_u8` launches `(scratch, original_scales, 2048, 256)`
   with grid `[8,64,1]`, block `[32,8,1]`, zero shared bytes, same stream.
6. Synchronize before the next projection or result construction.

Addresses and metadata bits remain identical. Original allocations remain in
their existing WeightStore/backend ledger; never free or re-register originals,
interior offsets or scalar pointers. Only the workspace owns/frees scratch.
There is no allocation, D2H, table building or forwarding during these steps.

Success returns a sealed `UnpublishedBTileLayer`, with no conversion into
`QuantizedWeight`, `ExpertPtrTable`, a Legacy layer or a dispatchable Ready
object. Failure returns no success descriptor, never resumes later projections
and never selects an old-layout kernel. Track whether writes may have started
so errors explicitly require abandoning construction, not a native fallback.
Attempt synchronization/cleanup as possible; preserve primary errors and
cleanup context. The actual CUDA backend removes its ledger entry before the
driver free; a failed free must retain explicit pointer/error diagnostics and
may require CUDA context/process teardown, not a promised later ledger sweep.
Do not retry a possibly poisoned
workspace in a later layer.

This type is not a magical revocation of existing raw DevicePtr aliases.
Because there is no loader caller in this partition, no existing MoeLayer can
be transitioned through it. Future activation must hold exclusive construction
ownership, withhold/invalidate native pointer tables AND routed expert views,
and abort model construction on any post-write error. Publication additionally
requires every reachable reader capability. Those are mandatory later gates,
not claims made by the Unpublished name.

## Actual production CPU TDD

Start with behavioral RED from missing/incorrect production validation or
operation ordering, then implement GREEN; preserve raw receipts persistently.
Use `WeightStore::from_map` fixtures and a recording `GpuBackend` modeled after
the existing shared-cache actual-dispatch tests. The backend decodes real
`KernelArg` values; do not implement a separate selector/oracle just for tests.

Positive tests invoke the actual constructor, workspace and transaction with
the complete 144-local/144-remote rank map and both projections, checking
representative nonzero ordinals and both rank partitions. Recording fixtures
hold metadata, not host copies of those device matrices. Check:

- Correct source-to-scratch copies before writes, exact source/destination and
  sizes, both handles/ABIs/grid/block/shared/stream, every synchronization,
  gate/up order and remote skips. No native/transposed GEMM launch at all.
- One 4MiB allocation, constant scratch address across projections/layers,
  zero table allocations/transaction D2H, exact original owner addresses and
  scalar/input bits, no original free, exactly one successful scratch free.
- Unpublished result only after the final successful synchronization; no raw
  forward-capability accessor. All legacy APIs remain untouched by source diff.

Negative tests vary each dtype/shape/marker, local map, null/alignment/overflow,
cross-expert/projection/scalar alias, unknown/missing source, wrong topology,
invalid handles, scratch alias/capacity and capture. Reject before mutation;
structural preflight faults reject before any scalar read. Inject failure at
each scalar read, allocation, D2D copy, packed launch, transpose launch,
synchronization and cleanup position, including after an earlier expert has
converted. Assert no later operation/publication, no legacy fallback, no
original free and explicit poisoned-state/error behavior.

Recording tests prove production dispatch/ownership/control flow, NOT numerical
GPU permutation correctness. Do not label simulated bytes as a native oracle.
Existing standalone native byte gates remain separate evidence. Root must
re-run native gates against final promoted CUDA helpers before activation.

## Review and stop boundary

Root reads this complete plan before code. Independent review checks the actual
provenance boundary, typed ABI, byte budget, cleanup/alias closure and the fact
that no model-loading path can call the transaction. Coordinate controller
Cargo with the current owner; use persistent receipts and the approved CPU-only
link environment. After RED/GREEN, relevant suite/fmt/SPDX and review, freeze
hashes. Root alone commits or authorizes any next partition.

## CPU receipts and measured scope

Persistent directory:
`/home/abc/storage/models/atlas-campaigns/20260908/btile-transaction/`.

- `initial-red.log` and `compile-privacy-red.log` are compiler failures while
  building the new fixture, **not behavioral RED**.
- `behavioral-red.log`: assertions execute, 3 pass/10 fail at the actual missing
  provenance constructor. `transaction-red.log`: 8 pass/5 fail after provenance
  implementation, at the actual missing transaction.
- `foreign-alias-red.log`: actual shared-owner scratch-alias test executes and
  fails before the all-WeightStore disjointness fix.
- `focused-final.log`: 14/14 including the foreign-alias fix, all 576 scalar
  read-failure positions and all 1,728 copy/launch/sync-failure positions.
- `full-suite.log`: 856/856 after the fifteenth new test invokes the actual
  WeightStore release, verifies all original owners persist through conversion
  and scratch cleanup, then verifies exactly-once/idempotent original teardown.
- `lib-check.log`: successful non-test, no-default-features library check.
- `fmt.log`: initial declaration-order formatting failure, fixed without
  changing behavior. `fmt-final.log`: complete workspace formatting check passes.

These are CPU production-control-flow/typed-ABI/ownership tests, not CUDA
execution or numerical byte-permutation oracles. The final suite is rerun after
the declaration-only formatting fix as `final-full-suite.log`. Scoped source
SPDX and whitespace checks accompany freeze; no Docker license checker or GPU
tool was run by this agent. Staged dead-code allowances are explicitly explained
in the private modules because this partition deliberately has no loader caller.
