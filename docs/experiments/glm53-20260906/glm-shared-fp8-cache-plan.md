# Target-only GLM shared-expert FP8 cache: bounded integration plan

## v20: retain valid oversized-prefill dispatch

Root observed a legitimate 1025-row solo prefill with an arena sized for 1025,
despite the configured chunk setting of 1024. Installed cache currently rejects
this existing path. Do not change scheduler/context bounds or claim FP8 M64 was
validated beyond 1024. Keep cache M64 for rows 1..1024 and dispatch larger valid
arena-bounded rows through the retained original transposed NVFP4 projection.

Before either installed-cache route, validate the projection ordinal (before
bit shifts), exact GU/down geometry, installed cache pointer, arena row limit,
output owner/capacity, and retained T handle/finite scalar/no per-row scales.
Check aligned non-null non-overflowing spans and disjoint output/input/weights.
VERIFY must refuse context or actual stream capture before the fallback, even
after diagnostic bits were checked; the existing outer overlap guard remains.
The large-row fallback calls existing `ops::w4a16_gemm_n128` with its retained
handle, unchanged scalar and stream. No GPU allocation, copy, synchronization,
oracle-pass log or diagnostic-bit update. Non-cache generic dispatch is unchanged.

TDD first: actual `MoeLayer::run_shared_fp8_cache` with a recording GPU backend
must fail on the current 1025-row rejection, then select cache at 1024 and old T
at 1025 for all three projections. Verify complete typed launch arguments,
grid/block/stream, zero extra backend operations, invalid projection/cache/T/
geometry/capacity/alias refusal, and both capture guards even with checked bits.
Independent source review and CPU suites precede root-only native rebuild,
near-cap/cold-prefill quality, both-rank diagnostics and matched clean timing.

The final positive dispatch matrix checks rows 1/4/5/16/63/64/65/148/1024 at the
cached handle and 1025 at the retained T handle, for every projection (30 exact
typed launches). A separate arena-2048 test accepts rows2048 through old T.
Missing candidate handles at eager K5 VERIFY also reject before the reference
launch: the added test first observed that reference launch, then passed after
moving the candidate-handle check before oracle work. The large-row fallback
does not require or launch the unused FP8 handle.

Raw CPU receipts are persistent under
`/home/abc/storage/models/atlas-campaigns/20260908/shared-fp8-fallback/`:
`dispatch-behavior-red.log` (actual1025 rejection), `oracle-handle-red.log`
(reference incorrectly launched before missing-candidate rejection), focused
`dispatch-green.log` and final `full-model-cpu-frozen.log` (824 model tests).
These are metadata/CPU tests, not CUDA numerical or serving-performance claims.
The link-only controller libcuda shim exports no CUDA API and therefore cannot
turn real CUDA calls into successful no-ops; it only satisfies the unused
link-library name for the no-default-feature CPU binary.

## v18 startup correction: validated BF16-derived provenance

The first cache-oracle startup failed safely before serving. Root's complete
checkpoint-header audit (`v18-shared-checkpoint-metadata.log`) shows all126 target
shared matrices are BF16-only:84 GU `[2048,4096]` and42 down `[4096,2048]`, each
16,777,216 bytes, with no quantization markers. The preexisting `quantized_any`
path already quantizes these to NVFP4 and **frees the BF16 source**. Consequently,
the native-checkpoint-pointer assumption below does not describe this checkpoint.

The correction captures explicit typed provenance at that unchanged quantizer
call: first validate BF16 dtype/exact shape/checked extent and absence of packed
or scale markers, then record the returned live NVFP4 packed/scales/scalar and
geometry. Deferred cache construction must match that token to the actual shared
weight. It must never read the freed BF16 pointer or require its stale address to
remain disjoint from later allocations. Original generated NVFP4/T weights and
quantization arithmetic remain unchanged; FP8-source fallback is not added.

TDD gates: a real Mock `quantized_any` call consumes/frees BF16, provenance matches
the exact returned NVFP4 view, changed pointers/scalars/shape and missing origin
reject, malformed BF16 source rejects before quantizer work, and a real cache
install succeeds from that provenance even though checkpoint BF16 is no longer
live. Existing native-checkpoint metadata and all non-cache paths remain strict.
The separately owned deferred-load/reserve correction keeps memory floors; it
does not turn the observed worker reserve rejection into permission to allocate.

Historical v18 status (before the startup correction above): source frozen and
independently approved with CPU gates passed; subsequent startup rejected the
real BF16-derived source and worker memory reserve. Current correction is WIP,
not yet approved or tested on GPU. Root reports standalone full-byte
predecode, three-way full-output, CPU-column, graph and both-FMA memcheck gates
passed. Three clean M5 micro repeats put gate/up around147.6→49.2us and down
75.8→26.6us for existing FP8 M64; these are hot-weight micro timings, not TPS.
First integration uses **only the existing converter and FP8 M64 kernel**.
Optional FP8 M16 is a separate later change, with separate symbols/handles from
the independent shared-NVFP4 M16 experiment.

**Mandatory large-prefill gate completed:** root reports source227828de passed
both FMA modes, full bit equality throughM1024 and both memchecks with0 errors;
receipt `shared-fp8-large-native-gates.log` in phase6. Populating shared FP8
fields also changes prompt and general prefill, so the separate profile compares the
unchanged W4A16 M64 and FP8 M64 kernels at M63/64/65/148/1024, with full outputs,
bytes, graphs, immutability and post-timing checks. Skip the optional M16 kernel
entirely above16. Preserve the original receipt's source identity67a7da3d; this
extension requires a new source commit and new native receipts.

The large profile's exact budget, with max_rows1024 and **two** output buffers,
is34,605,056 bytes for gate/up and38,799,360 bytes for down (eight individually
guarded allocations, sequential fixtures), below64MiB. Its separate compile-time
profile must enforce this cap before allocating. Initial integration eligibility
must limit configured prefill chunks to1024 until larger shapes are validated.
Cold-prefill full-model quality/TTFT remains part of promotion, not only K5 TPS.

## 1. Narrow entry and unchanged paths

- Proposed default-off `ATLAS_GLM_TARGET_SHARED_FP8=1`; separate eager diagnostic
  `ATLAS_GLM_TARGET_SHARED_FP8_VERIFY=1` requires it. Root owns launcher changes,
  both-rank propagation, memory checks, builds and deployment.
- Integrate in `weight_loader/glm5/components.rs::load_moe`, after current
  `transpose_for_prefill_unified_keep_shared`. Pass the actual layer index from
  `load_ffn`; require a target layer index below `num_hidden_layers`, not a dense
  FFN layer, and the existing `allow_prefill_layout=true`. MTP's call in
  `glm5/mtp.rs:88` passes false and uses the appended layer index: neither its
  weights nor proposer dispatch may allocate/use this cache.
- Exact first profile: `glm5_next`, hidden4096, expert/shared intermediate2048,
  target45 layers with42 MoE layers, TP2/EP2 native NVFP4 and unified T layout.
  Require the exact validated dense-layer map `[0,1,2]`, rather than blindly
  subtracting its vector length. Reject unsupported format/layout when enabled.
- Existing `helpers_c.rs::predequant_for_prefill` targets only an optional NVFP4
  dense router and three shared projections. GLM constructs `MoeLayer::new` with
  `gate_nvfp4=None` and a BF16 router, so the router must remain unchanged.
  Reuse its conversion operations and existing `shared_{gate,up,down}_fp8` fields,
  but do not blindly call its current partially-publishing allocation sequence.
- `forward_prefill_phase.rs` already chooses these three FP8 fields ahead of T
  GEMMs. Existing `fp8_gemm_t` uses identical A conversion/K32 MMA/BF16 stores.
  This affects shared projections in generic grouped/prefill execution, not the
  routed experts or routed down. Keep SiLU, shared residual/mHC/EP blending,
  all native original/T weights, exact-M GEMV paths and MTP unchanged.
- The new target-cache branch launches that handle **directly** with the
  existing six-argument ABI, M64 grid and128 threads. The generic
  `ops::fp8_gemm_n128` wrapper defaults to a different LDMAB kernel and allocates
  activation scratch; it is therefore intentionally retained only for preexisting
  non-cache paths. A real Mock launch test first failed on its unwanted scratch
  allocation, then checks direct handle/grid/block/no allocation at both GU/down
  dimensions and M1/4/5/16/63/64/65/148/1024.
- `K5_BATCHED_SHARED=1` still selects its earlier exact-M GEMV branch. For this
  experiment require0 at startup to avoid paying1GiB for an unused K5 cache.
  Its fused-GU flag remains inactive under0; do not conflate the experiments.

## 2. One-time shape, bytes and ownership contract

Introduce a small pure target-cache plan and one loader-only build helper in
dedicated MoE child files (new Rust files remain below500 lines). Before the
first allocation or conversion, validate all three original and T weight views:

- native NVFP4, BF16 input policy; originals `[N,K/2]` and `[N,K/16]`, T copies
  `[K/2,N]`/`[K/16,N]`; GU N2048/K4096, down N4096/K2048; scalar finite scale2
  agrees across original/T views; reject unexpected per-row scale2 semantics;
- cached FP8 stays the existing converter/consumer's row-major `[N,K]`, not
  the separate packed T weight layout;
- nonnull/aligned checked spans, sufficient checkpoint tensor extents validated
  at the loader boundary (a raw `DevicePtr` cannot prove allocation capacity),
  no source/destination overlap and no writable aliases; original/T bytes and
  their ownership are untouched;
- no BF16/FP8 shared alternative already installed, all three FP8 fields absent,
  no NVFP4 router cache to populate, converter and existing FP8 GEMM resolved;
- an explicit model-owned once-at-load state: `Absent → Building → Ready`.
  A repeated request for the same ready plan is a zero-allocation no-op or a
  clear duplicate-initialization error; malformed partial/mismatched state errors.
  No environment lookup, allocation or conversion in decode or graph capture.

Build three temporary output allocations, convert and synchronize, optionally
validate bytes, then publish all three existing fields together. If anything
fails, attempt to free every newly allocated output and leave all fields absent;
do not silently continue on another precision path. Avoid using the current
`QuantizedWeight::predequant_to_fp8` error path unwrapped: it allocates before a
possible launch failure and does not itself return an owner for cleanup.

## 3. Exact reserve and peak-memory accounting

Each projection contains8,388,608 coefficients/FP8 bytes. Three projections are
25,165,824 bytes (24MiB) per layer. Across42 target MoE layers the additional
resident allocation is **1,056,964,608 bytes =1008MiB =0.984375GiB per rank**.
Shared experts are replicated; do not divide by TP2 or EP2. No separate FP8 scale
array is needed because the current scalar scale2 is incorporated in each byte.

Derive these counts with checked arithmetic from validated target layer IDs and
dimensions; reject overflow before loops or allocations. Reserve the whole
remaining cache budget before beginning and recheck physical free memory before
each24MiB layer build. At minimum require remaining-cache bytes plus the existing
4GiB physical safety floor; do not lower any OOM watchdog threshold. Also retain
the existing loader's non-cache allocation/inference reserves: the4GiB condition
alone is not proof that later target/MTP/arena allocations fit.

The maximum extra device footprint along the existing load trajectory is1008MiB:
the converter consumes original resident weights directly and allocates only
the final8MiB output for each projection, with no temporary full weight twin.
Retain original and T copies. Host oracle staging must be chunked and explicitly
budgeted on GB10 unified RAM. Root must check measured **load peak and final
free memory on both ranks**, not infer safety from the previous~11GiB final free.

Factory currently sizes KV after target/MTP loading and `BufferArena::new`
(`factory/build.rs:333,404,473`). Inline cache allocations therefore enter its
actual-free/used-so-far accounting automatically. Do not subtract1008MiB again
from the KV budget after allocating it; a logical reservation is not a second
physical allocation. Log planned, allocated and remaining bytes once/per layer.

Two cache-enabled factory checks additionally preserve the existing reserves:
before target layers, require actual free memory >= full cache + actual
`BufferSizes::total_bytes()` + `inference_reserve`; immediately before arena
allocation, require free >= arena + inference reserve (no second cache debit).
The reserve additions are checked. `BufferSizes::from_config` retains its
existing supported-config arithmetic; this is not a new arbitrary-malformed
configuration validator, nor a proof of every intermediate target/MTP load peak.
The separate authoritative BufferSizes accounting correction includes its
previously omitted `o_latent` and `norm_unit_w` allocations.

## 4. Teardown and failure behavior without a broad ownership rewrite

Use `GpuBackend::alloc`, never managed oversubscription or raw untracked CUDA.
Layer fields are read-only views of new allocations owned for the model/backend
lifetime. Atlas currently has no `MoeLayer::Drop` GPU owner: successful derived
layer weights are tracked by `AtlasCudaBackend::live_allocs` and reclaimed by
`TransformerModel::release_pools`' final `sweep_unreleased`; failed construction
has the backend-drop sweep as a backstop. Preserve that explicit existing
contract in this first slice rather than inventing a destructor holding a dead
GPU reference or freeing original `WeightStore` pointers.

Temporary failed builds must still explicitly attempt cleanup, and repeated init
must never overwrite ready pointers. Test local cleanup and model-lifetime
ledger reclamation; cache pointers must not survive a model reload or be stored
in a process-global pointer-keyed map. A later dedicated `ModelResource` owner is
possible but not required to change the whole loader/teardown architecture here.
On a real CUDA fault/free failure, report fatal setup failure and stop that load;
do not promise recovery by resuming inference after a potentially poisoned context.

## 5. Independent resident-weight diagnostic (eager only)

Verify mode must fail before graph capture/lookup if decode/verify graphs are
enabled, and check actual stream capture state at the oracle entry.
The oracle launch must also explicitly set
`ATLAS_MOE_SHARED_REDUCE_OVERLAP=0`: general prefill can exceed64 rows and use
the auxiliary overlap path even though K5 itself is sequential. Startup and
actual shared-phase entry enforce this. Clean VERIFY=0 runs retain overlap.

1. At load, validate **every cached FP8 byte** for all126 target projections using
   original resident packed/scales and the independently tested CPU E4M3 RNE
   converter. Read one output row (or bounded row chunk) at a time; include
   scale2, both nibbles, sign of zero and saturation. No second device weight,
   no unbounded host snapshot and no original/T mutation. Publish only after
   that layer's three caches pass.
2. On the first actual eager grouped K5 shared call per target MoE layer, validate
   scratch spans before mutation. Execute old T GEMM into the existing output,
   synchronize and save its full5xN BF16 bytes; poison output, execute existing
   FP8 GEMM into the same output, synchronize, require finite/full bit equality.
   Repeat gate, up and shared down at their existing points around unchanged
   SiLU. The down comparison uses the same actual activated BF16 input, not a
   separately recomputed approximation. Full down output is40KiB at K5
   (5x4096x2); two snapshots require80KiB. Gate/up is20KiB per snapshot.
   There is no persistent GPU allocation or extra resident matrix.
3. Any mismatch stops before SiLU/down/blend can consume bad output. No timing
   claims with diagnostics enabled. Both ranks must report all42 layers/126
   projection gates and full-byte cache checks; a missing coverage trace fails
   the gate. Subsequent clean rollout uses the same binary with VERIFY=0.

## 6. TDD and rollout checkpoints

- Pure RED/GREEN: exact shapes/formats/target IDs/MTP exclusion, checked1008MiB
  total, remaining-budget floor boundaries, original/T mismatch/null/alias/short
  spans, partial cache and repeated init. Default off performs no new GPU work.
- Mock execution: the real install entrypoint proves three allocations/launches,
  atomic publication, original/T pointer retention, no router cache, repeat-init
  rejection, default-off/MTP no work, malformed/partial views rejected, and an
  actual VERIFY D2H failure releases its temporary without publication. Sources
  in this fixture are explicitly typed metadata; Mock does not execute CUDA.
  Separately, transaction-level injected
  failure at each allocation, launch, sync, byte check frees all prior outputs,
  with cleanup attempted even after another free fails. These are generic
  failure-closure tests, not claims of real CUDA fault injection. Byte-oracle
  fixtures use actual host-backed Mock source bytes and assert immutability.
- Real-path output-oracle Mock seam: each GU/down stage compares all BF16 bytes,
  rejects nonfinite/poison/mismatch before downstream calls, and graph-mode
  rejection occurs before any copy/kernel. Keep no-op tests for BF16 router,
  routed weights/down, MTP and unsupported/default-off models.
- Root gates: CPU suites/review → both-rank load/byte oracle → eager short coding
  and arithmetic with all resident output checks → diagnostics-off same-binary
  A/B versus cache0 using matched EH/M16/MTP flags, exact token hashes/quality,
  all-cap output and acceptance accounting. Recheck near-cap, cancellation and
  reused slots before deployment. Cache0 rollback retains existing original/T
  paths and allocates no new FP8 cache. No new physical cache until authorized.

Cold-model caveat: logical shared-weight traffic rises16/9x (567→1008MiB per
K5 pass across42 layers). Removing repeated conversion and one K32-stage barrier
is promising, but only a matched full-model A/B establishes throughput benefit.

### CPU/source receipts (phase6)

- `shared-fp8-cache-red.log`: initial pure policy/reserve RED,1 pass/2 failures;
  initial policy/transaction GREEN5/5 in `shared-fp8-cache-green.log`.
- `shared-fp8-cache-bytes-red.log`: independent full-byte oracle RED3 failures;
  subsequent byte/cache GREEN in `shared-fp8-cache-bytes-green.log`.
- `shared-fp8-cache-dispatch-red.log`: real generic-wrapper regression RED1
  failure (unexpected activation allocation), before direct M64 correction.
- `shared-fp8-cache-final-focused.log`: all22 cache tests passed, including the
  real loader publish/error seam and both GU/down direct launch geometry.
- `shared-fp8-cache-full-model-cpu.log`: all806 no-default-feature model library
  tests passed. This is CPU validation, not a resident CUDA numerical claim.
- Independent source reviews approved byte conversion, transactional ownership,
  reserve integration and corrected direct M64 candidate dispatch. Root retains
  native load/oracle, memory-monitoring, matched throughput and quality gates.
