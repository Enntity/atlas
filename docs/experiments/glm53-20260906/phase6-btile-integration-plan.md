# Phase6: typed equal-memory routed gate/up B-tile integration

Status: implementation plan only. No production edits are authorized by this
document. Root owns native builds, node operations, model stops, GPU gates,
measurements, commits, and eventual deployment decisions. The standalone
decode/M64 compatibility prototypes are not yet production capabilities.

## Evidence and decision boundary

### September8 loader audit (not an enabled implementation)

The existing unified helper allocates routed transposed gate/up and then frees
their native allocations before proceeding to down. Therefore conversion cannot
be appended after that helper, nor can `keep_originals` substitute for a bounded
transaction: a duplicate gate/up layout costs about53.16GiB per rank across42
layers. Extract shared/down phases explicitly and bypass both routed transpose
and routed-native frees when the new storage is eventually selected.

Publication must invalidate not only native/transposed pointer tables but also
the routed `weights.experts[*].gate_proj/up_proj` native views. Those remaining
views otherwise invite incompatible converters or frees. Preserve their real
allocation owner while transferring the checked spans into the private pair.
Validate native checkpoint dtype/shape/scale markers before `quantized_any`;
do not infer checkpoint-native ownership from a returned quantized weight.

The scripts now cover all required prequantized M64 ABIs through1024 with four
native ordinary/sanitizer passes (`40c4bf5c`). The further1088-row envelope is
committed as `ba37e742`, CPU-tested/reviewed but awaiting its own native gates.
Use actual arena rows for production eligibility, not just configured chunks.
Neither prototype validates production's native-source permutation: that
requires a separate full-byte oracle before resident conversion is enabled.

The next implementation partition is the private storage/handle contract and
bounded repack transaction, plus an unchanged-legacy extraction of shared/down
phases. Complete all reader selectors before exposing an enabled loader path.
This partition alone does not provide a speedup and cannot authorize repacking.

The original M16 B-tile standalone fixture passed C4/K5 full-output numerical
and memcheck gates. Root's three paired runs reported all15 useful comparisons
per width winning versus the current M16 baseline: C4 direct median1.299195x
(range1.213549–1.341430), builder-inclusive median1.303584x; K5 direct median
1.727232x (range1.523586–1.786981), inclusive median1.414485x. These are a
two-local-pair hot-working-set result, not a model-wide speed prediction.
The separately authored BF16-input N1/N2/N3 and prequant M64 compatibility
fixtures still require root-owned native gates. Root is measuring actual K5
gate/up time to establish the attainable whole-cycle ceiling.

There will be no model-wide duplicate weights, no activation-precision change,
and no old-layout fallback after conversion. A failed/incomplete conversion
aborts loading. The initial full-model A/B is a restart-time storage choice,
not a per-request or per-layer tuning switch. Default behavior stays unchanged.

## Minimal supported profile

Target GLM `glm5_next`, native NVFP4 routed experts, N2048/K4096,288 experts,
top8 sigmoid routing, TP=EP=world2, no LoRA, no swap/offload. Shared expert
remains in its existing native and transposed layouts; down remains in its
existing unified transposed layout. The appended MTP proposer is explicitly
excluded: `weight_loader/glm5/mtp.rs` constructs TP/EP1 and calls `load_ffn`
with `allow_prefill_layout=false`. Dense FFNs and attention weights are excluded.

For the first model gate choose the existing short-context C1+MTP4 recipe;
scalar bootstrap/accepted-history repair and general chunked prefill remain
mandatory even in that narrow recipe. C4 and mixed-width drains can be added
only after the complete compatibility matrix below passes. No assumption that
an operator's preferred grouped flag makes other consumers unreachable.

Proposed operator selection, subject to root approval:
`ATLAS_GLM_MOE_GATE_UP_LAYOUT=transposed|btile_n128k64_v1`. Absence keeps the
existing storage policy. Resolve once; reject unknown strings. Do not add a
global quantization variant: both layouts still contain identical NVFP4 codes.
Disable the old `ATLAS_GLM_MOE_GATE_UP_M16_VERIFY` diagnostic for B-tile because
its reference launch assumes transposed bytes. A separately bounded B-tile
oracle described below is needed instead.

## Storage type and invariant

Add a MoE-local `gate_up_layout.rs`, not a field on every `QuantizedWeight`.
The pair must have one authoritative storage discriminant. Recommended shape:

```text
RoutedGateUpStorage
  Legacy                     existing native/transposed/hybrid fields, unchanged
  BTileN128K64V1(BTilePair)    private checked gate/up tables + projection spans

BTilePair
  N=2048, K=4096, experts=288, group=16
  packed layout=[N/128,K/64,128,32], even-K nibble low
  scale layout=[K/16,N], E4M3, existing FP32 tensor scale2
  rank-local ownership map and nonaliasing allocation spans
  validated complete kernel family; immutable after publication
```

This is intentionally confined to MoE, not a cross-model weight type rewrite.
In the B-tile variant move the native gate/up pointer tables into `BTilePair`
and replace the legacy native tables with explicit null/unavailable tables;
leave `gate_ptrs_t/up_ptrs_t` absent. Never store B-tile tables in either `_t`
field. Make the typed accessor the only way B-tile pointers reach an op.
Legacy helper access must assert `Legacy` before any GPU launch, not merely
hope a null pointer makes an old kernel fail. Tests enumerate all entries.

Separate the question “has compatible routed prefill/decode storage” from
“is transposed.” Existing `use_t_layout_for_{decode,prefill}` must retain their
literal old-layout meaning. Add layout-aware dispatch-plan helpers for paths
that accept either variant; do not blindly broaden those predicates and then
continue returning raw `_t` tables. Required handles are validated before
conversion and kernel choice is bound to the storage variant thereafter.

## Equal-memory loader transaction

Prefer bounded **in-place-at-load replacement**, avoiding new slab ownership
and avoiding freeing checkpoint-owned gate/up pointers:

1. Validate policy, all local shapes/formats/scales/ownership, source pointer
   alignment/spans, and every compatibility handle before mutating any bytes.
   Reject source aliasing across projections or experts. The appended MTP
   layer and shared weights must not be in this ownership set.
2. Allocate one reusable4MiB temporary device buffer for this rank's GLM
   layer-loading pass, plus only small bookkeeping. Charge it in startup
   reserve. Its owner outlives all repack launches and frees it on every exit.
3. For each local expert, each gate/up projection: copy the native packed
   `[N,K/2]` allocation into scratch, then launch a native-to-B-tile byte
   permutation from scratch back into the **same original allocation**.
   For destination `(nt,kt,n,b)`, source is
   `(nt*128+n)*(K/2)+kt*32+b`. This is not the prototype's transposed-source
   mapping. No nibble unpacking, rescaling, FP conversion, or quantization.
4. Reuse the same scratch for the0.5MiB scale allocation: copy native
   `[N,K/16]` bytes, transpose back to `[K/16,N]` in the original scale
   allocation. Preserve scale2/input-scale metadata; reject unsupported
   per-row scale2 before this point.
5. Synchronize before scratch reuse/publication and any source verification.
   After all routed gate/up projections pass, publish `BTilePair` atomically
   at the host boundary. Build/retain only its checked pointer tables. Their
   addresses and the weight addresses stay fixed for all graphs.
6. Perform existing shared transpose and routed-down unified conversion,
   extracting/reusing the relevant phases of `helpers_a.rs`; do not call its
   current whole gate/up transpose on already tiled bytes. Preserve current
   shared native copies. No changes to down arithmetic or ownership policy.

Packed bytes remain4MiB and scales0.5MiB per projection/expert. For144 local
experts, gate+up remain1296MiB per layer, exactly as before. Transient additional
device bytes for this conversion are4MiB plus explicitly counted metadata,
not1296MiB. Existing down/shared conversion peaks remain separately budgeted.
GB10 host snapshots also consume system memory: diagnostic copies are bounded
to one projection, never one complete layer/model.

`WeightStore` remains the owner of those original gate/up allocations; only
their contents change. Do not free or register an offset as an allocation.
The backend's allocation ledger and `WeightStore::release` therefore retain
their existing addresses. The model must never export those mutated tensors
as checkpoint-native data. Conversion failure after the first write cannot
roll back to a partially native model: terminate construction, synchronize as
possible, and use existing model teardown. A diagnostic inverse conversion
does not become an automatic recovery path.

## All consumers and precision contract

| Entry/files under `crates/spark-model/src/layers/moe` | B-tile action |
|---|---|
| `forward.rs`, `forward_phase.rs` | N1 BF16-input compatible gate/up; original down/shared blend unchanged. Includes scalar bootstrap, repair and C4 scalar control. |
| `forward_k2/unified_t.rs`, `forward_k2.rs` | N2 BF16-input compatible gate/up; preserve routed-only/shared-disabled modes and EP reduction. |
| `forward_k3.rs` | Grouping-off N3 compatibility; grouping-on uses the layout-aware prequant route. Never change router arithmetic. |
| `forward_batched.rs` | Per-token fallback selects compatible N1 on each independent input; no MMQ-style redirect that changes activation arithmetic. |
| `forward_k4.rs`, `forward_k5.rs` | Their K2/K3 decompositions, grouped selection and generic fallback remain layout-safe; shared exact-M4/M5 and deferred mHC blend unchanged. |
| `forward_c4.rs` | Both reverse scalar control and grouped arm supported by typed dispatch. No changes to collective count or row copying. |
| `prequant_fp4.rs`, `forward_prefill_routed.rs` | Eligible compact N<=5 uses validated M16 B-tile; all other rows/worklists use compatible M64. Preserve existing A quantizer, scale handling and K64 accumulation. |
| `forward_prefill.rs`, `forward_prefill_phase.rs` | Preserve sorting, short-prefill and shared handling; assert supported storage before alternative paths. |
| `helpers_a/b/c.rs`, `mmq_layout.rs`, `forward_token_major.rs`, `forward_atomic_c4.rs`, FP8/BF16 prefill modules | Reject incompatible conversions/alternate expert families before mutation or launch. They must never reinterpret B-tile as native/transposed. |
| `dump.rs`, pointer-table diagnostics | Explicit layout in logs; metadata-only diagnostics need no byte conversion. No implicit native weight export. |

Scalar/batch2/batch3 compatibility must preserve BF16 A, FP32 group loop,
`acc += a_lo*w_lo + a_hi*w_hi`, group16 scale decode and FMA build policy.
The standalone direct accessor may regress coalescing; the cooperative stage
must earn selection through measured scalar gates. Do not replace these paths
with FP4 A just to reuse M16. Shared B stays transposed inside the fused kernels.

M64 production compatibility must remove the standalone129-row ceiling only
after testing actual maximum prefill rows, all M64 tails and dense/compact
work encodings. The current prototype only proves M<=129. Production must
support both gathered and existing supported no-gather calls or reject the
latter in the typed launch plan before GPU work. Never silently skip a larger
expert population or route it into the five-row M16 wrapper.

## Preflight and file boundaries

Exact proposed production files (not edits in this task):

- New `layers/moe/gate_up_layout.rs`, `gate_up_layout_tests.rs`: pure policy,
  typed pair, checked projection spans, dispatch plans and rejection tests.
- New `layers/moe/gate_up_repack.rs`, `gate_up_repack_tests.rs`: bounded
  transactional loader helper and fault-injected mocked production tests.
- New `layers/moe/gate_up_btile_oracle.rs` + tests if the resident diagnostic
  is approved; keep separate from ordinary dispatch.
- `layers/moe/mod.rs`, `init.rs`, `helpers_a.rs`, `helpers_b.rs`,
  `ptr_table_build.rs`, `weight_loader/glm5/components.rs`, `glm5/layers.rs`:
  declarations, eager handle validation, typed ownership and conversion.
- Consumer files in the table: only local storage-aware dispatch; no changes
  to their router, collectives, shared or down implementation.
- New `layers/ops/moe_gate_up_btile.rs` plus module declaration: checked
  launch ABI boundaries, no environment parsing inside ops.
- New GLM-specific CUDA implementation/helpers under
  `kernels/gb10/deepseek-v4-flash/nvfp4/`: gate/up native repack, BF16-input
  compatibility, M16 and M64 B-tile exports. Existing grouped `.cu` umbrella
  may include the private helpers rather than adding a new build system path.
  Keep original exports and arithmetic available as independent controls.
- New `spark-server/src/main_modules/serve_phases/preflight/glm_btile.rs` and
  one call from `preflight.rs`: parse/validate shared model-side policy before
  `load_weight_store`. `factory/build.rs` repeats the policy for direct API
  callers before layers are built (its WeightStore is already loaded).
- `scripts/start-glm53-ep2.sh`: root-owned forwarding/receipt of the resolved
  storage choice, only after model-side support exists.

Preload validation rejects non-GLM, nonnative quant source, wrong geometry,
topology, LoRA, swap, hybrid, MMQ, CUTLASS aliases, FP8 expert/dequant modes,
unported token-major/atomic paths, incompatible M128/fused alternative gate/up
selection and old M16 verification. Both ranks validate the same resolved
policy; local loader additionally checks actual metadata/handles before
repacking. The server's post-build `kernel_gate.rs` alone is too late to
protect bytes already converted. Do not expand general model-loader traits.

No derived GLM transpose cache/serializer was found in the inspected GLM and
safetensors loader path: repack only after original checkpoint/RDMA loading
is complete. Fast loading and RDMA manifests continue describing native disk
bytes. Do not teach either to transmit runtime tiled buffers in this phase.
Any future persisted B-tile cache needs checkpoint identity, versioned layout,
N/K, nibble order, scale layout/format, ownership and content checksums; reject
old/untyped cache entries. Native file names alone are not a valid cache key.

## Resident oracle without a model-wide twin

First verify every load-time permutation against original bytes while scratch
still contains them (bounded native inverse-pack/full-byte check), including
scales and all untouched bytes. This checks representation, not execution.

For a separate eager-only first-forward diagnostic, reserve a single model-
owned scratch transaction: one4MiB inverse-packed transposed projection,
small isolated reference pointer tables/worklist, and shadow BF16 reference
outputs bounded by the verified width (C4/K5 each pair <=327680 bytes).
For each *actually selected local expert* and projection, inverse-pack its
tiled B into scratch and invoke the **old production** kernel with that one
expert present in the isolated tables. Compare the expert's complete rows
against the new output. Gate and up are processed sequentially. Original
production output buffers, FP4 A, offsets, router IDs and weights stay live
and untouched. Shared/down are outside the oracle. Remote holes are checked
according to the relevant scalar-zero or grouped-untouched contract.

Record success per layer/path/width only after complete comparison and actual
local work; an empty rank does not count as checked. Check actual stream
capture status before any D2H/sync; reject graph-enabled diagnostic at the
model's pre-capture decision. Ordinary timing has this oracle disabled and
allocates no diagnostic scratch. Allocation/error cleanup is explicit, with
no host callback inside a graph. An oracle mismatch aborts the forward; never
silently select a wrong-layout old kernel as a fallback.

## CPU TDD and GPU acceptance sequence

1. Pure tests first: policy matrix, strict parse, format/layout independent,
   pointer alignments/extents/overflow, per-rank ownership/alias rejection,
   both projection spans validated before comparisons, exact bytes and
   scratch reserve. Positive tests must fail with deliberately absent planner.
2. Dispatch matrix tests call the production selector for C1, C2, C3 both
   modes, C4 both modes, K4/K5 decomposition, short/general prefill, grouped
   FP4 off and missing resources. B-tile returns a matching handle or an error
   before launch; legacy tests preserve their exact prior handles. Test disabled
   grouped flags deliberately, rather than testing only the intended recipe.
3. Mocked loader transaction tests drive actual repack helper: source->scratch
   before writes, scratch reuse only after synchronization, native input mapping,
   scales transpose, unchanged addresses/bytes/scale2, remote skips, shared/MTP/
   down untouched. Inject every copy/launch/sync/allocation failure; assert no
   ready storage or forward launch, no scratch leak, and no native fallback.
4. CPU ownership tests tie native allocation pointers to WeightStore/backend
   teardown exactly once. Converted gate/up allocations remain registered;
   no new offset-owned allocations. Verify graph pointers stay stable and
   incompatible reload uses a new model/graph lifetime.
5. Oracle tests: actual old/new launch callbacks, fresh shadow poisoning,
   wrong or omitted writes detected, complete selected expert rows checked,
   overflow/alias refusal, no-local-work does not mark success, capture refusal
   before copies, cleanup faults and full immutable-input checks.
6. Root native gates: repack/inverse pack byte oracle; scalar N1/2/3 and M64
   full output/memcheck under production no-FMA and default-FMA builds; root
   measures scalar/prefill regressions as well as M16 gains. Adapt standalone
   tests to consume the final production helpers, not stale script copies.
7. Root full model: same checkpoint/reserves/context and one resolved layout
   delta, eager resident oracle then graph-on quality. Bootstrap, prefill tails,
   accepted-history repair, rejection/mismatch paths, K5, C4 and C4->3->2->1
   drains must remain correct. Measure load peak and post-load free memory on
   both ranks, meaningful output quality, acceptance and capped outputs.
8. Only root decides promotion after paired full-model measurements without
   concurrent compilation/packaging. Preserve all runs and rollback recipe;
   retain the transposed layout as the default until then.

## Coordination

StageA policy/CPU planning can precede CUDA promotion, but source publication
waits for the compatibility gates and measured subphase ceiling. Suggested
ownership: index_tensorcore typed storage/repack/scalar integration; prefill
review M64/large-prefill compatibility; upstream review independent audit;
root server preflight/launcher, native builds and all hardware operations.
Do not edit a frozen source slice after announcing freeze: request or notify
before any change so the commit, build snapshot and GPU receipt stay aligned.

## Follow-up source audit: compatibility work required before replacement

Status: planning only. This section records remaining production-reader gaps,
not permission to repack resident weights or a fixed TPS forecast. Cache-v19
review/freeze takes priority over this next implementation stage.

### Current prototypes are not a complete reader family

- `scripts/dev/glm_moe_btile.cuh` is a **prequantized-FP4** M16 reader. Its
  body rejects more than5 rows per expert; its fused compact wrapper requires
  gathered token IDs and a worklist with `mt==0`. It is suitable for the existing
  small grouped path, not arbitrary prefill or a BF16-input fallback.
- `glm_moe_btile_m64.cuh` preserves prequantized-FP4 K64 arithmetic, but currently
  rejects `M_expert>129`; its compact wrapper also rejects `work_m_tile>=3`.
  A1024-token prefill can send all1024 distinct rows to one local expert.
  Reusing this prototype unchanged would silently leave later outputs unwritten.
- The M64 prototype exports only fused compact/gathered gate+up. Production
  `prequant_fp4_gate_up` also executes separate gate/up compact calls and an
  ordinary dense-grid path. Both scalar and vector scale policies must have
  compatible readers; a missing optional M16 handle must select matching M64,
  never an old transposed handle against tiled bytes.
- `glm_moe_btile_decode{,_register}.cuh` contains compatible **BF16-input**
  rows1/2/3 readers, with shared weights still transposed. Their remote-expert
  outputs are explicitly zero; grouped kernels instead leave remote outputs
  untouched. Preserve those different contracts and their existing reductions.
- When `nvfp4_prequant_moe` is false, general prefill reaches BF16-input grouped
  W4A16 kernels with FP8 activation conversion/K32 arithmetic. No existing
  B-tile M64 prototype implements that precision contract. Complete support
  requires a separately validated reader for that branch. An initial narrower
  prequant profile may reject it before repack, but this is a declared capability
  limit, not a substitute for scalar/bootstrap/verify/drain support and not
  permission to silently change BF16 activations to FP4.

### Smallest safe implementation partitions

1. **Storage and loader:** new `layers/moe/gate_up_layout.rs` + tests and
   `gate_up_repack.rs` + tests; focused declarations/handles in MoE `mod.rs` and
   `init.rs`; extraction of routed gate/up versus shared/down phases from
   `helpers_a.rs`; `weight_loader/glm5/components.rs` and `glm5/layers.rs` own
   target-only conversion. The existing helper cannot transpose/free a pair
   after its bytes have been repacked. Preserve exact resident addresses and
   original allocation ownership; native-source permutation needs its own
   byte oracle because existing fixtures start from transposed source bytes.
2. **Small-row readers:** new checked ops and CUDA family for BF16 rows1/2/3;
   storage-aware dispatch in `forward.rs`, `forward_phase.rs`,
   `forward_k2.rs`, `forward_k2/unified_t.rs`, `forward_k3.rs`, and
   `forward_batched.rs`. Keep scalar arithmetic/shared output/down/EP unchanged.
   `forward_k4.rs` and `forward_k5.rs` must preserve optimized shared projections
   and their routed-only K2/K3 decompositions; `forward_c4.rs` reverse scalar
   control must reach the compatible N1 reader for every row. Grouped C3/C4
   selection must use the typed capability as well.
3. **Grouped/prefill readers:** `prequant_fp4.rs` and
   `forward_prefill_routed.rs` bind M16/M64, fused/separate, compact/dense plans
   to the storage type. Keep quantizer inputs/output ownership and scale policy
   unchanged. `forward_prefill.rs`/`forward_prefill_phase.rs` retain sort/router,
   shared-expert and blending behavior. Extend the M64 CUDA body/wrappers and
   standalone fixture before production use; implement BF16-grouped compatibility
   separately if that existing precision profile is admitted.
4. **Incompatible routes and gates:** explicit pre-repack rejection for MMQ,
   CUTLASS, hybrid conversion, token-major/atomic and incompatible expert-format
   modes; defensive typed guards in `helpers_b/c.rs`, `mmq_layout.rs`,
   `forward_token_major.rs`, `forward_atomic_c4.rs` and affected original-reader
   helpers. Root owns server/factory policy, launcher and hardware gates.

Do not broaden `use_t_layout_for_decode/prefill` to mean “some usable layout.”
They currently require gate/up/down `_t` tables; leaving gate/up absent without
new dispatch makes `forward.rs` fall into native readers of now-tiled bytes.
A typed capability must choose the correct routed reader AND retain the actual
transposed down view. Legacy raw tables are unavailable in the B-tile variant.
Tests must cover disabled grouped/M16 flags and fallback decompositions, not
only the preferred K5 path. No normal scalar decode or C4 drain may be excluded
as a convenience workaround.

### Mandatory additional gates

- Large prefill: full-output old-kernel comparison for concentrated local
  expert populations130/148/255/256/257/1023/1024 in addition to existing M64
  boundaries; no-local, mixed local/remote, shuffled gather and nonzero offsets.
  Cover the highest configured production row bound, not just total expanded
  scratch size. Bound memory by streaming output/oracle chunks where necessary.
- Every actual work encoding: separate and fused compact, ordinary dense grid,
  scalar/vector scales, and any admitted no-gather mode. Test deliberately
  poisoned outputs so an early-returning/tail-skipping kernel cannot pass.
- Native exactness/memcheck for both build FMA policies, fixed-pointer graph
  metadata refresh, immutable inputs/scales after timing, and native-to-tile
  plus scale-transpose byte checks. No numerical relaxation for fallback paths.
- Full-model cold prefill, C1 bootstrap/serial fallback, K2/K3/K4/K5, C4 scalar
  and grouped controls, and C4→C3→C2→C1 drain/permuted slots. Retain the original
  transposed default and restart rollback. Measure actual cycle/acceptance and
  capped output separately; the microkernel speedup is not a TPS promise.
