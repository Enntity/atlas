# Shared/down unified-loader extraction: preserve legacy construction

Status: root approved the complete plan before implementation. The private
extraction and controller CPU characterization are implemented and frozen;
focused7/7, full-suite866/866, non-test lib check and formatting gates pass.
Independent exact-hash freeze review is pending.
Root owns activation, commits and hardware. The hidden-trace agent released
controller Cargo after its freeze; no trace source was changed here.

## Purpose and boundary

Extract the existing shared gate/up and complete down phases from
`transpose_for_prefill_unified_inner` into private, reusable construction
methods. The legacy caller still performs every operation in the same order.
This removes the structural requirement to transpose/free routed native gate/up
merely to obtain shared/down transposes in a future checked B-tile constructor.

No B-tile flag, layout selection, loader activation, Ready publication, reader
selection, weight conversion or new serving path. Do not connect the sealed
unpublished repack transaction to a MoeLayer in this partition. There is no
measured throughput gain from this extraction itself.

## Source audit: current behavior that must remain literal

`weight_loader/glm5/components.rs` calls `quantized_any` for local gate/up/down,
loads shared weights, constructs `MoeLayer`, then selects MMQ, CUTLASS or
`transpose_for_prefill_unified_keep_shared` when allowed. The last wrapper
passes `(keep_originals=false, keep_shared_originals=true)`. The ordinary
unified wrapper passes `(false,false)`; hybrid passes `(true,true)` and is also
used by non-GLM callers in `qwen3_attention/trait_impl.rs`. None of those loader
call sites or decisions changes here. The appended MTP is TP/EP1 and calls the
FFN loader with prefill layout disabled; it must remain untouched.

The current complete order in `helpers_a.rs` is:

1. Transpose routed gate, then up using `transpose_experts_gpu`. Each projection
   allocates local-only packed/scale slabs, builds source/destination scratch
   pointer tables, launches packed/scales batched transpose, synchronizes the
   default stream, then frees its six scratch-table allocations.
2. Build persistent gate-T table, then up-T table. This happens after **both**
   routed projection transposes, not after each individual projection.
3. Transpose shared gate, then shared up with `QuantizedWeight::transpose_for_gemm`.
4. If not keeping originals, free each routed expert's gate packed/scales then
   up packed/scales, in expert order, nulling each projection only after both
   frees succeed. Then, if also not keeping shared originals, free shared gate
   and shared up in the existing order and null their views.
5. Transpose routed down via the same batched helper, then build down-T table.
6. Transpose shared down with `QuantizedWeight::transpose_for_gemm`.
7. If not keeping originals, free/null routed down in expert order, then free/
   null shared down only when not keeping shared originals.

Routed scale group is32 only for `Mxfp4E8m0`, otherwise16. Shared transpose
continues using its existing group16 API; do not accidentally forward the
routed group32 into shared scales. Shared gate controls whether shared gate/up
transpose/free executes; shared down has its own existing null condition.
`shared_inter==0` remains a no-op for those shared phases.

`QuantizedWeight::transpose_for_gemm` currently selects GPU transpose when its
kernel is available, otherwise the existing host byte-transpose path (also
forced by `ATLAS_HOST_TRANSPOSE=1`). GPU shared transpose uses stream0; routed
batched transpose uses `gpu.default_stream()`. Preserve even this distinction.
No new lookup, synchronization, stream, allocation or arithmetic is introduced.

## Exact implementation ownership

- Modify only `layers/moe/helpers_a.rs`: replace the extracted blocks with
  calls at their original positions, and declare private child/test modules.
- New `layers/moe/helpers_unified_phases.rs`: private child of `helpers_a`,
  containing the three methods below, visible only to that parent and its
  descendants. Being a child lets it call the parent's existing private
  `transpose_experts_gpu`; no visibility expansion for that slab helper.
- New `layers/moe/helpers_unified_tests.rs`: actual legacy wrapper and extracted
  phase characterization tests.
- New `layers/moe/helpers_unified_test_gpu.rs`: allocation/byte/table/typed-ABI
  recorder used only by these tests, with explicit failure injection.
- New `layers/moe/helpers_unified_fault_tests.rs` if needed to keep every Rust
  file <=500 lines; planned as a child of the test fixture module.

No edits to `moe/mod.rs`, `ops.rs`, `helpers_b/c.rs`, `components.rs`,
`QuantizedWeight`, WeightStore, CUDA, prefill/forward files, or the frozen
transaction modules. No public convenience wrapper for partial construction.

Proposed exact methods in the private child:

```text
transpose_unified_shared_gate_up(&mut self, gpu, config) -> Result<()>
release_unified_shared_gate_up(&mut self, gpu, config) -> Result<()>
transpose_unified_down_phase(&mut self, gpu, config,
                             routed_group, keep_originals,
                             keep_shared_originals) -> Result<()>
```

The first extracts only step3, the second only the shared part of step4, and
the third steps5–7. The routed gate/up work and routed gate/up frees remain in
their existing parent function. The parent calls shared release only inside
the same `!keep_originals && !keep_shared_originals` condition. Do not create
another broad boolean that skips routed gate/up in the public unified helper.
That would make a currently unsupported partial layer selectable accidentally.

## CPU TDD: actual functions and exact operation/state contracts

Before extraction, characterize the current public unified/keep-shared/hybrid
wrappers using real `MoeLayer::new`, `ExpertWeight`/`QuantizedWeight`, table
builder and production transpose ops. No copied legacy implementation as an
oracle. Record typed kernel arguments and H2D table bytes, allocation/free
identities, operation order, streams and logical live/peak byte counts. Assert
those independently specified contracts before and after extraction.

Use a small deterministic fixture, e.g. H128, routed intermediate64, shared
intermediate32, four expert slots/two noncontiguous local experts. Explicit
original tensor payload is34,560 bytes at group16 (27,648 routed +6,912 shared),
excluding the small pointer tables/constructor scaffolding. Every allocation
is recorded exactly; there is no GPU call or claim of native CUDA correctness.
Distinct scalar/input metadata and stable source-address labels identify every
projection and remote slot. Clear constructor events before asserting the
transpose sequence, but retain its allocation ledger for ownership checks.

The fixture must support both the actual shared GPU-launch path and the actual
host transpose fallback. For the latter, supply deterministic tiny source bytes
and compare full host-transposed output bytes; this tests real Rust byte loops,
not a fake CUDA kernel. Select the fallback by the recorder reporting a missing
`transpose_u8` handle; avoid global environment mutations. Batched routed
transpose still uses its existing loaded handle.

Tests cover:

- Public wrappers `(false,false)`, `(false,true)`, `(true,true)`, plus the inner
  `(true,false)` edge: keeping originals must also suppress shared frees under
  its existing nesting. Verify every original pointer/null outcome, retained
  native shared matrices, persistent T tables, compact local slab offsets,
  table scalar bits and temporary-table frees. No slab-interior pointer is freed.
- Routed GS16 and GS32 with shared GS16; no local experts, absent shared expert,
  shared_inter0, nonzero/remote expert slots, missing shared GPU transpose
  handle and default-stream-versus-stream0 behavior.
- Literal operation ordering and equal allocation sizes/counts/live/peak profile
  for legacy construction, including GU persistent tables after both GU
  transposes and routed GU frees before down slab allocation. Do not merely
  compare final pointers while allowing a larger transient memory peak.
- New isolated shared/down methods with routed GU native views/tables left
  intact and GU-T tables absent. Their successful calls must not allocate,
  copy, launch from or free any routed gate/up source. Down/shared fields change
  exactly as specified. Both existing `use_t_layout_for_decode/prefill` remain
  false on this incomplete fixture; no forward is attempted or authorized.
- Start isolated phase methods as explicit unimplemented production stubs,
  observe the positive method tests fail at runtime, then extract their actual
  bodies and make those tests pass. Preserve real RED/GREEN receipts separately
  from compiler errors. Existing legacy characterization remains GREEN across
  this behavior-preserving refactor.
- Inject failure at every observed allocation, table H2D, transpose launch,
  synchronization and free position for representative legacy configurations;
  include shared host D2H/H2D fallback errors. Assert exact prefix effects,
propagation/no later phase and existing field-mutation ordering. Do not consume
an error as success or introduce a native fallback. Legacy T-table predicates
can already become true before a later shared/down error; the caller must
abandon construction rather than attempting to serve that partial layer.

This is not a broad cleanup rewrite. Existing helpers may retain partial
allocations after errors; current field nulling happens after paired frees.
Preserve/record that behavior instead of silently changing it in an extraction.
All loader errors require abandoning construction. CudaBackend forgets ledger
entries before driver free; never claim failed frees will be reclaimed by a
ledger sweep. Real WeightStore pointers remain original allocation owners, and
new transpose slabs remain backend-owned bases; extraction adds no owner or
serializer and does not make native aliases valid after future B-tile conversion.

## Reachable reader audit: why this cannot activate B-tile yet

| Entry family | Existing layout-sensitive behavior that remains unchanged |
| --- | --- |
| `forward.rs`, `forward_phase.rs` | Scalar BF16 input selects all-T helper only when `use_t_layout_for_decode`; otherwise native GU tables. Bootstrap and history repair use it. |
| `forward_k2.rs`, `forward_k2/{unified_t,originals}.rs`, `forward_k3.rs` | BF16 rows2/3, routed-only variants and per-token fallbacks retain distinct T/native readers. Grouped selection is optional, not a guarantee of reachability. |
| `forward_k4.rs`, `forward_k5.rs` | K2 pairs, K2+K3 decomposition, grouped arms, generic fallback and retained-native exact-M shared paths all remain reachable. |
| `forward_c4.rs` | Grouped prefill or reverse-order four scalar calls; C4-to3/2/1 drains cannot be excluded. Shared C4 GEMVs require native shared matrices. |
| `forward_batched.rs` | Per-token T/native selection based on the literal all-three-T predicate; leaving GU-T absent would select native GU against any wrongly converted bytes. |
| `forward_prefill.rs`, `forward_prefill_routed.rs`, `prequant_fp4.rs` | Prequant M16/M64 fused/separate compact and dense ABIs, plus BF16/FP8-activation alternatives. Quantization/precision cannot be changed to fit one reader. |
| `forward_prefill_phase.rs` | Shared BF16/cache/T/native exact-M3/4/5 variants and deferred blend need the original shared layout/ownership policy. |
| `forward_token_major.rs`, `forward_atomic_c4.rs`, `mmq_layout.rs`, `helpers_a/c.rs` | Alternative native readers/converters, CUTLASS SFB, FP8/BF16 expert families remain unported or require explicit future preflight rejection. |
| `forward_prefill_bf16/fp8.rs`, `lora*.rs`, `dump.rs` | Alternate format/adaptation and diagnostics stay unchanged; future layout admission/export contracts must be explicit. |

The reader primitives' standalone1088/native-repack gates are complete, but
those are not production dispatch coverage. In particular the original raw
native tables and `weights.experts[*].gate_proj/up_proj` must be withheld or
invalidated at future publication, and all compatible readers/handles bound to
one authoritative layout. Never broaden `use_t_layout_for_*` into a generic
capability test while it continues returning raw T tables. The sealed transaction
does not revoke WeightStore aliases, and this extraction does not change that.

## Review and completion gates

Root approves this full plan before code. The hidden-trace agent completes its
module/Cargo window first; new child/test files exist before their declarations
are introduced. Controller CPU RED/GREEN, relevant full library suite,
non-test library check, workspace fmt, scoped SPDX/whitespace checks and
independent source review precede freeze/root commit. Persistent receipts go to
`atlas-campaigns/20260908/shared-down-extraction/`.

No speed, native correctness, full-model safety or B-tile Ready claim follows
from these CPU gates. A later separately reviewed partition supplies complete
production reader capabilities and exclusive typed construction/publication;
only then can root authorize a loader activation and paired serving benchmarks.

## Implementation and CPU receipts

The three extracted methods are a private child of `helpers_a`; its routed
gate/up transpose, table publication and free blocks remain in place. The only
removed unrelated local is the unused `_num_experts` binding. There are no new
allocation, I/O, synchronization, dtype, CUDA, environment or selection paths.
The resulting GS16 fixture final-live/peak byte pairs are exactly35,040/58,000
for unified,41,952/58,000 for keep-shared, and69,600/69,600 for hybrid/keep-all.
Those include original and T pointer tables, not just packed/scales payload.

Persistent receipt directory:
`/home/abc/storage/models/atlas-campaigns/20260908/shared-down-extraction/`.

- `initial-tests.log` and `behavioral-red.log` preserve early characterization
  corrections (one operation-string typo, then remote scalar expectation).
  These are not the final baseline receipt.
- `behavioral-red-final.log`: actual unchanged legacy body,3 tests PASS
  (80 configurations, all host transpose bytes, exhaustive I/O failure prefixes)
  and exactly3 runtime failures at the three explicit phase stubs. The remote
  T table scalar canonicalization to zero is explicitly characterized; it is
  not changed to the scalar-one convention in `MoeWeights::empty`.
- `focused-green.log`:6/6 after literal extraction.
- `focused-final.log`:7/7 including exact six temporary-table free identities,
  source/destination scalar-table bytes and failure-time publication checks for
  all six T/shared fields. Representative GPU/host and four wrapper modes
  exhaust888 allocation/H2D/D2H/launch/sync/free failure positions; the failed
  free deliberately removes its recorded allocation before reporting failure,
  matching the existing CUDA ledger behavior.
- `full-suite.log`:866/866 tests PASS,23.98 seconds; `lib-check.log`: non-test
  no-default library check PASS. `fmt.log` and `diff-check.log` are clean;
  `license.log` records the scoped first-line SPDX check. Rust line counts are
  helpers_a461, phases105, characterization474, recorder193, extra tests294.

The byte recorder never executes/simulates CUDA numerical transposes. It
records real production typed launch ABIs and tests actual Rust host transpose
loops. These receipts do not claim CUDA numerical or serving performance gains.
