# GLM B-tile reader closure and resident activation outline

Status: next-slice outline for root review, not implementation authorization.
Current immediate work remains the checked family, shared-T receipt and
construction-bound arena. No serving selector or Ready has been added.

## Actual ownership seam

The old concern about borrowing a load-local WeightStore forever is resolved
by existing production ownership, not a new device-memory owner:

- `factory/build.rs:32` accepts WeightStore by value; target layer construction
  borrows it at line167. At line797, the same store moves into
  `model.adopt_weight_store(store)` before the model is returned.
- `model/types.rs:578` retains that store in the model. `release_pools` releases
  arenas/pools first, then drains the store at line615, then sweeps remaining
  backend allocations (including legacy table allocations).
- `spark-runtime/src/weights.rs:446` drains every entry and attempts exactly one
  free per stored base pointer. Original in-place B-tile gate/up addresses stay
  in that store. No gate/up frees, re-registration or extra layout are needed.

The stale comment near `model/types.rs:573` says the builder borrows the store;
the actual call above owns and adopts it. Correct that comment when this slice
is implemented; do not introduce a self-referential model/WeightStore borrow.

Resident Ready should own the authoritative host projection descriptors and
the moved existing gate/up table objects, plus a non-borrowing complete-kernel
receipt bound to the actual backend identity. It does not own/free a second set
of matrix allocations. Drop construction borrows before adopting the store.
No borrowed Unpublished or CheckedArena is stored as a serving owner.

Ready publication must be reachable only from the exclusive real construction
transaction, once per layer. A second conversion cannot be admitted merely
because unchanged WeightStore shape metadata still looks native after the first
conversion. Its matrix byte layout is no longer checkpoint-native.

## Partition 1: actual reader closure, still no loader selection

Use one MoE-private authoritative routed gate/up storage discriminant. Keep
the default Legacy case byte/control-flow unchanged. Ready has no raw table
getter and is the sole authority capable of submitting tiled GU reads.

Implement actual call-site selection, not a disconnected eligibility function:

| Existing path | Required closure |
|---|---|
| `forward.rs` / `forward_phase.rs` | BF16 word/vector N1 GU; includes scalar bootstrap, accepted-history repair and scalar C4 control. Preserve existing down, shared blend and collective. |
| `forward_k2.rs` / `forward_k2/unified_t.rs` / `forward_k2/originals.rs` | N2 active/shared-disabled GU, existing base routing/output and down semantics. Intercept the native GU+down fallback selected near k2:393 when old-T is false. |
| `forward_k3.rs` | N3 fallback and routed-only GU; grouped arm uses prequant reader below. |
| `forward_batched.rs` | Per-token N1 fallback with independently bound input, route and output origins. No activation-precision substitution. |
| `forward_k4.rs` / `forward_k5.rs` | Actual K2/K3 decomposition, grouped choices, exact-M shared precompute, copyback and deferred mHC contract. |
| `forward_c4.rs` | Reverse scalar control and grouped branch; preserve row copying and collective count. |
| `prequant_fp4.rs` / `forward_prefill_routed.rs` | Existing A quantizer and scalar/vector policy; gathered <=5 may use M16, every other supported gathered/route-major case uses M64 dense/compact. Dense includes 17 row tiles at1088. |
| `forward_prefill.rs` / `forward_prefill_phase.rs` | Existing sort/worklist producers, shared chain and padded prefill/graph refresh remain connected to exact typed consumers. |

Do not broaden `use_t_layout_for_decode/prefill` in `helpers_b.rs:170/193`:
their old meaning includes gate/up T tables. Introduce storage-aware dispatch
at the above sites while retaining a separately valid down-T contract.

Admission is part of reader closure: `prequant_fp4.rs:85`
(`glm_native_moe_resources`), `forward_k5.rs:52/82/97`, K3:29/136/161 and K4:27
must retain the supported grouped/deferred-mHC path with B-tile GU ownership.
GU `_t` absence must not silently disable those paths or select the native
fallback. The legacy predicates and default legacy branches remain unchanged.

The actual K4 decomposition is a useful boundary test: `forward_k4.rs:107`
passes a shifted input row to K2, which reroutes into base scratch and writes
base outputs before copying back into the input. A single shared row offset
cannot represent that contract. Require independent checked input, route and
output origins; also cover contiguous-global-output decomposition. Routed-only
zero sinks must not overwrite precomputed shared contributions.

Tests must obtain Ready through actual validated source/repack/builders and
publication, not manufacture it. Enumerate every actual entry above, both
scale/word policies, all fallback/drain widths, graph metadata refresh and
failure-before-launch. Keep existing oracle, transaction and legacy tests.
No environment activation in this partition.

## Partition 2: one bounded restart-time loader transaction

Resolve the policy once, strict unknown-value failure and default Legacy.
Before the first conversion, validate the complete42-target plan (exclude
dense FFNs and appended MTP), every native local GU source, all16 handles and
the actual model arena maximum. Predictable incompatible paths fail before
mutating matrix bytes, not on first serving use.

`weight_loader/glm5/components.rs:125` is the actual target MoE seam. It creates
the native ExpertWeight views, actual three tables via MoeLayer::new, and at
line255 currently invokes whole unified conversion. The B-tile branch must
bypass that whole GU transpose/free phase, not append after it:

1. Prevalidate native checkpoint metadata before `quantized_any` can hide a
   derived or incompatible source; preserve normal shared/down loading.
2. Build the actual layer/tables, run in-place native GU conversion with the
   single reusable4MiB workspace owned by the rank's loading pass.
3. Run the extracted real shared-only and routed/shared-down phases, retaining
   shared native copies and original legacy down arithmetic/order/peaks.
4. Validate actual table bytes and complete shared/down ownership; move existing
   gate/up table objects into Ready. Null/invalidate legacy native GU table slots,
   leave GU `_t` tables absent, and invalidate `weights.experts[*].gate_proj/up_proj`
   native views. Do not free the underlying WeightStore allocations.
5. Publish the storage discriminant only after all checks succeed. Any failure
   after a write abandons construction; no old-layout fallback or automatic
   inverse conversion. Continue normal model-owned store handoff at factory end.

`ep_prefill.rs` derives local ownership from config local ranges, not QW nullness;
Ready must retain the checked local map for any other consumer that formerly
derived locality from native GU views. Raw aliases elsewhere must be enumerated
and invalidated; null pointers alone are not a dispatch guard.

Every incompatible converter (`helpers_a/b/c`, `mmq_layout`, CUTLASS, token-major,
atomic C4, alternate BF16/FP8 grouped arms) must explicitly reject Ready before
any read/free/launch. Readiness must not mean that null weights are silently
treated as remote experts. No weight dump/export may label converted bytes as
checkpoint-native.

In particular helpers_a near229/241 and transpose_experts_gpu near304 infer
locality from native gate-pointer nullness. Reject Ready before their loops,
otherwise invalidated aliases can become a silent all-remote no-op. The CUTLASS
helper near396 falls back to native N-major scales when GU `_t` is absent;
reject before the first host snapshot/copy. Exercise actual forward_atomic_c4,
forward_token_major, MMQ and CUTLASS entry points in rejection tests.

Actual BufferArena construction follows layer loading at factory/build.rs:360;
it is passed into TransformerModel::new at line679.
Resident arena binding must therefore happen after the real model arena exists,
not retain the temporary construction lease. It should use the existing owning
model construction lifecycle, with no model-wide scan inside a launch. This
exact handoff is part of the reader/activation implementation review.

## Partition 1 ownership and actual TDD matrix for approval

Keep implementation split into private children under MoE. One resident storage
child owns the discriminant and non-borrowing Ready metadata; one checked launch
child reuses the current ABI bodies rather than copying argument assembly. A
construction-only publisher consumes the validated transaction and moves the
actual table objects. No public raw launch/table accessor, caller-created Ready,
or model-owned device-memory registry is introduced. Existing table receipts
remain stamped by real builders. Add only the storage field/default initializer
to mod.rs/init.rs. Evolve the current construction lease privately rather than
keeping both a detached serving API and an unused checked API.

The owner of this slice needs the actual forward files listed above,
helpers_a/b/c and any concrete alternate reader discovered by their call graph,
the private repack children, plus minimal factory/model construction hooks for
post-arena binding. Reserve these paths and controller Cargo exclusively before
editing; the hidden-trace author retains its separate graph window. No loader
environment parsing belongs to this partition.

| Actual behavior RED before implementation | Required GREEN evidence |
|---|---|
| Real validated source/repack/table publication cannot produce resident ownership | Consumes authority once; old GU views/tables invalidated without frees, originals remain in actual model-adopted store; failed publication cannot fall back |
| Actual N1/N2/N3 entry submits legacy GU for a published owner | Both policies, active/routed-only, bootstrap/repair/drain paths submit only matching tiled GU; unchanged down/shared/collective event sequence |
| Actual C4/K5 and K3/K4 admissions reject missing GU-T | Grouped and deferred-mHC still selected with Ready; existing quantizer/sort/builder refresh precedes exact consumers on each graph construction/replay input update |
| Actual shifted-input K4 pair misbinds origins | Nonzero input with base route/output and independent global-output forms preserve earlier routed rows and shared scratch |
| Actual padded-prefill/route-major consumer falls through | Rows1/5/6/64/1088, both scale policies, actual metadata producers, dense17 M tiles at1088; unsupported BF16 grouped entry rejects before GPU work |
| Actual converter/alternate reader sees nulled GU as remote | Atomic C4, token-major, MMQ, old transpose and CUTLASS host snapshots refuse before any copy/free/launch |
| Actual legacy entry is perturbed without Ready | Recording traces, output addresses and allocation/upload/free sequences equal existing legacy fixtures |

At factory's post-arena seam, validate the actual arena capacities and live
owner ranges against the still-live store and moved table/shared owners once.
Retain a fixed owner snapshot in Ready, not a borrow of the stack-local arena,
store or ForwardContext. An arena can move as a Rust value after construction;
therefore bind its allocation addresses/capacities plus supplied live backend,
not its temporary Rust address. Per launch require exact owner snapshot matches
and cheap subrange checks; never rescan the model or perform table D2H in a graph.
BufferArena still has no independent allocator/backend receipt; rely on the
existing actual factory/backend ownership contract, and do not claim more.

Before publication becomes callable by the loader (partition2), missing arena
binding must be a hard refusal, model release must prevent further forwarding,
and the actual release sequence must show each original allocation freed once.
Tests should construct and release the actual model/store fixture, not emulate
teardown with unrelated local counters. No new serving selector is approved by
this outline; root must read the finalized partition1 plan before code starts.

## Fail-before-conversion profile and final gates

Require exact Standard NVFP4 source proof; GLM4096/2048,288 experts,top8 sigmoid;
matching TP=EP2 ranks; no adapters/swap/offload/hybrid/MMQ/CUTLASS; no old T-layout
oracle; actual arena rows1..1088; complete supported shared/down construction.
Require existing prequantized A for grouped prefill: unsupported BF16-input
grouped prefill must be rejected BEFORE conversion. A preferred C4/K5 flag does
not prove scalar/bootstrap/drain/padded-prefill alternatives unreachable.

Final root gates: exact committed-source CUDA/PTX/output/memcheck checks, actual
model default-off/on correctness and equal resident-memory/4MiB transient gate,
then paired full-wall C4 performance. Keep numerical uncertainty and residual
MTP diagnostic differences separate from CPU ABI/control-flow evidence. No
additional standalone prototype is a prerequisite when existing promoted
readers already cover the required mathematical families.
