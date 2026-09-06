# State-indexed GLM KDA decode prototype

Scope: standalone files in `scripts/dev`, no production wiring or new feature
flag. Preserve FP32 state and Atlas arithmetic. GPU execution belongs to root.

## Rationale

Pinned vLLM `6865e67f0be02d53694517f6f71d7fb96492792d` batches independent
decode rows using device state indices for convolution and recurrence:
[GLM KDA orchestration](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/models/glm5next/nvidia/kda.py#L468).
Its [recurrent launch](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py#L33)
also includes sequence count in the grid. This is execution-structure evidence,
not a measured GB10 speedup. No external implementation is copied.

Atlas's existing `glm5_kda/multi_seq.rs` batches projections but submits
convolution and recurrence per row. First replace only that submission pattern:
one convolution grid and one recurrent grid, retaining the existing arithmetic.
Do not import vLLM's value tiling or change projection quantization here.

## ABI and ownership

- Real geometry: local heads32, dimension128, packed QKV channels12288,
  convolution width4. BF16 activations/output; FP32 H and convolution state.
- Each row represents exactly one independent token, never temporal verification.
- Device `int32 state_slots[N]` selects slots from explicit pool bases and
  64-bit element strides. Convolution grid `(48,N)`, block256; recurrence grid
  `(32,N)`, block128. Invalid indices leave state/output untouched.
- Host validates geometry, capacities and unique live indices before launch.
  Duplicate live indices are a data race and are never launched by the test.
- Graphs retain exact width, pointers and pool strides; device index contents
  change between replays. No allocation inside capture or kernels.
- Experimental bodies mechanically derive from Atlas's existing serial
  convolution and reference recurrence. The unchanged production kernels remain
  separately compiled comparators. On later production promotion, extract shared
  inline bodies rather than retain duplicate arithmetic implementations.

## Tests first and stop gates

1. Build the harness against the candidate ABI before implementing its bodies.
2. Compare candidate convolution with both existing per-row token-parallel
   (current default) and serial kernels. Compare every BF16 convolution output
   and every FP32 convolution-state element, not only sampled values.
3. Feed the respective convolution results into candidate/existing recurrence;
   compare all BF16 outputs and every FP32 H element bitwise after every step.
4. Eight padded slots; N1–4, distinct initial states, zero and nonzero histories,
   reordered/nonprefix indices, C4→3→2→1 drains, reset/reuse a freed slot, and
   repeated updates. Test invalid-slot masked rows without unsafe accesses.
5. Replay exact-width graphs while refreshing indices/inputs; compare against
   eager per-row reference. Check all slot-padding/allocation canaries and
   unchanged inactive slots; poison unused output rows.
6. Hard GPU allocation ceiling64MiB; expected footprint below40MiB. No model,
   weights checkpoint, communication, production change, or GPU run by agents.
7. Root runs correctness and compute-sanitizer before any production integration.
   Any state/output bit difference, modified inactive slot, guard failure or
   runtime error stops promotion. Throughput is not inferred from launch count.

The strengthened gate repeats the 13 slot sequences under exact-zero Q,
exact-zero K, both-zero Q/K, and stronger state/activation/gate magnitudes,
for 91 total steps. Zero Q/K is established in both convolution history and
current input; all convolution and recurrent comparisons remain active.
`--timing` is a separate CUDA-event mode (N2/3/4, median of five 100-step
intervals), comparing per-row TP convolution plus recurrence with the indexed
pair. State initialization is excluded; timing never substitutes for the gate.

Numerical diagnostic: production `--fmad=false` is required. Proving `dim==128`
inside the indexed kernel additionally changed one BF16 output while H remained
exact; retaining the original runtime dimension arithmetic passed all initial
39 cases and memcheck. Geometry belongs in host validation, not a CUDA equality
guard that changes compiler constant folding. No tolerance was relaxed.

## Proposed production promotion (not authorized or implemented yet)

Prerequisites: expanded 91-step exact gate, memcheck, and isolated timing.
No new tuning flag: select this implementation inside the existing
`ATLAS_GLM_KDA_MULTI_SEQ` independent-decode path. Preserve all projection,
gating/norm, output projection, collective, FFN, and mHC code around the core.

### Shared CUDA arithmetic and independent reference

1. Preserve a frozen, renamed scalar reference in the test directory BEFORE
   refactoring. Otherwise comparing two wrappers of the same new helper can
   miss an extraction regression. Keep this reference test-only.
2. Extract the existing serial convolution and recurrent bodies into private
   force-inline device helpers, in `kernels/gb10/common` private `.cuh` files.
   The existing exports in `causal_conv1d.cu` and `kda.cu` retain their exact
   ABIs; indexed exports in those same modules only select/rebase pool/row
   pointers and call the same helpers. Leave TP convolution unchanged.
3. Retain runtime dimension arithmetic and production `--fmad=false`; do not
   replace rsqrt, alter reduction/FMA grouping, or prove dim128 in a wrapper.
   Host validation owns the exact geometry gate. FP32 H stays FP32.
4. Re-run the indexed gate against the frozen scalar reference after extraction.
   Also test legacy temporal tokens1/2/3/5/17 to protect scalar prefill users,
   and compile/test the original scalar build flags as well as GLM's flags.
   Shared extraction itself is a numerical change until these gates pass.
5. New symbols remain in existing `kda` and `causal_conv1d` modules; load
   optional handles during GLM layer initialization, never during capture.
   No new module registration or device allocation should be necessary.

### Explicit pool and row contract

Use a dedicated optional `ForwardContext.ssm_batch` view, not attention/LoRA
metadata. A small typed view carries device i32 slot IDs, exact active width,
claimable slot capacity, FP32 H/conv element strides, and borrowed per-SSM-layer
base arrays. Its checked constructor is private to the model/layer contract.
Unrelated forward contexts explicitly carry `None`.

- Bases come from `SsmStatePool.h_state_pools` and `conv_state_pools`, never
  reconstructed from a row pointer or assumed contiguous across layers.
- H stride is `h_stored_bytes / 4`, and FP32 eligibility requires
  `h_stored_bytes == h_bytes`. Conv stride is `conv_bytes / 4`. Capacity is
  `max_slots`, excluding the reserved dummy slot. No checkpoint/intermediate
  pool is eligible. Check multiplication, addition, and pointer-address bounds.
- Row IDs come from each rank's actual `seq.ssm_slot` guard (`idx()`), NOT
  request `slot_idx`, KV slot, row ordinal, or `attn_metadata.seq_slot` (LoRA).
  IDs need not match across ranks, but each rank's row order must match the
  existing EP token/request order. Reject missing/duplicate/out-of-range live
  IDs before CUDA; invalid-slot masking is a kernel defensive test only.
- Before graph lookup, validate every participating SSM layer state's actual
  H and conv pointers against that layer's explicit base+slot*stride, including
  FP32 tags. This runs on every replay, not merely a layer capture/cache miss.
- Cache the KDA layer's SSM ordinal at construction by counting preceding
  `LayerType::LinearAttention` entries. Do not confuse global layer index with
  pool index (MLA layers interrupt the sequence). Validate it against the view.

Name `DecodeMetaLayout.ssm_slots_off() = 20 * rows`, using the existing
`[20R,24R)` gap at scratch+32768. Keep total metadata bytes and EVERY existing
offset unchanged. This gap is disjoint from LoRA `[4R,8R)` and KV `[8R,16R)`.
Upload the active IDs (and deterministic -1 padding) once per decode step on
the compute stream, outside capture and before graph lookup/replay. Existing
`upload_batch_metadata_fixed` writes separate regions and does not overwrite
this gap, so the SSM upload can precede its current later call safely.

`decode_batch_compute_main` is shared by head dispatch and
`impl_a2.rs::ep_worker_decode_batch`; construct, validate, and upload there so
BOTH ranks refresh their own IDs. Use the existing safe `copy_h2d_async`
contract for temporary host bytes, not a retained copy with short-lived memory.
Keep exact-width AND ordered-slot-vector graph keys unchanged. Do not enable
graph borrowing or reuse across slot permutations as part of this milestone.

### Dispatch and CPU rejection tests

Eligibility is independent GLM decode, local heads32/D128, conv channels12288
and width4, FP32 states, N2/3/4, and exact unpadded dispatch. C1, unsupported
geometry, absent view, missing optional kernels, prefill, verification, and
non-GLM paths keep their existing implementation. Existing FP16 refusal is
not bypassed. Invalid PRESENT metadata is an error, not a silent fallback.

Pure checked launch geometry returns conv grid(48,N)/block(256,1,1) and
recurrent grid(32,N)/block(128,1,1), both grid.z1/shared0. Validate full block
dimensions (not just x), conv input/output row strides >=channels, packed
QKV/gate/beta/output byte capacities, FP32 stride alignment/minimum bytes,
nonnull bases and slot pointer, state capacity, and checked total offsets.

CPU tests must cover: N0/1/2/3/4/5 selection; wrong head/dim/conv geometry;
row stride too small; block.y/z and grid.z errors; overflow/null/alignment;
FP16 storage/tag rejection; missing/duplicate/negative/out-of-range slots;
global-layer-to-SSM-ordinal mapping; mismatched actual H/conv pointers;
nonprefix/permuted slots and drain/reuse; metadata region nonoverlap at
R32/64/128; unchanged legacy offsets/total allocation; refreshed per-rank IDs
before replay; exact-slot graph keys unchanged; absent-vs-malformed fallback.

### Proposed file ownership after approval

- Index/CUDA agent: `kernels/gb10/common/{kda,causal_conv1d}.cu`, their new
  private helpers, and `scripts/dev` KDA candidate/reference/harness/plan.
- Root/metadata owner: `crates/spark-runtime/src/buffers/decode_meta.rs`,
  `crates/spark-model/src/layer.rs` plus a small `layer/ssm_batch.rs` contract,
  all mechanical unrelated `ForwardContext` None initializers,
  `model/ssm_pool.rs`, new `model/ssm_indexed_decode.rs`, and
  `model/trait_impl/decode_a2.rs`; root alone owns model module declarations.
- Layer/ops owner: new `layers/ops/kda_indexed.rs` plus its `ops.rs` export,
  `layers/glm5_kda.rs` handle/ordinal initialization, and
  `layers/glm5_kda/multi_seq.rs` replacing only the conv/recurrent loop.
  Coordinate the typed interface with the metadata owner before editing.
- Independent reviewer: verify metadata liveness, mixed layer mapping, both
  ranks, graph cache/replay ordering, and legacy fallback; no overlapping edits.

Production GPU acceptance remains root-only: bounded exact/memcheck first,
then eager and graph full-model C2/C3/C4 quality, nonprefix/permuted slots and
C4→3→2→1 drains, followed by matched performance. Retain an image built from
the preceding source for rollback; do not add a feature flag for the new core.
