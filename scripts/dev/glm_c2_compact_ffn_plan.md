# GLM C2 compact FFN: default-off same-binary experiment

2026-09-09. Root-approved implementation is CPU-gated, default off; native
numerical/whole-model quality and matched warm throughput remain pending.
No CUDA source, serving configuration, allocations or repacks changed.

## Why this is distinct from the old rejected C2 path

Frozen v26 `e40a9066` KDA `multi_seq.rs:317–327` and MLA
`multi_seq/mod.rs:361–411` invoke scalar FFN twice at C2; C3/C4 can use compact
prequantized native-FP4 MoE. Attention projection batching already includes C2.
The neutral batch-two FFN observation in `docs/glm53-dual-spark.md:189–194`
was committed in `e6b9ffa84` (September3), before compact C3 W4A4 integration
`ad42aac0` (September6). Preserve old `forward_k2` completely. This candidate
uses the current C3/C4 sorting, compact worklist and tensor-core kernels, not
that older batch-two routed GEMV. No predicted endpoint rate follows.

## One bounded implementation

Add explicit `ATLAS_GLM_C2_COMPACT_MOE`, default OFF; absent/0 selects the
existing scalar caller loops literally, 1 enables this experiment, malformed
values fail configuration before execution. Freeze the same value on both
ranks for each process; never change it during graph lifetime. No scheduler,
MTP width, factory layout, resident B-tile activation or model precision flag
changes. Nonspeculative active4/admitted4 v26 resources are the fixed baseline.

Call only from actual independent multi-sequence KDA/MLA FFN sites with n==2.
A small FfnComponent adapter returns no candidate for Dense/None, retaining
their existing scalar behavior. For an opted-in GLM MoE, fail closed on missing
required resources; do not silently benchmark a scalar fallback as compact.
The off branch adds no copies, reductions, allocations or changed mHC order.
One contiguous [2,4096] output is consumed by existing two-row hc_post before
any subsequent layer reuses scratch. Do not treat two independent rows as a
temporal K2 verifier or change convolution/recurrence state ownership.

Exact candidate envelope: glm5_next, H4096, routed/shared intermediate2048,
288 experts, unique top8 sigmoid/correction routing, TP2/EP2, actual decode
capacity4, actual attn_metadata.num_seqs==2, input==norm_output, no adapter,
mixed expert format, hash routing, pre-expert norm or gated shared expert.
Require the existing native-T NVFP4 resources and prequant/fused-SiLU/sparse-EP
reduce/compact builder/fused gate-up handles; shared projections are native
NVFP4 with actual nonzero batch2 handle. Refuse Published B-tile storage for
this v26-layout experiment; no repacking or alternate reader is implied.
Keep scalar/vector scale policy identical to the control. Keep C1/C3/C4 and
K5 selection unchanged. M16 eligibility stays rows4/5; C2 uses the established
M64 compact fallback without widening any CUDA kernel specialization.

Router logits retain exactly TWO original dense_gemv calls in scalar row
order, as C4's c4_router_logits does, then existing batched sigmoid/top8 and
sorting. Do not adopt C3's different dense-GEMM router arithmetic. Grouped
gate/up quantizes the two BF16 input rows once, uses the existing compact
fused gate/up, then existing fused SiLU+FP4 quantization/down/unpermute.
Shared gate/up/down use existing exact batch2 W4A16 projections and existing
SiLU. Reduce only routed [2,H] once, then add replicated shared output once;
never reduce replicated shared twice. At C2 the existing overlap threshold
`num_tokens > 64` is false: shared is computed before routed work and added
after reduction. Check both eager and captured modes; no C2 overlap is implied.

## Files and resource proof

Bounded existing implementation surfaces under `crates/spark-model/src/layers`:

- `ffn_c4.rs`: narrow optional independent-C2 adapter, no Dense path change.
- `glm5_kda/multi_seq.rs` and
  `qwen3_attention/trait_impl/multi_seq/mod.rs`: candidate branch only for C2;
  original row loop remains control. The Qwen-named implementation above is
  the actual GLM MLA/mHC caller, guarded by `model_type == "glm5_next"`.
- `moe/prequant_fp4.rs`: C2 shape/resource selector, include C2 in compact
  fused gate/up selection; preserve current C3/C4 predicates and M16 behavior.
- `moe/forward_c4.rs`: share only checked 2/4 arena arithmetic and scalar-order
  router; keep C4 wrapper semantics. Approved `moe/forward_c2.rs` owns the
  C2 guard and actual seven-argument batch2 shared adapter (no runtime M).
- `moe/forward_prefill.rs`, `forward_prefill_phase.rs`,
  `forward_prefill_routed.rs`: C2 routes to those shared helpers and compact
  builder. No generic prefill, remote-output or overlap cleanup rewrite.
- Existing C4/prequant tests plus one focused actual-call test child; no new
  general fixture framework. Resolve strict toggle once with existing MoE
  initialization conventions if needed; include that exact small init field
  scope in implementation review, not a new configuration subsystem.

No additional model/device allocation or weight twin is required. Check actual
BufferSizes before first write, using the established C4 arena proof restricted
to rows2: normalized/output16384B each; routed gate/up65536B each;
routed-down131072B; shared gate/up8192B each; shared-down16384B. Input FP4
staging4608B and down staging18432B reuse existing expert arenas at the same
phases. Worklist=16+2*8*16*8=2064B, max256 entries; routing sort metadata is
max(3*16*4+289*4,2*288*2)=1348B. All fit the already-qualified active4 buffers;
verify capacities/overlap explicitly rather than assuming an allocation name.
With16 expanded routes, ceil(16/64)=1: existing exact-tile code performs no
expert-offset D2H readback. Never add a truncating load-factor assumption.

Source-level changes sought: two 8192B routed reductions become one16384B
reduction; batched shared projections traverse each matrix once rather than
twice; routed weights can be reused when both rows select the same expert.
Distinct routes need not save routed weight reads, and C2's underfilled M64
tiles, sort/quantization overhead or larger working set may erase a win. No
claim that logical traversal savings equal cold DRAM traffic or endpoint speed.

## Minimal evidence and promotion gates

1. Actual CPU-call characterization: off path retains both scalar calls,
   reductions. New-entry behavioral RED then candidate GREEN:
   two real independent row inputs, both ranks, reversed/noncontiguous slot
   order; one grouped call, one16384B routed reduction and shared added after
   it. Preserve exact router handles/row offsets and shared batch2 calls.
   Shape, missing-handle, undersized-arena, wrong input/metadata and incompatible
   remote-output/reduction-policy negatives refuse before writes. Valid remote
   expert IDs remain required for EP2. Existing C1/C3/C4 tests remain regressions.
   Use focused/model-check/fmt/lint gates; root owns postcommit broad suites.

   CPU closure: actual FfnComponent→MoeLayer control/candidate exercised both
   ranks, both scale policies, eager/capture contexts; missing selected GU/shared
   and sparse-reduction handles plus input/metadata refused before GPU work.
   The 12 undersized cases exercise the actual shared validation helper, not
   an undersized live arena. KDA/MLA caller and mHC wiring are source-reviewed,
   not executed by this MoE fixture. Sparse recording backend is not numerics.
   Existing down remains prequant dense M64, not compact-down worklist.
2. Root-only bounded GPU comparison of the full router→MoE→mHC region at C2,
   not only GEMM: scalar control versus compact, exact routing indices/weights,
   finite outputs, full guarded buffers, identical/concentrated versus distinct
   routes and local/remote experts. Check eager and fixed-address graph replay
   with refreshed routing; memcheck before timing. Router/shared projection
   equality is distinct from routed output numerical tolerance.
3. BF16→FP4 activation quantization changes C2 routed arithmetic. Do not demand
   or advertise bit-identical final MoE outputs, and do not tune a tolerance to
   make a failing case pass. Use the existing grouped-precision acceptance
   criteria and mandatory whole-model quality: ordinary conversation, tools,
   distinct retrieval needles, mixed request lengths, cancellation/recovery
   and C4→C3→C2→C1 drain/slot reuse. No new exhaustive25 acceptance matrix;
   this is nonspeculative decode, not paired verdict ownership.
4. Same new binary, same v26 active4/admitted4/BF16-KV/no-prefix/no-spec/temp0
   recipe; only C2 compact toggle differs on both ranks. Warm separately,
   fixed148prompt/256output, three measured matched C2 batches per arm plus a
   fresh-process confirmation. Preserve per-run all-session output counts,
   full-wall aggregate, decode-window aggregate and per-session rates; compare
   those same denominators. Repeat C1/C3/C4 regressions, using existing v26
   workload/quality gates, and retain negative timings. No profiler syncs in
   throughput runs. Root controls idle health handling, container memory/swap,
   exact images/flags, stops and all hardware actions; no overlapping models.

Promote only after independent source review, numerical/quality/health gates
and reproducible matched endpoint improvement. Otherwise keep OFF and archive
the negative result. This experiment does not borrow expected gains from MTP,
resident B-tiles, QKV packing or M10, and does not delay their ownership gates.
