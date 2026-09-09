# Independent GLM C2–C8 host dispatch

2026-09-09. Root approved implementation of A and B; both are in progress.
Server admission C remains root-owned and is not implemented or native-qualified.
This is nonspeculative concurrent decode, not temporal K5 or paired MTP.
Root owns native, deployment, watchdog, health checks and commits.

## One bounded selection

Propose strict default-off `ATLAS_GLM_INDEPENDENT_DECODE=0|1` (reject malformed
and non-Unicode values). ON selects one complete independent path, not a family
of per-width experiment flags. Require exact glm5_next/local validated geometry,
world2/TP2/EP2, EP-v2, BF16 KV plus semantic index, FP32 KDA state, configured
active==admitted in2..=8, context1..=2048, no proposer/self/ngram/DFlash speculation,
no adapters, resident state/no HSS or swap, and no sparse/dynamic-sparse mode.
Do not alter old C4 or C2 flag semantics when this mode is OFF. Reject conflicting
experimental storage/numerical routes before load (resident B-tile, CUTLASS/MMQ,
M16/oracles and shared-FP8 cache are outside this first native-T/M64 envelope).
Existing grouped helpers are selected by the new mode directly; do not require
K5 flags or use K5 policy to recognize five independent requests.

Every actual drain width1..configured_cap is valid, including5/7. C1 goes through
the existing scalar decode/FFN/state path. C2..C8 use exact live widths, never
padding to8, duplicating a row, compacting slot identity or borrowing a peer state.
Keep existing graph enable/disable behavior and slot-vector graph keys; validate
new mode before lookup/replay as well as eager work. First native correctness is
eager, followed by explicit warmed graph qualification, not silent graph bypass.

## Source facts and ownership split

### A — model policy, KDA and FFN/MoE (prefill author after approval)

- `model/glm_c4.rs`: share existing geometry/checked row-sized scratch arithmetic
  with new small `model/glm_independent.rs`; preserve legacy wrappers/tests.
  `model/mod.rs` registers the child. No generic Model capability or MTP edit.
- `model/trait_impl/decode_a2.rs`: mode-aware immutable token/state-count, position,
  actual capacity/dtype/row-budget checks before head E0 and before compute/graph
  lookup; preserve scalar n1 branch, exact distributed width and graph lifecycle.
  `model/impl_a2.rs`: selected-mode E0 N bound against live slot capacity before
  receive Vec allocation; preserve packet order/duplicate-ID checks/slot mapping.
- `model/ssm_indexed_decode.rs`, `layer/ssm_batch.rs`, `layers/ops/kda_indexed.rs`:
  runtime, view and shape currently cap at4. Admit2..8 only through new runtime
  mode; expand structural view bound1..8 and retain every unique slot, live pool,
  disjoint span, FP32 stride and state-pointer equality check. Metadata upload
  remains rank-local before every replay. C1 still does not select indexed KDA.
- `layers/glm5_kda.rs`, `glm5_kda/{multi_seq,indexed_core,projection}.rs`:
  replace selected-mode width guards; require actual indexed pair before mHC or
  projections (no silent scalar state fallback for new multiseq mode). Existing
  hot projection adapter already handles4..8 via W4a16BatchmTiers; preserve2/3
  seven-argument ABI and4..8 runtime-M eight-argument ABI. BF16 batchm covers8.
  No temporal decode_batched/rollback intermediate logic changes.
- `layers/ffn_c4.rs`, new `layers/moe/forward_independent.rs`, `moe/mod.rs`:
  add an independent width-taking FFN entry, shared by KDA and MLA mHC callers.
  Dense FFN uses existing width-supported forward_km or true rows-N prefill;
  None remains identity. C1 bypasses this entry. No forward_k5/for_hc call.
- Reuse `moe/forward_c4.rs` checked arena/router/shared helpers with rows, and
  `prequant_fp4.rs`, `forward_prefill{,_phase,_routed}.rs` for one new independent
  eligibility predicate. Require actual norm input, attn_metadata.num_seqs==rows,
  cap>=rows, local comm rank, native-T/scales/selected handles before first write.
  Keep existing C3/C4/C2 OFF paths literal; scalar router GEMV per row in ascending
  order for new mode. Shared batch2/3 use dedicated ABI,4..8 actual tier handles.
  Reuse compact M64 GU, staged fused SiLU/FP4 and existing prequant M64 down.
  Shared remains before routed (all rows<=8, overlap threshold>64); replicated
  shared is added once after sparse-EP routed reduction. Valid remote expert IDs
  are required, not an invalid profile. No new allocations or weight repacks.

### B — MLA/export/init (separate author, coordinated graph window)

- CUDA prerequisite remains `glm_mla_c6_c8_standalone_plan.md`: actual6/7/8
  exports of unchanged MLA body, scalar/host numerical and memcheck gates before
  promotion. It does not authorize host activation by itself.
- `layers/qwen3_attention/{types,init}.rs`: resolve new exact MLA6/7/8 handles;
  no widening of another architecture's default. Existing projection tiers already
  resolve NVFP4 widths4..8; no new W4A16 or BF16 kernel is required.
- `trait_impl/multi_seq/{c4,mla_glm,mod}.rs`: share selected policy/positions/
  scratch preflight; select exact MLA2..8 absorption/extraction handles, validate
  each before cache writes, widen GLM projection match4..8 and use row-count FFN.
  Preserve cache assemble/causal paged attention per-row metadata and final TP
  reduce/mHC order. Independent5 is not K5 policy; temporal5 behavior stays literal.
  No sparse MLA changes. Author A owns policy/FFN APIs; B alone edits these callers.

### C — server admission (root-assigned last integration slice)

- Preflight ordering audit found raw, unsharded head counts are used for reserve
  before `resolve_topology` halves them for TP2. At the same1024-row envelope,
  C4 inference reserve is8327.5MiB raw versus4419.75MiB local; C8 is13682.5
  versus7097.25MiB. These numbers preserve the existing512MiB headroom and are
  not a complete corrected allocation budget. Proposed selected-only fix:
  resolve the actual Topology once before reserve, retain it, and reuse it at
  the later topology handoff. Do not duplicate division or mutate config twice;
  preserve ordinary OFF ordering. Review the actual prefill-plus-decode arena
  envelope (native C4 uses1028 rows, current preflight1024) and live SSM dummy
  slot as well: correcting topology alone does not prove exact accounting.
  Keep rollback rings, OOM reserve and post-load safety checks unchanged.
- `main_modules/serve_phases/preflight.rs` plus small policy-test child: actual
  resolved args call shared new policy before load; accept cap2..8 only with new
  mode, refuse9/partial capacities/speculation/dtype/context conflicts. Ordinary
  preflight and prefix/SSM rollback accounting remain unchanged when OFF.
- Existing build.rs worker slot vector, scheduler stable free-slot allocator,
  E0 serial protocol and EOS/cancellation retirement are already capacity-based;
  source-audit them, do not rewrite scheduler. Actual survivor slot7 must drain
  through n1 scalar without relabeling. No paired serial driver/T2 activation.
  Retain current watchdog, rollback rings, uncertainty/stop behavior and launch
  health gates; capacity expansion is not permission to relax any of these.

## Resource proof before admission

Use actual BufferSizes/DecodeMetaLayout and pool allocation counts at configured
cap, not a claimed fixed C4 footprint. At8/topk8/inter2048:64 routes, compact
worklist8208B, sort1924B but BF16 router logits4608B, input FP4 staging18432B,
down staging73728B, routed gate/up262144B each, down524288B. Worst-case M64 tiles
remain1 for<=64 routes, so existing exact-offset D2H branch should stay unused.
Derive KDA/MLA/mHC extents with checked row arithmetic (C8 mHC highway524288B
for hc4); do not blindly double fixed shared constants. Four additional FP32
KDA slots cost4*(2097152+196608) bytes per KDA layer for live state alone;
existing snapshots/rollback rings, target KV/index, graphs and arenas add more.
Keep pre/post-load reserve checks; root records actual high-water memory/no swap.

## Minimal real-dispatch TDD and promotion order

1. Policy/row-boundary tests2..8 plus0/9, cap/drain mismatch, context2047/2048,
   actual BF16/FP32 geometry and one-byte-short arenas. Keep old C4 tests intact.
2. Extend existing `forward_c2_tests` actual MoeLayer/FfnComponent recorder pattern:
   both ranks/policies, widths2..8, exact shared ABI/runtimeM, scalar router order,
   compact GU/down/reduction bytes and no alloc/free. Missing chosen handle or
   undersized arena refuses before writes. Real entry RED, not copied math.
3. Extend actual indexed ops/metadata tests for8 and noncontiguous permutations/
   draining subsets; actual KDA multiseq entry control at5/7/8 must issue one
   indexed conv+recurrent pair, not scalar/temporal fallback. Keep C1/legacy tests.
4. MLA owner tests actual selector+launch ABI at5/6/7/8 and missing-handle refusal;
   CPU dispatch evidence is not attention numerical equivalence.
5. Narrow actual Model E0 head/worker test with owned slots at8, reordered drain
   through7/5/2/1, per-row lengths and local slot maps; reuse existing recording
   backend and constructor patterns, no fake paired capability. Server startup
   negatives run before backend load. Existing cancellation/retirement regressions.

Freeze/review each owner slice; serialize Cargo. No redundant broad author suites.
Root then qualifies exact integrated binary: eager quality and graph reuse at
every1..8 width, skewed lengths/EOS/cancel/slot reuse, matched warmed C1/C2/C4 and
C6/C8 aggregate plus per-session latency, clean stop/health and memory receipts.
No throughput forecast or endpoint claim follows from available CUDA exports.
