# GLM fused gate/up M16: guarded full-model A/B

Plan before production edits. Standalone C4/K5 tests established complete
production-output bit equality, independent CPU columns, refreshed graphs and
zero-error memcheck. Three paired runs won all 24 useful builder-inclusive
comparisons (median speed ratios 1.0619 C4, 1.0364 K5); four distinct local
weight pairs are not a real-model route/cache distribution. Down remains the
old production implementation. No default promotion or throughput prediction.

## Scope and source ownership

- Author owns MoE registry/selection and new GLM-specific CUDA helper/exports,
  plus focused CPU tests. Root owns actual graph-decision preflight callsites;
  upstream reviewer owns launcher forwarding. No node/GPU operations or commits
  by the author.
- Move validated M16 arithmetic into one production-private header included by
  the existing native-FP4 translation unit. Its only production entry points
  are fused gate/up, N2048/K4096, real sorted-token gather, block128,
  projection-Y and the existing compact worklist. Test-only down wrappers reuse
  this arithmetic; no down entry point is selected or registered in production.
- Preserve all K64 MMA and scale arithmetic. No weight repacking, new GPU
  allocation, routing, builder, shared expert, SiLU, down or EP changes.

## Test-first host contract

Write CPU tests before eligibility implementation. Explicit
`ATLAS_GLM_MOE_GATE_UP_M16=1` enables selection; absence/0 leaves it off. Reject
invalid toggle values at initialization. Eligibility requires current native
GLM resource guards, exact model dimensions/top8/288 experts/TP2 EP2, native
NVFP4 transposed weights, a valid compact worklist and exactly 4 or 5 input
rows (the private arithmetic supports at most 5 rows per expert). The
existing fused-path caller still determines which C4/K5 paths reach selection;
do not widen generic prefill or alter C3 behavior. Bound work capacity to
rows*8*16 and preserve scalar/vecscale matching. Missing selected handles must
not silently claim M16 execution. Tests independently perturb each dimension,
model, topology, row count, layout/resource eligibility and worklist capacity.

## Optional native first-step oracle

`ATLAS_GLM_MOE_GATE_UP_M16_VERIFY=1` requires the main gate. This diagnostic is
for eager validation, not performance. Graph-decision hooks reject actual
graph-enabled use before graph lookup/capture; an additional layer guard
rejects captured execution before any host copy.

For each layer and eligible row count, once only after a successful comparison:

1. Poison existing gate/up output spans (rows*8*2048 BF16 each).
2. Launch old production fused gate/up with current quantized activations,
   real weight pointers, worklist and routing. Copy both full output spans to
   bounded host snapshots on the same stream.
3. Freshly poison both output spans; launch M16 into the identical destinations.
4. Read and compare every byte, including untouched remote rows. Stop on any
   difference; record the row-count success only after both comparisons pass.

The candidate result stays in the original destinations for the normal
SiLU/down chain. No extra GPU scratch or weight allocations are needed. Check
output buffer capacities and disjoint ranges, and preserve packed A in
expert_down_out. The oracle adds synchronization/host storage only when the
separate diagnostic is explicitly enabled. Performance runs disable VERIFY.

Remote-hole poison is intentionally retained: eligibility requires EP2,
communication, the indexed-EP reduction handle and no LoRA. The fused SiLU
quantizer computes each row's group scales independently. Existing dense down
returns for null remote expert weights before reading activation rows; indexed
EP reduction checks the expert ID range before reading down output. Shared
expert scratch is separate and blended after EP. Thus poisoned remote rows
cannot influence local results under this gate. Empty local worklists compare
successfully but do not mark the layer/row verified; retry until useful local
weights were exercised. Host snapshots peak at 655,360 bytes for K5.

Implementation checkpoint: focused MoE tests observed intentional behavioral
RED (the valid C4/K5 case failed, three rejection/graph tests passed), then
8/8 GREEN including real oracle sequencing through the mock GPU backend.
Post-extraction C4/K5 CPU fixture tests also pass. A byte comparison against
the committed standalone candidate confirms both shared arithmetic and fused
export sections unchanged. Native rebuild and full-model gates remain pending.

## Acceptance sequence (root only)

1. Focused RED/GREEN eligibility/oracle-span tests, full model CPU suite, source
   review. Recompile native kernels and rerun standalone C4/K5 gate/up plus
   unchanged down regressions after the arithmetic extraction.
2. Eager real C4 then K5 oracle checks on both ranks, with per-layer/per-row
   successful receipts and output quality. Abort on mismatch or missing proof.
3. VERIFY=0 graph-on quality plus drains/slot reuse, then repeated full-model
   M16 off/on A/B with identical model/config/request/output caps. Keep any KV
   change matched in both arms or validate it separately before combining.
4. Default remains OFF until full-model correctness and reproducible benefit
   justify a separate reviewed promotion decision.
