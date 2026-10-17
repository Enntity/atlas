# Q2 norm-causality experiment plan (next-repair #2–3)

Status (2026-10-12, after the independent review `CODE_REVIEW_2026-10-12.md`):
**C1 closed from native captures** (operator decision; flagged, see below).
**C2 is the next experiment.** Sequencing authority for the post-attribution
diagnostic work. The plan also
covers neighboring model/configuration coverage and a repeatable Rust/vLLM
implementation review. Those sections are planned work, not new
qualification or an expansion of GPU experiment authorization.

## Objective

Determine whether the attributed gated-norm arithmetic difference, and the
fixture-local correction `v24-minimal`, is causally responsible for any part
of the Q2 failures. If it is not, proceed
through the observed L0→L1 boundary and the revised code-path audit gap list
(G1–G9) to locate the dominant divergence.
For any shared-code correction, establish which neighboring types and model
configurations reach the changed code and validate their contracts too. Keep
the frozen GLM Q2 comparison as one explicit configuration, not a universal
numerical policy for every model.

## Frozen state and proven inputs (fingerprints)

- Registered v1 failure state (historical record): Q2 FAILS on the
  exchangeability rank plus 3 catastrophic `longer` rows (0–2). The
  `ulp_control` also fails, and held-out `ref13` fails `longer`/r0
  (reference-internal bimodality). The long-suite 2048-token failure is
  unlocalized. Evidence: `statistical-gate-20261011/final/`.
- Qualification rule going forward: gate v2
  (`docs/design/exl3-statistical-gate-v2.md`, with Amendments 1–2), pending
  final operator sign-off and evaluator verification; not yet collected.
  Fixed N=15; no adaptive top-ups. Diagnostic scoring in this plan
  still uses the frozen v1 ensemble/metrics as fixed diagnostics.
- Attributed gated-norm difference and correction:
  `q2-norm-ablation-20261012/analysis-20261012.md`. v24 = rsqrt form +
  reciprocal-multiply combine. It reproduces the reference on **7/7
  differing elements across 2 independent (rank × prompt) groups**; the 16
  fixtures are nested prefixes, and 2 of the 4 groups do not differ at all.
  v24 introduced no new differences elsewhere. v24 was selected after seeing
  these fixtures; other variant pairs were not run. The replay used NVRTC
  `--fmad=false`, and production flags are unverified. Fixture-local only.
- Runtime-bound fixtures and bindings: `q2-norm-replay-20261011/`
  (`fixtures/`, `common/`, `replay-ptx/`).
- Code-path audit gap list G1–G9 and discrepancies D1–D5:
  `q2-codepath-audit-20261012/comparison-report.md` §4–§5, corrected by
  the 2026-10-12 reconciliation after independent review.
  The archived report is preserved; its missing-config, disputed-head-padding,
  never-replayed and short-live-latent claims are superseded there.
  The operator-requested Luna source review and parent reconciliation are in
  [the pinned vLLM reference contract](../design/exl3-vllm-reference-contract.md).
  Use installed reference source/overlays and bound receipts as authoritative
  for engine behavior; short-prefill cache effects remain in scope. Seed 42
  and fresh processes do not establish deterministic forward arithmetic.
- Frozen harness: `tests/exl3_validation/` (statistical gate +
  compare/parity protocol + the norm replay/ablation tooling).

## Preconditions from the offline review

- Ablation runner repaired (`9ff39728`, `c71bb557`; review OK-with-notes).
  The repaired harness has **not yet run on a GPU**. Both archived ablation
  runs used the pre-repair harness. Engagement controls exist for
  `attested`, `v0-copy` and `v-all` only; **`v24-minimal` has none**. Any
  new v24 claim needs a GPU rerun under the repaired harness with v24's own
  controls.
- Reproduce the baseline final-logit fingerprint under the diagnostic build
  before interpreting an intervention. Record disagreement with the frozen
  candidate separately; stage tracing does not establish qualification.
- Checkpoint config is available: `vllm-reference.json`'s `config` serializes
  with `json.dumps(config, indent=4)` (no newline) to the exact registered
  `config.json` SHA256 `00e4a3aad5a2348d5a0ef8c6a1d8e8a945e77e2a1fddc1d7ddd6560f29e07066`.
  It matches all 24 suite manifests for ref04–ref13 and cand02/cand03. Use its
  EXL3 layer map and constants; do not import the enclosing historical serve
  environment. Runtime overrides/provider dispatch still require binding.

## Experiment C1 — local o_proj before TP: CLOSED from native captures

Operator decision 2026-10-12 ("C1 re-scoped"). Answers come
from the native pre-reduce captures under `q2-c1-execution-20261012/` and
`q2-c1-binding-2026013/` (sic). See `ERRATUM-20261012.md` there.

- **Same-input projection error:** none. Native local outputs are
  byte-identical across engines on all 8 identical-input rows.
- **Propagation of the natural ≤1-ULP norm delta:**
  - Rank0 rows 4–7: 6 differing inputs, 0 differing outputs.
  - Rank1 rows 0–3: 1 differing input element, 109 differing outputs on
    that token only, max |Δ| 6.1e-5.
- **v24 / reference-exact input:** byte-identical to each other 32/32. The
  candidate would reproduce the reference local output (inferred).
- **FLAG:** the last answer is inferred from 8 identical-input rows of 2
  prompts, not measured. The candidate o_proj input precision is unattested.
  These answers are local to layer-0 o_proj and say nothing about downstream
  effect.
- Do not replay o_proj over a BF16 reconstruction. Both engines multiply
  EXL3 trellis weights: the reference uses an FP16-input `h.forward` for
  `rows <= 144`. The earlier replay harness is archived
  (`tests/exl3_validation/archive/c1-local-o-proj-replay/`). Any future
  o_proj replay must bind the trellis bytes and the runtime input-precision
  branch.

## Experiment C2 — downstream intervention on both ranks (NEXT)

- Fresh two-node controlled captures with a **norm-output substitution knob**
  (diagnostic-only, off by default; production-path neutrality proven per
  step-5 discipline): run A = baseline; run B = gated-norm output replaced by
  reference-exact (and separately v24-corrected) values at the stage output.
  Initial intervention is **layer 0 only**, both ranks, all token/head
  elements of each of the eight captured requests. Bind the substitution to
  exact request IDs, layer, rank, invocation and tensor hash; reject a missing
  or mismatched fixture. Keep later layers unchanged in this experiment.
  An all-layer arithmetic correction is a separate experiment and cannot be
  inferred from this layer-0 substitution.
- Trace downstream on both ranks: local o_proj → reduced attention → FFN
  inputs → layer-3 router logits/IDs/membership → MoE post-TP outputs →
  final logits at the gate rows.
- Compare B−A deltas against the registered failing rows (`longer` 0–2,
  `arithmetic`, and the ref13 row-0 fingerprint over vocab
  {2619, 8062, 29296, 220, 576}).
- Diagnostic scoring fixed before execution: use `statistical_gate.row_metrics`
  against **every** member of the frozen ref04–ref12 ensemble. For each of the
  eight rows, report A and B mean symmetric KL, nearest-reference symmetric
  KL, continuation CE error where defined, and top-10 miss. Define the primary
  row delta as mean symmetric KL(B, refs) minus mean symmetric KL(A, refs):
  negative = closer, positive = farther, zero = no metric change; report raw
  logit byte equality separately. A mixed row-sign result stays mixed. Do not
  select a favorable reference, row or vocab subset; the five-token fingerprint
  is explanatory only. Record per-row catastrophic checks against the frozen
  limits, but do not call this short diagnostic slice a Q2 gate pass.
- One-variable A/B only; both ranks; identical serve flags/requests; fresh
  attempt names. Repeat the candidate baseline and treatment to confirm each
  is byte-deterministic; if either varies, report the variation and retain
  causality as unresolved until it is controlled. Preserve every attempt.
- The 2048-token long-suite slice required an explicit operator scope
  extension; **granted 2026-10-12**. It is authorized here
  subject to every other constraint in this plan.
- **First C2 measurement (discharges the C1 flag):** B's layer-0 local
  (pre-reduce) o_proj output must be byte-identical to the reference's
  captured local output on the same rows. That directly measures the
  corrected-input answer C1 only inferred. If it is not identical, record
  the difference and re-open C1 before interpreting downstream deltas.
- Because C1 found zero same-input o_proj error, a nonzero B−A at reduced
  attention or later is attributable to the substituted norm values
  propagating, not to o_proj arithmetic. That attribution still needs B's
  baseline determinism to be shown.
- Implementation notes:
  - The substitution knob is new diagnostic code: off by default, and
    neutral when off (final-logit archive byte-identical to the frozen
    baseline, as in the C1 neutrality check).
  - Captures reuse `Exl3FfnRouter` records (atomic metadata commit,
    `651ff083`).
  - Do not reuse the archived C1 o_proj binding/receipt code.

## Fallback-suspect ordering (folded from the audit gap list)

If C2 leaves residual failure, first inspect the **longer L0→L1 boundary**:
all four longer prefixes on both ranks have matching L0 reduced-attention and
FFN outputs, then differing L1 FFN inputs (464-pair analyzer rerun). Capture
the complete mHC residual streams and L1 attention input before attributing
this to L1 KDA, then bisect L1 core → gated norm → local/reduced o_proj → FFN
mHC input. Matching L0 branch outputs alone do not bind the residual streams.
This is the existing L1 escalation, not a claim that the full L0 hidden state
is already proven equal.
Bind `(x, residual, post, comb)`, actual local token count, sequence-parallel
setting, DeepGEMM/TileLang choice and split geometry. The ≤16-row fused path
uses unrounded FP32 residuals for prenorm, while the larger path materializes
BF16 first; the contraction denominator also uses pre-round FP32 squares.

Then continue through the audit gaps below. G8/G9 provenance is a prerequisite
for each affected replay, not work deferred until all numerical experiments:

1. **G2 — final RMSNorm + 1-row lm_head replay.** Bind final norm and rebind
   the existing head component replay to the failing-prefix inputs and current
   workers/embedded PTX. D4's padding question is resolved for the installed
   source and prior component run: `pad=false`, actual rows=1 despite global
   min_rows=3. Reuse `arithmetic-audit/linear-head-m1-production13-rank{0,1}-results/`;
   those NVRTC-source replays used one captured rank-0 input for both logical
   ranks and do not qualify current full-model output.
2. **G3 — router binding + same-input router-logit comparison + tie-rule
   evidence.** Reuse `arithmetic-audit/router-coherent131-controls-attempt2/`
   and extend its M1/M5/M10/M128 controls to the failing inputs. Checkpoint
   router dtype is float32; prior captured logits are BF16-representable.
   Bind the actual GEMM output/cast and sigmoid dtype, then check membership,
   order and ties separately. No stable top-k tie-break is promised by source.
3. **G4 — rebind/extend existing thin/fat expert replays.** Reuse the
   production thin/fat component controls, including
   `arithmetic-audit/fat178-production19-rank{0,1}-results/` (32/33 rows).
   Capture local expert IDs, FP16-rounded route weights, sorted token
   permutation, per-expert counts and routed FP32 accumulator before BF16
   conversion. Installed reference casts route weights to FP16 before sorting.
   Archived L3 longer IDs have max occupancy 21/22/22/22 (reference) and
   22/23/23/23 (candidate), identical across ranks: no expert exceeds cap32
   in these captures. Focus their same-input replay on the thin path; bind
   occupancy anew for other layers/long contexts before choosing thin or fat
   activation arithmetic. Atomic-order repeat probes remain separate.
4. **G1 — layers 4–44 hidden-state bisection captures** on both engines
   (operator scope extension granted 2026-10-12).
5. **G5 — effective cache/attention-path equivalence.** Reference `auto`
   resolves to `fp8_ds_mla`; candidate `fp8_g128` was selected to match that
   format (registered gate addendum). Different flag names do not establish
   dtype asymmetry. The logged SM120 backend disables dense MHA prefill and
   uses packed-cache MQA even for fresh short prompts. Bind actual writes/reads
   and QDQ placement for the short failing cases too; the merged report's
   live-BF16-latent assumption is superseded. Extend cache round-trip tracing
   to the 2048-token long suite under the operator scope extension granted
  2026-10-12.
6. **G6/G7 — mHC ≤16-row path-branch capture at a representative layer;
   MLA/indexer replay.** Checkpoint `index_topk=2048`, `index_kpool=4` and
   NoPE are recovered. Installed selector drops one selected pool in its
   ranked branch, while Atlas allows 2051 index slots. Compare actual key IDs,
   valid counts and positions beyond the cutoff; isolated prefills through
   2048 take the all-causal selector and bypass that pool-drop branch. A case
   named 2047/2048 can have teacher-forced continuation prefixes above 2048,
   so classify each actual prefix/chunk, not the case name. This semantic lead
   does not itself explain a row that stayed in the short branch.
7. **G8 — bind resolved runtime configuration/overrides** using the recovered
   exact checkpoint config; **G9 — compiled-binary ↔ source binding** (EXL3
   extension, `_C.*`, FlashInfer, cuBLAS). Reuse prior hashes and algorithm
   receipts, then establish identity with the current workers.

The ref13 held-out investigation is complete. Verdict: reference-internal
bimodality at `longer`/r0 (3/15 draws, 6 byte classes), with the mechanism
unidentified. It led to gate v2. A probe of the discrete runtime choice
(autotune/plan, NCCL algorithm, routing tie) is recommended before v2
collection (gate v2 Amendment 1, A1.5).

## Neighboring types and model/configuration coverage

This section governs regression scope for C2 and G1–G9 repairs. Derive the
current inventory from Rust enums, loader/dispatch predicates, kernel exports,
`kernels/gb10/*/MODEL.toml` and the model × quant matrix in
[`../GB10_DEPLOYMENT_GUIDE.md`](../GB10_DEPLOYMENT_GUIDE.md). An enum variant or
factory arm alone does not establish a supported model/configuration pairing.
Trace callers of every changed shared component through load, prefill, decode
and verification paths; a branch unused by this checkpoint may be active for
another supported model. Kernel source symlinks and common build flags count
as shared dependencies. Include string-valued and boolean configuration
policies as well as enum arms (for example `gdn_norm_sigmoid`,
`output_gate_type` and `scoring_func`); enum exhaustiveness alone is insufficient.

For each reachable combination, record model/checkpoint, effective config,
Rust variant and branch predicate, kernel/provider, reference counterpart,
fixture/test, evidence identity and coverage status. Classify it as:

- **Active GLM profile:** preserve the frozen Q2 contract and required controls.
- **Supported neighbor:** validate affected behavior using that model's own
  contract, including branches off in GLM. Record evidence or an explicit gap.
- **Implemented but unqualified:** record code reachability and the missing
  qualification; do not promote it from enum presence or compilation.
- **Unsupported combination:** verify explicit rejection before unsafe
  allocation/launch or state mutation. A hypothetical model is a design/test
  consideration, not authorization to enable a new serving path.

Use these axes as the starting inventory, then expand actual reachable enum
arms. The table specifies coverage obligations, not claims that every listed
combination is supported by Atlas or vLLM.

| Axis / source of truth | Neighbors and boundaries to cover | Required checks |
| --- | --- | --- |
| Tensor/diagnostic dtype: `atlas-core/src/dtype.rs`, runtime `WeightDtype`, `DiagnosticDtype`, EXL3 dense dtype | BF16/F16/F32 storage, compute and transport separately; signed integer IDs versus floats; mixed precision and rank-local output types | Item sizes/strides/endianness, overflow-safe byte counts, finite-value policy, rounding/FTZ/saturation and cast order; signed-zero/near-zero ULP controls; reject wrong dtype metadata. F32 transport is not proof of F32 computation. |
| Weight format and GEMM: loaders, weight maps, EXL3 packed numerics | EXL3 versus FP8/NVFP4/dense/PackedQ2 paths, mixed checkpoint formats and packed bit widths; GEMV/GEMM, reconstructed/packed, padded/unpadded, classic/Lt and compute modes | Detect checkpoint encoding, resident representation and compute dtype separately; bind shards/scales and provider/algorithm. Classify PackedQ2 and other code-only neighbors explicitly; preserve model-specific geometry and reject unsupported widths. |
| Normalization and gating: layer wrappers and kernel dispatch | RMSNorm/LayerNorm; sigmoid versus SiLU gate; affine/bias/residual/prenorm variants where implemented; gated/ungated, epsilons, head widths, contiguous/strided and in-place inputs | Same-input replay per supported variant; parameter and dispatch binding; alias/guard controls; RMS sum versus mean and numerator/denominator rounding. Never apply the sigmoid-gated v24 correction blindly to plain RMSNorm or SiLU gates. |
| Dense/shared/routed FFN: `FfnActivation`, `Exl3FfnActivation` and numerical policies | SiLU/GeLU, unclamped/clamped, pre-/post-activation clamp, shared expert present/absent, heterogeneous intermediate sizes | Match checkpoint activation and clamp policy, exercise clamp boundaries and both sides of signed inputs, merged/separate projection geometry and FP16/BF16 intermediates; do not copy thin EXL3 semantics into dense/shared or GeGLU paths. |
| Routing and expert execution: router policies and local expert mapping | Sigmoid/softmax and implemented `sqrtsoftplus` policy; correction bias on/off, grouping, top-k/scaling/normalization; ties/order/membership; thin/fat, absent/shared/routed FFN, empty and heavily loaded experts | Capture score dtype, unbiased weights, casts, local/global IDs, token permutation and counts; test occupancy cap−1/cap/cap+1 and mixed occupancy. Distinguish integer ticket/count atomics from floating-point accumulation; bind expert mapping under TP/EP. Classify code-only policies separately from supported configurations. |
| Attention, position and residual architecture: loader and layer-type schedule | KDA/GDN/Mamba-2 versus standard MHA/GQA and MLA; full/sliding/sparse attention; NoPE/RoPE and vision position variants; mHC versus ordinary residuals | Compare the appropriate equation/state transition, masks, scaling, positions and latent/head layout; capture complete recurrent/conv/mHC state. Keep learned contraction versus stream averaging distinct. Include shape-selected fused/unfused rounding. |
| Cache format: `KvCacheDtype`, `SparseIndexCacheDtype`, cache catalog/dispatch | BF16, per-tensor FP8, group-128 FP8, NVFP4, implemented Turbo/asymmetric K/V families; BF16/FP8 sparse index; recurrent/conv state distinct from KV | Enumerate all reachable supported enum arms, write/read/QDQ and scale layouts, K/V asymmetry, rotated bases, zero/saturated values and tail packing; bind logical-to-physical indices. A CLI spelling or main-KV dtype does not identify the indexer or recurrent-state dtype. Unsupported model/cache pairs must reject. |
| Shapes and dispatch thresholds: actual predicates | One-row head/decode, M=2 versus min-padding=3, 15/16/17 mHC rows, 31/32/33 expert occupancy; tile/vector/head/group boundaries; 127/128/129 prefill rows; 2047/2048/2049 selector boundary | For every reached threshold test below/equal/above using actual local rows/counts, plus non-multiple tails and capacity limits. Derive thresholds from code/config; listed GLM values are examples. Zero-sized work must safely no-op or reject per contract. |
| State lifetime and execution mode: scheduler/model/runtime | Fresh/chunked prefill, one-token final chunk, persistent decode and mixed requests; prefix reuse/eviction; eager/graphs; speculative/MTP verification and rollback where supported | Compare correct cache/history at boundaries; isolate sequences, stream ownership, graph addresses/capacity and scratch reuse; test cancellation/retirement and invalid reuse. Teacher-forced re-prefill cannot qualify persistent state or speculation. |
| Topology, backend and output boundary | Single-rank/TP/EP/SP and supported combinations; CUDA capability/provider variants; head output/logit scale, gather/truncation, sampler and API paths | Compare local and reduced tensors only at matching boundaries on every rank; verify output ownership/vocabulary layout and effective flags. Capture pre-sampler logits separately from sampled IDs, seeded behavior, masks and text. Unsupported topology/backend combinations must reject. |

Source paths in the table are under `crates/` unless stated otherwise. Useful
entry points are `spark-model/src/factory.rs`,
`spark-model/src/factory/exl3_preflight.rs`, `spark-model/src/weight_loader/`,
`spark-model/src/layers/`, `spark-runtime/src/kv_cache.rs`,
`spark-runtime/src/kv_cache/sparse_index.rs` and `atlas-core/src/capabilities.rs`.
Derive semantic architecture from the actual layer schedule as well as broad
capability enums; differently named models can share code and one model can
mix several layer families.
Immediate norm neighbors are GLM KDA sigmoid, Qwen GDN SiLU/configurable
sigmoid (`ModelConfig.gdn_norm_sigmoid`) and Nemotron Mamba-2 SiLU. Verify
their normalization/grouping and gate placement separately. Likewise,
`QuantFormat`, `QuantWeight`, `Glm5Projection` and `Bf16GemmNumerics` describe
different boundaries; do not force them into one assumed universal dtype enum.

Select representative checkpoints from the current support matrix for the
shared paths actually affected: Qwen dense and MoE/GDN siblings, Qwen3-Next
and VL where reached, Nemotron Mamba-2, Mistral MLA, Gemma GeGLU and
sliding/full attention, and MiniMax sigmoid MoE/EP. Respect each checkpoint's
actual activation and existing BF16-KV-only and topology restrictions. These
are coverage categories, not a request to boot every model or invent a new
model × quant × cache Cartesian product.
Registered factory branches such as `qwen4_exp`, and configuration policies
such as DeepSeek `sqrtsoftplus`, require their own support/evidence status;
factory registration is not an addition to the deployment support matrix.

Coverage selection is finite and reviewable:

1. Cover every affected supported branch at least once, plus every changed
   admission/rejection predicate and every numerical/layout threshold it uses.
2. Add pairwise combinations for independent axes and targeted higher-order
   cases for coupled paths: dtype × writer × reader × phase, activation ×
   occupancy × expert path, topology × reduction boundary × shard layout,
   graph/speculation × cache state × scratch lifetime. Pairwise coverage alone
   does not prove these interactions.
3. Use production parsing/dispatch and real kernels for numerical acceptance.
   CPU fixtures can prove selection/rejection/serialization; they cannot prove
   GPU arithmetic. Minimize mocks and avoid tests that only restate source.
4. For a supported neighbor with no matching vLLM implementation, name that
   limitation and use an independent operator oracle, mathematical invariant,
   or before/after non-regression evidence appropriate to its contract. Label
   each proof's scope; neither self-comparison nor approximate math proves
   cross-engine parity. Predeclare tolerances for new profiles, leaving Q2's
   frozen tolerances and controls unchanged.
5. Reuse current receipts only after their source, binary, input and config
   identities match the claim. Report missing fixtures/hardware as untested.
   A GLM pass cannot close another model's row in the coverage inventory.

## Repeatable Rust/vLLM/source/log review procedure

Use this procedure after an implementation change or new evidence. Review
source and existing artifacts first; follow existing authorization before
starting live experiments. Each iteration updates the same evidence ledger,
checklist and this plan, preserving failed runs and superseded conclusions.

1. **Freeze the review target.** Read `AGENTS.md`, `CONTRIBUTING.md`, the
   measurement-discipline skill, this plan and the current handoff. Record
   Git commit plus dirty diff, checkpoint revision/config hash, Rust binary,
   embedded PTX, image, installed vLLM/plugin/extension hashes, hardware,
   flags/env, topology and exact request/capture identities. Resolve effective
   config, overrides and dispatch from receipts; archived source alone is not
   proof of the running implementation. Search existing archives before
   declaring a config, component replay or log missing.
2. **Map implementation and neighbors.** Trace request → loader/resident
   weights → layer/state operations → local/collective outputs → final norm
   and head → captured logits → sampler. Include Rust/CUDA/FFI, unsafe pointer
   and lifetime boundaries, scratch ownership, I/O boundaries and every
   reachable neighbor in the matrix above. Assign an evidence status to each
   stage/branch; keep an explicit list of uninspected paths.
3. **Compare executed semantics.** Independently inventory Atlas's dispatch
   and the installed vLLM path, then reconcile disagreements using exact
   source lines, hashes, effective config and runtime logs. Compare operands,
   equations, layouts, casts/rounding/clamps, reductions, caching, state and
   collectives. Treat Python wrappers, patched plugins, compiled extensions,
   library algorithms and JIT products as distinct bindings. A similar API
   name is insufficient; a reference bug or an unsupported reference branch
   is not a prescription to change Atlas.
4. **Audit evidence and harnesses.** Verify hashes, dtype/shape/stride, request
   order, layer/rank/invocation and pre-/post-TP mapping. Verify original
   logs, raw arrays, failed attempts, controls and exit status agree with
   summaries. Reject missing/failed controls and incompatible fingerprints;
   use each engine's self-replay before same-input cross-comparison. Separate
   source inference, archived observation, fresh measurement and unknowns.
5. **Localize and establish causality.** Find the earliest comparable
   divergence while carrying complete residual/cache/recurrent state. Bind
   operands/constants and compiled dispatch, vary one numerical axis, then
   intervene downstream with explicit site, rows and metrics. Retain null,
   mixed and adverse results. Check reference variation independently; do not
   blame the model, dismiss a failure as noise or move a gate without proof.
   For this campaign execute C2 and the revised fallbacks above.
6. **Verify correctness at the claimed scope.** Run relevant existing CPU
   tests and review negative controls/admission errors. Proposed fixes need a
   reproducing regression before implementation and a passing regression
   afterward, plus the supported-neighbor slice. Apply required repository
   checks and real-model validation when code changes: GPU-free/stub checks
   do not establish runtime parity. Production changes require a clean image,
   current source/binary binding and fresh frozen Q2/required controls; use
   `tests/single_gpu_suite.py` or `tests/run_all_models.py` for relevant shared
   regressions with explicit model scope. Preserve host/service ownership and
   previously scoped experiment limits.
7. **Evaluate performance separately.** Follow the protocol below only for
   a clearly named configuration with its correctness/quality status recorded.
   Tracing an unqualified candidate may identify bottlenecks but cannot qualify
   it as a faster correct implementation. Propose optimizations from measured
   work, then repeat correctness and neighboring-branch checks after changes.
8. **Publish an evidence-backed verdict.** Report prioritized defects with
   path/line, triggering configuration, impact and reproducer; a stage/branch
   coverage table with pass/fail/untested/rejected statuses; raw artifact
   identities; source-versus-runtime gaps; and the smallest next experiments.
   State correctness scope and performance scope separately. "No finding in
   reviewed paths" is not proof for unreviewed branches. Reconcile new results
   into the experiment log and decision record without
   overwriting original evidence or marking unexecuted work complete.

## Performance and the meaning of optimal

Define the objective before measuring: latency (TTFT/TPOT and tails), throughput
at a named admission/concurrency level, memory/peak allocation and a fixed
quality/correctness contract. Report prefill, decode and end-to-end intervals
separately. Record prompt/output lengths, content, actual generated tokens,
reasoning/speculation policy, seeds, warm/cold/cache state, synchronization,
clocks and competing work on both ranks. Do not compare a traced run to an
untraced baseline or mix kernel timing with HTTP wall time.

Use representative authorized workloads, at least three matched repeats for
reported comparisons and independent timing confirmation. Trace both ranks to
identify launch gaps, synchronization/collectives, memory traffic, allocations,
occupancy and compute work; measure final rates without diagnostic tracing.
Identify alternatives from the bound dispatch/shape space (for example an
eligible fused/unfused or GEMM strategy), changing one factor at a time while
holding precision, quality and workload fixed. Different precision or serving
features require a separately labeled quality/performance tradeoff. A vLLM
end-to-end comparison and a same-operator microbenchmark answer different
questions and should be reported separately.

Acceptance requires a reproducible improvement at the declared objective,
the relevant correctness/behavior checks, unaffected-neighbor evidence and no
unexplained regression in other tracked objectives. Retain all candidates and
report uncertainty; a noisy or single-shot result is preliminary. A roofline
estimate is a hypothesis for investigation, not a measured bound or proof.
Use **best observed among named tested alternatives under configuration F**
when supported. Without exhaustive admissible alternatives or a justified
bound, **global optimality is unproven**. With no suitable timing evidence,
report performance as unmeasured rather than infer it from code appearance,
kernel count, reference parity or compilation.

## Reusable review prompt

```text
Review the current Atlas implementation and Q2 investigation end to end.
Read AGENTS.md, CONTRIBUTING.md and docs/plans/q2-norm-causality-plan.md. Apply the
measurement-discipline skill. Follow the plan's repeatable review procedure
and neighboring-types coverage matrix; use independent subagent reviews for
Atlas implementation, vLLM semantics and evidence/harness validity.

Reference source: /home/colum/Workspace/vllm-reference-src/
Evidence root: /home/colum/Workspace/atlas-exl3-validation-20261010/
Discover the current revisions and artifacts; do not assume previous source,
binary, plugin, configuration or host-state bindings still hold.

Trace the implemented Rust through CUDA/FFI, loaders, effective configuration
and actual runtime dispatch. Independently trace the pinned vLLM source and
installed patches. Compare executed operations, dtypes, arithmetic order,
layouts, cache/state transitions, routing and collectives. Separate semantic
requirements from reference-specific numerical policies and reference bugs.

Cover neighboring supported models/configurations that reach shared code,
including norm/gate/activation variants, weight and cache formats, attention
and recurrent types, routing policies, shape thresholds, execution modes and
topologies. Derive reachability from source. Distinguish supported neighbors,
implemented-but-unqualified branches and unsupported configurations requiring
explicit rejection. Select bounded branch/boundary and coupled-axis coverage;
do not infer coverage from enum presence or one GLM result.

Review raw logs, manifests, tensors and prior reports. Verify hashes, rank,
invocation, identical replay inputs, controls, failure propagation and complete
state. Reconcile contradictory reports, reuse valid prior evidence and find
the earliest defensible divergence. Separate observation, hypothesis, causal
proof and qualification. Preserve the frozen gate, failures and thresholds.

Run applicable read-only/CPU checks. This request authorizes review and updates
to existing docs, not production fixes or new GPU/service operations. Record
the smallest regression test and experiment needed for each unresolved claim.
Evaluate performance separately using matched repeated measurements and named
alternatives. Report best observed performance within a stated workload;
if evidence is insufficient, leave correctness or optimality unproven.

Update existing plans, diagnostic log, decisions and handoff with evidence,
corrections and unchecked follow-ups. Return prioritized findings with source
locations, trigger, impact and reproducer; a pass/fail/untested/rejected coverage
matrix; separate correctness and performance verdicts; and next experiments.
Do not mark planned work complete or alter historical evidence.
```

## Constraints (inherited, not renegotiated)

The v1 gate and its evidence stay frozen as the historical record.
Qualification uses gate v2 (with Amendment 1) only after operator sign-off.
No rerun selection. Preserve all failing runs and burned names. Use fresh
attempt names and capture roots per the runbook. Every production change
carries regression tests, a clean image rebuild, fingerprints and a fresh
registered Q2 qualification run with all controls. A production v24 change
must additionally:
- prove v24 under the production toolchain/flags (the ablation used NVRTC
  `--fmad=false`);
- carry v24's own engagement controls;
- cover the neighboring-types matrix.

## References (external to this repo)

- Evidence root: `/home/colum/Workspace/atlas-exl3-validation-20261010/`
  (`statistical-gate-20261011/`, `q2-diagnostic-20261011/`,
  `q2-norm-replay-20261011/`, `q2-norm-ablation-20261012/`,
  `q2-codepath-audit-20261012/`, `q2-c1-execution-20261012/`,
  `q2-ref13-investigation-20261012/`, `q2-neighbor-matrix-20261012/`,
  `q2-gate-v2-20261012/`).
- Reference engine source (hash-pinned): `/home/colum/Workspace/vllm-reference-src/`
  + `/home/colum/Workspace/atlas-exl3-validation-20261010/arithmetic-audit/`.
- In-repo: `tests/exl3_validation/` (harnesses),
  `docs/design/exl3-statistical-gate.md` (frozen v1 gate),
  `docs/design/exl3-statistical-gate-v2.md` (v2 + Amendment 1, pending
  sign-off).
