# EXL3 statistical logit gate — pre-registration

Status: **pre-registered rule, no qualification data collected under it.**
Committed before any reference run it governs. Implementation:
`tests/exl3_validation/statistical_gate.py` (tests:
`test_statistical_gate.py`). Approved direction: user decision 2026-10-11
(option B, a pre-registered statistical gate replacing the unattainable
frozen gate); rule refined from an independent review (Fable 5.1).

## Why the frozen gate is replaced

The frozen short16 gate (allclose abs 0.15 + rel 0.02, KL 0.005, CE error
0.15, top-10, exact argmax/text) is failed by the **reference against
itself**: repeats differ by max abs up to ~3.5 and KL up to ~0.06
(`arithmetic-audit/reference-three-run-diagnosis.json`,
`reference-same-arithmetic/raw-repeat-and-stage-hashes.json`).

Root cause of reference nondeterminism: f32 CUDA atomics in the EXL3 thin MoE
combine (`out.index_add_(0, token_idx, down * scale)` in the reference
plugin) and the fat kernel's float4 `atomicAdd`. The first change appears at
layer-3 `routed-weighted-f32` by one ulp and spreads chaotically (route-ID
flips by layer ~17). The noise is row-specific and multimodal: some rows are
byte-stable across runs, others never are. Pinning the reference would not
make the frozen gate attainable: any engine whose reductions are not
bit-identical deviates by the same chaotic magnitude.

## Scope and limitation

This gate certifies that the candidate shows **no error above the
reference's own chaos floor**: its logits look like one more reference run.
It cannot prove sub-floor arithmetic correctness. Atlas's own layer-3
deviation (~0.02 at the layer-3 input) is ~10^5 times the reference's 1-ulp
self-perturbation, yet both land at the same logit distance. **Stage-level
byte oracles remain the primary numerical-correctness evidence** (e.g. the
KDA stage oracle; layers 0–2 byte-exact in situ).

## Data status

All reference and candidate logits collected before this commit (including
`kda-fla-candidate-20261010/` and the reference three-run set) are **tuning
data**; the rule was evaluated on them during review. Qualification requires
fresh reference and candidate collections after this commit.

A configuration confound was found: the earlier three-run diagnosis used the
`--max-model-len 1024 --max-num-batched-tokens 1024` reference server, while
the candidate comparison used the context-v4 (4096 / chunk 128) reference.
The two configurations differ by as much as reference self-noise. **Never mix
configurations or instrumented builds in one reference set.**

## Hypothesis

H0: the candidate's teacher-forced logits are exchangeable with one more
fresh reference run. Under H0 the gate's false-fail probability is at most
1/(N+1), independent of correlation between rows and metrics.

## Collection protocol

- **References:** N = 9 fresh containers (one serve process each), the
  unmodified reference image `glm53-flash-mikeroz:dt-carry20` with the
  registered context-v4 create command (4096 context, 128 prefill chunk,
  TP2/EP1, BF16 KV, eager, no speculation, no prefix caching). Requests are
  serial, through `collect_logits.py`. Suites: every short16 row and every
  long-prefix case (2047/2048/2051/2053). No hooked or instrumented builds, no
  in-process repeats, no other configuration. Long prefix is a different
  regime (many chunked invocations): record whether the fat MoE path engages
  before assuming the same noise source.
- **Candidate:** two fresh-server runs of the identical candidate image and
  flags. They must be byte-identical (hard).
- Each run is passed to the tool as comma-joined artifact directories, one
  per suite, in a fixed suite order.

## Hard gates (any failure fails)

1. Finite logits; provenance, cases, token fingerprint, vocabulary and request
   policy identical across all runs; the reference runs share one serve
   configuration and image, and the candidate runs share theirs.
2. Candidate runs byte-identical.
3. Negative controls: a token-swapped candidate manifest must be refused, and
   a cyclic vocabulary shift of the candidate must fail the gate.
4. Argmax and HTTP text per row: if all references agree, the candidate must
   match; otherwise it must be one of the observed reference values.
5. Catastrophic rows (below).

## Exchangeability test

Metrics per row r, between runs X and Y:

- m1 = KL(X‖Y) + KL(Y‖X) over the full vocabulary;
- m2 = |log p_X(t) − log p_Y(t)| at the continuation token t (only rows that
  have one, so the final row of each case is excluded);
- m3 = 1 − top-10 overlap.

max_abs is an extreme value over the logit tail. It is reported but never
gated.

With S = {R_1..R_N, A}, s_{r,m}(X) = mean over Y ∈ S∖{X} of d_m(X, Y). Rank
s_{r,m} across S (higher = farther; ties take the average rank).
T(X) = mean over all (r, m) of that rank.

**Gate: T(A) ≤ max_k T(R_k).** Report p = (1 + #{k : T(R_k) ≥ T(A)})/(N + 1).
There is one combined test; per-suite T is a diagnostic only. No
multiplicative margin.

## Catastrophic-row rule

For m1 and m2, every row must satisfy
min_k d(A, R_k) ≤ max(c · max_{j<k} d(R_j, R_k), floor_m), with c = 3,
floor_KL = 0.005, floor_CE = 0.15. The floors are the frozen thresholds,
retained as the level below which a deviation is never catastrophic. The rule
is verified leave-one-out on the references. If any reference fails it, the
gate reports the rule **unverifiable** and fails; the rule is never loosened.

## Controls (evaluated under the identical rule)

- **Must pass:** an (N+1)-th held-out reference evaluated as the candidate
  (primary, `--held-out-reference`); every leave-one-out reference; the
  candidate moved by one f32 ulp (`--ulp-control`), which is a null sample.
- **Must fail:** the negative controls above, plus these pre-declared semantic
  perturbations of the candidate: mHC post multiplier 2.0 → 1.9, router
  normalization 2.5 → 2.4, top-k 8 → 7 on one layer, the shared expert
  dropped on one layer, BF16-rounded routed accumulation.
- **Measured, not assumed:** Atlas with `ATLAS_KDA_FLA_REFERENCE=0`, reported
  as the empirical minimum detectable effect.

## Secondary record

The frozen-threshold `compare_logits.py` report against each reference,
candidate determinism and reference self-repeat statistics are kept in the
report for continuity. They do not decide the gate.

## Addendum before collection (2026-10-11): candidate KV profile

The registered reference create command requests `kv_cache_dtype=auto` (BF16),
but its log shows `Using fp8_ds_mla KV cache format for
FLASHINFER_MLA_SPARSE_SM120 backend`: the reference stores the MLA latent as
E4M3 with one F32 scale (amax/448) per 128 dimensions and attends in latent
space. Atlas's BF16 latent was its first own divergence (layer-3 core
attention output; layers 0–2 byte-exact). Rounding the latent through that
format reproduced the reference's single-key rows (1 ulp), and Atlas's
existing `--kv-cache-dtype fp8_g128` cache implements the same format.
Therefore the **candidate profile is `--kv-cache-dtype fp8_g128`** (with
`ATLAS_KDA_FLA_REFERENCE=1`), image `sha256:eb972c10…` (source `00ffffa1…`,
commit `4adc3138`), otherwise the registered launch flags. The reference
create command is unchanged. The execution map keeps the requested
`kv_dtype: bf16` on both sides; the effective cache formats are recorded in
the provenance adjuncts. Multi-key rows still differ at implementation level
(p50 1–2 ulp at layer 3), which is the reorder class this gate tolerates.
Evidence: `kda-fla-candidate-20261010/{ref-attn3-capture,mla3b-capture,fp8-mla}`.

Second pre-collection note (2026-10-11): with `fp8_g128`, Atlas's sparse
(> 2047-token) MLA prefill has no fallback reader and fails closed unless
`ATLAS_GLM_SPARSE_PREFILL_TC=1` and `ATLAS_GLM_SPARSE_PREFILL_KV_REUSE=1` are set
(the fp8_g128 K=V tensor-core kernel). Both are part of the candidate profile.
A trial candidate run without them returned HTTP 500 on the first 2048-token
prefix and the TP group terminated; that run is discarded and recorded as a
robustness defect (one request error stops the server).
