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

### Semantic perturbation switches

Each must-fail perturbation is one value of `ATLAS_EXL3_GATE_PERTURBATION`
(at most one per run), honoured only by the native EXL3 GLM-5.3 loader
(`weight_loader/glm5/exl3_layers.rs`). It is parsed and validated once at
model load; an unknown value, a malformed layer, or a layer that is not a
routed MoE layer of the loaded model is a startup error. An active control
logs one `WARN` banner. Unset or empty means no perturbation. Each TP rank reads its own
environment, so every rank must be launched with the same value. Any set
value on a non-EXL3 model is refused at startup.

| Value | Effect |
|---|---|
| `mhc_post_1p9` | mHC post multiplier 2.0 → 1.9 at every native mHC site (MLA and KDA layers) |
| `router_scale_2p4` | routed scaling factor 2.5 → 2.4 on every routed MoE layer (requires configured 2.5) |
| `topk7_layer=<L>` | router top-k 8 → 7 on routed layer `L` only (requires configured top-8) |
| `drop_shared_layer=<L>` | shared expert skipped (not added) on routed layer `L` only |
| `bf16_routed_accum` | running routed-expert sum rounded to BF16 after each expert in `exl3_moe_combine_f32` |

`<L>` is a 0-indexed layer that is routed in the loaded config (for the
45-layer GLM-5.3 with three dense layers, 3..=44). `bf16_routed_accum` uses a
kernel flag that is zero on the reference path; at registration it had not
yet been exercised on a GPU.

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

## Collection restart (2026-10-11): instrumentation and single-row reader

The resumed diagnostic reproduced a separate failure with **both** prefill
flags present: short16 and the 2047/2048-token prefixes completed, then the
2049-token prefix failed at layer 3. Its one-token final chunk enters MLA
decode, whose required FP8 reader was incorrectly gated by the experimental
`ATLAS_GLM_SPARSE_DECODE_TC` flag. Both TP ranks exited. The durable record is
`statistical-gate-20261011/resume-diag01/diagnosis.json`, with both container
logs, identities, HTTP receipts and raw captures. This is diagnostic data.
The replacement candidate must initialize and select the existing FP8 reader
for ordinary serial inference; register its exact image/build identity before
collecting any new candidate samples. The serving flags remain unchanged.

The saved `ref01` and `ref02` runs are **diagnostic, not qualification samples**.
Their registered context-v4 command also enabled intermediate FFN/KDA/mHC and
head-input tensor captures. These copy GPU tensors to the CPU during forward
and do not meet the instrumentation restriction above. Identical returned
values do not establish that the hooks preserve atomic execution order.

For the fresh reference set, the only permitted observers are the existing
marker-gated **output logits collector** and the `serial128` **CPU chunk-position
observer** needed by the long-suite capture contract. Remove the environment
entries `ATLAS_EXL3_FFN_DUMP_DIR`, `ATLAS_EXL3_FFN_DUMP_LAYERS`,
`ATLAS_EXL3_UPSTREAM_DUMP_LAYERS`, `ATLAS_EXL3_MHC_DUMP_LAYERS`, and
`ATLAS_EXL3_CAPTURE_HEAD_INPUT`; setting a layer-list variable to `0` selects
layer zero and does not disable it. Bind the exact hook files and the effective
container environment in the fresh provenance before collection. Retain the
reference image, checkpoint, request order, serving flags and numerical rule.
“No hooked or instrumented builds” above therefore excludes stage tracing and
arithmetic modifications; these two explicitly registered collection observers
are required exceptions. Qualification applies to that observed profile, with
no assertion of equivalence to an unobserved reference process.

Start all nine reference samples and the held-out reference afresh. Never mix
the old instrumented runs into this set. Within each engine, every suite and
repeat must use the same effective environment and build identity. Neither
this collection correction nor the runtime repair changes the statistic,
catastrophic-row rule, sample count, or required controls.

The immutable collector-only command inputs are archived under
`statistical-gate-20261011/collector-only-profile/` (attempt names and host
capture directories are substituted at launch):

| Input | SHA256 |
|---|---|
| `ref-rank0-create-command.json` | `8afc990ca451dfc1ce45b26b7f45e017f138a7cdb15e39a5a933f04df01b3bcb` |
| `ref-rank1-create-command.json` | `a9a2e88325262de7f3ff73ce6aeb7b5140e2cd6731854edd05bdccde496e6eb3` |
| `sitecustomize.py` | `81771af5085bcd268576c49e911eed34d89616de9ac561196826e1ea8eadd53d` |
| `reference_chunk_trace.py` | `a92c2e642508595e21880725d9dcafe4481ca76e1639ac949f598d79c01ab081` |
| `ATLAS_EXL3_CHUNK_RUNNER_SHA256` | `f84255d75435e84f44972d3fd25e53447f9d4d2edd8bff4f8c19dfb793448415` |

Retain `PYTHONPATH=/opt/atlas-logits-patch` and the runner hash above. The
complete mounted-source audit, including both ranks' copies, is
`statistical-gate-20261011/reference-profile-audit/audit.json`, SHA256
`31363ff2ef1b4b400edcf63bdc5765d1a76b2ebfed574c9052298d28f9d6b7ec`.

### Replacement candidate identity (registered before requests)

The single-row FP8 reader repair is commit
`fb4f4b41de19a19badf155836cd8036b54e70280`. Its clean, no-cache build uses
Rust 1.97.1 and the pinned cuBLAS pair, with these immutable identities:

| Artifact | Identity |
|---|---|
| Source archive SHA256 | `8b6b47f14d5b7cefa354b970f24f95619576255d34b8ced063ee30a0eec48b13` |
| Image tag | `atlas-exl3-validation:src-8b6b47f14d5b-rust1.97.1` |
| Image ID | `sha256:e744b81d972b394cef6e6ec4a68f82c983b38b52747be6eab9dbc7498bc84b72` |
| Server binary SHA256 | `95b2a1d68c9127825d6a97e635d2c98e1f9606c245e3ae07560361a5664b9287` |
| Scratch checker SHA256 | `a8c6c5251321c902d50f35d3249692bf7cdc14089dfc5912181d8253e217f626` |

The full build manifest, source archive and build log are preserved in
`fp8decode-candidate-20261011/` under the validation evidence root. The serving
profile remains TP2/EP1/C1, context 4096, chunk 128, `fp8_g128`, KDA FLA reference
enabled, both sparse-prefill flags enabled, graphs disabled, no speculation.
The repair selects the existing FP8 decode reader without requiring the
experimental decode flag. It does not change CUDA arithmetic. No numerical
verdict follows from the build or this registration.
