# EXL3 statistical logit gate v2 — pre-registration

Status: **Pre-registration and Amendments 1–2 remain the governing rule.
Qualification collection completed 2026-10-14; corrected offline evaluation
is UNVERIFIABLE because the reference set fails H3/H4 leave-one-out checks.
No threshold or metric changed. The initial frozen evaluator returned
INCOMPLETE due to a held-out identity-validation bug; the correction and
outcome are recorded in the
[qualification report](exl3-statistical-gate-v2-qualification-20261014.md)
and [verification report](exl3-statistical-gate-v2-verification-20261012.md).**
Authored 2026-10-12 from reference-side data only, before any collection it
governs. Operator decision: `decisions.md` 2026-10-12 ("full v2
pre-registration authorized"). The v1 rule
([exl3-statistical-gate.md](exl3-statistical-gate.md)) is untouched and stays
the historical registered rule. Calibration evidence and script:
`/home/colum/Workspace/atlas-exl3-validation-20261010/q2-gate-v2-20261012/`
(`calibrate_v2.py`, `calibration.json`, `summary.md`), run against the 15
archived reference draws and the registered harness snapshot (hashes in the
ref13 investigation report §2).

## Why v2 exists: the v1 control failed

The v1 held-out control ("an (N+1)-th reference evaluated as the candidate
must pass") failed for **3 of 5 fresh reference draws** at N = 9
(`q2-ref13-investigation-20261012/report.md`, §6). Root cause, reference-side
and reproduced: at short16 `longer` row 0 the reference engine takes **two
numerically separated values** — the 12-draw ensemble mode (within-mode
sym-KL ≤ 0.0022) and the H-mode (3 of 15 draws: ref13, ref13rep2,
ref13rep3, byte-identical to each other, ≈ 0.056 (up to 0.067) from the
ensemble-mode draws). The v1 catastrophic floor (sym-KL 0.005) sits **below** the
reference's own cross-mode distance, and a 9-draw ensemble has a 13.4 %
chance (0.8^9 at the observed mode rate 3/15) of containing no H-mode draw
at all — which is what happened: the ref04–ref12 ensemble missed the mode,
so its measured envelope (3 × 0.00132 → floor 0.005) could not admit a
draw that landed in it. Separately, the exchangeability-rank bar (max of 9
near-tied T values) missed three fresh held-outs by 0.034–0.057.

v2's operative repair is **fixed N = 15** (with H = 3 held-outs and an explicit
leave-one-out rank exemption), so that the ensemble fills the bar and covers
demonstrated modes. The per-row envelope (H3) is v1's catastrophic-row
formula unchanged (c = 3, same floors, same leave-one-out check), stated for
every row; it widens as N grows because it is 3 × the measured spread. v2
adds a **mode-membership rule** (H4) so that a widened envelope cannot become
a free pass at bimodal rows. The v1 diagnosis is restated in Amendment 1:
v1 was **underpowered at N = 9 under a bimodal row**, with session drift not
excluded — not structurally unpassable.

## Data status and authoring discipline

Every number used to set v2's constants (G_MIN, K_MIN, C_RHO, c, floors, N,
H) comes from the 15 reference draws (ref04–ref13 archived +
ref13rep1–ref13rep5 fresh, collector-only profile, one fingerprint) and the
archived reference-side held-out reports. **No candidate artifact or
candidate result file was read.** The authors did know the v1 candidate
verdict (FAIL, rank-dominated); the design is therefore not blind to that
fact, although no candidate number entered any constant. All 15 archived
draws are v2 **tuning
data** and may not be reused as v2 qualification samples. Qualification
requires the fresh collection specified below.

## Hypothesis

H0: the candidate's teacher-forced logits are exchangeable with one more
fresh reference run, at the resolution of the measured per-row chaos
envelope and the measured reference modes. For the fixed, pre-designated
N = 15 ensemble, exchangeability of all N+1 runs and symmetric scoring imply
a marginal rank-only false-fail bound of 1/(N+1) = 1/16. This is not a bound
on the whole gate or conditional on passing reference-composition/held-out
checks. H1–H4 can reject additional runs. Three held-outs are operational
controls; passing does not prove exchangeability or establish a campaign
false-reject rate. Amendment 2 keeps N fixed rather than selecting it from
reference outcomes.

## Metrics (unchanged from v1)

Per row r, between runs X and Y: m1 = sym-KL(X‖Y) + sym-KL(Y‖X) over the
full vocabulary; m2 = |log p_X(t) − log p_Y(t)| at the continuation token t
(absent on final rows); m3 = 1 − top-10 overlap. Items i = (r, m); rows are
every row of every case of both suites (short16 and long); gated metrics are
sym_kl and ce_error (= m2). max_abs is reported, never gated.

## The v2 rule (any failure fails)

**H1. Hard gates (v1 #1–#3, unchanged).** Finite logits; identity,
provenance, serve configuration, environment and build identity identical
across all runs of an engine (v1 `validate_runs`); the two candidate runs
byte-identical; the token-swapped candidate manifest refused and the cyclic
vocabulary shift of the candidate failing the gate.

**H2. Discrete rules (v1 #4, unchanged).** Argmax and HTTP text per row: if
all references agree the candidate must match; otherwise it must be one of
the observed reference values.

**H3. All-row chaos envelope (v2 replacement for the v1 catastrophic-row
rule; applies to every row of both suites).** With N = 15 references
R_1..R_N and candidate A, for every gated item i = (r, m):

- measured chaos spread: S_i = max_{j<k} d_i(R_j, R_k);
- envelope: E_i = max(3 · S_i, floor_m), with floor_sym = 0.005,
  floor_ce = 0.15 (the frozen thresholds, kept as the level below which a
  deviation is never catastrophic);
- pass condition: min_k d_i(A, R_k) ≤ E_i;
- **leave-one-out verifiability**: for every reference k,
  min_{j≠k} d_i(R_k, R_j) ≤ E_i^{(−k)} computed from the other N−1
  references. If any reference violates it, the gate reports the rule
  **unverifiable** and fails. The rule is never loosened.

Calibration (reference draws, `calibration.json → D_envelopes`): the v1
floor 0.005 governs 4 of 32 sym-KL rows; measured chaos raises the envelope
on the other 28 rows, from 0.022 (short16/longer/r1) to 1.071 (long
2051/r1). At `longer`/r0 the
15-draw envelope is 0.202 — it contains the H-mode by construction because
the mode is present in the measuring set.

**H4. Mode membership (new; applies to every row that demonstrates
multi-modality on the reference set).** Demonstration, on sym-KL distances
among the N references at row r:

1. sort the positive pairwise distances; let d_lo / d_hi be the pair
   straddling the largest consecutive ratio, G = d_hi / d_lo;
2. G ≥ G_MIN = 4.0 (calibrated: the largest gap ratio any non-mode row
   reaches is 18.89 (`short16/code/r3`), and every non-mode row reaching
   G_MIN fails step 3 — see below);
3. single-linkage at t = sqrt(d_lo · d_hi), joining distances ≤ t, must yield **exactly two**
   components M_1, M_2, and the separation
   sep = min cross distance / max within-component distance must satisfy
   sep ≥ K_MIN = 20.

**Zero-distance and tie semantics (Amendment 2; apply before the gap scan).**
Byte classes use the recorded canonical FP32 row bytes. One byte class is
not multimodal. Exactly two byte classes with positive cross-class sym-KL
and zero within-class distances demonstrate two modes directly: the classes
are the components, sep = +infinity, and rho_r = 0. This avoids dropping
all within-mode evidence when step 1 excludes zero distances. Distinct byte
classes with zero sym-KL do not demonstrate separated modes by this exception.
For the ordinary scan, fewer than two positive distances means no
demonstration; tied largest ratios select the first pair in ascending sorted
order. Define a singleton's within-component spread as zero. Positive cross
distance divided by zero within-component spread means +infinity; zero over
zero does not demonstrate separation. No NaN/undefined division is accepted.
These semantics apply unchanged to candidate and leave-one-out evaluations.

Calibration: genuine-mode separation is 25.6–42.4 in **every** subset of
size 11..15 of the 15 draws, and no non-mode row demonstrates in any of the
1 941 scanned subsets (archived: `E3_subset_stability`). The maximum
separation reached by non-mode rows is not archived; K_MIN = 20 is
supported by the demonstration count, not by a measured non-mode
separation margin.

For a demonstrated row, with component byte-class hashes (draws whose row
values are byte-identical share a class):

- center(M) = the most frequent byte-class in M (ties: lexicographically
  smallest hash); the center is that class's row value;
- rho_r = max over components of max_{R_i in M} d(R_i, center(M)) — the
  **measured within-mode radius** (row-level, shared by both modes);
- **membership**: a row value x passes iff
  min_M d(x, center(M)) ≤ C_RHO · rho_r, with **C_RHO = 2.0**
  (calibration: the largest leave-one-out coverage ratio over every subset
  size the rule is exercised on — 11 through 15, candidate-outside and
  member-inside checks — is 1.652, always at `short16/longer/r0`,
  draw ref13rep5; C_RHO = 2 keeps ≥ 20 % headroom and keeps the radius ≥
  20× tighter than the 0.056 mode gap); when rho_r = 0, membership requires
  byte identity with a component center (no implicit epsilon or ULP allowance);
- **byte/ULP class evidence** (reported for every gated row): whether the
  evaluated row is byte-identical to a reference byte-class, which class,
  else the max |Δlogit| and max f32-ULP distance to the nearest class;
- **leave-one-out verifiability**: every reference must be a member of the
  modes measured on the other N−1 at every row those demonstrate; any
  violation → unverifiable → fail.

Calibration: with the constants above, exactly one row of 32 demonstrates
multi-modality in the 15-draw set — short16 `longer` row 0, modes
{12 draws} / {ref13, ref13rep2, ref13rep3}, sep 25.6, rho_r = 0.001395.
Leave-one-out membership: **0 failures in 15 checks** (and 0 in the
12-member and 11-member emulation subsets).

The zero-radius rule is intentionally strict: an ULP-perturbed candidate can
fail the must-pass ULP control in this case, blocking qualification. Do not
add a tolerance after seeing such a result. Archived calibration predates
these edge-case definitions; evaluator fixtures must cover them before use.

**Mode-frequency report (diagnostic, not a gate condition).** At every
demonstrated row the report must state the reference mode frequencies
n(M)/N with an exact (Clopper–Pearson) 95 % interval, and the candidate
mode frequency: how many of the candidate's (byte-identical) runs land in
each mode. Calibration at `longer`/r0: ensemble-mode 12/15, H-mode 3/15,
CP95 [0.043, 0.481].

**H5. Exchangeability rank (v1 statistic, kept — see decision below).**
With S = {R_1..R_N, A}, s_i(X) = mean over Y ∈ S∖{X} of d_i(X, Y); rank
s_i across S (ties: average rank); T(X) = mean over all items of that rank.
**Gate: T(A) ≤ max_k T(R_k).** Report p = (1 + #{k : T(R_k) ≥ T(A)})/(N+1)
and the per-suite T as a diagnostic. No multiplicative margin.

**T-statistic decision (evaluated, reference-only).** Three statistics were
compared on the 15 draws: (a) v1 T — mean of ranks of mean distances over
sym_kl + ce_error + topk_miss; (b) sym-KL-only T; (c) an envelope-free
scale-normalized mean distance (no ranks). Results
(`calibration.json → H_t_variants`, `H_leverage`):

- the leave-one-out rank false-fail is **identical (1/15) for all three**:
  under exchangeability exactly the single global-maximum-T draw fails its
  own LOO rank test, whatever the statistic — the false-fail rate cannot
  distinguish statistics, and measured margins confirm it (worst margin
  −0.5625 for v1, −0.703 for sym-KL-only, −0.0129 for normalized, each the
  same draw's relationship to the bar);
- v1 T reproduces all six archived held-out evaluations **bit-exactly**
  (implementation continuity with the registered harness);
- removing the bimodal row's item from T shifts mode-draw T by ≤ 0.074 and
  other-draw T by ≤ 0.086 under v1 (no systematic mode penalty), while the
  normalized statistic concentrates the shift 2.2× on mode draws
  (0.00444 vs 0.00203) — i.e. it re-introduces mode sensitivity through
  the scale factor that H4 already handles directly;
- the sym-KL-only statistic discards the ce_error and top-10 evidence the
  registered exchangeability claim is made on, with no reference-side gain.

**Decision: keep v1's T unchanged.** The N=9 rank misses are consistent
with a sample-size effect of the bar (below), but the evidence pools two
collection sessions and cannot exclude session drift (Amendment 1).

## Sample size: N = 15 references, H = 3 held-outs

**Rank arithmetic.** Under H0, T(A) ≤ max_k T(R_k) fails iff A is the
largest of N+1 exchangeable scores: false-fail at most 1/(N+1) (exactly,
absent ties; ties occur and pass) — 0.100 at
N=9, 0.0769 at N=12, 0.0667 at N=14, **0.0625 at N=15**. The smallest
reportable p-value is 1/(N+1) = 1/16.

**Held-out miss arithmetic (measured on the 15 draws).** At N=9 the
archived ensemble (ref04–ref12) missed 3 of 5 fresh held-outs, by margins
+0.0341, +0.0398, +0.0568 (`H_n9_vs_n14`). Re-evaluating the **same draws**
under the N=14 structure: exactly **1/15** fails — the single global-max
draw (ref13rep3). This comparison changes ensemble size/composition and pools
sessions, so it does not isolate sample size from drift. Running the full
v2 rule on the archived N=9 structure still fails 4 of 6 held-outs
(`H2_v2_rule_at_N9`): the redesign does **not** substitute for N — the
9-draw ensemble misses the H-mode, its envelope cannot contain it, and the
bar stays low.

**Mode-coverage arithmetic.** P(a fresh N-draw ensemble contains no draw
of a mode with rate p) = (1−p)^N: at the observed point rate p̂ = 3/15,
0.134 at N=9 (realized by v1) vs **0.035 at N=15**; at p = 0.1, 0.206; at
the CP lower bound p = 0.043, 0.517 (see residual risks). Composition with
exactly one mode draw makes the envelope leave-one-out unverifiable
(fail-closed): P(k = 1) = 0.132 at p̂, so P(k ≤ 1) = 0.167 at the point
estimate.

**Held-out control: H = 3, all must pass.** Three further fresh reference
runs, pre-designated by attempt name before evaluation, each evaluated as
the candidate under the entire v2 rule (H1–H5, including verifiability)
against the N=15 ensemble. **Pass rule: all three pass.** Bounds under
exchangeability, no independence assumed: family-wise rank false-alarm
≤ H/(N+1) = 3/16 = 0.1875, marginally under exchangeability at fixed N.
The candidate-dependent scores are recomputed separately for each held-out;
no independence or fixed scalar-order exact value is claimed for this gate.
Ensemble-composition events (lone-mode unverifiable, missed-mode) can affect
all three. Any control failure blocks qualification; three passes do not
establish exchangeability or guarantee coverage of unseen modes.

**Validated on the 15 draws** (`I_heldout_splits`): all 455 splits of
12-ensemble + 3-held-out were evaluated under the full v2 rule — the
conservative N=12 emulation (smaller bar, tighter envelopes, less mode
coverage):

| result | value |
|---|---|
| splits where all 3 held-outs pass | 352/455 = **77.4 %** |
| per-held-out fail rate | 199/1365 = 14.6 % (reasons below overlap; 199 distinct failing pairs, 241 reason incidences) |
| — rank | 104 (7.62 %; nominal 1/13 = 7.69 %) |
| — unverifiable (ensemble composition: exactly one mode draw) | 108 (36 splits × 3) |
| — envelope (the one split whose ensemble missed the mode entirely) | 3 |
| — discrete (long/index-boundary-2053/r1 argmax+text) | 26 |
| — membership | **0** |

The observed self-fail is consistent with its bounds: rank family union
3/13 = 0.231 plus the composition events (37/455 = 8.1 % conditional on
the observed 3-mode-draw composition). At N=15 both components shrink
(rank union 3/16 = 0.1875; composition P(k ≤ 1) = 0.167 at p̂).

## Controls (evaluated under the identical v2 rule)

- **Must pass:** the 3 held-out references (primary); the ulp-control —
  the candidate moved one f32 ulp toward +inf (a null sample); a
  leave-one-out reference evaluated as the candidate is *not* required to
  pass the rank condition (exactly one reference is the T-maximum by
  construction; v1's "every leave-one-out reference" applies to the
  verifiability checks of H3/H4, which are gate conditions in their own
  right).
- **Must fail:** the negative controls of H1, plus the pre-declared
  semantic perturbations of the candidate, each on its own run with the
  identical profile: `ATLAS_EXL3_GATE_PERTURBATION` ∈ {`mhc_post_1p9`,
  `router_scale_2p4`, `topk7_layer=<L>`, `drop_shared_layer=<L>`,
  `bf16_routed_accum`} (v1 switch semantics, validation and layer
  constraints unchanged).
- **Measured, not assumed:** Atlas with `ATLAS_KDA_FLA_REFERENCE=0`,
  reported as the empirical minimum detectable effect.

Reference-side control validation (`calibration.json → K`, `J`, `G`):
token-swap refused; vocabulary shift fails the v2 rule; the ulp
perturbation of 14 of 15 reference draws passes (the 15th is the global-T
rank artifact, not applicable to a candidate-derived ulp control);
pseudo-candidates built by amplifying a reference deviation away from the
reference mean (α = 2, 4, 6) are detected 15/15 by rank already at α = 2,
15/15 by membership at α ≥ 4, and 15/15 by the H3 envelope at α ≥ 4 —
**a candidate with excess beyond the envelope fails**.

## Collection protocol (fresh; nothing archived may be reused)

- **Profile:** the registered collector-only profile of v1 — unchanged pinned
  reference image `glm53-flash-mikeroz:dt-carry20`, the pinned create
  commands and observers (hashes in the v1 doc: `ref-rank0/1-create-
  command.json`, `sitecustomize.py`, `reference_chunk_trace.py`,
  `ATLAS_EXL3_CHUNK_RUNNER_SHA256`), registered context-v4 serve flags
  (4096 context, 128 prefill chunk, TP2/EP1, reference requested KV `auto`
  with effective MLA `fp8_ds_mla`; candidate `fp8_g128`; eager, no
  speculation, no prefix caching), marker-gated logits collector and the
  serial128 CPU chunk-position observer only. Intermediate-stage dump and
  head-input capture envs are absent; registered logits/chunk envs remain.
  Run `collect_logits.py` then `collect_phase_logits.py`, serial
  requests, temperature 0 / seed 42 / max_tokens 1, suites short16 + long
  in fixed order. **Never mix configurations, images, or instrumented
  builds in one set.** Candidate runs use the registered candidate profile
  (image identity recorded before any request, as in the v1 addendum).
- **Runs (all fresh containers, one serve process each):**
  1. `v2ref01..v2ref15` — the N = 15 reference ensemble;
  2. `v2refho01..v2refho03` — the H = 3 held-out references,
     pre-designated by attempt name before evaluation;
  3. `v2cand01`, `v2cand02` — the candidate, two fresh-server runs, hard
     requirement: byte-identical;
  4. `v2pert-*` — five candidate runs, one semantic perturbation each;
  5. `v2mde` — candidate profile with `ATLAS_KDA_FLA_REFERENCE=0`
     (measured, not gated).
- **Order and state:** freeze the 20-run seeded permutation before the first
  request (Amendment 1 A1.1). Within each attempt, preserve serial short16
  then long collection. Fresh processes control process-local state; mounted
  compilation/autotune caches can persist and must be recorded, not silently
  reset. See the [pinned vLLM source contract](exl3-vllm-reference-contract.md).
- **Cache-state receipt:** before startup and after shutdown on both ranks,
  archive a manifest for each registered vLLM, Triton, TileLang, CUDA compute
  and EXL3 tune-cache mount: host/container path, whether shared across
  attempts, and sorted relative file paths with size and SHA256 (including
  an explicit empty/missing state). Record pre/post changes; do not copy,
  purge or reseed caches during the campaign. A changed cache is reported
  process-external state, not proof of a mode's cause. Missing receipts make
  provenance incomplete. Apply the same accounting to candidate cache mounts
  where present; explicitly record absence otherwise.
- **Evaluation:** check H3/H4 verifiability on the 15 references with each
  evaluated reference excluded from its comparison set. LOO H5 scores are
  diagnostic, not required to pass. Evaluate each of the three held-outs
  against the fixed 15-reference ensemble; all three must pass applicable
  H1 provenance/finite checks and H2–H5. Candidate-pair byte identity and
  negative controls are campaign checks, not a requirement that distinct
  reference draws match each other. The candidate pair, ulp-control, negative
  controls, perturbations and MDE follow. No sample compares against itself.
  Provenance, manifests, captures, reports and the frozen
  `compare_logits.py` secondary record are archived per attempt.
- **Verdict semantics:** v2 produces its own verdict; it is not
  comparable to the v1 verdict (historical notes below).

## Calibration evidence summary (reference-only; full data in `calibration.json`)

| check | result |
|---|---|
| fingerprint of the 15 draws (identity + provenance + harness) | identical |
| reproduction of 6 archived v1 held-out evaluations | bit-exact (T, p, catastrophic, reasons) |
| byte classes at short16/longer/r0 | 6, matching the investigation report |
| rows demonstrating multi-modality (15 draws) | 1 of 32: longer/r0, sep 25.6, modes 12/3, rho_r 0.001395 |
| demonstration stability | 1928/1941 subsets of size 11..15 demo longer/r0 (the 13 non-demo subsets contain no H-mode draw); 0 non-mode-row demos anywhere |
| membership leave-one-out (def as registered) | 0/15 failures (alternatives b/c: 0/15, 0/15) |
| coverage ratios (all code-path subset sizes) | max 1.652 (ref13rep5); C_RHO = 2.0 |
| **LOO full-rule false-fail rate on the 15 draws** | **1/15 = 6.7 %** — rank 1 (the global-max draw, ref13rep3), unverifiable 0, envelope 0, membership 0, discrete 0; nominal 1/(N+1)\|N=14 = 6.7 % |
| ulp-control emulation (15 draws vs 14 refs) | 1/15 (same global-max artifact only) |
| v2 rule on the archived N=9 structure | 4/6 held-outs fail → N, not the statistic, repairs the control |
| held-out split emulation (455 × 12+3) | 77.4 % all-pass; rank fail 7.62 % ≈ nominal 7.69 %; membership 0 |
| pseudo-candidate power (α = 2/4/6) | detection 15/15 by rank at α=2; envelope 15/15 at α ≥ 4 |
| negative controls under v2 | token-swap refused; vocabulary shift fails |

Envelopes derived from the 15-draw pairwise distribution (calibration
example; qualification envelopes come from the fresh set): sym-KL envelope
= max(3·spread, 0.005) ranges from the 0.005 floor (4 rows: capital/r3,
code/r0–r2) to 1.071 (long/2051/r1); `short16/longer/r0` = 0.202; the
full 32-row table is `calibration.json → D_envelopes`. ce_error envelopes
range from the 0.15 floor to 5.73 (long/2053/r0).

## What this rule does and does not claim (explicit)

This redesign **addresses reference-control failures; it does not excuse
the candidate.** Fresh collection must establish whether the amended control
passes. Finite-sample composition, discrete rules and session effects can
still reject a reference-identical engine. Constants are unchanged, but the
measured H3 envelope can widen with N; this is not equivalent strictness to v1.
A PASS means this image meets H1–H5 and the controls on the registered
fixtures and observed profile. It does not establish identical output
distributions, mode frequencies, task quality or release readiness.
A candidate whose deviation exceeds
the measured chaos envelope at any row, whose T exceeds every reference's,
or that lands **between** the measured modes at a demonstrated-bimodal row
fails. Sub-chaos-floor arithmetic correctness remains the domain of the
stage-level byte oracles (v1 scope note, unchanged).

## Residual risks (measured, not hidden)

- **Mode-rate uncertainty.** The H-mode rate is bracketed 3/15, CP95
  [0.043, 0.481]. P(a fresh 15-draw ensemble has ≤ 1 mode draw) is 0.167
  at the point estimate and 0.86 at the CP lower bound. Exactly one isolated
  mode draw can make verifiability fail; zero mode draws can pass it. P(0 mode
  draws) = 0.035 at p̂; a candidate landing in an unseen mode then fails
  the envelope (false-fail, fail-closed). Held-outs can reveal a missing mode
  but are not guaranteed to sample it.
- **Discrete sensitivity.** In the N=12 emulation, held-out argmax/text at
  long/index-boundary-2053/r1 produced 26/1365 discrete fails (0/15 at
  N=14 LOO). The rule is unchanged; qualification at N=15 re-measures it.
- **Calibration concentration.** G_MIN = 4, K_MIN = 20, C_RHO = 2.0, c = 3
  and the floors are calibrated on these 15 draws only. The demonstration
  rule is data-driven per row on the fresh set; the constants are fixed.
- **Ulp-control.** Derived from the candidate, which must pass the rank
  bar first; reference-side emulation inherits the global-max artifact
  (1/15) and is not itself a qualification criterion.

## Amendment 1 (2026-10-12, pre-collection)

**Historical amendment, partially superseded by Amendment 2 below.** A1.1
randomisation and A1.3/A1.4/A1.6 remain applicable. A1.2 is withdrawn; the
A1.5 composition-risk statement is corrected below.

Source: independent review `CODE_REVIEW_2026-10-12.md` (finding A3), applied
before any v2 collection. Authored from reference-side reasoning only, with
no candidate numbers. The then-current amendment changed collection order
and added the now-withdrawn void rule for reference-ensemble composition.
Its numeric constants were unchanged.

**A1.1 Interleaved, randomised run order.** The 20 runs `v2ref01..15`,
`v2refho01..03`, `v2cand01`, `v2cand02` are executed in a random
permutation. It is generated before the first run from a seed recorded in
the attempt plan, written to the plan, and followed exactly. Held-out
designation stays by attempt name, fixed before evaluation. Perturbation and
MDE runs follow. Reason: the archived evidence cannot exclude drift between
collection sessions. Three fresh ref13 repeats scored T above the archived
ensemble's range (5.80/5.84/5.94 against a maximum of 5.76–5.89), while the
same-session ref13 sat well inside. Sequential reference → held-out →
candidate collection would load any drift onto the candidate.

**A1.2 Reference-composition void + top-up — SUPERSEDED.** The original
amendment allowed two reference-only top-ups to N ≤ 25 and claimed a
final-N rank bound. Amendment 2 withdraws both the top-ups and that bound:
reference-only adaptive stopping does not establish the exchangeability
needed by the fixed-N rank argument. Original text: commit `fd7b530b`.

**A1.3 Disclosures.**
- H3 is v1's catastrophic-row rule (same c, floors and leave-one-out
  check). Its envelope widens with N, so per-row H3 gating is looser than
  v1 at N = 9. Detection of a deviating candidate rests mainly on H5 (rank)
  and, at demonstrated rows, H4.
- H4 is **stricter** than the floor at demonstrated rows. At
  `longer`/r0 the membership radius C_RHO · rho_r = 0.0028 is below the
  0.005 floor that H3 calls never-catastrophic, so a value 0.004 from a
  mode centre passes H3 and fails H4.
- A deterministic candidate that always lands in the rarer reference mode
  passes H4 by design. The candidate mode frequency is reported
  (diagnostic).

**A1.4 Restated v1 diagnosis.** The v1 control failure is diagnosed as
*underpowered at N = 9 under a bimodal row, possibly with session drift*.
At N = 9, v1's probability that a held-out reference passes the
`longer`/r0 row condition (rank excluded) is ≈ 0.57–0.70 for mode rates
0.043–0.2, and higher at larger rates. It is not zero. "Structurally unpassable" is
withdrawn. v2 remains justified by N, the explicit leave-one-out rank
exemption (v1's every-LOO-must-pass text is incoherent) and H4.

**A1.5 Additional residual risks.**
- *Session non-exchangeability* (A1.1 mitigates temporal confounding; held-out
  failures can expose a problem, but passing does not prove exchangeability).
- *Mode-rate selection.* The 3/15 rate includes ref13, the draw that
  triggered the investigation. Excluding it gives 2/14; fresh draws alone
  give 2/5.
- *Unidentified discrete mechanism.* 15 draws fall into 6 byte classes, two
  of which hold 9 draws. That suggests a small discrete runtime choice
  (routing tie, autotune/plan selection, NCCL algorithm) rather than
  continuous chaos. "Fingerprint-identical" covers configuration and
  image, not these runtime choices. A probe that pins them is recommended
  before collection. If the mode can be pinned, H4 may become unnecessary,
  which would require a new amendment.
- *In-sample constants.* C_RHO = 2.0 is 1.21× the largest observed coverage
  ratio, and the membership definition was chosen as the first with 0 LOO
  failures. The LOO and split figures are therefore in-sample; only the
  fresh held-outs are out-of-sample.
- *Single-campaign false reject.* For a reference-identical engine, before
  A1.2, the combined probability is roughly 0.35 at p̂ = 0.2 and worse at
  the CP lower bound. The withdrawn A1.2 could not eliminate composition risk:
  its cap could be reached, and a wholly absent mode need not trigger its
  check. Amendment 2 retains fixed-N composition risk explicitly. The rough
  estimate is not a validated false-reject rate for the amended campaign.

**A1.6 Errata in `calibrate_v2.py`** (evidence script left unchanged; the
results are unaffected):
- The printed label for membership definition "a" (line 333) omits C_RHO,
  which the code applies (lines 355, 477).
- The comment calling definition "a" "least-strict" is wrong; it is the
  strictest.
- Definition selection (lines 366–373) is data-driven ("first with 0 LOO
  failures"), as A1.5 discloses.

## Amendment 2 (2026-10-12, pre-collection): fixed design and source contract

Operator request: apply the review recommendations after a Luna subagent
reviews the exact vLLM process; use pinned vLLM source as authoritative for
engine behavior. Source report and parent reconciliation:
[exl3-vllm-reference-contract.md](exl3-vllm-reference-contract.md).
No qualification data was collected or candidate result used for this revision.

**A2.1 Fixed collection and verdicts.** N = 15, H = 3, two candidate runs;
all attempt names, engine identities, the random seed/permutation and exact
perturbation layers are recorded before collection. There are no composition
top-ups, replacement draws or outcome-driven sample-count changes. Every draw
is retained. H3/H4 reference verifiability failure terminates the campaign as
`UNVERIFIABLE` (qualification blocked), not a numerical verdict on the candidate.
A held-out/control failure blocks qualification; a candidate rule failure is
FAIL. Missing/corrupt artifacts or interrupted runs are INCOMPLETE and cannot
PASS. All reasons and attempts remain archived. Any retry requires a separately
registered campaign retaining this campaign's outcome; never silently repeat
until PASS or present a successful retry as the only result.

The `1/16` H5 bound is marginal, rank-only, conditional on the exchangeability
assumption, and applies to the fixed experiment rather than campaigns selected
for passing other checks. Whole-campaign error rates are not established here.

**A2.2 H4 completeness.** The operative H4 section now defines exact two-class
zero-spread modes, zero radius, singleton spread, zero denominators and gap
ties. Do not introduce a tolerance or fallback clustering rule at runtime.

**A2.3 Source-bound interpretation.** Preserve registered observer mounts,
request/suite order and effective KV formats. The selected reference SM120
path reads cache even for short prefill. Seed 42 and a fresh process do not
promise deterministic forward logits. The recurring mode's mechanism is still
unidentified. A probe changing runtime policy is separate diagnostic work;
using that policy for qualification requires another pre-collection amendment.

**A2.4 Implementation readiness.** This is a documentation revision, not a
claim that the v1 evaluator implements v2. Before collection, freeze and review
an evaluator with fixtures for H3/H4 LOO exclusion, two exact byte classes,
zero cross-distance, zero radius/ULP control failure, singletons, tied gaps,
fixed attempt order/counts, blocked/incomplete campaigns and held-out isolation.
Verify unchanged T against the archived reference-only calibration. Recheck
H4 calibration with the edge-case definitions and archive any differences;
the old calibration alone does not validate the amended implementation.
These checks and evaluator review/freeze precede qualification collection.
Operator collection authorisation is recorded in `decisions.md` (2026-10-12).

## Historical notes

The v1 pre-registration and its archived evaluation are retained unchanged
as the historical record: under the frozen rule the candidate evaluation
recorded a FAIL verdict, and the v1 held-out control failed for 3 of 5
fresh reference draws (context only; no v1 candidate number is used
anywhere in this document). Direct comparability between the v1 and v2
verdicts is **surrendered by explicit operator choice**
(`decisions.md`, 2026-10-12): v2 is a different registered rule with a
different reference set, and no v2 verdict is a re-evaluation of the v1
verdict. The redesign repairs the control, not the verdict.

## Evidence index

| path | content |
|---|---|
| `/home/colum/Workspace/atlas-exl3-validation-20261010/q2-gate-v2-20261012/calibrate_v2.py` | calibration script (reference-only; sections A–L) |
| `…/q2-gate-v2-20261012/calibration.json` | full calibration output (envelopes, gap scan, LOO, splits, pseudo-power, N arithmetic) |
| `…/q2-gate-v2-20261012/summary.md` | calibration + rule summary for the operator |
| `…/q2-ref13-investigation-20261012/` | the reference-mode discovery this rule is built on |
| `…/statistical-gate-20261011/` | archived v1 tuning data (ref04–ref13 + final/) — untouched |
| `tests/exl3_validation/statistical_gate.py` | v1 harness (snapshot-hashed); v2 implementation follows this registration |
