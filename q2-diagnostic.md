# Q2 diagnostic and repair handoff

Initial handoff: 2026-10-11. Append dated findings at the end of this file.
Preserve earlier observations; explicitly supersede a claim when correcting it.
Read `AGENTS.md`, `CONTRIBUTING.md`, `continue.md`, and the measurement-discipline
skill before running experiments or publishing numerical conclusions.

## Objective and current verdict

*Rewritten 2026-10-12 after the independent review (`CODE_REVIEW_2026-10-12.md`);
the original objective text is in git history.*

Find the dominant cause of the candidate's numerical divergence and repair it,
so the candidate passes the registered qualification gate with all controls.

- **v1 gate (historical record):** candidate FAIL (rank-dominated, plus 3
  catastrophic `longer` rows). The held-out `ref13` control also fails,
  diagnosed as reference-internal bimodality at `longer`/r0. v1 was
  underpowered at N = 9 under a bimodal row, with session drift not
  excluded; it was not structurally unpassable.
- **Qualification rule going forward:** gate v2
  (`docs/design/exl3-statistical-gate-v2.md`, with Amendment 1), pending
  operator sign-off. No v2 data collected.
- **Norm attribution:** `v24-minimal` reproduces the reference gated-norm
  output on 7/7 differing elements in 2 independent fixture groups
  (fixture-local; NVRTC `--fmad=false`; v24 has no engagement controls of
  its own).
- **C1:** closed from native captures. There is no same-input o_proj error,
  and the natural norm delta is absorbed or confined locally. **Flag:** the
  corrected-input answer is inferred, and the o_proj input precision is
  unattested.
- **The dominant final-logit cause is unproven.** C2 (layer-0 norm-output
  intervention traced to final logits) is next. No runtime fix is qualified.

Do not discard failing reference rows or select favorable reruns.

## Evidence and source locations

Durable evidence root:

```text
/home/colum/Workspace/atlas-exl3-validation-20261010
```

Paths below are relative to that root unless stated otherwise:

| Path | Purpose |
| --- | --- |
| `statistical-gate-20261011/final/` | Registered Q2 results, controls and per-case breakdown |
| `q2-investigation-20261011/report.md` | Final-logit failure investigation |
| `q2-investigation-20261011/report-stage2-propagation.md` | Propagation investigation; read alongside the corrected review |
| `q2-investigation-20261011/report-mla-implementation-delta.md` | MLA implementation hypotheses |
| `q2-diagnostic-20261011/extended-plan.json` | Executed diagnostic plan, identities and invocation mapping |
| `q2-diagnostic-20261011/runbook.md` | Prior diagnostic execution procedure |
| `q2-diagnostic-20261011/dumps/` | Candidate/reference exact-byte internal tensors |
| `q2-diagnostic-20261011/analysis/localization-report.md` | Corrected diagnostic verdict; governs interpretation |
| `q2-diagnostic-20261011/analysis/review-validation.json` | Independent reviewed machine evidence |
| `q2-diagnostic-20261011/resume-analyzer-20261011/` | Repaired analyzer/report, verification logs and host shutdown receipts |
| `arithmetic-audit/glm5next/nvidia/kda.py` | Extracted reference KDA source matching archived installed-module hash |
| `arithmetic-audit/reference-kda-kernel-source/fla/ops/fused_norm_gate.py` | Extracted reference normalization implementation; runtime binding still needed |

Repository implementation paths:

- `tests/exl3_validation/statistical_gate.py`: existing Q2 evaluation and controls.
- `tests/exl3_validation/analyze_localization.py`: repaired diagnostic analyzer.
- `tests/exl3_validation/test_analyze_localization.py`: analyzer regression controls.
- `kernels/gb10/common/kda.cu`: candidate gated normalization kernel.
- `crates/spark-model/src/layers/ops/kda.rs`: kernel launch wrapper.
- `crates/spark-model/src/layers/glm5_kda/forward_attention.rs`: norm invocation.
- `crates/spark-model/src/weight_loader/glm5/exl3_kda_loader.rs`: norm weight loading.

Historical diagnostic identities:

- Candidate attempt: `cand-q2diag-ee5eaf0f-02`; source `ee5eaf0f`;
  image `sha256:7270d7ca16f7896da09e78e4a68cbac56f0aac1749ea9f8c6c671370aa3b94a6`.
- Reference attempt: `ref-q2diag-ctxv4diag-03`;
  image `sha256:6eb42fa20c145cabf9f788d7b850d8d3416706ad1955a003448b48962cb2845b`.
- Registered candidate: `cand-fp8decode02`, byte-identical logits to `cand03`.
- Reference ensemble: `ref04`–`ref12`; held-out must-pass control: `ref13`.

## What the corrected diagnosis establishes

The archived diagnostic covers arithmetic prefixes 10–13 and longer prefixes
36–39 on both ranks. It does not cover the long-suite 2048-token failure.

- Layer-0 embedding and attention mHC-normalized inputs match.
- KDA core/g1/g2 match across all 16 rank/prefix pairs after explicit layout
  conversion.
- The first verified mismatch among these captured KDA stages is gated
  normalization: arithmetic rank 1 differs by one BF16 element per prefix;
  longer rank 0 differs by six. Each difference is at most one BF16 ULP.
  Resident norm weights, epsilon and runtime dispatch were not bound.
- Longer reduced attention/FFN outputs match despite the internal norm mismatch.
  Small internal differences can disappear through downstream rounding.
- Arithmetic reduced attention and FFN inputs already differ. Neither its
  reduced o_proj output nor dense gate/up demonstrates same-input GEMM error.
- Layer-3 expert membership changes: two token rows per arithmetic prefix and
  one per longer prefix. This is more than a change in ordering.
- Correctly paired post-TP MoE outputs have p99 distances of 64–71 BF16 ULP.
  The historical 1507-ULP claim compared different TP boundaries and cannot
  establish amplification.
- Divergence precedes MLA in these cases. Its contribution to final-logit
  failure remains unmeasured.
- Three arithmetic-10 reference probes show within-process routed-f32
  variation. They do not bound other prompts or fresh processes.

Do not infer that the one-ULP norm difference explains the full gate failure.
Do not restore the superseded same-input GEMM or dominant MoE conclusions.

## Analyzer repair already completed

The repository fork reads declared little-endian integer IDs, uses monotonic
signed BF16 ordering, maps KDA layouts explicitly, compares corresponding
post-TP MoE outputs, and checks immediate producer inputs on both ranks for
reduced attention. It validates archived collector identities and records
evidence hashes. Producer checks bind dynamic inputs only; weights/config are
still required for arithmetic attribution.

Verification: eight regression controls failed before repair and pass after
repair. All 464 archived stage pairs load without missing/error results.
128 floating pairs and 16 expert-ID comparisons agree with the independent
review. The relevant Python suite passed 282 tests and 193 subtests.
These are offline diagnostic checks, not GPU qualification.

Reproduce with a new output filename:

```bash
python3 -m pytest tests/exl3_validation/test_analyze_localization.py -q
python3 tests/exl3_validation/analyze_localization.py \
  --root /home/colum/Workspace/atlas-exl3-validation-20261010/q2-diagnostic-20261011 \
  --output /tmp/q2-localization-NEW-ATTEMPT.json
```

Original external analyzer, outputs and dumps remain preserved.

## Required experiments — status (2026-10-12)

Sequencing authority: `docs/plans/q2-norm-causality-plan.md`. The original
five-step text is in git history.

1. **Bind normalization constants and dispatch: DONE**
   (`q2-norm-replay-20261011/`; runtime-bound fixtures, epsilon bits agree).
2. **Replay identical inputs and constants: DONE, fixture-local.**
   - Self-replay passes for both engines.
   - The ablation attributes the difference to the rsqrt and divide forms;
     `v24-minimal` matches 7/7 differing elements (2 groups).
   - Open items: a GPU rerun under the repaired harness with v24's own
     controls, and v24 under the production toolchain flags.
3. **Downstream causality:**
   - C1 (local o_proj) is CLOSED from native captures (flagged; see the
     post-review entry).
   - **C2 (layer-0 norm-output intervention → final logits) is NEXT.**
   - Then the L1 escalation for `longer`, then G1–G9.
   - Never replay o_proj over a BF16 reconstruction. Both engines run EXL3
     trellis kernels; bind the trellis bytes and the input-precision branch.
4. **Held-out reference control: DONE.** Reference-internal bimodality at
   `longer`/r0, mechanism unidentified. This led to gate v2 (pending
   sign-off). A probe of the discrete runtime choice is recommended before
   v2 collection.
5. **Rebuild and qualify:** only after a causally justified fix. Requires
   regression tests, a clean image, fingerprints and neutrality proofs, then
   a fresh run under gate v2 (once signed off) with all controls. The
   semantic perturbation GPU controls are still owed.

Diagnostic self-reproduction, internal exactness, neutrality and focused Q3
passes each prove their own scope. None substitutes for the final Q2 result.

## Host and evidence handling

At the last verified resume, both reference diagnostic containers were archived
and stopped; neither Spark had a running container. Production remained stopped.
No image was published. Reverify current state before another GPU session.

- Jump host: `ai-node` (user correction). Old `192.168.1.50` access timed out.
- Observed jump-host Tailscale address: `100.121.137.95`; verify before reuse.
- Spark 0: `colum@10.50.0.10`, hostname `gx10-eb0e`.
- Spark 1: `colum@10.50.0.11`, hostname `gx10-235f`.
- Each host archive: `~/gate/resume-norm-review-20261011-gate-ref-N/`.
- Local before/after inspections, logs and archive hashes:
  `q2-diagnostic-20261011/resume-analyzer-20261011/host-rankN/`.

Preserve previous containers, attempts and evidence. Never reuse burned attempt
names or delete failed captures. Record full serve flags, checkpoint/config,
toolchain/runtime, harness/request policy, both GPU identities and hashes beside
every experimental conclusion. Append a shutdown receipt when releasing hosts.

## Append-only experiment log

Add one entry per experiment or review. Use this template; include failures and
incomplete attempts. Distinguish measured facts from hypotheses.

### YYYY-MM-DD HH:MM UTC — experiment name

- Agent / objective:
- Attempt IDs and diagnostic plan/overlay hashes:
- Source, image, binary, kernels, checkpoint/config and runtime fingerprints:
- Hosts/ranks, serve flags, request inputs and invocation/layout/TP mapping:
- Variable changed; constants held fixed:
- Self-reproduction and positive/negative control results:
- Observations, with raw evidence paths and hashes:
- Interpretation and limits:
- Claims explicitly superseded, with reason:
- Q2/control verdict (or untested):
- Host shutdown/archive receipt:
- Next action:

### 2026-10-11 22:55 UTC — first-pass diagnostic review (offline verification + host state audit)

- Agent / objective: First-pass diagnostics per handoff. Scope agreed with
  operator: offline evidence-tree verification, analyzer/test reproduction,
  claim cross-check, implementation-path reading, and read-only host
  inspection. Facts and hypotheses separated below. **No code or other files
  were edited**; new outputs written only to `/tmp` and to this file.
- Attempt IDs and diagnostic plan/overlay hashes: no new attempt started.
  Re-verified archived identities only: candidate `cand-q2diag-ee5eaf0f-02`
  (source `ee5eaf0f`, image `sha256:7270d7ca16f7…`), reference
  `ref-q2diag-ctxv4diag-03` (image `sha256:6eb42fa20c14…`), registered
  `cand-fp8decode02`/`cand03`, refs `ref04`–`ref12`, held-out `ref13`.
- Source, image, binary, kernels, checkpoint/config and runtime fingerprints:
  local HEAD `f0ba28fd`; see "working-tree correction" below for what the
  burned image did and did not contain. No GPU execution, so no new
  fingerprints.
- Hosts/ranks, serve flags, request inputs and invocation/layout/TP mapping:
  read-only inspection only (no requests sent). See host section below.
- Variable changed; constants held fixed: none (verification-only pass).
- Self-reproduction and positive/negative control results:
  - `python3 -m pytest tests/exl3_validation/test_analyze_localization.py -q`
    → 8/8 pass (matches the eight repaired-analyzer regression controls).
  - `analyze_localization.py --root …/q2-diagnostic-20261011 --output
    /tmp/q2-localization-NEW-ATTEMPT-20261011.json` → exit 0, output
    **byte-identical** to archived
    `resume-analyzer-20261011/q2-localization-repaired-20261011-r3.json`
    (sha256 `a2fb3f65775b276dd7e91afe3671ed414d52525569e44633fa3c1a12dcb5ac38`).
  - `analysis/review_localization.py /tmp/q2-review-NEW.json` → "helper
    controls passed; validated 176 pairs", output **byte-identical** to
    `review-validation.json`
    (sha256 `0ec887c459dfdc0d59427fa76e0c2b717438c316dbba82b4d02c2a6a0347ad38`).
  - Full `tests/exl3_validation/` suite → 293 passed, 208 subtests passed
    (110.6 s). The archived `q2-resume-validation-tests-20261011.log` records
    282 tests + 193 subtests; that figure is intact and was the count at
    repair time. Current-tree delta is consistent with the uncommitted test
    tooling added afterwards (hypothesis; exact per-file attribution not
    recomputed). Neither figure contradicts the analyzer-repair claim.
- Observations, with raw evidence paths and hashes:
  1. **Evidence tree intact (fact).** All twelve paths in the "Evidence and
     source locations" table exist and are readable under
     `/home/colum/Workspace/atlas-exl3-validation-20261010`, including
     `statistical-gate-20261011/final/`, the three investigation reports,
     `dumps/` (cand-rank0/1 = 840 files each; ref-rank0/1 hold `diagnostic/`
     plus `reference-logits/`), `analysis/`, `resume-analyzer-20261011/`
     (incl. `host-rank0/`, `host-rank1/` inspection + SHA256SUMS receipts),
     and both extracted reference sources.
  2. **Registered Q2 failure detail (fact, from
     `statistical-gate-20261011/final/gate-cand02-cand03-vs-ref04-12.json`).**
     Gate FAIL with two reasons: exchangeability rank (candidate farther than
     every reference) and 3 catastrophic rows. All three catastrophic rows are
     suite 0 / case `longer` rows 0–2 (sym_kl nearest 0.032741 / 0.013552 /
     0.090346 vs row limits 0.005 / 0.01298892900657322 /
     0.056452221505187083). The `ulp_control` FAILs with the same two reasons
     and the same rows. `negative_controls` PASS (swapped-token and
     shifted-vocabulary both detected). `discrete_failures` is empty on both.
     Per-case (`per-case-cand02.json`): candidate nearest-sym_kl mean is
     elevated for `arithmetic` 0.0736, `longer` 0.0448, and suite-1
     index-boundary cases 0.032–0.075, but tiny for `code` 3.9e-5 and
     `capital` 0.00123 — the divergence is localized, not uniform.
  3. **Held-out control detail (fact, from `heldout-ref13-vs-9refs.json`).**
     ref13 FAILs exactly one row: suite 0 / case `longer` row 0, sym_kl
     nearest 0.056035 vs limit 0.005 — the same case/row as the candidate's
     worst catastrophic row. (Hypothesis, not established: case `longer`
     row 0 may be intrinsically unstable across fresh processes, or the two
     failures share a mechanism. Required experiment 4 remains the arbiter;
     do not call either failure an artifact without reproducing a specific
     faulty mechanism.)
  4. **Analyzer/review claims reproduce (fact).** The reviewed numbers quoted
     in the handoff reappear verbatim in the freshly generated review output:
     layer-0 core/g1/g2 exact on all sampled pairs; gated-norm first mismatch
     with the stated one-element/one-ULP pattern; arithmetic
     attn_reduced 32 elements max 16 ULP; ffn_input 56 elements max 10 ULP;
     post-TP MoE 45.33 % exact, p99 71 ULP, max 31089; router IDs 6 differing
     ordered slots with membership change in 2 rows (row-0/arithmetic sample).
     No contradiction found between `localization-report.md`,
     `review-validation.json` and the raw stage dumps inspected.
  5. **Working-tree correction (fact; supersedes any reading of "HEAD differs
     from `ee5eaf0f` only in documentation" as describing the working tree).**
     The commit-level claim holds (`git diff ee5eaf0f f0ba28fd` = `continue.md`
     + `todo.md` only), but the working tree carries **uncommitted Rust
     changes**: `glm5_kda/forward_attention.rs` (+16/−1),
     `glm5_kda/upstream.rs` (+134), `layers/ops/kda.rs` (±10, launch-geometry
     refactor into `kda_sigmoid_gated_norm_launch_dims`, same grid/block),
     `spark-runtime/src/weights/exl3/ffn_io.rs` (+39,
     `capture_with_metadata` + optional `binding` field in the
     `atlas-exl3-ffn-v1` record), `ffn_io_tests.rs` (+27). These implement the
     candidate-side **gated-norm binding capture** (schema
     `atlas-kda-gated-norm-binding-v1`: resident `weight` BF16 `[dim]`,
     `eps_f32_bits`, kernel symbol `kda_sigmoid_gated_rms_norm`, grid/block,
     activation `sigmoid`, affine true, bias/residual/prenorm false, full
     dtype set). Verified absent from the burned source:
     `git show ee5eaf0f:crates/spark-model/src/layers/glm5_kda/upstream.rs`
     contains no binding capture — which explains why no
     `gated_norm_binding` stage exists in the archived candidate dumps
     (grep over `dumps/` finds none). Consequences: (a) most of required
     experiment 1's *candidate-side* capture already exists and does not need
     to be re-invented; (b) any image built from the current tree is
     diagnostic-dirty vs `ee5eaf0f`, so step-5 neutrality and clean
     provenance requirements apply, and the changes should be committed
     before they appear in a qualification attempt.
  6. **Implementation-path facts relevant to experiments 1–2.** Candidate
     kernel `kda_sigmoid_gated_rms_norm` (`kernels/gb10/common/kda.cu:565`):
     per-row (token×head) FP32 reduction over `dim` (=128),
     `inv = rsqrtf(red[0]/dim + eps)`,
     `y = x * inv * weight[d] / (1 + expf(-g))`, one BF16 round;
     `eps = ctx.config.rms_norm_eps` (`forward_attention.rs:159`).
     Loader (`exl3_kda_loader.rs`): `o_norm.weight` read as checkpoint BF16,
     shape `[kd]`, `single_range(0..kd)` — a single 128-element norm weight,
     not TP-sharded. Reference (`arithmetic-audit/glm5next/nvidia/kda.py:278`):
     `FusedRMSNormGated(self.head_dim, activation="sigmoid")` with **no
     explicit eps** → class default `1e-5`
     (`fused_norm_gate.py:341-356`); `forward_native` (torch.rsqrt +
     torch.sigmoid) and the Triton `forward_cuda` use different arithmetic.
     (Hypothesis: the observed ≤1-ULP gated-norm difference is consistent
     with reduction-order / sigmoid-form / eps-constant differences; binding
     the effective values decides. Not established.)
  7. **Experiment 1–2 harnesses already exist in the tree (fact).** Untracked
     but present and passing: `tests/exl3_validation/reference_norm_binding.py`
     (binds resident reference norm operands + selected CustomOp dispatch,
     clones inputs before the call — exactly the aliasing hazard the handoff
     warns about), `reference_norm_replay.py` (reference self-replay with
     fresh clones per trial), `native_gated_norm_replay.py`/`.cu` +
     `native_ptx.py` (candidate PTX replay with explicit bound epsilon),
     with `test_reference_norm_binding.py` (11 tests + 11 subtests),
     `test_reference_norm_replay.py` (4 + 3),
     `test_native_gated_norm_replay.py` (4 + 10) — all green in this pass.
     Remaining work for experiments 1–2 is GPU-side execution and overlay
     binding, not tooling construction.
  8. **Host state (fact, observed 2026-10-11T22:42Z).** Reachable via jump
     host at `100.121.137.95` (DNS for `ai-node` failed intermittently again;
     the Tailscale address worked). Spark 0 `gx10-eb0e` (10.50.0.10) and
     Spark 1 `gx10-235f` (10.50.0.11): **0 running containers**, both GPUs
     idle (0 % util). Archives
     `~/gate/resume-norm-review-20261011-gate-ref-{0,1}/` present with
     SHA256SUMS (rank0 `inspect-after` `252b7cb1…`, rank1 `inspect-after`
     `e9021936…`; `captures.tar.gz` rank0 `f3f72732…`, rank1 `87bb4b64…`).
  9. **New post-handoff GPU activity (fact — read carefully).** Containers
     `gate-ref-normbind-0` (created 22:35:27Z, finished 22:41:12Z, exit 0) and
     `gate-ref-normbind-1` (created 22:35:26Z, finished 22:42:23Z, exit 0) ran
     today on image `glm53-flash-mikeroz:dt-carry20` on both ranks;
     `/tmp/atlas-norm-replay-20261011-r1` appeared on Spark 0 at 22:35Z. These
     postdate the handoff's "no running container" record and overlap the time
     of this inspection. (Hypothesis: a parallel operator/agent session is
     executing the norm-binding/replay slice of experiments 1–2. Not
     confirmed.) Also `gate-ref-0/1` show exit timestamps of ~22:11Z today, so
     they were restarted and stopped after the archived shutdown receipts.
- Interpretation and limits:
  - This pass verified reproducibility of the offline evidence and the host
    inventory; it produced no new numerical evidence and no GPU measurements.
    All numbers above are quoted from preserved artifacts, cross-checked but
    not re-measured.
  - The dominant final-logit cause remains unproven. Nothing here promotes
    the ≤1-ULP gated-norm difference to an explanation of the gate failure;
    the layer-3 routing-membership change and 64–71 ULP post-TP spread remain
    unexplained by it (hypothesis: amplification through routing; unmeasured).
  - The coincident `longer` row-0 failures of candidate and ref13 are a
    correlation only.
  - Host state is a point-in-time read; the concurrent `normbind` activity
    means it must be re-verified again immediately before any new session.
- Claims explicitly superseded, with reason:
  - Superseded: "both reference diagnostic containers were archived and
    stopped; neither Spark had a running container" as a *current* state
    description. Still true at 22:42Z, but `gate-ref-normbind-0/1` ran
    22:35–22:42Z and `gate-ref-0/1` were cycled at ~22:11Z today; the recorded
    shutdown receipts predate that activity.
  - Clarified (not superseded): the 282/193 suite figure is the archived
    repair-time count; the current tree runs 293/208, all green.
  - Clarified (not superseded): "HEAD `f0ba28fd` differs from `ee5eaf0f` only
    in documentation" is true of commits only; the working tree carries the
    uncommitted binding-capture changes described in observation 5.
- Q2/control verdict (or untested): **unchanged — Q2 FAILS** (exchangeability
  rank + 3 catastrophic `longer` rows; `ulp_control` FAILs; `ref13` held-out
  FAILs its `longer` row 0). Untested here: all GPU slices, experiments 1–5.
- Host shutdown/archive receipt: no hosts were changed by this pass; nothing
  to shut down. Both Sparks were verified with 0 running containers and idle
  GPUs at 22:42Z. No attempt names were burned; no captures were touched.
- Next action: coordinate with whoever ran `gate-ref-normbind-0/1` /
  `atlas-norm-replay-20261011-r1` before starting anything — adopt or
  cross-check that capture rather than duplicating it, and confirm no name
  collision. Then proceed with required experiment 1 using the existing
  `reference_norm_binding.py` + the uncommitted candidate binding capture
  (commit it first for provenance), per the runbook's fresh overlay/attempt
  discipline.

### 2026-10-11 23:20 UTC — second-pass diagnostic review (claim verification + output-level localization + live host snapshot)

- Agent / objective: Second diagnostic pass per operator instruction: verify
  the findings already recorded in this file, run another diagnostic pass,
  edit nothing except this file. Same shape as the first pass (offline
  verification + read-only host inspection), plus **new offline recomputation**
  from preserved raw logits. Facts and hypotheses separated below. Outputs
  written only to `/tmp` and to this file.
- Attempt IDs and diagnostic plan/overlay hashes: no new attempt. Archived
  identities re-verified only. A **parallel session** is executing the
  norm-binding/replay slice live on the GPUs during this pass (see observation
  10); its attempt names are not mine and were not touched.
- Source, image, binary, kernels, checkpoint/config and runtime fingerprints:
  local HEAD still `f0ba28fd`; working tree still carries the uncommitted
  binding-capture Rust changes (unchanged since first pass:
  `git diff --stat` = `forward_attention.rs` 16, `upstream.rs` +134,
  `kda.rs` 10, `ffn_io.rs` 39, `ffn_io_tests.rs` 27, plus `continue.md` /
  `todo.md` doc edits by the parallel session). Frozen harness integrity
  verified: `statistical_gate.py` = `dbd765a6…`, `compare_logits.py` =
  `6d20158c…`, `parity_protocol.py` = `545f62af…` — byte-identical to the
  `harness_sha256` recorded inside the registered gate JSON. New candidate
  image `atlas-exl3-validation:src-c12cc52e2ac2-rust1.97.1` exists on Spark 1
  (built by the parallel session; differs from burned `src-455b0da9f7cd`).
- Hosts/ranks, serve flags, request inputs and invocation/layout/TP mapping:
  read-only inspection only (no requests sent). Snapshot at 2026-10-11
  23:12Z; see observation 10.
- Variable changed; constants held fixed: none on hosts. Offline
  recomputation added one new comparison axis: **direct candidate↔ref13
  distances** and per-row ensemble spreads (script `/tmp/q2_pairwise_diag.py`,
  sha256 `b9b195d5…`, output `/tmp/q2-pairwise-diag-20261011.json` sha256
  `59257aa6…`).
- Self-reproduction and positive/negative control results:
  - Analyzer: `analyze_localization.py --root …/q2-diagnostic-20261011
    --output /tmp/q2-localization-VERIFY2-20261011.json` → exit 0,
    **byte-identical** to `resume-analyzer-20261011/q2-localization-repaired-
    20261011-r3.json` (sha256 `a2fb3f65…`, same as first pass).
  - Reviewer: `review_localization.py /tmp/q2-review-VERIFY2.json` → "helper
    controls passed; validated 176 pairs", **byte-identical** to
    `review-validation.json` (sha256 `0ec887c4…`).
  - `test_analyze_localization.py` → 8/8 pass. The four norm-harness test
    files (`test_reference_norm_binding.py`, `test_reference_norm_replay.py`,
    `test_native_gated_norm_replay.py`, `test_reference_norm_launch_binding.py`)
    → 27 tests + 31 subtests, all green.
  - Full `tests/exl3_validation/` suite at 23:00Z → **309 passed, 224 subtests**
    (110.4 s). See observation 11 for count bookkeeping.
  - **Rule-reproduction control (fact):** my independent pairwise script
    reproduces the frozen rule's registered numbers exactly — candidate
    nearest `longer` row-0 sym_kl `0.032740591222728745`, row-1
    `0.013551931051247568`, row-2 `0.09034580573518464`, and limits
    `0.005 / 0.01298892900657322 / 0.056452221505187083` match the gate JSON
    digit-for-digit; ref13 nearest row-0 `0.05603547069410249` matches the
    held-out JSON. New numbers below therefore sit on the registered footing.
- Observations, with raw evidence paths and hashes:
  1. **First-pass claims verified (fact).** Gate JSON: `passed=false`, reasons
     exactly ["exchangeability rank…", "3 catastrophic row(s)"], catastrophic
     rows = suite 0/`longer`/rows 0–2 with the quoted nearest/limit values;
     `ulp_control` FAILs with the same two reasons and same rows (nearest
     values differ only from fp re-association); `negative_controls` PASS
     (swapped-token and shifted-vocabulary both detected);
     `discrete_failures` empty; leave-one-out clean for all 9 refs.
     Held-out: ref13 FAILs exactly one row (`longer` row 0, nearest
     `0.05603547069410249` vs limit `0.005`), T_candidate `5.3011` sits
     mid-range of T_references. Per-case numbers (arithmetic 0.0736, longer
     0.0448, index-boundary 0.032–0.075, code 3.9e-5, capital 0.00123) match
     `per-case-cand02.json`. Review spot-checks match the first-pass quotes:
     `rank1.row0.gated` different=1/40960 max_ulp=1; `rank0.row0.attn_reduced`
     different=32 max_ulp=16; `moe_post_tp` p99 64–71, `rank0.row0` max_ulp
     31089 (global max 31171 at `rank1.row7`), exact fraction 45.33 %.
     Implementation claims verified: `kda_sigmoid_gated_rms_norm` at
     `kernels/gb10/common/kda.cu:565`, FP32 `rsqrtf(red/dim + eps)`, sigmoid
     gate division, one BF16 round; eps from `ctx.config.rms_norm_eps`
     (`forward_attention.rs:159`); loader `o_norm.weight` BF16 `[kd]` via
     `single_range(0..kd)` (`exl3_kda_loader.rs:311-314`); reference
     `kda.py:278` `FusedRMSNormGated(self.head_dim, activation="sigmoid")`
     with no explicit eps → class default `1e-5`
     (`fused_norm_gate.py:346`), `forward_native` (torch.rsqrt/torch.sigmoid)
     distinct from the Triton `forward_cuda`. Archived test log confirms
     282 passed + 193 subtests. Uncommitted binding capture confirmed present
     (`capture_gated_norm_binding`, schema `atlas-kda-gated-norm-binding-v1`)
     and absent from burned source `ee5eaf0f`.
  2. **Limit provenance of the failing rows (fact, new).** The catastrophic
     rule is `limit = max(3 × max pairwise ref-ref distance, floor)` with
     frozen floors sym_kl 0.005 / ce_error 0.15. For `longer` rows 0–3 the
     ensemble max pairwise sym_kl is `0.001324 / 0.004330 / 0.018817 /
     0.081074`. Row 0's limit is therefore the **floor** 0.005 (3×spread =
     0.00397 < 0.005); rows 1–3 are spread-governed (0.012989 / 0.056452 /
     0.243221). The three catastrophic rows are exactly the rows where the
     ensemble agrees most tightly. Candidate nearest/limit ratios: row 0
     6.55×, row 1 1.04×, row 2 1.60×, row 3 0.18×. Row 1 fails by only 4.3 %
     and row 3 (loudest absolute distance 0.0425) passes comfortably because
     the references themselves scatter 0.081 there. Also near-limit but
     passing: `arithmetic` row 3 (0.2096 vs 0.2371) and row 1 (0.0519 vs
     0.0634).
  3. **Candidate and ref13 do not share a failure location on the failing row
     (fact, new; refines the shared-mechanism hypothesis).** Direct
     candidate↔ref13 sym_kl on `longer` row 0 is `0.11366` — larger than
     either one's distance to the ensemble (candidate 0.0327–0.0379, ref13
     0.0560–0.0575). The two failures are three distinct points in logit
     space, not one shared "alternative mode". Across all `longer` rows the
     candidate↔ref13 distance never falls below either one's ensemble
     distance (rows 1–3: 0.0243 / 0.1174 / 0.0442). Hypothesis "the two
     failures share a mechanism" is thereby weakened; "row 0 is intrinsically
     sensitive" remains live (see 5–6).
  4. **Per-row determinism of `longer` is heterogeneous (fact, new).** Exact
     byte-identical rows occur between nominally independent reference runs:
     21 reference pairs share 27 byte-identical rows across suites, e.g. `longer/r0` cliques
     {ref05, ref07, ref09, ref11} (all pairs exact) and {ref06, ref08, ref12};
     `ref07`↔`ref11` share 3 rows; `capital/r2` is shared exactly by
     ref10/ref11/ref13. ref13 shares exact rows with ref10/ref11, so its
     capture is not wholesale corrupted — its divergence is confined to
     `longer/r0` (and marginally r1). ref-ref max |logit| difference on
     `longer/r0` is 0.2236; on that same row all nine ensemble runs agree on
     argmax 220 with identical logit 16.875. No case-level array is duplicated
     across runs, and the candidate shares **no** exact row with any reference
     anywhere. (Facts. Hypothesis: some rows execute a deterministic path
     while others are nondeterministic; row-level exactness is not a
     collection artifact — but a row-scoped collection error for ref13/r0 is
     not excluded.)
  5. **Output-level localization of the failing row (fact, new).** Sym-KL
     decomposition `longer` row 0 vs ref04: candidate total 0.03274 with 69 %
     of it in the top-6 vocab ids (2619: p 0.173 vs 0.133; 8062: 0.143 vs
     0.171; 29296: 0.041 vs 0.052; 220: 0.469 vs 0.437; 576: 0.041 vs 0.049;
     508); ref13 total 0.05728 with 86 % in the top-6, dominated by a single
     id, 8062: p 0.2566 vs 0.1710 (61 % of its total). All runs keep the same
     argmax (220), so this row has **no discrete divergence** — the failures
     are mass shifts among a small token set. Both failures move the same
     token set {2619, 8062, 220, 29296, 576} and in **different directions**
     (candidate under-weights 8062/29296/576, ref13 over-weights 8062).
     (Hypothesis, not established: row 0's distribution sits near a shallow
     boundary among these tokens, so distinct small perturbations — one
     implementation-related, one run-to-run — land on the same symptom. If
     true, symptom-size at this row is not a measure of mechanism-size.)
  6. **Case geometry (fact).** `longer` = 36-token prompt, continuation
     `[35665, 40451, 13]`, rows = prefixes 36–39 (consistent with the handoff).
     Diverging id 29296 occurs in the prompt at position 5; continuation id
     35665 is itself a contributor to ref13's row-0 divergence. ref13's
     row-0 ce_error to the ensemble spans 0.145–0.219 (nearest 0.145) against
     the ce floor 0.15 — it missed a second catastrophic trigger on the same
     row by ~3 % (fact from held-out distances; the row is marginal on two
     metrics, not one).
  7. **Dump-stage inventory refinement (fact; refines handoff experiment 1
     scope and first-pass observations 5–6).** The archived reference dumps
     already bind more than the handoff states: (a) `hook-source-*.json`
     records pin the installed module file hashes at capture time
     (`vllm.models.glm5next.nvidia.kda` = `ec090aab…`, `…quantization.exl3` =
     `6a062458…`, `…nvidia.model` = `a1214289…`, hook = `4774dc69…`, layers
     0–3, "exact-byte diagnostic hooks, original outputs returned
     unchanged"); (b) per-invocation `mhc-fused-policy` / `mhc-pre-policy`
     records bind `rms_eps 1e-5`, `norm_eps 1e-5`, `hc_pre_eps 1e-6`,
     `hc_sinkhorn_eps 1e-6`, `hc_post_mult_value 2.0`, `sinkhorn_repeat 20`.
     Still genuinely unbound for experiment 1: the reference resident
     `o_norm.weight` bytes (the only reference resident weights captured are
     `kda-resident-extra-weight` `[288,4096]` bf16, `kda-resident-g_b_proj-
     weight`, `kda-resident-f_b_proj-weight` — no `[128]` norm weight), any
     KDA-specific eps record (the mHC policy record is not the KDA o_norm),
     and the CustomOp selected forward method (`forward_native` vs Triton
     `forward_cuda`) / compilation-launch settings. Candidate dumps contain a
     `gated_norm` stage with one output tensor (`gated` `[10,4096]` bf16) and
     no binding record (re-verified: `gated_norm_binding` absent from
     `dumps/`); the reference side does have `kda-gated-norm-input0/input1`
     (core, g2) plus `kda-gated-norm` output for all 11 invocations.
  8. **Reviewed-artifact decomposition (fact; refines first-pass item 4).**
     `review-validation.json` holds exactly 176 pair entries = 128 floating
     (core, g1, g2, gated, oproj_input, attn_reduced, ffn_input, moe_post_tp
     × 16 rank/row slots) + 16 `ids` + 32 replicated-reference
     (`kda-o-proj_ref_replicated`, `post-tp-output_ref_replicated`). The
     first-pass phrase "128 floating pairs and 16 expert-ID comparisons"
     describes a subset of what the reviewer validated. `reference_probe-
     nonexact` shows 2 rank-level entries (`rank0.L3`/`rank1.L3`
     `routed-weighted-f32`, max_abs `2.38e-7`/`4.77e-7` = 1–2 f32 ULP) —
     consistent in kind with the handoff's within-process routed-f32
     variation claim, but the "three probes" count is not visible in this
     artifact (partially verified).
  9. **All quoted reproduction claims hold (fact).** Analyzer and reviewer
     outputs are byte-identical on re-execution (hashes in the controls
     section); the 464-pair load claim is consistent with the analyzer log
     and the byte-identical outputs. No contradiction found in anything the
     first pass asserted; the corrections in items 7–8 are refinements, not
     refutations.
  10. **Live host snapshot 2026-10-11 23:12Z (fact; read-only).** The
      parallel session is **actively running the norm-binding/replay slice**
      right now. Spark 0 `gx10-eb0e`: `gate-ref-normbind-r5-0` created
      22:54:37Z, finished 23:11:43Z, exit 0; `gate-ref-normbind-0` created
      22:35:27Z, finished 22:41:12Z, exit 0; `norm-dispatch-inspect-r2`
      22:45:43–22:45:49Z exit 0; `norm-binding-preflight-r2` exit 1,
      `-r3` exit 0, `-r4` exit 1 (22:50:11–28Z), `-r5` exit 0; GPU 0 % at
      snapshot, 0 running containers. Spark 1 `gx10-235f`:
      `gate-ref-normbind-r5-1` created 22:54:39Z, finished 23:11:53Z, exit 0;
      `gate-ref-normbind-1` 22:35:26–22:42:23Z exit 0;
      `norm-build-attestation-20261011` created 23:10:35Z (state Created,
      not running) on the new image `atlas-exl3-validation:src-c12cc52e2ac2-
      rust1.97.1`. Replay outputs `/tmp/atlas-norm-replay-20261011-r1`
      (22:35:26Z) and `/tmp/atlas-norm-replay-20261011-r5` (22:54:39Z, contains
      `gate-ref/`) are on **Spark 1**. Two preflight attempts failed (r2, r4,
      exit 1) and were retried — failures must be preserved in that session's
      receipts. Everything observed is post-shutdown-receipt activity again;
      host state must be re-verified before any further session.
  11. **Test-count bookkeeping (fact + limits).** 282/193 (archived repair
      log) → 293/208 (first pass) → 309/224 at 23:00Z (this pass). The tree is
      being edited concurrently: `test_reference_norm_launch_binding.py`
      appeared 22:51:30Z (5 tests + 4 subtests),
      `test_reference_norm_binding.py` grew from 11 to 14 tests since the
      first-pass quote, and `norm_replay_artifacts.py` +
      `test_norm_replay_artifacts.py` (3 tests + 8 subtests) landed 23:17:43Z
      — after my suite run. Counts are snapshots of a moving tree; exact
      per-file attribution is not recoverable (no per-file run log was kept at
      first-pass time). No bearing on Q2 or on the analyzer-repair claim.
- Interpretation and limits:
  - This pass added no GPU measurements and no runtime changes. The new
    numbers are offline recomputations from preserved registered logits
    (`statistical-gate-20261011/runs/`, manifests and harness hashes above),
    and the recomputation reproduces the frozen rule's published numbers
    exactly (control above).
  - The dominant final-logit cause remains unproven. Nothing here promotes
    the ≤1-ULP gated-norm difference to an explanation; the layer-3 routing
    change and 64–71 ULP post-TP spread remain unexplained by it.
  - New interpretive weight (hypothesis): the failing row is a tight-limit,
    high-sym_kl-concentration row where distinct perturbations produce
    similar symptom sizes. It follows that reducing the candidate's row-0
    sym_kl is the qualification target, but a small sym_kl at this row is not
    by itself evidence that the underlying mechanism was found. Pair with the
    internal captures (experiments 1–3) before claiming causality.
  - ref13's row-0 failure is not explained. What changed is the evidence
    distribution: three distinct points (ensemble / candidate / ref13), ref13
    otherwise reproducing ensemble rows exactly, and its divergence
    concentrated in one token's probability (8062: 0.257 vs 0.171). A
    row-scoped collection/alignment error and a genuine run-to-run mode are
    both still open; the concentration of ref13's error in a single token is
    the new lead for experiment 4.
- Claims explicitly superseded, with reason:
  - Superseded (first-pass observation 9): "/tmp/atlas-norm-replay-20261011-r1
    appeared on Spark 0 at 22:35Z" → the dir is on **Spark 1**
    (`gx10-235f`, mtime 22:35:26Z), and `-r5` is there too. Either the
    first-pass attribution was wrong or the path was relocated; current
    location is fact.
  - Refined (first-pass item 4): "128 floating pairs and 16 expert-ID
    comparisons" → the reviewer validates 176 pairs: 128 floating + 16 IDs +
    32 replicated-reference pairs (observation 8).
  - Refined (this file, "What the corrected diagnosis establishes" / first
    pass item 3): the open hypothesis "case `longer` row 0 may be
    intrinsically unstable across fresh processes, or the two failures share
    a mechanism" splits: the shared-mechanism branch is argued against by the
    0.1137 candidate↔ref13 distance (observation 3); intrinsic row
    sensitivity is now quantified (observations 2, 4, 5) but not proven.
    Required experiment 4 remains the arbiter.
  - Clarified (not superseded): suite counts are moving targets because a
    parallel session is adding test tooling (observation 11).
- Q2/control verdict (or untested): **unchanged — Q2 FAILS** (exchangeability
  rank + 3 catastrophic `longer` rows; `ulp_control` FAILs; `ref13` held-out
  FAILs its `longer` row 0). Untested here: all GPU slices, experiments 1–5.
- Host shutdown/archive receipt: no hosts were changed by this pass; nothing
  to shut down. Snapshot 23:12Z: 0 running containers on both Sparks, GPUs
  idle; one Created container (`norm-build-attestation-20261011`) and fresh
  r5 captures belong to the parallel session. No attempt names burned; no
  captures touched; outputs confined to `/tmp` (`q2-pairwise-diag-20261011.*`,
  `q2_pairwise_diag.py`, `q2-localization-VERIFY2-20261011.json`,
  `q2-review-VERIFY2.json`) and this file.
- Next action: (1) coordinate with the live normbind session before anything
  else — its `gate-ref-normbind-r5-*` captures and `src-c12cc52e2ac2`
  attestation image likely complete experiments 1–2's capture half; adopt,
  don't duplicate, and get the r2/r4 preflight failures into its receipts.
  (2) For experiment 4, use observation 5's token-level decomposition
  (ref13's error concentrated in vocab 8062; candidate's spread over
  2619/8062/29296/220/576) as the fingerprint to test against fresh reference
  repeats: does a fresh reference process reproduce either mode, and does any
  collection/alignment fault reproduce the 8062 shift? (3) Keep experiments
  2–3 as written; the candidate↔ref13 separation (observation 3) means a norm
  fix that improves the candidate does not automatically explain ref13.


## Runtime-bound gated-normalization replay — 2026-10-12T00:09:01.747937+00:00

**Outcome: replay complete; Q2 remains FAIL.** This diagnostic establishes the
local arithmetic difference under captured runtime constants. Its downstream
causal contribution has not been measured; no production numerical fix or
qualification rerun is claimed.

Evidence root:
`/home/colum/Workspace/atlas-exl3-validation-20261010/q2-norm-replay-20261011/`.
Reproduce the comparison with `python3 <evidence-root>/analyze_norm_replays.py`;
see `norm-replay-summary.json` and `evidence-hashes.json` for row results and
file identities. Each `fixtures/rankN-rowI/fixture.json` binds the exact request,
resident tensors, selected method, installed source, compiled launch and
self-replay receipts. Full captures, request/response records, overlays,
logs, inspections, hardware identities and test logs are preserved there.

### Changes and run identity

- Candidate captures resident BF16 norm weights, effective f32 epsilon bits,
  activation/affine/bias/residual flags, actual kernel handle, module/symbol
  and shared production launch geometry. Epsilon is converted once and the
  same value is captured and launched. Disabled diagnostics perform no I/O.
- Reference wraps the selected resident forward method; clones inputs before
  its in-place output alias can overwrite them; snapshots resident parameters;
  records installed source/JIT wrappers and actual compiled PTX/cubin plus
  metadata. Self-replays use independent parameter maps and fresh operands.
- Candidate native replay loads the exact PTX attested as embedded in the
  executable, rather than recompiling source or substituting a formula.
  It verifies the seven-argument ABI, geometry, finite output, operand
  immutability and output guard; input/gate/weight perturbations must engage.
- Clean diagnostic commit: `0dc3881e1837d4aa38fcd6639a2b12e62b53be68`, based on
  tested `ee5eaf0f19370f0dc302565d46b834a88630afc8`. Main workspace retains the
  five Rust edits plus Python diagnostic tooling; main was not switched.
- Fresh no-cache image: `atlas-exl3-validation:src-c12cc52e2ac2-rust1.97.1`,
  immutable `sha256:f4678136162f676d763d2a166a8e5eb962994bd6e4b9e8e437326c8a9c873bb0`.
  Server SHA256: `8842aff498390b3ddf5a1dbb97882a305d04a52489a9798d89f45e438f498905`.
  Actual build-source archive SHA256:
  `c12cc52e2ac2cf3dfd2a99dc621fa3ed4af554b2f9b85a160dc5febd2bd69596`.
  Local/remote archive difference is only the non-build LFS demo asset;
  `build-source.json` records it. All 232 PTX hashes, flags and toolchain
  match the tested candidate image. Gated-norm PTX SHA256:
  `95e5df1895c23edee7eebb4c351e5357cc319797524a60edb7f57a87ff887991`.
- Reference: unchanged image
  `sha256:6eb42fa20c145cabf9f788d7b850d8d3416706ad1955a003448b48962cb2845b`.
  Successful reference attempt `ref-normbind-20261011-02` (r5 overlay);
  candidate attempt `cand-normbind-0dc3881e-01`; standalone native containers
  `norm-native-replay-rank0-01` and `norm-native-replay-rank1-01`.

### Observed results (these fixtures only)

- All 16 rank/prefix pairs have byte-identical core, gate and resident norm
  weights. Effective epsilon agrees at the f32 ABI boundary: bits `925353388`,
  value `9.999999747378752e-06` (reference nominal epsilon `1e-5`).
- Reference: 11/11 exact self-replays per rank, including the three repeated
  arithmetic-prefix probes. All three operand controls engage.
- Candidate: 8/8 exact embedded-PTX self-replays per original physical rank;
  all three operand controls engage in every fixture.
- Instrumentation checks against the previous diagnostic captures: reference
  core/gate/output exact in 48/48 comparisons; candidate selected captured
  tensors exact in 64/64. These checks cover those tensors, not every possible
  runtime behavior or performance characteristic.

| Case / prefixes | Rank | Unequal output elements per prefix | Max signed BF16 ULP |
|---|---:|---:|---:|
| arithmetic / 10–13 | 0 | 0 | 0 |
| arithmetic / 10–13 | 1 | 1 | 1 |
| longer / 36–39 | 0 | 6 | 1 |
| longer / 36–39 | 1 | 0 | 0 |

**Source-assumption correction:** installed dispatch is
`vllm.third_party.flash_linear_attention.ops.kda.FusedRMSNormGated.forward_cuda`,
source SHA256 `ac2260c84a36936ad7d56ef63dbceb4618b2c499d7637e08b407f0cd706f9d02`.
It uses contiguous gates, `BT=32`, `BD=128`, four warps, compiled
`enable_fp_fusion=true`, `launch_pdl=false`. The older extracted
`fused_norm_gate.py` with eight warps/BT16/strided gates is not the selected
implementation and must not anchor an arithmetic repair.

The first full reference attempt failed because config lookup occurred outside
its initialization context; it was archived and stopped. r2 preflight exposed
JIT-wrapper identity handling; r3 passed self-replay; r4 failed a launch contract
inferred from that older source; r5 binds the actual installed implementation
and passes. Overlay versions are preserved in `preflight-overlays/`; use the
successful r5 capsules for conclusions. These failures are diagnostic tooling
failures and do not establish a model defect.

### Next repair experiments

1. Use these immutable fixtures to vary one arithmetic feature at a time:
   reduction tree/order, reciprocal-square-root implementation, sigmoid
   exp/reciprocal implementation, and multiply/divide ordering or contraction.
   Candidate source uses `rsqrtf` and division by `1+expf(-g)`; installed
   reference source uses `1/tl.sqrt` and multiplication by `tl.sigmoid`.
   Inspect the bound compiled PTX before attributing a source-level mechanism.
   Keep the exact native baseline and all engagement controls as anchors;
   record changed elements and coordinates across both ranks/all prefixes.
2. Before treating norm matching as a Q2 repair, replay local `o_proj` before
   TP with original versus reference-exact norm output, holding resident
   projection weights and GEMM arithmetic fixed. Test each engine against
   its own captured local projection before comparing them.
3. Perform a fresh controlled downstream intervention on both ranks: replace
   only the norm output and trace post-TP projection, routing membership,
   MoE and logits. Improvement or no improvement determines attribution;
   the existing one-ULP difference alone does not prove dominance.
4. A proven numerical repair needs a fresh build and the frozen registered
   Q2 qualification/control suite. Preserve thresholds, reference ensemble,
   held-out runs and unsuccessful attempts. Continue the independent
   longer-row reference-stability investigation.

Final checks: 313 Python tests + 235 subtests pass on the final tooling snapshot;
2 runtime binding tests and 2 model binding tests pass. Real no-cache CUDA build
and all 16 native GPU replays pass. Existing unrelated host clippy/metal-stub
failures remain outside this slice; this is not a full PR validation claim.
Both serving pairs are archived/stopped; native replay containers exited zero.
Both post-replay `docker ps` receipts show zero running containers.
Remote staging is `~/gate-norm-replay-20261011-r1/` (candidate/native) and
`~/gate-norm-replay-20261011-r5/` (reference) on the two Sparks. Connect through
`ai-node` (observed address `100.121.137.95`), targets `10.50.0.10/11`.

### 2026-10-12 04:40 UTC — dual independent code-path audits + Q2 gap report (offline)

- Agent / objective: Operator-directed review of the reference vLLM source for
  the exact GLM-5.3-Flash EXL3 code path, via two independent audits
  (`codex-exec`, isolated fresh contexts, identical briefs), compared by the
  parent session and mapped against the Q2 failure state to identify missing
  evidence. Offline only; no GPU work, no code changes.
- Attempt IDs and diagnostic plan/overlay hashes: no new attempt. Audits read
  hash-pinned sources only (`vllm-reference-src` checkout verified == installed
  module records `ec090aab…`/`a1214289…`/`ac2260c8…`/`6a062458…`/`4774dc69…`).
- Source, image, binary, kernels, checkpoint/config and runtime fingerprints:
  none new. Source identities verified above; compiled-binary binding still
  absent (retained limit of both audits).
- Hosts/ranks, serve flags, request inputs and invocation/layout/TP mapping:
  no hosts touched. Primary-record spot-checks this session: candidate create
  command carries `--kv-cache-dtype fp8_g128` + `--lm-head-dtype default`;
  reference carries `KV_CACHE_DTYPE=auto` + `--kv-cache-memory-bytes` +
  `--enable-chunked-prefill`; `LAYER_MAP.json` exists (`spark-glm53/ablit/`).
- Variable changed; constants held fixed: none (analysis-only pass).
- Self-reproduction and positive/negative control results: both audits
  independently re-verified all pinned source hashes (no mismatch) and both
  rejected the brief's `fp8_g128`-for-reference premise; agreement on the
  merged step inventory (embedding → mHC → KDA → MLA/indexer → FFN/MoE →
  final norm → 1-row lm_head → captured logits → sampler).
- Observations, with raw evidence paths and hashes:
  1. Full audit pair + comparison report: `q2-codepath-audit-20261012/`
     (`audit-a.md` `b83d8554…`, `audit-b.md` `6713c669…`,
     `comparison-report.md`). The report's §5 is the deliverable: gap matrix +
     prioritized "what is missing" (G1–G9).
  2. New mechanism-relevant facts (audit findings, source-level): reference
     thin expert path (≤32 rows — what the gate prompts execute) computes
     half2 SiLU with **post-SiLU** clamp and FP16 intermediates with FP32
     `atomicAdd` accumulation (order-sensitive); grouped-fat uses **pre-SiLU**
     clamp; router top-k has **no stable tie-break**; lm_head runs a 1-row
     packed EXL3 GEMM with disputed min-row padding (audit-a vs audit-b, D4);
     mHC has a ≤16-row rounding-path split; reference and candidate run
     **asymmetric KV cache dtypes** (auto vs fp8_g128).
  3. The captured-quantity boundary is confirmed: gate captures post-gather,
     pre-sampler logits (BF16 values widened to FP32); sampler/tie behavior
     cannot affect the compared rows.
  4. Largest evidence gaps: layers 4–44 entirely unobserved (G1); final norm +
     1-row lm_head never replayed (G2); router arithmetic unbound (G3); expert
     same-input replay never done (G4); KV asymmetry unexamined — live suspect
     for the 2048-token long suite (G5).
- Interpretation and limits: audits are source-level; runtime dispatch
  ambiguities remain open. No mechanism promoted; dominant final-logit cause
  still unproven. Audit discrepancies D3 (sampler V1 vs V2 — V2 is the bound
  path, runner hash matches preflight) and D4 (unresolved) recorded.
- Claims explicitly superseded, with reason: the brief's "reference FP8 paged
  KV (fp8_g128)" premise — wrong for the reference engine (its cache dtype is
  `auto`; `fp8_g128` is the candidate's flag). Any prior implicit assumption
  of symmetric cache dtype between engines is superseded by the create-command
  records.
- Q2/control verdict (or untested): **unchanged — Q2 FAILS**; all GPU slices
  still untested this session.
- Host shutdown/archive receipt: no hosts touched; nothing to shut down.
- Next action: use the gap list to sequence the remaining required
  experiments: (1) commit + review the ablation harness (in flight), run the
  gated-norm ablation; (2) design the router/expert same-input replay with
  binding records (G3/G4); (3) final-norm + 1-row lm_head replay (G2,
  resolves D4); (4) retrieve checkpoint `config.json` (G8, cheap); (5) scope
  the long-suite cache-tracing slice (G5) with the operator before running it.

### 2026-10-12 04:25 UTC — gated-norm arithmetic ablation (GPU replay, both ranks)

- Agent / objective: Required next-repair experiment 1 — attribute the ≤1-ULP
  cross-engine gated-norm difference by replaying immutable runtime-bound
  fixtures through single-feature candidate-kernel variants. Harness built,
  independently reviewed (blocker fixed: `__shared__` asm output → local
  copy-out; review in `q2-norm-ablation-20261012/harness-review-20261012.md`),
  committed (`f7ac1810`), then executed on both Sparks.
- Attempt IDs and diagnostic plan/overlay hashes: containers
  `norm-ablation-replay-rank{0,1}-01`, staging `~/gate-norm-ablation-20261012-r1/`
  on both hosts (fresh names; preflight: names free, image present, 0 running
  containers). Fixtures = immutable `q2-norm-replay-20261011` set (16 pairs).
  Durable evidence: `q2-norm-ablation-20261012/` (archives + per-row receipts +
  `evidence-hashes.sha256`).
- Source, image, binary, kernels, checkpoint/config and runtime fingerprints:
  harness commits `f7ac1810`/`f1736ae5`; staged `common/kda.cu` sha256
  `bf2149c3…` verified on both hosts; attested embedded PTX binding unchanged
  (`replay-ptx/binding.json`, gated-norm PTX `95e5df18…`); replay image
  `glm53-flash-mikeroz:dt-carry20`; variant sources derived at runtime by
  exact-string transforms with per-variant sha256 in receipts. No server runs;
  no checkpoint access.
- Hosts/ranks, serve flags, request inputs and invocation/layout/TP mapping:
  standalone replay containers, `--network none`, all GPUs, `/work:ro` +
  `/out`; rank0 on `gx10-eb0e`, rank1 on `gx10-235f`. Fixture operands and
  epsilon identical to the bound captures (equal-operand receipts in
  `fixture.json`).
- Variable changed; constants held fixed: one arithmetic feature per variant
  (reduction tree/order; rsqrt → `sqrt.approx.ftz`+`div.full`; expf →
  `mul-log2e`+`ex2.approx`; divide → `div.full`-reciprocal multiply), plus the
  composition `v-all`. Operands, epsilon, grid/block, ABI (7-arg) fixed.
- Self-reproduction and positive/negative control results: **attested
  self-reproduction byte-exact 16/16; `v0-copy` source reproduction byte-exact
  16/16 (both anchors); all input/gate/weight perturbation controls engaged on
  attested/v0-copy/v-all; both drivers exited 0**; `interpretation_blocked=false`.
  Baseline unequal pattern reproduces the norm-replay captures exactly (rank0
  longer 6×1ULP per row; rank1 arithmetic 1×1ULP per row).
- Observations, with raw evidence paths and hashes:
  1. **`v-all` reproduces the reference engine's gated-norm output
     byte-exactly on all 16 rank/prefix pairs (0 unequal elements, 0 ULP).**
     The ≤1-ULP cross-engine difference at this stage is fully attributable to
     the combined reciprocal-square-root + sigmoid reciprocal/divide
     arithmetic forms. (`native-ablation-rank{0,1}/r1/ablation-summary.json`,
     sha256 `06dde794…`/`014f3aad…`.)
  2. v1 (reduction order) and v3 (exp construction) are output-inert on all
     pairs — not sources. v2 alone: rank0 6→2 per longer row, rank1 unchanged.
     v4 alone: rank1 1→0 (fully reconciled), rank0 6→7. The two active
     features interact; no single feature reconciles rank0.
  3. Reference-side compiled forms confirmed as the reconciliation target:
     `sqrt.approx.ftz.f32` + `div.full.f32(1.0, root)`; combine
     `product * div.full.f32(1.0, 1+e)`. Candidate `rsqrtf` +
     `product / (1+expf(-g))` differs in exactly these forms (PTX inspection,
     `ptx-inspection/inspection.md`).
- Interpretation and limits: attribution holds for the **gated-norm stage and
  these fixtures only**. Downstream causality is unmeasured — the handoff's
  warning stands: do not infer the one-ULP difference explains the gate
  failure. `v-all` contains the inert v1/v3; the minimal sufficient change
  (v2+v4) is unproven. Any production fix triggers step-5 obligations (tests,
  clean build, arithmetic-neutrality proofs, fresh frozen Q2 qualification).
- Claims explicitly superseded, with reason: none (consistent with the
  norm-replay stage result; refines it from "≤1 ULP consistent with
  arithmetic-form differences" to "exactly reproduced by the composed
  reference-form arithmetic on these fixtures").
- Q2/control verdict (or untested): **unchanged — Q2 FAILS**; no gate slice
  run. This closes required experiment 2's cross-implementation question at
  the norm stage (self-reproduction + controlled attribution).
- Host shutdown/archive receipt: both replay containers `Exited (0)`,
  preserved; 0 running containers and idle GPUs on both Sparks at 04:19Z
  (`rank{0,1}-container-and-host-receipt.json`).
- Next action: (1) `v24` minimal-variant run (v2+v4) to prove the minimal
  fix; (2) next-repair #2 o_proj replay with original vs reference-exact norm
  output; (3) next-repair #3 downstream intervention on both ranks to measure
  whether the corrected norm reduces the registered Q2 failures.

### 2026-10-12 04:50 UTC — v24-minimal probe (GPU replay, both ranks)

- Agent / objective: minimal-fix probe for the previous entry's attribution —
  compose only the two active features (v2 rsqrt form, v4 reciprocal-multiply
  combine), dropping the proven-inert v1/v3, and test byte-exactness against
  the reference outputs.
- Attempt IDs and diagnostic plan/overlay hashes: containers
  `norm-ablation-replay-rank{0,1}-02`, staging `~/gate-norm-ablation-20261012-r2/`
  (copy of r1 with updated `gated_norm_variants.py`, sha256 `d578d7a5…`
  verified on both hosts; r1 preserved untouched). Commit `d04d8bfa`.
- Source, image, binary, kernels, checkpoint/config and runtime fingerprints:
  as previous entry (same fixtures, binding, frozen source `bf2149c3…`, replay
  image); variant `v24-minimal` source sha256 `7e8404ac…`.
- Hosts/ranks, serve flags, request inputs and invocation/layout/TP mapping:
  identical to previous entry (standalone replay containers, same operands).
- Variable changed; constants held fixed: variant set now includes
  `v24-minimal` (v2+v4 composition); everything else unchanged.
- Self-reproduction and positive/negative control results: anchors byte-exact
  16/16 both ranks (`attested`, `v0-copy`), all controls engaged,
  `interpretation_blocked=false`, both drivers exited 0.
- Observations, with raw evidence paths and hashes: **`v24-minimal` reproduces
  the reference gated-norm output byte-exactly on all 16 rank/prefix pairs
  (0 unequal elements, 0 ULP)** — identical to `v-all`
  (`v24-rank{0,1}/ablation-summary.json`, sha256 `6b606e07…`/`f8ca6080…`;
  archives `native-ablation-v24-rank*.tar.gz`).
- Interpretation and limits: the minimal sufficient correction at this stage
  is exactly two arithmetic changes (norm reciprocal form + gated reciprocal
  combine). With v1/v3 inert individually and v24 == v-all, no further
  interaction is present at the gated-norm stage. Same limits as the previous
  entry: stage-local attribution; downstream causality unmeasured.
- Claims explicitly superseded, with reason: analysis-20261012.md conclusion 4
  ("minimal change unproven") — superseded by this run.
- Q2/control verdict (or untested): **unchanged — Q2 FAILS**; no gate slice
  run.
- Host shutdown/archive receipt: containers `norm-ablation-replay-rank{0,1}-02`
  `Exited (0)`, preserved; 0 running containers, GPUs idle on both Sparks at
  collection (`v24-rank{0,1}-host-receipt.txt`).
- Next action: next-repair #2 — local o_proj replay before TP with original vs
  reference-exact norm output (each engine vs its own captured projection
  first), then #3 downstream intervention on both ranks to measure whether
  the corrected norm reduces the registered Q2 failures.

### 2026-10-12 — independent offline review of `614a2e24`

- Scope: read-only review of the handoff, recent binding/replay/ablation code,
  frozen gate reports and archived arrays. No new GPU experiment, production
  arithmetic change, image publication or live host inspection.
- Evidence independently checked: all 16
  `q2-norm-replay-20261011/fixtures/rankN-rowI/fixture.json` manifests and
  their array hashes; reference/candidate core, gate, weights and effective
  epsilon equality; raw `q2-norm-ablation-20261012/v24-rankN/rankN-rowI/`
  outputs (`attested`/`v0-copy` vs candidate expected, `v24-minimal`/`v-all`
  vs reference expected). All checked equalities and hashes passed. Archived
  controls passed. This supports only the fixture-local arithmetic result.
- Re-ran `analyze_localization.py` against `q2-diagnostic-20261011/`: all
  464 stage pairs completed. For every longer prefix on both ranks,
  `L0.kda_reduced` and `L0.out_after_tp` are exact, but `L1.ffn_input`
  differs. Prioritize the intervening mHC state/L1 attention boundary; the
  exact L0 branch outputs do not alone prove equal full residual streams.
- Focused CPU validation: 53 tests + 42 subtests passed across
  `test_analyze_localization.py`, `test_gated_norm_ablation.py`,
  `test_native_gated_norm_replay.py`, `test_norm_replay_artifacts.py`,
  `test_reference_norm_binding.py`, `test_reference_norm_launch_binding.py`
  and `test_reference_norm_replay.py` (`python3 -m pytest -q -p no:cacheprovider`).
- Open P2 tooling defects (not repaired in this documentation revision):
  1. `gated_norm_ablation.build_summary` checks anchors but drops engagement
     controls. Reproducer: set all required controls to `engaged=False` in
     existing `_synthetic_rows()` records; the saved summary still reports
     `interpretation_blocked=False`. The CLI later raises, so its exit status
     and summary disagree. Preserve controls and block the summary on missing
     or failed controls; add this regression test.
  2. The runner reads candidate epsilon and arrays without enforcing fixture
     hashes, row identity, actual cross-engine operand equality or epsilon
     agreement. The fixture builder intentionally permits differing operands
     as evidence. Reject those inputs before arithmetic attribution and add
     mismatch controls. Independent checks above protect the existing result.
- Correction to the 2026-10-12 code-path-audit entry and D1/G5: different
  flags (`auto` versus `fp8_g128`) do not prove effective dtype asymmetry.
  `docs/design/exl3-statistical-gate.md`, pre-collection candidate-profile
  addendum, already records `auto` resolving to `fp8_ds_mla`; Atlas's format
  was chosen to match it. Effective write/read/QDQ/attention paths still need
  comparison, especially for chunked long prompts. The historical audit
  reports remain preserved and must be read with this correction.
- Plan revision: C2 now starts at layer 0 only, binds every substituted
  request/rank/invocation, reports full-vocabulary metrics against every
  frozen reference and retains mixed outcomes. Baseline/treatment repeats
  check determinism; a short diagnostic slice cannot qualify Q2. Complete
  G8/G9 binding before affected replays. The L1 escalation precedes broad
  late-layer investigation. No GPU scope extension or gate change is implied.
- Verdict: candidate gate and held-out ref13 reports still fail. Dominant
  final-logit cause and downstream benefit of v24 remain unmeasured.
- Lesson: compare effective dispatch and operands, not flag spellings or
  source defaults; a preserved summary must reflect every required control.

### 2026-10-12 — vLLM report reconciliation after independent review

Scope: reviewed `q2-codepath-audit-20261012/audit-a.md`, `audit-b.md` and
`comparison-report.md`, then checked installed source, archived runtime logs,
earlier component receipts and `q2-investigation-20261011/report-mla-implementation-delta.md`.
Paths below are relative to the durable evidence root unless prefixed `vllm/`
(relative to `/home/colum/Workspace/vllm-reference-src/`) or described as
in-repo. Original reports and their hashes are preserved; this entry supersedes
the specific claims below. No GPU run, new image or gate change.

1. **Short-prefill attention is cache-backed on the bound reference backend.**
   The merged inventory's "fresh short prefill reads live latents, not cache"
   adopted audit-b §32 while overlooking audit-a §28's incompatible backend.
   Both norm-replay reference logs name `FLASHINFER_MLA_SPARSE_SM120` and
   `fp8_ds_mla` (`q2-norm-replay-20261011/ref-rank0/docker.log:92–93`, rank1
   lines 73–74); the held-out reference log agrees
   (`statistical-gate-20261011/batch2-attempts/spark1/ref13/docker-rank1.log:73–74`).
   In `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py:36`,
   `supports_dense_mha_prefill=False`; `vllm/model_executor/layers/attention/mla_attention.py:550–555`
   consequently disables the dense-prefill backend, and lines 824–826 move
   all tokens to MQA. The SM120 backend's lines 163–177 pass packed cache to
   FlashInfer. Source SHA256s: backend `d665ef2109b0183d48e3541ecd24e9fa8e1dc3e410983bc29b8d997af9d7cd01`;
   MLA wrapper `a309ca752b01e85360b268fba56245b816ff245340e9e361166ee96b99d93a94`.
   **Action:** examine short-case cache writes/reads and QDQ too. Selecting
   every causal index does not imply bypassing cache quantization. Actual
   cache bytes and FlashInfer internal arithmetic remain unbound; this is a
   dispatch correction, not a demonstrated Q2 cause.

2. **The checkpoint configuration is recoverable from existing evidence.**
   `vllm-reference.json` (SHA256
   `d469694511abaeb473891f06201f3505e7d36bd41c4bdf8d1a7d34e98ef57701`)
   contains the full `config`. Serializing it with
   `json.dumps(record['config'], indent=4).encode()` and no trailing newline
   produces 86,532 bytes whose SHA256 is
   `00e4a3aad5a2348d5a0ef8c6a1d8e8a945e77e2a1fddc1d7ddd6560f29e07066`.
   Independently matched this digest and checkpoint revision
   `dda99c7e935920fa6e235ce410e6f806e1c02720` against **all 24** suite manifests
   referenced by the frozen candidate and held-out gate reports (ref04–ref13,
   cand02/cand03, short16+long). The original audits' restricted input set did
   not include it; "configuration unavailable" is no longer an active gap.
   It supplies the 45-layer EXL3 map (MLA 3,7,…,43; dense FFN 0–2),
   `index_topk=2048`, `index_kpool=4`, 32 indexer heads, NoPE/rope dimension 0,
   RMS epsilon 1e-5, SwiGLU limit 10, `moe_router_dtype=float32`, groups 1/1,
   top-8 and routed scale 2.5. **Action:** reuse these exact checkpoint bytes;
   bind runtime overrides/providers separately. The enclosing JSON's older
   context/speculation/environment is not the gate's serving profile.

3. **Head padding is resolved and head component work already exists.**
   Installed plugin `arithmetic-audit/atlas-reference-active-exl3_mul1.py:3232`
   explicitly swaps `lm_head` with `pad=False`; line 3097 requires `self.pad`
   before min-row padding. Its pinned SHA256 remains `6a062458…dcaf`.
   `arithmetic-audit/linear-head-m1-production13-rank0-results/report.json`
   (SHA256 `e4a576daf7f994a01a740e2d997c2cabcac7445f245fbdae23dd6dd86d947d05`)
   records min_rows=3, pad=false, actual rows=1, K=6, cooperative autotuner
   tag 2, tile M/K/N=16/32/128 and grid [48,1,1]. Rank1 receipt agrees
   (SHA256 `2f126fc8165a6093e25c8f84ef214fb7ed3a6543a5cbfe20ea82692debf7e316`).
   These report exact experimental MMA outputs; preserve the failed scalar
   baseline distinction. The independent bit-check receipt
   `linear-head-m1-production13-independent-bit-check.json` has SHA256
   `559b4df5762312754b12cbe4ab5d6c2495ebfbdac3538ae30959993c0ab24995`.
   **Limit:** NVRTC production-source component runs using one captured rank-0
   head input for both logical ranks, not current gate-worker PTX or separate
   rank-1 input qualification. **Action:** rebind/extend those controls;
   investigate final RMSNorm separately. D4's min-padding dispute and G2's
   blanket "never replayed" statement are superseded at this scoped level.

4. **Reuse earlier router/expert controls and bind the missing route stages.**
   In-repo `docs/design/exl3-support.md` already records thin/fat same-input
   controls and coherent-13.1 router controls. The retained
   `arithmetic-audit/router-coherent131-controls-attempt2/root-independent-router131-raw-check.json`
   (SHA256 `1884f2d6ae0d7030802ada5f5622bd8ca1547b31bf715a49379869bc65aceb2b`)
   reports 369 checks covering both ranks at M1/M5/M10/M128 and exact captured
   router replay with BF16-representable logits. The
   `arithmetic-audit/fat178-production19-rank{0,1}-results/report.json` receipts
   cover the single-expert 32/33-row boundary (SHA256s `d94420e8…83` /
   `908a572b…c2ac`). These are historical component receipts, not fresh
   numerical replays or current full-model parity.
   Installed plugin lines 1676–1682 cast route weights to **FP16 before
   sorting** local expert IDs; lines 1703–1729 choose thin/grouped work, and
   line 1368 selects actual fat rows via `counts > cap`. **Action:** rebind
   earlier controls to failing Q2 inputs, and capture FP16 route-weight bytes,
   local IDs, token permutation, expert counts and pre-BF16 routed FP32 sums.
   Comparing membership alone cannot isolate these stages. Do not infer
   unstable sort execution merely because no stable-sort option is requested.

5. **Actual L3 occupancy narrows the thin/fat question for these fixtures.**
   Offline recomputation used `analyze_localization.load/tensors` on
   `q2-diagnostic-20261011/dumps/{ref,cand}-rankN/` (reference `diagnostic/`),
   layer 3 invocations 0–7, reference `route-ids` and candidate `router.ids`.
   Decode as i32, validate IDs in [0,288), and compute
   `np.bincount(ids.reshape(-1), minlength=288)` per request. Both ranks give:

   | Prefixes | Reference maximum count per expert | Candidate maximum count per expert |
   | --- | --- | --- |
   | arithmetic 10/11/12/13 | 10/11/12/13 | 10/11/12/13 |
   | longer 36/37/38/39 | 21/22/22/22 | 22/23/23/23 |

   No expert exceeds cap32 in these L3 captures. This supports thin-path
   replay here, but does not establish later-layer or long-suite occupancy.
   The report's global "gate prompts execute thin" should retain that scope.

6. **The installed pool-drop patch is a semantic lead with a precise cutoff.**
   `arithmetic-audit/installed-sparse_attn_indexer_kpool.py` (SHA256
   `00e32052b781723500987a463814116634c4d00b3b61066915f8cc70780c931e`)
   truncates to `pool_ids[:, :select_k - 1]` at lines 581/875 for the ranked
   prefill/decode branches. Atlas permits 2051 index slots. Preserve the
   selected-key-set hypothesis from the earlier MLA report, but correct its
   "around 2047" onset: installed lines 455–466 choose short prefill through
   context length 2048; lines 479–480 fill all causal indices and line 494
   skips ranked-pool work. For an isolated prefill that stays at or below the
   cutoff, this pool-drop branch cannot explain its failure. Classify each
   actual prefix/chunk and decode branch: teacher-forced continuations in a
   case named 2047 or 2048 can cross the cutoff. **Action:** compare selected
   IDs, valid counts, tail handling and positions before attributing any
   above-cutoff discrepancy. Long-suite GPU scope remains unchanged.

The mHC audit also strengthens the L1 plan: the next layer consumes deferred
`(x,residual,post,comb)` (`vllm/models/glm5next/nvidia/model.py:460–470,505–507`).
Bind all four, actual local rows, sequence-parallel setting and provider/splits.
The small fused path uses FP32 residuals before BF16 storage; larger shapes
round first (`vllm/model_executor/kernels/mhc/tilelang.py:510–532,584–613`).
This source-level distinction is relevant to capture design, not a proven
cause. The revised plan/checklist incorporate these leads and provenance reuse.

**Verdict unchanged:** v24 closes the captured norm stage; Q2 and held-out
ref13 remain FAIL, with downstream causality unmeasured. Reconciliation removes
false missing-work claims and sharpens experiments; it does not qualify a fix.

Documentation validation: `git diff --check` and `typos` on `continue.md`,
`todo.md`, `decisions.md`, this log and the causality plan passed. Changes are
confined to those five existing documents; runtime and harness repairs remain
open checklist items.

### 2026-10-12 — ref13 held-out investigation + neighboring-types matrix

- **ref13 investigation, verdict (i): reference-internal bimodal chaos.** The
  failing short16 `longer`/r0 row recurred **byte-identically** in 2/5 fresh
  fingerprint-identical reference repeats (ref13rep2/3 ≡ archived ref13;
  3/15 draws overall, exact CP95 [0.043, 0.481]); ensemble-mode draws sit at
  ≤0.002188 vs the 0.005 limit. Held-out-control emulation with the registered
  harness (archived ref13 report reproduced exactly first): fresh references
  fail 3/5 as candidates (2 catastrophic-row + rank, 1 rank-only T 5.795 vs
  max T(R) 5.756), pass 2/5. Config receipts identical across all 15 runs
  (canonical live-audit hash `3723c602aab8fa7d`; pinned create-command, hook,
  runner and image hashes match). No environment/config anomaly. The frozen
  gate was not changed; nothing here is a qualification sample; the arithmetic
  mechanism (f32 atomics / route flip) was not observed under the
  collector-only profile. The `provenance audit copy missing` log warnings were
  a batch-script scp path bug — the fixed finalizer ran per repeat and every
  receipt exists in 3 byte-identical copies (report.md §3). Evidence:
  `q2-ref13-investigation-20261012/{report.md,analysis/,provenance/}`.
- **Gate-design consequence** *(follow-up: operator decided full v2
  pre-registration, 2026-10-12; "structurally unpassable" was later
  corrected to "underpowered at N=9" — see the 2026-10-12 post-review
  correction entry at the end of this log)*:
  the pre-registered must-pass held-out control fails for the reference engine
  itself at ~3/5 rate when a draw lands in the failing mode, so the rule is
  structurally unpassable as written. Options (a) keep the registered gate and
  pursue reference-faithful numerics, (b) author a NEW pre-registration — are
  recorded in `todo.md`. Also recorded: the exchangeability-rank condition is
  marginal for held-out references at N=9 (fresh misses 0.034–0.057),
  diagnostic only.
- **Neighboring-types matrix instantiated (49 rows; none "passed").** 12 Active
  GLM profile / 19 supported neighbors / 12 implemented-but-unqualified / 6
  rejections; coverage 34 untested, 8 partial, 1 measured-fail. Decisive
  structural fact: the v24 change as attributed reaches exactly one kernel
  entry — `kernels/gb10/common/kda.cu:565 kda_sigmoid_gated_rms_norm` — whose
  only runtime requester is `Glm5KdaLayer` (`model_type == "glm5_next"`),
  while its build reachability is all 30 kernel targets (common dir). Fused
  twins embedding the same arithmetic (`gdn_verify_fused_k2.cu`,
  `gated_delta_rule.cu` `*_norm` entries across ≥4 families, `ssm_gdn_a.rs`
  fused chain) must be enumerated affected-or-out-of-scope by any shared-code
  fix; `norm::gated_rms_norm` resolves to three different semantics
  (whole-row common/Qwen, per-group `__expf` minimax fork, Nemotron). Evidence:
  `q2-neighbor-matrix-20261012/{matrix.md,reachability.md}`.

### 2026-10-12 — C1 execution: neutrality PASS, captures complete, anchor FAIL-CLOSED

> **CORRECTED 2026-10-12 (post-review):**
> - "binding/inputs are sound, gap is native kernel vs contract" and the
>   "approved remedy" below are wrong. Both engines multiply EXL3 trellis
>   weights, not the bound BF16 reconstruction.
> - C1 was closed from native captures by operator decision.
>
> See the post-review correction entry at the end of this log.

- **Build + neutrality:** fresh no-cache diagnostic image
  `atlas-exl3-validation:src-1687661e73e0-rust1.97.1` (`sha256:f4e47beb…faa81b`,
  commit `e5e18056`, PTX closure `92b03bf9…` byte-identical to the frozen
  norm-bind build; identity preregistered before collection). Neutrality
  precondition PASSED byte-identically: attempt `cand-c1neutr-e5e1-02`
  final-logit archive `28b7b3a8…d0df95` == frozen `cand-q2diag-ee5eaf0f-02`;
  48/48 L0 stage tensors byte-identical vs BOTH frozen baselines; capture
  attempt logits byte-identical too.
- **Captures + bindings (diagnostic runs only):** candidate X (load-time bf16
  common-space weight + resident shard + load record) and reference X+Y
  (hashed overlay; pre-all-reduce local o_proj outputs) complete across 7
  fresh attempts (3 failed tooling attempts preserved: descriptor typo,
  overlay `get_current_vllm_config()` context bug fixed at `install()` origin
  and hashed into `c1-overlay-binding.json`, driver heredoc bug). Placement
  attestation PASS: local outputs rank-divergent 8/8 (pre-reduce proof vs the
  archived rank-identical post-reduce `kda-o-proj`). Operand identity 16/16 vs
  fixtures on both engines. 32/32 rows assembled fail-closed,
  `$BIND=q2-c1-binding-2026013/` (inventory `e05fba33…b163`).
- **CPU preflight: self-anchor FAILED closed, both ranks, both engines**
  (exit 1). The f32-GEMM/RNE replay contract byte-reproduces neither engine's
  native projection GEMM: 40–47% of projection outputs unequal, median 1 ULP,
  max signed ULP ≈28k concentrated at near-zero/cancellation outputs; the
  numpy replay matches f64 truth to 6/40960 elements ⇒ binding and inputs are
  sound, the gap is native kernel vs replay contract. Per the fail-closed
  checkpoint: gate NOT relaxed, no variant replay ran, GPU `--backend cupy`
  skipped as backend-independent, interpretation blocked, propagation
  matrices empty (`rows: []`). C1's two questions remain unanswered by design.
- **Neutral facts (recorded, not interpreted):** (a) engines' native captured
  local outputs are byte-identical on all 8 rows with byte-identical inputs
  and differ on the 8 rows whose inputs differ; (b) bound cross-engine weights
  are NOT byte-identical (6.3% of elements, mostly 1 ULP, max |Δ| 9.8e-4) —
  a dequant-recipe delta, so literal "identical weights" is unattainable for
  this EXL3/bf16 pair (trellis cannot represent the reference matrix);
  (c) v24 ≡ reference-exact inputs byte-identical 32/32.
- **Approved remedy (parent, 2026-10-12; plan-prescribed checkpoint, not a
  gate redesign):** extend the binding with engine-matching GEMM algorithm
  receipts (candidate `launch_local_bf16` function-level identity — receipt
  kernel identity is function-level only by recorded scope of the `exl3-native`
  arm; reference real `quant_method`/cuBLAS algorithm behind
  `Glm53PackedLinearMethod.apply`) and add an engine-native replay backend
  (each anchor runs the engine's own GEMM code path), then re-run anchors and
  the three input variants. The f32-GEMM/RNE backend remains for common-space
  cross-engine comparisons with its own named semantics. Q2's achievable
  scope must be documented as dequant-recipe delta + GEMM-arith delta at
  bound weights.
- Execution-time fix `6568110b` (capture validator accepts the candidate's
  dispatch-nested receipt provider; 34+22 tests green) and the overlay
  amendment are recorded. Teardown receipts: 0 running containers on both
  hosts, GPUs idle, production `glm53-exl3-atlas-0/1` untouched.
  Note: the executing subagent was terminated by a provider content_filter
  AFTER writing all deliverables (`capture-report.md`, `summary.md`,
  matrices, receipts); no evidence was lost.

### 2026-10-12 — statistical gate v2 pre-registration authored (reference-only)

> Duplicate of the next entry's v2 record. Superseded in part by gate v2
> Amendment 1 (post-review correction entry at the end of this log).

- Full v2 pre-registration written per the operator decision: 
  `docs/design/exl3-statistical-gate-v2.md` (status: pre-registered rule;
  no qualification data collected under it). v1 doc untouched.
- Authored strictly from reference-side data (15 archived draws ref04–ref13
  + ref13rep1–5 and the reference-side held-out reports); no candidate file
  was opened. All 15 draws are v2 tuning data, not qualification samples.
- Rule in brief: all-row chaos envelopes E_i = max(3·max-pairwise,
  floor {0.005, 0.15}) replace the catastrophic-row rule; demonstrated-
  bimodal rows add a mode-membership gate (gap G≥4 + exactly-two-components
  + separation ≥20; membership within 2× the measured within-mode radius of
  a mode center; byte/ULP class evidence; mode frequencies reported);
  v1's T rank statistic kept; N=15 references + 3 held-out references
  (all must pass as candidate) + 2 byte-identical candidate runs.
- Calibration (`q2-gate-v2-20261012/`, reference-only): harness reproduces
  all 6 archived v1 held-out evaluations bit-exactly; LOO full-rule
  false-fail 1/15 (rank argmax only, nominal 1/(N+1)); membership LOO 0/15;
  held-out splits (455×12+3) all-pass 77.4 %, rank fail 7.62 % ≈ nominal;
  v2 on the archived N=9 structure still fails 4/6 held-outs → the repair
  comes from N=15, not the statistic; pseudo-candidates with amplified
  reference deviations fail the envelope 15/15 at α≥4 (excess beyond the
  envelope fails). Residual risk: ensemble-composition (lone/missed mode)
  fail-closed probability 0.167 at the observed mode rate 3/15.
- Committed as "docs: pre-register statistical gate v2 from reference chaos
  data". No v2 collection has run; comparability with the v1 FAIL verdict
  remains surrendered per decisions.md.

### 2026-10-12 — gate v2 pre-registered; C1 remediation boundary; overnight drive stopped

- **Gate v2 pre-registration (`f8061ad8`, `docs/design/exl3-statistical-gate-v2.md`)**
  authored from REFERENCE-ONLY data (15 draws: ref04–ref13 + ref13rep1–5, and
  archived reference-side held-out reports; no candidate file opened —
  independently verified by the review). Rule: all-row chaos envelope
  (E_i = max(3·max pairwise reference distance, floor_m); floors carried from
  v1), new mode-membership rule for demonstrated-bimodal rows (detection:
  gap ratio G_MIN=4, two single-linkage components, separation K_MIN=20;
  membership within C_RHO=2.0 × measured within-mode radius; byte/ULP class
  evidence), exchangeability rank kept (v1 T), N=15 references + H=3 held-out
  all-must-pass, 2 byte-identical candidate runs, all v1 must-fail controls
  retained. Calibration (reference draws only): leave-one-out false-fail
  1/15 = 1/(N+1) exactly (rank component only; envelope/membership/discrete
  0); mode detection fires on exactly 1/32 rows (short16/longer/r0) with
  1928/1941 subset stability and zero non-mode false positives; membership
  LOO 0/15; held-out pass rule validated over all 455 split emulations.
  Review verdict: OK with notes — rule-shopping check CLEAN (calibration
  inputs, constants and doc numbers all traced to reference-side archives,
  spot numbers hand-recomputed), 6 P2 wording/evidence-precision fixes staged
  as the FIRST pre-collection task (no rule/threshold impact). All 15 draws
  are v2 tuning data; v2 collection (N=15+3+2) awaits operator sign-off on
  the doc. The redesign repairs the control, not the candidate.
- **C1 remediation boundary (operator stop):** the implement stage timed out
  mid-harness-integration (its own recovery summary: no tracked files
  changed); the partial engine-native backend module is committed as WIP
  `bbd2d267` (unintegrated, untested — continuation in todo.md next-batch
  item 1). Its dependent GPU stage was stopped before any host activity.
  No remediation cycle was started; the approved anchor remedy stands.
- **Overnight drive ended at operator request** after this checkpoint: all
  work committed; continue.md/todo.md cleaned and the next batch staged;
  both Sparks verified idle (0 running containers).

### 2026-10-12 — post-review corrections (independent review `CODE_REVIEW_2026-10-12.md`)

Supersedes the claims named below. The original entries above are preserved.

- **C1 (supersedes "binding sound", "gap is native kernel vs contract" and
  "approved remedy: engine-native replay").**
  - *Verified:* the reference `Glm53PackedLinearMethod.apply`
    (`arithmetic-audit/atlas-reference-active-exl3_mul1.py:3087-3110`) uses
    `F.linear(x, layer.weight)` only for `rows > 144`. Otherwise it casts the
    input to FP16 and runs exllamav3 `h.forward` on the trellis. The C1
    rows (10–39 tokens) took the trellis path. The overlay receipt
    "torch linear (cuBLAS)" was hard-coded. The capture log confirms
    `prefill=bf16, min_rows=3`.
  - Neither engine multiplies its bound BF16 weight, so the 6.3% weight
    delta is between two BF16 reconstructions.
  - Native local outputs are byte-identical across engines on all 8
    identical-input rows. They are also byte-identical on rank0 rows 4–7,
    whose inputs differ at 6 elements. Rank1 rows 0–3 differ on one token
    only (109 elements, max |Δ| 6.1e-5).
  - Operator decision: C1 closed from these native captures, with the
    corrected-input answer flagged as inferred (8 rows, 2 prompts) and the
    o_proj input precision as unattested.
  - The replay harness, the engine-native WIP and the Rust capture hooks
    are archived (`651ff083`; tag `archive/c1-bf16-o-proj-replay-20261012`).
    Evidence erratum: `q2-c1-execution-20261012/ERRATUM-20261012.md`.
- **v24 attribution (supersedes "byte-exact 16/16" as stated).**
  - *Verified on raw arrays:* the 16 fixtures are nested prefixes of 2
    prompts × 2 ranks.
  - Candidate vs reference differ at 7 distinct elements: rank0 `longer`
    (6) and rank1 `arithmetic` (1). The other two groups do not differ.
  - v24 fixes all 7 and adds no new differences.
  - v24 was selected after these fixtures were seen, and the other variant
    pairs were not run. "v1/v3 are not sources" should read "inert on these
    fixtures".
  - The replay is NVRTC `--fmad=false`, and production flags are
    unverified. `v24-minimal` carried no engagement controls. Both GPU
    ablation runs used the pre-repair harness.
- **Gate v2.**
  - "Structurally unpassable" (v1) is corrected to "underpowered at N=9
    under a bimodal row, session drift not excluded". The 3 fresh ref13
    repeats with T above the archived range point to drift.
  - The six P2 review fixes and Amendment 1 are applied (`fd7b530b`):
    randomised interleaved run order; a reference-only composition
    VOID + capped top-up; H3 = v1 rule unchanged; H4 stricter than the
    floor; residual risks; calibration-script errata.
  - Operator sign-off is still required before collection.
- **Code.** The production default path is clean. `Exl3FfnRouter::capture_record`
  now refuses existing records before writing and commits its metadata
  atomically. The grammar-pause change (`19551fc6`) is correct; its tests
  are CUDA-gated and still owed on a Spark.


### 2026-10-12 — Gate v2 Amendment 2 after Luna source review

Operator requested documentation corrections after a Luna vLLM source review.
No GPU work, new image, qualification collection or numerical result.

- Supersedes the Amendment 1 composition VOID/top-up description above:
  `docs/design/exl3-statistical-gate-v2.md` now fixes N=15 before collection.
  Reference-only adaptive stopping did not establish the claimed final-N
  exchangeability bound. Reference unverifiability blocks qualification; all
  draws and failed/incomplete campaigns remain archived.
- H4 now defines two exact byte classes, zero distances/radius, singleton
  spread and tied gaps. Zero-radius membership is byte-exact; an ULP control
  can fail and must not cause an after-the-fact tolerance change.
- The rank bound is marginal and rank-only under exchangeability. Held-out
  passes do not prove exchangeability; a PASS is scoped to registered
  fixtures/profile, not distributional identity or release readiness.
- Source report: `docs/design/exl3-vllm-reference-contract.md`. Luna reviewed
  the exported reference source and registered observers; parent checked the
  selected SM120 cache path, V2 sampler, capture hook and installed EXL3
  plugin and rehashed the backend/wrapper/plugin. Pinned source plus bound
  receipts is authoritative for reference behavior. The older live-latent
  bypass description remains superseded; the recurring mode's cause is open.
- Revision authorised; final rule sign-off and verified v2 evaluator still
  precede collection. Existing calibration is not a test of the new edge
  semantics. Required verification is listed in Amendment 2 A2.4.
