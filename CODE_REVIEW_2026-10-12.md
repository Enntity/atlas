# Review — Q2 overnight drive (2026-10-11 08:00 → 2026-10-12 12:30 PDT)

Scope: commits `e33340d9`..`4e00b222` (30 commits, ~9k lines: Rust capture
hooks, spark-server grammar/stop changes, Python replay/ablation/gate
harnesses, docs) plus the conclusions recorded in `continue.md`, `todo.md`,
`decisions.md`, `q2-diagnostic.md`, `docs/plans/q2-norm-causality-plan.md`,
`docs/design/exl3-statistical-gate-v2.md` and the evidence under
`/home/colum/Workspace/atlas-exl3-validation-20261010` (`$EV` below).
Read-only review; nothing was edited, committed or run on the Sparks.
Python suites run locally: C1 56 passed; ablation/norm 65 passed (+42
subtests). Rust: `cargo check -p spark-server` clean; spark-model
o_proj/gated_norm/upstream tests 10/10. spark-server grammar tests are
gated out of non-CUDA builds and were **not** run.

**Not reviewed** (in-window conclusions with no independent check here):
the neighbor matrix's "v24 reaches exactly `kda.cu:565`, build-reachable by
30 targets"; code-path audit G1–G9 and the vLLM reconciliation
(q2-diagnostic.md:1068); the `a2bf22cd` offline review of `614a2e24`; and
the Q3 "15/15 focused pass / full byte neutrality" conclusions (only the
grammar code was reviewed, not that evidence).

Findings are in three buckets: (A) conclusion validity, (B) code defects,
(C) documentation staleness. Items marked *verified* were checked against
source or raw arrays by this review; others are inferences.

---

## A. Conclusion validity

### A1 (critical) — C1's "binding sound / gap is native kernel vs contract" is wrong; the approved remedy's reference half repeats the error

*Verified:* `$EV/arithmetic-audit/atlas-reference-active-exl3_mul1.py:3087-3110`
— `Glm53PackedLinearMethod.apply` uses `F.linear(x, layer.weight)` only when
`keep_bf16 and rows > 144` (`_GLM53_PACKED_DECODE_MAX_ROWS`). Otherwise it
casts the input to FP16 and runs exllamav3 `h.forward` on the packed EXL3
trellis. All C1 rows are 10–39 tokens, so the reference ran the trellis
path. The capture log (`q2-c1-execution-20261012/attempts/spark0/ref-c1bind-ctxv4c1-04/docker-rank0.log`)
confirms `prefill=bf16, min_rows=3`. The overlay's
`algorithm: "torch linear (cuBLAS) through quant_method.apply"`
(`ref-overlay/o_proj_binding_router.py:118`) is hard-coded, not observed.

Consequences:
- Neither engine multiplies its bound BF16 "common-space" weight. The
  reference casts the input to FP16 (`xp.half()`, :3104) and multiplies
  trellis weights. The candidate's KDA projections are built through
  `new_with_numerics`, which selects `Exl3InputPrecision::F16`
  (`native.rs:79-96`, `exl3_kda_loader.rs:220`); o_proj goes through
  `load_projection(..., select)`, and its input precision is **not attested**
  by the capture receipt. It must be bound before any engine-native replay
  is built.
  The 6.3% cross-engine weight delta compares two BF16 *reconstructions*;
  it is not a difference in what the engines compute.
- "replay ≈ f64 to 6/40960 ⇒ binding sound" only shows that numpy is
  accurate. The weight operand is unsound for both engines, so no
  f32/RNE-over-BF16 replay can pass the anchor.
- The recorded "neutral fact" (native outputs byte-identical across engines
  on identical inputs) is *expected*: same kernel family, same trellis.
  It is not suspicious.
- WIP `local_o_proj_engine_native.py`: the candidate half (EXL3 `_f16` +
  cooperative MMA, argument order matched to `exl3.cu` / `native.rs`) is
  viable and could anchor **both** engines. `project_reference_native`
  (:277-296) replays the >144-row `F.linear` branch and will fail exactly as
  numpy did. The docstring claims (:16-30) that same-weight comparison is
  impossible, which is backwards. `reference_algorithm_receipt` would
  propagate the wrong algorithm string.
- The summary omits that **rank0 rows 4–7 have different inputs** (6
  elements, 1 ULP, |x| ≤ 2.1e-4) yet byte-identical native outputs. Rank1
  rows 0–3 (1 differing input element) differ only on that token: 109
  output elements, max |Δ| 6.1e-5.

Implication: native captures already answer C1-Q2 (zero same-input
projection error on 8/8 identical-input rows). They largely answer C1-Q1
for the natural deltas: on rank0, 6 input elements differ by 1 ULP and 0
outputs differ (mechanism unconfirmed; FP16 can represent these BF16 values
exactly, so input rounding is not the explanation, and a sub-resolution
contribution is the likelier one); on rank1 the difference is confined to
one token. The third variant adds nothing, because v24 ≡
reference-exact on 32/32 rows. Re-scoping C1 is an operator decision.
Recommended: bind the trellis bytes as the reference operand, replay through
one EXL3 backend for both engines, and record the `apply` branch/row
threshold in the receipt.

### A2 (high) — v24 "byte-exact 16/16" rests on 7 elements in 2 independent groups

*Verified on raw arrays* (`staging/fixtures/*/expected.npy` vs
`reference-expected.npy`). The 16 fixtures are nested prefixes of 2 prompts ×
2 ranks. Only 2 of the 4 groups differ from the reference at all, at **7
distinct BF16 elements** (6 on rank0 `longer`, 1 on rank1 `arithmetic`).
v24 flips all 7 and leaves about 426k other unique elements unchanged; that
second part matters, since v4 alone moved rank0 from 6 to 7 differences. v24 was
also selected after seeing these fixtures: `d04d8bfa` was committed 6 min
after the r1 summary, and there is no held-out set. Other variant pairs were
never run, so "minimal" is inferred. "v1/v3 are not sources" should read
"inert on these fixtures": v1 cannot reproduce the reference tree, and v3's
ftz difference only bites on subnormals. Variants are NVRTC-compiled with
`--fmad=false` (`native_ptx.py:60`); production flags were not shown to
match, so the step-5 qualification must re-prove v24 under the production
toolchain. `v24-minimal` has `controls: null`. Both GPU ablation runs
predate the harness repair, so the repaired harness has never run on a GPU.

### A3 (high) — Gate v2: bookkeeping clean, framing overstated, single-campaign false-reject risk high

*Verified:* `calibrate_v2.py` reads only reference draws and
reference-side held-out JSONs; the CP95 [0.043, 0.481] and the 3/15 mode
counts are correct. The six P2 fixes are **not applied** (doc unchanged
since `f8061ad8`).

- **"Structurally unpassable" (todo.md:46, q2-diagnostic.md:1222) is
  overstated.** Under v1 the control passes `longer`/r0 with probability
  ≈0.57–0.70 at N=9 and ≈0.86 at N=15. v1 is underpowered under bimodality
  rather than impossible; the doc's own `H2_v2_rule_at_N9` (v2 still fails
  4/6 at N=9) concedes that N is the fix. v2 is still defensible for the
  explicit LOO rank exemption (v1's every-LOO-must-pass text is incoherent)
  and for H4.
- **Session drift is not excluded.** 3 of 5 fresh ref13 repeats have T above
  the archived ensemble's range (5.80/5.84/5.94 vs max 5.76–5.89), while the
  same-session ref13 sits well inside. p≈0.009 at the nominal rate.
  `H_n9_vs_n14` pools sessions and can't separate the two explanations.
  v2's sequential ref → held-out → candidate collection would put any drift
  onto the candidate. Interleave or randomise run order.
- **H3 is v1's catastrophic rule unchanged** (same c=3, floors, LOO). The
  envelopes widen with N, so per-row gating loosens: `longer`/r1 limit goes
  from 0.0130 to 0.0222, and the candidate's r1 (0.0136) would now pass.
  Detection rests on rank plus H4. Calling it an "all-row chaos envelope
  repair" overstates the change.
- **H4 membership (2·ρ = 0.0028 at `longer`/r0) is stricter than the 0.005
  floor**, and that is not disclosed. A deterministic candidate that always
  lands in the rare mode passes by design (low practical risk).
- **Estimated false-reject rate for a reference-identical engine** in one
  campaign: ≈35% at p=0.2, and much worse at p≈0.04. That is about 20 runs
  lost per miss. The doc pre-registers composition misses as FAIL with no
  void/top-up rule. C_RHO=2.0 and the membership definition were tuned and
  validated on the same 15 draws.
- "No candidate numbers set thresholds" holds at script level. The authors
  did know the candidate FAILs by a wide rank margin, so the design was not
  blind.
- ref13 rate: the triggering draw is included. Excluding it gives 2/14;
  fresh draws alone give 2/5. "Fingerprint-identical" excludes config/image
  drift but not runtime choices (autotune, NCCL algorithm, rank start
  order). Six byte classes, two of which hold 9 of the 15 draws, suggest a
  small discrete switch, not continuous chaos. If that switch can be pinned,
  making the reference deterministic would remove the need for H4. That
  option was not explored.

Before any v2 collection: apply the P2 fixes, interleave run order,
pre-register void+top-up for composition misses, disclose the H3/H4
changes, and restate the v1 diagnosis.

### A4 (medium) — Approval provenance

The C1 engine-native remedy was "approved" by the parent agent
(q2-diagnostic.md:1277, `3956b63c`); there is no decisions.md entry naming
the operator. Given A1, it should be re-decided by the operator rather than
resumed as todo item 1.

---

## B. Code defects

Production default path: **no defects found.** With the binding env vars
unset, the forward path performs no env reads, I/O, allocation or sync
(`upstream.rs:216-219` returns early; the gated-norm capture is behind
`if let Some(trace)`). Launch-dim refactors are arithmetic-identical. The
load-time `gate_dir` reads two env vars per KDA layer, at load only.

`19551fc6` grammar pause (*verified* against `first_token_thinking.rs:36-44`):
the flag `enable_thinking && think_end.is_some()` matches the decode loop's
initial `inside_thinking`. It fixes a real matcher off-by-one, and is a
no-op when reasoning is off. One edge behaviour change: with thinking on,
tools and a JSON grammar, the first token may now be the native tool opener
(previously masked). Its tests are CUDA-gated and were not run here.

| # | Sev | Location | Issue |
|---|---|---|---|
| B1 | Med | `weight_loader/glm5/o_proj_binding.rs` `capture_exl3` ~245-275 | `resident_sha256` is recorded, but the BF16 `.npy` is rebuilt from the host checkpoint (`plan.materialize`). It is never cross-checked against the uploaded resident bytes, so an upload/shard bug would still yield a "clean" binding. |
| B2 | Med-low | same + `spark-runtime/.../ffn_io.rs` `write_binding_with_receipt` | `create_new(true)` makes a reused binding dir fail model load. Writes are not atomic (no tmp+rename) and `write_all` has no context, so a crash leaves a truncated file with a sha recorded for the intended bytes. |
| B3 | Low | `upstream.rs:216-219` | The receipt re-derives the FP8 predicate (duplicated 2048 threshold; ignores graph_capture/scratch/arena bail-outs), so it can label `atlas-lt-fp8` when NVFP4 ran. Unreachable in practice today. |
| B4 | Low | `local_o_proj_replay.py:293`, `:304-314` | `identical_operands=True` is hard-coded and `build_matrix` doesn't block on `weights_cross_engine_equal=False`, so cross entries could be read as same-input GEMM error. |
| B5 | Low | `local_o_proj_replay.py:169`; capture :222 / replay :182 | `resident` is optional in replay validation but required in capture; `resident['file']` is used as a path without sanitising; `own_output_bin` is not hash-linked to the capture record. |
| B6 | Low | `ffn_io.rs` | `write_binding_with_receipt` omits the `stream` field that `capture_record` writes under the same schema. |
| B7 | Low | `calibrate_v2.py:333,366-373` | The label for membership definition "a" omits C_RHO, which the code applies; the definition is picked as "first with 0 LOO fails"; the comment calls it "least-strict" when it is the strictest. |
| B8 | Nit | `gemm_dense.rs` | The `w4a16_gemm` doc comment moved onto `w4a16_gemm_launch_dims`. |

Test gaps: no end-to-end `capture_exl3` test on the mock GPU; the receipt
FP8 predicate is not tested against `project_hot_cast`; the grammar pause
is not tested through the real prefill callers or
`resolve_with_tool_boundary`; C1 synthetic own-outputs are produced by
`project_numpy` itself; `main()` wiring of `validate_fixture` /
`enforce_summary_gate` is untested (already deferred in todo.md).

---

## C. Documentation staleness (fix before the next drive)

1. todo.md:71-73 — "`gate-ref-0/1` IN-FLIGHT … do not stop them". This
   contradicts "both Sparks idle" and survived `94317570`, which claimed to
   fix it.
2. continue.md:225 — ref13 evidence row still "IN FLIGHT, do not touch".
3. continue.md:92-95 — Next action #1 describes C1 as not yet run ("C1-prep
   running"); todo.md #1 is the correct version.
4. continue.md / todo.md #5 — the neighbor matrix is done; only the repeat
   Rust/vLLM/source review remains.
5. decisions.md — the v2 entry does not mark as superseded the "Q2 gate
   stays frozen / no gate change" entries (:38, :48-49, :65, :132-139).
   ":24 existing runs are tuning data" should say *reference* runs. The
   entries are out of date order.
6. q2-diagnostic.md:8-24 still states the v1-only objective, with no pointer
   to v2. :1219-1225 has no follow-up note. :1296 and :1324 record the v2
   event twice (the first at `##` level).
7. docs/plans/q2-norm-causality-plan.md — :3/:5 say "nothing executed"; :48-51
   says the runner defects are open (fixed); :24/:426/:442 are v1-only;
   :182 says ref13 is pending; the C1 remedy is absent.
8. todo.md:51 still says "(in flight)" for the committed v2 doc. :227/:259
   ("HEAD after `ee5eaf0f` is documentation only") are stale (`b85623c0`
   changed Rust).
9. continue.md evidence index has no rows for `q2-neighbor-matrix-20261012`,
   `q2-c1-prep-20261012`, `q2-c1-execution-20261012`,
   `q2-c1-binding-2026013` (sic — missing digit; leave the directory name,
   annotate it) or `q2-gate-v2-20261012`.
10. continue.md:112/120 still say "reviewed HEAD `11671adf`". The production
    image `src-48624a8605de` differs from the Q0 final image
    `src-8b6b47f14d5b` (todo.md:131); confirm which image the stopped
    containers actually ran.
11. All v24 claims (continue.md:22-23, todo.md:153-157, plan :29-30,
    `analysis-20261012.md`) should read "7/7 differing elements across 2
    independent groups; fixtures-only; NVRTC `--fmad=false`".

Orphaned untracked files (predate this window, created 2026-10-10 22:40,
unreferenced): `PROJECT_WIDE_CODE_REVIEW_2026-10-11.md`,
`CODE_REVIEW_PASS2_SUPPLEMENTAL.md`, `docs/plans/atlas-kv-{coord,cache-server}-plan.md`.
They are a generic repo audit and an unrelated KV-coordinator design. Commit
or discard them deliberately.

---

## Suggested next steps (in place of todo "Next batch" items 0–1)

1. **Operator decision on C1 re-scope (A1).** Either accept the native-capture
   answers for Q1/Q2, or rebuild the reference binding on trellis bytes and
   replay both engines through one EXL3 backend. Do not resume the WIP
   reference half as written.
2. Before any v2 collection: P2 fixes + A3 amendments (interleaving,
   void/top-up, H3/H4 disclosure, restated diagnosis). Consider a cheap
   probe of the ref13 discrete switch (autotune/NCCL algorithm pinning)
   first; if it can be pinned, H4 may be unnecessary.
3. Fix B1/B2 before the next capture run.
4. Doc sweep (C1–C11) in one commit.
5. C2 / L1 escalation: with o_proj now shown to be same-input exact,
   prioritise the downstream path (routing at L3, L1 KDA for `longer`).

---

## Disposition (2026-10-12, same day)

Operator decision: C1 → Option 1 (close from native captures, flagged).

| Finding | Disposition |
|---|---|
| A1 C1 premise | Fixed. C1 closed from native captures (decisions.md). Harness, engine-native WIP and Rust hooks archived (`651ff083`, tag `archive/c1-bf16-o-proj-replay-20261012`, archive README with import guard). Evidence erratum `q2-c1-execution-20261012/ERRATUM-20261012.md`. |
| A2 v24 evidence | Recorded. Wording corrected everywhere to "7/7 elements, 2 groups". Evidence erratum `q2-norm-ablation-20261012/ERRATUM-20261012.md`. GPU rerun with v24 controls, production-flag proof and variant pairs are todo Next batch item 5 obligations. |
| A3 gate v2 | Fixed in doc. Six P2 fixes + Amendment 1 (`fd7b530b`). Sign-off and the ref13 mechanism probe are todo item 0. |
| A4 approval provenance | Fixed. decisions.md records the remedy as withdrawn and the operator's C1 decision. |
| B1, B3, B6, B8 | Resolved by archiving the C1 Rust capture (code removed from crates). |
| B2 | Fixed for the remaining capture path: `capture_record` refuses existing records before writing tensors and commits metadata via tmp + fsync + rename, with a test. |
| B4, B5 | Resolved by archiving the C1 Python harness. |
| B7 | Recorded as errata in gate v2 Amendment 1 (A1.6). The evidence script is unchanged. |
| Grammar-pause test gap | Owed on a Spark (todo "Owed verification"). |
| Ablation harness test gaps / review nits | Pi agent branch `pi/ablation-hardening-20261012` (see todo "Deferred hardening"). |
| C1–C11 doc staleness | Fixed in continue.md, todo.md, decisions.md, q2-diagnostic.md and the causality plan. Append-only logs keep history with SUPERSEDED/CORRECTED markers. |
| Orphaned untracked files | Left untouched pending the operator's decision. |

### Gate follow-up — Amendment 2 (2026-10-12)

The A3 disposition above records Amendment 1 at that time. The operator's
subsequent requested review/correction produced
[Amendment 2](docs/design/exl3-statistical-gate-v2.md) and the
[pinned vLLM source contract](docs/design/exl3-vllm-reference-contract.md).
A1.2 top-ups are withdrawn; N is fixed, H4 edge cases are explicit, and
collection awaits final sign-off plus evaluator verification. No new
qualification evidence was collected.
