# todo.md — GLM-5.3-Flash-Uncensored-EXL3-mul1 on Atlas

Launch-first follow-ups. Goal ordering: (1) inference works — DONE, (2)
parity, (3) performance. Everything here is follow-up; nothing blocks the
running launch. See `docs/design/exl3-support.md` and
`docs/design/exl3-kda-reference-core.md` for context.

## Launch status (2026-10-10)

DONE. Atlas TP=2 / EP=1 serving `MikeRoz/GLM-5.3-Flash-Uncensored-4.05bpw-h6-exl3`
on both Sparks via `scripts/start-glm53-exl3-tp2.sh`, image
`atlas-exl3-validation:src-48624a8605de-rust1.97.1`
(`sha256:82b23cd19e59…`). `/health` ready on rank0 and rank1. Smoke:
greedy short, 612-token coherent long generation, multi-turn recall,
sampled — 5/5. Took over port 8000; production vLLM `glm53-exl3-head`/
`-worker` preserved stopped as fallback. Restart policy `unless-stopped`.

## Correctness follow-ups (parity, before any perf claim)

- [ ] **Reference-policy KDA core (EXL3-only).** Atlas computes the
  correctly-rounded result; the frozen gates compare against lossy FLA
  chunk arithmetic, so short16 still FAILS even though Atlas is the more
  accurate one. Implement an EXL3-only opt-in core that reproduces FLA
  bit-for-bit to close the gates. **Groundwork DONE 2026-10-10:**
  - Spec + 8-task implementation plan (T2 prefill-M5 → T8 GPU gates) with
    PTX-level open questions Q1–Q6: `docs/design/exl3-kda-reference-core.md`.
  - CPU acceptance oracle (4/4 pass; skip path clean; negative control:
    Ai_bf16 off → 533/621 mismatches = the documented ≥533):
    `tests/exl3_validation/test_exl3_kda_reference.py`.
  - New `KdaArithmeticPolicy {CorrectlyRounded, FlaReference}` gated by
    `ATLAS_KDA_FLA_REFERENCE`, env-unset path byte-identical to today
    (§3.6 of the design doc).
  - **Open items the groundwork surfaced:** inter-chunk h carry ≥3 chunks
    unverified (no fixture); inter_bk 32 vs 64 unpinnable from outputs
    (recover from reference Triton cache/PTX fingerprint, Q4); gate cumsum
    order (sequential vs ieee triangular dot) indistinguishable on
    fixtures but not literally identical (Q6).
  - Next: T2 — implement the M5 prefill CUDA kernel against the oracle.
  Fixtures in
  `~/Workspace/atlas-exl3-validation-20261010/arithmetic-audit/kda-core-divergence/`.
- [ ] **M128 residual.** ~1728/585 of 524288 1-ulp flips unexplained; hypothesised
  ex2.approx/div.full GPU behaviour not NumPy-modellable. Needs a bit-level
  GPU comparison harness.
- [ ] **Rerun short16 + long/persistent/behavior gates** after the KDA core,
  before declaring parity. Reference self-repeat already fails the frozen
  gates (nondeterminism at layer-3 routed combine) — preserved, not an excuse.
- [ ] **Indexer boundary oracle retry** (continue.md step 8): mount the exact
  frozen reference source closure; prior attempt failed its source guard pre-Torch.
- [ ] **Graphs unqualified.** Launch runs eager with `ATLAS_NO_DEBUG_GRAPHS*=1`.
  Qualify decode graphs for the EXL3 path before enabling by default.

## Performance follow-ups (SECONDARY — only after parity)

Smoke throughput was ~2.17 tok/s (612 tok / 281.7s) — **preliminary,
single-shot**, no repeats; fingerprint = attested image 48624a8605de, no
speculation, eager, max_batch=1, BF16 KV, GB10 TP2. Not a benchmark. Ranked
areas to improve once parity lands, per continue.md + GB10_KERNEL_OPTIMIZATION_REVIEW:
- [ ] Decoder specialization / remove local-accumulator LDL/STL in repeated MMA regions.
- [ ] Ordered split-K.
- [ ] mHC parallelization.
- [ ] Expert grouping.
- [ ] Profile the real serving path first; do not touch legacy NVFP4/Qwen router-tier advice on the native EXL3 path.
- [ ] FP8 KV, DFlash/speculation, BF16 head transport — each needs separate qualification.
