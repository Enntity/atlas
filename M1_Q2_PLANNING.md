# M1 / Q2 planning note (2026-10-18, followup/pr7-curation / dev)
Source: todo.md §A + docs/plans/q2-norm-causality-plan.md; continue.md "A. M1 causality (open)".
Status: C2 done; escalation / G-fallbacks / reference pin / I-residuals owed.

Done:
- C2 L1 bisect — registered correction (44fbdc2a dtype + 8b4854b2 layout binding);
  CPU tests 66+65 green; old f099-01 payloads half-token-only (option b2 NOT taken).
- L1 tooling fixed (l1-prepare.py / l1-collect.py scripts available; operator-gated).

Open — needs operator / scheduling:
- Capture-change neutrality proof (candidate F32 vs reference BF16); fresh L1 collection
  (operator-gated); bisect on both ranks.
- Reference runtime pin: EXL3_FUSED_MOE=0 deterministic <2052 tok but NOT NEUTRAL
  (4 short16 to 1.19–1.93x H3); adoption needs registered amendment.
- G-fallback sequence (G8/G9 binding first, then G2→G3→G4→G1→G5→G6/G7).
- I-a sqsum (~6.3 µs), I-b KDA recurrent (32 µs vs ~15 µs), I-c exl3_mhc_pre_partials
  (12 µs vs ~8 µs), I-e graph-launch, I-f NCCL SendRecv (90 exchanges/token) — all
  need PC-sampling ncu (host sudo /tmp/atlas-bench.lock).

No GPU experiment launched — planning only; GPU uses require /tmp/atlas-bench.lock.
