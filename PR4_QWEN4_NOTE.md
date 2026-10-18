# PR-4 qwen4_exp — verification (2026-10-18, worktree followup/pr7-curation)
Open row per todo.md §PR-4: build upstream qwen4_exp EXL3 + its existing EXL3 tests.
Status: verified non-invasive; build not yet executed (needs upstream target).

Evidence:
- Our changes scoped to glm5_next (parsers/glm5_next.rs, dispatch.rs,
  atlas-core config). grep of those files shows NO mention of "qwen4",
  "qwen4_exp", or cross-model EXL3 paths.
- Upstream independent EXL3 path identified: bench/qwen4_exp/ (ARCHITECTURE.md,
  PLAN.md, cmp_repos.py, ckpt_split.py) + per continue.md: quant_format/exl3.rs,
  weight_map/exl3/, kernels/gb10/qwen3.8-flash-next/exl3 ([build] base_quant).
- Our ensure_not_exl3 / vocab-cap / TP2 preflight lives only under
  glm5_next dispatch; no shared-trait or kernel-register change touches
  qwen3.8-flash-next target.
- Build owed: nvcc compile of upstream qwen4_exp EXL3 target + run its
  existing EXL3 tests; requires builder container (at atlas-dev-builder,
  do not rebuild into production paths per continue.md §0).

Acceptance: PR-4 closes when upstream qwen4_exp EXL3 builds on the rebased
branch and passes its own test suite with our glm5_next changes present.
Not started — needs separate build window; not a PR blocker (PR-4 is
non-regression of shared GLM code, already verified on NVFP4 lane PR-4 first row).
