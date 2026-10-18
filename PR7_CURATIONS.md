# PR-7 commit curation — followup/pr7-curation @ 2e8b54db + 4 groups
Branch: followup/pr7-curation (worktree ../atlas-pr7); not pushed.
Source: pr/glm53-exl3 @ 2e8b54db (rebased onto spark-shared a9464810).
Diff vs upstream base: ~70 commits, 714 files, +95,982.

Exclude (docs/design/pr5-description-draft.md §Packaging; todo.md PR-7):
- Root working notes (todo.md, continue.md, decisions.md, assumptions.md,
  q2-diagnostic.md, prefill-*.md, p13-*, CODE_REVIEW_*)
- docs/design/ + docs/plans/ validation records (45 files)
- tests/exl3_validation/ harnesses (237 files)
- atlas-kv-cache-server / atlas-codecs crates
- F4 config stub (dev-only); *_opt kernel drafts (already dropped 6a5110c2)

Reviewable groups (this branch — 4 marker commits above a9464810):
G1 build/kernel — EXL3 target, MODEL.toml, nvfp4 build, symlink repair
G2 server/scheduler — EP=0xFFFF_FFF9 wire, request-id, stop/seed, grammar
G3 model/wire — qwen3_attention EXL3 guards, glm_sparse_decode, MTP, prefill
G4 CI/license — fmt clean, SPDX green, 500-line cap inherited; exclusions noted

Accept: open PR from this grouping; no new code — packaging only.
