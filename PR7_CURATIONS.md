# PR-7 commit curation — followup/pr7-curation @ 2e8b54db
Branch: followup/pr7-curation (worktree ../atlas-pr7); not pushed.
Source: pr/glm53-exl3 @ 2e8b54db (rebased onto spark-shared a9464810, 33 topic
squashes + fmt). Diff vs upstream base: ~70 commits, 714 files, +95,982.

Exclude (already in docs/design/pr5-description-draft.md §Packaging):
- Root working notes: todo.md, continue.md, decisions.md, assumptions.md,
  q2-diagnostic.md, prefill-*.md, p13-*, CODE_REVIEW_*
- docs/design/ + docs/plans/ validation records (45 files, ~55k lines with harness)
- tests/exl3_validation/ (237 files, harness + receipts)
- atlas-kv-cache-server / atlas-codecs crates (unrelated to GLM EXL3)
- F4 config stub (not in PR tip; dev-only)
- Kernel *_opt drafts already dropped by merge (6a5110c2)

Reviewable groups (to be finalized into 4–6 commits before PR):
1. Build / kernel target (glm-5.3-flash EXL3, MODEL.toml, symlink repair)
2. Server / scheduler / wire (EP_CMD_REQUEST_ID 0xFFFF_FFF9, request-id,
   stop strings/seed fixes, thinking-loop watchdog, grammar mask)
3. Model / weight / MTP / sparse-decode admission (qwen3_attention EXL3
   guards, glm_sparse_decode, prefill_stream, choose_seed)
4. CI / license / format (SPDX, 500-line cap inherited red only, fmt clean)
5. Evidence / benchmark harness — EXCLUDED from PR (held in branch,
   merged separately if needed)

Accept: PR opens from this grouping once exclusions confirmed; no new
code added to this branch — packaging only.
