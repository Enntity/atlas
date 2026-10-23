# references/pr-and-ci.md — sync-pr automation, CI-green preflight, fork contribution policy

## Remotes (know which is which)

Enntity/atlas is this fork. Verify `origin` points to
`https://github.com/Enntity/atlas.git` before pushing or opening a PR.
Atlas-Inf/atlas is an upstream source, not a publication target for this fork's
work. The sync procedure below is inherited upstream guidance; do not run it
against Enntity branches without a separately requested engine sync.

## `/atlas-release sync-pr` — the loop
1. **Detect:** `git fetch monumental && git log --oneline origin/main..monumental/main`.
   If empty, nothing to do.
2. **Sync (fast-forward only):** advance `origin/main` to `monumental/main`. If it
   won't fast-forward, **stop and report** — a diverged fork is a human decision,
   not an auto-merge.
3. **Build + verify:** `/atlas-release build *` → `/atlas-release verify`. A sync
   that fails the serve matrix is a blocker to surface, not to ship past.
4. **Enablement PR (only if warranted):** if the merge added a new
   `kernels/gb10/<model>/MODEL.toml` or a new loader that needs a
   `MODEL.toml`/`factory.rs`/docker stanza to be *deployable*, open **that**
   enablement PR on `origin`. Otherwise there is no PR — a plain engine sync just
   feeds the next image cut.
5. **Preflight (below) must be green before submit.** AI-attributed. Enntity/atlas requires no CLA.

**No-takeover rule:** sync-pr opens *additive* enablement PRs or pushes a trivial
fix to an existing branch. It never opens a parallel reimplementation of someone's
in-flight work — reference their branch and comment instead. Contribute to the
author's work, don't fork around it.

**Attribution:** commit as the contributing maintainer; state AI authorship in the
PR body per `CONTRIBUTING.md` §Authorship (this is an AI-first repo — AI authorship
is the norm), not as a tool/bot signature trailer in the commit.

## CI-green preflight — run locally before every PR
CI is **entirely GPU-free** (stubs `libcuda`/`libnccl` so no-GPU runners link).
The kernel work happens on your box; CI only proves the Rust compiles + hygiene.
Run exactly what CI runs, in this order — a red preflight is a red PR:

```bash
# 1. fmt          (ci.yml: fmt)
cargo fmt --all -- --check
# 2. clippy       (ci.yml: clippy — deny warnings comes from [workspace.lints], so no -Dwarnings needed)
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo clippy --workspace --tests
# 3. license hdrs (SPDX AGPL-3.0-only line 1 of every new .rs/.cu/.cuh/.h/.hpp/.cpp;
#    wraps the same apache/skywalking-eyes engine ci.yml runs against .licenserc.yaml)
bash scripts/check-license-headers.sh
# 4. typos        (ci.yml: typos)
typos                                   # cargo install typos-cli
# 5. unit tests   (ci.yml: test — non-#[ignore] only, GPU-free)
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo test --workspace --locked
# 6. file-size    (file-size-cap.yml: ≤500 LoC per crates/**/*.rs, allow-list aside)
find crates -name '*.rs' -not -name '*.bak' -not -path '*/target/*' | xargs wc -l | awk '$1>500 && $2!="total"'
# 7. deny         (security.yml: cargo-deny)
cargo deny check advisories licenses sources bans
```

CI job → check map (so a red check tells you which command to rerun):
`fmt` · `clippy` · `license-headers` (skywalking-eyes, `.licenserc.yaml`) ·
`typos` · `test` · `test-macos-metal` (asserts the metal binary links **neither**
libcuda nor libnccl) · `File size cap` · `Security Audit` (weekly + on Cargo.*
changes) · `Docs` · `Site`.

⚠️ **CI-green ≠ shippable.** None of the above boots a model. Only
`/atlas-release verify` (the serve matrix) proves the image serves. Keep the two
gates distinct in your head and in PR descriptions.

## Enntity fork contribution policy

Enntity/atlas does not require a CLA. Do not request signatures, add contributors
to an inherited CLA allowlist, or edit historical signature records. See
[CONTRIBUTING.md](../../../../CONTRIBUTING.md#license). Keep pull requests in
Enntity/atlas unless the user explicitly requests a separate upstream contribution.

## CI convergence — the drift to reconcile
The maintainer flagged "our PRs and CI are all over the place." The concrete,
low-risk reconciliations (the convergence pass applies the unambiguous ones to the
working tree, unpushed, for review):

| Drift | SSOT / fix |
|-------|-----------|
| **File-size cap: 250 vs 500.** `AGENTS.md` + the global guide say "≤250 lines"; `CONTRIBUTING.md` + `file-size-cap.yml` enforce **≤500 LoC**. | CI is SSOT → **500**. Fix `AGENTS.md` to `≤500` and cross-reference the workflow. (Applied.) |
| **Dead links.** `docs/EP2-TROUBLESHOOTING.md` (never created) was referenced by `docs/DEPLOYMENT.md`, `book/src/operations/multi-gpu.md`, and `docs/adr/0007-tp-ep-composition.md`. | All three repointed to `docs/GB10_DEPLOYMENT_GUIDE.md` §7. (Applied.) |
| **Dead script.** `AGENTS.md`, `.github/pull_request_template.md`, and `book/src/project/contributing.md` all told contributors to run `scripts/check-license-headers.sh`, which didn't exist — CI uses `apache/skywalking-eyes`. | Created `scripts/check-license-headers.sh` as a thin wrapper over that same engine, so all three references now resolve. (Applied.) |
| **Image provenance.** `release.sh`/`release.yml` pass `--build-arg ATLAS_GIT_SHA` but the Dockerfile had no `ARG`, so it was dropped. | `ARG`+`LABEL org.opencontainers.image.revision` added to `docker/gb10/Dockerfile`. (Applied.) |
| **Action-version drift.** `release.yml` uses `checkout@v4` + `upload-artifact@v4`; the rest use `checkout@v6` / `upload-artifact@v7` / `download-artifact@v8`. | Pin all workflows to one major per action. (Recommend — touch when next editing release.yml.) |
| **`cap` is eroding.** The allow-list has grown to ~40 files, several far over 500 (metal_backend 734), which only warn. | Machine-enforce the "rationale + tracking issue per addition" policy, or schedule splits. (Recommend.) |
| **No image in CI.** `release.yml` ships tarballs only; the image is manual. | The Stage-1→3 self-hosted workflow in `references/pipeline.md` §Closing the gap. |
| **Required-check trap.** `security.yml`/`docs.yml` deliberately drop the PR path filter so kernel-only PRs aren't left permanently "expected"; `site.yml` still has one. | If `Site / Build` is ever made required, give it the same no-path-filter treatment. (Recommend.) |
| **Version scheme ×3.** crate `0.1.0` vs image `alpha-2.x`/`3.0.0` vs distro `vX.Y.Z`. | Anchor the image to the git SHA tag (`references/pipeline.md` §Image); let moving tags point at it. |
