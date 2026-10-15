# L1 reference observer under `EXL3_FUSED_MOE=0` — registration

Registered 2026-10-15, before collection. This is a **diagnostic** that
establishes reference observer verifiability for L1 (todo "Now" §4), the
requirement left by the [L1 record](l1-execution-20261014.md). It is not a gate
amendment and no sample from it is a qualification draw. The gate-v2 campaign
stays UNVERIFIABLE.

## Why this design

`l1-refexact-f099-02` withheld attribution because the reference's control
and observed final logits matched on only 1 of 8 rows. That single pair could
not separate intrinsic reference variation from capture effects. Three
2026-10-15 results now separate them:

- [Hidden-state localization](exl3-hidden-state-localization-result-20261015.md):
  in 96/96 comparisons the first differing reference tensor is the fused-MoE
  FP32 output at layer 3 or later. All of layers 0–2, including every mHC and
  attention boundary, were byte-identical across fused-on processes.
- [Fused-MoE determinism](exl3-fused-moe-determinism-result-20261015.md): with
  `EXL3_FUSED_MOE=0`, every row of up to 2,051 tokens was byte-identical across
  three processes.
- [Indexer localization](exl3-indexer-localization-result-20261015.md): the
  remaining source starts at 2,052 tokens.

The L1 rows are eight cold `serial128` teacher-forced terminal-prefill rows
(`arithmetic` and `longer`, 10–39 tokens; `l1_prepare.py` enforces at most 128).
They are below both thresholds. L1 captures layers 0 and 1, which run before
the first routed MoE layer (3). So under `EXL3_FUSED_MOE=0`:
- the observer's final-logit comparison becomes decidable: control and
  observed should match unless arming the capture perturbs the reference;
- the layer-0/1 captures should not depend on the MoE setting at all. That is
  testable against the fused-on `f099-02` captures (O2).

`EXL3_FUSED_MOE` is a runtime environment setting that the installed plugin
reads. It was already used in a registered test. Setting it is not a
modification of the reference kit under the 2026-10-15 scope decision. As the
determinism result records, the loop path is a different reference profile,
not a neutral pin. This study never uses its final logits as a qualification
reference.

### What the switch changes besides MoE dispatch

Only the plugin reads `EXL3_FUSED_MOE`, through `fused_moe_enabled()` at
plugin l.964–965, called at l.1927 and 2447. The plugin is the installed
`exl3.py`, SHA256 `6a0624582898…`, archived as
`$EV/arithmetic-audit/atlas-reference-active-exl3_mul1.py`. The start script
and the mounted patches do not read it. With the fused path on,
`build_exl3_fused_state` (l.1486 onward, with the concurrency query at l.1514) allocates FP16 temporaries, builds
pointer tables and queries `exl3_moe_max_concurrency`. With it off, none of
that happens, so allocator state differs. The KV cache is fixed by
`--kv-cache-memory-bytes`, and `ENFORCE_EAGER=1` is set. O2 is the test that
none of this reaches layers 0/1.

### Autotune cache (second state variable, controlled)

Every reference window mounts the live exllamav3 cooperative-GEMM tune cache
read-write: `/tmp/atlas-exl3-upstream-matched-context-v4-20261010/reference-autotune`,
named through `EXLLAMAV3_TUNE_CACHE`. Neither the plan nor the lifecycle hashes
it. Its frozen state is 1,008 bytes:
- rank 0 `62e13d3d4d1f1acd2f4eb01c1201e2472da3f94b5d5a4771c3c80091f5dc3620`;
- rank 1 `6844e72b825a59afa776e3b0a9e8c36361a2822dee04b6a73157aca8299d0abc`.

The probe `cache-freeze/` recorded this state after `f099-02`, and it was
re-verified on 2026-10-15. Fused-on runs leave it unchanged.

In all three determinism runs the fused-off loop path appended records,
growing it to 1,072 bytes. The post-state was the same every time:
`e78106dc…` on rank 0 and `e66f8bd2…` on rank 1. It was the only one of 4,372
cached files to change.

Here, reference-control tunes those shapes and reference-observed reads them.
The installed source shows this should not change outputs, but it is
registered and controlled rather than assumed:
- After tuning, the real result comes from one final launch with the chosen
  configuration (`coop_autotune.cu:641–660`). That is the same launch a cached
  run makes (`:626–639`).
- The GEMM writes C without reading it ("Does not need to be
  zero-initialized", `exl3_gemm.cu:28`). Split-K is lock-ordered.
- Observed reads exactly the configuration control chose.
- Layer-0/1 shapes are pretuned at load (plugin l.3333) and already present
  in the frozen state.

Controls:
1. Before collection, require the frozen hashes on both ranks, then back up
   the file on each host and locally.
2. After collection, record the post-state. The expected value is the
   determinism post-state: it implies control appended exactly those records
   and observed added none.
3. Then restore the backup and verify the frozen hashes again. Later fused-on
   runs mount this file.

## Attempt (frozen)

One attempt: `l1-fusedoff-f099-01`. It is not retried unchanged, and a failure
is retained. A fresh attempt needs a registered correction first. The preserved
failed attempts `l1-refexact-f099-01` and `l1-refexact-f099-02` stay bound in
the plan.

## Profile (frozen)

Evidence root: `$EV/l1-observer-fusedoff-20261015/`
(`$EV=/home/colum/Workspace/atlas-exl3-validation-20261010`).

- Configuration `config.json`, SHA256
  `21c99e679865d0f0e83b196a24055fdefb1fde8a39add6c695ea4ec56bd4587f`. It is
  derived from the `f099-02` configuration
  (`8b3c461f95b81fff210572ba44e48c67a74be677f8dc7e4cee6e3b607b2440a5`).
- Prepared plan `prep/plan.json`, SHA256
  `3aeee295a0e617923ba2d9f5f386b94ce6263f9d65ab0d2c870c739ed3982376`, built by
  the unchanged `l1_prepare.py`. Every bound input rehashed equal before
  preparation.
- **One variable:** `inputs/derive.py` writes `inputs/derivation.json`. It
  changes `EXL3_FUSED_MOE=1` to `0` at reference argv index 57 (rank 0) and
  37 (rank 1), and in the reference provenance environment. After attempt and
  prep paths are normalized, a structural diff of the two plans shows exactly
  10 `EXL3_FUSED_MOE` changes, all in the two reference windows (commands,
  environment, rank environments). Every other plan difference is a path, the
  attempt name, an input hash or the added preserved attempt.
- Candidate windows match `f099-02` apart from attempt-scoped paths and names:
  image `src-4ee96c12b44c` (`sha256:82173ea6…`), selected-treatment
  `reference-exact` fixture and the same descriptors.
- Reference windows otherwise keep the `f099-02` profile: image
  `glm53-flash-mikeroz:dt-carry20` (`sha256:6eb42fa2…`), v5 overlay
  `a07370fe…`, runner `f84255d7…`, model source `a1214289…`, and the
  `controlled-headless137` worker shutdown policy.
- Collector: unchanged `l1_collect.py`
  (`e4579c7d976d03b913709c18538cdf6a90553b6dba6d21e4feffa70514b672ba`), with
  its schedule: candidate control, candidate observed, reference control,
  reference observed. The observer rule is unchanged.

## Criteria (frozen before collection)

**O1, observer (the collector's unchanged rule):** for each engine, control and
observed match on all eight final-logit rows, generated text, thresholds and
numerical/source profile. A pass makes the collector write `binding.json`.

**O2, cross-attempt capture reproducibility:**
`tests/exl3_validation/l1_capture_reproducibility.py` (SHA256
`cab25b590a771a36b2b67d038061702cf934b13ec5235d0a590971711061fb29`, 14 CPU
tests) compares `f099-02` `reference-observed` (fused on) with this attempt's
`reference-observed` (fused off), on both ranks. O2 passes only if all of the
following hold:
- every compared file is first verified against its attempt's collection
  `receipt.json` (`file_sha256`), in both directions;
- the capture file sets are identical;
- every payload is byte-identical;
- sidecars are identical, compared type-strictly, apart from three registered
  per-run fields:
  - the random 8-hex suffix of `chunk_position.request_id`;
  - the absolute paths of `chunk_position.active_markers`. Each marker must be
    the window's own `logits/` or `diagnostic/` marker, and the
    window-relative set must match (rank 0 arms both, rank 1 only
    `diagnostic/`);
  - the process ID in hook-source record file names. The records are compared
    as a multiset of contents, so per-module counts must match.

A refusal by the comparator is an O2 **fail**, and its report records the
reason. File names carry the overlay's per-process invocation counter. It
advances only while the capture is armed (`overlay/reference_ffn_trace.py:81–89`),
so warmup or profiling forwards cannot shift the numbering.

A self-comparison of the archived `f099-02` window passes: 2,857
receipt-verified files, 686 payloads and 1,421 compared files per rank.

**Bisect (only when authorized below):** unchanged `l1_bisect.py`
(`31084bbb4b9e6d4c378290a17da45bdfd59d0e3838cf8f9565dc169d018dee7a`, which
imports `analyze_localization.py`
`32f57ed1fe4318ca406e4db43ee719b6af793f84c1ba408f6bfe2cd4450ef2a5`). It runs at
layer 1 on ranks 0 and 1, on this attempt's `binding.json`, observed
diagnostics and collections.
- Primary: `longer` rows `4,5,6,7`.
- Secondary: `arithmetic` rows `0,1,2,3`, descriptive.

The exact command lines are frozen in `post_collection.sh` (below).

Bisect outcome rules:
- `analysis_complete=false` for a rank/row set means no attribution for it.
  Record `incomplete_pairs`. Do not rerun with changed inputs.
- A row's attribution is its earliest divergent stage in spine order across
  both ranks, with the diverging rank(s) recorded. If one rank shows no
  divergence for that row, say so.
- If a row has no divergent stage and a complete prefix on both ranks, the
  candidate matches the reference through every captured layer-1 stage. Its
  divergence then lies downstream of the layer-1 boundary, which a
  separately registered later-layer bisection (G1) must take up.
- `attribution_blocked` rows are reported as blocked.
- A single-stage attribution for `longer` is claimed only if all four rows
  share the earliest stage. Otherwise report the per-row distribution.

**Descriptive, no verdict:**
- each window's final-logit rows against the same window in `f099-02`. The
  candidate windows are expected to be identical; the reference windows are
  expected to differ, because the loop path changes routed arithmetic;
- the autotune post-state against the determinism post-state.

## Interpretation (frozen)

| O1 candidate | O1 reference | O2 | Meaning and action |
|---|---|---|---|
| pass | pass | pass | Arming the capture does not perturb the reference. Layer-0/1 reference captures reproduce byte-for-byte across processes, attempts and the MoE setting. Run the bisect. Its layer-1 result transfers to the fused-on qualification profile, because the compared reference tensors equal the fused-on `f099-02` ones (and HSL found layers 0–2 identical across fused-on processes). It says nothing about layers ≥ 2. |
| pass | pass | fail | The capture is non-perturbing in this profile, but layer-0/1 captures differ across attempts or with the MoE setting (allocator state, tuning or something else). The bisect may run, labelled fused-off profile only; nothing transfers. Record the first mismatch; localizing it needs a new registration. |
| pass | fail | any | Final logits of rows below 2,052 tokens differ between control and observed. The fused-off loop path was deterministic for such rows in the determinism test, which used frozen caches and the context-v4 overlay, not this overlay or the live cache. Candidate explanations: capture perturbation, a profile-specific source, or a tuning-run versus cached-run launch difference, which the source review above says should not occur. Attribution is withheld. Compute O2 descriptively. Do not retry unchanged or add a tolerance. |
| fail | any | any | Candidate observer regression against a previously passing image and profile. Stop and investigate before any reference conclusion. |

Autotune rules:
- If the frozen pre-state check fails, collection does not start.
- If the post-state differs from the determinism post-state, record it. O1 and
  O2 remain interpretable, because observed reads control's choice and
  layer-0/1 shapes are pretuned. The determinism-test configurations no longer
  describe the loop shapes, and the result says so.
- If the restore fails, stop all GPU work until the file is restored.

A lifecycle or collection failure leaves the attempt failed and retained.

This study cannot resolve the H3/H4 reference-ensemble failures, change gate
v2 or qualify anything. Any adoption of a deterministic reference profile for
qualification needs its own registered amendment and campaign.

## Execution (frozen)

Run the scripts in the evidence root, in this order. `run_collection.sh` wraps
all of them:
1. `scripts/inspect-sparks.sh` to `host-inspection-pre/`. Requires idle GPUs
   and no other experiment owner. The P0 trace session must have released
   both GPUs.
2. `autotune_guard.sh pre`.
3. `l1_collect.py --plan prep/plan.json --plan-sha256 3aeee295…`.
4. `autotune_guard.sh post`, which always runs and restores the cache.
5. `scripts/inspect-sparks.sh` to `host-inspection-post/`.
6. `post_collection.sh`: O2, the descriptive comparisons, then the four bisect
   runs only if the receipt says `observer_passed`.

The script SHA256s, also recorded in `scripts.sha256` in the evidence root:
- `autotune_guard.sh`
  `073a15fbb2df5e7b2b200ec8f0cb331078a2c14c2fcaf3563ab622ff27e2c1e1`;
- `run_collection.sh`
  `541ca98288a5d415fba1b399f95fc82e98dcb56e299eac2c01e12ad27f6c821f`;
- `post_collection.sh`
  `ef864a1237d00d51f04a1dbd6caa1d04364a8197f906cea51a706b482bac51d7`;
- `inputs/derive.py`
  `f1b4b6a1a0d344b36c7e194fa64ec538366ad231e2c9cbf015759e97dcdbd43f`.

`run_collection.sh` refuses to start unless both hosts report no containers
and no GPU processes. Its `EXIT` trap always runs `autotune_guard.sh post`
once `pre` has succeeded, and it exits nonzero if the restore fails.
`autotune_guard.sh` checks both ranks (frozen hash and `sudo -n`) before
backing up either. In `post` it attempts the restore on both ranks before
failing. Its failure paths were exercised against mock hosts:
- a rank-0 restore failure still restores rank 1 and exits nonzero;
- a retry restores both ranks;
- re-running `pre` over an existing backup refuses. spark0 had 19 GB free on 2026-10-15. The `f099-02` C2+L1 footprint
there was 3.1 GB. Retained `idx`/`hsl` dumps are not deleted for space.

## Pre-collection review

An independent review on 2026-10-15 returned **BLOCK** on one major finding,
the unfrozen read-write autotune cache, plus minor findings and nits:
switch side effects, refusal and bisect outcome rules, full hashes, baseline
receipt verification, comparator strictness and wording. All are addressed
above. A confirmation re-review returned **GO-with-notes**. Its three script
failure-path bugs (restore skipped on a rank-0 failure, `pre` blocking its
own re-run, no trap or exit propagation) and three nits (full post-state
hashes, script hashes here, row-count assertion) are fixed. The reviewer
independently confirmed:
- the one-variable plan diff;
- every plan, config, module and tool hash;
- that the comparator refuses candidate captures;
- that sidecars carry no other per-run fields;
- that counters advance only while armed;
- that rows 4–7 are `longer` with `2r`/`2r+1` phase pairing;
- that the tune key buckets rows as `min(roundup_pow2(m), 16)`
  (`exl3_gemm.cu:72`), so every L1 row uses the load-time-pretuned bucket 16;
- that the input transform goes to a separate buffer
  (`linear.cpp:41–43`), so tuning launches never modify the GEMM input.

## Execution record

- `l1-fusedoff-f099-01` ran 2026-10-15 05:19–05:48Z through
  `run_collection.sh`, with the hosts idle-checked first. All four windows
  completed and the collector exited 0. The registration freeze record
  `registration-freeze.json` (05:18:42Z) precedes it.
- **O1** passed for candidate (8/8) and reference (8/8).
- **O2** passed on both ranks (686 payloads per rank; 2,857 receipt-verified
  files per side).
- The autotune post-state equals the determinism post-state, and both ranks
  are restored to the frozen state.
- **Bisect:** all four runs ended with `analysis_complete=false`
  (`boundary_streams`, `boundary_post`) → no attribution, as registered.
- See the [result](l1-fusedoff-observer-result-20261015.md).
