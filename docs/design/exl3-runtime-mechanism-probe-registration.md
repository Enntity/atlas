# EXL3 reference runtime-mechanism probe — registration

Registered 2026-10-14. **Executed 2026-10-14/15; see the
[result](exl3-runtime-mechanism-probe-result-20261015.md).** This registers the **diagnostic** designed in
[the runtime-mechanism plan](../plans/exl3-reference-runtime-mechanism-plan.md)
(the SSOT for mechanism rationale, telemetry boundaries and probe design).
It is **not** a gate amendment: H3/H4 constants, component count, singleton
treatment, fixed campaign N and reference identity are untouched, and no
probe sample may ever be used as a qualification draw. The frozen
UNVERIFIABLE v2 campaign result stands.

Per the plan, no Spark may be obtained before the freeze below completes.

## Attempt list (frozen)

Fixed order, fresh processes, three repeats per arm; the **only** arm
difference is the telemetry package/profiler enabled state:

    off01, on01, on02, off02, off03, on03

Every predetermined attempt is reported. No attempt is added, replaced or
retried unchanged; a failed attempt is retained. Do not add attempts until
the historical singleton reappears.

## Profile (frozen)

- Checkpoint `MikeRoz/GLM-5.3-Flash-Uncensored-4.05bpw-h6-exl3` revision
  `dda99c7e935920fa6e235ce410e6f806e1c02720`; TP=2/EP=1; temperature 0;
  seed 42; eager; full short16-then-long request order; chunk128 runner;
  reference effective MLA format `auto`→`fp8_ds_mla`
  (continue.md §6 frozen comparison profile).
- Cache handling (registered here explicitly because it changes both arms
  identically): freeze a per-rank copy of the **present** cache state, then
  initialize an isolated writable copy from that same snapshot for each
  attempt, mounted at the original container paths. Original campaign
  caches are never reset or overwritten.

## Engagement and incompleteness (frozen rules)

From the plan's "Minimum next reference probe", steps 3–5:

- One complete telemetry chain per target request on **both** ranks.
  Startup-only logs, missing child CUDA traces, or zero callback counts
  fail engagement; analysis terminates as **incomplete**, never "inert".
- Router engagement is judged from **per-request** counters
  (`ids_written`/`gate_logits_written` with a marked context), not from
  `capture_callback_calls` alone: that counter includes unmarked calls,
  so a nonzero value alone does not prove engagement on a target request.
- Before any mechanistic attribution, compare on/off final-logit
  distributions and capture identity. Same-class byte matches are scoped
  evidence of neutrality only; a telemetry-induced systematic class is an
  observer effect and reduces observer overhead before any new design.

## Harness freeze (CPU-first deliverable, this batch)

Reviewed GO-with-notes 2026-10-14 (independent review); all P2 notes
dispositioned (`_bound_identity` SSOT dedup, `group_reason` binding,
setattr rollback, capture-callback zombie cleanup). 63 focused
decoder/binding tests plus the full `tests/exl3_validation` CPU suite
pass; SPDX license gate clean.

| Frozen harness file | SHA256 |
|---|---|
| `tests/exl3_validation/linear_launch.py` | `ab1ec6407a43d35970791ab81bfd911d0ef4dd8c69dc2851e17386c5bd1ec81a` |
| `tests/exl3_validation/reference_router_capture.py` | `faf13fc5c86624a9850a7b8561570c84650f45d16c57f15311d00131217d8a26` |
| `tests/exl3_validation/reference_collective_trace.py` | `f82a175b4635b7b899c423fe90f509e512d56f4509a8299b4531459541a1b26a` |

Any change to these files requires re-freezing these hashes (and a fresh
review record) before collection.

## External bindings (SSOT pointers, not duplicated here)

- Reference source anchors (hash table): plan §"Source and binary contract".
- Registered reference image/build/checkpoint/observer hashes:
  `docs/design/evidence/gate-v2-reference-diagnosis-20261014/diagnosis.json`.
- Reference behavior contract:
  [the vLLM reference contract](exl3-vllm-reference-contract.md).

## Owed before obtaining either Spark (not yet frozen)

Recorded 2026-10-14 by `tests/exl3_validation/probe_package.py` /
`probe_records.py` (CPU suite green: 695 tests + 280 subtests, license gate
clean). Evidence root: `probe-runtime-mechanism-20261014/`.

- [x] Evidence-root overlay package assembly (sitecustomize lineage copy +
  hash manifest) wiring the frozen harness modules; package hash recorded.
  Assembled at `probe-runtime-mechanism-20261014/probe-overlay/` from the
  lineage `q2-l1-tooling-20261013/ref-overlay` (binding SHA256
  `a07370fec1ba57c300e0367a6e7e72dab525025c9024b7328cd8168db8dd328a`).
  `package-binding.json` SHA256
  **`9068c0972f0995095b5aa5f8734f3a6a75ae0be7ee7b2e686a4679a45a07ece9`**
  (21 files, re-verified after assembly; `__pycache__`/`.pytest_cache`
  excluded). It binds this registration at SHA256
  `9e39cc3a865e79ba5489721944c3a827157b9874b6140d0fd74592e19fdad8f6`,
  recorded inside `package-binding.json`; the superseded first assembly
  (which copied lineage `.pytest_cache` junk) is preserved untouched at
  `probe-overlay-v1-superseded-pytestcache/`. The package's
  `sitecustomize.py` chains the renamed lineage copy
  (`lineage_sitecustomize.py`); `reference_norm_binding.py` — the trio's
  imported dependency, not listed in the freeze table — is bound in the
  package manifest. On-arm processes install a SIGTERM flush handler for
  teardown persistence; handler presence is part of the registered arm
  difference (telemetry enabled state), not a cross-arm invariant.
- [x] Profiler/CUPTI selection, version and exact invocation hash; if
  launch tracing is unavailable, record that limitation here **before**
  the attempts rather than at collection time (plan: stop at a known
  tooling limitation instead of rebuilding the native binary ad hoc).
  **Selection and invocation registered now:** `nsys profile
  --trace=cuda,nvtx --sample=none --cpuctxsw=none -o {output}` plus the
  engine argv; invocation SHA256
  `fdb01a69956577795cbbed79e41b9ae49987446deb9c3d350b71aa07ef4f3fe2`
  (recompute with `probe_records.py`'s `invocation_sha256()`; the flags live
  in `tests/exl3_validation/probe_records.py`). Owed pre-attempt (before
  `off01`) on both Sparks: run `probe_records.py profiler bind --nsys
  <path>` — or `profiler unavailable --reason "..."` — and paste the
  emitted record here; version and availability cannot be bound before a
  host check.
  **Bound 2026-10-14** (`probe_runner.py profiler`, records
  `probe-runtime-mechanism-20261014/profiler/rank{0,1}.json`): both ranks
  report `NVIDIA Nsight Systems version 2025.3.2.474-253236389321v0`,
  binary `/opt/nvidia/nsight-systems/2025.3.2/target-linux-sbsa-armv8/nsys`
  SHA256 `f8eabc8c88cf08b874401fba616321cbe418cc3a815cbf3140edece6339c74dd`,
  invocation SHA256 `fdb01a69…` as registered, no limitation. The binary is
  host-installed and bind-mounted read-only into the container. A tooling
  check (`tooling-nsys-check/`, not an attempt, no model) recorded CUDA
  kernels from a spawned child inside the reference image and a finalized
  report after `docker stop`. **Superseded by the recorded limitation**
  (`profiler/limitation.json`, pre-collection correction 6): forked
  workers are not traced, so the profiler is disabled in both arms.
- [x] NCCL loaded-library path hashes on both ranks.
  Prior loaded-run evidence bound by hash for both ranks plus the reference
  runtime library: `probe-runtime-mechanism-20261014/nccl-prior-binding.json`
  (spark0/spark1 loaded-sha files SHA256
  `21d5e7c70a1269bc39eb62e33a96a51a1a96e293d2e45c8392947fd8aa0362fd`
  each; reference runtime evidence SHA256
  `085567870297f457dfd91640a888146477947c96d7baf8f4d4dd86c73bf37f0e`).
  Stated limitation inside the binding: live re-bind from running reference
  processes is owed during pre-attempt setup (`probe_records.py nccl live
  --maps /proc/<pid>/maps --proc-root /proc/<pid>/root --rank N`).
- [x] Attempt-list attestation copied into each attempt receipt.
  Code-side freeze and registration parse agree:
  `probe-runtime-mechanism-20261014/attempt-attestation.json` (schema
  `atlas-probe-attempt-attestation-v1`, 6 attempts, arms off/on/on/off/off/on).
  Every attempt receipt is stamped with `probe_records.py stamp
  --registration <this document> --receipt <dir>` at attempt creation; a
  second stamp into one receipt fails closed.

## Pre-collection correction (2026-10-14, before any probe attempt)

A pre-execution source check found three defects in the package recorded
above that would have voided attempts. No Spark attempt had started; the
attempt list, arms, profile and frozen trio are unchanged.

1. **Observer lineage.** The package chained the L1 context-v5 overlay
   (`q2-l1-tooling-20261013/ref-overlay`, sitecustomize `741e422b…`), but
   the gate-v2 reference attempts whose classes this probe studies ran the
   context-v4 observer (`/opt/atlas-logits-patch`, sitecustomize
   `81771af5…`). "Keep the exact … observer mounts" requires the v4
   package: lineage is now `upstream-reference-package-context-v4/`
   (binding SHA256
   `69e54138297b9bb10ddc3d0e895fd37419bed570051deab988fc861013177513`),
   byte-identical to the copy mounted on both Sparks. The probe overlay
   replaces that mount at the same container path in **both** arms.
2. **Off arm captured nothing.** `probe_sitecustomize.install()` returned
   before chaining the lineage when telemetry was disabled, so off attempts
   would have written no logits. The off arm now installs the lineage
   observer only (`5162eef1`, regression test).
3. **On arm would fail closed.** Router discovery searched
   `named_modules()`, but the pinned `FusedMoERouter` is a plain `router`
   attribute of the `MoERunner` module (`moe_runner.py:257`); fixed in
   `d11a42d0`. Separately, every Python process in a rank container shares
   the telemetry directory, so flushes are now keyed `pid-<pid>/`
   (`c4ac3c32`).
4. **Communicator trace silently absent.** A CPU-only import check in the
   reference image (no GPU, no model) showed that startup `.pth` hooks
   import `cuda_communicator` before `sitecustomize` runs, so the import
   finder never patched it. Targets already in `sys.modules` at install
   are now patched in place and listed in the receipt as
   `patched_after_import`. The first assembled v4 package (`3f0acbcb…`)
   predates this fix and is preserved as superseded; no attempt used it.
5. **On arm would crash at the first capture, and the last request would
   be lost.** Found by independent review. The lineage chunk trace owns
   `GPUModelRunner.prepare_inputs` and raises at capture time if that
   attribute is rewrapped after its install
   (`reference_chunk_trace.position_context`). The probe now marks requests
   from `execute_model` (`num_scheduled_tokens`; idle and dummy steps
   unmark) and leaves `prepare_inputs` alone. vLLM workers are forked
   (`VLLM_WORKER_MULTIPROC_METHOD=fork`) and exit through `os._exit`, and
   the executor replaces the SIGTERM handler, so neither atexit nor SIGTERM
   flushes run. The probe therefore flushes at the end of every
   `execute_model` step: a defined boundary outside any routing or dispatch
   frame. Analyzers sum per-request counters across flush windows.
6. **Launch tracing is unavailable for the engine's workers (tooling
   limitation, recorded before any attempt).** The plan anticipates this
   case. With nsys 2025.3.2.474 in the reference image, a CUDA workload in
   a fork-without-exec child records no kernels and only 3 runtime calls.
   This held with the registered flags, with `--trace-fork-before-exec=true`,
   and with `--cuda-flush-interval=1000`, for both single and double fork
   (`tooling-nsys-fork-check{,2,3,4}/`). A spawned child is traced
   (`tooling-nsys-check/`), but vLLM forks its GPU workers by default, and
   switching to spawn would change the engine profile. Per the plan, the
   probe stops at this limitation and does not rebuild: **neither arm runs
   nsys**. The arm difference is the two `ATLAS_PROBE_*` variables only.
   The probe therefore does not observe executed EXL3 launch plans or
   collective kernels. Router decisions and the collective dispatch
   sequence remain instrumented. The plan's "engagement" rule for CUDA
   traces does not apply. Plan selection stays unresolved unless a later,
   separately registered design (e.g. a CACHEDEBUG observer build) observes
   it.

Execution bindings (runner `tests/exl3_validation/probe_runner.py`):

- Container commands derive from the gate-v2 reference create commands
  (`gate-v2-qualification-20261013-luna1/driver/profile/ref-rank{0,1}-*`).
  Changes: fresh `probe-rm-<attempt>-ref-<rank>` names, an attempt-scoped
  logits mount, the probe overlay at `/opt/atlas-logits-patch`, and a
  telemetry mount in both arms. The vLLM
  (triton/tilelang/nv subdirectories) and EXL3 tune caches are mounted from
  the attempt's copy of the frozen snapshot. On arm only: the two
  `ATLAS_PROBE_*` variables (no profiler; correction 6).
- Cache snapshot: one `cp -a` per rank of the campaign vLLM cache parent and
  `reference-autotune` into `/home/colum/probe-rm-20261014/cache-snapshot/`;
  originals are only read. Every attempt's copy must hash equal to the
  snapshot before start; the post-shutdown manifests are recorded.
- Probe config: `probe-runtime-mechanism-20261014/probe-config.json`
  (installed-source hashes verified in the reference image; backends
  PYNCCL `all_reduce`/`all_gather`, the only backend vLLM logged for
  `tp:0`/`ep:0`). Limitation: the logits head uses `gather`, which the
  frozen collective trace does not wrap. Head-gather dispatch is
  therefore unobserved; kernel traces still show it.
- Provenance: the gate-v2 finalizer, changed only to read the attempt's
  profile directory (`driver/finalize-probe-reference.py`). The draft is
  the campaign `prov-ref.json` with the probe hook hashes, plus the
  telemetry environment on the on arm.
- Collection: the campaign's short16-then-long collector commands and
  ledger-verified inputs on spark0.
- Stop: `docker stop --timeout 120` in both arms. Fixed mount sources are
  frozen once by type and hash (`mount-freeze/`); every attempt must match
  them, so a missing `/tmp` patch file can't silently become an empty
  directory. Preflight failures, before any host change, do not consume an
  attempt. After the stamp, every outcome tears down, archives and advances
  the order.
- NCCL live re-bind: during `off01`, after the API is ready and before
  collection, from every container process that maps libnccl.

### Package and records that run (supersede the package item above)

- Package `probe-runtime-mechanism-20261014/probe-overlay/`, lineage
  context-v4. `package-binding.json` SHA256
  **`9b6b2be8cda751c0c0b39b5c8a9b095108e48f4ef383484ab295685417938c68`**
  (20 files) binds this registration at `f974b47bf6da598620e949b0666eb5e2e6677be1aaad1f05e9b3f98b1291a488`, i.e. this document before
  this subsection. It is deployed read-only at
  `/home/colum/probe-rm-20261014/probe-overlay` on both Sparks, and each
  attempt's preflight re-verifies it. Superseded packages are preserved:
  `probe-overlay-v2-superseded-l1lineage/`,
  `-v3-superseded-preimport/` and `-v4-superseded-prepareseam/`. No attempt
  used them.
- CPU-only in-image import check on both ranks and arms
  (`import-check-9b6b2be8.json`). Off arm: lineage observer only, with
  `prepare_inputs` still the lineage wrapper. On arm: `execute_model`,
  `Glm5NextMoE.__init__` and `CudaCommunicator.__init__` (patched in place)
  patched, the lineage wrapper intact, all 8 source targets bound.
- `probe-config.json` SHA256 `c27cf2269eecf24f82fa8b412422a804e1014480a34459aeaf82516485923b34`; finalizer
  `driver/finalize-probe-reference.py` SHA256 `14b64604625b76f04c4f6db980b6b412c4bde6e51b1d6987c591b13b76a5fcfb`; profiler limitation
  `profiler/limitation.json` SHA256 `154dfc7b5370e4eb23f71c84a65899c5554811980d7a4c57345578216f3f6c09`.
- Cache snapshot `cache-freeze/rank{0,1}.json`, equal to the originals.
  Fixed mount sources `mount-freeze/rank{0,1}.json` (37 per rank, none
  missing).
- Runner: `tests/exl3_validation/probe_runner.py` at the commit that adds
  this subsection. Every attempt receipt records the runner, wiring,
  registration, package, config and finalizer hashes.

### Mid-campaign record: on01 failed (2026-10-14, after off01 and on01)

- `off01` **complete** (package `9b6b2be8…`; both ranks exit 0, caches
  unchanged pre→post).
- `on01` **failed** during engine warmup, before the API was ready, on
  both ranks: `FileExistsError` in the probe flush. Within one step every
  routed layer writes the stage name `router-gate-logits-000` (stage
  counters are per context), and the flush named files by stage only. It
  is a wiring defect, not a reference observation. The attempt is retained
  as failed and is neither retried nor replaced. The arms are now two on
  and three off.
- Fix: tensor files are named `<window position>-<stage>.bin`, with a
  42-layer regression test. The package is rebuilt for `on02` onwards
  (recorded below); `off02`/`off03` run the same package. The on-arm
  telemetry code changes between `on01` and `on02`, but the off arm's
  executed code does not: off attempts never import `probe_wiring`.
- Before `on02`, an **unregistered engagement smoke** ran the full engine
  with on-arm telemetry and one short request, with logits capture never
  enabled. It runs outside the attempt list, its outputs never enter any
  class or metric analysis, and it is recorded under `tooling-engagement-smoke/`.
  **Result (package
  `c97451459768273b9f299e6e8229ccffc1ebc0c4ee13bcc9a407f6c02f3ac8e6`):**
  both ranks exit 0. The request is marked with the same ID on both ranks,
  with 336 gate-logit, weight and ID captures each (42 routed layers × 8
  steps). There are 42 router installs and 2 dispatch installs. All 1013
  collectives went through PYNCCL/PYNCCL_GATHER, with 0 coverage
  violations and no absent backends. 593 collectives fall outside MoE
  forward frames and carry ctx `None`; that is the frozen trace's design,
  and they remain sequenced. `on02`, `off02`, `off03` and `on03` run this
  package.

## Overlay-author obligations (carried from review residual risks)

- `context()` must be side-effect-free, bound **per router/communicator
  instance** (the bindings cannot cross-check the layer index), and carry
  invocation/chunk identity in its 4-tuple.
- `write_tensor`/`write_metadata` callbacks must buffer and flush at a
  defined boundary; blocking synchronous I/O inside dispatch/routing
  frames is an observer-effect risk.
- The collective coverage-violation raise happens **inside** the dispatch
  frame: treat it as attempt-failing, never retryable.
- The collective trace's frame stack is not thread-safe; it assumes the
  one-forward-per-process TP execution model.
- Stage names use `%03d` counters per (context, stage); parsers must not
  assume exactly three digits.
