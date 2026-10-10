# vLLM research ideas to implement and test in Atlas

Review date: 2026-10-09. This is an implementation backlog, not a claim that
the proposed changes are qualified for Atlas. No GPU experiments, serving-node
changes, or runtime implementation were performed during this review.

## Highest priority: native EXL3 support

### T00 — Add EXL3 checkpoint and kernel support to Atlas [V/M source; U in Atlas]

User priority: implement and test this before the other optimization candidates.
Build the loader/runtime foundation first, then qualify the specialized kernels
in separate bounded PRs. T01/T02 evidence and telemetry gates accompany this work.

- **Sources:** Spark-GLM53 [EXL3 mul1 integration](../../spark-glm53/docs/MIKEROZ_MUL1.md), [cooperative MoE](../../spark-glm53/docs/cooperative-moe.md), [thin-decode kernels](../../spark-glm53/docs/sm121-perf-paths.md), [independent kernel references](../../spark-glm53/docs/exl3-sm121-kernel-lab.md); Mindreader [packed dense implementation](../../mindreader/patches/dense-packed/README.md) and [reviewed integration](../../mindreader/docs/research-reports/69-packed-dense-stage-b.md).
- **Evidence boundary:** the source establishes EXL3 execution and bounded kernel/serving observations. It does not establish Atlas compatibility, a universal speedup, or complete numerical clearance of the thin/cooperative paths. Keep MCG and mul1 codebooks, checkpoint-specific transforms, and quantization metadata distinct.
- [ ] **Checkpoint/ABI contract:** inventory packed tensors, codebook variants, scales/transforms, permutations, shard geometry and supported shapes from pinned sources. Add explicit EXL3 format dispatch and validation; reject unknown or mismatched metadata before allocating/publishing runtime weights. Preserve upstream license notices and register the quant target through Atlas's existing kernel registry.
- [ ] **Correctness foundation:** write independent CPU dequantization/matmul reference tests with frozen fixtures before loader/kernel integration. Cover MCG and mul1 where supported, gate/up/down and dense projections, scale/permutation boundaries, TP/EP shard reconstruction, malformed tensors and unsupported geometry. Add native Rust/CUDA loading and a baseline EXL3 execution path; keep Python confined to research/test harnesses.
- [ ] **Kernel portfolio:** integrate packed dense GEMM and routed MoE first, then add capability-gated small-row/thin and cooperative decode variants. Preserve grouped prefill/fat-expert handling. Transform sharing requires all-expert scale equality and owned pointer lifetime; scratch/worklists must support actual stream and graph ownership. Do not select MCG-only kernels for mul1 weights.
- [ ] **Native qualification:** compare complete layer outputs against independent references and calibrated baseline controls; exercise changing routes/data, concentrated experts, row/shape boundaries, graph replay, non-default streams and compute-sanitizer. Run real EXL3 checkpoints through serial/speculative decode, prefill, supported TP/EP, useful answers, tools, retrieval, cancellation and clean shutdown. Verify current supported model/quant paths remain correct.
- [ ] **Performance promotion:** perform same-checkpoint, same-image EXL3 baseline-versus-candidate A/Bs with explicit dispatch witnesses, repeated fresh-process controls, full-wall/TTFT/per-stream measurements and allocation/headroom receipts. Measure dense, thin and cooperative additions separately. A comparison against an NVFP4 checkpoint is a separate quality/performance tradeoff, not an isolated kernel gain.
- **Done when:** Atlas loads and serves the declared EXL3 variants using its native runtime, correctness/memory gates pass, and each enabled optimized path has Atlas-specific performance evidence. Experimental kernels remain opt-in until individually qualified; support does not depend on every optimization proving a gain.

## Narrow shortlist: measured source improvements with missing Atlas mechanisms

Follow-up source/code check distinguishes candidates worth porting from the
larger audit/qualification backlog. T00 above takes precedence by user direction;
the table below ranks the measured source mechanisms after that foundation.
No improvement has yet been proved in Atlas.

| Priority | Mechanism to add for testing | Source evidence | Atlas gap and boundary |
| --- | --- | --- | --- |
| 1 | Role-aware CPU-thread placement separating runner, NCCL progress, callback/copy, and background work (T14) | Mindreader report 66 Part B: reviewed same-boot ABBA, supported step-time improvement with manipulation checks. | Current Atlas recipes expose broad container CPU sets, but the reviewed paths do not implement this per-role placement policy. Source control concentrated tasks on one CPU; Atlas's broader placement may have less headroom. Map Rust/NCCL thread roles and isolate background/tuner changes in the experiment; one boot and unresolved source quality/acceptance differences limit the result. |
| 2 | Fair-prefill service credit with fixed-plus-token cost, reservation/refund and async completion accounting (T08) | Spark-GLM53 v5 live TP2/TP3 retests demonstrate useful newcomer service while preserving incumbent streaming; individual runs only. | Atlas's current SLAI suppresses overdue prefill and otherwise uses the full chunk; it lacks this specific shared service-credit/affine-cost controller. Existing Atlas evidence rejects naive chunk shrinking. Test service allocation and cost estimation against that control; a fairness/latency improvement is not a universal throughput gain. |

Compact draft pages/exact boundary lookup (T25) also have measured source reuse
benefit, but the current Atlas DFlash implementation uses separate paged caches
rather than the source's padded slot-shared hybrid group. First demonstrate an
analogous reservation/replay bottleneck; the page-ID patch is not a direct port.
Packed dense, adaptive speculation, B-tile MoE, and FlashKDA already have Atlas
implementations. EXL3-specific cooperative/thin kernels require a different
weight ABI and carry incomplete numerical qualification. None is a newly
missing, numerically cleared drop-in kernel improvement identified by this review.

## Scope and evidence rules

Reviewed the research reports, plans, decision/task ledgers, relevant code,
tests, and retained evidence in both repositories, then checked the current
Atlas implementation for overlap. Repository revisions at review time:

| Repository | HEAD |
| --- | --- |
| Atlas | `3a6de55f78ac0a60aabf27b7136ad0eb8106473a` |
| Mindreader | `ce30f24c729b49f6ba92bc4d81d3b91b4c865667` |
| Spark-GLM53 | `205a5ab957fc5e2353f1ab70a0684ffc0393014f` |

The graph registry contains no registered repositories. Direct graph queries
found Mindreader's index stale and Spark-GLM53's empty; Atlas's supplied graph
also reports an incomplete build. Consequently, this review uses source files
and research/code entities directly, not graph completeness as evidence.
It does not claim exhaustive inspection of every graph node or artifact.
Links to the sibling repositories assume their present workspace locations.

**Evidence levels:** **V** = source-reported empirical validation, limited to
the stated experiment; **M** = mixed evidence, negative result, or inconclusive
qualification; **U** = design, source analysis, or untested hypothesis.
These labels describe source evidence, not Atlas readiness. Most historical
Spark-GLM53 GPU receipts are referenced at remote paths absent from this checkout;
they were not independently rerun. Committed APC comparison JSON and Mindreader
evidence are stronger local audit inputs, but reading them is not a fresh experiment.

Use the corrected report bodies and Mindreader's
[October 5 review](../../mindreader/docs/research-reports/review-full-2026-10-05.md)
over stale `continue.md` claims. Later Spark-GLM53 `todo.md`/`decisions.md`
entries supersede introductory design statuses. Mindreader contains October 10
addenda later than this session's date; do not rely on that chronology as
established evidence. Reports 88/89 separately record October 8 reviews.

No source throughput or compression ratio is promised as an Atlas gain.
EXL3 trellis, ModelOpt NVFP4, FP8, MTP, and DFlash have different representations
and costs. Transfer the mechanism and experiment design, then measure Atlas.

## What Atlas already implements

These are starting points to extend or qualify, not new missing subsystems:

| Mechanism | Current Atlas anchor | Remaining translation boundary |
| --- | --- | --- |
| Checkpoint-native packed GLM dense weights | [GLM dense loader](crates/spark-model/src/weight_loader/glm5/dense.rs) | Audit actual expansions/conversions and missing shape dispatch; EXL3 packing is a different format. |
| Small-row KDA projections and optional fused QKV | [projection](crates/spark-model/src/layers/glm5_kda/projection.rs), [FP8 plan](crates/spark-model/src/layers/glm5_kda/projection_fp8_plan.rs) | Qualify actual row distributions, large-row paths, retained bytes, and graph use. |
| B-tile grouped/decode/resident MoE and weight retirement | [grouped B-tile](crates/spark-model/src/layers/moe/gate_up_btile_grouped.rs), [resident path](crates/spark-model/src/layers/moe/gate_up_btile_resident.rs), [retirement](crates/spark-model/src/weight_loader/glm5/retirement.rs) | Prove selection and numerical/serving benefit for each new variant; do not rebuild this subsystem. |
| Adaptive speculation | [width policy](crates/spark-server/src/scheduler/dflash_width.rs), [rung policy](crates/spark-server/src/scheduler/adaptive_rung.rs), [suspend/reprobe](crates/spark-server/src/scheduler/adaptive_spec.rs), [MTP depth](crates/spark-server/src/scheduler/mtp_dcut.rs) | Cost/phase/concurrency calibration and engagement tests; paired GLM capability exclusions still apply. |
| Mixed prefill scheduling | [mixed execution](crates/spark-server/src/scheduler/phase_continue_prefills/run_batched_mixed.rs), [typed execution plan](crates/spark-model/src/traits/execution_plan.rs) | Qualify supported EP/model lanes and reader/writer fairness against the existing policy. |
| Prefix reuse and recurrent snapshots | [prefix cache](crates/spark-runtime/src/prefix_cache.rs), [radix tree](crates/spark-runtime/src/radix_tree.rs), [SSM snapshots](crates/spark-model/src/model/ssm_snapshot.rs) | Group retention, replay seams, capacity accounting, and placement experiments. |
| GLM single-latent storage and bounded raw tails | [cache plan](crates/spark-model/src/model/glm_cache_plan.rs), [allocation](crates/spark-runtime/src/kv_cache/paged_impl.rs), [tail slots](crates/spark-runtime/src/kv_cache/tail_slots.rs), [factory](crates/spark-model/src/factory/build.rs) | Factory selects V-aliasing; slotted tails are conditional on speculation with prefix cache/HSS excluded. Broader ownership qualification remains separate. |
| Indexed independent-row KDA | [indexed execution](crates/spark-model/src/model/ssm_indexed_decode.rs) | Already integrated; test new widths/state layouts rather than proposing its initial implementation. |
| Opt-in FlashKDA recurrence prefill | [adapter](crates/spark-model/src/layers/glm5_kda/flash_prefill.rs), [research qualification](research/flash-kda-prefill/README.md) | Bounded operator qualification exists; full-model qualification remains distinct from projection tuning and combined source-profile results. |
| Quantized KV and grouped/direct collectives | [KV formats](crates/spark-runtime/src/kv_cache.rs), [communication API](crates/spark-comm/src/lib.rs) | Existing lossy KV formats differ from lossless offload codecs; identify remaining uncoalesced live collectives before adding communication machinery. |

The older [vLLM infrastructure roadmap](docs/experiments/glm53-20260906/vllm-infrastructure-roadmap.md)
is useful history, but its missing-latent/missing-tail statements are superseded
by current code. Code existence alone does not prove a particular deployed lane
is qualified or that a feature engages.

## Priority 0: make experiments and serving failures trustworthy

### T01 — Fail-closed experiment receipts and numerical qualification [V/M]

- **Sources:** Mindreader [report 82](../../mindreader/docs/research-reports/82-evidence-harness-track-a.md), [full review](../../mindreader/docs/research-reports/review-full-2026-10-05.md); Spark-GLM53 [equivalence design](../../spark-glm53/docs/SPEC-EQUIVALENCE.md) and [thin-path qualification](../../spark-glm53/docs/sm121-perf-paths.md).
- [ ] Extend Atlas's benchmark harness with immutable build/binary/image, checkpoint revision, effective flags, graph shapes, quant/cache formats, sampler, request ownership, cold/warm state, and raw-evidence identities. Record independent boot/process assignments separately from repeated steps.
- [ ] Write refusal tests first: missing comparisons, incomplete candidate rows, mixed fingerprints, absent effective sampler fields, rank-deficient causal designs, unsupported evidence schema, and silently failing quality gates must not PASS.
- [ ] Distinguish fixed-output contracts from natural delivery: valid natural EOS is not under-delivery, while aborts, missing finish reasons and zero-token responses still require refusal. Preserve refused artifacts before raising. Mindreader's [live-receipt repairs](../../mindreader/docs/research-reports/54-live-receipt-arms-2026-09-28.md) establish these harness failure modes.
- [ ] Calibrate numerical tolerances with independent stock-repeat controls. Use exact equality for an exact-arithmetic promise; use declared distribution/quality tests for arithmetic-changing paths. An unsuccessful stock control makes the candidate verdict inconclusive, not cleared.
- **Done when:** receipt validation runs through real producer → stored record → comparator and cannot turn incomplete evidence into a promotion. Apply [measurement discipline](.claude/skills/measurement-discipline/SKILL.md) to every later task.

### T02 — Round-level speculative work and applied-policy telemetry [V/M]

- **Sources:** Mindreader [receipt repairs](../../mindreader/docs/research-reports/52-theory-testing-repairs-2026-09-27.md), [carry validation](../../mindreader/docs/research-reports/57-one-client-carry-validation.md), [watcher corrections](../../mindreader/docs/research-reports/83-mindreader-x0-x2.md); Spark-GLM53 [live tuning](../../spark-glm53/docs/LIVE-TUNING.md).
- [ ] Extend existing scheduler/model telemetry with request/round IDs, actual graph/depth selection, physically drafted and verified positions, accepted prefix, bonus token, draft/verify/repair/state-copy costs, thinking phase, and requested/effective sampler settings.
- [ ] Test complete and partial windows through the aggregation path. Marginal per-position acceptance and survival-conditional acceptance need distinct denominators. Verified-position counts do not establish physical draft work.
- [ ] Refresh attribution for each record and retain just-finished requests across the producer/aggregation seam; an empty/stale live registry must not turn a mixed completed step into a pure regime. [Report 54](../../mindreader/docs/research-reports/54-live-receipt-arms-2026-09-28.md) validates that narrow stale-registry failure and repair. Test finish/cancel concurrent with collection, not just steady active requests.
- **Done when:** a positive-control request demonstrates the selected feature on both ranks; ordinary, speculative, cancelled, and drained requests retain attributable complete records without charging diagnostic overhead to uninstrumented timings.

### T03 — Startup usefulness and no-progress detection [V/U]

- **Sources:** Spark-GLM53 [UVM incidents](../../spark-glm53/docs/uvm-livelock-gb10.md), [warmup script](../../spark-glm53/scripts/boot-shape-warmup.sh), [latest task ledger](../../spark-glm53/todo.md). Latest canary code has CPU checks, but default-on behavior is not fully live-qualified.
- [ ] Audit readiness and worker health; add a bounded useful-answer canary exercising the configured graph/speculative shape if existing readiness does not cover it. Add progress detection using active work and prompt/generation counters, not HTTP health alone.
- [ ] Test idle service, progressing long prefill, capture/loading grace periods, a stalled active request, peer failure, cancellation, and visible startup failure. Do not classify a long idle command wait as a failed collective.
- **Done when:** normal work stays healthy and stalled work produces an actionable error through the existing supervisor lifecycle. Reboots/swap resets remain operator actions.

## Priority 1: bounded implementation and qualification work

### T04 — Audit useful cache bytes versus actual allocations [V/U]

- **Sources:** Mindreader [report 89 and its storage-map evidence](../../mindreader/docs/research-reports/89-kv-cache-compressibility-screen.md); Spark-GLM53 [capacity accounting](../../spark-glm53/docs/DESIGN-kv-capacity-log.md).
- [ ] Derive a per-rank Atlas ledger from allocation owners: MLA latent, index values/scales/raw tail, attention KV, recurrent/convolution state, draft cache, snapshots, graph buffers, scratch, metadata, and reserved blocks. Separate allocated capacity, useful content, active occupancy, and retained-conversation capacity.
- [ ] If measured waste exists, prototype group-specific pools/page geometry or removal of structurally unused fields. GLM NoPE's zero RoPE field in vLLM is not proof Atlas stores it; current Atlas latent aliasing already addresses a different duplication.
- [ ] Test recycle/window ownership, reserve accounting, maximum legal requests, prefix forks, rollback, cancellation, and shutdown; reconcile allocator receipts with the typed plan. Unknown group semantics must remain explicitly unmodelled.
- **Done when:** a real model's before/after allocation ledger reconciles and pressure/quality tests pass. The source's padding calculation is not evidence of recovered Atlas capacity.

### T05 — Qualify GLM aliasing and tail-slot ownership beyond the current lane [U in Atlas]

- **Sources:** report 89 above; Atlas [cache plan](crates/spark-model/src/model/glm_cache_plan.rs), [tail lifecycle](crates/spark-runtime/src/kv_cache/tail_slots.rs), and [factory guards](crates/spark-model/src/factory/build.rs).
- [ ] First qualify the selected alias/slotted-tail lane end to end. Separately design capability checks for prefix reuse, HSS, and non-speculative arbitrary rewind before lifting their present tail-slot exclusions.
- [ ] Test pool-completion seams, prefill chunk seams, rollback spanning blocks, copy/free without duplicate ownership, shared prefixes, slot reuse, capacity exhaustion, changed graph metadata, and multi-rank agreement. Retain full tails for unbounded resumes until an exact reconstruction contract exists.
- [ ] Include a targeted padded-tail seed regression from Spark-GLM53's [confirmed stride defect](../../spark-glm53/overlay/patch_kpool_tail_seed_stride.py) and [CPU layout tests](../../spark-glm53/tests/test_kpool_tail_seed_stride.py): seed using the actual tail block/plane strides and slot map, not a dense assumed layout. Assert canaries in unrelated indexer blocks remain untouched. Atlas already passes explicit strides in its GLM prefill path; verify the CUDA consumer and every supported tail geometry. Local source tests establish the address contract, not a newly measured serving gain.
- **Done when:** full cache/state equality, memory-safety checks, retained-byte accounting, and near-capacity native serving pass for each newly admitted lane.

### T06 — Per-group prefix retention with exact suffix replay [V/M]

- **Sources:** Spark-GLM53 [qualification](../../spark-glm53/docs/apc-retention-qualification.md), [design](../../spark-glm53/docs/DESIGN-apc-per-group-retention.md), [committed branch-matrix evidence](../../spark-glm53/docs/qualification/apc-2026-09-05/branch_matrix.py).
- [ ] Extend existing Atlas prefix/snapshot ownership with explicitly selected retention policies, if T04 shows snapshot/drafter pressure. Reconcile attention, recurrent state, raw index tail, and draft window at one committed boundary; missing draft state must trigger bounded rebuild.
- [ ] Test append-only reuse, early/middle/late edits, branches, partial windows, eviction, state-generation mismatch, interleaved sessions, cancellation, and suffix replay. Check cached/computed tokens against actual tokenized prompts.
- [ ] Audit hit granularity separately: Spark-GLM53's [task ledger](../../spark-glm53/todo.md) proposes excluding nonparticipating tail managers from partial-hit restrictions, but that fix is source-reading only. Compare Atlas hash/block/checkpoint boundaries and prove a partial hit restores every required state before relaxing alignment.
- **Done when:** correctness and workload reuse both pass at equal physical memory. Source all-zero retention passed bounded correctness but regressed late-edit/branch reuse; do not make it the default.

### T07 — Legal sparse-index workspace bounds [V/U]

- **Sources:** Spark-GLM53 [workspace analysis](../../spark-glm53/docs/DESIGN-indexer-workspace.md), [implementation](../../spark-glm53/overlay/patch_indexer_workspace.py), [tests](../../spark-glm53/tests/test_indexer_workspace.py).
- [ ] Audit Atlas sparse-index scratch allocations for token-versus-pooled-entry confusion. Derive maximum simultaneous work from actual scheduled rows, pool ratio, context/token ceilings, and speculative lookahead, including a complete longest legal request.
- [ ] Add boundary/property tests for rounding, heterogeneous lengths, ragged drains, speculative tails, maximum legal batches, and fail-before-launch undersizing. Compare chunk lists and all top-k outputs with an independent reference.
- **Done when:** the bound is checked at initialization and actual allocation receipts establish any reclaimed bytes. vLLM's stock allocation was observed; the candidate's projected recovery is not an Atlas measurement. Do not shrink undocumented pitch/alignment requirements.

### T08 — Mixed-load fairness and EP scheduling qualification [V/U]

- **Sources:** Spark-GLM53 [scheduler overlay](../../spark-glm53/overlay/patch_scheduler_decode_floor.py), [v5 design and accounting corrections](../../spark-glm53/docs/astra-fix.md), [independent live TP2/TP3 overlap retest](../../spark-glm53/docs/diditfix.md), [other-recipe interference follow-ups](../../spark-glm53/todo.md); Atlas [execution contract](crates/spark-model/src/traits/execution_plan.rs), [mixed path](crates/spark-server/src/scheduler/phase_continue_prefills/run_batched_mixed.rs).
- **Validated boundary:** source v5 retests demonstrate newcomer progress during a long incumbent generation on TP2/TP3. These were individual runs, not a repeated multi-newcomer matrix or TP4 qualification; host busy-time estimates do not guarantee a hard GPU-step deadline. The separate NNNtrance/EXL3-Zeus TP2 reader-stalls-writer probe in `todo.md` remains unrun. The original backlog omitted the v5 evidence.
- [ ] Run a long reader arriving while short writers decode, then newcomer/short-request admission, against existing Atlas scheduling. Measure writer gaps, per-request TTFT, total work, queue delay, and fairness.
- [ ] Implement only a demonstrated policy/capability gap: decode floor, bounded prefill work, or supported EP packed execution. Preserve actual SSM identities and ordered rank-consistent work plans; do not simply remove EP exclusions.
- [ ] Screen v5's specific transferable mechanisms: fit fixed-plus-token step cost rather than scaling one tiny-chunk sample linearly; reserve/refund one shared credit balance, save credit for a useful legal rung, distinguish solo work from contention, and count overlapping async busy time once. Test late/duplicate completions, aborted grants, shared debt, alignment-rounded zero work, and several candidates spending the same credit. Do not import the policy constants.
- [ ] Test ragged drains, starvation bounds, cancellation, worker agreement, state reuse, and pooling/sparse transitions. Compare with the current policy and full chunk control; smaller chunks are not a proven universal improvement.
- **Done when:** repeatable native mixed-load improvement preserves useful answers, tool calls, retrieval, progress, and bounded memory on supported lanes.

### T09 — Qualify cost-aware adaptive width across workload and concurrency [V/M]

- **Sources:** Mindreader [prompt-mix study](../../mindreader/docs/research-reports/58-prompt-mix-acceptance-result.md), [width study](../../mindreader/docs/research-reports/55-width-ceiling-ab-result.md), [phase study](../../mindreader/docs/research-reports/56-answer-vs-reasoning-acceptance-result.md); Spark-GLM53 [adaptive graph implementation](../../spark-glm53/overlay/patch_adaptive_k.py).
- [ ] Extend existing Atlas policies only where T02 demonstrates a missing cost term or regime. Compare against the best fixed width and speculation off across code/chat/tools/math/prose, reasoning/answer phases, concurrency, and natural long contexts.
- [ ] Constrain changes to supported/captured widths and launch-time ceilings; verify actual applied selection and charge repair, bonus sampling, copies, observation, and policy costs. GLM paired plans that reject adaptive depth remain excluded.
- [ ] Test phase transitions, graph cache selection, low-acceptance suspend/reprobe, context changes, drains, rollback, and stale policy epochs.
- **Done when:** complete-request cost improves under matched repeated runs. Width/phase studies were inconclusive; no universal optimal depth follows from them.

### T10 — Native dense packing, tiny-row GEMM, and large-row projection portfolio [V/M]

- **Sources:** Mindreader [packed A/B](../../mindreader/docs/research-reports/69-packed-dense-stage-b.md), [trace](../../mindreader/docs/research-reports/70-kernel-trace-packed.md), [adoption limits](../../mindreader/docs/research-reports/76-adopt-packed.md); Spark-GLM53 [large-M KDA](../../spark-glm53/docs/kda-bf16-large-m.md).
- [ ] Profile current packed GLM loader and KDA/MLA/shared/output projections. Add only missing native-layout/row-shape variants; consider bounded retained BF16 copies of logical FP8 weights for large-row compute as an explicit memory tradeoff.
- [ ] Preserve checkpoint scale/rounding semantics; test decode/verifier/prefill crossover boundaries, fused QKV splits, TP shard layout, graph replay, non-default streams, finite outputs, retained memory, and independent full-model numerical controls.
- **Done when:** a shape-qualified portfolio improves complete requests without silently reducing KV capacity. Packed bundle source validation does not price each component; large-M source qualification was inconclusive. Activation W8A8 QDQ is a separate behavior-changing arm, not a substitute for this task.

### T11 — Router result reuse where duplicated [V/U]

- **Sources:** Mindreader [residual projection map](../../mindreader/docs/research-reports/71-residual-bf16-map.md), [dedup deployment](../../mindreader/docs/research-reports/74-gate-dedup-deploy.md), [patch](../../mindreader/patches/moe-gate-dedup/exl3_mul1.py).
- [ ] Inspect Atlas MoE router dispatch for repeated projection of identical hidden rows. If duplication exists, explicitly own/reuse its result across consumers. If absent, close the task with a trace/source witness.
- [ ] Test expert IDs, weights and tie behavior, TP/EP reduction order, changing hidden rows, lifetime, graph replay, multi-request isolation, and shared/routed branches.
- **Done when:** selection and full-layer behavior match and the removed projection is observed in a trace. Source deployment agreement does not establish a separately measured router saving.

### T12 — MoE tile, staging, reduction, and cooperative worklist experiments [V/M]

- **Sources:** Mindreader [compiled variants](../../mindreader/docs/research-reports/77-moe-compiled-variants.md), [benchmark](../../mindreader/docs/research-reports/79-moe-variant-benchmark.md), [N128 forensics](../../mindreader/docs/research-reports/80-n128-slowdown-analysis.md); Spark-GLM53 [cooperative kernel](../../spark-glm53/docs/cooperative-moe.md), [later ledger](../../spark-glm53/docs/astra-experiment-ledger.md), [thin path](../../spark-glm53/docs/sm121-perf-paths.md).
- [ ] Use Atlas's existing B-tile/grouped/resident paths as controls. Screen quant-compatible tile widths, stage depths, packed scale loads, gate/up versus down geometry, shared scratch, and cooperative scheduling at measured row/expert distributions.
- [ ] Record compiled registers, spills, shared-memory reservation and actual worklist/grid. Test concentrated/spread/invalid routing, shape edges, graph mutation, concurrent streams, scratch ownership, numerical oracles and sanitizer. Never retry another kernel after a partially mutating launch.
- [ ] Explicitly screen gate/up input-transform reuse where the quant layout supports it. The source thin kernels distinguish shared and independent transforms; sharing requires equality of every expert's packed transform scales plus owned pointer lifetime, not merely equal hidden inputs. Test unequal-scale refusal and independent fallback. Source kernel checks do not override its inconclusive model-wide numerical verdict.
- **Done when:** standalone correctness precedes production dispatch witnesses and a matched full-model A/B. Source N128 was slower in tested cells; occupancy/stall explanations remain hypotheses. Cooperative source evidence is bounded and thin-path formal quality remains inconclusive; EXL3 ABI is not Atlas NVFP4 ABI.

### T13 — UMA-safe loading and allocation-tail diagnosis [V/M]

- **Sources:** Spark-GLM53 [cold-load UMA](../../spark-glm53/docs/cold-load-uma.md); Mindreader [tail diagnosis](../../mindreader/docs/research-reports/73-tail-diagnosis.md), [compaction window](../../mindreader/docs/research-reports/75-compaction-verify.md).
- [ ] Audit Atlas mmap/copy/page-cache eviction and memory admission against host available/reclaimable memory and GPU allocations. Profile long-tail requests alongside page faults, allocations, compaction, thermal state, and active competing work.
- [ ] Test cold/full file cache, supported large-page host, repeated load/teardown, discrete-GPU handling, allocation refusal and sustained pressure. Put probes/staging behind existing I/O boundaries.
- **Done when:** a reproduced mechanism supports a bounded loader/allocator remedy and repeat tests. One zero-tail observation does not establish a compaction cure; no global sysctl/cache-dropping policy should follow automatically.

### T14 — Collective and CPU-submission bottleneck attribution [V/M]

- **Sources:** Mindreader [collective provenance](../../mindreader/docs/research-reports/62-drafter-collectives-provenance.md), [root-step profile](../../mindreader/docs/research-reports/60-root-decode-step-profile.md), [reviewed same-boot CPU-placement ABBA](../../mindreader/docs/research-reports/66-plan58-stage0-and-placement-v2.md); Spark-GLM53 [fabric and pinning follow-ups](../../spark-glm53/todo.md).
- **Validated boundary:** report 66 Part B supports its CPU-placement treatment on one boot with manipulation/receipt/swap checks. It does not resolve bundled tuner/remainder-task effects, validate every sampling regime, or supply independent multi-boot replication. Its separate FP8 Stage 0 stopped on worker loss; do not merge that failed campaign into the placement finding.
- [ ] Trace Atlas's actual communication/submission critical path and distinguish link latency from receive starvation, idle waiting, launch gaps, and quant compute. Inventory TP/EP/DFlash collectives by payload and ordering.
- [ ] Only then prototype a bounded communication batching/ordering change or CPU-affinity experiment. Hold topology/HCA count, workload, checkpoint, graph and kernel settings fixed. Test rank agreement, async buffer lifetimes, timeouts, cancellation and shutdown.
- [ ] Compare remaining per-token reductions in [batched MoE](crates/spark-model/src/layers/moe/forward_batched.rs) with existing coalesced [token-major MoE](crates/spark-model/src/layers/moe/forward_token_major.rs). Atlas already supports grouped communication and direct multi-rail two-rank RDMA; weights do not travel over these collectives.
- [ ] Verify CPU-affinity manipulation after sleeping threads wake, including newly created threads and rollback. Mindreader's [failed first placement campaign](../../mindreader/docs/research-reports/65-placement-ab-stage1-stopped.md) demonstrates that a sampled last-CPU field can be stale after an affinity-mask change; a mask update alone is not a runtime-placement witness.
- **Done when:** independent timing/counter evidence supports the bottleneck and a matched end-to-end change. RNR timer and CPU-cluster explanations are leads, not established Atlas causes.

### T15 — Shared tool-prefix reuse without tool generation under `none` [V/U]

- **Sources:** Spark-GLM53 [execution record and plan](../../spark-glm53/docs/tool-choice-none.md), [sampler patch](../../spark-glm53/overlay/patch_tool_choice_none.py); Atlas [request preparation](crates/spark-server/src/api/chat/prepare.rs).
- [ ] Treat this as an optional semantics/design task: Atlas currently removes tools under `none`. If shared-prefix reuse is worth retaining tool declarations, preserve identical template bytes and mask parser-owned opener token sequences in serial and speculative sampling.
- [ ] Test a useful answer independently of absence of structured tool calls; test literal token leakage, multi-token openers, client forbidden-token union, auto/required/named choice, streaming/usage, and OpenAI/Anthropic parity.
- **Done when:** the chosen contract is explicit and native answers plus cache telemetry demonstrate it. Output suppression alone can swallow the answer; source opener enforcement does not prove its separate shared-prefix benchmark succeeded.

## Priority 2: research prototypes with no local Atlas validation

Start these after T01/T02 and a profile showing relevant cost. Each begins
outside production dispatch and remains explicitly disabled until qualified.

### T16 — Draft-only context reduction [U]

- **Source:** Mindreader [report 37](../../mindreader/docs/research-reports/37-todo-draft-only-context.md).
- [ ] Compare full draft context, a supported recent-window/sink policy, and a named compressed draft representation while leaving the target's required context authoritative.
- [ ] Test proposal-law recording, refresh, rejected-prefix repair, context/adapter/tokenizer invalidation, cancellation, target prefix/state equality, long-context retrieval and total physical memory. Charge draft rebuild and lost acceptance.
- **Gate:** actual target distribution and correction remain valid; complete-request cost beats the strongest fixed draft policy and target-only control. Acceptance alone is insufficient.

### T17 — Exact constrained output-head/sampler fusion [U]

- **Source:** Mindreader [report 38](../../mindreader/docs/research-reports/38-todo-output-sampling-fusion.md); Atlas [sampling kernels](crates/spark-model/src/layers/ops/sampling.rs).
- [ ] Begin with a small hard-allowed token set and narrow greedy contract. Extend projection/selection only after proving the current Atlas transform order, tie rules, dtype, penalty history, masks, EOS/min-token handling and logprob stage.
- [ ] Compare with an independent full-head reference; unsupported transforms/raw logprobs trigger explicit fallback. For stochastic speculation preserve full required target/proposal support and residual law, including bonus positions.
- **Gate:** token/distribution/API contract equality precedes timing. Static vocabulary pruning, altered RNG mapping, and lower-precision heads are separate approximation experiments.

### T18 — Stochastic joint block verifier [U]

- **Source:** Mindreader [corrected report 36](../../mindreader/docs/research-reports/36-todo-joint-block-verification.md).
- [ ] Build an exhaustive tiny-vocabulary CPU probability-law reference first, with recursively clipped prefix coefficients, correctly oriented residual, full-block special case, continued sub-block consideration, and target bonus token.
- [ ] Test zero-probability branches, ties, all acceptance/rejection positions and output law; then integrate bounded GPU state/rollback verification under equal draft/work/memory budgets.
- **Gate:** exact law and complete state restoration precede full-request A/B. Atlas's joint multi-owner FFN traversal is not this stochastic algorithm, and a greedy toy matcher is not its proof.

### T19 — Recurrent checkpoint placement and bounded replay [U]

- **Source:** Mindreader [report 41](../../mindreader/docs/research-reports/41-todo-recurrent-checkpoint-placement.md); Atlas [snapshot tiers](crates/spark-model/src/model/ssm_tier.rs).
- [ ] Separately compare prefix checkpoint placement and generation rollback strategies at equal total memory. Use existing complete snapshots as the reference; account for recurrence inputs, convolution history, attention KV, metadata and replay scratch.
- [ ] Test forks, partial prefix matches, changed checkpoint positions, rejected drafts, cancellation, eviction/fault-in and restored full state/output; learn overlap depths only from past requests.
- **Gate:** exact suffix reconstruction and lower total cost at the same memory ceiling. Recurrent checkpoints do not replace attention KV.

### T20 — KV/session offload compression after ownership filtering [V/U]

- **Sources:** Mindreader [report 89](../../mindreader/docs/research-reports/89-kv-cache-compressibility-screen.md), [cache research](../../mindreader/docs/research-reports/03-cache-kv-engram.md), [codec constraints](../../mindreader/docs/research-reports/87-nvcomp-weight-decompression.md).
- [ ] After T04, screen lossless codecs for Atlas snapshot/offload families separately. Prefer cold session restore/offload over a hot attention codec until decode costs are measured.
- [ ] Filter live ancestry, recycled blocks, sliding windows and padding before computing useful-content ratios. Test byte-exact roundtrip, allocation metadata, corruption/refusal, cancellation, staging memory, capture support and ARM64 installation for any chosen GPU library.
- **Gate:** restore/capacity benefit exceeds encode/decode/copy/scratch cost on Atlas data. Source host compression is not a GPU decoder or serving result; stale source summaries include recycled drafter pages.

### T21 — Decision-time adaptive watcher [M/U]

- **Source:** Mindreader [corrected X0–X2 report](../../mindreader/docs/research-reports/83-mindreader-x0-x2.md), [watcher code](../../mindreader/src/mindreader/).
- [ ] Collect real round records via T02 over independent boots. Use chronological train/calibration/test splits, features available before the action, decision-time constant/history controls and genuinely different actions.
- [ ] Test future-data leakage, marginal/conditional confusion, missing data, policy epochs and observation cost. Keep a do-nothing fallback and offline/shadow mode first.
- **Gate:** prospectively useful control after charging measurement/interference costs. X1 was retrospective and future-informed; its internal feature point estimate beat history, contrary to stale summaries. X2 never compared distinct proposal actions.

### T22 — Training-free parallel proposals or draft overlap [U]

- **Sources:** Mindreader [parallel-proposal design](../../mindreader/docs/research-reports/39-todo-training-free-parallel-proposals.md), [overlap research](../../mindreader/docs/research-reports/06-overlap-cross-tokenizer.md).
- [ ] Proceed only if serial drafting is exposed after current MTP/DFlash baselines. Prototype bounded lookahead/n-gram or overlap with explicit proposal ancestry, masks, positions and request-owned candidate state.
- [ ] Test full target conditionals, branch dedup/selection, hybrid-state rollback, stale hidden inputs, common UMA bandwidth contention, padding and cancellation. Compare with native MTP/block drafting, not just serial target-only.
- **Gate:** valid proposal/verification law and measured total latency after wrong-branch work. Parallel guessed tokens cannot automatically be treated as autoregressive conditionals.

## Additional mechanisms identified in the follow-up audit

These Priority 1 candidates were absent or insufficiently explicit in the first
version. They complement T04–T08 rather than imply that Atlas has the same bugs.

### T23 — Multimodal history semantics and aggregate encoder admission [V]

- **Source:** Spark-GLM53 [images-per-request experiments](../../spark-glm53/docs/images-per-request.md).
- **Validated boundary:** source reproduces rejection when a stateless client resends history beyond the configured image count. It records completed larger single-request input and a proxy test combining explicit history trimming with an aggregate in-flight image guard. Those observations are not an arbitrary safe concurrency envelope or proof that raising a count reserves memory.
- [ ] Audit Atlas image/video limits and memory admission across full message history. Define an explicit request limit and aggregate encoder-work budget derived from image pixels/patches, video frames, transient buffers and model shape; count alone is a coarse proxy. Reuse Atlas's existing pixel-budget config and vision harness.
- [ ] Test repeated full-history turns including later text-only turns, distinct and reused images, large-resolution inputs, concurrent requests, cancellation/error release and recovery. A retained prefix hit must not bypass admission validation.
- [ ] Keep history trimming an explicit client/gateway policy with a meaningful placeholder; silently discarding images inside Atlas changes the request. Test OpenAI/Anthropic history semantics and useful visual answers separately from admission success.
- **Done when:** bounded aggregate work and cancellation-safe permits prevent one valid request from overcommitting concurrent encoder memory, with native multi-image quality checks and measured headroom.

### T24 — Retire every superseded recurrent state at its real checkpoint boundary [V/M]

- **Sources:** Spark-GLM53 [follow-up qualification](../../spark-glm53/CHANGELOG.md), [state-lifetime and group-alignment contract](../../spark-glm53/README.md), [retirement fix](../../spark-glm53/overlay/patch_mamba_align_state_free.py), [chunk-alignment fix](../../spark-glm53/overlay/patch_mamba_align_chunking.py).
- **Validated boundary:** source finds a single pending-state index overwritten during async multi-chunk prefill, retaining older states for the request lifetime. It also identifies recurrent state labelled at the wrong boundary when alignment follows a smaller draft page. A composed workload/reference/rollover qualification passes; it does not isolate each fix's performance contribution.
- [ ] Audit Atlas snapshot leases and retirement queues against several in-flight chunks. Track all unreleased predecessors, retire only after committed progress passes the correct boundary, and reserve the actual running/speculative/in-flight peak.
- [ ] Test multiple predecessors superseded before completion, late/out-of-order completions, cancellation, request reuse, forks and leak-free teardown. With heterogeneous attention/recurrent/draft page sizes, assert checkpoint label and actual complete state refer to the same prefix; test chunk tails before/at/after each boundary and progress under sub-block token budgets.
- **Done when:** a long async prefill maintains bounded retained-state count, every release is ownership-safe, and resumed output/full recurrence matches an independent scalar reference. Existing Atlas snapshots are the baseline.

### T25 — Compact draft page IDs and exact draft-prefix boundary lookup [V/M]

- **Sources:** Spark-GLM53 [compact-page contract and follow-up](../../spark-glm53/README.md), [qualification/limitations](../../spark-glm53/CHANGELOG.md).
- **Validated boundary:** source TP2 OFF/ON/OFF and follow-up reference/rollover tests establish a bounded page-ID reservation/reuse tradeoff. Some prose and cold-load cells regress. TP3 confirms geometry only; tensor-level parity and TP4 are unqualified. This reduces block-ID demand, not backing tensor bytes or quantization precision.
- [ ] First determine whether Atlas target/draft allocation shares a comparable page-ID constraint. If so, derive a compatible draft page divisor and typed per-group reservation/replay plan; require complete-window and ancestry checks.
- [ ] For a DFlash cache whose per-position projection depends only on the target hidden state at that position, prove whether exact boundary lookup can omit a future lookahead page. Do not apply that rule to EAGLE or MTP caches with different dependencies.
- [ ] Test exact-fit/padded geometry, short repeated tails, branch replay, window expiry, rollover, non-DFlash refusal, pressure, admission and graph metadata. Compare reuse latency, allocated bytes and reserved IDs separately at identical KV budget.
- **Done when:** target/draft state parity and complete-request tradeoffs pass on Atlas. Do not replace conservative shared-KV commitment accounting merely because the source removed a draft-only lookahead.

### T26 — Qualify the existing FlashKDA recurrence adapter at model level [V/M]

- **Sources:** Spark-GLM53 [TP3 components and combined-profile limits](../../spark-glm53/docs/tp3-throughput-results.md); Atlas [existing operator gates](research/flash-kda-prefill/README.md) and [production adapter](crates/spark-model/src/layers/glm5_kda/flash_prefill.rs).
- **Validated boundary:** source evaluates a bundle containing FlashKDA, cooperative MoE, precision scope, graph coverage and admission changes; its serving gain cannot be attributed to FlashKDA alone. Atlas already records bounded independent recurrence/packing/resume gates and smoke evidence, while model-level qualification remains pending. This is recurrence compute, not T10's projection GEMM.
- [ ] Run an isolated same-image OFF/ON model qualification within the adapter's declared shape/state/scratch eligibility before extending it. Test full FP32 recurrent matrices, convolution history, raw versus normalized Q/K, asymmetric nonzero state, split/resume, resumed native recurrence and both-rank state-slot identity.
- [ ] Preserve prefill-only and graph/MTP exclusions unless separately proved. Validate scratch overlap/retirement and ABI/library identity; a partially mutating launch failure must abort rather than retry the chunk.
- **Done when:** native long-prefill/mixed-load quality, retrieval, tool calls and state parity pass, followed by matched preparation-inclusive whole-request measurements. A combined source-profile result or a standalone operator result does not complete this gate.

## Screened negatives and work to defer

| Idea | Disposition and reopening condition |
| --- | --- |
| Extra generic lossless compression of EXL3 trellis weights | [Report 88](../../mindreader/docs/research-reports/88-exl3-trellis-compressibility-screen.md) found no worthwhile result for tested host codecs/transforms on sampled trellis. Small float remainder differs. Reopen only with a different supported representation/decoder and cost screen; this is not an Atlas NVFP4 entropy result. |
| Materialize all compressed weights before compute | [Report 87](../../mindreader/docs/research-reports/87-nvcomp-weight-decompression.md) models added scratch traffic; its roofline is conditional, not an impossibility proof for every fused codec. Verify GB10 support, device-decoder versus host API, installation/capture, and end-to-end traffic before any implementation. |
| Smaller MoE tile necessarily raises occupancy | Report 80 disproves that inference for its compiled comparison: resource reservations did not improve. Reopen with compiled resource changes and a new controlled benchmark. |
| Generic BLAS/kernel swaps or contiguous copies automatically improve tiny-row GEMM | Mindreader [report 68](../../mindreader/docs/research-reports/68-gemm-microbench.md) did not clear the tested generic swaps; sampled strided inputs did not show the assumed penalty. Its packed path also required an M=1 numerical exception/fallback. Reopen with Atlas-specific shape/precision evidence, not an unconditional copy or backend switch. |
| W8A8 KDA activation QDQ as a transparent speedup | Source reproduces probability shifts; keep as a negative numerical-control arm. A quality-changing quantization project needs its own acceptance contract. |
| Zero retention for every hybrid group | Source APC matrix demonstrates reuse regressions. Require workload-aware retention plus replay qualification, not a blanket default. |
| Copy larger prefill ceilings, utilization fractions, HCA/CPU settings, or kernel crossover constants | Backend/checkpoint/topology costs differ. Derive legal geometry and reserve first; measure complete latency/fairness/capacity on Atlas. |
| Disable compaction, claim a universal tail cure | Only association and a bounded zero-event observation survive the source corrections. Require a reproduced mechanism and pressure/repeat qualification. |
| Fresh SHA pins prove an allocator/ABI port | Spark-GLM53's display-KV notes report pin mismatches. Re-derive layout, ownership and driver compatibility; changing a pin is not a safety proof. |
| JIT-cache persistence/skip graph profiling as an immediate Atlas port | Audit actual Atlas startup consumers first. Eliminate only unused work and account for real graph memory; vLLM Python/JIT costs need not exist in Atlas. |
| Broad alternate runtime replacement | [Report 40](../../mindreader/docs/research-reports/40-todo-serialization-alternative-runtime.md) is research only. First locate a measured CPU/I/O boundary; make a bounded serialization/lifetime improvement before a separate runtime project. |
| External kernel repo lists, training/distillation, shared-weight drafting or expert prefetch | [Reports 84–86](../../mindreader/docs/research-reports/84-external-cuda-kernel-repo-triage.md) and broad research are hypothesis inventories. Architecture/license/precision compatibility and a cheap cost/oracle screen precede a dedicated proposal. |

## Implementation order and common promotion gate

1. Start T00 EXL3 support as the highest-priority implementation, with T01/T02 gates alongside it; check T03 readiness/progress coverage before native serving qualification.
2. Use T04 to select cache tasks T05–T07 and T24/T25; run T08/T09 engagement and mixed-load controls. Add T23 when qualifying vision concurrency.
3. Profile before choosing T10–T14 kernel/loader/communication work. T15 is an independent API/prefix decision.
4. Open T16–T22 only when the relevant cost or capacity limit is observed. T26 qualifies an existing adapter before any further recurrence port.

Keep each PR bounded, with failing boundary/oracle tests before production
changes. T00 spans separate contract/loader, baseline kernel, optimized kernel,
and serving-qualification PRs. Derive config/accounting from the existing authoritative
model/cache plans and `MODEL.toml`; use explicit capability rejection and the
existing I/O abstraction. Keep Rust modules within the repository's size cap
and record shared trait/build/CI changes prominently in the PR.

For promotion, require an independent numerical/state oracle, actual dispatch
and both-rank engagement evidence, memory-safety/pressure tests, and native
useful answers, real tool calls, retrieval, cancellation/recovery and clean
shutdown. Exercise changing graph metadata and ragged drains where relevant.
Run at least three measured repetitions for headline results with matched
fingerprints and a fresh-process/order control; separate full-wall throughput,
decode-window rate, TTFT and per-stream gaps. Hold checkpoint, quantization,
prompt class, thinking, sampler, KV budget and topology fixed.

Runtime PRs require a fresh Docker rebuild and the relevant validation-suite
slice, followed by supported-model regression coverage proportional to the
shared paths changed. CPU checks do not qualify a numeric kernel. Preserve
failed/inconclusive evidence and explicit rollback; enable no experimental
path by default solely on source-reported gains.
