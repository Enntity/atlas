# Phase 2: dual-Spark aggregate-decode comparisons and feasibility

Research date: 2026-09-06. Read-only source/receipt review; no competing engine
was installed or run. This is not an Atlas benchmark or a promise of gains.
The Atlas graph-safe C2/C3 implementation is separate work in progress.

## Normalize the metric before choosing a target

Use three separate metrics:

- **Aggregate end-to-end:** all completion tokens / batch wall time, including
  admission, prefill and first-token latency.
- **Aggregate post-first-token:** total completion tokens minus one per stream,
  divided by last completion minus earliest first token. It still includes
  later streams' prefill/admission interference; it is not steady full-width
  GPU throughput.
- **Per-stream decode:** each stream's remaining tokens / its own decode
  window. Summing these rates does not generally equal aggregate throughput.

Record offered client concurrency, server running/waiting counts, and actual
GPU target/verifier width separately. A configured million-token context is
not a million-token benchmark prompt. All results below use two Sparks and TP2,
unless explicitly excluded; none establishes equivalence to Atlas TP2+EP2.

### Most useful published receipts

| Lane | Offered C; prompt/output | Published result | Interpretation |
|---|---|---|---|
| PixelML Apollo, LibertAIDAI NVFP4 + native MTP4 | C7; short Python validator / forced 256 per stream | 82.12 aggregate e2e; 14.01 mean-stream decode | Median of three warm runs; FP8 E4M3 KV; eager vLLM, TP2/Ray, Marlin MoE. C4 was 58.36. Same checkpoint repository, pinned revision below. |
| Same Apollo recipe, fresh revalidation | C6; short coding / forced 256 | 67.55 aggregate e2e; 14.80 mean-stream decode | Median of three; C4 56.00; C7/C8 queued. Keep this lower repeat beside 82.12. |
| Apollo EXL3/TR3 4-bpw + DFlash2 K7 | C4; short salted prompts / max 400 | 154.86 counting, 88.96 code-only, 47.03 planning aggregate e2e | Three rounds; corresponding mean-stream decode 41.66/26.84/16.67; acceptance 95.8%/70.0%/38.3%. Different target quantization; FP8 KV; graphs. |
| loud1990 NVFP4 + MXFP8 DFlash2 K7 | C4; **27-token counting** / 400 | 194.15 aggregate post-first-token; 49.10 median per-request decode | Five batches, 97.49% median acceptance. Recalculated e2e median **183.68**, not 194.15. Different NVFP4 publisher; FP8 KV; draft TP1. |
| Beastllama SGLang NVFP4 + DFlash2 | C8/C12 offered; distinct short code/infra / max 400 | Current LC4: 77.4/83.2 aggregate | Eight configured running slots; C12 does not establish twelve simultaneous GPU rows. FP8 KV, BF16 recurrent state, D=5 including bonus, chunk4096. |
| SparkGLM posted long-C4 | C4 staggered 0/1/2/3s; actual 15,807–15,810 / 400 | 24.267 aggregate post-first-token; **18.573 e2e** | 1600 tokens / 86.148650 seconds. EXL3+DFlash2 K7, FP8 KV, graphs, chunk7168. Not comparable to short counting. |

Sources and limitations for these rows:

- Apollo MTP [original receipt](https://github.com/PixelML/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark/blob/main/results/APOLLO-2026-08-27.md)
  and [fresh revalidation](https://github.com/PixelML/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark/blob/main/results/APOLLO-2026-08-27-REVALIDATION.md).
  Target revision `11d73216cd636238e82e1d77fe1042ffab36e7fa`;
  vLLM `0.1.dev20051+g487ecf187`, recipe `aed98a1`. Temperature0,
  low reasoning, ignore-EOS. Exact rendered short-prompt token count is not
  given in these aggregate tables. The fresh run records zero prefix hits.
- Apollo EXL3 [receipt](https://github.com/PixelML/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark/blob/main/results/APOLLO-2026-08-28-EXL3-DFLASH2.md)
  and [planning raw result](https://github.com/PixelML/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark/blob/main/results/raw/APOLLO-2026-08-28-EXL3/planning-concurrency.json).
  Temperature0, thinking off, unique prefixes. The headline prose says decode
  excludes TTFT, but its **aggregate** column is e2e: the
  [harness](https://github.com/PixelML/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark/blob/main/exl3/tests/bench_concurrency.py)
  divides all tokens by the sum of complete wave walls. Do not multiply its
  mean-stream rate by C and relabel that aggregate. Exact rendered prompt
  counts are not retained in this summary JSON.
- loud1990 [five-run table](https://github.com/loud1990/GLM-5.3-Flash-NVFP4-MXFP8-2x-DGX-Sparks/blob/main/benchmarks/all-benchmark-results-20260828.md)
  and [per-request JSON](https://github.com/loud1990/GLM-5.3-Flash-NVFP4-MXFP8-2x-DGX-Sparks/blob/main/benchmarks/glm53-nvfp4-mxfp8-dflash2-20260828.json).
  Recomputed from `completion_tokens / batch_wall_s`, C4 e2e runs are
  186.989, 179.768, 188.396, 179.916, 183.684. The published aggregate exactly
  matches `(tokens-C)/(max(call_start+wall)-min(call_start+ttft))`.
  This is the highest repeated dual-Spark rate found in this review, not a
  general coding result. The old EXL3 served-model ID is a compatibility alias,
  not the target's actual quantization.
- SGLang [current recipe](https://github.com/beastllama/GLM-5.3-Flash-DFlash2-SGLang-2x-DGX-Spark/blob/main/README.md)
  and [chronological ladder](https://github.com/beastllama/GLM-5.3-Flash-DFlash2-SGLang-2x-DGX-Spark/blob/main/LADDER.md).
  The older claim of a 55 tok/s hardware ceiling was explicitly retracted:
  recurrent-state allocation had silently capped speculation at two streams.
  Current short-prompt rates are relevant fleet evidence, but per-request raw
  concurrency receipts/harness are not in the inspected tree. Do not assign
  them the same reproducibility strength as the retained JSON above.
- SparkGLM [raw video receipt](https://github.com/Enntity/sparkglm/blob/main/results/legacy/2026-09-03-current-best-posted-video/current-best-posted-video.json)
  has the actual prompts, completion counts and timestamps. The 24.267 value
  is `(1600-4)/(86.148650-20.380570)`, whereas e2e is `1600/86.148650`.
  Its [ten-repeat restoration check](https://github.com/Enntity/sparkglm/blob/main/results/candidates/2026-09-04-video-runtime-isolation/TEN_REP_RESULT.md)
  reports restored-image median23.701 post-first-token, median88.629s wall;
  no full quality qualification or new optimization is claimed.

Exclude single-Spark 2.05-bpw EXL3 headlines, four-Spark results, the 743B
non-Flash model, and counting-only figures from any claim about this deployment's
general coding speed. Do not disable host safeguards to copy memory-tight
competitor launch settings.

## What the recipes actually change

Apollo's native-MTP lane uses weight-only Marlin NVFP4 routed experts and
FlashInfer CUTLASS dense projections, patched SM90 NoPE sparse MLA/FA2, and FP8
KV. Eager execution means CUDA graphs are disabled, **not** that requests must
execute serially. Its [launcher](https://github.com/PixelML/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark/blob/main/start.sh)
passes four speculative tokens and eight sequence slots with no explicit
one-sequence MTP cutoff. Running/waiting telemetry confirms admission pressure,
but these receipts do not expose target/verifier GPU width. Therefore do not
claim C7 means a 35-row verifier or assert a serial-MTP fallback without a
pinned runtime trace. Atlas needs its own target-width tracing.

EXL3 changes both the weight representation and expert execution: packed
trellis/MCG experts, TP-sharded gate/up/down, and fused `exl3_moe`; DFlash2
proposes a block instead of Atlas's serial native-MTP drafts. SparkGLM adds
grouped prefill and cooperative decode with device work sharing, plus mixed
prefill/decode scheduling. These are not drop-in NVFP4 kernel substitutions.
Its [runtime mapping](https://github.com/Enntity/sparkglm/blob/main/docs/PUBLISHED_VIDEO_CONFIGURATION.md)
also pins the FP16 sparse indexer, native top-k/DeepGEMM foundation, TP2 drafter,
and right-sized indexer workspace.

loud1990's [patch ledger](https://github.com/loud1990/GLM-5.3-Flash-NVFP4-MXFP8-2x-DGX-Sparks/blob/main/PATCHES.md)
names `local-inference-lab/GLM-5.3-Flash-NVFP4`, MXFP8 selector/convolution/context
projections and draft TP1. Draft narrowing produced only +5.64% at C4 and
slightly slower C1–C3 in its matched counting test. Thus most of the gap to
Atlas is not explained by MXFP8 alone. Humming's alternative indexed NVFP4
expert path reached193.15 on the same protocol, not a clear gain over194.15.

FP8 KV and BF16 recurrent state need their own numerical and capacity
validation in Atlas; neither is a supported performance flag for this GLM
path today. EXL3 and DFlash2 also have distinct published license restrictions;
review their actual artifacts before any weight/runtime adoption rather than
treating a code recipe's license as permission for its checkpoints.

## Local feasibility: C4 first, C7/C8 later

Paths below are relative to the Atlas repository. They describe the reviewed
baseline; graph work may subsequently move functions without lifting width
restrictions.

| Boundary | Current condition | Required work before increasing width |
|---|---|---|
| Launcher/server | `scripts/start-glm53-ep2.sh` and `spark-server/.../serve_phases/preflight.rs`: active1..3, admitted active..5; long sparse only C2/C3 | Add a separate explicit validated width opt-in only after model/kernel tests. |
| EP wire protocol | `spark-model/src/model/impl_a2.rs::ep_broadcast_decode_batch_dispatch` and `ep_worker_decode_batch`: dynamic N, slot IDs, tokens | No inherent C3 wire ceiling found. Preserve exact row order and collective counts; test noncontiguous/permuted slots and drains on both ranks. |
| KDA | `layers/glm5_kda.rs::decode_multi_seq` and `glm5_kda/multi_seq.rs` accept only2/3 | Widen the true batched mHC path, not the fallback: fallback calls scalar decode with shared mHC base pointers and is not a validated C4 path. |
| KDA projections | `glm5_kda/projection.rs::project_hot_multi_decode` already has exact-M NVFP4 kernels2..8; BF16 side projection is batch-M | C4 can reuse existing projection kernels after checking handles and arenas. Recurrence remains per-state-row; do not use sequential-token verifier recurrence as independent rows. |
| MLA | `qwen3_attention/.../multi_seq/mla_glm.rs`: projection dispatch2..5, absorption/extraction optimized2/3/5 | C4 projections exist but absorption/extraction fall back row-wise; C7/C8 need wider dispatch/kernel support or an explicitly checked fallback. Sparse validation currently rejects all but2/3. |
| MoE/mHC | KDA FFN optimized only n3; MLA FFN specialized n3/5; `moe/prequant_fp4.rs::c3_grouped_shape` requires3 | Port the measured compact grouped pipeline to exact C4 with row-count-aware shared projections/worklists, then validate. Merely raising guards would lose the new C3 benefit. C7 is another dispatch tier, not a config tweak. |
| Graphs | `model/trait_impl/decode_graph_key.rs` ordered SSM-slot vector; exact EP width | Preserve slot-vector identity; no wider-graph borrowing for distributed drains. Include every actual width in tests; missing pool slots may force eager execution. |

### Memory and scratch, not just KV capacity

TP2 local KDA geometry is32 heads x128x128 FP32 state, with four-tap
convolution and34 layers. From `Glm5KdaLayer::new`, `config/methods.rs`, and
`model/ssm_pool.rs`, each live sequence needs68MiB recurrent H plus6.375MiB
convolution = **74.375MiB per rank**, before snapshots. The pool also allocates
a dummy slot. `model/impl_a1.rs` sizes the resident pool by active batch cap;
additional admitted requests and slot availability need an allocation audit.

Ordinary decode's watchdog rollback ring is eight full states per active slot
when enabled (`ssm_reserve::decode_rollback_ring_slots` and
`atlas-kernels::DECODE_ROLLBACK_RING_SLOTS`). Base plus ring is approximately
**669.375MiB per extra active slot**: C3->C4 about0.654GiB/rank;
C3->C7 about2.615GiB/rank; C3->C8 about3.269GiB/rank, before other reserve
changes. Do not disable the watchdog or ring to manufacture a larger limit.
Native MTP instead adds per-position verification snapshots/checkpoints;
use `ssm_pool_reserve_bytes` rather than multiplying only live-state bytes.

Main BF16 MLA K+V alone costs22KiB per active token per rank. At16K context,
one additional full request costs352MiB, plus semantic-index tail/pool and
allocator metadata. Thus C7 at16K is a materially larger safety envelope than
short C7. FP8 would not remove recurrent-state or rollback costs.

`BufferSizes::from_config` uses max(prefill budget, speculative staging,
batch width), and decode metadata has a floor of32 rows. Existing stage512
may already cover C4/C8 activation volumes, but check mHC highway, KDA QKV
planes, absorbed Q, expert permutations, compact worklist and output arenas
against the proposed **actual** row count. The new sparse chain deliberately
reuses selector scratch sequentially; no reason to allocate C copies of full
history scores just to widen independent concurrency.

### Concurrent speculation is a different project

Server preflight, launcher, `model/impl_a2.rs`, and
`model/trait_impl/speculative.rs` currently restrict distributed GLM MTP to
active C1; verifier semantic indexing remains unavailable beyond2048. The
independent sparse flag explicitly rejects speculative execution. It cannot
be reused as proof that C4 x K5 verification is correct. A batched proposer
needs per-request predictor state, accepted-position ownership, rollback,
draft KV, and matched distributed proposal/target ordering. Long-context MTP
additionally needs selector state for each verifier token, not just one token
per independent row. DFlash2 integration is larger still and adds a separate
checkpoint/license decision.

## Recommended sequence after the graph work

1. Measure eager versus graph C2/C3 with actual batch/slot traces and equal
   workloads. Report full-wall and common post-first-token throughput alongside
   per-stream rates; preserve unequal-history and drain correctness gates.
2. Highest likely incremental ROI: **guarded non-speculative C4**, using
   existing KDA M4 projections and extending the measured C3 grouped MoE path.
   Start at short context with unchanged precision and rollback safeguards;
   widen sparse history only after row/state isolation passes. This is a
   hypothesis to test, not an extrapolated throughput promise.
3. Only after memory accounting and C4/C5 drain coverage, consider C7/C8.
   Measure true GPU width and admission separately; published fleet receipts
   show queueing can masquerade as a bandwidth plateau.
4. Separately evaluate concurrent native MTP and then DFlash2 feasibility.
   Counting ceilings around194tok/s justify investigation of block drafting,
   but planning/code receipts around47/89tok/s are the more useful workload
   targets. Do not expect graphs alone to close that architecture gap.

For an attributable comparison, retain the current checkpoint, BF16 KV,
reasoning mode, generated length, prompt bytes and safety caps first. Add
short-code, planning and counting cases as separately named workloads, then
a mixed long-prefill interference case. This document changes no runtime
configuration or code.
