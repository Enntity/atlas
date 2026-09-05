# GLM-5.3-Flash-NVFP4 on two DGX Sparks

Atlas support targets the exact `LibertAIDAI/GLM-5.3-Flash-NVFP4` checkpoint and
uses expert parallelism (`EP=2`). Set `TP_SIZE=2` to overlap tensor parallelism
on the same two physical ranks: routed experts remain EP-sharded while MLA and
KDA heads are split between the Sparks and their row projections are summed
once per mixer. `TP_SIZE=1` preserves the original pure-EP fallback.

## Safe first launch

Build the same image on both aarch64 Sparks:

```bash
docker build \
  -f docker/gb10/glm-5.3-flash-nvfp4/nvfp4/Dockerfile \
  -t atlas-glm53-flash:latest .
```

Then launch from the head node. If the checkpoint is already on both machines,
use the same absolute path for `MODEL`:

```bash
HEAD_IP=169.254.179.82 \
WORKER_IP=169.254.128.113 \
SSH_TARGET=mangokid@192.168.8.187 \
MODEL=/var/tmp/models/glm53-flash-nvfp4 \
TP_SIZE=2 \
FP4_PREFILL=1 \
./scripts/start-glm53-ep2.sh
```

The launcher intentionally starts with:

- 1,024 maximum sequence tokens;
- a separately configurable 1,024-token prefill chunk budget;
- one batch item and one concurrent sequence;
- BF16 KV cache;
- a 92% total GPU-memory budget and 4 GiB OOM guard;
- a 114 GiB container memory ceiling, leaving host headroom if loading runs away;
- no speculative decoding or prefix cache;
- cuBLASLt BF16 projection dispatch for checkpoint-native MLA matrices;
- the worker rank first and no automatic restart policy.

The KDA register-resident prefill path, KDA multi-sequence decode, width-three
batched FFN, guarded GLM MLA projection batching, unified MoE layout, and
cuBLASLt projection dispatch are enabled by default. Their diagnostic fallbacks
are `KDA_REGRESIDENT_PREFILL=0`, `KDA_MULTI_SEQ=0`, `KDA_BATCHED_FFN=0`,
`MLA_MULTI_SEQ=0`, `UNIFIED_MOE_LAYOUT=0`, and `CUBLAS_GEMM=0`.
KDA Q/K/V and row-parallel output projections keep independent decode-native
NVFP4 weights plus transposed M=128 prefill twins. The output twins add about
0.3 GiB per rank across all 34 KDA layers without changing decode numerics.

`FP4_PREFILL=1` enables Atlas's native W4A4 tensor-core path for GLM's first
three dense FFN layers. It quantizes each BF16 activation to NVFP4 once for the
gate/up pair and once for the down projection, then uses the M-fast schedule so
CTAs sharing a weight panel reuse it from L2. This is opt-in because activation
quantization is lossy. Hybrid dense/MoE models allocate about 78 MiB of shared
scratch per rank at the guarded 1,280-token test size; pure MoE models still
allocate none.

This checkpoint declares a 1,048,576-token model context. Atlas uses causally
masked dense MLA through `index_topk=2048`, then the checkpoint's 32-head,
128-wide semantic index to select 2,048 raw history tokens from four-token
pools. Pooled keys and unfinished raw key/gate groups share the main KV cache's
physical block IDs, so chunk boundaries, recycled blocks, and a later
multi-sequence implementation have one ownership model. The launcher rejects
only values above the checkpoint limit.

`MAX_PREFILL_TOKENS` is independent of `MAX_SEQ_LEN` and defaults to the smaller
of 2,048 and `MAX_SEQ_LEN`.
Long prompts stream through that reusable activation arena while paged
persistent state grows across the request. Atlas accepts arbitrary scheduler
chunk boundaries: raw index inputs are staged by physical token offset, and a
separate stream-ordered kernel finalizes every completed four-token pool.

## Guarded 100K launch

The first long-context milestone keeps BF16 KV and admits one sequence:

```bash
MAX_SEQ_LEN=100000 \
MAX_PREFILL_TOKENS=1024 \
MAX_BATCH_SIZE=1 \
MAX_NUM_SEQS=1 \
./scripts/start-glm53-ep2.sh
```

On the measured TP2+EP2 pair at a 90% GPU budget and 4 GiB OOM guard, startup
left 9.1/9.8 GiB available on the two hosts and allocated capacity for 275,696
KV tokens per rank. A warmed 10,000-token prompt completed coherently at 251.6
prefill tok/s and 9.47 decode tok/s. An independent 10,000-token prompt with an
early `QUARTZ-9051` needle recovered that exact value. These are correctness
receipts for the scalar sparse baseline, not the final performance target.

The first sparse-attention optimization scores eight selected tokens per CTA
tile while retaining the same online-softmax recurrence. With the same warmed
requests it raised 3K prefill from 303.9 to 362.0 tok/s and 10K prefill from
251.6 to 285.7 tok/s; 10K decode rose from 9.47 to 10.04 tok/s. The 10K early
needle still returned `QUARTZ-9051` exactly.

GPU-synchronized stage profiling at 10K then attributed 13.355 seconds to the
one-head sparse-MLA kernel, versus 1.077 seconds for semantic-index logits and
0.011 seconds for exact top-k. At 32K the same stages took 48.937, 11.322, and
0.118 seconds respectively. The follow-up kernel groups eight query heads per
CTA, reusing each head-shared compressed K/V row while retaining the original
one-head kernel as `GLM_SPARSE_HEAD_GROUP=1`. Sparse-attention time at 10K fell
to 4.801 seconds (2.78x faster). With profiling disabled, warmed exact-10K
requests measured 379.2--383.4 prefill tok/s and 10.14--10.16 decode tok/s,
up 32.7% in prefill from the 285.7 tok/s tiled baseline. Decode continues to
use the latency-favorable one-head kernel. The exact-10K early needle receipt
from `scripts/benchmark_glm53_niah.py` recovered `NEBULA-2847` verbatim.

With sparse attention no longer dominating 10K, whole-model profiling showed
the routed/shared MoE FFNs consuming about 0.64 seconds of a 1.19-second 1K
model pass. Enabling Atlas's existing prequantized NVFP4 activation path and
fused SiLU-quant kernel raised warmed 1K prefill from the previous 942.9 tok/s
best to 963.3 tok/s, and raised 10K from 379.2 to 392.7 tok/s. Decode remained
10.18 tok/s on the 10K workload. These two levers are now default-on in the GLM
launcher; `NVFP4_PREQUANT_MOE=0 NVFP4_FUSED_SILU_QUANT=0` restores the BF16
activation path. An exact-10K early-needle run still recovered `NEBULA-2847`.

The long-context semantic scorer now tiles eight query rows by eight pooled
keys per CTA. The GB10 kernel stages each pooled BF16 key once in shared memory
and reuses each query value across all eight scores; decode retains the original
one-row/eight-pool kernel. In a same-image, profiler-enabled A/B, semantic-logit
time fell from 1.093 to 0.793 seconds at 10K (27%) and from 11.293 to 7.766
seconds at 32K (31%). Exact-32K end-to-end prefill rose from 341.9 to 358.5
tok/s (+4.9%), while 10K rose from 394.0 to 401.4 tok/s (+1.9%). Set
`GLM_INDEX_ROW_GROUP=1` to restore the original scorer. With profiling disabled,
the exact-10K needle receipt again recovered `NEBULA-2847`, at 399.0 prefill
tok/s and 10.16 decode tok/s.

A subsequent isolated chunk-size A/B kept the 32K sequence cap, one admitted
sequence, the 90% memory budget, and the 4 GiB guard fixed. Raising only the
prefill chunk from 1,024 to 2,048 increased warmed exact-10K prefill from
398--400 to 445--447 tok/s (about 11.8%). At exact 32K it increased prefill
from 354.5 to 375.8 tok/s (6.0%). The 2,048-token run left 8.2/10 GiB host
memory available, and its exact-10K early-needle receipt recovered
`NEBULA-2847`. The launcher therefore defaults to 2,048 when the configured
sequence limit permits it, while the 1,024-token safe-first-launch remains
unchanged because the default chunk is capped to `MAX_SEQ_LEN`.

Re-testing chunk geometry after the native NVFP4 prequant path was established
showed that 2,048 was no longer the throughput optimum. On the same image and
exact-10K prompt, raising only the chunk budget from 2,048 to 4,096 increased
warmed prefill from 447.3 to 534.4 tok/s (+19.5%). Exact-32K prefill was 397.9
tok/s versus 375.8 tok/s at 2,048 (+5.9%). The 4,096-token run retained more
than the 4 GiB memory guard, produced coherent output, and recovered the exact
10K `NEBULA-2847` needle at 535.4 prefill tok/s.

A guarded follow-up at 6,144 tokens reduced exact-10K to two prefill passes and
reached 663.1 tok/s on the M=64 NVFP4 expert kernel, another 24.1% over 4,096.
The exact-32K receipt reached 415.0 tok/s (+4.3% over 4,096), and the exact-10K
needle recovered `NEBULA-2847` at 662.5 tok/s. Both ranks retained 9--10 GiB
available after the run. The adaptive launcher default therefore caps at
6,144; safe-first launches with `MAX_SEQ_LEN=1024` remain at 1,024
automatically.

BF16 is deliberate for the first 100K correctness gate. FP8 halves the main KV
value width and should improve capacity and bandwidth, but it needs its own
needle/coherence A/B because quantization can change attention rankings. The
semantic index is also BF16 in this milestone.

## Guarded concurrency launch

The first validated concurrency step keeps the exact per-sequence limit at
2,048 and admits five sequence slots, for a 10,240-token aggregate admission
envelope. At most three sequences execute in one decode batch:

```bash
MAX_SEQ_LEN=2048 \
MAX_BATCH_SIZE=3 \
MAX_NUM_SEQS=5 \
./scripts/start-glm53-ep2.sh
```

This is deliberately not advertised as a 10K per-request context. Serving one
sequence beyond 2,048 requires GLM's semantic indexer; increasing only the
allocation limit would silently run a different attention algorithm. Batch
sizes above one also require the v2 EP protocol; the launcher enables it on
both ranks automatically.

GLM uses zero-width RoPE in its MLA layers. The multi-sequence path must skip
that empty projection, as the single-sequence path already does, or it launches
a CUDA kernel with a zero-width grid. Distributed decode also keeps the EP
protocol's exact batch width: rounding three live requests to Atlas's generic
four-row CUDA-graph bucket wastes a full KDA/MLA pass on a dummy row.

GLM KDA decode now reads each checkpoint-native NVFP4 Q/K/V/O matrix once for
the active two- or three-row batch. Its small BF16 projections, normalization,
and hyper-connections are also batch-aware; only convolution and recurrent
state updates remain per sequence. At width three, Atlas's existing grouped FFN
path processes all rows together. Width two deliberately retains the sequential
FFN path because the grouped version was neutral in isolated GB10 measurements.

GLM's 11 full-attention layers use checkpoint-native BF16 MLA matrices. The
guarded width-two/three path now reads each large Q-down, Q-up, KV-latent, and
output matrix once for the active batch through Atlas's existing batched dense
projection primitive. Absorption, cache writes, paged attention, and value
extraction remain sequence-private. This confines the optimization to stateless
work and preserves the established cache semantics.

The concurrency receipt uses simultaneous streaming requests with identical
1,000-token prompts and 96 requested output tokens. Each cell has one warm-up
and three measured repetitions. `Aggregate window` divides all completion tokens
by the interval from the first emitted token to the last completed stream;
`sum receipts` sums the server-reported per-session decode rates.

| Concurrent sessions | Per-session decode (median tok/s) | Sum receipts (tok/s) | Aggregate window (tok/s) | Median TTFT |
|---:|---:|---:|---:|---:|
| 1 | 12.627 | 12.627 | 12.760 | 1.066 s |
| 2 | 8.003 | 16.006 | 15.342 | 1.066 s |
| 3 | 6.852 | 20.654 | 19.175 | 1.066 s |

Aggregate window throughput is monotonic through C=3: C=2 is 20.2% above C=1
and C=3 is 50.3% above C=1. Against the preceding KDA-batched result, MLA
projection batching raises C=2 from 14.760 to 15.342 tok/s (+3.9%) and C=3 from
17.872 to 19.175 tok/s (+7.3%). The server-reported summed C=3 rate reaches
20.654 tok/s. Before MLA work, KDA batching had raised C=2 from 12.820 to
14.760 tok/s and C=3 from 13.136 to 17.872 tok/s. A normal-mode C=3 A/B measured
16.003 tok/s with batched FFN disabled, so grouped width-three FFN contributed
another 11.7%. Before exact-width EP dispatch, C=3 was 10.571 tok/s because it
executed the padded fourth row.

The guarded MLA path also passed a three-stream long-prompt needle test: the
independent `SAPPHIRE-7319`, `EMBER-4826`, and `QUARTZ-9051` codes were all
recovered exactly. The receipt used raw completions to prevent hidden reasoning
tokens from exhausting a short chat response budget.

Reproduce the table on the head node with:

```bash
python3 scripts/benchmark_glm53_concurrency.py \
  --prompt-tokens 1000 --output-tokens 96 \
  --max-concurrency 3 --repetitions 3
```

## 1K prompt benchmark

After `/health` is ready, run the included Python receipt on the head node:

```bash
python3 scripts/benchmark_glm53.py --prompt-tokens 1000 --max-tokens 16
```

The driver asks Atlas to tokenize a deterministic seed locally, sends exactly
1,000 token IDs to `/v1/completions`, and reads the server's streamed usage
receipt. At the safe 1,024-token startup limit, 16 generated tokens keep the
request at 1,016 total tokens. Use a 2,048-token launch before increasing the
decode sample to 128 tokens.

The first dual-Spark validation produced 11.407 seconds time-to-first-token
(87.66 prompt tokens/s) and 4.15 decode tokens/s. Profiling showed that KDA
layers were sending single-token MoE work through the prefill dispatcher. After
routing decode through Atlas's single-token MoE path, the same safe launch and
1,000-token prompt produced 5.873 seconds time-to-first-token (170.26 prompt
tokens/s) and 9.42 decode tokens/s. A separate 100-token, 64-output-token run
repeated at 9.52 decode tokens/s.

The optimization campaign uses one warm-up followed by five measured 1,000-token
requests with profiling disabled. Medians on the same two Sparks are:

| Atlas path | Prefill tok/s | Decode tok/s | Disposition |
|---|---:|---:|---|
| Unified layout + prequantized NVFP4 activation path (v18) | 834.67 | 12.7–12.9 | BF16-activation dense-FFN baseline |
| Dense FFN M-fast W4A4 (v30) | **942.86** | 12.83 | Current fastest prefill; opt-in |
| Equal-memory grouped MMQ (v19) | 795.1 | 14.30 | Decode gain, prefill regression; experimental only |
| Native grouped CUTLASS NVFP4 (v23) | 725.57 | **14.67** | Decode gain, prefill regression; experimental only |
| CUTLASS with reused exact-tile offset snapshot (v24) | 723.14 | 14.62 | Neutral; confirms the extra D2H was not the bottleneck |

The external comparison target is approximately 1,500 prefill tok/s and 20
decode tok/s. Optional MMQ and CUTLASS routes remain disabled by default; they
are diagnostic branches, not recommended launch settings. These are controlled
receipts from two DGX Sparks, not general performance claims.

The v30 result is the median of `943.507, 940.538, 929.662, 942.860,
942.878` prompt tok/s after one warm-up, a 13.0% gain over v18. The median TTFT
was 1,060.603 ms and median decode was 12.83 tok/s. A 968-token needle prompt
returned the exact middle-of-prompt recovery code `SAPPHIRE-7319`; a separate
deterministic arithmetic check returned 703 with the correct derivation. With
profiling enabled, the 1K model pass fell from 1,198.6 ms to 1,082.8 ms.

### K=4 MTP verification optimization

With `SPECULATIVE=1`, `NUM_DRAFTS=3`, TP2+EP2, a 2,048-token context cap,
one admitted sequence, and the standard 4 GiB OOM guard, the fixed-width K=4
path now uses the narrow batch-4 NVFP4 projections in both KDA and GLM MLA.
The KDA MoE keeps two fused K=2 passes for routed experts, preserving their
parallel CTA occupancy, but evaluates GLM's always-on shared expert once with
the exact-M=4 GEMV tier. The unified-layout loader retains only the shared
expert's decode-native weights for this purpose (about 0.57 GiB per rank);
routed-expert originals are still freed. Setting `KDA_BATCHED_FFN=0` restores
the generic per-row expert path.

| Warmed receipt | Original K4 | K2-pair v66 | Shared-M4 v70 |
|---|---:|---:|---:|
| K=4 target forward | ~194 ms | 183–185 ms at 1K | **173–177 ms at 1K** |
| 256-token decode, short prompt | 9.1 tok/s | 10.6 tok/s | — |
| 1,000-token prompt + 96-token decode | — | 767.94 prefill / 9.50 decode | **763.64 prefill / 10.79 median decode** |

The scoped v70 decode repeats were `10.345, 10.792, 11.288` tok/s after
warm-up, with a 1,309.5 ms median TTFT and coherent output. A sampled router diagnostic found
14.5 unique experts per 16 K2 routed slots (9% overlap). A full expert-union
CTA was tested and rejected because serializing four row accumulators more than
doubled target-forward latency; only the shared expert is amortized in the
shipping path. A layer-boundary diagnostic (`VERIFY_PROFILE=1`) attributed
about 133 ms of the pre-pairwise K=4 target pass to 34 KDA layers and 46 ms to
11 MLA layers; KDA FFN alone accounted for roughly 102 ms. The pairwise change
reduced that profiled KDA FFN total to about 89 ms. `VERIFY_PROFILE` and
`MOE_UNION_STATS` are diagnostic only and must remain disabled for performance
measurements.

### MTP prompt-state correction and batched KV primer

GLM's MTP module consumes the base model's post-final-norm hidden state for
each shifted prompt pair. Atlas previously captured prompt rows before that
normalization, although its decode path supplied the normalized state. This
prefill/decode mismatch substantially reduced acceptance immediately after a
prompt. Atlas now captures the same post-final-norm representation used by
upstream GLM and initializes the appended MLA layer's prompt cache with a
batched KV-only pass. The latter computes the shifted embeddings, both input
normalizations, `eh_proj`, and compressed MLA K/V, while intentionally skipping
historical attention output and MoE work that no future proposal consumes.

`GLM_MTP_BATCHED_PREFILL` defaults to the value of `SPECULATIVE`. Set it to
`0` and `GLM_MTP_SERIAL_PREFILL=1` only to compare against the slow full-layer
correctness oracle; the two modes are mutually exclusive.

On the same TP2+EP2, K=4 launch, the KV-only primer populated 999 prompt rows
in 26.7--27.6 ms. After one warm-up, two exact 1,000-token requests that ended
naturally after 113 generated tokens measured:

| Run | TTFT | Prefill | Decode | Mean accepted drafts (of 3) |
|---|---:|---:|---:|---:|
| 1 | 1,296.959 ms | 771.034 tok/s | 18.734 tok/s | 2.645 |
| 2 | 1,301.142 ms | 768.555 tok/s | 19.496 tok/s | 2.767 |

Speculative results depend strongly on the generated sequence. A separate
1,000-token chat benchmark forced to generate all 256 tokens measured `12.699`
and `13.293` tok/s (12.996 median), with only 1.462 and 1.550 mean accepted
drafts. These longer receipts are the appropriate sustained-output baseline;
the 18.7--19.5 tok/s result demonstrates the corrected high-acceptance path,
not a universal decode rate. The external repository's advertised 23--30
tok/s single-session range does not include enough workload detail for a
strict apples-to-apples comparison.

### K=5 MTP verification

`NUM_DRAFTS=4` verifies five rows per target pass. The generic K-gamma path
was initially correct but slow: its target forward took about 322 ms. Two
GLM-specific dispatch gaps caused the cliff at five rows:

- KDA q/k/v/o projections used the small-prefill GEMM even though Atlas
  already shipped exact-M=5 NVFP4 GEMVs;
- every KDA MoE evaluated five rows independently instead of composing the
  fused K2 and K3 routed paths and reading the shared expert once.

The KDA projection dispatcher now uses the common exact-M tier resolver for
M=4..=8. The K=5 MoE path evaluates the shared expert once with the M5 tier,
then combines routed-only K2 and K3 passes. A failed experiment that applied
the same FFN composition to the 11 MLA layers was measured slightly slower and
was removed rather than shipped.

The target forward fell from ~322 ms to 223--224 ms. With a 1,536-token safety
cap, TP2+EP2, one admitted sequence, BF16 KV, and a 4 GiB OOM guard, one warm-up
followed by two exact 1,000-token requests produced:

| Run | TTFT | Prefill | Decode | Mean accepted drafts (of 4) |
|---|---:|---:|---:|---:|
| 1 | 1,316.850 ms | 759.388 tok/s | 19.490 tok/s | 3.708 |
| 2 | 1,315.246 ms | 760.314 tok/s | 19.551 tok/s | 3.708 |

Both requests ended naturally after 113 generated tokens. This makes K=5
competitive with the corrected K=4 high-acceptance path while verifying one
additional draft. It does not supersede the forced-256 K=4 sustained-output
baseline above; K=5 still needs an equivalent low-acceptance receipt before it
can be selected as a general default.

The first K=5 MLA follow-up removes another width cliff. GLM's guarded
zero-RoPE MLA projection path originally admitted only two through four rows,
so a five-row verify reread the four large BF16 projection matrices once per
row in all 11 MLA layers. The path now admits five rows and selects an exact-M5
BF16 GEMV instantiation instead of the generic M<=8 accumulator tier.

With the same deployment and one warm-up, two high-acceptance 1,000-token
requests measured `21.374` and `21.575` tok/s (113 generated tokens), up from
`19.490` and `19.551` tok/s. Target-forward time fell from 221--223 ms to
197--201 ms. A separate chat request with an exact 1,000-token rendered prompt
and `min_tokens=max_tokens=256` measured `11.781` and `11.915` tok/s after
warm-up, versus `10.447` and `11.731` tok/s before the change. That is a 6.8%
median sustained-output gain despite normal acceptance-path variance. Prefill
remained effectively unchanged at 766--768 tok/s, and both output workloads
remained coherent.

The next guarded step batches the rest of GLM's exact five-row MLA chain. A
dedicated BF16 GEMV shares each absorbed-Q and extracted-V weight load across
the five verification rows; cache assembly/write and paged attention are also
submitted once for all five rows while preserving each row's sequence length
and block table. Across twelve timed target-forward samples, median target
time fell from about 196.5 ms to 194.6 ms (roughly 1%). End-to-end forced-256
decode remained acceptance-bound and noisy: the two measured requests were
12.623 tok/s with 166 accepted drafts and 10.374 tok/s with 143. Prefill stayed
at 771--774 tok/s and coherent-output checks passed. This is a modest kernel
scheduling win, not the remaining route to vLLM-class sustained decode.

An eager per-layer verifier profile locates that remaining work. Median K=5
time was 137.53 ms across the 34 KDA layers and 75.15 ms across the 11 MLA
layers. The KDA FFN accounted for 91.48 ms by itself, about two thirds of KDA
time. A true five-row routed-expert dispatch was tested but discarded: it
reduced launch count while expanding the active expert wave from K2/K3 to 40
slots, lowering acceptance-normalized target throughput by about 10%. The
proven K2+K3 routed schedule therefore remains in place.

The next K=5 change preserves those small expert waves but defers their
expert-parallel reductions. K2 and K3 write rank-local routed contributions
into one contiguous five-row buffer, which is reduced once before the shared
expert result is added. Existing K2 and K3 callers retain their independent
reductions; the change is isolated to GLM's K=5 verifier.

An exact A/B used a 1,280-token sequence cap, one admitted sequence, BF16 KV,
and the same 1,000-token rendered chat prompt with
`min_tokens=max_tokens=256`. The v84 control's two measured requests decoded
at `19.324` and `18.672` tok/s (18.998 median). Across two v86 measurement
sets, five requests decoded at `20.305`, `18.048`, `18.897`, `19.288`, and
`19.217` tok/s (19.217 median, +1.15%). Median accepted-draft counts were 195
and 194 respectively, so the comparison is not explained by a favorable MTP
acceptance shift. Twelve timed v86 target-forward samples had a 195.48 ms
median, versus 199.02 ms for five v84 control samples (-1.78%). Median v86
prefill was 855.55 tok/s versus 849.03 tok/s for the control. All responses
generated the requested 256 tokens and preserved the control's deterministic
output prefix.

K=5 routing is also shared across the two expert waves. GLM's five BF16 router
rows and sigmoid top-k are evaluated once, then the device-resident indices and
weights are sliced between K2 and K3. The optimized arm explicitly excludes
hash routing and sqrt-softplus scoring, so those models retain their established
paths. Expert execution remains K2+K3 and the single EP reduction above is
unchanged.

Against v86 with the same launch and request, v87's three measured forced-256
runs decoded at `19.630`, `19.999`, and `18.677` tok/s (19.630 median, +2.15%).
Median accepted drafts were 195 versus v86's 194. Seven timed target-forward
samples had a 191.55 ms median, down 2.01% from v86's 195.48 ms. Median prefill
was effectively flat at 852.61 tok/s. The deterministic output prefix remained
identical, and a separate arithmetic chat completed with the correct simplified
fraction `5/16` in both separated reasoning and visible content.

The routed-expert follow-up reuses Atlas's existing grouped W4A16 tensor-core
pipeline for the complete five-row verifier. It sorts the 40 top-k routes by
expert, keeps the source activations in BF16, then converts activation and
dequantized weight tiles on chip to E4M3 for FP8 MMA. Unlike Atlas's W4A4 MMQ
path, it does not quantize and stage verifier activations in FP4. The grouped
path also skips the expert-offset host copy when the worst case is already one
64-row tile; at K=5 and top-k=8, all 40
routes fit in that tile by construction. `GLM_K5_GROUPED_MOE=0` restores the
K2+K3 verifier path.

When the exact-M shared path is selected, `GLM_K5_FUSED_SHARED_GATE_UP=1`
submits its independent gate and up projections as two planes of one native
NVFP4 launch. Each plane executes the unchanged batch-five dot-product body;
`=0` restores the two-launch path for direct A/B and compatibility testing.
On the same v21 image, 20 target-forward timing windows measured 117.635 ms
median / 117.578 ms mean with fusion versus 117.975 ms / 118.017 ms without it
(-0.29% / -0.37%). One warm-up plus five forced-256 endpoint runs improved
from 19.010 to 19.187 tok/s median (+0.93%) and from 18.958 to 19.599 tok/s
mean (+3.38%); the larger endpoint-mean movement includes normal accepted-draft
variation, so the target-forward delta is the conservative kernel claim. A
separate arithmetic chat returned `323` with reasoning and visible content
still separated.

`GLM_K5_DENSE_EXACT=1` selects Atlas's existing exact-five-row BF16 GEMV for
the KDA side projections instead of executing the general eight-row kernel
with three inactive accumulator lanes. `=0` restores the general batch-M
kernel for same-image comparison. On the same v22 image, 12 target-forward
windows improved from 117.075 to 116.580 ms median (-0.42%) and from 116.923
to 116.560 ms mean (-0.31%). Three endpoint samples were acceptance-limited
(160 versus 157 median accepted predictions), so the target-forward delta is
the performance claim. A separate arithmetic chat returned `899` correctly
with reasoning and visible content separated.

`GLM_K5_FUSED_DENSE_PAIRS=1` then submits KDA's same-shape `f_a/g_a` and
`f_b/g_b` BF16 projections as two planes per pair. Each plane uses the same
exact-five dot-product body and its original input, weight, and output; `=0`
restores four individual launches. On the same v23 image, 12 target-forward
windows improved from 116.965 to 115.870 ms median (-0.94%) and from 116.834
to 115.891 ms mean (-0.81%). Three forced-256 endpoint runs were essentially
flat at the median (-0.37%) and +0.76% at the mean with the same median accepted
draft count; the less acceptance-sensitive target-forward timings are the
performance claim. An arithmetic chat returned `1517` correctly and kept its
reasoning separate from the visible answer.

`GLM_K5_FUSED_DENSE_TRIPLE=1` extends the same submission pattern to the
same-input `beta/f_a/g_a` projections, including beta's narrower output plane.
The kernel keeps each plane's original output width and exact-five reduction;
`=0` restores the separate beta launch plus the fused `f_a/g_a` pair.
On the same v24 image, 15 target-forward timing windows improved from 116.230
to 115.680 ms median (-0.47%) and from 116.177 to 115.777 ms mean (-0.34%).
Three forced-256 endpoint runs were acceptance-limited (19.855 versus 19.320
tok/s median), while their means were effectively flat at 19.569 versus 19.513
tok/s. A separate arithmetic request still produced the correct result `1517`.

With one warm-up and five measured copies of the same exact 1,000-token,
forced-256 request, grouped W4A16 decoded at `19.046`, `19.414`, `21.029`,
`20.243`, and `20.646` tok/s (20.243 median, +3.1% over v87). Median accepted
drafts were 194 versus v87's 195, and median prefill remained flat at 852.42
tok/s. Target-forward samples from the benchmark had a 177.0 ms median, 7.6%
below v87's 191.55 ms. A direct arithmetic check returned `5/16`, and a
1,002-token needle prompt recovered `SAPPHIRE-7319` exactly.

An equal-memory MMQ comparison was also made before selecting W4A16. It lowered
target-forward time further to 158--161 ms, but quantizing verifier activations
to FP4 reduced accepted drafts to 154 and sustained decode to 13.68 tok/s on
the forced workload. MMQ therefore remains experimental and disabled. Its
small-batch fallback now routes through the layout-aware grouped dispatcher,
preventing repacked weights from being consumed by checkpoint-layout kernels.

The next prefill-only optimization defers the shared expert until routed MoE is
complete, then runs the shared GEMMs on Atlas's auxiliary CUDA stream while the
main stream performs the EP all-reduce. This pairs compute with communication
instead of overlapping two LPDDR5X-heavy expert paths. It is deliberately
restricted to more than 64 rows: applying it to the five-row MTP verifier
reduced decode from 15.04 to 12.23 tok/s in the rejection test.

On the exact 1,000-token benchmark, one warm-up plus five measured requests
gave 875.35 tok/s median versus 862.82 with the overlap disabled (+1.45%). A
second prompt followed by a deterministic 97-token continuation measured
880.98 versus 861.77 prefill tok/s (+2.23%); decode was 15.97 versus 15.04
tok/s, confirming that the >64-row guard preserved the verifier path. All
measured continuation outputs were byte-identical across the A/B comparison.
Arithmetic still returned `5/16`, and a 964-token rendered needle prompt
recovered `SAPPHIRE-7319`. `MOE_SHARED_REDUCE_OVERLAP=0` restores the fully
sequential schedule.

The sustained verifier path subsequently gained two more guarded changes.
First, the eleven MLA-layer FFNs run their native NVFP4 projections as one
three- or five-row operation instead of repeating the single-row decode path.
On the exact 1,000-token, forced-256 benchmark, fixed K=3 improved from a
13.34 tok/s median to 15.99 tok/s (+19.9%), while fixed K=5 improved from
13.38 tok/s to 15.94 tok/s (+19.2%). The K=5 gain remained 12.9% after
normalizing target-step throughput by accepted drafts.

Second, the grouped K=5 routed-expert path now has an opt-in exact-M shared
expert dispatch. `GLM_K5_BATCHED_SHARED=1` reuses the existing batch-five
NVFP4 W4A16 kernels for the shared gate, up, and down projections, avoiding
the generic prefill GEMM's padded tile at only five rows. An eager profile
reduced the 34 KDA layers from roughly 105 ms to 96 ms per target step. With
profiling disabled, one warm-up plus five exact 1,000-token, forced-256 runs
decoded at `16.716`, `14.927`, `16.932`, `17.144`, and `16.076` tok/s: a
16.716 median, +4.9% over the 15.941 tok/s control. The low run coincided with
137 accepted predictions versus 147--153 in the other runs. All completions
retained separated reasoning and coherent technical output.

The routed gate/up follow-up keeps the same native Blackwell FP4 MMA tile but
replaces its dense `[16, 1, 288]` expert grid with a device-built compact
worklist. At K=5, each gate and up projection now launches at most 640 CTAs
instead of 4,608; empty and rank-remote experts never enter the MMA kernel.
The dispatch is restricted to GLM's exact five-row, 40-route shape and is
opt-in with `GLM_K5_COMPACT_MOE=1`. Its device tile count is clamped to the
host-provided capacity and every work item validates its expert and N tile.

With one warm-up and five measured exact 1,000-token, forced-256 requests, the
gate/up-only image decoded at `16.656`, `17.028`, `17.238`, `17.135`, and
`15.921` tok/s (17.028 median), +1.9% over the 16.716 tok/s control above.
A rebuilt, hardened image repeated at `17.547`, `15.607`, and `17.304` tok/s
(17.304 median); the low sample again tracked fewer accepted predictions.
A separate arithmetic request completed naturally at 19.16 tok/s with cleanly
separated reasoning and the correct visible answer `5/16`.

Compacting the down projection was tested independently and as part of the
full routed path. Although it completed short isolated requests, the combined
variant produced one stalled long request and later measured only 14.967 tok/s
median on the same forced workload. It is therefore absent from the shipping
dispatch; down retains the established dense native-FP4 kernel.

A smaller M=32 compact gate/up CTA was also tested to reduce inactive-row MMA
at K=5. It retained the native FP4 instructions but cut the block from 128 to
64 threads. On the exact 1,000-token, forced-256 workload, the M32 image
decoded at `15.747` tok/s median versus `17.523` tok/s for the M64 control from
the same binary (-10.1%). The reduced loader participation and duplicated
B-tile staging outweighed the saved inactive-row math, so the M32 variant was
removed rather than retained as another runtime branch.

The next guarded change keeps the proven M64/block-128 kernel and multiplexes
the compact gate and up projections in one two-dimensional launch.
`GLM_K5_FUSED_COMPACT_GATE_UP=1` selects gate/up with grid Y while grid X
indexes the unchanged device worklist. The native Blackwell FP4 helper, K
accumulation order, scale handling, outputs, and EP ownership are unchanged;
the optimization removes one submission per routed layer.

With the phase ledger enabled, 16 steady fused K=5 windows had a 127.015 ms
median target-forward time versus 128.085 ms across 12 unfused windows from
the same image (-0.84%). Three exact 1,000-token, forced-256 measurements
decoded at `19.091`, `17.416`, and `16.735` tok/s (17.416 median); the matched
unfused runs were `18.050`, `14.355`, and `17.599` tok/s (17.599 median). The
end-to-end medians are acceptance-noisy—the low unfused run accepted only 136
drafts—but the direct target-forward ledger consistently resolves the launch
reduction. Short 8/32-token smoke tests completed normally, and an arithmetic
check kept reasoning/content separated and computed `5/16` correctly.

Compact K=5 also changes the depth-controller economics. The controller was
calibrated when a K=5 target step cost about 1.20x K=3, but the optimized
native-FP4 K=5 path now completes at roughly the same step cost while emitting
every token K=3 can emit plus up to two more accepted drafts. On the exact
1,000-token, forced-256 workload, fixed K=3 decoded at `15.130`, `14.061`, and
`14.856` tok/s (14.856 median), versus 17.626 tok/s for fixed K=5 (+18.6%).
The old adaptive default measured 16.396 tok/s because it moved into the
now-dominated K=3 path after its initial 12-step probe.

Accordingly, `start-glm53-ep2.sh` now defaults
`MTP_SINGLE_DEPTH_ADAPT=0` whenever `GLM_K5_COMPACT_MOE=1`. Non-compact
launches retain the prior adaptive default, and an explicitly supplied
`MTP_SINGLE_DEPTH_ADAPT` always wins. This changes no model math or request
semantics; it keeps the optimized launcher on the measured faster verifier.

The KDA verifier now batches its stateful K=5 section without weakening
rollback correctness. Previously, each of the 34 KDA layers submitted five
one-token convolutions, five one-token recurrences, four FP32 convolution-state
copies, and four FP32 H-state copies. The batched variants perform one
five-token convolution and one five-token recurrence, writing the four
post-token rollback states inline. Dispatch requires the exact five-row GLM
shape and contiguous snapshot pools; otherwise it falls back to the original
interleaved path. `GLM_K5_BATCHED_CONV_SNAPSHOT=0` and
`GLM_K5_BATCHED_RECURRENT_SNAPSHOT=0` independently restore the old paths.

The dedicated GB10 hardware check compared outputs, final states, and all four
rollback states byte-for-byte against the legacy sequence: every comparison
had zero differing bytes. With verifier profiling enabled, the combined
pack/conv/recurrent/snapshot phase fell from 180 us to 104 us per KDA layer,
and the 34-layer KDA section fell from a 92.77 ms median to 89.55 ms. A separate
five-window MTP ledger measured median target-forward time at 123.62 ms versus
126.85 ms for the same-image control (-2.55%), and median complete MTP-step
time at 143.26 ms versus 145.87 ms (-1.79%). On the warmed 1,000-token,
forced-256 endpoint test, five optimized requests measured 17.566 tok/s median
versus 16.286 tok/s for the control (+7.9%); accepted-prediction variation
(153.0 versus 147.4 mean) exaggerates that endpoint delta, so the direct
target-forward ledger is the conservative speed claim. Both snapshot levers
default on in the GLM launcher after the exactness and end-to-end checks.

The next native-NVFP4 change multiplexes the KDA Q, K, and V projections in
one three-plane launch. Grid Z selects the independent projection weight and
output plane, while each plane calls the unchanged exact-M=5 FP4 GEMV body;
no dot product is combined or reassociated. The GB10 hardware oracle reported
zero differing bytes across all three output planes versus three separate
launches. Per-layer Q/K/V-plus-beta projection time fell from about 194 us to
169 us. Five-window MTP timing on the same image measured target-forward at a
121.57 ms median versus 122.97 ms with `GLM_K5_FUSED_QKV=0` (-1.14%); complete
MTP-step medians were 141.58 and 142.35 ms respectively. The GLM launcher now
defaults the fused path on, with the environment flag retained as an exact
fallback switch.

The K=5 verifier now also sends each five-row mHC pre-mix through Atlas's
existing batched TF32 cuBLASLt path instead of evaluating 24 independent
matrix-vector reductions per token inside `hc_pre`. The final RMS scaling,
sigmoid gates, Sinkhorn iterations, residual collapse, and FP32 highway remain
in the existing Atlas kernel. `GLM_K5_HC_CUBLAS=0` restores the serial-FP32
pre-mix; the launcher defaults the measured batched path on.

With verifier profiling enabled, both KDA mHC-plus-RMS boundaries fell from
about 182 us to 94--98 us per layer. Across all 34 KDA layers, two measured
K=5 forwards fell from about 90.1 ms to 84.0--84.2 ms. With profiling disabled,
one warm-up plus five exact 1,000-token, forced-256 requests decoded at
`18.695`, `19.712`, `19.162`, `20.732`, and `18.762` tok/s: 19.162 tok/s
median and 19.412 mean. The same-image control measured `18.830`, `17.578`,
`18.249`, `18.414`, and `19.372` tok/s: 18.414 median and 18.489 mean. That is
+4.1% by median and +5.0% by mean, while median accepted predictions remained
effectively unchanged (159 versus 160), so the gain is not an MTP-acceptance
artifact.

The appended MTP layer's input combiner is a BF16 `[4096, 8192]` matrix. It
was streamed once for each of the four serial drafts even though prompt KV
prefill is the only phase that benefits from its BF16 batched-GEMM layout.
`GLM_MTP_NVFP4_EH=1` now keeps an additional compact NVFP4 copy for the
one-row draft path while retaining BF16 for prompt prefill; setting it to `0`
restores the original projection exactly. The extra rank-0 allocation is about
18 MiB including block scales.

On a same-image A/B, four 25-step timing windows reduced the proposer from a
17.345 ms median to 16.56 ms (-4.6%). On the established one-warm-up plus five
exact 995-rendered-token, forced-256 chat workload, the NVFP4 arm measured
`20.278`, `20.067`, `19.631`, `19.211`, and `17.873` tok/s (19.631 median,
19.412 mean) versus BF16's `18.304`, `17.468`, `19.815`, `19.500`, and
`19.784` tok/s (19.500 median, 18.974 mean). Median accepted predictions were
160 and 159 respectively, so the measured raw proposer saving did not trade
away acceptance. A separate arithmetic request completed naturally with the
correct exact result `10/1` and decimal `10.0`.

### Mirrored MTP body with split vocabulary projection

The target verifier already uses both Sparks, but the appended GLM MTP
proposer historically ran only on rank 0. `GLM_MTP_DISTRIBUTED=1` now mirrors
the checkpoint-native full TP1/EP1 proposer body on both ranks and assigns each
rank one contiguous half of the exact vocabulary projection. The two BF16
logit halves are exchanged before the unchanged full-vocabulary argmax. BF16
and GS16 NVFP4 row offsets are aligned for this model, so the projection does
not change any per-row reduction arithmetic.

An earlier TP2/EP2 body prototype reduced proposer time further, but its BF16
MLA and routed-MoE reductions changed draft acceptance enough to erase the
endpoint gain; that form was rejected. Mirroring the proven body recovered
acceptance while still reducing steady proposer windows from about 17.6 ms to
14.7 ms (-16.5%). On a warmed 1,005-token prompt with 256 forced decode tokens,
five measured requests reported `20.224`, `18.800`, `19.160`, `19.913`, and
`18.218` tok/s: 19.160 median and 19.263 mean. The matched rank-0 proposer
control from the same binary reported `19.288`, `18.219`, `19.434`, `18.898`,
and `18.879` tok/s: 18.898 median and 18.944 mean. The split projection gained
1.4% by median and 1.7% by mean. Median accepted predictions were 154 versus
156, confirming that the decode gain was not purchased by the earlier material
acceptance regression.

This path is deliberately opt-in and currently requires two ranks with
TP=EP=2, `MAX_BATCH_SIZE=MAX_NUM_SEQS=1`, one to four draft tokens, batched MTP
prompt KV prefill, and serial MTP prefill disabled. Both ranks must use the same
image and flag value. The default rank-0 proposer remains unchanged when the
flag is off.

The split projection initially exchanged its two contiguous logit halves with
two synchronous broadcasts per draft. `GLM_MTP_ALL_GATHER=1` replaces them with
one NCCL in-place all-gather. Rank `r` already writes at
`logits + r * local_vocab`, which is NCCL's defined in-place layout, and the
collective uses the same default CUDA stream as projection and global argmax.
The fallback broadcasts remain available with `=0`; the launcher defaults the
one-collective path on only when `GLM_MTP_DISTRIBUTED=1`.

On the same v29 binary, steady 25-step proposer windows fell from a 14.37 ms
median with the broadcast oracle to 13.91 ms with all-gather (-3.2%). Five
warmed 1,005-token, forced-256 endpoint requests were effectively neutral:
19.317 versus 19.295 tok/s median, and 19.148 versus 19.264 tok/s mean. The
direct phase result is the conservative claim; it removes one collective and
one host synchronization without changing projection or argmax arithmetic.

### Exact five-row router

The K=5 verifier previously sent its `[5, 4096] x [4096, 288]` BF16 router
through the generic exact router kernel's 16-row tile. Eleven row lanes did
the complete scalar K loop on padding. `GLM_K5_ROUTER_M5=1` selects a
shape-guarded five-row kernel with 16 columns per CTA. Every real output keeps
the same increasing-K FP32 accumulation chain and final BF16 conversion; the
existing exact router remains the fallback for every other shape. The GLM
launcher defaults the specialization on after validation.

The GB10 microtest produced zero bit differences across all 1,440 logits and
reduced kernel-only time from 0.1579 ms to 0.0901 ms (-42.9%). In the complete
42-layer verifier profile, router time fell from 8.67 ms to 7.20 ms per target
forward (-17.0% after launch and pipeline effects). A same-image endpoint A/B
with one warm-up and five 1,005-token-prompt, forced-256 requests measured
18.456 tok/s mean and 18.779 median with the specialization, versus 17.731
mean and 17.680 median for the fallback. Accepted drafts varied between the
samples, so the bit-exact microtest and per-target profile are the conservative
speed claims rather than the larger endpoint delta.

### Fused two-rank KDA reduction and mHC post

The five-row KDA output projection previously exchanged the peer BF16 tensor,
launched `bf16_add_inplace`, stored the reduced tensor, and immediately loaded
it again in `hc_post`. `GLM_K5_FUSED_TP_HC=1` instead exchanges the unmodified
peer contribution into Atlas's already-registered MoE scratch and evaluates
the same `__hadd(local, peer)` inside an `hc_post` twin. The rounded BF16 sum,
FP32 conversion, mHC accumulation order, and output stores are unchanged. The
path is restricted to eager K=5 verification with exactly two ranks; graph
capture, every other shape, and other communication backends retain the
existing all-reduce path. The dual-Spark launcher defaults it on after the
exactness and endpoint checks.

Across 137 profiled target forwards, the fused communication phase measured
8.05 ms per KDA stack versus 10.14 ms across 127 same-image control forwards
(-20.6%). On the profiler-off 1,005-token-prompt, forced-256 A/B, acceptance
variation made raw endpoint rates misleading: 18.011 tok/s candidate mean
versus 18.511 control. Normalizing each request by its actual number of target
verification steps showed the model work falling from 139.89 to 138.67 ms per
step by mean (-0.87%), and from 140.67 to 138.57 ms by median (-1.49%). A
separate natural arithmetic request completed coherently with the exact result
`73 * 19 = 1387`.

If either rank exits during model load, remove both Atlas containers before a
retry. Do not configure a Docker restart policy: repeatedly reloading a model
under unified-memory pressure can make both Sparks unreachable.

### Distributed K=5 CUDA graph

The generic five-row target verifier used by four-draft MTP was permanently
eager whenever a communication backend was present, even though this exact
GLM path has fixed shapes, fixed arena/state pointers, and identical collective
order on both ranks. `GLM_TP_VERIFY_GRAPH=1` now permits CUDA/NCCL capture only
for `glm5_next`, K=5, and TP=2. Other models, verifier widths, and distributed
topologies retain the established eager behavior. Setting the flag to `0` is
the immediate fallback; the dual-Spark launcher defaults it on after validation.

Both ranks captured and replayed the graph successfully without CUDA, NCCL, or
OOM errors. A natural arithmetic request remained coherent and returned the
exact result `73 * 19 = 1387` with two independent checks. On the same v35
binary, one warm-up plus five 1,005-token-prompt, forced-256 requests measured
18.693 tok/s mean and 18.889 median with capture, versus 17.517 mean and 17.660
median eager (+6.7% mean, +7.0% median). Because draft acceptance varied,
target-forward work was also normalized by `completion_tokens -
accepted_prediction_tokens`: capture measured 135.58 ms per target step mean
and median, versus 139.24 mean and 138.41 median eager (-2.6% mean, -2.1%
median). The normalized result is the conservative decode claim.

### Fused MTP embedding and hidden-state normalization

Each speculative draft previously copied one embedding row to scratch and
then launched two independent RMSNorm kernels to construct
`[enorm(embed(token)), hnorm(target_hidden)]`. `GLM_MTP_FUSED_EH_NORM=1`
replaces those three operations with one two-CTA kernel: one CTA reads the
embedding table directly and the other reads the target hidden state. The
kernel deliberately preserves the established thread mapping, FP32 reduction
tree, multiplication order, and BF16 conversion. The exact GLM launcher now
defaults it on; `=0` retains the original path.

For bring-up, `GLM_MTP_FUSED_EH_CHECK=1` saves the fused result, reruns the
original copy plus two-normalization oracle, and requires every output byte to
match. Both ranks independently reported zero differences across 16,384 bytes.
With one warm-up followed by five forced-256 requests using a 1,005-token
prompt, steady proposer windows improved from 13.791 ms mean / 13.780 ms
median to 13.654 ms mean / 13.640 ms median (-1.0% for both). Raw endpoint
throughput was inconclusive because accepted-draft counts varied, so the
direct proposer timing is the performance claim.

### Fused K=5 shared-expert blend and mHC post

GLM's K=5 MoE path previously materialized the BF16 sum of the globally
reduced routed experts and sigmoid-gated shared expert, then immediately read
that temporary into the hyperconnection post kernel. With
`GLM_K5_FUSED_MOE_HC=1`, the mHC kernel consumes the routed contribution,
shared contribution, input row, and shared gate directly. It preserves the
original FP32 reduction and sigmoid, including the explicit BF16 rounding at
the former kernel boundary, while removing one launch and one complete
five-row output store/read per MoE layer. The optimization is restricted to
GLM's exact K=5, EP=2 grouped path; the launcher defaults it on and `=0`
retains the established implementation.

The one-shot `GLM_K5_FUSED_MOE_HC_CHECK=1` oracle reruns the old blend and mHC
post and compares the complete hyperconnection state. Both ranks matched all
327,680 bytes exactly. Across steady draft-forward samples, the fused path
improved from 110.736 ms mean / 110.900 ms median to 110.397 ms mean /
110.450 ms median (-0.31% / -0.41%). One-warm-up, five-request endpoint A/B
testing normalized by target steps improved from 136.43 ms to 135.17 ms mean
(-0.92%); medians were 135.75 ms and 135.52 ms. Raw token rates remain
acceptance-dependent, so the normalized and direct phase measurements are the
reliable claims.

### Mixed-precision MTP vocabulary policy

The appended predictor's per-draft profile showed that its full MLA+MoE body
cost about 1.8 ms, while each BF16 half-vocabulary projection cost another
1.7 ms. The already-established NVFP4 projection used by drafts three and four
needed only 0.45--0.50 ms. `GLM_MTP_BF16_DRAFTS` now controls how many leading
drafts retain the exact tied BF16 head. The dual-Spark launcher defaults to one:
draft one remains BF16, while drafts two through four use the compact head.
`=2` restores the previous policy exactly, and values zero through four remain
available for controlled acceptance studies.

On the same image and 1,004-token-prompt, forced-256 workload, changing the
leading BF16 count from two to one reduced steady proposer windows from about
13.6 ms to 12.4--12.6 ms (roughly 8%). One warm-up plus six measured requests
reduced endpoint wall time from 14.727 s mean / 14.752 s median to 14.250 s /
14.117 s (-3.2% / -4.3%). Mean accepted predictions did not regress in that
sample (148.0 control versus 150.8 candidate). A natural arithmetic request
continued to return the correct result. `GLM_MTP_PROFILE=1` enables the
synchronized MTP-only phase profiler used for this analysis; it remains off by
default and should never be used for throughput measurements.

### Decode-native MTP MLA output projection

The appended predictor's full-rank MLA output projection remained a BF16
`[4096, 16384]` matrix and was streamed once for every serial draft. The
predictor-only `GLM_MTP_NVFP4_WO` path now constructs a compact NVFP4 copy at
load time and sends the one-row output projection through Atlas's established
decode GEMV. Target-model MLA layers are unchanged, the BF16 predictor weight
is retained as the `=0` fallback, and the launcher defaults the measured path
on.

On the same v49 image, 1,004-token prompt, forced-256 output, and fixed seed
set, one warm-up plus six measured requests took 13.778 s mean / 13.440 s
median with the native-NVFP4 projection versus 14.508 s / 14.448 s with the
BF16 control (-5.0% / -7.0%). Mean endpoint completion rate increased from
17.654 to 18.621 tok/s (+5.5%). Steady proposer windows fell from roughly
12.45--12.77 ms to 10.68--11.03 ms (about 14%). Sampled acceptance did not
regress; the mean of the five reported accepted-draft windows was 1.455 for
NVFP4 and 1.397 for BF16.

### Distributed MTP top-1 reduction

The split-vocabulary proposer previously all-gathered both BF16 logit halves
before running the ordinary full-vocabulary argmax. For unconstrained decode,
`GLM_MTP_DISTRIBUTED_ARGMAX=1` now runs the identical first-strict-max
reduction over each contiguous local half and exchanges only an 8-byte
`(value, local_index)` pair per rank. Rank zero wins equal values, exactly
matching the lower-token-ID tie break of the concatenated argmax. Requests
with a grammar bitmask retain the established full-logit gather because their
mask must inspect the complete vocabulary. The dual-Spark launcher enables
the reduction whenever distributed MTP is enabled; `=0` is the direct
full-gather fallback.

On the same v51 image and 1,004-token-prompt, forced-256 workload, one warm-up
plus six measured requests took 13.907 s mean / 13.714 s median with the
distributed top-1 path versus 14.477 s / 14.259 s for the full-logit control
(-3.9% / -3.8%). Mean endpoint decode increased from 17.719 to 18.432 tok/s
(+4.0%). Direct steady proposer windows show the conservative gain is smaller,
roughly 1--2% (about 10.70--10.85 ms versus 10.80--11.00 ms); endpoint spread
still reflects generated-sequence variance.

### One-pass exact BF16 target vocabulary head

GLM's K=5 verifier retains the model's exact BF16 target vocabulary head, but
the five normalized hidden rows previously fell through to the generic tiny-M
GEMM. With `GLM_K5_BF16_LMHEAD_BATCHM=1`, Atlas dispatches those exact five
rows through its existing batched BF16 GEMV instead. That kernel streams each
vocabulary weight row once and reuses it across all five hidden rows. No model
weights or output logits are quantized; `=0` restores the generic GEMM path.
`LM_HEAD_DTYPE=default|bf16|nvfp4|fp8` is also exposed by the launcher so head
precision experiments are explicit, while the production default remains the
model's configured dtype.

The one-shot `GLM_K5_BF16_LMHEAD_BATCHM_CHECK=1` oracle reruns the former GEMM
and requires all five target argmax IDs to match. It passed independently on
both physical ranks. On an identical image with one warm-up followed by eight
forced 96-token requests, the optimized path reached 24.411 tok/s mean and
24.423 tok/s median versus 21.471 tok/s mean and 21.691 tok/s median for the
disabled control (+13.7% mean). Mean accepted predictions were 59.5 versus
59.125. Steady verifier forwards fell from roughly 108.2 ms to 97--99 ms. A
separate natural decode returned the correct `73 * 19 = 1387` result with its
reasoning kept separate from visible content.
