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

`MAX_PREFILL_TOKENS` is independent of `MAX_SEQ_LEN` and defaults to 1,024.
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

If either rank exits during model load, remove both Atlas containers before a
retry. Do not configure a Docker restart policy: repeatedly reloading a model
under unified-memory pressure can make both Sparks unreachable.
