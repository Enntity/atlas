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
- one batch item and one concurrent sequence;
- BF16 KV cache;
- a 92% total GPU-memory budget and 4 GiB OOM guard;
- a 114 GiB container memory ceiling, leaving host headroom if loading runs away;
- no speculative decoding or prefix cache;
- cuBLASLt BF16 projection dispatch for checkpoint-native MLA matrices;
- the worker rank first and no automatic restart policy.

The KDA register-resident prefill path, KDA multi-sequence decode, width-three
batched FFN, unified MoE layout, and cuBLASLt projection dispatch are enabled by
default. Their diagnostic fallbacks are `KDA_REGRESIDENT_PREFILL=0`,
`KDA_MULTI_SEQ=0`, `KDA_BATCHED_FFN=0`, `UNIFIED_MOE_LAYOUT=0`, and
`CUBLAS_GEMM=0`.
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

This checkpoint declares 1M model context, but initial Atlas support does not.
GLM's full-attention layers select 2,048 tokens using an indexer. Atlas currently
uses causally masked dense attention for those layers, which is mathematically
equivalent while the entire sequence is no longer than `index_topk=2048`.
Startup therefore rejects values above 2,048 rather than silently changing the
model. KDA recurrent state is FP32 and allocated for one sequence; increasing
concurrency multiplies that state and should follow a measured memory audit.

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

The concurrency receipt uses simultaneous streaming requests with identical
1,000-token prompts and 96 requested output tokens. Each cell has one warm-up
and three measured repetitions. `Aggregate window` divides all completion tokens
by the interval from the first emitted token to the last completed stream;
`sum receipts` sums the server-reported per-session decode rates.

| Concurrent sessions | Per-session decode (median tok/s) | Sum receipts (tok/s) | Aggregate window (tok/s) | Median TTFT |
|---:|---:|---:|---:|---:|
| 1 | 12.606 | 12.606 | 12.738 | 1.063 s |
| 2 | 7.683 | 15.366 | 14.760 | 1.070 s |
| 3 | 6.345 | 19.119 | 17.872 | 1.065 s |

Aggregate window throughput is monotonic through C=3: C=2 is 15.9% above C=1
and C=3 is 40.3% above C=1. Against the prior exact-width result, KDA batching
raises C=2 from 12.820 to 14.760 tok/s and C=3 from 13.136 to 17.872 tok/s. A
normal-mode C=3 A/B measured 16.003 tok/s with batched FFN disabled, so grouped
width-three FFN contributes another 11.7%; the rest comes from KDA projection
and hyper-connection batching. Before exact-width EP dispatch, C=3 was 10.571
tok/s because it executed the padded fourth row.

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

If either rank exits during model load, remove both Atlas containers before a
retry. Do not configure a Docker restart policy: repeatedly reloading a model
under unified-memory pressure can make both Sparks unreachable.
