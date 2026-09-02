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

The KDA register-resident prefill path, unified MoE layout, and cuBLASLt
projection dispatch are enabled by default. Their diagnostic fallbacks are
`KDA_REGRESIDENT_PREFILL=0`, `UNIFIED_MOE_LAYOUT=0`, and `CUBLAS_GEMM=0`.
KDA Q/K/V and row-parallel output projections keep independent decode-native
NVFP4 weights plus transposed M=128 prefill twins. The output twins add about
0.3 GiB per rank across all 34 KDA layers without changing decode numerics.

This checkpoint declares 1M model context, but initial Atlas support does not.
GLM's full-attention layers select 2,048 tokens using an indexer. Atlas currently
uses causally masked dense attention for those layers, which is mathematically
equivalent while the entire sequence is no longer than `index_topk=2048`.
Startup therefore rejects values above 2,048 rather than silently changing the
model. KDA recurrent state is FP32 and allocated for one sequence; increasing
concurrency multiplies that state and should follow a measured memory audit.

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
| Unified layout + prequantized NVFP4 activation path (v18) | **834.67** | 12.7–12.9 | Current prefill baseline |
| Equal-memory grouped MMQ (v19) | 795.1 | 14.30 | Decode gain, prefill regression; experimental only |
| Native grouped CUTLASS NVFP4 (v23) | 725.57 | **14.67** | Decode gain, prefill regression; experimental only |
| CUTLASS with reused exact-tile offset snapshot (v24) | 723.14 | 14.62 | Neutral; confirms the extra D2H was not the bottleneck |

The external comparison target is approximately 1,500 prefill tok/s and 20
decode tok/s. Optional MMQ and CUTLASS routes remain disabled by default; they
are diagnostic branches, not recommended launch settings. These are controlled
receipts from two DGX Sparks, not general performance claims.

If either rank exits during model load, remove both Atlas containers before a
retry. Do not configure a Docker restart policy: repeatedly reloading a model
under unified-memory pressure can make both Sparks unreachable.
