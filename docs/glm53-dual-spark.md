# GLM-5.3-Flash-NVFP4 on two DGX Sparks

Atlas support targets the exact `LibertAIDAI/GLM-5.3-Flash-NVFP4` checkpoint and
uses pure expert parallelism (`TP=1`, `EP=2`). Routed experts are split between
the two ranks; attention, KDA, shared-expert, and dense weights are replicated.

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
./scripts/start-glm53-ep2.sh
```

The launcher intentionally starts with:

- 1,024 maximum sequence tokens;
- one batch item and one concurrent sequence;
- BF16 KV cache;
- a 92% total GPU-memory budget and 4 GiB OOM guard;
- a 114 GiB container memory ceiling, leaving host headroom if loading runs away;
- no speculative decoding or prefix cache;
- the worker rank first and no automatic restart policy.

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
repeated at 9.52 decode tokens/s. These are bring-up receipts from two DGX
Sparks, not general performance claims.

If either rank exits during model load, remove both Atlas containers before a
retry. Do not configure a Docker restart policy: repeatedly reloading a model
under unified-memory pressure can make both Sparks unreachable.
