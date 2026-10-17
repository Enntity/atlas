#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Launch Atlas GLM-5.3-Flash EXL3 (mul1) TP=2 serving on port 8000.
#
# Takes over the production port from the stopped vLLM containers
# (glm53-exl3-head / glm53-exl3-worker are preserved as fallback; this script
# never touches them). Serve profile = Atlas validated defaults: BF16 KV,
# eager/no decode graphs, EP=1, max context 4096, prefill chunk 128,
# serial single-sequence admission (C1). See
# docs/design/exl3-support.md for what is and is not qualified.
#
# Usage (spark0): RANK=0 bash scripts/start-glm53-exl3-tp2.sh
#        (spark1): RANK=1 bash scripts/start-glm53-exl3-tp2.sh
set -euo pipefail

RANK="${RANK:?set RANK=0 (spark0/head) or 1 (spark1/worker)}"
IMAGE="${IMAGE:-atlas-exl3-validation:src-48624a8605de-rust1.97.1}"
NAME="${NAME:-glm53-exl3-atlas-${RANK}}"
PORT="${PORT:-8000}"
MASTER_ADDR="${MASTER_ADDR:-192.168.100.1}"
MASTER_PORT="${MASTER_PORT:-29721}"
MODEL_PATH="${MODEL_PATH:-/root/.cache/huggingface/hub/models--MikeRoz--GLM-5.3-Flash-Uncensored-4.05bpw-h6-exl3/snapshots/dda99c7e935920fa6e235ce410e6f806e1c02720}"
HF_CACHE="${HF_CACHE:-/home/colum/glm53-local}"
CUDA_CACHE="${CUDA_CACHE:-/home/colum/.cache/atlas-exl3-validation/src-48624a8605de}"
# KDA arithmetic policy: 0 = Atlas correctly rounded core (default),
# 1 = EXL3 reference-policy core (requires an image with module kda_fla).
KDA_FLA_REFERENCE="${KDA_FLA_REFERENCE:-0}"

case "$RANK" in
  0) NIC_ETH=enp1s0f0np0; NIC_IB=rocep1s0f0 ;;
  1) NIC_ETH=enp1s0f1np1; NIC_IB=rocep1s0f1 ;;
  *) echo "RANK must be 0 or 1" >&2; exit 1 ;;
esac

docker rm -f "$NAME" 2>/dev/null || true
docker create \
  --name "$NAME" \
  --restart unless-stopped \
  --network host \
  --ipc host \
  --gpus all \
  --device /dev/infiniband:/dev/infiniband:rwm \
  --cap-add CAP_IPC_LOCK \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --env NCCL_CUMEM_ENABLE=0 \
  --env "NCCL_IB_HCA=$NIC_IB" \
  --env NCCL_IGNORE_CPU_AFFINITY=1 \
  --env NCCL_IB_ROCE_VERSION_NUM=2 \
  --env NCCL_IB_MERGE_NICS=0 \
  --env NCCL_DEBUG=WARN \
  --env NCCL_IB_DISABLE=0 \
  --env NCCL_NET=IB \
  --env NCCL_NVLS_ENABLE=0 \
  --env NCCL_IB_GID_INDEX=3 \
  --env "NCCL_SOCKET_IFNAME=$NIC_ETH" \
  --env NCCL_NET_PLUGIN=none \
  --env NCCL_CROSS_NIC=0 \
  --env "GLOO_SOCKET_IFNAME=$NIC_ETH" \
  --env CUDA_CACHE_PATH=/tmp/cuda-driver-cache \
  --volume "$CUDA_CACHE:/tmp/cuda-driver-cache" \
  --env RUST_LOG=info \
  --env ATLAS_NO_DECODE_GRAPHS=1 \
  --env ATLAS_NO_DECODE_GRAPHS_MULTISEQ=1 \
  --env ATLAS_DEBUG_NO_GRAPH=1 \
  --env "ATLAS_KDA_FLA_REFERENCE=$KDA_FLA_REFERENCE" \
  --volume "$HF_CACHE:/root/.cache/huggingface:ro" \
  "$IMAGE" \
  serve "$MODEL_PATH" \
  --world-size 2 \
  --tp-size 2 \
  --ep-size 1 \
  --rank "$RANK" \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  --bind 0.0.0.0 \
  --port "$PORT" \
  --max-seq-len 4096 \
  --max-prefill-tokens 128 \
  --max-batch-size 1 \
  --max-num-seqs 1 \
  --kv-cache-dtype bf16 \
  --lm-head-dtype default \
  --language-model-only \
  --gpu-memory-utilization 0.84 \
  --swap-space-gb 0 \
  --oom-guard-mb 1024 \
  --request-timeout 1200

docker start "$NAME"
echo "started $NAME (rank $RANK, image $IMAGE, port $PORT)"
