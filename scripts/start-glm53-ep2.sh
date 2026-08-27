#!/usr/bin/env bash
# Safe initial GLM-5.3-Flash-NVFP4 bring-up on two DGX Sparks using pure EP.
#
# Usage:
#   HEAD_IP=169.254.179.82 WORKER_IP=169.254.128.113 \
#     MODEL=/var/tmp/models/glm53-flash-nvfp4 ./scripts/start-glm53-ep2.sh
#
# Raise MAX_SEQ_LEN only after a stable load and measured free-memory check.

set -euo pipefail

MODEL="${MODEL:-LibertAIDAI/GLM-5.3-Flash-NVFP4}"
IMAGE="${IMAGE:-atlas-glm53-flash:latest}"
HEAD_IP="${HEAD_IP:-127.0.0.1}"
WORKER_IP="${WORKER_IP:-127.0.0.1}"
SSH_TARGET="${SSH_TARGET:-$WORKER_IP}"
MASTER_PORT="${MASTER_PORT:-29500}"
PORT="${PORT:-8888}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
OOM_GUARD_MB="${OOM_GUARD_MB:-4096}"
MODEL_MOUNT="${MODEL_MOUNT:-}"
CONTAINER_MEMORY="${CONTAINER_MEMORY:-114g}"
NCCL_IFNAME="${NCCL_IFNAME:-enp1s0f1np1}"
NCCL_HCA="${NCCL_HCA:-rocep1s0f1}"
PROFILE="${PROFILE:-0}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-}"

if (( MAX_SEQ_LEN > 2048 )); then
  echo "ERROR: initial GLM-5.3 Atlas support is capped at 2048 tokens." >&2
  echo "Its sparse-attention layers use exact dense attention only while seq_len <= index_topk (2048)." >&2
  exit 2
fi

if [[ "$HEAD_IP" == "127.0.0.1" || "$WORKER_IP" == "127.0.0.1" ]]; then
  echo "ERROR: set HEAD_IP and WORKER_IP to the two Sparks' fabric addresses." >&2
  exit 2
fi

if [[ -z "$MODEL_MOUNT" && "$MODEL" == /* ]]; then
  MODEL_MOUNT="$MODEL"
fi

MOUNT_FLAGS=()
if [[ -n "$MODEL_MOUNT" ]]; then
  MOUNT_FLAGS=(-v "$MODEL_MOUNT:$MODEL_MOUNT:ro")
fi

RDMA_FLAGS=(
  --device=/dev/infiniband
  --cap-add=IPC_LOCK
  --cap-add=SYS_NICE
  --ulimit memlock=-1
  --security-opt seccomp=unconfined
)

NCCL_ENV=(
  -e NCCL_SOCKET_IFNAME="$NCCL_IFNAME"
  -e NCCL_NET=IB
  -e NCCL_IB_DISABLE=0
  -e NCCL_IB_HCA="$NCCL_HCA"
  -e NCCL_IB_GID_INDEX=3
  -e NCCL_IB_ROCE_VERSION_NUM=2
  -e NCCL_IB_ADDR_FAMILY=AF_INET
  -e NCCL_IB_TIMEOUT=22
  -e NCCL_IB_RETRY_CNT=7
  -e NCCL_NET_GDR_LEVEL=0
  -e NCCL_NET_GDR_C2C=0
  -e NCCL_DMABUF_ENABLE=0
  -e NCCL_NVLS_ENABLE=0
  -e NCCL_CUMEM_HOST_ENABLE=0
  -e NCCL_CUMEM_ENABLE=0
  -e NCCL_PROTO=Simple
  -e NCCL_ALGO=Ring
  -e NCCL_BUFFSIZE=33554432
  -e NCCL_MIN_NCHANNELS=1
  -e NCCL_MAX_NCHANNELS=2
  -e NCCL_DEBUG=WARN
)

COMMON_SERVE_ARGS=(
  serve "$MODEL"
  --world-size 2
  --tp-size 1
  --ep-size 2
  --master-addr "$HEAD_IP"
  --master-port "$MASTER_PORT"
  --max-seq-len "$MAX_SEQ_LEN"
  --max-prefill-tokens "$MAX_SEQ_LEN"
  --max-batch-size 1
  --max-num-seqs 1
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --kv-cache-dtype bf16
  --oom-guard-mb "$OOM_GUARD_MB"
)
if [[ "$PROFILE" == "1" ]]; then
  COMMON_SERVE_ARGS+=(--profile)
fi
if [[ -n "$TOOL_CALL_PARSER" ]]; then
  COMMON_SERVE_ARGS+=(--tool-call-parser "$TOOL_CALL_PARSER")
fi

echo "Atlas GLM-5.3 dual-Spark safe bring-up"
echo "  model: $MODEL"
echo "  image: $IMAGE"
echo "  context/concurrency: $MAX_SEQ_LEN / 1"
echo "  GPU budget: $GPU_MEM_UTIL; OOM guard: ${OOM_GUARD_MB} MiB"
echo "  container memory ceiling: $CONTAINER_MEMORY"
echo "  NCCL: $NCCL_IFNAME / $NCCL_HCA"
echo "  profiler: $PROFILE"
echo "  tool-call parser override: ${TOOL_CALL_PARSER:-model default}"

# Never leave one stale rank in an old communicator.
docker rm -f atlas-glm53-ep0 2>/dev/null || true
ssh "$SSH_TARGET" "docker rm -f atlas-glm53-ep1 2>/dev/null || true"

# Worker first: it waits for rank 0 without exposing a half-initialized API.
REMOTE_MOUNT=""
if (( ${#MOUNT_FLAGS[@]} )); then
  printf -v REMOTE_MOUNT '%q ' "${MOUNT_FLAGS[@]}"
fi
printf -v REMOTE_RDMA '%q ' "${RDMA_FLAGS[@]}"
printf -v REMOTE_NCCL '%q ' "${NCCL_ENV[@]}"
printf -v REMOTE_SERVE '%q ' "${COMMON_SERVE_ARGS[@]}"
ssh "$SSH_TARGET" "docker run -d \
  --name atlas-glm53-ep1 --gpus all --ipc=host --network host \
  --memory $CONTAINER_MEMORY --memory-swap $CONTAINER_MEMORY \
  $REMOTE_RDMA $REMOTE_NCCL -e RUST_LOG=info \
  $REMOTE_MOUNT $IMAGE $REMOTE_SERVE --rank 1 --port 0"

docker run -d \
  --name atlas-glm53-ep0 \
  --gpus all \
  --ipc=host \
  --network host \
  --memory "$CONTAINER_MEMORY" \
  --memory-swap "$CONTAINER_MEMORY" \
  "${RDMA_FLAGS[@]}" \
  "${NCCL_ENV[@]}" \
  -e RUST_LOG=info \
  "${MOUNT_FLAGS[@]}" \
  "$IMAGE" "${COMMON_SERVE_ARGS[@]}" --rank 0 --port "$PORT"

echo "Rank 0 logs: docker logs -f atlas-glm53-ep0"
echo "Rank 1 logs: ssh $SSH_TARGET 'docker logs -f atlas-glm53-ep1'"
echo "API (head node): http://127.0.0.1:$PORT/v1"
