#!/usr/bin/env bash
# Safe GLM-5.3-Flash-NVFP4 bring-up on two DGX Sparks using EP2, optionally
# composed with overlapping TP2 on the same two ranks.
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
TP_SIZE="${TP_SIZE:-1}"
OOM_GUARD_MB="${OOM_GUARD_MB:-4096}"
MODEL_MOUNT="${MODEL_MOUNT:-}"
CONTAINER_MEMORY="${CONTAINER_MEMORY:-114g}"
NCCL_IFNAME="${NCCL_IFNAME:-enp1s0f1np1}"
NCCL_HCA="${NCCL_HCA:-rocep1s0f1}"
PROFILE="${PROFILE:-0}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-}"
KDA_REGRESIDENT_PREFILL="${KDA_REGRESIDENT_PREFILL:-1}"
UNIFIED_MOE_LAYOUT="${UNIFIED_MOE_LAYOUT:-1}"
CUBLAS_GEMM="${CUBLAS_GEMM:-1}"
HC_CUBLAS_PREFILL="${HC_CUBLAS_PREFILL:-1}"
NVFP4_GATE_UP_M128="${NVFP4_GATE_UP_M128:-0}"
NVFP4_DOWN_M32="${NVFP4_DOWN_M32:-0}"
NVFP4_PREQUANT_MOE="${NVFP4_PREQUANT_MOE:-0}"
NVFP4_FUSED_SILU_QUANT="${NVFP4_FUSED_SILU_QUANT:-0}"
NVFP4_MMQ_MOE="${NVFP4_MMQ_MOE:-0}"
NVFP4_CUTLASS_MOE="${NVFP4_CUTLASS_MOE:-0}"
FP4_PREFILL="${FP4_PREFILL:-0}"
MOE_PREFILL_EXACT_TILES="${MOE_PREFILL_EXACT_TILES:-1}"
MOE_PREFILL_MAX_LOAD_FACTOR="${MOE_PREFILL_MAX_LOAD_FACTOR:-8}"
DUMP_EXPERT_IDS="${DUMP_EXPERT_IDS:-0}"

if (( MAX_SEQ_LEN > 2048 )); then
  echo "ERROR: initial GLM-5.3 Atlas support is capped at 2048 tokens." >&2
  echo "Its sparse-attention layers use exact dense attention only while seq_len <= index_topk (2048)." >&2
  exit 2
fi

if [[ "$TP_SIZE" != "1" && "$TP_SIZE" != "2" ]]; then
  echo "ERROR: dual-Spark GLM-5.3 supports TP_SIZE=1 or overlapping TP_SIZE=2." >&2
  exit 2
fi

if [[ "$FP4_PREFILL" != "0" && "$FP4_PREFILL" != "1" ]]; then
  echo "ERROR: FP4_PREFILL must be 0 or 1." >&2
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

OPTIONAL_ENV=()
if [[ "$FP4_PREFILL" == "1" ]]; then
  OPTIONAL_ENV=(-e ATLAS_FP4_PREFILL=1)
fi

COMMON_SERVE_ARGS=(
  serve "$MODEL"
  --world-size 2
  --tp-size "$TP_SIZE"
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
echo "  parallelism: TP=$TP_SIZE / EP=2 on two physical ranks"
echo "  GPU budget: $GPU_MEM_UTIL; OOM guard: ${OOM_GUARD_MB} MiB"
echo "  container memory ceiling: $CONTAINER_MEMORY"
echo "  NCCL: $NCCL_IFNAME / $NCCL_HCA"
echo "  profiler: $PROFILE"
echo "  KDA register-resident prefill: $KDA_REGRESIDENT_PREFILL"
echo "  unified MoE layout: $UNIFIED_MOE_LAYOUT"
echo "  cuBLASLt BF16 projections: $CUBLAS_GEMM"
echo "  cuBLASLt TF32 mHC prefill: $HC_CUBLAS_PREFILL"
echo "  NVFP4 MoE gate/up M128: $NVFP4_GATE_UP_M128"
echo "  NVFP4 MoE down M32: $NVFP4_DOWN_M32"
echo "  NVFP4 MoE prequant FP4: $NVFP4_PREQUANT_MOE"
echo "  NVFP4 fused SiLU + quant: $NVFP4_FUSED_SILU_QUANT"
echo "  NVFP4 equal-memory grouped MMQ: $NVFP4_MMQ_MOE"
echo "  NVFP4 grouped CUTLASS MoE: $NVFP4_CUTLASS_MOE"
echo "  dense FFN native FP4 prefill: $FP4_PREFILL"
echo "  MoE exact grid / fallback load factor: $MOE_PREFILL_EXACT_TILES / $MOE_PREFILL_MAX_LOAD_FACTOR"
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
REMOTE_OPTIONAL_ENV=""
if (( ${#OPTIONAL_ENV[@]} )); then
  printf -v REMOTE_OPTIONAL_ENV '%q ' "${OPTIONAL_ENV[@]}"
fi
printf -v REMOTE_SERVE '%q ' "${COMMON_SERVE_ARGS[@]}"
ssh "$SSH_TARGET" "docker run -d \
  --name atlas-glm53-ep1 --gpus all --ipc=host --network host \
  --memory $CONTAINER_MEMORY --memory-swap $CONTAINER_MEMORY \
  $REMOTE_RDMA $REMOTE_NCCL $REMOTE_OPTIONAL_ENV -e RUST_LOG=info \
  -e ATLAS_KDA_REGRESIDENT_PREFILL=$KDA_REGRESIDENT_PREFILL \
  -e ATLAS_UNIFIED_MOE_LAYOUT=$UNIFIED_MOE_LAYOUT \
  -e ATLAS_CUBLAS_GEMM=$CUBLAS_GEMM \
  -e ATLAS_HC_CUBLAS_PREFILL=$HC_CUBLAS_PREFILL \
  -e ATLAS_NVFP4_GATE_UP_M128=$NVFP4_GATE_UP_M128 \
  -e ATLAS_NVFP4_DOWN_M32=$NVFP4_DOWN_M32 \
  -e ATLAS_NVFP4_PREQUANT_MOE=$NVFP4_PREQUANT_MOE \
  -e ATLAS_NVFP4_FUSED_SILU_QUANT=$NVFP4_FUSED_SILU_QUANT \
  -e ATLAS_NVFP4_MMQ_MOE=$NVFP4_MMQ_MOE \
  -e ATLAS_MOE_GROUPED_CUTLASS=$NVFP4_CUTLASS_MOE \
  -e ATLAS_MOE_PREFILL_EXACT_TILES=$MOE_PREFILL_EXACT_TILES \
  -e ATLAS_MOE_PREFILL_MAX_LOAD_FACTOR=$MOE_PREFILL_MAX_LOAD_FACTOR \
  -e ATLAS_DUMP_EXPERT_IDS=$DUMP_EXPERT_IDS \
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
  "${OPTIONAL_ENV[@]}" \
  -e RUST_LOG=info \
  -e ATLAS_KDA_REGRESIDENT_PREFILL="$KDA_REGRESIDENT_PREFILL" \
  -e ATLAS_UNIFIED_MOE_LAYOUT="$UNIFIED_MOE_LAYOUT" \
  -e ATLAS_CUBLAS_GEMM="$CUBLAS_GEMM" \
  -e ATLAS_HC_CUBLAS_PREFILL="$HC_CUBLAS_PREFILL" \
  -e ATLAS_NVFP4_GATE_UP_M128="$NVFP4_GATE_UP_M128" \
  -e ATLAS_NVFP4_DOWN_M32="$NVFP4_DOWN_M32" \
  -e ATLAS_NVFP4_PREQUANT_MOE="$NVFP4_PREQUANT_MOE" \
  -e ATLAS_NVFP4_FUSED_SILU_QUANT="$NVFP4_FUSED_SILU_QUANT" \
  -e ATLAS_NVFP4_MMQ_MOE="$NVFP4_MMQ_MOE" \
  -e ATLAS_MOE_GROUPED_CUTLASS="$NVFP4_CUTLASS_MOE" \
  -e ATLAS_MOE_PREFILL_EXACT_TILES="$MOE_PREFILL_EXACT_TILES" \
  -e ATLAS_MOE_PREFILL_MAX_LOAD_FACTOR="$MOE_PREFILL_MAX_LOAD_FACTOR" \
  -e ATLAS_DUMP_EXPERT_IDS="$DUMP_EXPERT_IDS" \
  "${MOUNT_FLAGS[@]}" \
  "$IMAGE" "${COMMON_SERVE_ARGS[@]}" --rank 0 --port "$PORT"

echo "Rank 0 logs: docker logs -f atlas-glm53-ep0"
echo "Rank 1 logs: ssh $SSH_TARGET 'docker logs -f atlas-glm53-ep1'"
echo "API (head node): http://127.0.0.1:$PORT/v1"
