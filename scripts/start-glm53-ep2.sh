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
BIND_ADDRESS="${BIND_ADDRESS:-0.0.0.0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
# Keep scratch and the live chunk bounded when only the context window is
# raised. The measured dual-Spark sweet spot is 6144 tokens, capped to the
# configured sequence limit so the 1024-token safe-first-launch still works.
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-$(( MAX_SEQ_LEN < 6144 ? MAX_SEQ_LEN : 6144 ))}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
TP_SIZE="${TP_SIZE:-1}"
OOM_GUARD_MB="${OOM_GUARD_MB:-4096}"
# The server deadline includes chunked prefill. A 100K request currently takes
# longer than Atlas's 300-second default on two Sparks, so expose the existing
# serve flag without weakening the default for short-context deployments.
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
MODEL_MOUNT="${MODEL_MOUNT:-}"
CONTAINER_MEMORY="${CONTAINER_MEMORY:-114g}"
NCCL_IFNAME="${NCCL_IFNAME:-enp1s0f1np1}"
NCCL_HCA="${NCCL_HCA:-rocep1s0f1}"
PROFILE="${PROFILE:-0}"
MS_PROFILE="${MS_PROFILE:-0}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-}"
KDA_REGRESIDENT_PREFILL="${KDA_REGRESIDENT_PREFILL:-1}"
KDA_MULTI_SEQ="${KDA_MULTI_SEQ:-1}"
KDA_BATCHED_FFN="${KDA_BATCHED_FFN:-1}"
KDA_MS_PROFILE="${KDA_MS_PROFILE:-0}"
MLA_MULTI_SEQ="${MLA_MULTI_SEQ:-1}"
UNIFIED_MOE_LAYOUT="${UNIFIED_MOE_LAYOUT:-1}"
CUBLAS_GEMM="${CUBLAS_GEMM:-1}"
HC_CUBLAS_PREFILL="${HC_CUBLAS_PREFILL:-1}"
NVFP4_GATE_UP_M128="${NVFP4_GATE_UP_M128:-0}"
NVFP4_DOWN_M32="${NVFP4_DOWN_M32:-0}"
# The GLM routed-MoE path benefits from quantizing each sorted activation tile
# once and consuming it directly in the native W4A4 gate/up and down kernels.
NVFP4_PREQUANT_MOE="${NVFP4_PREQUANT_MOE:-1}"
NVFP4_VECSCALE="${NVFP4_VECSCALE:-1}"
NVFP4_FUSED_SILU_QUANT="${NVFP4_FUSED_SILU_QUANT:-1}"
NVFP4_MMQ_MOE="${NVFP4_MMQ_MOE:-0}"
NVFP4_CUTLASS_MOE="${NVFP4_CUTLASS_MOE:-0}"
FP4_PREFILL="${FP4_PREFILL:-0}"
MOE_PREFILL_EXACT_TILES="${MOE_PREFILL_EXACT_TILES:-1}"
MOE_PREFILL_MAX_LOAD_FACTOR="${MOE_PREFILL_MAX_LOAD_FACTOR:-8}"
DUMP_EXPERT_IDS="${DUMP_EXPERT_IDS:-0}"
SPECULATIVE="${SPECULATIVE:-0}"
NUM_DRAFTS="${NUM_DRAFTS:-1}"
# GLM's shipped template enters <think> for ordinary chat requests. Atlas's
# generic safety gate otherwise keeps speculative decode out of that phase,
# which would leave MTP idle for most benchmark and agent workloads.
MTP_SPEC_THINK="${MTP_SPEC_THINK:-1}"
MTP_GATE_FORCE="${MTP_GATE_FORCE:-0}"
# Per-request K=3/K=5 selection. Resolve its default after the compact-K5
# lever below: the compact native-FP4 verifier made K=5 cheaper than K=3 on
# dual GB10, so stepping down is no longer economical in that configuration.
MTP_SINGLE_DEPTH_ADAPT="${MTP_SINGLE_DEPTH_ADAPT:-}"
MTP_TIMING="${MTP_TIMING:-0}"
MTP_ACCEPT_DEBUG="${MTP_ACCEPT_DEBUG:-0}"
VERIFY_PROFILE="${VERIFY_PROFILE:-0}"
GLM_INDEX_PROFILE="${GLM_INDEX_PROFILE:-0}"
GLM_INDEX_ROW_GROUP="${GLM_INDEX_ROW_GROUP:-8}"
GLM_SPARSE_HEAD_GROUP="${GLM_SPARSE_HEAD_GROUP:-8}"
MOE_UNION_STATS="${MOE_UNION_STATS:-0}"
GLM_K5_GROUPED_MOE="${GLM_K5_GROUPED_MOE:-1}"
GLM_K5_BATCHED_SHARED="${GLM_K5_BATCHED_SHARED:-0}"
GLM_K5_FUSED_SHARED_GATE_UP="${GLM_K5_FUSED_SHARED_GATE_UP:-1}"
GLM_K5_DENSE_EXACT="${GLM_K5_DENSE_EXACT:-1}"
GLM_K5_FUSED_DENSE_PAIRS="${GLM_K5_FUSED_DENSE_PAIRS:-1}"
GLM_K5_FUSED_DENSE_TRIPLE="${GLM_K5_FUSED_DENSE_TRIPLE:-1}"
GLM_K5_COMPACT_MOE="${GLM_K5_COMPACT_MOE:-0}"
GLM_K5_FUSED_COMPACT_GATE_UP="${GLM_K5_FUSED_COMPACT_GATE_UP:-0}"
GLM_K5_HC_CUBLAS="${GLM_K5_HC_CUBLAS:-1}"
# The appended predictor rereads its 4096x8192 input combiner once per draft.
# Keep a compact NVFP4 decode copy; BF16 remains resident for batched KV prefill.
GLM_MTP_NVFP4_EH="${GLM_MTP_NVFP4_EH:-1}"
GLM_K5_BATCHED_CONV_SNAPSHOT="${GLM_K5_BATCHED_CONV_SNAPSHOT:-1}"
GLM_K5_BATCHED_RECURRENT_SNAPSHOT="${GLM_K5_BATCHED_RECURRENT_SNAPSHOT:-1}"
GLM_K5_FUSED_QKV="${GLM_K5_FUSED_QKV:-1}"
if [[ -z "$MTP_SINGLE_DEPTH_ADAPT" ]]; then
  if [[ "$GLM_K5_COMPACT_MOE" == "1" ]]; then
    MTP_SINGLE_DEPTH_ADAPT=0
  else
    MTP_SINGLE_DEPTH_ADAPT=1
  fi
fi
MOE_SHARED_REDUCE_OVERLAP="${MOE_SHARED_REDUCE_OVERLAP:-1}"
GLM_MTP_SERIAL_PREFILL="${GLM_MTP_SERIAL_PREFILL:-0}"
# MTP proposals need the appended layer's prompt K/V history. Build it with
# the batched KV-only path by default whenever speculative decode is enabled.
# The serial path remains an explicit correctness/debugging oracle.
GLM_MTP_BATCHED_PREFILL="${GLM_MTP_BATCHED_PREFILL:-$SPECULATIVE}"

MODEL_MAX_SEQ_LEN=1048576
if (( MAX_SEQ_LEN > MODEL_MAX_SEQ_LEN )); then
  echo "ERROR: MAX_SEQ_LEN exceeds GLM-5.3's ${MODEL_MAX_SEQ_LEN}-token model limit." >&2
  exit 2
fi

if (( MAX_PREFILL_TOKENS < 1 || MAX_PREFILL_TOKENS > MAX_SEQ_LEN )); then
  echo "ERROR: MAX_PREFILL_TOKENS must be in 1..MAX_SEQ_LEN." >&2
  exit 2
fi

if [[ ! "$REQUEST_TIMEOUT" =~ ^[0-9]+$ ]]; then
  echo "ERROR: REQUEST_TIMEOUT must be a non-negative integer number of seconds." >&2
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

if [[ "$MS_PROFILE" != "0" && "$MS_PROFILE" != "1" ]]; then
  echo "ERROR: MS_PROFILE must be 0 or 1." >&2
  exit 2
fi

if [[ "$KDA_MULTI_SEQ" != "0" && "$KDA_MULTI_SEQ" != "1" ]]; then
  echo "ERROR: KDA_MULTI_SEQ must be 0 or 1." >&2
  exit 2
fi
if [[ "$KDA_BATCHED_FFN" != "0" && "$KDA_BATCHED_FFN" != "1" ]]; then
  echo "ERROR: KDA_BATCHED_FFN must be 0 or 1." >&2
  exit 2
fi
if [[ "$KDA_MS_PROFILE" != "0" && "$KDA_MS_PROFILE" != "1" ]]; then
  echo "ERROR: KDA_MS_PROFILE must be 0 or 1." >&2
  exit 2
fi
if [[ "$MLA_MULTI_SEQ" != "0" && "$MLA_MULTI_SEQ" != "1" ]]; then
  echo "ERROR: MLA_MULTI_SEQ must be 0 or 1." >&2
  exit 2
fi

if (( MAX_BATCH_SIZE < 1 || MAX_BATCH_SIZE > 3 )); then
  echo "ERROR: validated GLM-5 dual-Spark MAX_BATCH_SIZE range is 1..3." >&2
  exit 2
fi

if [[ "$SPECULATIVE" != "0" && "$SPECULATIVE" != "1" ]]; then
  echo "ERROR: SPECULATIVE must be 0 or 1." >&2
  exit 2
fi
if [[ "$SPECULATIVE" == "1" && "$MAX_BATCH_SIZE" != "1" ]]; then
  echo "ERROR: initial GLM-5 MTP validation requires MAX_BATCH_SIZE=1." >&2
  exit 2
fi
if [[ "$MTP_SPEC_THINK" != "0" && "$MTP_SPEC_THINK" != "1" ]]; then
  echo "ERROR: MTP_SPEC_THINK must be 0 or 1." >&2
  exit 2
fi
if [[ "$MTP_GATE_FORCE" != "0" && "$MTP_GATE_FORCE" != "1" ]]; then
  echo "ERROR: MTP_GATE_FORCE must be 0 or 1." >&2
  exit 2
fi
if [[ "$MTP_SINGLE_DEPTH_ADAPT" != "0" && "$MTP_SINGLE_DEPTH_ADAPT" != "1" ]]; then
  echo "ERROR: MTP_SINGLE_DEPTH_ADAPT must be 0 or 1." >&2
  exit 2
fi
if [[ "$MTP_TIMING" != "0" && "$MTP_TIMING" != "1" ]]; then
  echo "ERROR: MTP_TIMING must be 0 or 1." >&2
  exit 2
fi
if [[ "$MTP_ACCEPT_DEBUG" != "0" && "$MTP_ACCEPT_DEBUG" != "1" ]]; then
  echo "ERROR: MTP_ACCEPT_DEBUG must be 0 or 1." >&2
  exit 2
fi
if [[ "$VERIFY_PROFILE" != "0" && "$VERIFY_PROFILE" != "1" ]]; then
  echo "ERROR: VERIFY_PROFILE must be 0 or 1." >&2
  exit 2
fi
if [[ "$GLM_INDEX_PROFILE" != "0" && "$GLM_INDEX_PROFILE" != "1" ]]; then
  echo "ERROR: GLM_INDEX_PROFILE must be 0 or 1." >&2
  exit 2
fi
if [[ "$GLM_INDEX_ROW_GROUP" != "1" && "$GLM_INDEX_ROW_GROUP" != "8" ]]; then
  echo "ERROR: GLM_INDEX_ROW_GROUP must be 1 or 8." >&2
  exit 2
fi
if [[ "$GLM_SPARSE_HEAD_GROUP" != "1" && "$GLM_SPARSE_HEAD_GROUP" != "8" ]]; then
  echo "ERROR: GLM_SPARSE_HEAD_GROUP must be 1 or 8." >&2
  exit 2
fi
if [[ "$MOE_UNION_STATS" != "0" && "$MOE_UNION_STATS" != "1" ]]; then
  echo "ERROR: MOE_UNION_STATS must be 0 or 1." >&2
  exit 2
fi
if [[ "$GLM_K5_GROUPED_MOE" != "0" && "$GLM_K5_GROUPED_MOE" != "1" ]]; then
  echo "ERROR: GLM_K5_GROUPED_MOE must be 0 or 1." >&2
  exit 2
fi
if [[ "$GLM_K5_COMPACT_MOE" != "0" && "$GLM_K5_COMPACT_MOE" != "1" ]]; then
  echo "ERROR: GLM_K5_COMPACT_MOE must be 0 or 1." >&2
  exit 2
fi
if [[ "$GLM_K5_FUSED_SHARED_GATE_UP" != "0" && "$GLM_K5_FUSED_SHARED_GATE_UP" != "1" ]]; then
  echo "ERROR: GLM_K5_FUSED_SHARED_GATE_UP must be 0 or 1." >&2
  exit 2
fi
if [[ "$GLM_K5_DENSE_EXACT" != "0" && "$GLM_K5_DENSE_EXACT" != "1" ]]; then
  echo "ERROR: GLM_K5_DENSE_EXACT must be 0 or 1." >&2
  exit 2
fi
if [[ "$GLM_K5_FUSED_DENSE_PAIRS" != "0" && "$GLM_K5_FUSED_DENSE_PAIRS" != "1" ]]; then
  echo "ERROR: GLM_K5_FUSED_DENSE_PAIRS must be 0 or 1." >&2
  exit 2
fi
if [[ "$GLM_K5_FUSED_DENSE_TRIPLE" != "0" && "$GLM_K5_FUSED_DENSE_TRIPLE" != "1" ]]; then
  echo "ERROR: GLM_K5_FUSED_DENSE_TRIPLE must be 0 or 1." >&2
  exit 2
fi
if [[ "$GLM_K5_FUSED_COMPACT_GATE_UP" != "0" && "$GLM_K5_FUSED_COMPACT_GATE_UP" != "1" ]]; then
  echo "ERROR: GLM_K5_FUSED_COMPACT_GATE_UP must be 0 or 1." >&2
  exit 2
fi
if [[ "$GLM_K5_HC_CUBLAS" != "0" && "$GLM_K5_HC_CUBLAS" != "1" ]]; then
  echo "ERROR: GLM_K5_HC_CUBLAS must be 0 or 1." >&2
  exit 2
fi
if [[ "$GLM_K5_BATCHED_CONV_SNAPSHOT" != "0" && "$GLM_K5_BATCHED_CONV_SNAPSHOT" != "1" ]]; then
  echo "ERROR: GLM_K5_BATCHED_CONV_SNAPSHOT must be 0 or 1." >&2
  exit 2
fi
if [[ "$GLM_K5_BATCHED_RECURRENT_SNAPSHOT" != "0" && "$GLM_K5_BATCHED_RECURRENT_SNAPSHOT" != "1" ]]; then
  echo "ERROR: GLM_K5_BATCHED_RECURRENT_SNAPSHOT must be 0 or 1." >&2
  exit 2
fi
if [[ "$GLM_K5_FUSED_QKV" != "0" && "$GLM_K5_FUSED_QKV" != "1" ]]; then
  echo "ERROR: GLM_K5_FUSED_QKV must be 0 or 1." >&2
  exit 2
fi
if [[ "$MOE_SHARED_REDUCE_OVERLAP" != "0" && "$MOE_SHARED_REDUCE_OVERLAP" != "1" ]]; then
  echo "ERROR: MOE_SHARED_REDUCE_OVERLAP must be 0 or 1." >&2
  exit 2
fi
if [[ "$GLM_MTP_SERIAL_PREFILL" != "0" && "$GLM_MTP_SERIAL_PREFILL" != "1" ]]; then
  echo "ERROR: GLM_MTP_SERIAL_PREFILL must be 0 or 1." >&2
  exit 2
fi
if [[ "$GLM_MTP_BATCHED_PREFILL" != "0" && "$GLM_MTP_BATCHED_PREFILL" != "1" ]]; then
  echo "ERROR: GLM_MTP_BATCHED_PREFILL must be 0 or 1." >&2
  exit 2
fi
if [[ "$GLM_MTP_SERIAL_PREFILL" == "1" && "$GLM_MTP_BATCHED_PREFILL" == "1" ]]; then
  echo "ERROR: GLM_MTP_SERIAL_PREFILL and GLM_MTP_BATCHED_PREFILL are mutually exclusive." >&2
  exit 2
fi

if (( MAX_NUM_SEQS < MAX_BATCH_SIZE || MAX_NUM_SEQS > 5 )); then
  echo "ERROR: validated GLM-5 MAX_NUM_SEQS range is MAX_BATCH_SIZE..5." >&2
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

if (( MAX_BATCH_SIZE > 1 )); then
  NCCL_ENV+=(-e ATLAS_EP_PROTOCOL=v2)
fi
if [[ "$MS_PROFILE" == "1" ]]; then
  NCCL_ENV+=(-e ATLAS_MS_PROFILE=1)
fi

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
  --bind "$BIND_ADDRESS"
  --max-seq-len "$MAX_SEQ_LEN"
  --max-prefill-tokens "$MAX_PREFILL_TOKENS"
  --max-batch-size "$MAX_BATCH_SIZE"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --kv-cache-dtype bf16
  --oom-guard-mb "$OOM_GUARD_MB"
  --request-timeout "$REQUEST_TIMEOUT"
)
if [[ "$PROFILE" == "1" ]]; then
  COMMON_SERVE_ARGS+=(--profile)
fi
if [[ -n "$TOOL_CALL_PARSER" ]]; then
  COMMON_SERVE_ARGS+=(--tool-call-parser "$TOOL_CALL_PARSER")
fi
if [[ "$SPECULATIVE" == "1" ]]; then
  COMMON_SERVE_ARGS+=(--speculative --num-drafts "$NUM_DRAFTS")
fi

echo "Atlas GLM-5.3 dual-Spark safe bring-up"
echo "  model: $MODEL"
echo "  image: $IMAGE"
echo "  API bind: $BIND_ADDRESS:$PORT"
echo "  per-sequence context / active / admitted: $MAX_SEQ_LEN / $MAX_BATCH_SIZE / $MAX_NUM_SEQS"
echo "  prefill chunk budget: $MAX_PREFILL_TOKENS"
echo "  parallelism: TP=$TP_SIZE / EP=2 on two physical ranks"
echo "  GPU budget: $GPU_MEM_UTIL; OOM guard: ${OOM_GUARD_MB} MiB"
echo "  container memory ceiling: $CONTAINER_MEMORY"
echo "  NCCL: $NCCL_IFNAME / $NCCL_HCA"
echo "  profiler: $PROFILE"
echo "  multi-sequence phase profiler: $MS_PROFILE"
echo "  KDA register-resident prefill: $KDA_REGRESIDENT_PREFILL"
echo "  KDA multi-sequence decode: $KDA_MULTI_SEQ"
echo "  KDA batched FFN: $KDA_BATCHED_FFN"
echo "  GLM MLA multi-sequence decode: $MLA_MULTI_SEQ"
echo "  unified MoE layout: $UNIFIED_MOE_LAYOUT"
echo "  cuBLASLt BF16 projections: $CUBLAS_GEMM"
echo "  cuBLASLt TF32 mHC prefill: $HC_CUBLAS_PREFILL"
echo "  NVFP4 MoE gate/up M128: $NVFP4_GATE_UP_M128"
echo "  NVFP4 MoE down M32: $NVFP4_DOWN_M32"
echo "  NVFP4 MoE prequant FP4: $NVFP4_PREQUANT_MOE"
echo "  NVFP4 vectorized scale staging: $NVFP4_VECSCALE"
echo "  NVFP4 fused SiLU + quant: $NVFP4_FUSED_SILU_QUANT"
echo "  NVFP4 equal-memory grouped MMQ: $NVFP4_MMQ_MOE"
echo "  NVFP4 grouped CUTLASS MoE: $NVFP4_CUTLASS_MOE"
echo "  dense FFN native FP4 prefill: $FP4_PREFILL"
echo "  MoE exact grid / fallback load factor: $MOE_PREFILL_EXACT_TILES / $MOE_PREFILL_MAX_LOAD_FACTOR"
echo "  tool-call parser override: ${TOOL_CALL_PARSER:-model default}"
echo "  MTP speculative / draft tokens: $SPECULATIVE / $NUM_DRAFTS"
echo "  MTP during thinking / force gate: $MTP_SPEC_THINK / $MTP_GATE_FORCE"
echo "  MTP per-request K3/K5 adaptation: $MTP_SINGLE_DEPTH_ADAPT"
echo "  MTP phase timing: $MTP_TIMING"
echo "  MTP acceptance telemetry: $MTP_ACCEPT_DEBUG"
echo "  GLM verifier layer profile: $VERIFY_PROFILE"
echo "  GLM semantic-index profile: $GLM_INDEX_PROFILE"
echo "  semantic-index rows per CTA: $GLM_INDEX_ROW_GROUP"
echo "  sparse MLA heads per CTA: $GLM_SPARSE_HEAD_GROUP"
echo "  sampled MoE expert-union stats: $MOE_UNION_STATS"
echo "  GLM K5 grouped W4A16 MoE: $GLM_K5_GROUPED_MOE"
echo "  GLM K5 exact-M shared expert: $GLM_K5_BATCHED_SHARED"
echo "  GLM K5 fused shared gate/up: $GLM_K5_FUSED_SHARED_GATE_UP"
echo "  GLM K5 exact BF16 side projections: $GLM_K5_DENSE_EXACT"
echo "  GLM K5 fused BF16 projection pairs: $GLM_K5_FUSED_DENSE_PAIRS"
echo "  GLM K5 fused beta/f_a/g_a: $GLM_K5_FUSED_DENSE_TRIPLE"
echo "  GLM K5 compact native-FP4 MoE: $GLM_K5_COMPACT_MOE"
echo "  GLM K5 fused compact gate/up: $GLM_K5_FUSED_COMPACT_GATE_UP"
echo "  GLM K5 batched TF32 mHC: $GLM_K5_HC_CUBLAS"
echo "  GLM MTP decode-native NVFP4 eh_proj: $GLM_MTP_NVFP4_EH"
echo "  GLM K5 batched conv snapshots: $GLM_K5_BATCHED_CONV_SNAPSHOT"
echo "  GLM K5 batched recurrent snapshots: $GLM_K5_BATCHED_RECURRENT_SNAPSHOT"
echo "  GLM K5 fused native-FP4 QKV: $GLM_K5_FUSED_QKV"
echo "  shared-expert / EP-reduce overlap: $MOE_SHARED_REDUCE_OVERLAP"
echo "  GLM serial MTP prefill probe: $GLM_MTP_SERIAL_PREFILL"
echo "  GLM batched MTP KV prefill: $GLM_MTP_BATCHED_PREFILL"

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
  -e ATLAS_GLM_KDA_MULTI_SEQ=$KDA_MULTI_SEQ \
  -e ATLAS_GLM_KDA_BATCHED_FFN=$KDA_BATCHED_FFN \
  -e ATLAS_GLM_KDA_MS_PROFILE=$KDA_MS_PROFILE \
  -e ATLAS_GLM_MLA_MULTI_SEQ=$MLA_MULTI_SEQ \
  -e ATLAS_UNIFIED_MOE_LAYOUT=$UNIFIED_MOE_LAYOUT \
  -e ATLAS_CUBLAS_GEMM=$CUBLAS_GEMM \
  -e ATLAS_HC_CUBLAS_PREFILL=$HC_CUBLAS_PREFILL \
  -e ATLAS_NVFP4_GATE_UP_M128=$NVFP4_GATE_UP_M128 \
  -e ATLAS_NVFP4_DOWN_M32=$NVFP4_DOWN_M32 \
  -e ATLAS_NVFP4_PREQUANT_MOE=$NVFP4_PREQUANT_MOE \
  -e ATLAS_NVFP4_VECSCALE=$NVFP4_VECSCALE \
  -e ATLAS_NVFP4_FUSED_SILU_QUANT=$NVFP4_FUSED_SILU_QUANT \
  -e ATLAS_NVFP4_MMQ_MOE=$NVFP4_MMQ_MOE \
  -e ATLAS_MOE_GROUPED_CUTLASS=$NVFP4_CUTLASS_MOE \
  -e ATLAS_MOE_PREFILL_EXACT_TILES=$MOE_PREFILL_EXACT_TILES \
  -e ATLAS_MOE_PREFILL_MAX_LOAD_FACTOR=$MOE_PREFILL_MAX_LOAD_FACTOR \
  -e ATLAS_DUMP_EXPERT_IDS=$DUMP_EXPERT_IDS \
  -e ATLAS_MTP_SPEC_THINK=$MTP_SPEC_THINK \
  -e ATLAS_MTP_GATE_FORCE=$MTP_GATE_FORCE \
  -e ATLAS_MTP_SINGLE_DEPTH_ADAPT=$MTP_SINGLE_DEPTH_ADAPT \
  -e ATLAS_MTP_TIMING=$MTP_TIMING \
  -e ATLAS_MTP_ACCEPT_DEBUG=$MTP_ACCEPT_DEBUG \
  -e ATLAS_GLM_VERIFY_PROFILE=$VERIFY_PROFILE \
  -e ATLAS_GLM_INDEX_PROFILE=$GLM_INDEX_PROFILE \
  -e ATLAS_GLM_INDEX_ROW_GROUP=$GLM_INDEX_ROW_GROUP \
  -e ATLAS_GLM_SPARSE_HEAD_GROUP=$GLM_SPARSE_HEAD_GROUP \
  -e ATLAS_MOE_UNION_STATS=$MOE_UNION_STATS \
  -e ATLAS_GLM_K5_GROUPED_MOE=$GLM_K5_GROUPED_MOE \
  -e ATLAS_GLM_K5_BATCHED_SHARED=$GLM_K5_BATCHED_SHARED \
  -e ATLAS_GLM_K5_FUSED_SHARED_GATE_UP=$GLM_K5_FUSED_SHARED_GATE_UP \
  -e ATLAS_GLM_K5_DENSE_EXACT=$GLM_K5_DENSE_EXACT \
  -e ATLAS_GLM_K5_FUSED_DENSE_PAIRS=$GLM_K5_FUSED_DENSE_PAIRS \
  -e ATLAS_GLM_K5_FUSED_DENSE_TRIPLE=$GLM_K5_FUSED_DENSE_TRIPLE \
  -e ATLAS_GLM_K5_COMPACT_MOE=$GLM_K5_COMPACT_MOE \
  -e ATLAS_GLM_K5_FUSED_COMPACT_GATE_UP=$GLM_K5_FUSED_COMPACT_GATE_UP \
  -e ATLAS_GLM_K5_HC_CUBLAS=$GLM_K5_HC_CUBLAS \
  -e ATLAS_GLM_MTP_NVFP4_EH=$GLM_MTP_NVFP4_EH \
  -e ATLAS_GLM_K5_BATCHED_CONV_SNAPSHOT=$GLM_K5_BATCHED_CONV_SNAPSHOT \
  -e ATLAS_GLM_K5_BATCHED_RECURRENT_SNAPSHOT=$GLM_K5_BATCHED_RECURRENT_SNAPSHOT \
  -e ATLAS_GLM_K5_FUSED_QKV=$GLM_K5_FUSED_QKV \
  -e ATLAS_MOE_SHARED_REDUCE_OVERLAP=$MOE_SHARED_REDUCE_OVERLAP \
  -e ATLAS_GLM_MTP_SERIAL_PREFILL=$GLM_MTP_SERIAL_PREFILL \
  -e ATLAS_GLM_MTP_BATCHED_PREFILL=$GLM_MTP_BATCHED_PREFILL \
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
  -e ATLAS_GLM_KDA_MULTI_SEQ="$KDA_MULTI_SEQ" \
  -e ATLAS_GLM_KDA_BATCHED_FFN="$KDA_BATCHED_FFN" \
  -e ATLAS_GLM_KDA_MS_PROFILE="$KDA_MS_PROFILE" \
  -e ATLAS_GLM_MLA_MULTI_SEQ="$MLA_MULTI_SEQ" \
  -e ATLAS_UNIFIED_MOE_LAYOUT="$UNIFIED_MOE_LAYOUT" \
  -e ATLAS_CUBLAS_GEMM="$CUBLAS_GEMM" \
  -e ATLAS_HC_CUBLAS_PREFILL="$HC_CUBLAS_PREFILL" \
  -e ATLAS_NVFP4_GATE_UP_M128="$NVFP4_GATE_UP_M128" \
  -e ATLAS_NVFP4_DOWN_M32="$NVFP4_DOWN_M32" \
  -e ATLAS_NVFP4_PREQUANT_MOE="$NVFP4_PREQUANT_MOE" \
  -e ATLAS_NVFP4_VECSCALE="$NVFP4_VECSCALE" \
  -e ATLAS_NVFP4_FUSED_SILU_QUANT="$NVFP4_FUSED_SILU_QUANT" \
  -e ATLAS_NVFP4_MMQ_MOE="$NVFP4_MMQ_MOE" \
  -e ATLAS_MOE_GROUPED_CUTLASS="$NVFP4_CUTLASS_MOE" \
  -e ATLAS_MOE_PREFILL_EXACT_TILES="$MOE_PREFILL_EXACT_TILES" \
  -e ATLAS_MOE_PREFILL_MAX_LOAD_FACTOR="$MOE_PREFILL_MAX_LOAD_FACTOR" \
  -e ATLAS_DUMP_EXPERT_IDS="$DUMP_EXPERT_IDS" \
  -e ATLAS_MTP_SPEC_THINK="$MTP_SPEC_THINK" \
  -e ATLAS_MTP_GATE_FORCE="$MTP_GATE_FORCE" \
  -e ATLAS_MTP_SINGLE_DEPTH_ADAPT="$MTP_SINGLE_DEPTH_ADAPT" \
  -e ATLAS_MTP_TIMING="$MTP_TIMING" \
  -e ATLAS_MTP_ACCEPT_DEBUG="$MTP_ACCEPT_DEBUG" \
  -e ATLAS_GLM_VERIFY_PROFILE="$VERIFY_PROFILE" \
  -e ATLAS_GLM_INDEX_PROFILE="$GLM_INDEX_PROFILE" \
  -e ATLAS_GLM_INDEX_ROW_GROUP="$GLM_INDEX_ROW_GROUP" \
  -e ATLAS_GLM_SPARSE_HEAD_GROUP="$GLM_SPARSE_HEAD_GROUP" \
  -e ATLAS_MOE_UNION_STATS="$MOE_UNION_STATS" \
  -e ATLAS_GLM_K5_GROUPED_MOE="$GLM_K5_GROUPED_MOE" \
  -e ATLAS_GLM_K5_BATCHED_SHARED="$GLM_K5_BATCHED_SHARED" \
  -e ATLAS_GLM_K5_FUSED_SHARED_GATE_UP="$GLM_K5_FUSED_SHARED_GATE_UP" \
  -e ATLAS_GLM_K5_DENSE_EXACT="$GLM_K5_DENSE_EXACT" \
  -e ATLAS_GLM_K5_FUSED_DENSE_PAIRS="$GLM_K5_FUSED_DENSE_PAIRS" \
  -e ATLAS_GLM_K5_FUSED_DENSE_TRIPLE="$GLM_K5_FUSED_DENSE_TRIPLE" \
  -e ATLAS_GLM_K5_COMPACT_MOE="$GLM_K5_COMPACT_MOE" \
  -e ATLAS_GLM_K5_FUSED_COMPACT_GATE_UP="$GLM_K5_FUSED_COMPACT_GATE_UP" \
  -e ATLAS_GLM_K5_HC_CUBLAS="$GLM_K5_HC_CUBLAS" \
  -e ATLAS_GLM_MTP_NVFP4_EH="$GLM_MTP_NVFP4_EH" \
  -e ATLAS_GLM_K5_BATCHED_CONV_SNAPSHOT="$GLM_K5_BATCHED_CONV_SNAPSHOT" \
  -e ATLAS_GLM_K5_BATCHED_RECURRENT_SNAPSHOT="$GLM_K5_BATCHED_RECURRENT_SNAPSHOT" \
  -e ATLAS_GLM_K5_FUSED_QKV="$GLM_K5_FUSED_QKV" \
  -e ATLAS_MOE_SHARED_REDUCE_OVERLAP="$MOE_SHARED_REDUCE_OVERLAP" \
  -e ATLAS_GLM_MTP_SERIAL_PREFILL="$GLM_MTP_SERIAL_PREFILL" \
  -e ATLAS_GLM_MTP_BATCHED_PREFILL="$GLM_MTP_BATCHED_PREFILL" \
  "${MOUNT_FLAGS[@]}" \
  "$IMAGE" "${COMMON_SERVE_ARGS[@]}" --rank 0 --port "$PORT"

echo "Rank 0 logs: docker logs -f atlas-glm53-ep0"
echo "Rank 1 logs: ssh $SSH_TARGET 'docker logs -f atlas-glm53-ep1'"
echo "API (head node): http://$BIND_ADDRESS:$PORT/v1"
