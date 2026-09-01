#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

# Launch the two-Spark-native GLM-5.3 Flash EXL3 pure-TP2 appliance. The development
# executable is bind-mounted so normal Rust/CUDA-kernel iterations avoid an
# entire multi-gigabyte image transfer.

set -euo pipefail

IMAGE=${IMAGE:-sparkglm-atlas:glm53-exl3-20260831}
MODEL_DIR=${MODEL_DIR:-/home/enntitysparkadmin/.lloom/models/Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw}
DFLASH_MODEL_DIR=${DFLASH_MODEL_DIR:-/home/enntitysparkadmin/.lloom/models/incoai--GLM-5.3-Flash-DFlash2}
DEV_BINARY=${DEV_BINARY:-/tmp/sparkglm53-dev/spark}
RANK0_SSH_HOST=${RANK0_SSH_HOST:-ennspark01}
# The fixed appliance's rank-1 LAN address avoids both MagicDNS and NVIDIA
# Sync's command-length-sensitive proxy. Override these when the lab moves.
RANK1_SSH_HOST=${RANK1_SSH_HOST:-192.168.1.249}
RANK1_SSH_USER=${RANK1_SSH_USER:-enntitysparkadmin}
RANK1_SSH_IDENTITY=${RANK1_SSH_IDENTITY:-/Users/jmac/Library/Application Support/NVIDIA/Sync/config/nvsync.key}
RANK1_SSH_HOST_KEY_ALIAS=${RANK1_SSH_HOST_KEY_ALIAS:-spark-6f19.local}
MASTER_ADDR=${MASTER_ADDR:-10.100.16.2}
MASTER_PORT=${MASTER_PORT:-29553}
# Four decode/metadata rows are reserved beside prefill, so 7,164 admitted
# tokens fill the appliance's exact 7,168-row arena. Solo and idle bursts use
# the full slab, exposing fat experts to the large-M EXL3 path. When decoding
# is active, phase interleave still bounds each prompt slice to 128 tokens.
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-7164}
PHASE_PREFILL_SLICE_TOKENS=${PHASE_PREFILL_SLICE_TOKENS:-128}
PHASE_DECODE_STEPS=${PHASE_DECODE_STEPS:-1}
PHASE_PREFILL_STEPS=${PHASE_PREFILL_STEPS:-1}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-4}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-$MAX_BATCH_SIZE}
# TP2 verification contains rank-symmetric collectives and has a validated
# graph-safe metadata path. Graphs are part of the fast appliance, not an
# operator-only experiment; set 0 only for an eager-path diagnostic.
GLM53_EP_GRAPHS=${GLM53_EP_GRAPHS:-1}
GLM53_PROFILE=${GLM53_PROFILE:-0}
GLM53_DFLASH=${GLM53_DFLASH:-1}
# Seven drafts are the measured production optimum for both solo and
# concurrent streams. A 17-row build remains supported for depth experiments,
# where the scheduler can demote weak requests, but the 5x400 structured gate
# showed that accepting roughly 9/16 does not repay the larger target pass.
GLM53_DFLASH_GAMMA=${GLM53_DFLASH_GAMMA:-8}
GLM53_DFLASH_DEPTH_LADDER=${GLM53_DFLASH_DEPTH_LADDER:-1}
GLM53_DFLASH_TIMING=${GLM53_DFLASH_TIMING:-0}
GLM53_DFLASH_SPEC_THINK=${GLM53_DFLASH_SPEC_THINK:-1}
GLM53_MTP_GATE_FORCE=${GLM53_MTP_GATE_FORCE:-1}
# The phase-interleave scheduler is the measured default. The current fused
# mixed verify+prefill path raises decode latency on GB10, so keep it as an
# explicit A/B until it beats interleave on the workload-matched canary.
GLM53_DFLASH_PREFILL_FUSION=${GLM53_DFLASH_PREFILL_FUSION:-0}
GLM53_DFLASH_BLOCK_DUMP=${GLM53_DFLASH_BLOCK_DUMP:-0}
GLM53_DFLASH_BLOCK_DUMP_AT_POS=${GLM53_DFLASH_BLOCK_DUMP_AT_POS:-0}
GLM53_DFLASH_PRECOMPUTE_DUMP=${GLM53_DFLASH_PRECOMPUTE_DUMP:-0}
GLM53_DFLASH_CONTIG_ATTN=${GLM53_DFLASH_CONTIG_ATTN:-0}
GLM53_MULTI_PROFILE_ONCE=${GLM53_MULTI_PROFILE_ONCE:-0}
GLM53_NO_DFLASH_MULTI_GRAPH=${GLM53_NO_DFLASH_MULTI_GRAPH:-0}
GLM53_NO_MTP_BATCH_VERIFY=${GLM53_NO_MTP_BATCH_VERIFY:-0}
GLM53_EXL3_FAT=${GLM53_EXL3_FAT:-1}
GLM53_DSA_PREFILL_FAST=${GLM53_DSA_PREFILL_FAST:-1}
GLM53_DSA_PREFILL_TC=${GLM53_DSA_PREFILL_TC:-1}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-/tmp/atlas-glm53-ssh-%C}

ssh_opts=(
  -o ControlMaster=auto
  -o ControlPersist=600
  -o "ControlPath=$SSH_CONTROL_PATH"
)

ssh_rank() {
  local host=$1
  shift
  if [[ $host == "$RANK1_SSH_HOST" ]]; then
    ssh "${ssh_opts[@]}" \
      -i "$RANK1_SSH_IDENTITY" -o IdentitiesOnly=yes \
      -o "HostKeyAlias=$RANK1_SSH_HOST_KEY_ALIAS" \
      "$RANK1_SSH_USER@$host" "$@"
  else
    ssh "${ssh_opts[@]}" "$host" "$@"
  fi
}

common_env=(
  -e ATLAS_EP_PROTOCOL=v2
  -e RUST_LOG=info
  # Collect a concurrent HTTP burst before native ragged prefill. Without
  # this, four simultaneous requests arrive as 2+2/3+1 cohorts and cannot
  # share either prefill or their first DFlash target sweep.
  -e ATLAS_PREFILL_CODISPATCH=1
  -e ATLAS_PREFILL_CODISPATCH_WINDOW_MS=100
  -e ATLAS_PREFILL_CODISPATCH_SETTLE_MS=25
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0
  -e NCCL_IB_HCA=rocep1s0f0
  -e NCCL_IB_DISABLE=0
  -e NCCL_IB_ADDR_FAMILY=AF_INET
  -e NCCL_IB_ROCE_VERSION_NUM=2
  -e NCCL_IB_TIMEOUT=22
  -e NCCL_IB_RETRY_CNT=7
  # Do not set NCCL_NET_GDR_LEVEL here. On the two-Spark RoCE path, NCCL's
  # automatic GPUDirect selection is materially faster than the host-bounce
  # path forced by LEVEL=0. An operator can still pass an explicit override
  # through the host environment; it is appended below.
  -e NCCL_NET_GDR_C2C=0
  -e NCCL_DMABUF_ENABLE=0
  -e NCCL_CUMEM_HOST_ENABLE=0
  -e NCCL_NVLS_ENABLE=0
  -e NCCL_ALGO=Ring
  -e NCCL_PROTO=Simple
  -e NCCL_MIN_NCHANNELS=1
  -e NCCL_MAX_NCHANNELS=1
)

if [[ -n ${NCCL_NET_GDR_LEVEL:-} ]]; then
  common_env+=(-e "NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL")
fi

# Diagnostic escape hatch. The graph-safe GLM metadata path is the default;
# set GLM53_NO_MULTI_GRAPHS=1 only for an eager-vs-graph A/B.
if [[ ${GLM53_NO_MULTI_GRAPHS:-0} == 1 ]]; then
  common_env+=(-e ATLAS_NO_DECODE_GRAPHS_MULTISEQ=1)
fi

# GLM decode is launch-bound at TP2 when every layer and collective is issued
# eagerly. The GLM state-pointer tables are fixed-address and graph-safe, so
# capture the complete rank-symmetric decode step by default. Keep a one-line
# rollback for graph/NCCL diagnostics.
if [[ $GLM53_EP_GRAPHS == 1 ]]; then
  common_env+=(-e ATLAS_EP_GRAPHS=1)
fi

if [[ $GLM53_DFLASH == 1 ]]; then
  common_env+=(
    -e ATLAS_DFLASH_OPTION_B=1
    -e ATLAS_DFLASH_CTX_WINDOW=2048
  )
  if [[ $GLM53_DFLASH_DEPTH_LADDER == 1 ]]; then
    common_env+=(-e ATLAS_GLM_DFLASH_DEPTH_LADDER=1)
  fi
  if [[ $GLM53_DFLASH_TIMING == 1 ]]; then
    common_env+=(-e ATLAS_DFLASH_STEP_TIMING=1)
  fi
  if [[ $GLM53_DFLASH_SPEC_THINK == 1 ]]; then
    common_env+=(-e ATLAS_DFLASH_SPEC_THINK=1)
  fi
  if [[ $GLM53_MTP_GATE_FORCE == 1 ]]; then
    common_env+=(-e ATLAS_MTP_GATE_FORCE=1)
  fi
  common_env+=(-e "ATLAS_GLM_DFLASH_PREFILL_FUSION=$GLM53_DFLASH_PREFILL_FUSION")
  if [[ $GLM53_DFLASH_BLOCK_DUMP == 1 ]]; then
    common_env+=(
      -e ATLAS_DFLASH_BLOCK_DUMP=1
      -e "ATLAS_DFLASH_BLOCK_DUMP_AT_POS=$GLM53_DFLASH_BLOCK_DUMP_AT_POS"
    )
  fi
  if [[ $GLM53_DFLASH_PRECOMPUTE_DUMP == 1 ]]; then
    common_env+=(-e ATLAS_DFLASH_PRECOMPUTE_DUMP=1)
  fi
  if [[ $GLM53_DFLASH_CONTIG_ATTN == 1 ]]; then
    common_env+=(-e ATLAS_DFLASH_CONTIG_ATTN=1)
  fi
fi

if [[ $GLM53_MULTI_PROFILE_ONCE == 1 ]]; then
  common_env+=(-e ATLAS_GLM_MULTI_PROFILE_ONCE=1)
fi

# Correctness/performance bisects for the native concurrent DFlash verifier.
# These are deliberately independent: the first keeps the N x K kernels but
# executes them eagerly; the second falls back to the proven per-sequence
# verifier. Both default off in the production appliance.
if [[ $GLM53_NO_DFLASH_MULTI_GRAPH == 1 ]]; then
  common_env+=(-e ATLAS_NO_DFLASH_MULTI_GRAPH=1)
fi
if [[ $GLM53_NO_MTP_BATCH_VERIFY == 1 ]]; then
  common_env+=(-e ATLAS_NO_MTP_BATCH_VERIFY=1)
fi

# Independent numerical/performance rollback switches for the two GB10
# prefill fast paths. Both retain the original kernels in the same binary.
if [[ $GLM53_EXL3_FAT == 0 ]]; then
  common_env+=(-e ATLAS_GLM_EXL3_FAT=0)
fi
if [[ $GLM53_DSA_PREFILL_FAST == 0 ]]; then
  common_env+=(-e ATLAS_GLM_DSA_PREFILL_LEGACY=1)
fi
if [[ $GLM53_DSA_PREFILL_TC == 0 ]]; then
  common_env+=(-e ATLAS_GLM_DSA_PREFILL_TC=0)
fi

common_args=(
  serve --model-from-path /model --model-name GLM-5.3-Flash-EXL3
  --kernel-target glm-5.3-flash
  --world-size 2 --ep-size 1 --tp-size 2
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT"
  --max-seq-len 32768 --max-num-seqs "$MAX_NUM_SEQS" --max-batch-size "$MAX_BATCH_SIZE"
  --gpu-memory-utilization 0.92 --kv-cache-dtype bf16
  --max-prefill-tokens "$MAX_PREFILL_TOKENS"
  --scheduling-policy phase-interleave
  --phase-decode-steps "$PHASE_DECODE_STEPS" --phase-prefill-steps "$PHASE_PREFILL_STEPS"
  --phase-prefill-slice-tokens "$PHASE_PREFILL_SLICE_TOKENS" --tbt-deadline-ms 500
  --swap-space-gb 0 --no-auto-swap
)

if [[ $GLM53_PROFILE == 1 ]]; then
  common_args+=(--profile)
fi

if [[ $GLM53_DFLASH == 1 ]]; then
  common_args+=(
    --dflash --draft-model /dflash --dflash-gamma "$GLM53_DFLASH_GAMMA"
    --dflash-window-size 2048
  )
fi

launch_rank() {
  local host=$1 rank=$2 port=$3 name=$4
  ssh_rank "$host" docker run -d \
    --name "$name" \
    --gpus all --network host --ipc host \
    --device /dev/infiniband:/dev/infiniband \
    --ulimit memlock=-1 --cap-add IPC_LOCK --security-opt label=disable \
    -v "$MODEL_DIR:/model:ro" \
    -v "$DFLASH_MODEL_DIR:/dflash:ro" \
    -v "$DEV_BINARY:/usr/local/bin/spark:ro" \
    "${common_env[@]}" \
    "$IMAGE" "${common_args[@]}" --rank "$rank" --port "$port"
}

# Both ranks are one distributed process, so stop them concurrently. Serial
# removal needlessly pays each rank's container teardown latency back-to-back.
ssh_rank "$RANK0_SSH_HOST" docker rm -f sparkglm53-r0 >/dev/null 2>&1 &
stop_rank0=$!
ssh_rank "$RANK1_SSH_HOST" docker rm -f sparkglm53-r1 >/dev/null 2>&1 &
stop_rank1=$!
wait "$stop_rank0" "$stop_rank1" || true

# Start the worker first so it is already entering rendezvous when rank 0 joins.
launch_rank "$RANK1_SSH_HOST" 1 0 sparkglm53-r1
launch_rank "$RANK0_SSH_HOST" 0 8888 sparkglm53-r0

echo "GLM pure TP2 launched; rank 0 API: http://ennspark01:8888"
