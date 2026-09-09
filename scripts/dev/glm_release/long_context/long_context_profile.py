# SPDX-License-Identifier: AGPL-3.0-only
"""Literal nonspeculative TP2/EP2 C4 sparse profile; no ambient ATLAS environment."""
import ipaddress
import re

LIMIT = 114 * 1024**3
MODEL = "/var/tmp/models/glm53-flash-nvfp4"
SERVER = "/usr/local/bin/spark"
ENV = {
    "PATH": "/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "LD_LIBRARY_PATH": "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64",
    "RUST_LOG": "info", "NCCL_NET": "IB", "NCCL_IB_DISABLE": "0",
    "NCCL_IB_ROCE_VERSION_NUM": "2", "NCCL_IB_ADDR_FAMILY": "AF_INET",
    "NCCL_IB_TIMEOUT": "22", "NCCL_IB_RETRY_CNT": "7", "NCCL_NET_GDR_LEVEL": "0",
    "NCCL_NET_GDR_C2C": "0", "NCCL_DMABUF_ENABLE": "0", "NCCL_NVLS_ENABLE": "0",
    "NCCL_CUMEM_HOST_ENABLE": "0", "NCCL_CUMEM_ENABLE": "0", "NCCL_PROTO": "Simple",
    "NCCL_ALGO": "Ring", "NCCL_BUFFSIZE": "33554432", "NCCL_MIN_NCHANNELS": "1",
    "NCCL_MAX_NCHANNELS": "2", "NCCL_DEBUG": "WARN", "ATLAS_EP_PROTOCOL": "v2",
}
for _name in (
    "GLM_C4_DECODE GLM_C4_SPARSE GLM_MULTI_SEQ_SPARSE GLM_C3_GROUPED_MOE "
    "GLM_C4_GROUPED_MOE GLM_KDA_MULTI_SEQ GLM_KDA_BATCHED_FFN GLM_MLA_MULTI_SEQ "
    "GLM_MLA_BATCH23 GLM_MLA_BATCH4 GLM_INDEX_WMMA KDA_REGRESIDENT_PREFILL "
    "FP4_PREFILL CUBLAS_GEMM HC_CUBLAS_PREFILL UNIFIED_MOE_LAYOUT "
    "NVFP4_PREQUANT_MOE NVFP4_VECSCALE NVFP4_FUSED_SILU_QUANT "
    "MOE_PREFILL_EXACT_TILES MOE_SHARED_REDUCE_OVERLAP DECODE_BATCH_LOG "
    "NO_DECODE_GRAPHS NO_DECODE_GRAPHS_MULTISEQ DEBUG_NO_GRAPH"
).split():
    ENV["ATLAS_" + _name] = "1"
for _name in (
    "GLM_INDEPENDENT_DECODE GLM_MULTI_SEQ_SPARSE_GRAPHS KV_OVERCOMMIT "
    "GLM_MTP_DISTRIBUTED GLM_MTP_BATCHED_PREFILL GLM_MTP_SERIAL_PREFILL "
    "GLM_MTP_REPAIR GLM_MTP_KV_REPAIR_VERIFY GLM_MTP_K5_LEDGER GLM_MTP_HIDDEN_TRACE "
    "GLM_MOE_GATE_UP_M16 GLM_MOE_GATE_UP_M16_VERIFY GLM_TARGET_SHARED_FP8 "
    "GLM_TARGET_SHARED_FP8_VERIFY GLM_C2_COMPACT_MOE GLM_TP_VERIFY_GRAPH "
    "NVFP4_GATE_UP_M128 NVFP4_DOWN_M32 NVFP4_MMQ_MOE MOE_GROUPED_CUTLASS"
).split():
    ENV["ATLAS_" + _name] = "0"
ENV.update(ATLAS_GLM_INDEX_ROW_GROUP="8", ATLAS_GLM_SPARSE_HEAD_GROUP="8",
           ATLAS_MOE_PREFILL_MAX_LOAD_FACTOR="8")


def digest(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value) or value == "0" * 64:
        raise ValueError("expected full lowercase SHA256, without sha256: prefix")
    return value


def argv(context, rank, fabric, port):
    if type(context) is not int or context not in (4096, 8192, 16384):
        raise ValueError("context must be exactly 4096, 8192, or 16384")
    if rank not in (0, 1) or not 1024 <= port <= 65535:
        raise ValueError("invalid rank/port")
    ipaddress.IPv4Address(fabric)
    return [SERVER, "serve", MODEL, "--world-size=2", "--tp-size=2", "--ep-size=2",
            f"--rank={rank}", f"--master-addr={fabric}", "--master-port=29500",
            "--bind=0.0.0.0", f"--port={port}", f"--max-seq-len={context}",
            "--max-prefill-tokens=1024", "--max-batch-size=4", "--max-num-seqs=4",
            "--gpu-memory-utilization=0.90", "--oom-guard-mb=4096", "--kv-cache-dtype=bf16",
            "--ssm-h-dtype=f32", "--ssm-rollback-mode=snapshot", "--block-size=16",
            "--ssm-cache-slots=0", "--ssm-checkpoint-interval=0",
            "--gpu-ordinal=0", "--swap-space-gb=0", "--no-auto-swap", "--no-tui",
            "--disable-tool-grammar=true", "--lm-head-dtype=nvfp4", "--request-timeout=600"]


def environment(node, paged_prefill_bf16_gemm):
    if type(paged_prefill_bf16_gemm) is not bool:
        raise ValueError("paged_prefill_bf16_gemm must be an explicit boolean")
    fabric = node["fabric"]
    if not isinstance(fabric, dict) or set(fabric) != {"interface", "hca", "gid_index"}:
        raise ValueError("explicit per-node RDMA interface/hca/gid_index required")
    for key in ("interface", "hca"):
        if not isinstance(fabric[key], str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", fabric[key]):
            raise ValueError("invalid RDMA " + key)
    if not isinstance(fabric["gid_index"], str) or not re.fullmatch(r"[0-9]{1,3}", fabric["gid_index"]):
        raise ValueError("invalid explicit RDMA GID index")
    result = dict(ENV)
    result.update(NCCL_SOCKET_IFNAME=fabric["interface"], NCCL_IB_HCA=fabric["hca"],
                  NCCL_IB_GID_INDEX=fabric["gid_index"],
                  ATLAS_GLM_PAGED_PREFILL_BF16_GEMM="1" if paged_prefill_bf16_gemm else "0")
    return result


def create_args(node, session, context, fabric, port, paged_prefill_bf16_gemm):
    name = f"atlas-longctx-{session}-r{node['rank']}"
    args = ["docker", "create", "--name", name, "--label", f"atlas.longctx={session}",
            "--restart=no", "--gpus=all", "--network=host", "--ipc=private", "--shm-size=1g",
            "--memory", str(LIMIT), "--memory-swap", str(LIMIT), "--cpuset-cpus=0-19",
            "--runtime=runc", "--user=0:0", "--device=/dev/infiniband:/dev/infiniband:rwm",
            "--cap-add=IPC_LOCK", "--cap-add=SYS_NICE", "--ulimit=memlock=-1:-1",
            "--security-opt=no-new-privileges=true", "--security-opt=seccomp=unconfined",
            "--security-opt=label=disable", "--stop-timeout=35",
            "--mount", f"type=bind,src={node['weights_host_path']},dst={MODEL},readonly",
            "--entrypoint=/usr/bin/env", "sha256:" + digest(node["image_sha256"]), "-i"]
    # env -i discards ALL image-inherited selected-mode/FD/MTP flags at exec.
    args += [f"{k}={v}" for k, v in sorted(environment(node, paged_prefill_bf16_gemm).items())]
    return args + argv(context, node["rank"], fabric, port)
