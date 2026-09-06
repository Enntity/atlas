# Current bounded dual-Spark GLM deployment

As of 2026-09-06, the active validated short profile is v11: native NVFP4 GLM-5.3-Flash,
TP2/EP2, BF16 KV/index, FP32 recurrent state, **2048 context per request**,
four active/admitted requests, 1024-token prefill chunks, no speculation.
Eager and graph-enabled answer/needle gates passed. Matched C4 coding throughput
is47.309 full-wall aggregate tokens/s versus45.669 for v10 (+3.6%, two measured
batches); see [phase 4](phase4-infrastructure-results.md) for workload and limits.
This is not a claim of general model-quality parity or a full serve-matrix run.

The v11 binary was built from `e8771f30`; `d3b0ac9f` contains only subsequent
equivalent style predicates and is not the binary's source revision.
Both nodes have image `atlas-glm53-flash:kernel-20260906-v11` and binary SHA256
`949ec12d3aac6024927cbd070c16b0a1658c1129343bedb222fb48539a5e3f29`.
The corrected dense MLA512 binding is included. Do not prefer old v4/v5/v65
rollback images, which predate that fix; v6 is a known failed experiment.

## Reproduce the profile

Run on head `192.168.8.181`. Drain clients, then gracefully stop **both** active
model containers before another launch. Inspect `docker ps` on both nodes:
the launcher only replaces standard `atlas-glm53-ep0/1` names and cannot detect
an active, differently named rollback. Preserve containers needed for rollback
by renaming them to an unused, explicit name before invoking the launcher.
Do not run a GPU benchmark or another model beside the resident service.

```bash
HEAD_IP=169.254.179.82 WORKER_IP=169.254.128.113 \
SSH_TARGET=mangokid@192.168.8.187 \
MODEL=/var/tmp/models/glm53-flash-nvfp4 \
IMAGE=atlas-glm53-flash:kernel-20260906-v11 \
TP_SIZE=2 PORT=8890 GPU_MEM_UTIL=.90 CONTAINER_MEMORY=114g \
OOM_GUARD_MB=4096 REQUEST_TIMEOUT=600 \
MAX_SEQ_LEN=2048 MAX_PREFILL_TOKENS=1024 \
MAX_BATCH_SIZE=4 MAX_NUM_SEQS=4 KV_OVERCOMMIT=0 SPECULATIVE=0 \
KDA_MULTI_SEQ=1 MLA_MULTI_SEQ=1 \
GLM_INDEX_WMMA=1 GLM_MLA_BATCH23=1 GLM_MLA_BATCH4=1 \
GLM_C3_GROUPED_MOE=1 GLM_C4_DECODE=1 GLM_C4_GROUPED_MOE=1 \
GLM_C4_SPARSE=0 GLM_MULTI_SEQ_SPARSE=0 GLM_MULTI_SEQ_SPARSE_GRAPHS=0 \
NO_DECODE_GRAPHS_MULTISEQ=0 FP4_PREFILL=1 DECODE_BATCH_LOG=1 \
bash /home/mangokid/atlas-glm53-deploy-20260906/start-glm53-ep2.sh
```

Use a fresh shell so unrelated experiment variables are not inherited. Keep
the memory guard and container ceiling. Check host `MemAvailable` on both
nodes, not just `MemFree` or CUDA free memory. Normal startup took roughly
75 seconds in this campaign; wait for readiness before sending requests.

```bash
curl --fail http://127.0.0.1:8890/health
docker logs --tail 30 atlas-glm53-ep0
free -m
ssh mangokid@192.168.8.187 'free -m; docker logs --tail 30 atlas-glm53-ep1'
```

API: `http://192.168.8.181:8890/v1`. FP4 prefill and grouped MoE retain their
lossy activation formats; the exact indexed KDA update does not change that.
For contexts above2048, this is the wrong profile. The separate v9 long-C4
eager lane passed bounded quality checks but was **not** promoted for
performance. Raising the context or enabling sparse graphs is not a safe
extension of this recipe.

## Matched fallback: v10 short control

Both preserved containers `atlas-glm53-v10-short-control-ep0/1` have the same
short-C4 graph profile, without the indexed KDA change. Binary SHA256:
`316cfa33bb72bb08709ee699880f909d631dc137453e5a91e396bed065d039ce`.
After draining clients and confirming no other named model is active:

```bash
docker stop -t 20 atlas-glm53-ep0
ssh mangokid@192.168.8.187 'docker stop -t 20 atlas-glm53-ep1'
ssh mangokid@192.168.8.187 'docker start atlas-glm53-v10-short-control-ep1'
docker start atlas-glm53-v10-short-control-ep0
```

Wait for readiness and run answer checks. Before relaunching a standard-name
profile, explicitly stop both `atlas-glm53-v10-short-control-ep0/1` containers.
Do not perform a unilateral NCCL reconnect or reset a host to clear a warning.
The [command-idle health investigation](ep-command-idle-health-plan.md) explains
a preexisting warning that counts successful idle command waits as slow
broadcasts. It remains unfixed in v11; independent transport/CUDA errors still
require investigation.

Receipts and launch logs are under
`/home/mangokid/atlas-glm53-deploy-20260906/phase3/` on the head. No automatic
restart policy, host-clock change, swap-policy change, or sudo operation was
introduced by this deployment.
