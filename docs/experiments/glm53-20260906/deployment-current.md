# Current bounded dual-Spark GLM deployment

As of 2026-09-07, the active short profile is v13: native NVFP4 GLM-5.3-Flash,
TP2/EP2, BF16 KV/index, FP32 recurrent state, **2048 context per request**,
four active/admitted requests, 1024-token prefill chunks, no speculation.
Graph-enabled answer/needle and stream-lifecycle gates passed. This retains
v11's indexed KDA and fixes first-token thinking state, non-speculative
cancellation, and terminal stream flushing. Matched C4 coding148/256 measures
47.319 full-wall aggregate tok/s, effectively unchanged from v11's47.309;
all eight measured responses reach256 tokens. See [phase 5](phase5-vllm-gap.md)
for fixed-output throughput controls and the still-open vLLM capability gap;
[phase 4](phase4-infrastructure-results.md) records the preceding kernel wins.
This is not a claim of general model-quality parity or a full serve-matrix run.

The v13 binary was built from `2465672f`. The later pure CPU legacy codec
commit `79c01f03` is unused by serving and not included in this binary.
Both nodes have image `atlas-glm53-flash:kernel-20260907-v13` and binary SHA256
`06a77b12a0f920d6823ce5a7466b2fb4a4ebf1ddcd672b80dc370461c1d00ac1`.
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
IMAGE=atlas-glm53-flash:kernel-20260907-v13 \
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

## Diagnostic rollback: v12 short control

Both preserved containers `atlas-glm53-v12-terminal-control-ep0/1` have the same
short-C4 graph profile and indexed KDA, but retain a known guarded-Done text
leak fixed in v13. This is a diagnostic rollback, not a fully equivalent
stream-correctness profile. Binary SHA256:
`d06eff5e8dadf20c84b5bb14eb5ea7c75cf201bfddf3c221235306e6ec2d0054`.
After draining clients and confirming no other named model is active:

```bash
docker stop -t 20 atlas-glm53-ep0
ssh mangokid@192.168.8.187 'docker stop -t 20 atlas-glm53-ep1'
ssh mangokid@192.168.8.187 'docker start atlas-glm53-v12-terminal-control-ep1'
docker start atlas-glm53-v12-terminal-control-ep0
```

Wait for readiness and run answer checks. Before relaunching a standard-name
profile, explicitly stop both `atlas-glm53-v12-terminal-control-ep0/1` containers.
Older v11/v10 short controls remain stopped and preserved; v11 also has the
first-token thinking/cancellation defects documented in phase5.
Do not perform a unilateral NCCL reconnect or reset a host to clear a warning.
The [command-idle health investigation](ep-command-idle-health-plan.md) explains
a preexisting warning that counts successful idle command waits as slow
broadcasts. It remains unfixed in v13; independent transport/CUDA errors still
require investigation.

Receipts and launch logs are under
`/home/mangokid/atlas-glm53-deploy-20260906/phase5/` on the head; preceding
v10/v11 controls remain under `phase3/`. No automatic
restart policy, host-clock change, swap-policy change, or sudo operation was
introduced by this deployment.
