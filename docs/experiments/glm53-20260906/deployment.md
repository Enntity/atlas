# Deployment profiles and rollback

**Historical phase-1 recipes, not current deployment instructions.** The later
dense MLA512 binding fix supersedes these images. Use
[the current bounded deployment](deployment-current.md), including its v10
rollback; do not restore v4/v65 merely because their original needle checks
passed. The recipes below are retained to interpret historical receipts.

Run these commands **on the head Spark**, not the EPYC development host.
Both Sparks already have image `atlas-glm53-flash:kernel-20260906-v4`.
Its binary SHA256 is
`d880d3826e2e9a41030443b927a3f6f1be53eb48ad5b17ff790675f93794c2ba`.
v4 adds only the direct-server MTP context guard to v3.

Before changing profiles, drain clients and stop both active test/service
containers gracefully. Do not start another model beside the resident one.

```bash
docker stop -t 20 atlas-glm53-ep0
ssh mangokid@192.168.8.187 'docker stop -t 20 atlas-glm53-ep1'
```

Use a fresh shell, then set the shared configuration:

```bash
export HEAD_IP=169.254.179.82 WORKER_IP=169.254.128.113
export SSH_TARGET=mangokid@192.168.8.187
export MODEL=/var/tmp/models/glm53-flash-nvfp4
export IMAGE=atlas-glm53-flash:kernel-20260906-v4
export TP_SIZE=2 PORT=8890 GPU_MEM_UTIL=.90 CONTAINER_MEMORY=114g
export REQUEST_TIMEOUT=600 OOM_GUARD_MB=4096
export GLM_INDEX_WMMA=1 GLM_MLA_BATCH23=1
```

Choose **one** profile below. Keep the memory guard and container ceiling.
Do not raise context or admission counts without a fresh free-memory check.
Changing prefill chunking can change floating-point results and MTP acceptance.

## Long independent concurrency

16K per sequence, three active/admitted sessions, no speculative decoding.
The eager semantic-index path is experimental and tested for retrieval and
batch drains; it is not graph-accelerated. The 27.4 tok/s short-profile result
below must not be attributed to this long profile. See the campaign report
for final-profile measurements and validation status.

```bash
MAX_SEQ_LEN=16384 MAX_PREFILL_TOKENS=4096 \
MAX_BATCH_SIZE=3 MAX_NUM_SEQS=3 SPECULATIVE=0 \
GLM_MULTI_SEQ_SPARSE=1 GLM_C3_GROUPED_MOE=1 FP4_PREFILL=1 \
bash /home/mangokid/atlas-glm53-deploy-20260906/start-glm53-ep2.sh
```

Both FP4 activation options are lossy. Set `GLM_C3_GROUPED_MOE=0` and/or
`FP4_PREFILL=0` for their respective precision fallbacks. Keep sparse mode
enabled for independent sessions beyond 2048; turning it off is not a valid
long-context fallback.

## Short-context throughput

This is the configuration that measured 27.366 aggregate tok/s at C3 with
1K prompts and 96 output tokens per session, versus 22.129 control.

```bash
MAX_SEQ_LEN=2048 MAX_PREFILL_TOKENS=1024 \
MAX_BATCH_SIZE=3 MAX_NUM_SEQS=3 SPECULATIVE=0 \
GLM_MULTI_SEQ_SPARSE=0 GLM_C3_GROUPED_MOE=1 FP4_PREFILL=0 \
bash /home/mangokid/atlas-glm53-deploy-20260906/start-glm53-ep2.sh
```

## Single-session MTP

1536 context, one session, fixed four-draft distributed MTP. The repeated
1K-prompt prefill A/B measured about 1050 tok/s with FP4 prefill versus 923
without it. Decode depends strongly on acceptance and prompt; the bounded
256-output repetitive test measured 32.379 tok/s, not a general guarantee.

```bash
MAX_SEQ_LEN=1536 MAX_PREFILL_TOKENS=1024 \
MAX_BATCH_SIZE=1 MAX_NUM_SEQS=1 SPECULATIVE=1 NUM_DRAFTS=4 \
MTP_GATE_FORCE=1 MTP_SINGLE_DEPTH_ADAPT=0 \
GLM_K5_BATCHED_SHARED=1 GLM_K5_COMPACT_MOE=1 \
GLM_K5_FUSED_COMPACT_GATE_UP=1 GLM_MTP_DISTRIBUTED=1 \
GLM_MULTI_SEQ_SPARSE=0 GLM_C3_GROUPED_MOE=0 FP4_PREFILL=1 \
bash /home/mangokid/atlas-glm53-deploy-20260906/start-glm53-ep2.sh
```

The launcher and server reject MTP when context plus draft lookahead exceeds
the 2048-token exact-attention threshold. Multi-session MTP remains unsupported.

## Health and rollback

```bash
curl --fail http://127.0.0.1:8890/health
docker logs --tail 30 atlas-glm53-ep0
free -m
ssh mangokid@192.168.8.187 'free -m; docker logs --tail 20 atlas-glm53-ep1'
```

The original v65 containers are preserved, stopped, with their original
512-context/64-chunk MTP configuration. Inspection confirmed the original
head also uses port8890; the health/client endpoint remains the same.
To restore them after draining clients:

```bash
docker stop -t 20 atlas-glm53-ep0
ssh mangokid@192.168.8.187 'docker stop -t 20 atlas-glm53-ep1'
ssh mangokid@192.168.8.187 'docker start atlas-glm53-v65-rollback-ep1'
docker start atlas-glm53-v65-rollback-ep0
```

Wait for both ranks to finish loading and health to pass before sending
requests. Do not run the launcher while rollback containers are active:
their names differ, so it cannot detect/stop them for you. Stop both rollback
containers explicitly before returning to a new profile. No restart policy
was enabled; model loading after a reboot remains an explicit operator action.
