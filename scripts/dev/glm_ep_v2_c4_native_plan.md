# v26 native EP-v2 C1/C2/C3/C4 idle and drain gate

2026-09-08, root-owned continuation after both v26 C1/v1 idle gates pass.
The actual protocol fix is CPU-tested for v1 and v2, but native C1 selected v1.
Close that native v2 gap using the existing nonspeculative C4 profile; this is
not permission to enable multi-request speculation or unfinished B-tile code.
User explicitly requested the complete C1/C2/C3/C4 matrix on continuation.
Sweep offered concurrency1..4 on this one unchanged active4/admitted4 server
profile. Its nonspeculative C1 is a scaling control, not a replacement for the
separately qualified MTP4 C1 profile (v26 initial28.699/fresh28.618).

## Exact source and historical profile

Reuse the already verified v26 image/executable from `e40a9066`, CUDA189db87e,
SHA256 `a29c261991b94309a2eb4193f3354a429cb60f77d851604b62fe1f3ea5538cbd`.
No new build, overlay, kernel change or resident layout conversion. Both nodes
are stopped before this gate. Keep current controller reader WIP off the nodes.

Root inspected the preserved head `atlas-glm53-v13-short-control-ep0` command
and selected runtime environment, retained as
`atlas-campaigns/20260908/v26-c4-historical-v13-config.log`. It is the established
v13 image, context2048, prefill1024, active/admitted4, utilization.90, BF16 KV,
NVFP4 target head,4096MiB guard, request timeout600, TP2/EP2, EP_PROTOCOL=v2,
no speculative flag. Reproduce those settings with the frozen phase7 launcher.

Explicit C4 settings: KDA/MLA multi-sequence ON, MLA batch23/batch4 ON, grouped
C3/C4 MoE ON, C4 decode ON, indexed WMMA ON, FP4 prefill ON, nonsparse decode,
multisequence graphs ON and DECODE_BATCH_LOG1. Keep114GiB container limits,
KV overcommit0 and swapspace0. All newer C1-only measurement arms, MTP repair/
distributed primer, hidden/ledger probes, M16 experiments and target shared FP8
cache remain OFF. Shared/EP overlap keeps its historical value1. No C1 recipe
environment may leak into this profile. The launcher explicitly selects v2
when max batch exceeds1; confirm actual both-rank environment/CLI after load.

The historical C4 figure47.319 used v13 and two measured batches. A v26 result
is a new bounded baseline, not a one-variable historical A/B speedup claim.
Keep the acceptance fixture unchanged and use one warmup plus three measured
batches and a fresh repeat before any60-aggregate claim.

## Automated native sequence

Root preflight verifies no competing model/build/GPU workload and preserves
all standard-name containers. Refuse any existing receipt tag or target name.
Prepare the entire runner before launch. Require both hosts at least4096MiB
MemAvailable at readiness, then deliberately wait35 seconds with no requests.
Record local idle boundaries and both worker/head log timestamps. Exact NCCL
call duration is not independently instrumented; do not claim it is.

Run unchanged literal148/256, temperature0/seed1, offered concurrency1,2,3,4,
one warmup and three measured waves per width. Every measured wave must finish
all requested256-token outputs; no forced EOS/repetition policy or partial-
response rate substitution. The existing harness discards warmup receipts;
do not claim independent warmup output-cap verification. Each measured result
must retain its actual counts, per-session rate and full-wall aggregate rate.
Freeze complete both-rank performance-window logs before quality traffic.

Sequential quality/lifecycle steps:

1. Four strict answer checks with concurrency4 and context limit2048.
2. Existing `benchmark_glm53_concurrent_niah.py`: prompt lengths
   768/800/832/896 and output caps32/16/48/64, two repetitions, context2048.
   Require each correct unique needle and no foreign needle. Preserve all
   outputs/lengths; these mixed-cap requests are not throughput qualification.
3. Deliberate two-second cancellation using the unchanged LRU148/128 stream
   payload, require curl28 with received bytes, then four concurrent recovery
   answers. No simultaneous standalone GPU work.
4. Gracefully stop both services with30-second timeout. Preserve full post-stop
   logs, selected settings, memory and renamed containers. Require stopped,
   exit0, OOMfalse and swap0. Any failure remains a failed receipt.

Parse actual `ATLAS_DECODE_BATCH` records on both ranks. Report exact width/
slot-vector counts and matching head/worker coverage: sustained N4, observed
N3/N2 drains, non-prefix/permuted slots when present. Never infer batching from
four successful HTTP replies alone, invent missing widths or assume every
scalar N1 step emits this batch log. Separate single-request cancellation and
recovery/observed scalar execution from proven same-wave drain coverage. If
coverage is incomplete, report that limitation and design a bounded targeted
follow-up before claiming complete native drain validation.

Require no ERROR-level/unhealthy/CUDA fault/panic or hidden/ledger records in complete logs,
stripping ANSI before classification. Benign INFO-level `os error111` during
startup is not an ERROR-level failure. Test the actual log scanner against
both real marker spellings and benign startup lines before native execution.
and an idle boundary exceeding the unchanged30-second timed-data threshold.
If gates pass, repeat from a fresh service with the same35-second idle and full
sequence. Archive ALL receipts and exact recipes on controller/head. Do not
reconnect/reset hosts, raise limits, disable error checks, or convert this gate
into arbitrary-context/concurrency or general transport-failure qualification.
