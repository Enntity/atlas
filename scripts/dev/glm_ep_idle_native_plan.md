# Explicit EP idle receive: bounded native validation

Root execution plan, 2026-09-08. No native execution until the separate
`ep_idle_command_receive_plan.md` implementation has actual-entry behavioral
RED/GREEN, full CPU and NCCL-feature compile checks, independent exact-source
review and a committed freeze. The current v25 result is retained unchanged;
the old376.9s/50.5s/32.9s health errors are never relabeled as clean.

## Frozen candidate and unchanged limits

Reserve v26 for the reviewed idle-command classification fix. Record its exact
commit before building. It may include the independently committed GLM
checkpoint-retirement infrastructure, but that code has no enabled whole-model
loader selection and no resident B-tile publication. Do not include unfinished
reader/activation files. Archive committed spark-comm and spark-model source
for the existing stopped native Rust builder; retain enabled CUDA189db87e.
This gate does not promote the separately compiled B-tile kernels.

Root verifies BOTH nodes idle before build/package: no model or standalone GPU
workload, native Rust builder8GiB/two CPUs/jobs2. Preserve v25 runtime and build
an executable-only v26 image over it, verifying the same new executable SHA256
inside both images. No reset, forced kill, sudo, clock or swap changes.

Use the exact v25 C1 recipe with image changed only: TRACE0/CACHE1/VERIFY0,
MTP4 repair, one admitted/active request, context2044, prefill1024,114GiB limit,
4096MiB guard, utilization.91, KV overcommit0/swapspace0, same graph/overlap and
numerical settings. Refuse existing standard names and any prior receipt tag.
Require at least4096MiB host MemAvailable on each node at readiness.

## Native idle discriminator and normal workload

Prepare the entire runner before launch. At readiness record both timestamps
and memory, then deliberately leave the service without requests for35 seconds.
Use a bounded asynchronous runner so root can report progress during that
interval. This exceeds the unchanged30-second timed-collective threshold, but
the outer worker command receive is now explicitly classified as idle.
No keepalive request is permitted inside the interval.

Afterward run the unchanged literal148/256 benchmark, one warmup plus three
measured C1 requests, temperature0/seed1, no forced-cap/repetition override.
All measured outputs must reach256. Preserve raw full-wall rates separately
from session/decode-window metrics; any timing comparison is against v25's
28.632 initial/28.726 fresh-repeat baseline, not a30-tok/s success assumption.
The intentional pre-request idle interval is not included in client request
wall time. Do not claim this tests arbitrary idle durations or transport loss.

Then run four answer checks,1984/16 needle, deliberate two-second stream cancel,
and four recovery checks, sequentially. Do not change the idle35 duration,
threshold, workload, output policy or guards after seeing a failure. Preserve
complete both-rank logs before and after graceful30-second-timeout shutdown,
and verify running=false, exit0, OOMfalse and swap0. Keep renamed containers.

Require no new communicator-unhealthy, diagnostic, CUDA or ERROR-level message
in the complete run. The worker-ready-to-first-command interval must visibly
exceed30 seconds. HTTP readiness alone cannot prove communicator health.
The actual protocol tests and NCCL adapter checks must independently prove that
follow-on/data broadcasts retain timing/error behavior; never manufacture a
physical transport stall to test a timeout on these Sparks. No blind reconnect.

If the idle and normal gates pass, perform a fresh confirmation using the same
35-second idle and full request sequence before updating a bounded serving
recommendation. Archive all attempts and exact recipes on controller/head.
This does not implement an interrupting watchdog, connect the unused health
latch to HTTP, validate multi-turn carry, enable C4 speculation, or establish
30 C1/60 aggregate C4 unless the unchanged full-wall measurements actually do.
