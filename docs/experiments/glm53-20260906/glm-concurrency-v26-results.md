# v26 warm C1/C2/C3/C4 matrix and native EP-v2 idle gate

2026-09-08. Two fresh services passed the unchanged nonspeculative active4
profile, warm fixed-output matrix, four-answer smoke check, mixed independent
needles, cancellation/recovery and automatic clean shutdown. This closes bounded
native-v2 idle coverage; it does not enable concurrent speculation or B-tile.

## Frozen engine and workload

Image on both ranks: `atlas-glm53-flash:kernel-20260908-v26`.
Rust e40a9066673057ceb99f76abddc8ae0d29826272; CUDA
189db87e0b7ce22e262643a160adf3e51263b203; executable SHA256
`a29c261991b94309a2eb4193f3354a429cb60f77d851604b62fe1f3ea5538cbd`.
The later resident-reader commit4416ede2 is NOT in this image.

Both actual commands/environments: TP2/EP2, EP v2, no speculation,
context2048, prefill1024, active/admitted4, utilization.90, BF16 KV, NVFP4
target head,114GiB memory and memory+swap ceilings,4096MiB guard, swapspace0,
KV overcommit0. C3/C4 grouped MoE, MLA batch23/batch4, indexed WMMA, FP4 prefill
and multisequence graphs ON. C4 sparse/indexed sparse modes, C1 shared-FP8
cache, M16 arms, distributed MTP, repair and hidden/ledger diagnostics OFF.
The frozen launcher is the same4286b334... phase7 launcher used for C1.

Root used `run-v26-c4.sh` (SHA1ae8a59cfbf391a66dd7ad0f8f3bf300c3d95bb19eb4ae5b8aba8ce69fc0f5e3)
and controller `run-v26-c4-gated-native.sh`
(SHA28cdc972a99b9340ab29bf22e93ddc055996e4f471188bf0403ba43fd8e4d571).
The latter now sweeps offered widths1..4 on this SAME active4 profile. It checks
actual configuration, output counts, memory, error logs and stopped/OOM state;
existing serving/receipt names are refused. The reviewed plan and log gate are
committed257937a4. Log scanner CPU tests execute real benign-startup RED then
4/4 GREEN, and correctly classify retained clean/faulted native logs.

Literal LRU148-token prompt,256 requested output tokens per stream,
temperature0/seed1, normal EOS/repetition behavior. Each width has one warmup
then three measured waves. Initialization is excluded; warmup receipts are not
retained by this harness, so independent warmup-cap verification is not claimed.
Every one of60 measured streams across both launches reaches256 tokens.

## Warm rates

| Offered concurrency | Initial full-wall aggregate | Fresh full-wall aggregate | Initial decode-window aggregate | Fresh decode-window aggregate |
|---|---:|---:|---:|---:|
| C1, nonspeculative |13.478|13.468|13.812|13.806|
| C2 |19.067|18.976|19.392|19.301|
| C3 |35.037|34.828|35.774|35.556|
| C4 |47.440|47.222|48.451|48.222|

Full-wall measured runs, initial then repeat:

- C1:13.491/13.475/13.478;13.459/13.468/13.487.
- C2:19.095/19.067/19.056;19.005/18.976/18.973.
- C3:35.037/35.041/35.022;34.828/34.874/34.759.
- C4:47.318/47.440/47.474;47.168/47.246/47.222.

Median per-stream server decode rates, initial/fresh: C1 13.759/13.753,
C2 9.765/9.718, C3 12.160/12.085, C4 12.481/12.424. These are not aggregate
full-wall metrics and must not be multiplied into a replacement acceptance rate.
The separately qualified MTP4 C1 result remains28.699/28.618; that profile is
not the C1 scaling control in this table. Neither30 C1 nor60 C4 is established.
Historical v13 C4 47.319 is effectively similar, not evidence of v26 speedup.

Output hashes are retained per request. C1/C2 use one common hash09db1ed1...
in both launches. C3 has two hashes with counts6/3 per launch; C4 has two with
counts4/8 initially and3/9 on repeat. No cross-width or universal text-bitidentity
claim is made. Correctness smoke checks below remain separate from performance.

## Native protocol, memory and lifecycle

Deliberate no-request idle windows after readiness:17:11:24–17:11:59 UTC and
19:17:49–19:18:24 UTC,35 seconds each. Both actual rank environments select v2.
These are controller idle boundaries, not independent NCCL-call instrumentation.
No idle unhealthy latch, ERROR-level message, CUDA fault, panic or hidden/ledger
record appears in either complete final log. Both gate exits are0.

Ready MemAvailable head/worker:11782/11779MiB initial;11746/11889MiB repeat.
Both launches pass four strict visible-answer checks at concurrency4, then two
independent-needle repetitions with prompts768/800/832/896 and caps32/16/48/64.
All needles correct; no foreign needle. The832-token request naturally emits14
tokens in repetition1 on both launches; these quality requests are not fixed-
output performance samples. Cancellation returns curl28 after4125 received
bytes on each launch, followed by four passing concurrent recovery answers.

Both complete rank logs have exactly equal ordered batch-width/slot records:

| Launch | N2 records/rank | N3 records/rank | N4 records/rank | Non-prefix records/rank |
|---|---:|---:|---:|---:|
| Initial |1062|1071|1104|98|
| Fresh |1061|1083|1092|108|

Observed non-prefix subsets include[0,2],[1,2],[1,3],[2,3],[0,2,3],[0,1,3];
the repeat also has[1,2,3]. This proves actual multirow work and observed drains,
not just four HTTP clients. All observed slot vectors are ascending; arbitrary
permutations are NOT natively qualified. N1 delegates to scalar decode and
does not emit this batch logger. These records do not prove arbitrary mixed
prefill/verification scheduling or every cancellation point.

Initial stops17:18:36/17:18:38; repeat19:25:03/19:25:05 UTC. Every container
reports running=false, exit0, OOM=false; host swap used0. Post-stop available
head/worker118778/118843MiB and118790/118847MiB. Preserved containers are
`atlas-glm53-v26-matrix-initial-ep0/1` and
`atlas-glm53-v26-matrix-repeat-ep0/1`. Both nodes are stopped after this gate.

## Interpretation and next work

C2's small full-wall/decode-window difference does not explain its gap against
the user's MTP4 vLLM reference. Actual Atlas KDA and MLA mHC callers keep C2
FFNs scalar, whereas C3/C4 use grouped execution. The vLLM table also uses
concurrent speculation, unlike this profile; it lacks matched workload details.
See [C2 speculation plan](../../../scripts/dev/glm_c2_speculation_plan.md):
per-request hidden/KV ownership, serialized C2 correctness, then segmented
two-session verification. No unsupported guard is bypassed for these results.

Raw persistent controller receipts: `atlas-campaigns/20260908/v26-matrix-*`.
Initial matrix JSON SHA256
`bc0c776482c2fda795a5f67234f8e25c8d383534f049aa3fe26f63316fd734aa`;
repeat `d59afa0176a829613fa8b4b853449b54f3a5b73956d861382ebe18abe02d9e1c`.

Closed archive `v26-concurrency-verified-receipts.tar` (15MiB) is verified on
controller and head phase7, SHA256
`7b1718f1e784107020b3b12545ff4a3756b7c36b39db19bb3d93df9a207165a5`.
It includes both complete runs, preflight/fault-test receipts, frozen recipes,
committed harness/fixture sources and the preceding v26 engine-provenance archive.
