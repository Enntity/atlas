# Independent C2–C8 native serving qualification

2026-09-09. Runtime source `c0f7b0efac262e7f1a1baf7031306587c517e609`;
binary SHA256 `780e3ee33774061e9e18528beb7ba6ead7b1c93bcc41d33ec473589b233d8ca4`.
Both Sparks use this exact binary in `atlas-glm53-flash:kernel-20260909-v29`.
The CPU-only release build completed successfully with all 191 target kernels.
The previous v28 deployment remains recoverable and unchanged.

## Fixed configuration and scope

Two ranks, TP2/EP2-v2, independent decode selected, active/admitted cap8,
context2048, prefill1024, BF16 KV, FP32 KDA state, nonspeculative.
Temporal K5 HC and legacy per-width experiment selectors are off. Memory
utilization0.90, container memory=memory-swap114GiB, allocation guard4GiB,
KV overcommit off and swap budget0; existing watchdogs/rollback rings retained.
No competing native build, model or microbenchmark ran during qualification.

This exercises new independent-row KDA/MLA/grouped-MoE dispatch through eight
streams. It does not activate concurrent MTP4 or establish full reference parity.
The reference's original workload/denominator is not published; our literal LRU
completion workload remains an internal regression benchmark.

## First eager qualification — passed

Multi-sequence graphs off; C1 retains its existing scalar graph behavior.
One warmup and one measured batch per width, literal148-input/128-output coding
prompt, temperature0/seed1, ordinary EOS and repetition policy. This is a short
occupancy/correctness run, not the repeated256-output headline benchmark.

| C | Aggregate full-wall tok/s | Median per-stream decode tok/s | Client TTFT ms |
|---|---:|---:|---:|
| 1 | 13.174 | 13.700 | 445.068 |
| 2 | 23.875 | 12.717 | 691.783 |
| 3 | 34.011 | 12.278 | 866.657 |
| 4 | 42.317 | 11.606 | 1150.700 |
| 5 | 50.635 | 11.252 | 1351.736 |
| 6 | 55.936 | 10.451 | 1571.622 |
| 7 | 61.697 | 9.969 | 1774.733 |
| 8 | 66.535 | 9.481 | 1986.730 |

All72 warm/timed completions reached128 tokens with retained text, matching
byte counts and SHA256. Four unique outputs are coherent, truncated LRU-cache
code beginnings; this is not complete-program correctness. Identical coding
prompts cannot alone establish cross-request isolation.

Distinct visible answers and forced named tool calls passed at every C1..8
before and after the coding run. Two eight-request NIAH waves before and after
passed own-needle retrieval and no-foreign-needle checks with mixed prompt/output
lengths. NIAH is a substring retrieval gate, not exact-format validation: all16
before outputs contain extra text, often repeating answers/questions or filler.
Do not describe that result as clean answer-only termination or broad quality.
A streamed request was cancelled after2s, followed by passing C8 answer and
tool recovery waves. Tools validate actual structured API output/arguments;
they do not execute external actions or prove automatic tool selection.

Both ranks have exactly the same ordered3209 batch-entry records and unique
slot IDs in0..7. Counts by actual width: C2=597,C3=606,C4=421,C5=428,C6=371,
C7=359,C8=427. Narrowing proper-subset slot transitions include every adjacent
width8→7→6→5→4→3→2. Entry traces alone are not kernel-completion proof; the
successful requests and final logs provide the accompanying evidence. C1
bypasses this logger and is covered by its separate successful requests.

Ready system MemAvailable was7835/7941MiB on head/worker. Later sampled values
were7179/7672MiB during high-concurrency quality and7184/7648MiB after the coding
matrix. These are observations, not continuous high-water measurements. Every
observed swap-use value was zero. Both ranks stopped exit0/OOMfalse, released
their GPU processes, and recovered118809/118484MiB available. Native error scan
and overall gate exit0. Logs retain the existing padded-vocabulary-stride and
allocation-ownership teardown warnings; absence of fatal errors is not proof
that those warnings or ownership accounting have been resolved.

## Graph-enabled repeated qualification — passed

Same binary/configuration with multi-sequence graphs enabled,
148-input/256-output workload, one warmup plus three measured batches at every
C1..8. Medians exclude warmups. These are repeats within one fresh process;
a second fresh-process graph repetition is still required for reproducibility.

| C | Full-wall tok/s, median | Three measured batches, tok/s | Per-stream decode, median | Client TTFT ms |
|---|---:|---|---:|---:|
| 1 | 13.410 | 13.409 / 13.410 / 13.428 | 13.685 | 454.447 |
| 2 | 25.589 | 25.646 / 25.589 / 25.573 | 13.229 | 696.073 |
| 3 | 36.732 | 36.732 / 37.320 / 36.672 | 12.777 | 941.856 |
| 4 | 46.839 | 46.958 / 46.839 / 46.758 | 12.327 | 1163.055 |
| 5 | 56.227 | 56.227 / 56.118 / 57.069 | 11.919 | 1363.944 |
| 6 | 63.585 | 63.585 / 63.666 / 63.433 | 11.296 | 1577.583 |
| 7 | 70.549 | 70.549 / 71.421 / 70.418 | 10.801 | 1787.337 |
| 8 | 76.923 | 76.894 / 76.926 / 76.923 | 10.360 | 2000.435 |

Client TTFT includes batch/request waiting; server-reported median TTFT is
approximately420–422ms. The full-wall denominator starts before request launch
and ends at the last completion. Do not replace it with the smaller post-first-
token window to call C6 a pass: that alternative gives64.543tok/s atC6 and
77.948 atC8, whereas conservative full-wall C6 remains0.415 below64.

On this internal workload C8 exceeds72 by6.8%, and C4 exceeds43. C1/C2 remain
below30/37. C1 is nonspeculative, not the separate earlier MTP4 result. C2's
measured pairs all contain the earlier mixed hashes `72891e8fd4d5…` and
`f79f067e466f…`, preserving the conservative approximately25.5 baseline rather
than matching the faster identical-output pairs. C4 is0.9% below the earlier
47.273 result; capacity and independent dispatch differ, so this is not a
single-kernel A/B. No claim of complete reference parity or goal completion.

All144 completions (36 warmup,108 measured) reached256 tokens, with hashes and
UTF-8 byte counts independently verified. Five unique texts are coherent,
truncated LRU implementations without obvious looping/gibberish. Outputs differ
across widths and sometimes within a width; this is not bit-exact parity or
full-program correctness. All56 summary medians were independently recomputed
from measured batches, excluding warmup.

Before/after C1..8 distinct answers and forced tools, both two-wave eight-needle
retrieval checks, and cancellation followed by C8 answer/tool recovery passed.
The same NIAH formatting and tool-choice limitations noted above apply.
Both ranks have exactly matching ordered8550 batch-entry records, with counts
C2=1325,C3=1364,C4=1218,C5=1195,C6=1130,C7=1129,C8=1189. Native capture logs
confirm graphs at every width2..8, including noncontiguous slot vectors; capture
counts by width are21/26/23/15/14/8/1 on each rank. Narrowing subset transitions
include every adjacent width8→7→6→5→4→3→2. These observations accompany successful
requests; entry/capture logs alone are not numerical proof.

Late-run sampled MemAvailable was6411/6841MiB, swap0. Both ranks stopped
exit0/OOMfalse, with no remaining GPU applications, recovered118798/118847MiB
available and zero swap use. Native error scans and overall graph gate exit0.
As with eager, sampling does not prove the exact minimum free memory. No node
reset, reboot, driver/clock/swap changes or native fault injection was used.

Next: repeat the warmed graph matrix in another fresh process, retaining output
mix; then integrate actual supervised paired MTP4 serving. The nearest missing
pieces are the guard/child channel and two-rank quiescent release, actual Model
registration/paired construction, and selected head/whole-worker admission and
shutdown—not another implementation of the already committed cold F0 or serial
K5 transactions. First qualify the connected fixed-profile C2 control, then
optimize its batching and widen concurrency; do not claim the serialized control
alone satisfies the full C1/C2/C4/C6/C8 reference objective.

## Evidence

Controller campaign:
`/home/abc/storage/models/atlas-campaigns/20260909/glm-independent-c2-c8`.
Receipt prefixes `v29-c8-eager-` and `v29-c8-graphs-`: matrix JSON includes all text/counts,
per-request timing and warmup records; before/after quality JSON and NIAH JSONL,
live configuration, binary hashes, ready memory, both final rank logs, shutdown
and gate-exit receipts are retained. Stopped containers are preserved as
`atlas-glm53-v29-c8-{eager,graphs}-ep{0,1}` (each on its respective rank node).

Root gate SHA256 `125522ccf43cf356d6bf324ac773a2cfc496c639dc0622098ff9ed2ec54ae7fb`;
measurement runner `65939c9e72948262afe9a5f0d4a65c284849f877aab4cc8f042e998f63c7b064`.
The global native lock, exact binary/config checks, memory guard and
evidence-preserving shutdown apply to both arms. No full-suite or release-matrix
claim follows from this GLM-specific experiment.
