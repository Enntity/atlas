# First bounded C4-MTP native serving result

2026-09-09, source `6e1e37f4`, TP2/EP2, two DGX Sparks. One fresh process,
one warmup plus three measured waves at each width. Fresh-process repetition
is still required. This establishes correct four-owner serving on the checked
workload, not a throughput improvement or full reference parity.

| Width | Aggregate full-wall tok/s | Decode-window tok/s | Median client TTFT ms | Median server TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| C1 | 27.309 | 28.704 | 457.432 | 428.060 |
| C2 | 36.891 | 38.151 | 687.954 | 426.926 |
| C3 | 33.054 | 33.716 | 881.870 | 426.096 |
| C4 | 36.894 | 37.522 | 1098.774 | 425.540 |

Measured full-wall waves: C1[27.309,27.296,27.382],
C2[36.972,36.833,36.891], C3[33.131,33.054,33.002],
C4[36.908,36.894,36.867]. Every request reaches the256-token output cap.
The prompt remains the frozen148-token LRU completion with temperature0/seed1.
Client TTFT includes queue/admission effects that server TTFT does not fully
represent; report both. No fresh long-prompt prefill-rate benchmark was run.

C1/C2 preserve the previous repeated selected-MTP performance envelope. C4
does not improve aggregate throughput over C2, and C3 is slower: the scheduler
executes separate pair/single target traversals. This supports prioritizing
wider layer-major weight reuse, not further admission flag changes. These
C3/C4 rates are below the earlier nonspeculative approximately37/47.3 rates;
preserve that profile as a separate baseline. None of the strict30/37/43
targets is established here, and C6/C8 concurrent MTP remains outstanding.

## Quality and safety

Before timing, four simultaneous distinct answer requests pass exact arithmetic,
stable sorting, constrained Python-function structure and JSON checks. Four
distinct structured auto tool calls pass exact names/arguments/finish checks.
Four768/800/832/864-token needle prompts at early/middle/late positions return
their own identifier without any of the three peers' identifiers. This is
bounded short-context evidence, not broad quality or large-context qualification.

All40 coding streams, including warmups and widths1..4, retain the identical
256-token output seen in source290cf248. C1/C2 complete output sequences also
match that prior receipt. The capped program is not executed or certified.

Context remains2044, prefill1024, BF16 KV, BF16 MTP4, FP32 state, eager selected
serving, and JointSharedM10 arithmetic. Worker logs confirm four retained slots;
both ranks log actual JointSharedM10 E6 commits. The process still computes
two-owner pairs, not M20. Preflight logs7231MiB inference reserve and637MiB
arena. Controller minimum available memory is10,354,384KiB on rank0 and
9,136,204KiB on rank1; maximum observed swap is0 on both. Keep the4GiB guard.

Workload and controller exit0. Both actual quiescent/release exchanges complete;
independent Docker inspections show both ranks exited0, OOMKilled=false and
RestartCount=0. Both nodes are idle afterward. Temporary helper sudo rules were
withdrawn into recoverable root-owned campaign files; images/helpers and all
evidence remain available. No node reset or driver change occurred.

## Pinned artifacts and receipts

Campaign root:
`/home/abc/storage/models/atlas-campaigns/20260909/glm-native-controller`.

- Source archive `native-6e1e37f4-source.tgz`:
  `bf1082dc6a34e486fa464dfc92fa459eff764ef9b98225fe62803a3c39f975cc`.
- Server ELF:
  `9a1afb09c1a71b1e5121c342d29dcad0e719f3688d72202d8b12b7dd60f7fc8a`.
- Guard ELF:
  `d36a20d040ce1e96810e0b62372d9c0b520466d9e38e100d315837191502bb39`.
- Supervisor ELF:
  `8a49af1e04134d5cd24a5bcb61900b3199684528ea405e6c8cb340f103b7ba37`.
- Relay ELF:
  `72d83ae1919aadf3c5d8bced96071acae17b841d184073ed0de27293cf1d23c9`.
- Head image:
  `89eae53e521782363b2d3fd97b0cb03b85a062bdefc9b9872d061c152d16f2aa`.
- Worker image:
  `47a72e3b0e2566dcc86ba0662d0c53e5e04865de7633e35a5cc2b637b2c38ca9`.
- `native-workload-c4.py`:
  `e659e3a44b3146d7f2fa05f24ea605f150052f04dd8f64ed4ff4ca1b3cabac1c`.
- Prepared directory `native-prepared-6e1e37f4-c4-joint-shared-m10-first`;
  bundle `b92e606c7ffb3bc27e4b9d9709438991231a7ce2e566f1641d61252e0da32ae0`.
- `native-c4-first-summary.json`:
  `396ab37e5b106e38d4d048a56f992270d605ab8a8b1f236daa6f4967097e1414`.

Retained files include the prepared directory's raw workload/evidence frames,
`native-c4-first-{head,worker}.log`, both `-inspect.json` files,
`native-c4-first-output-comparison.log`, source hash/install receipts, native
build/image receipts and helper withdrawal logs. Head container:
`bd16981b7f1b761fd9184b4202a9a6b86208fbaba3c62c61cfd3d7d352931d38`;
worker:`727c6728f8decdba99144627623d546fd48c3791ba13b27babedd98e7d6b551c`.

Next: fresh-process regression repetition, then the checked wider producer,
workspace and layer traversal in `mtp-owner-batch-implementation-plan.md`.
Keep the full C1/C2/C4/C6/C8 goal and large-context follow-up intact.
