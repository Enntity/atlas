# v23 initial-prefill no-overlap control

2026-09-08. Disabling shared-expert overlap changed the observed divergence
distribution but **did not eliminate it**: cross-rank final-hidden differences
fell from 192/192 sampled step pairs in the earlier run to 32/192, all in
one request. This single cold-load control does not establish a race, a cache
bug, a reliable remedy or a throughput improvement.

## Controlled scope and complete evidence

The [control plan](../../../scripts/dev/glm_mtp_overlap_control_plan.md) changes
exactly `MOE_SHARED_REDUCE_OVERLAP=1` to `0` in the prior v23 recipe. The actual
predicate requires EP shared work, more than 64 rows, no graph capture and no
profiling: this affects the initial 148-token target prefill, **not K5 verifier
overlap**, which is already sequential. Source event_a/event_b joins exist;
no missing synchronization has been demonstrated.

The intervention changes placement/order as well as concurrency: flag0 runs
the shared expert before routed compute (`forward_prefill.rs:189`), while
flag1 runs it after routed compute alongside the routed collective (`:468`).
It therefore does not isolate auxiliary-stream concurrency alone. Read-only
source audit found no obvious missing join or base-pointer alias; that negative
inspection is not proof of race freedom or equality of private state.

Root retained the v23 executable from `1f675c8b`, SHA256
`a5b9e20c11a12bf24d9a2a3fb226b0722cd1507d58df4b248bed1a818e96bed9`.
Both effective environments show cache enabled, cache VERIFY disabled,
target verification graphs enabled, hidden tracing enabled and overlap disabled.
The run kept C1, four drafts, accepted-pair repair, the literal 148-token prompt
and 64-token output cap, with one warmup then five sequential repeats.

The two complete logs were frozen before quality requests. The manifest assigns
all six generations 1..6, slot0 on both ranks; nothing is extracted to hide
unmatched trace records. Each rank has 192 valid version2 records: eight
attempts × four steps × six requests. Independent replay using the committed
v2 analyzer from `1f675c8b` exactly reproduced both the baseline and control
JSON reports. No missing, duplicate, foreign-owner or schema/chain failure.

## Cross-rank distribution

Counts below are paired steps, not independent experiments or generated tokens.

| Run/request | Compared steps | Input-hash differences | Final-hidden differences | Equal sampled post-EH pairs |
|---|---:|---:|---:|---:|
| Earlier overlap-on, all six requests | 192 | 144 | 192 | 48/48 |
| No-overlap warmup, generation1 | 32 | 0 | 0 | 8/8 |
| No-overlap repeat1, generation2 | 32 | 24 | 32 | 8/8 |
| No-overlap repeat2..5, generations3..6 | 128 | 0 | 0 | 32/32 |

All 192 control pairs agree on chosen draft, gathered argmax-pair payload and
reported causal metadata. In exceptional repeat1, every step0 has equal input
and post-EH hashes but different final hashes. Its 24 later steps consume the
previous differing final rows, so their classification is input difference.
Post-EH is not sampled on those later steps; absent hashes are not equality.

The five agreeing requests also agree with each other on all compared hashes,
tokens, pair payloads and causal metadata in all seven declared comparison
edges that avoid repeat1. This does not prove
equal private KV, static weights or unobserved scratch.

## Earliest exception and changed drafts

At generation2/attempt1/position149/seed1304/step0, cache148→149 and hidden_row0:

- Both input hashes: `6c03a73705a818c124de98781562cbaf9fa239cdf3ba860ce4f1f1694ad75965`.
- Both post-EH hashes: `1970b738b46101eed56d8b0989801578c713a42d2fd312d8399da94e03c6ea27`.
- Rank0 final: `f581529b85f08ce53e7ef938790c5e64c11cc83d7e934a03402e2e948aef93b5`.
- Rank1 final: `b2cdc71699edf14687cef52097bb437c42923e7b058d57a380e05a26f9231ec3`.

That rank1 final is also the agreeing control warmup's final on both ranks and
the earlier overlap-on warmup's rank1 final at this exact key. The earlier
overlap-on rank0 final was
`ea9724864b0d65f6981ded09e923ec8d5e5f841dbc893179c33ca6a036a58009`.
The control's first exceptional proposal therefore differs after the measured
EH boundary, before any later differing draft can explain that first result.
Its existing private prefix remains unobserved in version2.

The first changed draft between control warmup and repeat1 is attempt1/step2,
still position149/seed1304: 284→49449 on both ranks. Rank1's input and final
hidden hashes remain identical across those two requests at that step, but
the gathered shard0 pair changes (17.125,284)→(18.875,49449); shard1 remains
(9.5625,3799). These are gathered shard-local pairs, not an assumption that
rank-local maxima should match. Subsequent selected inputs/histories change.

## Cross-request matching caveats

The two declared edges involving repeat1 (warmup↔repeat1 and repeat1↔repeat2)
match only **five position/seed attempts, 20 steps per rank**, not all 32.
For warmup→repeat1, unmatched keys are:

- Warmup: (153,3489), (172,1376), (177,982).
- Repeat1: (152,284), (154,792), (173,5856).

At matched position157/seed957, warmup attempt3 and repeat1 attempt4 carry
hidden_row3 versus2, affecting all four metadata comparisons. Position/seed
matching does not establish identical causal history. On each exceptional
edge, rank0 differs on 15 input and 20 final hashes; rank1 differs on 12 input
and 17 final hashes. Each rank has five changed drafts and 20 changed gathered
pair payloads among these matched steps. All five sampled post-EH pairs agree.

Across the nine declared edges there are 66 post-EH comparisons per rank, all
equal; these reuse the existing 96 captured post-EH records, not 132 new
observations. Comparing baseline and control warmups likewise matches only
five attempts: do not pair all rows across runs merely by attempt number.

## Quality, safety and interpretation

All five measured client requests reached 64 tokens with identical completion
text SHA256 `1bdda4caa55ae2df2226c4c3c080d7ab23a6ca80ba56488a1de6b75e0ee2f669`.
The complete logs additionally cover the warmup. Four answer checks passed;
the 1,984-token needle case passed. A deliberate two-second client cancellation
returned curl28 after 9,772 response bytes; four subsequent recovery answer
checks passed. Both preserved containers then stopped with exit0/OOMfalse.
These are bounded health gates, not broad model-quality certification.

The evidence supports an execution/history-sensitive difference, not a proven
overlap cause: the exception persists even with this overlap disabled. The
next approved probe observes the valid private prefix and appended row only
at attempt1/step0; no broad reset or synchronization fix follows from this
control. Trace synchronization invalidates all throughput comparisons, and
none of this run's timing qualifies the 30 C1 / 60 C4 goal.

## Frozen receipts

All files are under `/home/abc/storage/models/atlas-campaigns/20260908/`, with
prefix `v23-nooverlap-hidden-`. Baseline evidence and limits are in
[the v23 post-EH report](glm-mtp-hidden-trace-v23-results.md).

| Suffix | Bytes | SHA256 |
|---|---:|---|
| `clean-rank0.log` | 323188 | `11fa0843bad309165b007d5081e67af936e4241a81a01c771dab5cda02cec71d` |
| `clean-rank1.log` | 297907 | `49ad3511a2fc8a800f9cd06344ae666a528c78996e46f6339d930c2fe4b1958e` |
| `analysis-manifest.json` | 1796 | `848efe77554aaab4e6eb32f48de656fdfbfe07f0f9c377fd69311f42cb90d23d` |
| `analysis.json` | 1646384 | `bc5a3d52091e57a3610a363ad2969f540e13ef19df0c2c2b9f628e39e41bebd3` |
| `c1-64.json` | 5865 | `0f17a1bf1979be6572759386591199c67f826a59e3a8d20f883980845c70ba1c` |

Additional receipts: `effective-env.log`, `live-identity.log`,
`load-memory-both.log`, `answers.json`, `niah.json`, `cancel.stderr`,
`cancel-response.txt`, `recovery.json`, `stop-head.log`, `stop-worker.log`.
