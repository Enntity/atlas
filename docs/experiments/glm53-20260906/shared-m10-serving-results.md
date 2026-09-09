# Same-kernel shared M10: native serving qualification

2026-09-09, source `290cf248`. Explicit default-off
`ATLAS_GLM_C2_PAIR_FFN=joint-shared-m10` retains the qualified Joint routed
FFN/dense down path and replaces two generic-T M5 shared chains with one M10
chain. No new kernel, numerical policy, workspace allocation or serial C1 path.
Both ranks compare mode word3 in E6 before worker target computation.

The standalone full-FFN bitwise/memcheck evidence is in
`paired-followup-kernel-results.md`. Runtime dispatch has a genuine M5-vs-M10
RED/GREEN check. Actual Model/worker continuation passes all25 intra-pair
accepted lengths and both owner orders; mode mismatches refuse without worker
target writes. These CPU fixtures are not native all25 attention rollback proof.

## Fresh-process comparison

The unchanged internal148-input/256-output workload uses one warmup and three
measured waves at each width, eager MTP4, two slots, context2044/prefill1024,
BF16 KV, FP32 SSM and114GiB no-swap container limits.

| Mode/source | C1 full-wall tok/s | C2 aggregate full-wall tok/s | C2 decode-window tok/s |
| --- | ---: | ---: | ---: |
| Same-binary Joint,290cf248 control | 27.207 | 34.243 | 35.329 |
| Shared M10,290cf248 first | 27.298 | 36.957 | 38.221 |
| Shared M10,290cf248 repeat | 27.243 | 36.763 | 38.042 |

Candidate measured C2 waves37.000/36.856/36.957; warmup37.016. The median is
7.9% above the same-binary Joint control but **still below37**. The control
also passes all quality, output equality and two-rank release/exit checks.
Run order is candidate, control, candidate. The repeat measures
36.763/36.828/36.745, warmup36.842: median7.4% above control and0.52% below
the first candidate process. Do not round up the strict target or claim exact
MiaAI workload parity. Relative to the prior TwoK5 control27.320, these candidate
medians improve34.6–35.3%; that longer comparison spans two source binaries.

All four answer, two strict auto-tool and two retrieval checks pass. Full
answer/tool/retrieval/coding outputs match the same-binary and previous Joint
processes and both candidate repetitions. Each C2
coding wave again has65 paired commits with diagonal acceptance counts
4/5/12/13/31 for0/1/2/3/4 accepted drafts. The gain is not changed acceptance.
Every coding output reaches256 tokens. Generated programs are not certified.

Controller exit0, actual two-rank quiescence/release and independently inspected
Docker exit0/OOMKilled=false/restarts0. Minimum available host memory
10,270,296/9,045,904KiB, zero observed swap. Both nodes recover about116GiB.
The same-binary control minimum is10,444,632/9,931,160KiB with zero swap;
its C2 measured waves are34.243/34.215/34.253.
Candidate repeat minimum is10,362,368/10,204,820KiB, again zero swap, normal
controller/Docker exits, no OOM/restarts and actual paired release on both ranks.
Both rank logs confirm `Some(JointSharedM10)` E6 commits. Server TTFT medians
are426.506ms first and426.374ms repeat at C2; client-observed median TTFT is
670.372/680.406ms respectively (queued admission included).

## Provenance

Evidence root `/home/abc/storage/models/atlas-campaigns/20260909/glm-native-controller/`.
First receipts/logs: `native-prepared-290cf248-joint-shared-m10-first`,
`native-{summary,head,worker,run}-290cf248-joint-shared-m10-first.{json,log}`.
Bundle `aeb31562dac57141a68b26d2ee62e5428e76aa7695e8bc40080ca7e9284a9da5`.
Control receipts follow `290cf248-joint-control`, bundle
`32e4ed04dbadb4fbf5c78fbd398f3bf83cc9219a922c7373a6a2b2adb6a79eb6`.
Repeat receipts follow `290cf248-joint-shared-m10-repeat`, bundle
`1b618338d875260a9ddfb45edf946344f8aac90cfdc6704bce22e5f8ca120be9`.
Server ELF `6f4498fcee1037ed0f5e9ce9bae89ed1be7bb5868ea236ef59753de78e7c54cd`.
Head image `f4145d9158e09e635eda15448072b97dfa5afeb306f446c8b989ba4029053a2f`;
worker image `61b9f746cf73fee4e30d9069d13e87052f2ef13552dab5d7f575ae8d21fd5271`.
Unchanged digest-pinned9aca8d8f guard/relay/supervisor. Source archive and complete
model-crate hash readback retained as `native-290cf248-{delta.tgz,source.sha256}`
and `native-290cf248-source-install.log`. Native release build exits0.
After all three campaigns, the temporary9aca8d8f sudoers rule was withdrawn on
both nodes by moving the exact verified rule into the corresponding retained
`phase7/paired-9aca8d8f/withdrawn-sudo-rule-9aca8d8f` asset. Images, helpers,
stopped containers and evidence remain recoverable; no model is left resident.

Selected C3..C8 MTP remains unavailable. See `mtp-c3-c4-next-plan.md` for the
next ownership/residency and cross-request batching work; no full-goal claim.
