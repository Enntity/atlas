# v25: ordered target-to-drafter prefill

2026-09-08. The first six-request diagnostic has complete cross-rank and
repeat-request agreement after stream fix `d3f989c7` and separate v4 probe
`aa2b7e16`. This is a correctness observation, **not throughput qualification**.
It also contains a worker communicator-health error after a long idle period;
the complete receipt is retained and is not called a clean health pass.
A third, fully automated diagnostic passes the complete health/quality sequence
with the same complete agreement. Separate trace-OFF148/256 full-wall medians
are28.632 initially and28.726 on a fresh restart. The30 C1 target is not met.

## Exact executable and profile

Rust source: `d3f989c73003a1ac792e0a7993c29435694d8f4a`.
Enabled CUDA remains `189db87e0b7ce22e262643a160adf3e51263b203`.
Both nodes' `atlas-glm53-flash:kernel-20260908-v25` executable SHA256:
`6166a51d36256860bc176e02debdbba4f689ba9181333ba050daafac5364f51e`.
The separate promoted B-tile CUDA and unfinished resident integration are not
enabled. Only frozen committed model source was overlaid in the stopped native
Rust builder. Its 8GiB/two-CPU/jobs2 build finished in2m12s, exit0/OOMfalse.

The v24 recipe changed only its image: TP2/EP2 C1, four-draft accepted-pair
repair, shared cache ON, overlap ON, existing target verification graph,
BF16-drafts1, context2044, prefill1024,114GiB containers,4096MiB guard,
KV overcommit0 and swapspace0. Hidden trace/ledger ON invalidates its timing.
The independent stream fix resolves the existing effective stream in three
public prefill wrappers before both target dispatch and eager consumption;
it adds no event, global fence, allocation or numerical kernel.

## First complete diagnostic

Unchanged literal148/64 fixture, one warmup plus five measured requests.
Freeze both complete logs before quality traffic. Strict committed analyzer
and independent replay agree exactly:

- 384 version4 records:192 per rank, six generations, eight attempts and four
  steps per request; no missing or duplicate records.
- 192 cross-rank pairs agree in input, final hidden state, selected draft and
  raw gathered argmax-pair payload. All48 post-EH pairs agree.
- All12 first-step source observation sets agree in shifted tokens, primer
  and bootstrap capture sources, immediate primer/bootstrap KV, composed
  written prefix, later prefix and newly appended row.
- Composed prefix equals later same-rank prefix12/12. Physical block-map hashes
  alternate between two allocations; canonical KV remains identical.
- All nine request comparison edges have32 matched steps on EACH rank, no
  unmatched steps or metadata differences:576 observed-agreement comparisons.
- All measured outputs reach64 tokens with text SHA256
  `1bdda4caa55ae2df2226c4c3c080d7ab23a6ca80ba56488a1de6b75e0ee2f669`.

The first prefix is `492a374877bb35504750b89914c150d097d753e510d2f88ce1e077063daa838b`
and first final hidden is
`b2cdc71699edf14687cef52097bb437c42923e7b058d57a380e05a26f9231ec3`,
matching the preceding v24 rank1 first key. No claim is made that every later
trajectory equals v24. v24 had192/192 differing cross-rank final states and six
different rank0 first prefixes. v25 combines a new observation probe with a
separately tested ordering fix, so this is not isolated native causal A/B proof.
The actual public-wrapper CPU regression independently proves the old stream
mismatch and its repair. Equal hashes are not comprehensive model correctness.

## Quality, idle warning and safety

Four answer checks PASS;1984/16 needle PASS; deliberate streaming cancellation
returns curl28 after9772 bytes; four recovery checks PASS. No diagnostic
validation failure occurred. Both services gracefully stopped and were preserved
as `atlas-glm53-v25-hidden-clean-ep0/1`, exit0/OOMfalse. Post-stop host available
memory118802/118876MiB and swap used0 on both nodes.

The worker was ready15:51:39.316247 and first received a command15:57:56.237523.
It logged `NCCL broadcast took 376.9s ... marking communicator unhealthy`.
Source inspection shows its command-receive broadcast timer includes the
normal wait for a head command: this is not a measured376.9s network transfer.
Nevertheless the unhealthy latch really is set; this receipt does not qualify
as clean communicator health. HTTP readiness uses a different fault latch and
cannot disprove this error. A fresh run must dispatch promptly at readiness,
preserve this initial receipt, and inspect complete logs again. Do not disable
or raise the watchdog threshold to make a gate pass.

Raw receipts live under `/home/abc/storage/models/atlas-campaigns/20260908/`:
`v25-hidden-*`, native build/image identities, `run-v25-c1.sh`, and the frozen
`v25-stream-fix-slice.tar`. Archive closure and subsequent fresh gates are
recorded below. The30 C1 /60 aggregate C4 goal remains unmet.

## Fresh diagnostic repetitions

`v25-fresh-hidden-*` records a second fresh restart:384 records again yield192
cross-rank and576 repeated-request observed agreements. The six-request window
is free of errors, but later manual gaps before quality requests caused50.5s
and32.9s idle-receive warnings. Answers, needle and recovery still passed, and
both containers stopped0/OOMfalse. This is also not a clean whole-run health
gate; no attempt is omitted or overwritten.

`v25-gated-hidden-*` is the third restart, with the exact same source, image,
flags, workloads and caps. An independently reviewed fixed runner (SHA256
`e2399515d009d60b0ee56afdf31e2e1427836e25156055a5f2ead2b42baebbc5`)
submits promptly at readiness and automates the entire sequential gate through
shutdown. It validates capped outputs and JSON quality results explicitly,
refuses existing receipts/containers, preserves failure status and requires
stopped/exit0/OOMfalse. No timeout setting changes.

This run has384 complete records and the same192 cross-rank/576 repeat
agreements. Four answers,1984/16 needle, deliberate curl28 cancellation and
four recovery checks PASS. Complete post-stop logs contain no ERROR or
communicator-unhealthy message. Ready memory is11527/11547MiB; both ranks stop
normally at16:10:31/16:10:33UTC, swap0, post-stop available118819/118504MiB.
The normal OOM-watchdog startup INFO messages are not OOM failures. The
automated short idle intervals establish this bounded native gate only; they
do not fix idle serving. Explicit idle-command classification is a separate
infrastructure change under development.

## Separate trace-OFF performance qualification

The same frozen v25 executable and profile, with hidden trace/ledger OFF,
receives the unchanged literal148/256 fixture, temperature0/seed1, one warmup
plus three measured requests on EACH fresh restart. No forced output cap or
repetition override. All six measured outputs finish at256 tokens with SHA256
`12046a6857a2c4411efb58c919331ecc16c7243ae22a1d12d9281857c0903fc2`.

| Fresh service | Measured full-wall tok/s | Median |
| --- | --- | --- |
| Initial |28.691 /28.632 /28.630|28.632|
| Confirmation |28.641 /28.739 /28.726|28.726|

The initial decode-only median is30.088, which **does not** satisfy the30
full-wall target. The confirming full-wall median is0.197 tok/s (about0.69%)
above historical v20's28.529. This is a small observed difference, not proof
of a statistically significant or isolated causal speedup. Do not pool runs
or select the fastest individual sample as the headline.

Both timing runs also pass four answers,1984/16 needle, curl28 streaming
cancellation and four recovery checks. Complete post-stop logs contain no
hidden/ledger records, ERROR, unhealthy or CUDA fault. Selected actual Docker
environment snapshots on both preserved ranks confirm trace/ledger0 with
cache/repair/verifier-graph/overlap1 and BF16-drafts1, not just launcher intent.
Initial ready memory11563/11589MiB; confirmation11544/11311MiB. Both runs stop
each rank with running=false, exit0, OOMfalse and swap0. Confirmation services
are preserved as `atlas-glm53-v25-timing-repeat-ep0/1`, stopped16:17:21/22UTC.
This is a bounded cold C1 benchmark, not multi-turn, long-context, C4 or
idle-service qualification. C4 remains the separate historical47.319 result.

## Persistent receipt closure

`v25-verified-receipts.tar` (27MiB) is retained and SHA256-verified on controller
and head's persistent `atlas-glm53-deploy-20260906/phase7` directory:
`ddc58b5cca70c8538ed9eebf60f513b4233d8425adf2d279599b3775a0576017`.
It includes ALL three diagnostic attempts, both timing restarts, full clean-
window and later/post-stop logs, manifests/analysis, quality/cancellation,
memory/identity receipts, native build/package evidence, exact source archive,
committed plans/analyzer, both CPU proof directories, and exact launch recipes.
The archived v25 launcher SHA256 is
`4286b334ea34227d1c783a4bbbd78c2afc6c0543f6e3be33960ed1703e83d10a`.
No known failed attempt was discarded. Both nodes remain stopped while the
next source partitions undergo CPU validation.
