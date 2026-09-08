# v26: explicit idle-command receive

2026-09-08. Both fresh native35-second idle gates pass with clean subsequent
quality/recovery/shutdown. Full-wall C1 medians28.699/28.618 remain below30.
This is an idle-health infrastructure
fix, not a kernel speedup or B-tile activation.

## Defect and bounded change

The old worker's outer command receive started a broadcast timer before
waiting for the next head command. Normal idle time therefore triggered the
post-completion30-second slow-broadcast latch. The earlier v25 failures remain
preserved:376.9s initial idle and50.5s/32.9s later manual gaps. This is not an
interrupting watchdog; the old call returned successfully despite latching
unhealthy. HTTP health used a separate GPU-fault latch.

Commit `e40a9066673057ceb99f76abddc8ae0d29826272` adds an explicit receiver-only
outer-command-word operation. Only that first word omits elapsed-duration
marking: the v1 command or v2 sequence-ID preamble. The following v2 command,
arguments, payloads and head broadcasts retain normal timing. Non-root rank,
world and aligned/non-null word validation happen before I/O. Real NCCL
arguments, synchronization/error propagation and async-error checks remain
unchanged; no existing unhealthy latch is cleared. There is no size/rank
heuristic that exempts ordinary four-byte data broadcasts.

Actual runner and actual Model worker/protocol RED→GREEN are retained. Seven
communication tests and940 model tests pass, plus non-test and real NCCL-feature
compile checks; strict NCCL communication/test Clippy passes. Model Clippy is
still blocked by four existing runtime Metal-stub errors, not claimed green.
Both protocol versions are covered by CPU tests. No CPU fixture is claimed to
prove physical transport-loss handling or GPU numerical correctness.

## Exact native artifact

Both `atlas-glm53-flash:kernel-20260908-v26` images contain executable SHA256
`a29c261991b94309a2eb4193f3354a429cb60f77d851604b62fe1f3ea5538cbd`.
Rust source is the full commit above; enabled CUDA remains
`189db87e0b7ce22e262643a160adf3e51263b203`. Frozen archive of spark-comm and
spark-model, `v26-idle-source-slice.tar`, SHA256:
`55ac1e4eecf7e2fbeed6c6570ed104fb960937e31d98312581c6b22136356941`.

Only committed source was overlaid in the actual native builder. The old
grouped CUDA TU still matches189db87e SHA256
`8697eb5fc6140977da3409fb8dac4c07271335c501a1527aceb7a667e6593843`.
The8GiB/two-CPU/jobs2 offline Rust build ran16:35:13.353630→16:37:26.565792UTC,
reported2m13s, exit0/OOMfalse. Executable-only images were packaged over v25
after the builder stopped, without any resident model or GPU fixture overlap.

The committed checkpoint-retirement infrastructure `6a486364` is included but
is unselected by the legacy whole-model loader. Unfinished resident readers,
loader activation and the promoted B-tile CUDA are NOT included/enabled.

## Unchanged profile and initial idle test

The head recipe changes only v25's image to v26: C1/TP2/EP2, MTP4 repair,
TRACE0/CACHE1/VERIFY0, same numerical policy and graph/overlap settings,
context2044, prefill1024,114GiB containers,4096MiB guard, utilization.91,
KV overcommit0 and swapspace0. Actual selected container environments confirm
trace/ledger0 and the unchanged settings. `ATLAS_EP_PROTOCOL` is absent;
the model's explicit environment check therefore selects v1. Native v2 is
not claimed by this C1 gate; later concurrency gates must cover it.

At readiness the fixed runner records both host memory levels, then waits
35 seconds without any request/HTTP keepalive. Initial local timestamps are
16:40:01→16:40:36UTC. Worker-ready16:39:58.934766→head's first prefill-start
16:40:36.244954 spans37.310188s. This corroborates the idle discriminator but
is NOT an exact measurement of the NCCL call's start/end; that call is not
independently timestamped. Complete post-stop logs contain no ERROR,
communicator-unhealthy, diagnostic trace/ledger, panic or CUDA fault.

Then the unchanged literal148/256 benchmark runs, temperature0/seed1, one
warmup plus three measured C1 requests, no forced-cap/repetition override:

| Restart | Full-wall measured tok/s | Median |
| --- | --- | --- |
| Initial |28.700 /28.670 /28.699|28.699|
| Fresh confirmation |28.633 /28.618 /28.591|28.618|

All six measured outputs reach256 with SHA256
`12046a6857a2c4411efb58c919331ecc16c7243ae22a1d12d9281857c0903fc2`,
matching v25. The initial decode-only median30.141 does not satisfy30 full-wall.
The deliberate pre-request idle is not part of any client request wall time.
This rate is effectively near v25's28.632/28.726 initial/fresh medians; no
significant or causal throughput improvement is claimed.

Four answers,1984/16 needle, deliberate curl28 streaming cancellation after9772
bytes, and four recovery checks PASS. Initial ready memory11553/11545MiB.
Both containers stop normally at16:41:48/50UTC, running=false, exit0, OOMfalse,
swap0, post-stop available118798/118832MiB. Their complete logs and preserved
containers retain the exact initial attempt.

This validates the bounded cold C1/v1 idle boundary only. It does not add an
interrupting watchdog, wire communicator health into HTTP, prove reconnect or
transport-loss recovery, qualify multi-turn carry/long context, or establish
the30 C1/60 aggregate C4 target. C4 remains the separate47.319 historical rate.
Raw receipts are `atlas-campaigns/20260908/v26-idle-initial-*` and
`v26-idle-repeat-*`.

## Fresh confirmation and receipt closure

The second fresh service waits16:43:35→16:44:10UTC without requests. Worker-
ready16:43:32.715820→head first prefill16:44:10.605162 spans37.889342s, again
a boundary corroboration rather than directly instrumented NCCL duration.
The same quality/cancellation/recovery sequence passes, all post-stop logs
remain clean, and both ranks stop normally at16:45:23/24UTC. Ready memory
11482/11578MiB; post-stop available118799/118802MiB; swap0 throughout the
recorded snapshots. The repeat's30.027 decode-only median is not full-wall
qualification. The two medians remain separate, with no speedup claim.

`v26-verified-receipts.tar` (9.6MiB) is retained and SHA256-verified on controller
and the head's persistent `atlas-glm53-deploy-20260906/phase7` directory:
`634ba56311205a181bf43356b942e4c3ffefb3c856b25dd3239eee4e1ef9e9fd`.
It contains both complete native attempts, selected live/stopped environment
identities, build/overlay/package evidence, exact committed source and workload,
CPU RED/GREEN/check receipts and frozen manifests, and exact launch runners.
Recipe SHA256: `741defe738987230b9bdd31dfea33e3e2b9c04e4b038383f264b42e41646b061`;
gated runner: `2b762248391fa5ac1908e88814eaa34e1f2224e2e71cda387da772eb97f589b6`.
All services are stopped and preserved. Native C4/v2 coverage is a subsequent
bounded gate, not implied by this C1/v1 evidence.
