# First native paired MTP measurement — shutdown qualification failed

2026-09-09, source `656667cef44354a76cd9a2f530185766875507c2`.
This is an actual two-Spark TP2/EP2 eager MTP4 run, not the earlier CPU fixture.
The quality/timing workload passed, but the campaign did **not** qualify:
shutdown failed before either quiescent receipt reached the controller.
Do not promote this image or call its shutdown clean.

## Measured workload

Two selected slots, context2044, prefill1024, BF16 KV, FP32 SSM snapshots,
full-vocabulary BF16 appended MTP head and NVFP4 target projection. Existing
K5 optimizations retained. The selected dispatcher still completes each owner's
verification/acceptance/repair/proposal transaction serially.

Matched internal LRU coding prompt:148 input tokens,256 output cap, temperature0,
seed1, one warmup plus three measured batches per width. All outputs reached256.

| Width | Aggregate full-wall tok/s, median | Aggregate decode-window tok/s | Client TTFT ms | Server TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| C1 | 27.359 | 28.774 | 456.751 | 427.230 |
| C2 | 27.312 | 27.994 | 672.407 | 426.455 |

Full-wall repeats: C1=27.355/27.399/27.359; C2=27.331/27.308/27.312.
C2 is aggregate, not per stream. Its lack of scaling is consistent with the
serial target-verification implementation; this is not a profile attribution.
The unchanged v29 nonspeculative baseline remains the rollback/control.
C3..8 were not issued against this two-slot candidate.

Before timing, four distinct answer checks (arithmetic, stable ordering, Python
AST, exact JSON), two genuine automatic structured tool calls with exact JSON
arguments, and two NIAH own/no-peer checks all passed. Named forced tools remain
unsupported because they require compiled grammar. NIAH uses768/800 input tokens,
not a broad long-context or exact-format quality claim. Coding output is retained,
not executed or certified as a complete program.

The new stdlib client uses read1 instead of requests.iter_lines: client buffering
can affect TTFT/decode-window comparisons, while the full-wall denominator is
unchanged. MiaAI's precise workload remains unavailable; these are internal
regression measurements, not exact reference parity or full-goal completion.

## Deployment and safety

Each container: memory=swap114GiB (no container swap), CPUs0..19, privatePID,
privateIPC/shm1GiB, hostnetwork, no-new-privileges, RDMA device mapping.
Across909 actual node observations, minimum MemAvailable was9,210,856KiB head
and10,491,668KiB worker; maximum observed SwapUsed was0 on both. Both returned to
about116GiB available after cleanup, OOMKilled=false, restart count0.

Prepared bundle `68bbea5f277bf17dc9a9bd8a66d974d1408111b1435383f62f1255174d8842f1`.
Twenty-two paired renewal rounds completed. After workload exit0 the controller
sent the real head drain request. Head logged SIGINT and HTTP drain completion;
the server process then disappeared before any observed Q/release. The controller
latched failure and killed only its two recorded full container IDs. Both final
Docker statuses were137, not clean exit0; no node reboot/reset or OOM occurred.
The HTTP log alone does not prove main returned: serve_supervised already retains
a pending selected future. Thread-local parent-death protection at the release
handoff was subsequently reproduced in the actual registered CPU fixture below.

Earlier attempt `1256b2b3f20d0a02542cc1211fb6586f667c9d7b029ec23690d3b11294c5b931`
stopped before either gated report, both exit74/OOMfalse. Read-only inspection
found inherited NVIDIA library directories absent in these images, which the
guard refuses; source also requires exact PATH=/usr/bin:/bin. The succeeding
launch uses that PATH and observed root-owned755 CUDA/system library directories.
Runtime checks were not relaxed. The merged environment has127 entries after
six explicitly verified redundant zero diagnostic removals (startup cap128).

## Artifact identity and retained evidence

Head image: `7314d940262df68e621c8bc2eabedecb274ef9146dceba756905cb8cd83fafc6`.
Worker image: `45399cec84ed2a557c27c458cc0baf3059257b80978dbf1af1d3ac6c5f599670`.
Spark SHA256: `60373a0d148de139baa530b0d7515efae2dd33dd9701db522bbb2e0f825f967e`.
Workload SHA256: `aeee714fa6001655ad45e8c8c04acc4d9afb08ed617dd5f5de2ab9e0b35513b1`.

External base: `/home/abc/storage/models/atlas-campaigns/20260909/glm-native-controller/`.
`native-prepared-656667ce-loader/` contains immutable launch/session/recipes/input
and actual event receipts. `native-summary-loader.json` summarizes them;
`native-head-loader.log`, `native-worker-loader.log`, corresponding `*-final.json`
and `native-run-656667ce-loader.log` retain runtime/final observations.
Native source/build identities and exact-tip CPU closure are retained alongside.

Next: reproduce/fix the threaded shutdown failure, repeat this healthy campaign,
then implement joint layer-major target verification, initially batching routed
FFN rows across both K5 owners while preserving independent state/rollback.

## Thread-handoff reproduction and candidate fix

The new `registered-drain-threaded` mode moves the real registered Model,
inherited session and retained sequences to new OS threads after both real
registrations and the actual drain request. Both witnesses recorded
`thread-pdeathsig=0`; the unchanged production shutdown paths then failed74
before Q (`thread-handoff-red.log`, retained `/tmp/atlas-live-main-sYpMSe`).
Linux clears the task's parent-death signal on clone, including CLONE_THREAD;
see [kernel/fork.c](https://raw.githubusercontent.com/torvalds/linux/master/kernel/fork.c).
The original fixture did not perform this thread handoff.

The candidate explicitly binds SIGKILL parent-death protection once on the
actual head/worker execution thread before its Model bind/alloc/receive.
Retained actual process identity and pinned server/guard ELF identities are
validated before/after; a preexisting signal other than0/SIGKILL is refused.
The strict release check is unchanged, with no per-token syscall or relaxed
guard authority. No router-return/join workaround was added.

Root observed threaded GREEN (`/tmp/atlas-live-main-aBD8eU`), ordinary drain
GREEN and unchanged replay/foreign refusals, plus focused IO/server checks.
These are CPU fixture results, not a substitute for repeating the native run.
