# Current goal: match MiaAI's dual-Spark TP=2 serving reference

User revision, 2026-09-09. This supersedes the earlier alternative finish line
of 30 C1 tok/s OR 60 aggregate C4 tok/s. Continue improving Atlas serving of
GLM-5.3-Flash NVFP4 on the two DGX Sparks toward the performance and serving
capabilities of this specific reference, not SparkGLM's newer DFlash2 profile:

https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark

## Performance targets

Reference README inspected 2026-09-09:

| Concurrent streams | Published aggregate tok/s | Working target, aggregate tok/s |
| --- | ---: | ---: |
| C1 | 23–30 | >=30 |
| C2 | 31–37 | >=37 |
| C4 | 43 | >=43 |
| C6 | 64 | >=64 |
| C8 | 72 | >=72 |

Use the upper end of published ranges as the conservative completion target;
report reaching the range separately. C3 remains a required diagnostic and
regression point, although the reference gives no C3 number. Passing C1 alone
does not complete this goal. Preserve previous stronger C4 results as baseline
regressions, rather than deliberately settling for the reference's lower C4 rate.

Our qualification runs must be warmed, reproducible, unprofiled and retain
per-stream counts, aggregate full-wall and decode-window rates, TTFT, output
quality and fresh-process repeats. The reference marks only C6 explicitly
warm; its README does not specify enough prompt/output workload or denominator
detail to assert a matched comparison from its table alone. Establish those
conditions from available source/receipts or reproduce a safe reference run.
Until then, label comparison to the published table as indicative, not exact
benchmark parity. Keep the existing 148-input/256-output C1..4 benchmark as
an internal regression test, not proof of full reference parity.

## Serving scope and safety

Reference configuration: TP=2 on two GB10 Sparks, NVFP4 target, concurrent
MTP-4, up to eight scheduled sequences, FP8 E4M3 KV, Marlin/eager execution,
OpenAI-compatible API, tool calling, reasoning and image/video input. Track
functional gaps explicitly; copying vLLM's backend or exact flags is not a
requirement for Atlas if its implementation provides comparable capabilities.
Do not silently substitute another model, quantization or weaker quality policy.

The reference advertises a 262144-token maximum context. The user explicitly
permits a smaller total context to prevent OOM/crashes. Choose a bounded context
and KV budget from measured per-rank memory headroom, state it alongside results,
and grow only after lower-cap qualification. The existing 2048-token C2 experiment
is an interim short-context gate, not a claim of comparable long-context capacity.
An exact new long-context cap is not selected or qualified yet.

Retain TP=2, no competing resident models/builds/microbenchmarks, no swap use,
strict memory guards and recoverable images. Never trade node stability or
correctness for headline throughput. Do not copy the reference's memory fraction
blindly: vLLM and Atlas have different allocations. No reboot/reset/clock/driver
changes are part of this goal.

The default-off compact C2 FFN candidate now has native eager quality and two
fresh-process warmed OFF/ON C1..4 pairs. Conservative mixed-output C2 throughput
improves about19.0→25.5tok/s; identical-output pairs can be faster, so the repeat's
29.746 median is not a general heterogeneous-request baseline. See
`v27-c2-compact-results.md` for full-wall rates, TTFT, quality and limitations.

Independent-row KDA/MLA/grouped-MoE host support through every draining width1..8
is now committed in `c0f7b0ef`, with selected server admission, TP-local reserve
accounting and preserved slot identity. MLA exports6/7/8 are committed and
standalone GPU-qualified (exact scalar equality, zero memcheck errors and repeated
projection speedups); see `mla-c6-c8-kernel-results.md`. The integrated v29 binary
now passed eager and graph native C1..8 quality, retrieval, cancellation recovery,
actual width/slot tracing and clean shutdown with zero observed swap/OOMs.
One fresh graph process, warmup plus three measured148-input/256-output batches,
gives full-wall medians C1=13.410,C2=25.589,C3=36.732,C4=46.839,C5=56.227,
C6=63.585,C7=70.549,C8=76.923tok/s. C8 exceeds72 on this internal workload;
C6 remains0.415 below64 and C1/C2 remain below30/37. All are nonspeculative.
See `independent-c2-c8-results.md` for raw evidence, TTFT, quality limitations
and full tables. A second fresh-process graph repeat also passed: medians
C1=13.434,C2=25.810,C3=36.998,C4=47.303,C5=56.469,C6=63.761,C7=70.805,
C8=77.014tok/s, all within1% of the first process. Quality/retrieval/recovery
and both-rank clean shutdown passed with zero observed swap/OOMs. There
is no exact reference-workload or full-goal-completion claim. Preserve the
rollback/watchdog behavior and bounded memory throughout subsequent work.
Concurrent MTP also remains required: checked cold F0 transport is
committed, and `bdaa2558` adds actual Model health/strict stream-quiescence checks
with independent source and focused CPU qualification. Shared wire/credentialed
transport, the existing guard's inherited child channel and actual server ticket
consumer now pass a connected controller-only two-PID1/proc handshake gate.
See `live-pair-handshake-results.md`; this is not native serving activation.
The production LIVE guard entry/loop, canonical recipe codec, pinned startup
ingress and two-rank release now pass the actual two-guard CPU fixture, including
delayed-peer renewal and rejected unreleased/replayed exits. See
`live-pair-release-results.md`. No throughput gain is attributed to this slice.
Actual Model registration, selected factory/admission and supervised head/worker
dispatch are now connected in the development source. Focused controller checks
cover cold selection, owner health, memory accounting, serial transactions and
retirement; see `selected-paired-serving-results.md`. The real registered-Model/
two-guard CPU fixture passes valid release and three registration refusals,
using local scripted model transport, not NCCL. The native recipe/controller
adapter remains unqualified.
Recipe v2, the actual held-child healthy drain path and the native controller's
Docker/state/subprocess adapters are now implemented; see
`native-controller-adapter-results.md`. The SSH launch/renewal run loop is now
connected and has passed an actual two-container Docker/SSH CPU Model rehearsal,
including active lease renewals and independently observed paired exit0; see
`connected-controller-results.md`. ARM64/native MTP qualification remains next;
no native throughput gain is claimed for this slice.
The first actual native selected C2 MTP campaign now completed quality and warmed
148/256 timing: C1=27.359 and C2=27.312 aggregate full-wall tok/s. However, paired
shutdown failed before Q/release, and exact-ID cleanup ended both containers137
with OOMKilled=false and zero observed swap. This is not a qualified deployment;
see `paired-native-first-results.md`. The execution-thread fix in `fcbe8da6`
then passed the same native quality workload at C1=27.280/C2=27.245 full-wall
tok/s. Both real Q/release exchanges and independently observed container exit0
completed without OOM/swap. The controller still failed on a two-snapshot Docker
Running-to-Exited race, so full campaign qualification remains pending. Fix that
narrow observer transition while building joint layer-major verification: the
two-owner routed FFN is the next performance target, not an achieved gain.
The observer fix is now committed in `7d0d220b`. The fixed-pair model/scheduler
and joint routed-FFN candidate are implemented default-off, with actual CPU
ownership/continuation checks; native numerical and throughput qualification
remain outstanding. See `paired-temporal-verification.md` for the implementation,
explicit controls and evidence boundaries.
Native source `5b4662a9` subsequently passes the first fully supervised paired
FFN A/B, including quality and both-rank normal shutdown: warmed C1 remains
approximately27.3, while C2 improves27.314→34.333 aggregate full-wall tok/s
(25.7%) with identical retained outputs. See `paired-ffn-serving-results.md`.
Fresh-process repetition and explicit committed-pair telemetry are next; C2
remains below37 and selected MTP capacity remains two, not reference parity.
See `scripts/dev/glm_c2_live_integration_plan.md` and
the exact proposed `glm_c2_live_wire_plan.md`. Neither standalone kernels nor inactive infrastructure complete
the goal; require measured native serving gains and all quality checks.

The user has now updated and resumed the actual product goal; `get_goal`
confirmed this reference-parity objective on 2026-09-09. Every candidate must
include coherence, real tool-calling and needle-in-haystack checks. Never
substitute a repository document change for changing the product goal itself.
