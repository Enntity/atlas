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
projection speedups); see `mla-c6-c8-kernel-results.md`. The host integration has
source/CPU qualification only; see `scripts/dev/glm_independent_c2_c8_plan.md`.
The native candidate build has started; no selected C6/C8 serving result exists
yet. Launcher and distinct answer/tool/needle clients now support C1..8.
Next qualify eager native quality, actual memory headroom and drain behavior,
then warmed graphs and reproducible throughput. Preserve rollback/watchdog
behavior throughout. Concurrent MTP also remains required: checked cold F0 transport is
committed, but scheduler admission and actual supervised head/worker integration
are not live. Neither standalone kernels nor inactive infrastructure complete
the goal; require measured native serving gains and all quality checks.

The user has now updated and resumed the actual product goal; `get_goal`
confirmed this reference-parity objective on 2026-09-09. Every candidate must
include coherence, real tool-calling and needle-in-haystack checks. Never
substitute a repository document change for changing the product goal itself.
