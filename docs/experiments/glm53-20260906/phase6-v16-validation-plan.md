# Phase 6 v16 controlled validation

The user target remains >=30 C1 or >=60 aggregate C4 tokens/s, measured by
the existing coding148/256 full-wall median, not summed per-session rates.
This plan specifies the next candidate's checks; it is not a passed receipt.

The existing four-answer validator will expose explicit concurrency1 or4 and
the already tested context2044 envelope. Add parser regression tests first;
keep the default C4 behavior and all answer validators/payloads unchanged.
C1 runs the same four requests sequentially with a one-party start barrier,
so a constrained deployment is never accidentally sent four active requests.

## Frozen artifacts and recovery

Root controls all native operations. Keep both v15 C1 control containers
stopped and preserved, plus the earlier validated v13 C4 control containers.
Build only committed source, syncing changed paths without rewriting unrelated
CUDA/runtime timestamps. Kernel-cache priming uses committed `502cda12`;
the eventual runtime candidate must include the separately reviewed repair.
Do not package until the builder reports exited0/OOMKilled=false; reject the
old v15 SHA and verify identical new binary hashes on both nodes. CPU builds
and standalone GPU gates do not overlap performance measurements.

Use the existing 114GiB model-container cap, GPU utilization0.90, 4096MiB
load guard, and >=4GiB host MemAvailable after load/warmup. No host swap,
clock, privileged configuration, watchdog, reset or reboot changes. Runtime
`--swap-space-gb 0` refers to Atlas's request-state spill facility, not host
swap. It is matched in both repair arms and required by the bounded new lane.

## Accepted-history A/B

Both arms: TP2/EP2, active/admitted C1, context2044 plus four lookahead rows,
BF16 KV, native distributed MTP4, batched primer1/serial primer0,
MTP_SPEC_THINK1, MTP_GATE_FORCE1, MTP_PREFILL_ONLY1, SWAP_SPACE_GB0,
no prefix reuse, adaptive depth, confidence discard, catchup or LoRA.
Keep the existing leading-BF16/later-NVFP4 proposer precision unchanged.
M16 gate/up stays OFF throughout this comparison. Preserve identical normal
repetition and output-cap policies; forced continuous MTP is matched, not
credited as a kernel or accepted-history gain.

1. Repair OFF / KV diagnostic OFF: quality checks and one warmup plus three
   coding waves establish a fresh fixed-policy control on the same binary.
2. Repair ON / KV diagnostic ON: short text/strict visible answers and coding
   exercise the existing loaded BF16 weights through the resident row oracle.
   Inspect both rank logs for bootstrap, accepted counts and bitwise checks;
   oracle timings are not performance evidence. Stop on mismatch, stale state,
   unexpected fallback, unsupported lane or missing native execution proof.
3. Repair ON / KV diagnostic OFF: repeat quality, bounded near-context
   retrieval/output caps, cancel then slot reuse, and the matched benchmark.
   Use explicit short request deadlines and context-safe token sums. A model
   error is not permission to reset hardware or keep using unknown GPU state.
4. If the threshold is reached, run a separate confirming three-wave repeat.
   Otherwise retain all results and continue with the next measured bottleneck.

The KV writer runs eagerly outside the completed target verification graph;
this does not by itself require disabling target graphs. No diagnostic host
copies may execute during graph capture. Device-fault restoration is only
best effort, never a guarantee that a distributed communicator can resume.

## Separate fused gate/up A/B

The committed production implementation has standalone C4/K5 full-output,
refreshed-graph and memcheck receipts. Those are not resident-model proof.
First use eager execution with M16 ON and its separate VERIFY switch ON;
require useful local-work comparisons for each selected layer/row count on
both ranks. Empty worklists are valid but cannot establish real-weight proof.
Then disable VERIFY and restore the ordinary graph policy for matched M16
OFF/ON quality, draining/slot-reuse and repeated throughput tests. Keep the
accepted-history mode identical in both arms. Down remains unchanged.

No default kernel promotion, general concurrency support, support-matrix
validation or vLLM parity is implied by this bounded GLM C1 repair experiment.
