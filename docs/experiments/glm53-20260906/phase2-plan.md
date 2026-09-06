# Phase 2: aggregate decode campaign

Started after source commit `ad42aac0` on2026-09-06. The prior v4 deployment
remains the rollback baseline. No new throughput gain is assumed.

## Ordered plan and acceptance gates

1. Research primary dual-Spark receipts. Normalize concurrency, prompt/output
   lengths, precision, speculation, and throughput definitions. Separate
   fixed-output aggregate windows from steady-state decode and token-rate sums.
   Identify whether larger batches, kernels, or scheduling explain the gap.
2. Implement default-off graph-safe independent sparse C2/C3. Share existing
   CUDA arithmetic; use device lengths and fixed capacity-bounded grids.
   Preserve the existing dense kernel below2048 via a scratch dense-length
   scalar and a device-guarded sparse kernel above2048. Keep C1 eager, current
   slot-vector graph keys, and exact distributed batch widths. No persistent
   allocation, concurrency increase, MTP relaxation, or KV precision change.
3. Add tests before deployment: pure geometry/overflow/scratch tests; small
   GPU capture/replay tests across threshold, pool/page phases, capacity and
   slot/table reuse. Compare old/new score arithmetic, selected sets and
   attention outputs. Reject stale-length or inactive-branch writes.
4. Build offline with two CPU cores and4GiB memory, keeping builds off GPUs.
   Before model tests, preserve the current v4 containers. Serialize GPU tests;
   retain114GiB runtime ceiling and4GiB memory guard. Watch both nodes. Do not
   use unbounded kernels or revive the rejected compact-down experiment.
5. Same-image graph-off/on A/B at unchanged16K context,4096 chunk, C3, and
   FP4 settings. Test mixed threshold lengths, independent10K needles,
   C3-to-C2 noncontiguous-slot drains and C1 tail, slot reuse, and fixed-output
   concurrency. Measure latency and aggregate throughput, not just kernel time.
6. Keep positive validated changes, document negative results, commit the
   completed next step, and select the best validated deployment. Based on
   measured bottlenecks and external evidence, plan the next independent
   batch-width or scheduling change rather than conflating it with graph work.

Implementation detail: [graph-safe design](graph-safe-concurrency-next.md).
CUDA and Rust integration are independently delegated with a shared explicit
ABI; root owns node operations, launch configuration, benchmarking and final
review. External research is read-only and runs in parallel.
