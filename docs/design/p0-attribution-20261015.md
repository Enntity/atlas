# P0 Nsight attribution — 2026-10-15

Classification: **unqualified, instrumented, attribution only.** Each run is
one serial request under Nsight Systems, from an unknown initial state. The
timings below are observed under the fingerprint, as single shots. They are
not headline rates, a baseline or a target. P0's done-when (≥ 3 matched
repeats with independent timing, then targets) stays open.

## Fingerprint

- **Image:** `atlas-exl3-validation:src-4ee96c12b44c-rust1.97.1`, ID
  `sha256:82173ea631a8c78e4fb565f0e23bc6d55f94a43090dbfb27c850f8cc1e6b41d9`,
  asserted on both ranks. This is the worker-binding image (`f099ac58`) with
  a full neutrality proof. The re-pin was an operator decision on 2026-10-15.
- **Serve profile:** the collector-only rank descriptors
  (`$EV/statistical-gate-20261011/collector-only-profile/cand-rank{0,1}-create-command.json`).
  The image token is replaced and the logits observer env/mount removed.
  Settings: TP=2/EP=1, `fp8_g128` KV, default LM head, max-seq-len 4096,
  prefill chunk 128, batch 1, one sequence, memory utilization 0.84, eager
  (all three graph disables), `ATLAS_KDA_FLA_REFERENCE=1`, sparse-prefill
  TC/KV-reuse on.
- **Checkpoint:** `MikeRoz/GLM-5.3-Flash-Uncensored-4.05bpw-h6-exl3` at
  revision `dda99c7e…`.
- **Harness:** the re-pinned driver copy, `$EV/p0-trace-20261015/driver/`
  (11 tests). Each request uses the `perf_baseline.py` prose prompt,
  temperature 0, seed 20261011 and `reasoning_effort: none`. The server still
  emitted reasoning tokens (63/64, 10/64 and 67/600).
- **Profiler:** nsys 2025.3.2.474, trace `cuda,cublas,nvtx,osrt`, CPU
  sampling off, a named session started after health.
- **Hosts and date:** spark0/spark1 (GB10), 2026-10-15 04:42–05:18Z.
- **Evidence:** `$EV/p0-trace-20261015/`.
  - `runs/<run>/` holds the plan, request, raw `.nsys-rep` and stats.
  - `summary.json` comes from `analyze.py`.
  - `phase-split/` comes from `phase_split_remote.py` (SHA256 `5d6d0977…`)
    over each rank's `.sqlite` export. Its split point is the first kernel
    plus TTFT, so prefill/decode boundaries are approximate.

## Observed (instrumented, single shot per run)

| Run | Prompt / completion tokens | TTFT | Median inter-token gap (p10–p90) |
|---|---|---|---|
| `p0-short-20261015` | 108 / 64 | 11.8 s | 0.665 s (0.62–0.72) |
| `p0-nearlimit-20261015` | 2,573 / 64 | 272.3 s | 0.686 s (0.61–0.78) |
| `p0-decode-20261015` | 108 / 600 | 12.0 s | 0.664 s (0.63–0.72) |

Kernel share of GPU busy time, rank 0. Rank 1 is within about 1 point
everywhere. In every phase of every run, kernels cover 95–100% of the
phase's wall span.

| Phase (run) | Routed MoE (EXL3) | Packed projections (EXL3 mul1) | NCCL | mHC | Rest |
|---|---|---|---|---|---|
| Prefill, 2,573 tokens (`nearlimit`) | 55% | 42% | 2% (rank 1: 4%) | <1% | ~1% |
| Decode, 600 tokens (`decode`) | 21% | 45% | 25% | 5% | ~4% |

Decode in `p0-decode-20261015` breaks down as follows:
- about 2,067 kernels per token;
- `exl3_mul1_matmul_mma` averages 1.60 ms per call over 114,000 calls;
- `exl3_moe_mul1_matmul_precision_mma` averages 1.18 ms over 75,600;
- `ncclDevKernel_SendRecv` averages 1.79 ms over 54,000 calls (about 90 per
  token). Its median in the short run was 0.27 ms, so the tail is heavy.

## What this does and does not support

- **The time is in GPU kernels, not host gaps or launch overhead.** Kernels
  are busy for 95–100% of every phase span. Decode graphs would not recover
  much time unless the kernels themselves get faster.
- **Prefill is the packed EXL3 GEMMs:** routed MoE plus projections, about
  97%. At about 106 ms per prompt token it runs only about 6× faster per
  token than decode, despite 128-token chunks. That is consistent with
  row-serial work in the packed kernels (P2: real multi-row GEMM and
  expert-grouped prefill). It is a hypothesis until per-kernel counters
  confirm it.
- **Decode is the packed projections (45%), NCCL (25%) and routed MoE
  (21%).** A 1.6 ms average per M=1 projection call is far from a
  bandwidth-bound GEMV on 273 GB/s unified memory. Shapes vary per call, so
  this record makes no roofline claim. NCU DRAM bytes and occupancy for the
  dominant kernels are the owed measurement (P0 scope, then P1).
- **The NCCL share is attribution, not a diagnosis.** A heavy-tailed
  SendRecv mean can mean peer skew (one rank waiting on the other) or
  transport latency. The trace does not separate them. Per-call skew
  analysis across both ranks' timelines is P6 work.

**Ledger.** The prior unqualified record (`p0-unqualified-baseline-20261010/`:
decode about 0.45 s/token, 2,573-token TTFT about 139 s) is a different image
and was not profiled. These are two observations, not an A/B, so no
regression or improvement is claimed. Nsight CUDA tracing of about 2,000
kernel launches per token is a plausible source of overhead, but this record
does not measure that overhead.
