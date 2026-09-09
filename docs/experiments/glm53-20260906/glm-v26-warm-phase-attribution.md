# v26 warm C1: existing phase timing, not a new benchmark

2026-09-09. Reanalysis of the two qualified September 8 v26 runs. No model
launch, profiling flag change, CUDA timing synchronization or node mutation.
This supplements the historical timing in
[the GPU-feedback source audit](../../../scripts/dev/glm_gpu_feedback_reference_audit.md).

## Selected windows and result

Both rank-zero logs contain run-wide 25-step timing summaries. Select lines
751, 752, 764, 765, 776 and 778 in each log: two interior windows from each of
the three measured requests, 150 steps per run. These have `GAP=0.00ms(x1.0)`.
Exclude the first two zero-gap windows (warmup), all summaries spanning request
boundaries, and the later answer/recovery workload. The windows are not a
whole-request timing average; their sequence-position labels are respectively
247, 339, 258, 355, 270 and 365. Each phase fires once per step in these windows.

Arithmetic means of the six printed, two-decimal window averages:

| Run | Target `fwd` ms | `propose` ms | `TOTAL` ms | Target / total | Proposal / total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial | 106.133 | 11.447 | 117.698 | 90.17% | 9.73% |
| Repeat | 106.643 | 11.388 | 118.157 | 90.26% | 9.64% |

At the actual v26 Rust commit `e40a9066673057ceb99f76abddc8ae0d29826272`,
`scheduler/mtp_timing.rs` uses host `Instant` elapsed time, summing microseconds
and dividing by 25. `scheduler/verify_dflash_step.rs` brackets
`decode_verify_dflash` as `VerifyForward` and the subsequent proposal call as
`Propose`. These are host-observed phase boundaries, not GPU kernel attribution.
`TOTAL` includes other work within the verify step. Do not add `step_mtp`, GAP
or loop accumulators to these nested phases, or interpret their run-wide idle
and admission accumulation as decode-only overhead.

Inference: target verification dominates this C1 control. Sharing target work
across two independent five-row verification segments remains the larger
architectural opportunity. Device-resident draft feedback is a narrower
candidate; its removable readback fraction is still unmeasured, and eliminating
all proposal work is neither possible nor a throughput prediction. These logs
do not predict C2 acceptance, paired scheduling overhead or M10 endpoint speed.

## Keep the throughput definitions separate

The corresponding warm 148-prompt / 256-output benchmark JSONs report median
full-request aggregate rates **28.699 / 28.618 tok/s**. Their median server
decode-window rates are **30.253 / 30.138 tok/s**. The latter exclude prefill and
are not evidence that the >=30 full-wall C1 target has been reached. No measured
rate changed through this analysis. All six measured outputs reached 256 tokens
with the same recorded text hash; no forced-output or repetition allowance.

## Reproduction and provenance

Source directory: `/home/abc/storage/models/atlas-campaigns/20260908/`.
Read the six line numbers above, extract `fwd`, `propose`, `TOTAL`, average each
column and divide phase mean by total mean for the percentages. Benchmark
fields are `median_aggregate_e2e_tps` and
`median_aggregate_decode_window_tps`; there is one C1 result per JSON.

SHA-256:

```text
2c23a774131be238fd111c71494d544f0fb6fb3ab9b19372b6d7bd6d690a88fb  v26-idle-initial-clean-rank0.log
9e4c4bdd0be35981099c8996c3e1e60e8f5e2646ece652b24ef4ac76f23d47b8  v26-idle-repeat-clean-rank0.log
07bd4356b62b93247267611b106878bc63a08e1e519897df8276101c0b9e4c13  v26-idle-initial-c1.json
b3782fb9d8b0d603e7b61f7055d989f3a097842448b25d45f0874080dc92aa53  v26-idle-repeat-c1.json
```

The v26 qualified-receipt archive already retains these files; no new native
qualification record is minted for a source-only reanalysis.
