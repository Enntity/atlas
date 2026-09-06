# Phase 3: bounded C4 correctness investigation

Status: **experimental; not promoted**. The initial full-model C4 scalar
control fails independent-needle correctness. Kernel-only speedups are not
serving throughput results.

## Initial image and configuration

Source contents: `2329bf66`; image `atlas-glm53-flash:kernel-20260906-v6`.
Binary SHA256:
`de7892be673f17523e4adf3779750168f74a5afad2a25e473a38f2ab8dcf3ca9`.
TP2/EP2, native NVFP4 weights, BF16 KV, FP32 recurrent state, non-speculative,
context 2048, prefill chunk 1024, four active/admitted requests. C4 enabled,
grouped C4 MoE disabled, multi-sequence graphs disabled, sparse mode disabled.
WMMA prefill index, FP4 prefill, MLA batch2/3, and C3 grouped MoE enabled.
Docker memory remains 114 GiB per node; GPU utilization fraction remains 0.90.

## Failure and process-isolation evidence

Raw client receipts and both-rank logs are under
`/tmp/atlas-glm53-phase3-20260906/` on the controller. These are exploratory
receipts, not clean-tip benchmark gate records.

| Order | Test | Result |
| --- | --- | --- |
| First process, first request set | C4: prompts 768/800/832/896, output caps 32/16/48/64 | All four failed; initial correct fragment followed by `!` tokens |
| Same process afterward | C3: prompts 768/800/832, caps 16 | All three failed similarly |
| Same process afterward | C1: fresh 768-token prompt, cap 16 | Failed similarly |
| Fresh container restart, first request | C1: `FRESH-9421`, 768-token prompt, cap 16 | Passed |
| Fresh process, before any C4 | C3: independent needles, prompts 768/800/832, caps 16 | All three passed |

The fresh C3 test has actual N=3 server traces, followed by N=2 slots `[1,2]`.
Thus four clients are not being confused with four-way model execution.
The failed initial test likewise records actual N=4 execution. Fresh C1/C3
success narrows the failure to something triggered by C4; it does not identify
the cause by itself. Token `!` is consistent with argmax choosing token zero
from invalid logits, but logits were not inspected in this run.

No GPU reset, host reboot, OOM kill, or privileged host change occurred.
After gracefully stopping both model containers for investigation, both report
`OOMKilled=false`; available host memory is about 118,300 MiB per node.

## Confirmed attention binding defect

The source audit found a separate concrete defect that must be fixed before
further full-model tests: GLM inherits the DeepSeek kernel source bundle, whose
`paged_decode_mla` module hardcodes 576 dimensions. GLM uses a 512-dimensional
cache, but the layer initialization selects that module unconditionally.
Runtime row strides do not change the compiled vector width. This causes
cross-head reads and overlapping output writes. It affects the dense fallback
used by both ordinary and sparse-enabled multi-sequence paths.

The C4 path computes absorbed queries one row at a time, whereas the enabled
C2/C3 projection path prepares all absorbed queries first. The invalid reads
therefore need not produce identical symptoms across batch sizes. Whether
correcting this binding fully resolves the C4 failure remains to be tested.

Next gates: dimension-selection regression tests, bounded guarded GPU oracle
checks for the 512-dimensional kernel, then a freshly loaded full model with
C1/C3 before C4, batch drain, and post-C4 fresh-request checks. Only after
correctness passes will scalar/grouped C4 throughput be measured.
