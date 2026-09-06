# Phase 3: bounded C4 correctness investigation

Status: corrected short-context C4 has passed the scoped serving gates below;
long-context C4 remains under validation. The initial failure and its resolution
are retained chronologically. Kernel-only speedups are not serving throughput
results.

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

## Corrected image: scalar control

The dimension correction is committed as `c7a574f3`. Image
`atlas-glm53-flash:kernel-20260906-v7` has identical binary SHA256 on both ranks:
`a8c67a8fe583a17272f3a78feea63217d7cdae1e70b06cb3868f2b7ba63b8bd0`.
Only the dense attention binding changes from the v6 serving source. The
configuration remains the initial scalar C4 control described above.

Fresh C1 and C3 pass, then all four independent needles pass on the previously
failing C4 smoke. A fresh C1 request afterward also passes. Two further C4
batches use prompts 1900/1920/1950/1984, output caps 96/64/48/64, and needle
position 0.9: all eight pass without foreign needles. One 1984-token request
actually returns all 64 output tokens, reaching the configured total limit.
These tests exercise slot reuse and noncontiguous drains, not just the first
four requests after load. Some other needle requests stop early; they are
quality checks, not fixed-output throughput receipts.

The matched throughput workload uses exactly 1024 prompt tokens and 64 output
tokens per request, one warmup and two measured batches per concurrency.
Temperature 0, seed 1, per-request repetition allowance enabled; every measured
output reaches the cap. Prompt token hash:
`e74375dce562b280752663f16202b17f37f388a64155a15e2db604c43fda23ec`.

| Scalar-control profile | Aggregate full-wall tokens/s | Aggregate post-first tokens/s |
| --- | ---: | ---: |
| C3 (existing grouped C3 MoE enabled) | 22.799 | 25.412 |
| C4 (four scalar MoE rows) | 17.212 | 18.149 |

Medians over the two measured batches. This is a negative C4 scaling result,
not a reason to increase concurrency by default. The next same-binary A/B
enables only grouped C4 MoE. The historical pre-fix dense outputs cannot serve
as an exact numerical oracle for this corrected image.

## Same-binary grouped C4 result

Only `GLM_C4_GROUPED_MOE` changes from 0 to 1; image v7 and all other serving
flags remain the same. The CPU-only v8 build was paused before throughput
measurement so it could not compete for CPU cycles. It was resumed afterward.

| Grouped-C4 profile | Aggregate full-wall tokens/s | Aggregate post-first tokens/s |
| --- | ---: | ---: |
| C3 unchanged control | 22.776 | 25.431 |
| C4 grouped MoE | 26.840 | 29.471 |

Same workload hash, one warmup and two measured batches; every output reaches
64 tokens. Grouped C4 improves full-wall throughput by 55.9% over scalar C4
and 17.8% over its C3 control. This is a short-context, non-speculative result,
not comparable without qualification to speculative peer counting workloads.

All four short needles and all eight repeated near-limit needles pass without
foreign needles. A request again reaches 1984+64 tokens. Raw quality and timing
receipts are `v7-grouped-c4-smoke.jsonl`, `v7-grouped-c4-boundary.jsonl`, and
`v7-grouped-1k-64.json` in the controller directory above. Grouped routed
activations are quantized to FP4; semantic checks do not prove unrestricted
quality equivalence to scalar BF16 activations.

## Exact M4 MLA batching and sustained coding workload

Image v8 includes the exact-M4 projection implementation (`16e59a66`; later
test-only commits do not change its serving binary). Both ranks have SHA256
`9bfe0b6283590cc9f8cf166d41d3185cde55e58518226259b1dce093773f098b`.
All flags match grouped v7 except the new `GLM_MLA_BATCH4` A/B switch; graphs
remain disabled. The standalone numerical gate is documented in
[c4-mla-batch4.md](c4-mla-batch4.md).

| v8, grouped C4, 1024 prompt / 64 output | Full-wall tokens/s | Post-first tokens/s |
| --- | ---: | ---: |
| M4 MLA off | 27.069 | 29.747 |
| M4 MLA on | 27.691 | 30.508 |

One warmup and two measured batches per arm; all outputs reach64. Full-model
improvement is2.3%, much smaller than the approximately2x isolated projection
speedup. M4-on passes four short and eight repeated near-limit independent
needles. These are same-image A/B results, not cross-version claims.

The four-request chat harness initially requested thinking off and failed two
visible-answer checks: sorting and code used all128 tokens for reasoning.
Atlas's template-forced thinking detection overrides that request for GLM.
The arithmetic and exact-JSON requests passed. After explicitly requesting a
32-token reasoning budget within the unchanged128-token cap, all four checks
pass on M4-off and M4-on. Validators are unchanged and no generated code is
executed. Raw initial failure is preserved in `v8-off-chat-quality.json`;
corrected-policy receipts are `v8-{off,on}-chat-budget32.json`. This is not a
full quality evaluation or a claim that disabling thinking now works.

To measure sustained decoding without the repetitive synthetic prompt, use
the committed [LRU coding workload](workloads/lru-cache-completion.txt):148
prompt tokens,256 output tokens, temperature0, seed1, **no repetition override**.
Token hash: `8b104308b377752ad9803d298be34e01cedf9f8ac577cacfea653c3359e7aebf`.

| Same v8 M4-on eager profile | Full-wall tokens/s | Post-first tokens/s |
| --- | ---: | ---: |
| C3 | 32.981 | 33.499 |
| C4 | 44.246 | 44.945 |

One warmup and two measured batches per width; all streams reach256 output
tokens. C4 improves full-wall aggregate by34.2% on this workload. Do not compare
44.246 directly with the1024/64 test or a single-stream decode number. The
coding completion is a throughput workload; generated implementations were
not executed or graded. Raw receipts: `v8-on-coding-c3-256.json` and
`v8-on-coding-256.json`.

## Short-context C4 graphs, matched v8 comparison

Same v8 binary and M4-on/grouped configuration; only
`NO_DECODE_GRAPHS_MULTISEQ=1` changes to `0`. Sparse indexing remains disabled,
context2048/chunk1024. No CPU build runs during either benchmark.

| C4 workload | Eager full-wall tokens/s | Graph full-wall tokens/s | Graph post-first tokens/s |
| --- | ---: | ---: | ---: |
| 1024 prompt / 64 output, repetition allowed | 27.691 | 28.321 | 31.270 |
| 148 coding prompt / 256 output, normal watchdog | 44.246 | 45.948 | 46.713 |

One warmup and two measured batches, matching the hashes above; all measured
streams reach their requested caps. Graph gains are2.3% and3.8% respectively,
not evidence that graphs resolve the remaining execution bottlenecks.
Four budgeted chat answers, four short needles, and eight repeated near-limit
needles pass. Server logs show captured C4 plus C3/C2 nonprefix slot subsets.
This result does not extend to long sparse C4, whose graph path is not enabled.
Receipts: `v8-graphs-{1k-64,coding-256}.json`,
`v8-graphs-{chat-budget32,c4-smoke,c4-boundary}` and both-rank logs in the
phase3 receipt directory. The tested containers are preserved, stopped, as
`atlas-glm53-v8-graphs-control-ep0/1`.
