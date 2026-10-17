# EXL3 reference runtime mechanism diagnostic plan

Prepared 2026-10-14 by CPU-only review of extracted reference source. This is a
separate diagnostic design, not an amendment, qualification campaign or GPU
execution receipt. Root currently owns both Sparks for C2; do not launch this
work until that owner releases them. No remote actions were taken for this plan.

The immediate objective is to bind final-logit byte classes to **executed**
EXL3 launch plans, router decisions and collective dispatch. Identical disk
cache hashes and environment settings do not establish identical runtime
choices. A policy that removes the mode has not been identified.

The fresh failures are H3 zero-based index 13, attempt `v2ref14`,
`short16/longer/r3`, and H4 index 14, attempt `v2ref15`, `short16/code/r1`.
Keep these distinct from the historical recurring `short16/longer/r0` mode
in the old ref13 investigation. Observe all three rows without assuming a
common cause. The
[reference diagnosis](../design/exl3-reference-verifiability-diagnosis-20261014.md)
contains their full reference-only pair structures; the fixed campaign remains
UNVERIFIABLE.

## Source and binary contract

`SRC=/home/colum/Workspace/vllm-reference-src` and
`AUDIT=/home/colum/Workspace/atlas-exl3-validation-20261010/arithmetic-audit`.
The source export has no Git metadata. Use exact hashes, then verify imported
module paths and hashes on both ranks before execution. The installed plugin
extract takes precedence over the different checkout `quantization/exl3.py`.
Keep the
[source contract](../design/exl3-vllm-reference-contract.md) and the registered
reference image/build/checkpoint/observer hashes from
[diagnosis.json](../design/evidence/gate-v2-reference-diagnosis-20261014/diagnosis.json).
Extracted native source describes possible behavior; bind it to the actually
loaded `exllamav3_ext` and NCCL shared libraries before claiming engagement.

| Local source anchor | SHA256 |
|---|---|
| `AUDIT/atlas-reference-active-exl3_mul1.py` | `6a0624582898f5f840fdd5524d72f174188648f299be170b910f6d596539dcaf` |
| `AUDIT/installed-linear-exl3.py` | `c010bd18aaf5363632db25c0a4f7c4be0938011656f0446f933505a59b8d6cc0` |
| `AUDIT/installed-extra-source/quant/coop_autotune.cu` | `f6096afdc617b038c09b42cd4fab331a929ef5d191843e9c80366affde01987a` |
| `AUDIT/installed-extra-source/quant/coop_autotune.cuh` | `1a2bfaae6151ea45229d839885099803a028896b8ad70fc322e832a4b5213b71` |
| `AUDIT/installed-extra-source/quant/exl3_gemm.cu` | `2455154041c8573832fbe7ef814efe7bef0cf2cd6be1c3ed5050fb16b1f5f8fc` |
| `SRC/vllm/models/glm5next/nvidia/model.py` | `a12142897fc5bdc5cff693d4d63c88a782f9ef7f35879963f45e1682d7faaed8` |
| `SRC/vllm/model_executor/layers/fused_moe/router/base_router.py` | `47fa2324bcd9b6ebdc138d69ae6b6ecc0f2463d34bda8c5988bdb0a2a76d3fae` |
| `SRC/vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py` | `6a541bfdcf9e9a09fc888ea3f4a2e9689181e889e3a3ac9f49716d8d59cca142` |
| `SRC/vllm/distributed/device_communicators/cuda_communicator.py` | `a9a0242175c3d3e0d46d586a38cb76139e6e4e92d15032d5b1e33237af2f6757` |
| `SRC/vllm/distributed/device_communicators/pynccl.py` | `28596817ee9fb842026624a8a6c32b0637630b6b908ddb2a79ecccbeac531e35` |
| `SRC/vllm/envs.py` | `461c316ff917e35d985b41eb611fc6d38dc53729639e618e4545665615dd30c9` |

The native hashes above match `AUDIT/installed-extra-source/source-sha256.json`.
Do not copy permissively licensed external implementation into Atlas; implement
small observer/decoder glue against these interfaces with Atlas source headers.

## Real telemetry boundaries and their limits

**EXL3 startup and cooperative kernels.** The plugin's `exl3_shape_key:1845`,
`iter_exl3_handles:1856`, and `pretune_exl3_shapes:1869` enumerate real native
handles and warm row buckets 1/2/4/8/16. Record named-layer identity, in/out
features, K, codebook, rows, activation/output dtype and trellis shard identity.
`LinearEXL3.forward:114` calls `self.bc.run_alloc` for the small-row path;
it returns tensors, not a public Python launch-plan object.

`exl3_gemm.cu::gemm_autotune_hash:53` includes bucketed rows, k/n, K,
FP32-output flag, device, compute capability, maximum SM count and codebook;
`mgemm_autotune_hash:84` also includes batch dimensions. Apply the version-3
salt from `coop_autotune.cu::salt_hash:68` before matching disk records. Do not
assume the Python five-field shape key is the entire native key.

`coop_autotune.cu::load_disk_cache_locked:132` accepts magic `EX3ATUNE`, format
1, and a record size at least the native structure size. `DiskCacheRecordV1` stores hash, tag,
block dimension, SM count and concurrency; duplicate hashes are overwritten
in file order, so the final record wins. Decode with explicit little-endian
layout only after verifying the producing ABI's header/record sizes (expected
16/32 bytes for the bound aarch64 build); the diagnostic decoder conservatively rejects any size outside the attested
ABI, unknown versions and truncated records. Keep both raw file SHA256 and canonical final
record-map SHA256. This is passive **disk** telemetry.

`launch_from_disk_cache:191` validates the record against candidate tag,
block dimension, SM range and expected concurrency. Rejection falls through
to tuning. `CoopKernelAutotuner::launch_locked:577` uses the in-memory map and
launches the cooperative kernel; `launch:610` tries memory, then validated
disk, then measurement. Equal disk maps do not prove cache hits or executed
geometry. Existing `cache loaded/lookup/hit/rejected/store` printf calls are
guarded by **compile-time `CACHEDEBUG`**, commented out near line 22. Setting an
imaginary environment debug variable will not enable them. A `CACHEDEBUG`
extension rebuild would be a separately bound diagnostic binary with neutrality
checks, and would still need per-call memory-hit telemetry.

For the first observational experiment, preserve the native binary and use
Nsight Systems/CUPTI CUDA launch tracing, if available, to record actual kernel
identity, grid, block, shared-memory size and stream on both ranks. Bind the
tool version and exact invocation; validate child-worker coverage rather than
assuming profiling the HTTP parent sees kernels. Add observer NVTX spans around
named native-handle calls only after reviewing the installed NVTX API. Match
actual cooperative geometry to disk records where unambiguous; mark unmatched
or many-to-one joins unresolved. The plan does not claim profiler availability
or install it during C2.

**Routing.** `Glm5NextMoE.forward:252` computes external gate logits before
`self.experts`. `BaseRouter.set_capture_fn:185` is a supported logical-ID
capture hook; `_select_experts:260` invokes it after `_compute_routing` and
before EPLB mapping. Register per actual router instance and record its concrete
class, EPLB state and bound callback count. The installed EXL3 method's
`apply:2490` supplies the actual `topk_ids` and `topk_weights` to
`apply_exl3_experts:1906`; capture that boundary too when weights or physical
IDs are needed. Record gate-logit, ID-order and weight bytes, not histograms
alone, for the three target requests and each routed layer. Derive layer IDs
from named modules; bind request prefix and runner context on both ranks.

The real `EXL3_FAT_EXPERT_LOG` path (`fat_expert_log_enabled:236`) adds host
synchronization under grouped execution. `exl3_fat_diag:487` reports tiers and
counters, but neither histogram nor tier proves identical expert ordering.
Use existing counter snapshots first; introduce detailed tensor snapshots as
explicit instrumentation and report synchronization/allocator effects. Buffer
events and flush at a defined boundary; never retain a mutable tensor reference
as an alleged earlier snapshot. Avoid extra forwards or random warmups.

**Collectives.** `CudaCommunicator._log_all_reduce_backend_selection:209`
lists enabled backends, explicitly not the per-tensor selection.
`all_reduce:275` dispatches through supported alternatives and eventually
PyNCCL or torch-distributed fallback. Instrument the selected backend's actual
method calls, not only the top-level return, and fail coverage if an unknown
branch executes. Record group/rank, call sequence, tensor shape/dtype/count,
stream, operation and return/fallback. Include gather, since captured logits
are post-TP-head-gather. `PyNcclCommunicator.all_reduce:166` and
`all_gather:199` bind to actual `ncclAllReduce`/`ncclAllGather`, but do not
select or report NCCL's internal algorithm/protocol. Hash loaded NCCL library
paths on both ranks. Kernel traces can show executed collective kernels;
algorithm/protocol attribution requires version-bound NCCL telemetry. This
local source review provides no validated NCCL algorithm pin and proposes none.

## Minimum next reference probe: observation before intervention

Finish the decoder and observer harness locally first, with fixtures exercising
real capture callbacks, exact argument forwarding, unknown-source rejection,
duplicate/truncated cache records and request/layer binding. Freeze source,
binary, package, tool hashes and the exact attempt list before obtaining either
Spark. The proposed count is diagnostic only and is not a change to gate N.

1. Register six fresh-process attempts in the fixed order `off01,on01,on02,
   off02,off03,on03`, where the only arm difference is the combined telemetry
   package/profiler enabled state. This is three repeats per arm. Freeze a
   per-rank copy of the present cache state and initialize an isolated writable
   copy from that same snapshot for each attempt, mounted at the original
   container paths. This changes cache handling for both arms, so explicitly
   register the diagnostic profile; never use its samples as qualification
   draws. Do not reset or overwrite original campaign caches.
2. Keep the exact checkpoint, TP=2/EP=1, flags, observer mounts, temperature 0,
   seed 42, full short16-then-long request order and chunk128 runner. Effective
   reference MLA format is `auto`→`fp8_ds_mla`, despite the legacy BF16 field.
   Collect every row, HTTP response, prefix receipt and both-rank cache manifests.
3. In the on arm record the actual native-handle calls and CUDA launches,
   logical/physical routing signatures and chosen collective backend/sequence.
   Require one complete telemetry chain for each target request on **both**
   ranks. Startup-only logs, missing child CUDA traces or zero callback counts
   fail engagement and terminate analysis as incomplete rather than “inert.”
4. Before mechanistic attribution compare on/off final-logit distributions and
   capture identity. Any same-class byte match is scoped evidence of neutrality;
   three matches do not prove distributional neutrality. If telemetry causes
   a new systematic class, retain it as an observer effect and reduce observer
   overhead before another separately registered design. Do not label an
   unobserved singleton eliminated.
5. Report every predetermined attempt, class frequency, pairwise row metrics,
   earliest differing gate/route/collective/plan signature and missing joins.
   The result may simply be “no recurrence, mechanism unresolved.” Do not add
   attempts until the historical singleton reappears.

This is the smallest proposed fresh probe with repeats and visibility into the
three candidate mechanisms. If full telemetry needs a native rebuild, stop at
that known tooling limitation, preserve the reference binary, and separately
register a rebuilt-observer comparison before runtime collection.

## Later one-variable interventions, selected by observational evidence

Register a separate fixed experiment for one observed mechanism, with three
fresh processes per arm in alternating order and **two predesignated fresh
held-out processes per arm**, excluded from mechanism/center fitting. Freeze
the choice and held-out rule before collection; retain all attempts. A held-out
check compares complete telemetry coverage plus whether each held-out row lies
within its arm's fitted measured nearest-reference envelope under unchanged
H3 constants. This is a diagnostic generalization check, not gate qualification
or a false-reject guarantee. Record H4 applicability without inventing a new
fallback when two modes are not demonstrated.

| Observation | One supported intervention | Engagement and limits |
|---|---|---|
| Same shape reaches differing EXL3 plans or fresh tuning | Freeze that rank's validated final disk plan map in both arms; change only tune-cache mount write permission, writable versus read-only, at the same container path. Preserve non-tune caches. | Verify all used keys are present, accepted and execute the expected geometry; no cold tuning. Missing keys/rejections make the experiment incomplete. Read-only disk does not itself pin in-memory choices. If both arms execute the same geometry, this treatment did not vary the hypothesized mechanism. |
| Identical gate inputs yield differing selected ID orders/weights on the grouped path | Set only `VLLM_USE_FUSED_MOE_GROUPED_TOPK=0` in the treatment; baseline records its effective current value. `envs.py:1573` and `grouped_topk_router.py:92` implement this path selector. | Require actual grouped routing, eligible group count/top-k and correction bias; verify the baseline calls `fused_grouped_topk:28` and treatment executes the nonfused branch. Invalid grouping may instead call fused_topk_bias/fused_topk and makes this switch inapplicable. This changes routing arithmetic, not a guaranteed tie policy. |
| Actual collective dispatch or NCCL kernel differs while inputs match | First collect a version-bound NCCL library/source capability record. Choose one documented, supported algorithm/protocol setting only after that review, in a new exact registration. | No NCCL pin is authorized or specified by this plan. vLLM backend switching, network switching or disabling PyNCCL would change multiple numerical/runtime properties and cannot establish a pure NCCL algorithm mechanism. |

`VLLM_BATCH_INVARIANT` is a real broad setting, but also affects collective
dispatch; do not use it as a one-variable router-tie probe. Native
`exl3_gemm` accepts `force_shape_idx/force_num_sms`, but `LinearEXL3.forward`
does not expose those as a serving pin. `GLM53_COOP_GEOMETRY` appears in the
environment but its launcher-to-native implementation was not located in this
review. None is a proven end-to-end reference pin.

If an intervention changes final logits with engaged telemetry, report that
observation and its fingerprint. A causal claim requires reproducing the
baseline mechanism and isolating the changed runtime choice; absence of modes
in a few repeats is insufficient. Use an independent observation path (CUDA
launch trace versus cache/native logs, actual apply inputs versus router hook)
before dismissing any historical record as an artifact.

## Qualification boundary

This plan does not modify H3/H4 constants, component count, singleton treatment,
fixed attempt count or reference identity. No candidate arrays inform mechanism
selection. A demonstrated pinning policy would need a separately reviewed
pre-collection amendment, evaluator/control validation, and a newly registered
qualification campaign retaining the existing UNVERIFIABLE result. Until then,
the next actionable result is runtime-bound diagnostic evidence, and the
qualification policy remains unresolved.
