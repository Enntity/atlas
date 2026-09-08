# GLM speculative GPU feedback: source comparison

2026-09-08. Read-only followup while the paired E1/F5 transport is implemented.
No kernel, runtime, factory, scheduler dispatch or native profile changes here.

The local reference `/home/abc/storage/models/dcp-audit/vllm` is at
`e232d262369b8c918cf478a7a96a0fcf8127cf65`, not the earlier online checkpoint's
`487ecf...`. The inspected sampler/speculator files are unmodified. Treat these
as separate source snapshots, not proof of the engine used by a headline rate.

## Concrete dependency differences

In vLLM's [autoregressive speculator](https://github.com/vllm-project/vllm/blob/e232d262369b8c918cf478a7a96a0fcf8127cf65/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py),
draft input updates and sampled tokens remain device tensors across steps.
Its [rejection sampler](https://github.com/vllm-project/vllm/blob/e232d262369b8c918cf478a7a96a0fcf8127cf65/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py)
returns device tokens/counts indexed by requests. This avoids needing a host
acceptance count merely to prepare the next draft inputs.

The [model runner](https://github.com/vllm-project/vllm/blob/e232d262369b8c918cf478a7a96a0fcf8127cf65/vllm/v1/worker/gpu/model_runner.py)
starts output copying before postprocessing and proposal. Its
[async output owner](https://github.com/vllm-project/vllm/blob/e232d262369b8c918cf478a7a96a0fcf8127cf65/vllm/v1/worker/gpu/async_utils.py)
retains source tensors, orders the copy stream and waits on its event later.
This is explicit lifetime management, not removal of necessary synchronization.

Atlas's actual dependencies are:

- `model/trait_impl/verify_d.rs`: K5 argmax readback is 20 bytes. Checked server
  selection can add small positivity probes or a `5 * vocab * 2` logits copy.
- `layers/glm5_mtp.rs`: four host-driven draft iterations. Each iteration reads
  a 16-byte TP argmax pair, or a four-byte global argmax, before the next token
  becomes a scalar host input to `forward_body_one`. The common unfused embedding
  path derives its source pointer from that host token. Changing only the
  argmax return type cannot remove this dependency.
- `cuda_backend/gpu_copy.rs`: ordinary D2H explicitly synchronizes the default
  stream before returning. The issue is feedback latency, not the byte volume.
- Two serialized owners therefore have eight per-draft readback barriers per
  full proposal round, per rank. A future M10 target kernel does not remove them.

## Candidate experiment, not a speed prediction

After the paired control is correct, assess device-token embedding and device
selection of the gathered TP argmax pair. Four fixed draft steps could retain
their token IDs in reserved device storage and copy the final four IDs once.
The target/body mathematical dependency remains sequential; only host feedback
and submission gaps can disappear. Preserve existing local argmax reduction,
rank-0 tie preference, nonfinite comparison behavior, KV metadata and stream
ownership. Hidden diagnostics and grammar are separate eligibility concerns.

An independent feasibility review found a narrow reuse route: keep the current
local argmax/value kernel and TP all-gather, add an exact two-pair device
selector, then use existing `batched_embed(n=1)` for device-selected input IDs.
The existing fused EH normalization can consume that staged row with token zero.
This preserves its arithmetic but adds an embedding-gather launch, so it is not
automatically faster. Pair selection must use the existing strict `v1 > v0`
comparison, including rank-zero ties and unordered/NaN behavior.

Metadata remains a separate potential host barrier. `forward_body_one` uploads
one pageable packed metadata vector per draft via `copy_h2d_async`. The backend
does not explicitly synchronize that pageable case, but the
[CUDA synchronization rules](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)
permit driver synchronization during pageable staging. Removing draft D2H does
not prove four steps can be submitted without host waits. Measure actual API
durations, or separately design retained pre-uploaded four-step metadata; do
not weaken the general upload lifetime contract or merely rename an API async.

First measure distinct readback/selection paths and host submission gaps without
adding timing synchronizations to a qualifying TPS run. Keep detached-row and
KV-writer completion costs separate: their current fences protect actual owned
state and host metadata lifetimes. They cannot simply be deleted.

Historical, not current-v26, receipts in `docs/glm53-dual-spark.md` put proposal
near 10.7–11 ms versus target K5 near 94–96 ms. Those numbers bound the old
proposal opportunity and do not establish today's readback fraction. No new
profiling run was made for this audit.

Do not transplant vLLM's sampler wholesale. Its non-greedy rejection uses
probability ratios and its penalties incorporate preceding draft positions;
Atlas's current path preserves sampled-token equality and its existing history
semantics. GPU residency is the transferable design principle, not a claim of
identical sampling, quantization, TP/EP topology or expected throughput.
