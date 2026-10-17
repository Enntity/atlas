# Gate v2: pinned vLLM reference contract

Reviewed 2026-10-12 by a user-requested Luna subagent and reconciled against
primary source by the parent agent. Read-only review; no new GPU evidence.
The installed reference implementation, registered overlays and run receipts
are authoritative for reference-engine behavior. Gate v2's statistical rules
are Atlas experiment policy, not rules supplied by vLLM.

## Source identity and precedence

`SRC` below is `/home/colum/Workspace/vllm-reference-src/`; `EV` is
`/home/colum/Workspace/atlas-exl3-validation-20261010/`. The source export
has no Git metadata. Bind files to the archived source hashes and each run's
manifest, not an inferred checkout HEAD or the older root provenance profile.

The installed plugin extract takes precedence over the different checkout
`quantization/exl3.py`. Registered observer files take precedence over older
collection packages. Parent review rehashed these three source files:

| Source | SHA256 |
|---|---|
| `SRC/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py` | `d665ef2109b0183d48e3541ecd24e9fa8e1dc3e410983bc29b8d997af9d7cd01` |
| `SRC/vllm/model_executor/layers/attention/mla_attention.py` | `a309ca752b01e85360b268fba56245b816ff245340e9e361166ee96b99d93a94` |
| `EV/arithmetic-audit/atlas-reference-active-exl3_mul1.py` | `6a0624582898f5f840fdd5524d72f174188648f299be170b910f6d596539dcaf` |

Observer/launcher hashes remain in the [v1 collector-only registration](exl3-statistical-gate.md#collection-restart-2026-10-11-instrumentation-and-single-row-reader).
The authoritative archived observer copies are under
`EV/statistical-gate-20261011/reference-profile-audit/rank0/sources/tmp/atlas-exl3-vllm-upstream-context-v4-patch/`
(rank1 has the matching copies). Source identity alone does not prove which
compiled kernel or runtime plan executed; keep the existing binary/provider
binding obligations.

## Source report compared with the qualification plan

| Process | Source/receipt evidence | Required interpretation in the plan |
|---|---|---|
| Teacher forcing | In-repo `tests/exl3_validation/collect_logits.py:42` and `EV/long-context-chunk128-registration/collect_phase_logits.py:53` construct prompt IDs plus prescribed continuation IDs, then submit serial `/v1/completions` requests with `max_tokens=1`, temperature 0, seed 42. | Each row is a separate prescribed-prefix request; generated tokens do not choose later prefixes. Short16 then long stays fixed within each attempt; only attempt order is randomised. |
| Chunk positions | Registered `reference_chunk_trace.py:44` wraps the original V2 `GPUModelRunner._prepare_inputs`, checks its source hash, requires one unpadded request, no speculative tokens and at most 128 scheduled tokens; `:118` and `:122` gate/bind terminal one-row captures. | Context 4096/chunk 128 is a checked collection profile, not an assumption based on prompt length or `max_tokens=1`. Preserve this observer and its receipts. |
| Captured quantity | Registered `sitecustomize.py:44` calls the original logits processor, then `:55` exports rank-0 logits as FP32. `SRC/vllm/model_executor/layers/logits_processor.py:70–82` applies the head and configured scale/soft-cap; `:84–96` implements gather and `:143–153` applies the head, TP gather and vocabulary truncation. | Compare post-head/post-gather, pre-sampler logits. Export as FP32 does not imply FP32 head arithmetic. The image is unchanged but the two registered observers instrument its process. |
| Sampling | Bound V2 path: `SRC/vllm/v1/worker/gpu/sample/sampler.py:264` selects the Gumbel implementation for greedy/explicit-seed requests; `sample/gumbel.py:135` adds random noise only when temperature is nonzero. | Seed 42 is recorded request policy, not a guarantee of deterministic model-forward arithmetic. Keep HTTP text/argmax checks separate from logit metrics. |
| Effective MLA cache | SM120 backend `:36` disables dense-MHA prefill, `:67` requires `fp8_ds_mla`; MLA wrapper `:550` disables its dense-prefill backend and `:824` routes all such tokens to MQA. SM120 `:163` passes packed cache directly to FlashInfer. Ref13 rank1 log `EV/statistical-gate-20261011/batch2-attempts/spark1/ref13/docker-rank1.log:73` records backend/format. | Reference requests `auto`; effective MLA storage is `fp8_ds_mla`. Candidate uses `fp8_g128`. Matching format policy does not prove identical attention arithmetic. Short prefill is cache-backed too; no live-latent bypass exemption. |
| Native EXL3 projection | Installed plugin `:3087` uses the FP16-input trellis path below its large-row threshold; `:3232` installs the head with `pad=False`. | Do not substitute reconstructed BF16 weights or assume head min-row padding. Existing component receipts remain scoped; qualification still needs the current image. |
| Fresh processes and recurring modes | Fresh-run protocol and repeated row modes are documented in `EV/q2-ref13-investigation-20261012/report.md`. No inspected source/receipt identifies the mechanism producing that mode. | Fresh processes are an experimental control, not a vLLM determinism guarantee. Atomics, routing ties, autotune and collective choices remain hypotheses for this particular mode until runtime-bound evidence isolates one. |

The broad code-path audit is historical. Its short-live-latent, missing-config
and head-padding claims were already corrected in
the 2026-10-12 reconciliation after independent review.
This review agrees with those corrections. Luna's initial source-unavailable
statement was withdrawn after locating `SRC`; its generic context-cache
upconversion observation is not the selected SM120 execution path and is not
used as evidence for that path. The parent checked the V2 sampler rather than
transferring seed behavior from the older `gpu_model_runner.py` implementation.

Fresh containers are not empty runtime environments: the pinned rank create
commands mount host vLLM, Triton, TileLang, CUDA compute and EXL3 autotune
caches. Record which mounts persist across attempts and their pre/post content
manifests using the gate's cache-state receipt contract. This documents a
possible source of process-external state without attributing the mode to it.

## Consequences for Gate v2

- Keep the exact pinned engine, installed plugin, checkpoint, observer mounts
  and serving profile. A different upstream vLLM version is not this oracle.
- A runtime-choice probe is diagnostic and uses separate attempts. If pinning
  changes the reference process/profile, amend the registration before fresh
  qualification; do not mix those draws with the existing profile.
- Fixed sample count, H4 membership and acceptance criteria remain explicit
  Atlas decisions. vLLM source cannot justify adaptive-stopping rank bounds.
- Passing the gate qualifies these fixtures under this observed profile. It
  does not establish distributional identity, general task quality or release
  readiness. The mode's cause and compiled-kernel path equivalence remain open.
