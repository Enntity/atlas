# Bounded merged-image argmax diagnostic

Status: source-audited plan only; no diagnostic launch or causal result.
Context qualification has priority. Root owns image selection, deployment,
execution, artifact export and shutdown. No sampling or watchdog relaxation.

## Question and controls

The a069-era and merged native outputs reportedly share initial spaces and
then differ (`__slots__` versus a docstring). Different accepted-round counts
or later output hashes do not identify the cause. The exact branching token
position is **unknown until the original token IDs are compared**; four text
spaces need not be four tokens.

Use ordinary nonspeculative C1 workload, the same weights/tokenizer, explicit
BF16 KV and NVFP4 target head, original numerical flags, and the original
148-token prompt **as exact token IDs**, not a re-rendered or re-tokenized
string. Preserve actual sampling parameters; cap the diagnostic at one output
token. Record complete argv/env and both immutable image/server hashes.
Do not enable the new paged-prefill projection experiment for this comparison.

Four cases, each with separate rank0/rank1 dump directories:

| Case | Image | Exact prompt |
| --- | --- | --- |
| `old/prompt` | pinned a069-era image | original148 IDs |
| `new/prompt` | pinned merged image | identical148 IDs |
| `old/shared-prefix` | same old image | original IDs + agreed common emitted-token IDs |
| `new/shared-prefix` | same merged image | identical extended IDs |

The common prefix must stop before the first differing token. Never force a
prefix chosen only from one divergent branch. Fresh process/cache ownership
is preferred for each case; do not compare a cold run with a prefix-cache hit.
These are teacher-forced **prefill** comparisons, not a reproduction of the
same autoregressive state evolution. A row difference localizes a numerical
difference at this boundary; it does not by itself explain the original run.

## Existing capture and limits

Use `ATLAS_NEMO_DUMP=<fresh rank-specific directory>` on the actual chunked
prefill path, including when the entire148-token prompt fits one chunk.
The hooks exist in both `a069efc3` and the merged source:

- [`forward_layers.rs:296`](../../../crates/spark-model/src/model/trait_impl/prefill_b/forward_layers.rs#L296)
  writes an H-element slice after each layer, on the final chunk:
  `atlas_L0.bin` through `atlas_L44.bin` for the observed45-layer model.
  Its offset is `(proc_count-1)*H` elements, not an independently checked
  HC4 row-stride calculation. Do not label these files the full mHC residual
  or use them as proof of exact last-token layer divergence; prioritize the
  actual final-norm input to the head and the resulting logit row.
- [`finalize_last.rs:156`](../../../crates/spark-model/src/model/trait_impl/prefill_b/finalize_last.rs#L156)
  writes `atlas_final_norm.bin` (H elements).
- [`finalize_last.rs:219`](../../../crates/spark-model/src/model/trait_impl/prefill_b/finalize_last.rs#L219)
  writes `atlas_logits.bin` (V elements), before scheduler sampling, and logs
  top10 IDs/values. Its BF16-to-FP32 expansion preserves each stored logit.

At H4096/V154856/L45, exact raw-file payload per rank is
`45*4096*4 + 4096*4 + 154856*4 = 1,373,088 bytes` in47 files.
Both ranks total2,746,176 bytes/case; all four cases total10,984,704 bytes.
These are not weights or full-token activation dumps. Directory/filesystem
overhead and bounded ordinary logs are additional; reserve under32MiB total
for these raw artifacts and keep the complete diagnostic below100MB.

The code creates directories and overwrites files, not append-only history.
It ignores filesystem write failures. Root must precreate private writable
directories, verify47 files and their exact sizes, and export/hash the raw
files **after the scoped request and before any next prefill**. Rank directories
must be distinct even if both containers mount the same host export root.
Missing files, unexpected shapes or mixed requests invalidate the capture.

## Stream and vocabulary meaning

The logit hook first calls `gpu.synchronize(stream)`, then synchronous D2H
from `buffers.logits()`. Per-layer and final-norm hooks likewise synchronize
the producing stream before reading. Existing async NCCL reductions join
back into the compute stream through completion events
([`comm_impl.rs:46`](../../../crates/spark-comm/src/nccl_backend/comm_impl.rs#L46));
this diagnostic must run only on the healthy normal path, not as error cleanup.
These synchronizations perturb timing: no dump-enabled throughput claim.

For this ordinary GLM BF16-logit path, `lm_head` writes the entire configured
V-vector on each rank (`impl_a3.rs:341`), and `decode_logits_ptr` resolves to
`buffers.logits()` (`impl_a3.rs:495`). This is **not** the paired draft head's
77428-column local vocabulary shard and requires no concatenation of rank dumps.
Compare rank0/rank1 full rows independently; equality is useful corroboration,
not a guarantee supplied by the dump. Restrict to C1 row0: the hook hardcodes
the base buffer, so it is not a valid general co-dispatched-row or FP32-logit dump.

## Interpretation and unavailable shortcuts

Compute finite-value checks, exact row equality, differing-element count,
maximum absolute difference, each image's top contenders and exact margins,
and both contenders' values in the other image. Compare raw bytes/FP32 values,
not rounded log text. Exact BF16 ties must be reported as ties, not evidence
of identical sampler tie-breaking. Retain sampled IDs and original response.

`ATLAS_DUMP_LOGITS_PATH` is absent from a069 runtime/server source. Merged
hooks alone cannot provide an old/new comparison. `ATLAS_LOGIT_DUMP` exists
in both, but GPU greedy can bypass it; its caller passes already processed
logits despite the historical `raw_topk` label. Forcing host sampling via
logprobs could change tie resolution. Neither shortcut proves unchanged-policy
raw autoregressive equivalence. A later true branch-position decode comparison
would require a separately reviewed narrow diagnostic backport, not inference
from these prefill rows.
