# Code Review — 2026-10-10

Scope: current uncommitted changeset, with emphasis on inference hot paths, GB10 memory traffic and capacity, and profiling/debug visibility. No source files were changed.

## Findings

### P1 — Routed MoE scratch scales with full token capacity in every layer

`crates/spark-model/src/layers/glm5_exl3_ffn/moe/setup.rs:106-126` sizes and allocates eleven scratch buffers from `capacity`, including per-route F32 gate/up/down tensors and two additional F32 transform buffers. The native layer loader passes the complete `max_batch_tokens` value as this capacity for every layer (`crates/spark-model/src/factory/build.rs:258-265`). In particular, buffers 7 and 8 alone each reserve `capacity * topk * max(hidden, local_intermediate) * 4` bytes per routed layer. With a 4,096-wide hidden state, top-k 8, and capacity 1,024, that is 256 MiB per layer for those two buffers alone; the gate/up/down tensors add more. Repeating this across the model can consume many GiB before KV cache allocation, reducing GB10 headroom or preventing startup.

The layers execute sequentially, so their transient workspaces can be shared at model scope, or otherwise sized from the actual maximum rows concurrently processed by one layer. Account for this scratch in the serving memory budget before allocating it.

### P2 — Vocabulary head stages every logit through oversized F32 buffers

`crates/spark-model/src/model/glm_exl3_head.rs:134-145` allocates three capacity-sized F32 buffers for local, gathered physical, and active vocabulary data, in addition to a BF16 local buffer. `forward` writes BF16 projection output, converts it to F32, gathers and reorders F32 logits (`:227-254`), and `project_exl3_vocab` immediately converts the active logits back to the serving BF16 buffer (`:261-282`, `:378-386`). The F32 staging costs roughly 1.6 MiB of persistent scratch per row for a 154,880-token vocabulary at TP=2, and the conversion, collective, reorder, and down-conversion add substantial full-vocabulary memory traffic on a bandwidth-constrained GB10.

Keep the serving path in BF16 through gather/reorder where supported, and reserve the F32 result path for consumers that require F32 logits or explicit validation capture.

### P2 — Native EXL3 FFN has no stage-level timing instrumentation

The new dense and routed paths expose optional tensor dumps (`crates/spark-model/src/layers/glm5_exl3_ffn/dense.rs:211-270` and `crates/spark-model/src/layers/glm5_exl3_ffn/moe/forward.rs:29-168`), but they do not emit stage timings or tracing spans. The enclosing KDA profiler measures the whole FFN as one interval (`crates/spark-model/src/layers/glm5_kda/forward_ffn.rs:74-80`), leaving router GEMM/top-k, gate/up transforms and matmuls, activation, down projection, shared experts, and combine indistinguishable when diagnosing slow inference. This makes it difficult to identify whether the GB10 path is launch-bound, compute-bound, or moving excess memory.

Add opt-in per-stage timing or spans around those operations, including rows, selected routes, and the chosen thin/fat precision profile. Keep synchronization limited to profiling mode so normal inference remains asynchronous.

## Review notes

The report was produced from the current working-tree changes, including untracked additions. No tests were run because this was a read-only code review; only this report was written.
