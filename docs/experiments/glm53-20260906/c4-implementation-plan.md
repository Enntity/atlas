# Bounded C4 independent-decode implementation plan

2026-09-06. Design only; no runtime changes or hardware validation are claimed.
Implement only after the current C2/C3 graph campaign has a retained control.
Paths below start at the repository root; Rust layer paths are under
`crates/spark-model/src/layers/` unless stated otherwise.

## Scope and first launch contract

Add one explicit experimental `ATLAS_GLM_C4_DECODE=1` opt-in, propagated
identically to both ranks. Keep its default off. A separate default-off
`ATLAS_GLM_C4_GROUPED_MOE=1` isolates grouped activation precision from the
independent-row implementation. First scope:

- Exact `glm5_next`, TP2+EP2, EP protocol v2, independent non-speculative rows.
- BF16 MLA KV and semantic index, FP32 KDA state, unchanged NVFP4 checkpoint.
- Active cap4, admitted cap4 initially; no C5/C7/C8 admission expansion.
- `max_seq_len <= 2048`, preferably1024 for first smoke tests. Prompt plus
  completion must fit the cap, not merely the initial prompt.
- Existing `ATLAS_GLM_KDA_MULTI_SEQ=1` is mandatory; sparse and sparse-graph
  flags remain off for this initial short-context lane.
- Start eager; enable the existing exact-width, slot-keyed dense graphs only
  after C4 correctness and all drain widths pass. No graph borrowing in EP.
- Preserve watchdog, rollback ring, KV allocation safeguards, safety reserves,
  thermal limits and bounded startup/benchmark deadlines.

The short dense multi-sequence path does not maintain semantic-index history.
Therefore this is a separately restarted, context-capped lane, not a way to
switch a live C4 process into long sparse decoding. Do not lift the current
C2/C3 dynamic-selector width guards during this first implementation.

## Minimal files and dispatch changes

| Location / function | Bounded change |
|---|---|
| New small model policy module, exported for server use | Pure C4 policy accepts exact model/topology, flag, no-spec, required KDA mode and short context. Keep existing C1/C2/C3 behavior unchanged. Give C4 an explicit failure if prerequisites are missing. |
| `scripts/start-glm53-ep2.sh` | Validate/propagate the opt-in to both containers; allow active4 only under that policy, retain admitted<=5 globally but use4 initially. Reject C4 long context, any speculation and sparse flags in the first lane. |
| `crates/spark-server/src/main_modules/serve_phases/preflight.rs` | Extend `glm5_concurrency_supported` narrowly through the same policy. Direct `spark serve` must enforce the same contract; keep long-context and MTP guards unchanged. |
| `crates/spark-model/src/model/trait_impl/decode_a2.rs` | Validate C4 policy/positions and required arenas every step before graph lookup. An active token's host position must be<2048. Preserve exact distributed `decode_dispatch_width` and ordered SSM-slot graph keys. |
| `glm5_kda.rs::decode_multi_seq` | Dispatch opted-in N4 into the genuine batched implementation. Reject unsupported N4 instead of falling through to scalar decode. |
| `glm5_kda/multi_seq.rs::decode_multi_seq_inner` | Permit N4 only through policy. Existing row-major mHC, projections and per-state convolution/recurrence loops are N-parametric. Extend FFN selection separately as below. |
| `glm5_kda/projection.rs::project_hot_multi_decode` | Reuse existing `w4a16_batchm.kernel(4)` arm and BF16 batch-M side projections. Validate nonzero M4 handles; no new projection CUDA required. |
| `qwen3_attention/trait_impl/multi_seq/mla_glm.rs` | Existing eligibility/projection dispatch already covers N4. Keep row-wise Q absorption/V extraction fallback for first C4: it has explicit row pointers. No need to create M4 absorption CUDA before measuring. |
| `qwen3_attention/trait_impl/multi_seq/mod.rs` | Add an explicit N4 FFN arm in `decode_multi_seq_inner_hc`; do not route N4 through `forward_k5_for_hc`. Consume contiguous four-row MoE output using ordinary N4 `hc_post`. |
| `moe/prequant_fp4.rs`, `moe/forward_prefill_phase.rs`, `moe/forward_prefill_routed.rs` | Add narrowly gated C4 grouped eligibility, exact-M4 shared projections, and reuse of the existing compact prequantized routed pipeline. Keep existing C3 behavior unchanged. |

No EP wire-format change is needed. `model/impl_a2.rs` already broadcasts
dynamic N, sequence IDs and tokens; its worker reconstructs the head's exact
row order. The protocol's ability to carry N4 is not itself a kernel/state
correctness guarantee.

### Critical KDA distinction

The fallback in `glm5_kda.rs::decode_multi_seq` offsets `hidden`/`residual` but
calls scalar `decode` with the shared mHC buffer bases. It is not a valid C4
control or a safe implementation shortcut. Widen both the outer dispatch and
the inner true-batch guard together. Keep convolution and recurrent H updates
one independent state per row; never substitute the sequential-token verifier
recurrence merely because its token count is four.

The inner implementation's **FFN-only** scalar fallback is different: it
explicitly offsets `hc_streams`, `hc_post`, and `hc_comb`, and consumes each
row's MoE output before reuse. It can serve as the initial N4 FFN control once
the surrounding true batched KDA path is validated.

## Grouped N4 reuse: what exists and what must change

`MoeLayer::forward_prefill(input, 4, ctx, stream)` already accepts four
independent activation rows on this NVFP4 backend. It performs router/top-k,
expert sorting, grouped gate/up, SiLU/down, EP-aware unpermute/reduce and shared
blend. MoE has no sequence-recurrence semantics, so this is reusable for
independent decode. Prefer a small guarded `forward_c4` entry that validates
eligibility and delegates to this body, rather than pretending N4 is K3/K5.

Required details:

1. Factor common C3/C4 eligibility without broadening C3's existing flag.
   C4 must retain current exclusions: LoRA, mixed BF16/FP8 experts, non-native
   NVFP4 layout, missing T-layout/prequant kernels, non-sigmoid router,
   pre-expert norm, shared gate, missing EP-aware reduction, and grouped
   CUTLASS alternatives. Check exact-M4 handles, not batch3 handles.
2. `MoeLayer` already resolves `w4a16_batchm` in `moe/init.rs`. Reuse
   `.kernel(4)` with `ops::w4a16_gemv_batchm` for shared gate, up and down;
   use SiLU over `4 * shared_inter`. Destinations remain
   `ssm_deinterleaved`, `ssm_qkvz`, `attn_output`. No new shared CUDA kernel.
3. Extend compact-worklist selection in `forward_prefill_routed.rs` and the
   compact fused gate/up condition in `prequant_fp4.rs`. Generic grouped
   kernels already take N-derived offsets and device tile counts. N4 has32
   routed assignments; the worst-case grouped M tile bound is1, so the exact
   tile D2H path is disabled even in eager execution. Graph capture retains
   a fixed upper grid with device work counts.
4. Preserve router numerics explicitly. The current C3 branch in
   `forward_prefill.rs` uses `dense_gemm` to match `forward_k3`. Do not blindly
   extend that condition to C4. Implementation review found that the generic
   `router_gate_gemm_dense` helper also has a different reduction from the
   proven scalar decode GEMV. N4 therefore needs four row-offset launches of
   the same scalar router GEMV for its first grouped/control comparison.
   C3's router behavior must stay unchanged.
5. KDA and MLA call the same guarded N4 FFN entry and consume its contiguous
   `[4,H]` output through ordinary mHC post-mix. Do not add the K5 deferred
   shared blend or communication overlap to this first step. Both ranks must
   select the same grouped/control branch and collective sequence.

This reuse changes routed activation precision relative to scalar W4A16,
as the measured C3 path already does. Require numerical/quality validation;
do not describe the whole grouped path as bit-identical merely because its
router and independent-row ownership are preserved.

Implementation review also rejected generic `forward_batched` as the C4 EP
control: it blends the shared expert before its per-row EP reduction. The
proven scalar `forward` defers that blend until after reduction. Use the
proven scalar path in reverse row order3,2,1,0, copying each row0 result to
its final `[4,H]` output row before the next call. Audit that all input rows
and already-copied output rows remain live. This preserves four scalar
collectives in the control versus one grouped collective in the candidate;
do not quietly repair an unrelated generic path as part of this experiment.

## Explicit scratch and memory bounds

Check `BufferSizes` against actual row counts, not just a stage512 assumption.
For TP2 GLM H4096, local KDA P4096, MLA heads32, KDA dimension128:

| Simultaneous allocation / phase | N4 lower bound |
|---|---:|
| KDA Q/K/V plus beta/f-a/g-a in `qkv_output` | 98,304 +256 +1,024 +1,024 =100,608 bytes |
| KDA packed QKV and convolved BF16 rows, each | 98,304 bytes |
| KDA two gate planes in `ssm_deinterleaved` | 65,536 bytes |
| MLA Q latent in `ssm_ba` | 12,288 bytes |
| MLA expanded Q / extracted V, each | 65,536 bytes |
| MLA absorbed Q / attention output, each | 131,072 bytes |
| MLA assembled K+V entries | 8,192 bytes |
| Routed gate / up output, each (32x2048 BF16) | 131,072 bytes |
| Routed down output (32x4096 BF16) | 262,144 bytes |
| Input NVFP4 pack + scales, temporary in down arena | 9,216 bytes |
| Fused SiLU down pack + scales, staged in down arena then copied to dead up | 36,864 bytes |
| Compact gate/up worklist including counter prefix | 16+4x8x16x8 =4,112 bytes |
| Sort metadata in `gate_logits` | 3x32x4 +289x4 =1,540 bytes |
| Shared gate/up, each; shared down | 16,384; 32,768 bytes |

mHC needs `N * hc_mult * H * sizeof(float)` highway plus N-row post/comb
planes; derive hc_mult from the loaded configuration. Keep metadata at its
fixed floor32 layout. Router scores and sort metadata share their arena only
after top-k; index scratch is irrelevant in this short dense-only lane.

Per added active slot on each rank: live KDA state is74.375MiB, and the enabled
eight-state decode rollback ring adds595MiB, totaling **669.375MiB** before
other reserves. C3->C4 BF16 MLA KV adds44MiB for a full2048-token request
(22KiB/token/rank), plus semantic-index storage and metadata. Initial C4
therefore needs roughly0.697GiB extra per rank for these terms alone, not an
assumption that this much free memory suffices. Account using
`ssm_reserve::ssm_pool_reserve_bytes`, the real pool/dummy allocations and
server preflight; retain original host/GPU headroom. No rollback narrowing.

## Tests-first sequence and stop gates

1. Pure policy/dispatch tests: opt-in off preserves existing limits; N4 needs
   TP2/EP2/v2, KDA enabled, no speculation, no sparse flag, context<=2048.
   Boundary positions2047 pass and2048/usize::MAX fail before graph lookup.
   Prove N4 selects true-batch KDA; test missing handles and undersized arenas.
2. Pure layout/worklist tests: checked byte arithmetic above; concentrated
   expert routing, distinct experts, remote-only expert assignments and
   zero-local-work ranks must all fit and reduce correctly.
3. Small CUDA tests before serving: four distinguishable inputs and isolated
   KDA H/conv states, permute/unpermute equivalence, repeated updates, exact-M4
   projection versus row oracle. Grouped FFN compares router IDs/weights first,
   then outputs with explicit tolerances; poison remote outputs to prove the
   EP unpermute mask never consumes them. Preserve C3 regression fixtures.
4. Restart with cap1024, active/admitted4, eager and scalar FFN control.
   Run distinct short needles with unequal histories and output lengths that
   force actual C4->C3->C2->C1 drains. Then enable grouped N4 alone and repeat.
5. Include reordered/noncontiguous SSM slots: `[3,1,0,2]`, then `[3,0,2]`,
   `[3,2]`, `[2]`; reuse a freed slot for a different prompt and repeat both
   row orders. Confirm actual traces on both ranks, not just four HTTP clients.
6. Enable graphs only after eager passes. Replay across16-token KV boundaries,
   changing positions/block tables and slot reuse. Compare outputs and state
   against eager; test near the2048 cap without allowing an over-cap decode.
7. Retain matched C3/C4 full-wall and post-first-token aggregate throughput,
   per-stream rates, TTFT, context/output counts, memory and thermal receipts.
   Increase admitted cap to5 only as a separate bounded admission/drain test.

Any needle/state-isolation failure, unsupported path, collective stall,
unexpected allocation growth, or breached headroom stops promotion. Root
controls node operations and benchmark deadlines; do not reset a Spark as a
routine test recovery mechanism. Long-context C4 is a later change requiring
its own sparse-selector/graph width guards and memory validation.
