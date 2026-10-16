# F4 — DFlash2 speculative decoding for GLM-5.3 EXL3, TP=2/EP=1 (plan, 2026-10-15)

Scoping result from a read-only planning pass; nothing implemented yet.
Status of serial decode on `perf/decode-vllm-parity-20261015`: ~0.0535 s/token
(~18.7 tok/s, graphs + all perf flags, bit/output-identical to the
pre-optimization Atlas).

## Reference (what "25–35 tok/s" is)

- Drafter: `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c` (licence CC BY-NC-ND 4.0:
  local use only, not shippable in a published image). Qwen3-shaped, 1.171B
  BF16 (2.34 GB), 5 non-causal sliding-window layers, no embed/lm_head (shares
  the target's). `block_size` 8, `mask_token_id` 154856,
  `target_layer_ids` [5,14,24,33,42], `selector_top_k` 16.
- Block-diffusion draft: one forward per step over (bonus + k mask) rows; context
  K/V from an fc over the 5 concatenated aux hiddens (mHC post streams,
  averaged); DFlash2 grouped convs + low-rank selector walk over lm_head top-16.
- Measured on this ring (mindreader `glm-tp2-20260925/summary.json`): k=7,
  mean acceptance length 3.75–4.09, C1 step median 131.7 ms → ~30 tok/s. The
  kit README's 59 tok/s was never reproduced locally.
- Rollback: KDA keeps k+1 recurrent-state slots per request (accepted slot read
  next step), conv state widened by num_spec; MLA/indexer caches are
  position-addressed and overwritten (indexer kpool tail).

## Atlas reuse / gaps

Reusable: `layers/dflash_head/dflash2.rs`, `weight_loader/dflash_loader.rs`,
`precompute_ctx_kv.rs`, `model/trait_impl/verify_d.rs`,
`scheduler/verify_dflash_step.rs`, rank-1 command `0xFFFFFFF5`
(`impl_a2.rs`), KDA verify snapshots (`glm5_kda.rs` decode_batched,
`forward_recurrent.rs` recurrent_verify_snap_k), `rollback_ssm_states`,
mHC capture (`impl_b3_dflash.rs`, `decode_a3.rs`, `prefill_b/forward_layers.rs`).

Gaps: EXL3 speculation guards (`serve_phases/weights.rs`, `factory/build.rs`,
`preflight.rs`); the existing `ATLAS_GLM_DFLASH` lane admits only
TP2/EP2/BF16 KV (`mla_glm_verify.rs`); the drafter head takes a BF16/NVFP4
shared lm_head, not the packed vocab-sharded EXL3 head; verify MoE issues one
M=1 slot per (row, expert) without dedup.

## Phases

1. Drafter load + parity (~1 wk): guards behind `ATLAS_GLM_EXL3_DFLASH=1`,
   replicate drafter on both ranks, BF16 head copy, aux-hidden capture at the 5
   layers; exit: cosine ≥ 0.999 vs reference dumps, draft tokens match.
2. Verify with r = k+1 rows (2–3 wk): EXL3 verify admission (KDA batched
   snapshots, MLA EXL3 chunk path with FP8 KV + kpool, MoE coop thin up to
   r·8 ≤ 32 slots, EXL3 head on r rows); exit: verify-row vs serial argmax 100%.
3. Rollback (1–2 wk): KDA recurrent/conv intermediates, MLA via seq_len, indexer
   kpool finalize audit; exit: forced accept counts j ∈ 0..k → 512 tokens
   byte-identical to serial.
4. Acceptance + end-to-end (~1 wk): k sweep 1..7, ≥3 repeats; exit: accepted
   tokens per wall second > serial.
5. Perf (2–3 wk): verify graphs keyed by r, drafter graph, expert-grouped MoE.

## Speedup model (modelled)

tok/s = τ / (T_verify(r) + T_draft). With serial 56 ms (MoE 15), non-MoE ×1.1
for r ≤ 4, MoE ∝ unique experts, T_draft ≈ 10 ms replicated: k=3 → 24–36
tok/s; k=7 → 23–39 tok/s. Expect ~28–30 tok/s first; MoE efficiency is the main
lever (≈52% of bandwidth ideal today).

## First step

Guards behind `ATLAS_GLM_EXL3_DFLASH=1`; load Dflash2 weights against the EXL3
target with capture + BF16 head copy; run drafts without verify, scored against
serial decode → per-position acceptance curve on a fixed prompt set. This
measures τ and the capture contract before kernel work.
