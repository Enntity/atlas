# MTP emission: thinking EOS and hard-ceiling parity

Observed during v14 validation: explicit32-token thinking-budget arithmetic
returned a correct calculation only in `reasoning_content`, empty visible
content and finish=stop after24 tokens. This is a failed answer check, not
qualified deployment. The initial zero-budget smoke also failed; both raw
receipts are retained. Whether v13 produces the identical failure remains
to be tested; the following source discrepancy predates v14.

`emit_token` (native MTP acceptance/bootstrap) lacks the non-speculative
`process_decode_logits` thinking-only EOS suppression. Its own comment
explicitly acknowledges that only grammar-armed thinking requests suppress
EOS. Plain thinking can terminate before the closing-think boundary. It also
returns from the suppressed-EOS branch before the bottom hard-ceiling check.

Plan before implementation:

1. Regression tests invoke the real emitter with an in-thinking grammarless
   sequence. EOS must consume its ordinary budget/accounting but neither
   finish nor stream; a later closing-think token, visible answer and EOS
   must finish normally. Preserve normal content EOS and stray-close policy.
2. Use the existing shared `hard_ceiling_hit` and
   `eos_suppressed_by_thinking` helpers, matching non-spec semantics. At
   either output or context ceiling, an EOS must retire even while a
   suppression condition is armed; otherwise the early return bypasses the
   bound. Cover requested and spontaneous thinking, min-token suppression,
   both ceilings, and cooperative cancellation.
3. No sampler, accepted-count/SSM rollback, model kernel or EP protocol
   changes. Do not claim this fixes all verify-time policy differences.
4. Independent review, CPU server suite and native rebuild, then repeat the
   unchanged strict visible-answer cases and matched coding benchmark.
   Retain negative results; do not accept reasoning as a visible answer.
