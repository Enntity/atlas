# First-token thinking lifecycle correction

## Evidence and scope

The saved v11 reference log identifies token 154842 as `</think>` and shows it
as the first sampled token of the short C1/C2 requests. The chat stream consumes
that close correctly, while prefill constructs the scheduler state from the
request flag alone. Later content is therefore counted as reasoning and can be
subject to thinking-phase EOS suppression. This proves a lifecycle mismatch,
not whether the repeated text was generated or replayed by a parser.

Subsequent baseline capture in
`/tmp/atlas-glm53-phase5-20260907.gu145h/v11-reference-token-ids.sse` and
`v11-reference-token-ids-decoded.json` retained 124 sampled token IDs. Their
decoded text contains two complete code blocks separated by another
`</think>`, establishing generated repetition rather than full-buffer replay.
Usage reports 125 completion tokens and 62 reasoning tokens. The separate
stream-guard issue drops punctuation from the second block on the wire.

## Test-first implementation

1. Add a pure, shared `FirstTokenThinking` resolver in the scheduler. First
   reproduce the current behavior and observe a failing CPU assertion for
   requested thinking whose first token is the configured end marker.
2. Resolve `inside_thinking`, `think_ended`, and `think_just_ended` from the
   actual first token. A requested block closed by that token becomes
   `false / true / true`, matching a real close in the decode path. Ordinary
   first reasoning tokens, spontaneous starts, disabled thinking, and missing
   marker IDs retain their existing behavior.
3. Use the resolver in single-chunk prefill A/B and final-chunk promotion.
   Leave beam-result construction alone. Preserve first-token
   `thinking_tokens = 0`, token-budget accounting, spontaneous-start token
   omission and budget selection, grammar behavior, and immediate termination.
4. Run focused CPU tests for all resolver cases and constructor integration;
   check formatting and the scoped diff. No GPU actions in this work item.

## Boundaries and validation

No watchdog threshold, EOS policy, cancellation handling, stream sanitizer,
kernel, dtype, or rollout flag changes. Cancellation/terminal streaming is a
separate correction. CPU success is not a serving-quality or throughput gate:
the deployment owner must rebuild and rerun bounded raw SSE/blocking checks,
then the reference workload, retaining raw token/finish/usage evidence.

## CPU checkpoint

The original-state resolver produced the expected failing close assertion
(one pass, one fail). After correction, the focused `first_token_thinking`
server tests passed 3/3, including the real final-chunk promotion constructor
and its unchanged output-token, remaining-budget, thinking-token, and
spontaneous-budget assertions. The close test was then expanded to include
absent/equal start-marker IDs; its final rerun is part of the combined CPU gate.
Scoped formatting and `git diff --check` passed. No live deployment is claimed
by this CPU checkpoint.
