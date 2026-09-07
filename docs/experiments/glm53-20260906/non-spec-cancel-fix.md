# Non-speculative cancellation parity and terminal streaming

Status: source frozen after CPU TDD; root-owned live validation pending.

## Confirmed cause and scope

`scheduler/emit_step.rs::emit_token` observes the stream's shared cancellation
flag before committing a sampled token. The independent non-speculative
`decode_logits_step.rs::process_decode_logits` commit loop does not. A stream
watchdog can therefore request cancellation while non-speculative generation
continues until a different stop condition. Separately, `handle_token.rs`
sanitizes and emits subsequent token text even after `stop_string_triggered`
became terminal. Sanitization removes markup, not duplicate ordinary content.

This change concerns host request lifecycle only. No kernels, wire protocol,
watchdog thresholds, scheduler admission, prefill thinking policy or GPU
allocation changes. Existing large source files retain their unrelated content;
new helper/test files stay below500 lines.

## Minimal implementation

1. Extract the existing emit cancellation check into one shared helper: Acquire
   load of the optional flag, set only `ActiveSeq.finished`, and report whether
   to skip committing. Keep emit's existing earliest cancellation precedence.
2. Invoke the same helper before either mutable host-sampling arm and again for
   each non-speculative result before modifying
   `last_token`, timing, grammar, output history or generation budget. Preserve
   row order and process unaffected rows normally; never compact the batch while
   its logits still use the original row mapping.
3. Refuse subsequent stream token handling once the stream terminal flag is
   already set, before detokenization or accumulator mutation. A genuine stop
   string first matched by the current token may still emit its sanitized
   pre-stop prefix exactly once. Existing explicit stop-string holdback and
   sanitizer behavior are unchanged for that triggering token.

The flag is cooperative: an already-issued forward/host sampling may finish,
and cancellation racing after a row's check may leave one in-flight event.
The stream terminal gate discards that event; the next scheduler commit check
retires the row. No claim of GPU interruption or atomic cancellation across
threads is made. Normal lifecycle retirement remains responsible for cache,
SSM, EP and response cleanup; this helper neither rewinds nor frees state.
The existing finish-reason ladder and stream guard overrides remain authority:
plain cancellation/stop strings report stop, named quality guards report length,
and exhausted token/context budgets retain their existing behavior.

## Tests before implementation

- Pre-set cancellation prevents output/history/budget/last-token mutation and
  marks the real ActiveSeq finished; missing/false flags preserve delivery.
- Cancellation observed between independent rows skips only the affected row,
  preserving row mapping; repeated observation is idempotent.
- Both emit and non-speculative commit routes use the same check before their
  first token commit; exercise the real production gate, not a copied predicate.
- A terminal stream never calls its token-processing body, emits content or
  reasoning, or changes accumulators; normal open streams still call it.
- Explicit stop matching preserves the pre-stop prefix and sanitization, while
  later tokens are suppressed. Existing stop holdback and finish-reason tests
  remain green, including context and token-budget caps.

CPU-only server tests use the shared supplied CUDA stub/runtime libraries.
Root runs live cancellation/normal generation/explicit-stop checks only after
source freeze and image rebuild. No performance claim follows from CPU tests.

## CPU evidence and limitations

Before the fix, the real non-speculative post-logits entry point failed all
three new cancellation cases: pre-set cancellation, cancellation after argmax,
and cancellation of row1 during row0's checkpoint callback. Existing emit parity
passed. The stream entry-gate seam failed both terminal tests while normal
delivery passed. After the fix, all6 cancellation tests and all3 entry-gate tests
pass, including serial/parallel host adaptive-state preservation, midbatch row
mapping, repeated cancellation, explicit-stop prefix preservation and unchanged
hard token/context limits. The complete scheduler unit slice passes250/250
(including the separately owned first-token-thinking tests), and the chat-stream
slice passes50/50. Rustfmt and diff checks are clean.

The stream tests execute the production gate with real StreamState, without an
AppState/tokenizer fixture; existing stop-string holdback and stream finish tests
cover those neighboring policies. In-flight tokens suppressed after cancellation
must not enter token-ID accumulation. Returned token IDs need not equal scheduler
usage counts in this race; usage still comes from the normal scheduler lifecycle.
