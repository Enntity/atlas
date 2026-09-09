# Selected paired scheduler retirement

Root authorized this bounded implementation on 2026-09-09. No serving activation,
factory change, T2 authority, supervisor, native action or ordinary policy change.
Implemented and focused CPU-gated; independent final source review requested.

## Exact boundary

`mod_helpers::retire_selected_finished_sequences(model, &mut Vec<ActiveSeq>, limit)
-> Result<()>` requires the actual paired capability and EP-v2, validates unique
slots0/1 over the whole slice, and handles finished owners in slot order.
The owner remains borrowed in `active` through actual `Model::free_sequence`
and `ep_broadcast_cmd_for_seq(original_slot, F1)`. On either Err return immediately:
no later owner, completion, cache insertion, removal or retry. F1 is never sent
after failed free. Successful local free can already retire/release device state;
retaining the host owner is not a claim of transactional rollback on F1 failure.

Only after both calls succeed invoke a shared host-only completion body and remove
that owner without changing the survivor's slot identity. Reuse actual finish
reason/response/log accounting. Existing `finish_sequence` retains literal old
host completion -> cache -> best-effort free -> best-effort F1 ordering.

The new helper has no serving caller. A future T2 caller must remain armed around
the entire helper, inspect Result before any normal cleanup, and send Err directly
to the no-unwind fatal sink. Returning Err alone neither exits nor prevents an
unreviewed caller dropping `active`; no such protection is claimed here. F1 send
completion is not a new worker ACK: worker serial command order and existing Model
retirement govern reuse. Existing worker F1 performs actual selected free+allocate.

## Owned files and tests

- `scheduler/lifecycle.rs`: mechanical extraction of existing host completion.
- `scheduler/mod_helpers.rs`: new private uncalled checked helper and test module.
- `scheduler/glm_c2_retirement_tests.rs`: existing real paired dependency Fixture,
  pointer-free Observer/Wire, ActiveSeq initializer and actual serial driver.

New-entry non-sending stub supplies genuine behavioral RED on valid retirement.
Then both finished-slot choices/vector orders; real worker F1 replay; slot1 survivor
continues actual serial transactions. Control-derived local completion and both F1
word transfer faults prove owner retention, no completion/removal/later work and
peer bytes/cursor preservation. Legacy finish/retirement tests remain regressions;
no new model mock/capability or fabricated Ready. Local fixture disposal after
assertions is not serving recovery. GPU bytes are sentinel evidence, not numerics.

Focused server CPU tests/check/fmt/scoped lint only, existing offline wrapper;
root owns broader qualification and commits. Preserve intermediate failures and
exact final hashes. No repeated all25 matrix or native fault injection.

## CPU checkpoint

Actual new-entry RED: both tests reach the non-sending stub after successful
real two-owner bootstrap. Final 3 tests pass, including both retired-slot choices,
both vector orders, six real injected free/F1 failures, malformed whole-slice slot
metadata, deterministic two-finished F1 order, and two healthy survivor rounds.
Legacy lifecycle13, cancellation6 and existing serial-driver5 tests pass.
Non-test check passes18.24s. Clippy stops in unchanged spark-model source; no lint
lowering or full clippy pass claim. Initial cancellation filter matched zero;
correct `emit_step::cancellation_tests` receipt supplies the6 actual tests.
No runtime no-cache spy or live T2 proof: absence of cache call and literal legacy
body/order are source-reviewed. Failure bytes remain observable, but a revoked
private cursor is not treated as authority to resume. Root owns native/full gates.
