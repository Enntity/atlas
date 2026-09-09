# Checked scalar bootstrap sampling prerequisite

2026-09-09. Root and independent review approved; explicit engine source/Cargo
handoff received after bootstrap closure. Implement this bounded TDD slice only.
No serving call, admission or native authority. Read alongside the serial-driver plan.

## Actual gap and fixed scope

`scheduler/sample_step.rs::sample_token_with_grammar` already returns Result,
but its reduce-only fast path calls bool `fast_greedy::logit_is_positive`.
That wrapper turns a failed one-logit copy into false, so the scalar sampler
can perform a subsequent full-vocabulary copy and return success. The actual
argmax kernel/token readback and the full-vocabulary read already propagate
errors with `?`. Checked VERIFY selection is a different entry and does not
close this scalar bootstrap gap.

Add a grammarless checked sibling over one shared scalar sampler body. Keep
the existing legacy wrapper's grammar, fast-path, copy fallback, mask/penalty
ordering, suppression, tie behavior and sampling calculations unchanged.
The checked sibling has no serving caller in this slice. No Model trait,
capability, fixture numerical handler or scheduler transaction change.

The inherited scalar slow path reads BF16, and its final SamplingParams uses
`seed=None` and `top_n_sigma=0`. Preserve these literals and existing min_p
handling; do not silently implement seeded or FP32 sampling. The first paired
driver will require temperature0 before wire issuance and the actual selected
BF16 profile. This slice claims greedy parity and strict copy failure handling,
not stochastic seed parity or broader dtype support. The existing stochastic
branch remains the same shared body; no new selection policy is invented.

## Exact shared-body and policy partition

1. Move the existing function documentation/signature/body literally into
   `scheduler/sample_step_checked_scalar.rs`, registered from sample_step.rs.
   Keep `sample_step::sample_token_with_grammar` available at its existing path
   through a re-export. No existing caller changes.
2. In that child, use two thin entry wrappers and one private body:
   - Legacy entry supplies `CopyFailurePolicy::LegacyFallback`.
   - New `sample_token_with_grammar_checked` takes the same arguments, rejects
     an actual `Some(GrammarState)` before model/backend work, then supplies
     `CopyFailurePolicy::Propagate`.
   - One private body retains all numerical/sampling code. The only fast-path
     substitution is the existing lazy immunity proof through policy.immune,
     with `logit_is_positive_checked` as its Result-returning probe. Neutral
     penalties still short-circuit without a probe; history membership still
     short-circuits before any probe. Legacy probe Err still means false and
     then the same full read; checked probe Err returns immediately.
3. Reuse, rather than copy, the existing concrete policy from
   `verify_pipeline_helper/selection_io.rs`. Move that small enum/immune impl
   literally to `scheduler/fast_greedy_copy_policy.rs`, privately registered
   and scheduler-only re-exported by fast_greedy.rs. Keep selection_io.rs as
   a compatibility alias so existing verify children keep their paths. Both
   scalar and verify callers use the same policy and `argmax_immune` proof.
   Do not introduce a generic I/O router, sampler framework or new policy flags.
4. Add `scheduler/sample_step_checked_scalar_tests.rs`, registered only under
   the new child's cfg(test), with a bounded subprocess/helper child if needed.
   New files <=500 lines with SPDX. sample_step.rs is an existing allowlisted
   714-line parent; this literal extraction decreases it, without unrelated
   splitting to meet a new artificial scope.

Root-approved subprocess SSOT amendment: reuse the existing
`glm_c2_fixture_test_process.rs` from the scalar test child rather than copying
its environment. In the exclusive implementation window, change that helper
to accept a complete exact test path (remove its fixture-specific path format),
preserving its existing marker and environment literals. Adapt the existing
fixture-test wrapper/calls to supply `scheduler::glm_c2_fixture_tests::{name}`;
scalar tests supply `scheduler::sample_step::checked_scalar::tests::{name}`.
These two existing test files are the only added compiled-file scope. Confirm
each actual child runs exactly one test; exit0 from a zero-match filter is not
a successful gate. No helper/compiled-graph edits before the explicit handoff.

## Actual tests and behavioral RED

Use the committed actual dependency fixture and its real Model APIs. The test
itself performs actual prefill plus scalar bootstrap via Fixture::parts_mut,
then samples the real produced logits. No painted logits, new fake Model,
capability implementation or private Ready setter. The fixture has no Wire
installed because this prerequisite tests local sampling, not transport.
Keep the caller's actual SequenceStates and observer alive; compare tokens,
seq_len, private cursor and saved canonical/slab bytes before/after sampling.

First characterize the unchanged legacy entry, then add the checked signature
as a transparent legacy forwarder for a runnable behavioral RED. The real
reduce-only probe-copy failure must currently recover via a later full read
and return Ok, while the new negative requires Err and the exact trace ending
at that failed probe. Capture that runtime assertion failure before enabling
the policy. Actual grammar rejection has its own negative; a compile failure
or unrelated invalid-profile control is not counted as the probe RED.

Derive fault ordinals from a successful actual control and match the exact
fresh-fixture event at each injected ordinal. Exercise both local rank
configurations and owners with unequal prompt histories. Tests cover:

- Neutral greedy: actual argmax kernel plus its4-byte token read; no positive
  probe and no full logits copy. Both entry points return the same actual ID.
- Reduce-only, argmax absent from scoped history: actual positive2-byte BF16
  probe, then fast return. No disabled fast-path flag as the correctness fix.
- Argmax in scoped history: lazily skip the probe and run the actual full
 16-byte vocabulary8 read; checked and legacy output/event traces agree.
- Blocked penalty/bias and suppression controls: the same host mask/penalty/
  greedy path and token result. Reuse penalty_params_for(PositionKind::Verify)
  and penalty_history_scope where caller parameters are constructed.
- Argmax kernel failure and actual4-byte token-readback failure: both wrappers
  return the original error without later probe/full read.
- Positive-probe copy failure: legacy still performs the subsequent full read
  and succeeds; checked returns the injected error with no subsequent event.
- Full-vocabulary copy failure, reached by a valid history/blocked control:
  both return the original error, with no later sampling/backend action.
- Actual nonempty GrammarState passed to checked: refusal before argmax/copy,
  unchanged grammar history and owner bytes. Legacy grammar behavior remains
  separately characterized using existing grammar machinery; checked must not
  silently discard a grammar argument.

The untouched one-logit helper's existing zero/negative/NaN/FP32 tests remain
part of regression coverage; they are not a new FP32 scalar-sampler claim.
The actual fixture's byte argmax/numerical handlers are boundary sentinels, not
GLM arithmetic or statistical sampling oracles. No native failure is injected.

## Qualification and handoff

Capture baseline/RED/GREEN raw receipts separately; run focused scalar tests,
existing checked/legacy verify-selection and fast-greedy tests, fixture3 and
relevant server regression controls. Check ordinary non-test server compilation,
formatting, scoped headers/size and truthful clippy status. Do not fix or
suppress the inherited CUDA model lints as part of this prerequisite.

Freeze exact source and receipts for independent/root review. Root owns commit
and exact-tip full engine gates. No selected-driver call or temperature
admission branch is added here; the future driver must independently stop on
the propagated sampler error before emission, E1, peer work or cleanup, under
its separately reviewed terminal-session policy.
