# Bounded native GLM K5 token ledger

Status: source frozen after CPU TDD; independent review/native gates pending.

Repeated matched EH-BF16 requests produce identical streamed text hashes but
different acceptance counts. A text hash does not establish identical token IDs
or hidden states. Locate the first host-visible divergence before adding device
readbacks or changing numerical kernels.

## Contract and scope

- Explicit `ATLAS_GLM_MTP_K5_LEDGER=1`, off otherwise; root owns forwarding.
- The existing concrete `Glm5MtpProposerState` identifies native GLM. DFlash,
  other models and widths other than four drafts/five targets are excluded.
- At most eight completed K5 records per request. A dedicated tiny counter
  belongs to existing `RequestAccept`, whose normal default construction resets
  it. Its existing MTP counter includes bootstrap, so cannot count actual K5
  verifications. Request snapshots preserve the diagnostic count.
- Capture pre-verify position, seed, four draft IDs and five raw target IDs;
  finish with five selected IDs and accepted prefix count before terminal token
  emission. Include slot and ordinal in the log; ordinal one starts each request
  window. No prompt/text strings are logged. Token IDs remain sensitive derived
  data: use trusted local diagnostic logs, not public receipts.
- This scheduler ledger is rank-zero only. It reports the already selected
  distributed draft tokens, not separate per-rank private hidden states.
- Use fixed arrays only after enabling and checking the limit. The raw target
  vector follows its existing ownership path; never clone it on the normal path.
  The environment flag is read once; the disabled per-step path has no new
  allocations or token-array copies.
  No new GPU operation, collective, synchronization, forced proposal, sampling
  policy, request completion behavior or persistent device memory.
- Invalid shape, out-of-vocabulary IDs, overflowing position, or an inconsistent
  accepted prefix suppress only the diagnostic; they do not modify inference.
- Capture raw IDs before selection may move their vector. Record the actual
  verdict before emission so short requests and terminal verifications appear.
  Failed forwards do not generate a completed record.

## Implementation and tests first

New pure `verify_dflash_ledger.rs` and sibling tests, nested under the existing
K5 module; one defaulted counter field in `RequestAccept`. The public K5 entry
resolves eligibility and delegates to the same internal implementation. This
permits existing host-only Target tests to enable diagnostics without mutating
process environment or constructing GPU-owned GLM state. The public concrete
state check remains separately inspectable/tested for non-native rejection.

1. Observe meaningful RED for a valid prepared/completed record on a no-op stub.
2. Test off/invalid flags, unsupported widths, raw/selected length, token bounds,
   position overflow, accepted count/prefix mismatch, all acceptance counts0..4,
   limit eight and new-request reset. Rejected records must not consume budget.
3. Exercise the actual K5 inner runtime with the existing host Target: normal
   verification, terminal remaining-one, verdict error, and disabled path.
   Preserve its exact GPU-free call sequence and next-proposal behavior.
4. Focused server tests, formatting/diff check, independent review, then freeze.

Root alone builds/deploys. A bounded diagnostic request should yield at most
eight `GLM MTP K5_LEDGER` records, including a short terminal verify. Compare
position/seed before comparing draft arrays; differing accepted prefixes change
subsequent verification positions. These diagnostic runs are not throughput
receipts because host logging can perturb scheduling.

## CPU receipts and file-size maintenance

Actual server TDD RED: four new tests discovered, two passed and two failed
on the no-op stub (valid-record construction and eight-record window).
Receipt: `/tmp/atlas-glm53-phase6-20260907.J5PkkO/k5-ledger-red.log`.
The existing `mtp_accept_debug.rs` was already 511 lines; adding its counter
required moving its unchanged test module to `mtp_accept_debug_tests.rs`.
The moved content, including comments, matches HEAD modulo indentation.
No acceptance-accounting logic changes are included in that split.

GREEN: all seven ledger tests and all263 scheduler tests passed. Receipts in
the same directory: `k5-ledger-green.log` and `k5-ledger-scheduler-green.log`.
The actual runtime fixture tests all acceptance counts0..4 with normal progress,
terminal remaining-one, and injected post-verification record failure, comparing
diagnostic ON/OFF call order, emitted tokens, sequence state, and pending drafts.
One terminal fixture captures the real log event and checks every token array.
Native concrete-state admission is source-inspected; CPU fixtures inject the
resolved eligibility into the same runtime body, not GPU-owned GLM state.
All six touched Rust files are under500 lines; formatting/diff checks pass.
# Follow-up precision isolation

After the v17 native first-eight trace, compare an explicit existing
`GLM_MTP_BF16_DRAFTS=4` vocabulary arm against the prior value1, retaining
BF16 EH, NVFP4 WO, accepted-pair repair, routed M16 and every other launch
setting. This changes draft-head precision only, not the target, sampler or
output cap. First run the same148/64 diagnostic with five repetitions to
compare proposals; then strict answers and matching148/256 warmup+three waves.
Trace rates are diagnostic only. Any throughput candidate needs trace-OFF
matching runs and confirmation. Different draft IDs under mixed precision
do not prove a cache bug or a particular kernel race.
