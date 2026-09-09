# Explicit logical position for the existing token emitter

2026-09-09. Approved B2 serial-driver prerequisite, root-owned. No live paired
scheduler or factory/admission caller is added. Existing ordinary decoding must
retain exactly its current emission, cancellation, penalty and budget behavior.

The selected driver will commit a complete accepted prefix before emitting any
token. The canonical target seq_len therefore already describes the final row.
The two context-ceiling operands in emit_step currently read that final seq_len
for every token, prematurely stopping an earlier accepted row at the ceiling.
Do not rewind canonical SequenceState while streaming. Add a private explicit
logical-position entry over the existing body; ordinary emit_token passes its
unchanged a.seq.seq_len. Replace only the two ceiling operands, retaining the
shared seqlen_force_stop/hard_ceiling_hit arithmetic and all ordinary side effects.
Callers remain responsible for validating bounded logical positions; this is
not a new request authority, budget ledger or separate emission algorithm.

TDD: an initial new-entry scaffold delegates to the existing emitter. Tests
require all earlier logical rows to emit while canonical state stays fixed,
and require suppressed EOS to respect the logical rather than final position.
Those behavior failures demonstrate the old emitter is unsuitable for the new
commit-before-emission order, not a claim that its ordinary callers regress.
A same-position legacy-wrapper parity test is a positive control. Test actual
streamed tokens and generation-budget accounting, output and context ceilings.
Thinking consumes generation budget in the existing body and remains unchanged.

Files: emit_step.rs (shared body/wrapper and test registration), one bounded
glm_c2_emit_position_tests.rs child, and this plan. The existing emitter parent
is explicitly allowlisted by file-size-cap.yml; new tests remain <=500 lines.
Record actual RED before replacing the scaffold, then focused emitter/cancel
tests, full server/model postcommit CPU gates, non-test server check, scoped
fmt/SPDX and honest inherited lint status. Independent review must confirm only
the intended two body operands changed and ordinary wrapper behavior is retained.
No model/GPU run or speed claim; selected driver integration remains later work.
