# Paired GLM eager-bootstrap error boundary

2026-09-08. Code `b9dbbef7a25ceb9837ce60c9bef44a4aec72adf1` makes the actual
paired scalar bootstrap eager and returns body errors without invoking graph
cleanup. Legacy model graph selection and error cleanup are unchanged. This
is unactivated controller CPU qualification, not a throughput improvement.

The defect was observable even with graphs disabled: the ordinary abort helper
calls `cuStreamEndCapture`, and may destroy a partial graph, before returning
the original body error. An outer selected terminal handler cannot stop those
inner calls. The paired branch now avoids capture and that abort path entirely;
its existing outer Model wrapper still restores and quarantines proposer state.

Receipts live in
`/home/abc/storage/models/atlas-campaigns/20260908/glm-c2-eager-bootstrap/`:

- `behavioral-red-attempt.log`: one legacy control passed, two selected tests
  failed at runtime. The failures observed target-error followed by abort, and
  actual begin/end capture under inherited graph flags.
- `focused-green.log`: all three new tests passed. Nine isolated subprocess
  profiles cover both ranks/owners, all four EP/GDN graph-flag combinations and
  real legacy construction. Peer private KV/slab/token state is preserved.
- `final-handoff-green.log`: all 96 paired-model regression tests passed, 3.64 s.
- `final-nontest-check.log`: library check passed, 4.42 s.
- `postcommit-model-cpu.log`: all 1,072 model tests passed, 81.17 s, at the exact
  code commit. Unreferenced next-stage ownership-test drafts were present but
  were not registered or compiled; all compiled source was committed/frozen.
- `postcommit-server-cpu.log`: 2,363 passed, 12 ignored, 23.37 s, at the same
  exact code commit with `NO_COLOR=` and `RUST_TEST_THREADS=1`.

Independent review and root verified all five frozen paths, manifest SHA
`b135f41a29cf5195bdaaf3b95f9c60398ce9b417846dd3d9b2407bf2627e864b`.
Formatting, scoped diff, SPDX and file-size checks pass; all four Rust files
are below 500 lines. No whole-workspace lint/native qualification is claimed.
The graph fixture records calls, not CUDA capture/replay arithmetic.

Allocation/retirement, selected scheduler/worker integration, admission and
exact-peer supervision remain separate work. Prefix-cache OFF is not sufficient
to disable the independently allocated SSM snapshot region; the initial selected
profile must exclude that optional error-swallowing path. Model-resource
teardown and optional cuBLAS descriptor handling also retain their documented
limits. No image build, model load, node reset, native fault injection or push
occurred. Both Sparks remain stopped on v26; no new C1/C2/C3/C4 rate is reported.
