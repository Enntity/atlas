# GLM paired pre-dispatch and checked-selection CPU checkpoint

2026-09-08. This is host-side development qualification, not a native serving
qualification or throughput improvement. Both Sparks remain stopped; the
verified v26 image and binary are unchanged. No public factory or scheduler
caller selects the new paired capability.

## Committed changes

- `335c9956`: shared immutable actual-model/proposer validation before selected
  K5 and owned proposal work. Checks reuse actual owner, history, SSM, buffer,
  private KV plans and target allocator capacity. The optional sealed execution
  capability is exposed, but its transport methods still explicitly refuse.
- `676403680ffe983024e4a48cd923a28c63d2756f`: checked grammarless verification
  selection propagates actual small-probe and full-logits-copy errors. The
  legacy wrapper retains its old recovery behavior; both share the same driver
  and unchanged per-position penalty/processor/sampling implementation.

Behavioral RED found two existing selected-model defects: a target-cache
exhaustion discovered after K5 claimed shared state poisoned a healthy owner,
and a Pending reproposal could proceed without capacity for its following K5.
Both now fail before the claim/write, and restoring capacity permits the
original real operation. Preflight is not a reservation; serialized execution
must not interpose another allocator user before dispatch. Target maps retain
the inherited allocator ownership contract, not an independent provenance seal.

Checked selection RED independently showed that failed copies could become
successful raw or recovered picks, and that the new entrypoint initially failed
to reject grammar on an empty span. The checked entrypoint now returns errors;
legacy behavior is separately characterized and preserved. No checked grammar
support or changed sampling math is claimed.

## Evidence and limitations

Controller receipts live under
`/home/abc/storage/models/atlas-campaigns/20260908/`:

| Gate | Result | Receipt subdirectory/file |
| --- | --- | --- |
| Model checkpoint RED | 6 executed failures; four new API stubs, two real capacity defects | `glm-c2-predispatch/checkpoint1-red.log` |
| Model focused GREEN | 75 passed, 2.62 s | `glm-c2-predispatch/checkpoint1-green-attempt.log` |
| Model non-test check | passed, 5.49 s | `glm-c2-predispatch/checkpoint1-lib-check.log` |
| Selection legacy characterization | 5 passed | `glm-c2-checked-selection/characterization-runtime.log` |
| Selection behavioral RED | 8 passed, 3 intended failures | `glm-c2-checked-selection/behavioral-red.log` |
| Selection focused GREEN | 12 passed | `glm-c2-checked-selection/focused-green.log` |
| Server non-test check | passed, 17.42 s | `glm-c2-checked-selection/final-nontest-check.log` |
| Exact-tip model suite | 1,051 passed, 81.36 s | `glm-c2-predispatch/postcommit-model-cpu.log` |
| Exact-tip server suite | 2,359 passed, 12 ignored, 23.17 s | `glm-c2-checked-selection/postcommit-server-cpu.log` |

Both exact-tip suites ran at `67640368`, with no uncommitted compiled source.
Only next-stage Markdown audit/planning files were uncommitted. These are not
GPU tests: model fixtures execute real host ownership/dispatch against recorded
numerical boundaries, and server fixtures execute the actual sampler against
owned host bytes. Ignored tests were not run.

The server suite uses `NO_COLOR=` and `RUST_TEST_THREADS=1`. The initial parallel
run under inherited `NO_COLOR=1` had four unrelated TUI failures (three color
assertions and one shared-log renderer race); its raw failure is retained in
`full-server.log`. No TUI source or test was altered. Strict Clippy is **not
passing**: dependency analysis stopped on existing model style errors; scoped
server analysis stopped on the unchanged `serve_phases/preflight.rs`
items-after-test-module lint. No lint levels were lowered.

Controller CUDA linking uses the existing installed cudart/cuBLASLt libraries
and an untouched official NVIDIA driver development stub, extracted without
installation. Its package/library hashes and URLs are in
`cuda-driver-link.hFicnP/PROVENANCE.md`; `cuInit` returns error 34. This permits
CPU tests to link, not to execute CUDA. Never deploy these test dependencies.

Independent review verified both frozen manifests and actual RED/GREEN logs:
model 18 paths, manifest SHA
`123d43bf4c3cb07e75afd575fbe07d994f8203179703727772c3bf4f5168ae4e`;
selection 10 paths, manifest SHA
`2e87e9436bef54bc1c94027886453739b4fcb095147b3fc1566f4272530bb29c`.
The implementation plans are committed alongside each slice. The next owned
E1/F5 transport checkpoint, selected scheduler integration and exact-peer fatal
supervision remain separate work. The 30 C1 / 60 C4 performance target has not
been reached by this checkpoint.
