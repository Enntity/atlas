# Selected paired scalar bootstrap transport

2026-09-09. Code `3d74e19cc05db73e8e21b20bc66c5d2eb62490bf`.
This adds the bootstrap prerequisite for concurrent speculation, not a serving
caller or a throughput result. Both Sparks retain the stopped v26 deployment.

Exact post-commit CPU suites passed: **1108 model tests** (85.93s, four test
threads) and **2369 server tests**, 12 ignored (23.85s, one test thread).
No compiled-source changes occurred during those gates.

The sealed actual-model capability now validates bootstrap ownership and
storage before sending either scalar command word. Its immutable private plan
is reused at begin; validation does not reserve or write. The head owns the
existing slot/token transport, actual default-stream target decode and immediate
owned H[P] publication. Issued errors latch both owners; the selected worker
scalar helper applies the same local validation and latch. Ordinary nonpaired
scalar dispatch is unchanged. Earlier worker preamble errors and process-fatal
T2/supervision remain outside this slice.

Bootstrap proves one additional target row, while preserving the established
five-row arena envelope and the separate K5 budget. Actual P15/P16/P17 tests
exercise exhausted/free-block boundaries: P15 bootstrap can fit while its later
K5 cannot. Other tests cover repeated inert validation, genuine head/worker
replay in both owner orders, wrong rank/profile/owner/map, missing/retired owners,
and an unfinished peer target. Upload/transfer/target/bonus-copy/immediate-sync
faults are derived from completed controls; no later successful sync reopens
either owner. These CPU sentinels do not prove GLM numerics or NCCL progress.

New-entry nonsending scaffolds produced six intended runtime failures, followed
by six passes. Expanded coverage totals eight focused tests (0.63s); the entire
handoff family passed 128 tests (4.02s). A missing import and wrong subprocess
control environment are retained separately, not counted as product bugs.
Two old cleanup healthy controls decoded after installing deliberately
incompatible cleanup-only SSM geometry. Their producer now runs before that
installation; failure cases and production validation remain unchanged.

Non-test compilation and formatting/whitespace/SPDX checks passed. New test
children are 415/182 lines. Scoped no-CUDA clippy stops on four inherited runtime
stub lints; it is not a model lint pass or CI-green claim. Root and independent
reviewer approved all 13 frozen paths, manifest SHA-256
`078bda8847c8596ed06e266c9c3e242217c346a6982d81ed681256a687993f48`.
Raw commands and intermediate failures are retained under campaign
`20260908/glm-c2-bootstrap-transport`; the committed plan describes their scope.
Root also ran the CUDA server clippy path: it reaches nine inherited model
lints across eight unchanged files (exit101), with no new suppressions.
Source/receipt archive: `glm-c2-bootstrap-transport-3d74e19c-receipts.tar`,
with detached SHA-256; no targets, libraries or native binaries included.
