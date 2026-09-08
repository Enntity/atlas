# GLM paired transport and local terminal core: CPU checkpoint

2026-09-08. No native deployment or new throughput measurement. Both Sparks
remain stopped, with the verified v26 binary/image unchanged. The performance
target of 30 C1 or 60 C4 aggregate tok/s remains unmet.

## Code changes

- `0d5a6977`: the actual sealed paired Model capability now sends selected E1
  proposal and F5 verification commands after immutable local preflight.
  Selected E1 carries eight words, including private generation/attempt and
  position; the worker validates its own owner before proposal work. Legacy C1
  retains its four-word/global-hidden protocol. F5 retains the existing command
  body, with selected checks and an error latch covering both private owners.
- `c4a08bad4e9505b79c92d980af913d9717506b4c`: server-private, one-shot terminal
  core, immediate `_exit(74)` sink and pre-unwind panic interception, including
  the late TUI hook. The production core is inert: no live Model registration,
  launch ticket, scheduler caller or supervisor activates it.

An issued-command error is not a retryable request error. Model tests establish
irreversible metadata refusal, not process or native GPU containment. The head
capability's F5 return precedes checked selection, accepted-count exchange and
state completion; the future scheduler must guard that entire transaction.

## Development evidence

Receipts are under `/home/abc/storage/models/atlas-campaigns/20260908/`.

| Gate | Result | Receipt |
| --- | --- | --- |
| Transport behavioral RED | 3 intended runtime failures | `glm-c2-predispatch/checkpoint2-red.log` |
| Transport focused final | 93 passed, 3.65 s | `glm-c2-predispatch/checkpoint2-focused-final.log` |
| Model non-test check | passed, 5.15 s | `glm-c2-predispatch/checkpoint2-lib-check.log` |
| Terminal behavioral RED | 2 passed, 2 failed: C atexit and nested unwind ran | `glm-terminal-session/behavioral-red.log` |
| Terminal focused final | 4 passed; eleven real subprocess modes | `glm-terminal-session/focused-thread-green.log` |
| Server development suite | 2,363 passed, 12 ignored, 23.59 s | `glm-terminal-session/final-full-suite.log` |
| Server non-test check | passed, 11.86 s | `glm-terminal-session/final-nontest-check.log` |
| Exact-tip model suite | 1,069 passed, 81.80 s | `glm-c2-predispatch/checkpoint2-postcommit-model-cpu.log` |
| Exact-tip server suite | 2,363 passed, 12 ignored, 23.81 s | `glm-terminal-session/postcommit-server-cpu.log` |

Root ran both exact-tip suites at `c4a08bad`, with no uncommitted compiled
source. Only this results note and the next inner-owner audit were untracked.
Both commands exited zero. Server tests use `NO_COLOR=` and
`RUST_TEST_THREADS=1`; ignored tests were not run.

The first wider transport run was 89 passed / 1 failed: an old failure test used
a live peer getter after the new both-owner latch correctly invalidated it.
The test now compares pointers captured before failure; its byte-preservation
assertions remain. The unsuccessful receipt is retained.

Actual head command bytes are replayed through the actual worker adapter for
all 25 acceptance pairs, both owner orders and repeated unequal histories.
Tests also cover real retirement/reuse, stale generation, malformed identities,
target block exhaustion before wire traffic, and command/read/write failures.
These are host-owned numerical-boundary fixtures, not CUDA arithmetic or NCCL
agreement tests. Legacy F5 controls prove transport and state-update ordering,
not normalized numerical equivalence of their preparatory hidden row.

Terminal subprocess controls prove that selected errors/panics skip caller/error
destructors, C atexit handlers, prior hooks and caught-panic continuation. Ordinary
and explicitly closed controls retain normal cleanup. They cannot prevent
cleanup that a callee already performed before returning Err to its caller.

## Review and scope limits

Independent review verified the 22-path transport manifest, SHA
`fc1e61f5eab6d02d5e636b1f928c0b8f47d27e9756d2d8b4d9786aff7be2a450`,
and the seven-path terminal manifest, SHA
`2c29149a37825c34f2ab1a750180e13019f19e6297c088f6a4d92afaf532f7ee`.
Root separately reviewed the independent author's real legacy fixture/tests.
New Rust files meet SPDX/500-line checks; touched `impl_a2.rs` is an existing
oversized workflow-allowlisted file. Formatting and scoped diff checks pass.

Scoped server Clippy fails at the unchanged
`main_modules/serve_phases/preflight.rs:460` items-after-test-module lint.
No lint level was relaxed; this is not whole-workspace CI-green. CPU linking
uses the previously documented official NVIDIA development stub and installed
libraries, never deployed to the Sparks. No native failure injection occurred.

Read-only node checks at approximately 23:35 UTC found no running Docker
containers, 121,609,268 / 121,707,276 kB MemAvailable (head / worker), and
SwapTotal equal to SwapFree at 10,485,756 kB on both nodes. No restart, reset,
model load, image build or runtime replacement was performed.

The next selected-only work is allocation/retirement cleanup, actual worker and
scheduler integration, resolved admission and exact-session peer supervision.
See `scripts/dev/glm_c2_result_owner_audit.md` for concrete inner cleanup gaps.
The existing vLLM source comparison identifies device-resident draft feedback
as a later optimization; neither this serial transport nor the unused M10
kernel experiments establish batched-target speedup.
