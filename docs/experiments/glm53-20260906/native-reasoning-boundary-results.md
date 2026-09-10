# GLM native reasoning boundaries: evidence and changes

## Source and qualification status

Committed source: `14b4e485a8719d763b1b043e1c7a8844994ae4c2`.
Native source archive SHA-256:
`2506dce744e767569683079c837e6a02a88240992db648073e99c94c472e4fa8`.
The native build and both images are hash-verified; exact pins are in the
[RC handoff](../../releases/glm53-dual-spark-20260910-rc.md). The earlier
`1cb0267e` artifact pins do not identify this new code. Its fresh arithmetic
diagnostic still fails unchanged; no GPU quality or performance PASS is claimed here.

Retained logs below are relative to the operator evidence directory
`/home/abc/storage/models/atlas-campaigns/20260909/glm-native-controller`
(`BASE`); these external receipts are not bundled repository artifacts.

## Native observation and reference contract

The failed `1cb0267e` 4K ON run provides a concrete native tool-boundary example.
Root verified `longctx-1cb0267e-4096-on-chat-run/00297-rank0-collect.json`:
startup at 00:04:51 identifies `<tool_call>` as token **154843**; the final
3483-token prompt for session `0xa1fef7ecfa2712cf` is logged at
00:06:17.433009, followed at 00:06:22.245452 by prefill first token **154843**.
The corresponding `quality-receipts/quality/http-0030.json` records 99 output
tokens, normal stop and approximately 4811.9 ms TTFT, but the tool envelope
remained in reasoning rather than becoming the required structured call.
This connects the actual native opener to the failed API result; it is not
a synthetic-token or tokenizer-text inference.

The official vLLM parser at pinned revision
`6c379b9e5439ae305913e4a87ebf2b2e816072b4` treats a tool opener during reasoning
as both a reasoning end and a tool-call start. Its natural thinking-end
transition has no preceding-character condition, and declared tool names are
validated. This is parser-contract evidence, not evidence about native logits
or this model's output distribution.
[Native tool transition](https://github.com/vllm-project/vllm/blob/6c379b9e5439ae305913e4a87ebf2b2e816072b4/vllm/parser/glm47_moe.py#L138),
[natural end transition](https://github.com/vllm-project/vllm/blob/6c379b9e5439ae305913e4a87ebf2b2e816072b4/vllm/parser/glm47_moe.py#L103).

## Actual CPU regressions and corrections

| Boundary | Retained RED → GREEN | What the tests establish |
|---|---|---|
| Native tool opener | `glm-tool-boundary-red2.log` (1 pass, 5 fail) → `glm-tool-boundary-green2.log` (7 pass) | Actual blocking/streaming API parsing, ordinary decode, first-token construction and MTP emission preserve reasoning then a validated tool call. Controls cover explicit/implicit ends, first-token and malformed envelopes, request/model restrictions and tool names. |
| Natural thinking end | `glm-natural-end-red.log` (1 pass, 2 fail) → `glm-natural-end-green.log` (3 pass) | The actual `MODEL.toml` parser and logits/emission pipeline preserve a supplied natural end-token winner; generic-model masks/floors and explicit thinking-budget enforcement remain tested. These supplied logits are not a native-model numerical oracle. |
| Selected phase transition | `glm-selected-phase-red2.log` → `glm-selected-phase-green.log` | Real issued owners reached E7 and published accepted count **4 instead of 0**. The unchanged corrected test passes across E7, E6 and singleton F5, including actual worker replay, canonical target suffix, next E1 seed, emitted token and remaining budget. The backend is byte-backed, not a CUDA numerical oracle. |

The initial `glm-selected-phase-red.log` is retained but **is not the intended
regression proof**: its caller-bias setup failed the nonboundary acceptance
control. The corrected test uses supported history/presence conditioning
without changing canonical sequence state or forging issued receipts; the
nonboundary control genuinely accepts before the boundary assertions run.

Final committed-source receipt `glm-native-reasoning-14b4e485.log` reports
**44 passed, 0 failed in 10.92 s**. Formatting and kernel-shadow checks also
passed according to the root gate receipt; this was not a full GPU serve gate.

## Production scope and remaining uncertainty

The change resolves the native opener from actual tokenizer/model metadata
and gates the special behavior to `glm5_next`. It preserves the opener through
sampling and ends reasoning at that token in first-token, ordinary and MTP
emission paths, with corresponding API handling. GLM rejects undeclared tool
names **before** generic fuzzy single-tool repair can reinterpret them.
Other models retain their prior behavior.

For GLM natural endings, the generic mid-word thinking-end defer is bypassed
and the model's minimum-reasoning floor is explicitly zero. Explicit budgets,
loop safeguards and health/lease checks remain. Selected verification makes
the first phase boundary the bonus token before publishing accepted counts,
trimming targets or detaching producers; later rows chosen under the old
phase's masks are not committed.

These are source-backed boundary corrections, not a demonstrated explanation
or resolution of the fresh arithmetic failure. The earlier fresh, repeated
and post-cancellation requests already failed identically, so cancellation
was not necessary to trigger it. Fresh math, tool, near-full-context and
performance qualification must be rerun on the new pinned image; historical
failures remain recorded in [qualification progress](release-qualification-progress.md).
