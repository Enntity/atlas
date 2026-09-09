# v27 compact C2: native qualification

2026-09-09. Candidate source `b72a2dd553e43c97a8fa1ce6aaf74e6965fd1806`,
native binary SHA256
`ab2b25459267d2d23b903600df9c3e50b3f3733f4ed156953581cecc9a614e2f`,
installed and verified on both Sparks in `atlas-glm53-flash:kernel-20260909-v27`.
The tool checker is committed at `85d01931`; runtime source is unchanged.

## OFF eager: tool-quality failure, no timing claim

Campaign receipts: `/home/abc/storage/models/atlas-campaigns/20260908/`, prefix
`v27-compact-off-eager-`. TP=2/EP=2, active/admitted4, context2048,
BF16 KV, no speculation, compact C2 OFF, multisequence graphs OFF.

- Four answer checks run in pairs passed (arithmetic, stable sorting, Python
  syntax/semantics by AST comparison, exact typed JSON).
- Both uneven C4 retrieval waves passed at 768/800/832/896 prompt tokens.
  This is short-context retrieval evidence, not long-context qualification.
- Both concurrent forced named-tool requests failed: correct function names
  and arguments appeared in raw markup, but neither response contained
  structured `tool_calls`; both incorrectly finished with `stop`.
  The checker did not accept hidden reasoning as a tool call.
- No throughput matrix or optimization-ON run was started. This failure occurs
  with the new C2 path disabled, so it does not establish a C2 numeric regression.
- Ready MemAvailable: head11200MiB, worker11781MiB. Both processes stopped
  with exit0, OOMKilled=false; zero used swap and no remaining GPU applications.
  Retained containers: `atlas-glm53-v27-compact-off-eager-ep0` / `ep1`.

Primary evidence: `before-tools.json` retains full payloads and HTTP responses;
`before-answers.json`, `before-niah.jsonl`, `live-config-rank*.log`,
`final-rank*.log`, `stop-rank*.log` and `gate-exit.log` retain the other gates.
The overall gate exit is1, despite clean process shutdown.

Source and log diagnosis: the GLM template ends tool prompts with
`<think></think>`, even when the request explicitly enables thinking. The chat
template boundary detects an unclosed opener to enable scheduler thinking, but
does not reconcile an explicitly closed suffix in the opposite direction.
This leaves the scheduler/response decoder expecting reasoning when generation
has already entered visible output. The head log resolves `<tool_call>` to
token154843 and records that exact first token for both requests. First-token
sampling advances the grammar past the opener; the incorrect thinking state
classifies it as reasoning. When thinking ends, grammar resumes after the
opener, explaining the bare function name in visible output. Confirm the fix
with a focused regression and a fresh native run; do not reinterpret the failed
receipt, promote hidden tool calls, or weaken the checker.
