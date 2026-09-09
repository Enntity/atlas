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

## v28 boundary fix and rebuild

Commit `4091d8da80415a59b727189eb3ea06c36d12b3df` adds an exact terminal
`[think_start, think_end]` check at the rendered prompt boundary, returning
initial thinking=false/budget=None. Existing unclosed-tail and no-marker
behavior remains unchanged; tool parsing and the native checker are unchanged.
Actual GLM template rendering/encoding reproduced the failure before the fix
and passed afterward. The existing hidden-Poolside-call rejection still passes.
Independent source review approved both files. Controller server checkpoint:
2380 passed,12 ignored (26.74s; NO_COLOR unset as in the v27 checkpoint).

Only those two server files differ from v27's compiled runtime source. The
committed delta archive SHA256 is
`f3cecc516f11d3514e9b1987ea402d11a8074c691b73c334b9a6c396c706c10b`.
The retained CPU-only builder completed in1m21s, exit0/noOOM. Native binary
SHA256 `d32875231dab0d1deae0546428a90a9b960c34d7f931582c9247e5b61dcf8faf`
matches on both nodes in `atlas-glm53-flash:kernel-20260909-v28`:

- Head image `598afb9c3559a961d29c8bb3dcb91abc34209fcfc32c3d2d9abe139c85f5831c`.
- Worker image `b32d3bac8b405ffdaaa0ced728ef5e25378e5a4544771818ca8a056764d462a4`.

The campaign's four v28 launch/measurement scripts are mechanical v27→v28
renames with identical settings, checks and payloads. Retain v27 failure
receipts; v28 uses a fresh prefix and fresh processes for each arm. Build,
packaging and source checks do not themselves prove native tool correctness
or a throughput improvement; the v28 quality and timing gates remain required.

### v28 OFF eager: native tool fix confirmed

The unchanged two tool requests now return exactly one structured function call
each, with exact arguments and `finish_reason=tool_calls`; no reasoning markup
is promoted into calls. Four paired answer checks and both uneven C4 retrieval
waves also pass. Overall gate exit0; both ranks exit0/OOMKilled=false, no used
swap or remaining GPU applications. Ready head11311MiB/worker11752MiB available.
Receipts use prefix `v28-compact-off-eager-`; this remains quality-only, with
no throughput claim. Compact C2 ON requires its own quality qualification next.

### v28 ON eager: compact C2 quality passed

Same binary and recipe, compact bit1 on both ranks. All four answer checks,
both C4 retrieval waves and both exact structured tool calls pass. The uneven
retrieval requests exercise shrinking batches; the run is not speculative.
Ready head11808MiB/worker11802MiB available. Overall gate exit0, both ranks
exit0/OOMKilled=false, zero used swap and no remaining GPU applications.
Receipts use prefix `v28-compact-on-eager-`. Both eager prerequisites are now
qualified; same-binary warmed OFF/ON graph matrices are the next performance
evidence. No native throughput improvement has yet been established.
