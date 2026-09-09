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

### v28 OFF graphs: qualified warmed baseline

Same 148-input/256-output LRU workload, temperature0/seed1, ordinary stop rules;
one retained warm-up and three measured waves per concurrency. Median full-wall
aggregate tok/s: C1=13.469, C2=18.991, C3=34.791, C4=47.160. Median server TTFT
respectively419.503,420.601,419.002,419.605ms. All warm-up/measured outputs reach
the requested cap. Before/after answers, tool calls and retrieval pass;
cancellation and recovery pass. Overall gate exit0, clean non-OOM process exits
on both ranks, no used swap or remaining GPU applications. Prefix
`v28-compact-off-graphs-` retains complete timing and quality receipts.

The original benchmark retained completion hashes/byte counts, not generated
text. Its quality evidence comes from the separate retained answer/tool/retrieval
requests, not inspection of the timed coding completions. The first OFF/ON pair
loaded that original harness. Subsequent runs retain completion text as well,
outside the measured window with unchanged payloads and rate formulas. Do not
claim retroactive text inspection of the first pair or semantic equivalence from
hashes alone. Fresh-process repeats should use the same updated harness on both
arms and inspect their retained coding output.

### v28 ON graphs: first paired performance result qualified

Both arms use native source4091d8da/binaryd3287523 and identical workload
metadata. Later staged retirement source3680ee36 is NOT in either native binary.

| Concurrency | OFF aggregate tok/s | ON aggregate tok/s | Change | OFF/ON client TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| C1 | 13.469 | 13.449 | -0.15% | 453.758 /451.459 |
| C2 | 18.991 | 25.560 | +34.59% | 695.098 /700.051 |
| C3 | 34.791 | 34.946 | +0.45% | 948.901 /941.272 |
| C4 | 47.160 | 47.208 | +0.10% | 1159.974 /1154.520 |

These are warmed full-wall medians over three measured waves, not sums of
per-stream rates or speculative decode. ON C2 decode-window rate26.140tok/s;
server-reported TTFT420.566ms excludes some client-observed waiting and must not
be substituted for700.051ms client TTFT. Each arm has40 completions (1+3 waves
times1+2+3+4 streams). All80 paired completions reach256 tokens:60 measured
completions plus20 warm-ups.

ON before/after answers, exact tool calls and both retrieval waves pass;
cancellation/recovery passes. Gate exit0; both processes exit0/OOMKilled=false,
zero used swap and no remaining GPU applications. Prefix
`v28-compact-on-graphs-` retains the evidence. Fresh-process OFF/ON repeats are
required before calling the improvement reproducible. C2 remains below37tok/s;
C6/C8 and concurrent MTP are not qualified, and the short workload is not proof
of exact MiaAI benchmark parity.

### v28 fresh-process repeats: gain reproduced, output-mix sensitivity exposed

Both repeat arms completed with gate exit0 on 2026-09-09. Native binary and
recipe are unchanged; the updated benchmark from123840bb retains text after
timing on both arms. Same literal148/256 workload, one warm-up plus three
measured waves per width. Prefixes `v28-compact-off-repeat-graphs-` and
`v28-compact-on-repeat-graphs-` retain raw evidence in the campaign directory.

| Concurrency | OFF repeat full-wall tok/s | ON repeat full-wall tok/s | OFF/ON client TTFT ms |
| --- | ---: | ---: | ---: |
| C1 | 13.428 | 13.477 | 462.318 /450.151 |
| C2 | 19.025 | 29.746 | 704.896 /657.965 |
| C3 | 34.764 | 35.008 | 954.203 /862.923 |
| C4 | 47.116 | 47.273 | 1160.248 /1153.450 |

C2's repeat median is56.35% above its OFF control, but do not interpret the
larger gain as a new stable baseline. The unchanged workload metadata and
retained output hashes expose two clusters:

- Mixed output pairs (`72891e8f…` plus `f79f067e…`) measured25.446–25.566tok/s
  across the first ON run and repeat; the repeat's mixed wave is25.455tok/s.
- Identical output pairs (`72891e8f…` twice) measured29.746 and29.862tok/s in
  the repeat. The first ON warm-up also had this pair and reached29.971tok/s;
  the repeat warm-up instead had mixed outputs and reached25.547tok/s.

This is an observed output-mix/throughput correlation, not a proven account of
the scheduling or expert-routing cause. There is a reproducible conservative
C2 improvement around34% (about19.0→25.5tok/s); do not advertise29.7 as general
heterogeneous-request throughput. Longer, distinct-prompt workloads and a matched
reference run remain necessary. C1/C3/C4 full-wall medians stay within0.7% of
their repeat controls. These C1 results are nonspeculative, not the separate
single-request MTP4 baseline.

All80 repeat completions reached256 tokens; every retained UTF-8 byte count
and SHA256 was recomputed successfully. Root inspected all seven distinct texts:
coherent partial LRU implementations, not repetition collapse. Some docstrings
refer to an `_dll`/`_list` attribute absent from the visible implementation, and
the cap interrupts functions. This is limited coherence evidence, not complete
program correctness, semantic equivalence, or bit-identical numerical output.

Before/after answer and exact forced-tool checks, both uneven retrieval waves,
cancellation and answer recovery pass for both repeats. Ready memory was
head11732/worker11807MiB OFF and11746/11759MiB ON. Both ranks exited0 with
OOMKilled=false after each arm; zero used swap and no remaining GPU applications.
All four timed arms are now retained, with no node reset/crash. C2>=37, C6/C8,
concurrent MTP and exact reference parity remain unmet/unproven.
