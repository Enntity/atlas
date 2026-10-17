# EXL3 CUDA validation

These checks require a real CUDA device and NumPy/CuPy. Missing dependencies,
compiler errors, or failed assertions produce a nonzero exit status. A CPU-only
self-check is not a GPU qualification result.

```bash
python3 tests/exl3_cuda_oracle.py
python3 tests/exl3_validation/kernel_moe.py
```

The first checks packed decoding at every integer bitrate, batches, H128,
SwiGLU bounds and in-place output, TP partition arithmetic, F32 vocabulary
outputs, padding trim, and rank-major gather layout. The second checks indirect
expert pointer tables with mixed bitrates, expanded down inputs, F32 routing,
bias-only selection, stable ties, shared/routed combination, and invalid IDs.
Neither substitutes for Atlas's two-rank NCCL execution or full-model parity.

For source-only checks on a CPU host:

```bash
python3 tests/exl3_cuda_oracle.py --cpu-only
python3 tests/exl3_validation/kernel_moe.py --cpu-only
```

## Full-model evidence

Set `ATLAS_EXL3_LOGITS_DIR` to an empty, explicit destination on rank zero and
disable CUDA graphs for the capture run. Native head capture writes raw F32
pre-softmax logits after complete TP vocabulary gather and checkpoint-padding
trim, before conversion into the existing BF16 serving sampler buffer. Preserve
each `.f32` file and its JSON sidecar, with a mapping from the executed request
and teacher-forced prefix to the captured row. Directory reuse must not silently
overwrite an earlier run.

Freeze prompt token IDs and teacher-forced continuation IDs once. At each prefix,
execute both engines with the same checkpoint, tokenizer/template hashes,
activation/cache dtypes, topology, eager execution, and disabled speculation and
prefix caching. Compare the full vocabulary at identical positions, including
finite-value checks, explicit numerical tolerances, argmax margins and distribution
error. Text similarity and top-k HTTP logprobs cannot establish raw-logit parity.
Keep deliberate changed-token and shuffled-vocabulary controls that must fail.

Record engine/build/binary/source hashes, actual checkpoint revision and shard
hashes, full flags and relevant environment, token IDs, harness/source hashes,
GPU UUIDs/driver/CUDA/NCCL versions, hostname and UTC date beside every artifact.
If a layer diverges, capture KDA/MLA, router, selected expert IDs and FFN outputs at
the first differing position before changing numerical tolerances.

## Reference compatibility

The pinned EXL3 plugin's [compatibility audit](https://github.com/vcruz305/vllm-exl3/blob/223e246ff8da552d2ace976c4bcbed804a8ec04c/docs/VLLM_COMPATIBILITY.md)
checks vLLM v0.30.0 source interfaces; its last GPU-qualified runtime is the
0.28.1rc1.dev324 fork. This is not evidence that installing the plugin into a stock
v0.30.0 runtime serves this exact checkpoint. Preserve the working reference image,
its local patches and runtime fingerprint before preparing an Atlas image.

## Isolated native image

Use `docker/gb10/glm-5.3-flash/exl3/Dockerfile`. It requires explicit CUDA
base images, toolchain, CPU affinity, commit and source snapshot hash. Derive the
CUDA image arguments from the digest-pinned `docker/gb10/Dockerfile`.
Use the installed stable release version explicitly, rather than an unrecorded
rolling toolchain. Build with `--no-cache`, a unique source-hash validation tag,
and no GPU-serving processes competing for RAM. The single-CPU affinity also
bounds the kernel build's internal compiler pool; Cargo's job flag alone does not.
Record the resolved image ID/digest and packaged fingerprint files. The image
is a validation candidate until model/collective/logit checks pass.

From the extracted, hash-verified source snapshot, invoke:

```bash
bash tests/exl3_validation/build_image.sh \
  "$SOURCE_SNAPSHOT_SHA256" "$SOURCE_COMMIT" "$RUST_STABLE_VERSION" "$BUILD_CPU_ID"
```

`BUILD_CPU_ID` must name one CPU permitted by the build host's affinity mask.
Preserve the complete build log. This script builds only the isolated validation
tag and prints the resolved image ID and labels; it never publishes an image.

## Teacher-forced collection and comparison

`collect_logits.py` sends one serial `/v1/completions` request for every fixed
prefix, including the final continuation prefix. It saves the full vocabulary
of the final captured row, requests/responses, capture mappings, source hashes,
and token/archive hashes. Each request must produce exactly one sequential
`logits-NNNNNN.json`/`.f32` capture. Use a fresh capture directory and artifact
output for each engine. Run no other requests during collection.

Supply one shared cases JSON with this structure (the IDs below illustrate the
format; freeze IDs from the actual checkpoint before model validation):

```json
[{"id":"case1","prompt_ids":[1,2],"continuation_ids":[3,4]}]
```

Optionally freeze a local text dataset of
`[{"id":"case1","prompt":"...","continuation":"..."}]` once:

```bash
python3 tests/exl3_validation/freeze_cases.py \
  --checkpoint "$CHECKPOINT" --input "$TEXT_CASES" --out "$TOKEN_CASES" \
  --add-special-tokens no
```

This uses a local tokenizer without remote code or downloads. Prompt and
continuation are encoded separately; continuation receives no special tokens.
The adjacent `.tokenizer.json` records tokenizer source hashes and this policy.
Use the resulting token IDs unchanged on both engines.

Each engine supplies its own provenance JSON. Required fields are `engine`
(`atlas` or `vllm`), `engine_build` (commit, binary_sha256, source_sha256, image),
`checkpoint_revision`, `checkpoint_sha256`, `tokenizer_sha256`, `config_sha256`,
`serve_flags` (complete argument list), `environment`, `hostname`, `utc`, and
`hardware` (two gpu_uuids, driver, cuda, nccl). All three checkpoint/tokenizer/config
hash fields are nonempty filename-to-SHA256 maps. `execution` must include:

```json
{"tp":2,"ep":1,"kv_dtype":"bf16","activation_dtype":"bf16","speculative":false,"eager":true,"prefix_caching":false}
```

Add shared execution requirements to `execution`; the complete map must match
between engines. Record engine-specific numerical policies in a separate
provenance adjunct, such as `native_numerical_policy`, including its source
identity. Engine builds, flags, and hardware are recorded separately and may
differ. Preserve original manifests if an adjunct was placed in the shared map:
create separately named derived copies and record the exact field move and
parent/derived file hashes before scoring. Raw tensors, requests, preregistration
and tolerances must remain unchanged. Hash actual artifacts and record revisions and flags;
provenance validation does not independently attest the running server.

```bash
python3 tests/exl3_validation/collect_logits.py \
  --endpoint "$ENDPOINT" --model "$MODEL" --cases "$TOKEN_CASES" \
  --capture-dir "$CAPTURE_DIR" --out "$ARTIFACT_DIR" \
  --provenance "$PROVENANCE_JSON" --timeout "$REQUEST_TIMEOUT_SECONDS" \
  --thresholds tests/exl3_validation/thresholds.json

python3 tests/exl3_validation/compare_logits.py \
  --reference "$VLLM_ARTIFACT_DIR" --candidate "$ATLAS_ARTIFACT_DIR" \
  --out "$COMPARISON_REPORT" --abs "$ABS_TOL" --rel "$REL_TOL" \
  --kl "$KL_TOL" --ce-max-error "$CE_TOL" --top-k "$TOP_K"
```

The supplied `thresholds.json` preregisters absolute 0.15, relative 0.02, KL
0.005, cross-entropy 0.15, top-k 10 and exact argmax. The collector records these
thresholds and provenance in `pre_registration.json` before requests. The
comparator rejects thresholds differing from this record. Numerical failures
require investigation; increasing a threshold requires a justified new run.

The collector creates an exclusive `capture-enabled` marker immediately before
requests and removes it on success or failure. Both engine capture hooks must
require that marker so startup dummy forwards do not contaminate captures.

When running a multi-rank gate driver, wrap one run at a time so failed output,
raw captures, and both rank logs survive retries:

```bash
tests/exl3_validation/run_gate_attempt.sh ref ref01 /path/to/gate_run.sh
```

The wrapper refuses reused run/attempt names. It archives under
`~/gate/attempts/<name>/` on both hosts and returns failure if evidence
preservation fails.

Choose tolerances explicitly before examining the comparison. The pass gate
requires full-shape finite logits, per-element absolute/relative tolerance,
matching raw argmax and exactly matching actual HTTP generated text per prefix,
bounded reference-to-candidate KL, and bounded continuation
cross-entropy error. Reports also include top-k overlap and reference argmax
margins. Changed token-order and cyclic vocabulary-permutation controls must both
be detected; uniform or insensitive fixtures cannot qualify the negative control.
Atlas serves through its existing BF16 sampler after F32 head capture, so raw
argmax alone does not establish actual sampled-token agreement. Generated text
is checked without whitespace or other normalization; equal decoded text does
not independently prove token-ID identity. This comparison makes no speed claim.

CPU tests use a real local HTTP server and filesystem captures:

```bash
python3 -m unittest discover -s tests/exl3_validation -p test_parity.py -v
```

## Persistent autoregressive decode

The teacher-forced collector re-prefills each prefix. Separately exercise actual
KV/recurrent-state decode with one serial, greedy, 16-token request per frozen
prompt. `collect_decode.py` uses the same cases/provenance format, preserving
full HTTP requests and responses. It sends only `prompt_ids`; prescribed
`continuation_ids` are retained in the fingerprint and do not constrain generation.
It does not enable logits capture and rejects an existing capture marker.

```bash
python3 tests/exl3_validation/collect_decode.py \
  --endpoint "$ENDPOINT" --model "$MODEL" --cases "$TOKEN_CASES" \
  --out "$DECODE_ARTIFACT_DIR" --provenance "$PROVENANCE_JSON" \
  --timeout "$REQUEST_TIMEOUT_SECONDS"

python3 tests/exl3_validation/compare_decode.py \
  --reference "$VLLM_DECODE_DIR" --candidate "$ATLAS_DECODE_DIR" \
  --out "$DECODE_COMPARISON_REPORT"
```

The comparison requires exact generated text, terminal finish reason, and
completion-token count, with no normalization. Generated token IDs are compared
when both responses expose them; unavailable token-ID comparisons are reported
explicitly. Prompt-token usage must match the prescribed IDs. This supplements
full-vocabulary teacher-forced validation and makes no speed claim.

```bash
python3 -m unittest discover -s tests/exl3_validation -p 'test_*.py' -v
```

## Same-engine reproducibility

Before attributing a numerical difference to Atlas, capture repeated identical
requests with the reference. Use fresh artifact directories, identical input
IDs and execution fingerprints, and the same preregistered thresholds:

```bash
python3 tests/exl3_validation/repeat_logits.py \
  --run "$REFERENCE_REPEAT_1" --run "$REFERENCE_REPEAT_2" \
  --run "$REFERENCE_REPEAT_3" --out "$REPRODUCIBILITY_REPORT"
```

This checks every pair and reports bitwise equality, full-vocabulary errors,
distribution errors, argmax and HTTP text. Changed hooks, environment, flags or
binaries fail the fingerprint check; compare those as separately labeled
diagnostics. Reference repeatability is a prerequisite for interpreting a strict
parity result. A failed repeat check does not qualify the Atlas candidate or
authorize changing its thresholds.

## Native precision component checks

Run these in an isolated CUDA environment with the exact installed reference
source and compiled extension. They require Torch, CuPy and the pinned vLLM
installation; missing exports or dependencies fail the command. Use fresh
artifact directories and preserve failed attempts before changing source.

```bash
python3 tests/exl3_validation/index_quant_oracle.py \
  --native-source kernels/gb10/glm-5.3-flash/exl3/glm_indexer_quant.cu \
  --reference-source "$INSTALLED_KPOOL_SOURCE" \
  --reference-sha256 "$KPOOL_SOURCE_SHA256" \
  --reference-attention-sha256 "$ATTENTION_SOURCE_SHA256" \
  --out "$INDEX_COMPONENT_ARTIFACT_DIR"

python3 tests/exl3_validation/dense_activation_oracle.py \
  --native-source kernels/gb10/glm-5.3-flash/exl3/exl3.cu \
  --input-bin "$REFERENCE_L0_GATE_UP_BIN" \
  --input-sidecar "$REFERENCE_L0_GATE_UP_JSON" \
  --input-sha256 "$REFERENCE_L0_GATE_UP_SHA256" \
  --reference-binary-sha256 "$VLLM_STABLE_EXTENSION_SHA256" \
  --limit 10 --out "$DENSE_ACTIVATION_ARTIFACT_DIR"
```

The index check calls the installed quantizer and compiled index normalization
and weight-scale leaves. It checks FP8 ties and saturation, paged key layout,
query in-place writes and guard storage. Its scorer comparison checks the
representation against independent dot-product math; it does not reproduce
DeepGEMM accumulation order. The dense activation check calls the actual
`torch.ops._C` operators registered by `vllm._C_stable_libtorch`, checks captured
BF16 inputs, clamp limits 10 and 1, signed zeros, separate and in-place outputs,
and a control that must reject the old rounding/clamp policy. The compiled
extension hash is distinct from the vLLM console entrypoint hash.

These reports qualify components only. Full-model logits and persistent decode
still require the frozen comparison gates above. The general
`tests/exl3_cuda_oracle.py` also checks the explicit F16 input-transform export
against an independent CPU oracle while preserving the original F32 contract.

For the router, `differential_router.py` can additionally test the existing Atlas
cuBLASLt descriptor and first heuristic with explicit `--cublaslt-library` and
`--runtime-wrapper` paths. Both arguments are required together. The runner
records the exact library version/hash, loaded library mappings, opaque selected
algorithm, 64 MiB workspace, and guards. Its exit gate requires byte equality
with the actual captured logits. A reference-library result does not qualify a
candidate using different library bytes.

`shared_reconstruction.py` tests bounded shared-expert weight materialization;
it prohibits routed-expert prefixes. Supply the checkpoint, shared prefix,
installed plugin path/hash, rank, and a new output directory. Optional paired
`--gate-up-json` and `--down-json` arguments bind the actual resident BF16 shared
weights. Reconstruction uses the full checkpoint projection geometry before TP
slicing and checks native decoded inner weights independently. Its strict byte
gates preserve FP16-to-BF16 double rounding and signed zero. Ordered scalar H
matrix accumulation is an experimental candidate until it matches the installed
Torch reconstruction and loaded weights; compilation alone is insufficient.

`shared_gemm_oracle.py` binds the actual resident merged BF16 gate/up and down
weights, captured BF16 input, installed activation operator, and exact candidate
cuBLASLt library. It checks every stage byte for byte and records separate
gate/up GEMMs as a geometry control. Shared reconstruction and forward reports
qualify only their explicitly captured layers and ranks.

For routed arithmetic, `mma_probe.py` compares the same packed native tensor-core
helper against frozen reference Hadamard inputs with an explicit tile and
split-K policy. Preserve the `--half-partial 0` negative control alongside the
qualified `32/256/8/1` profile. `differential_expert.py` and
`differential_fat.py` require both `--native-mma-profile 32 256 8 1` and
`--native-fat-mma-profile 16 128 1 0` to run the scoped
production export through all expert stages; the former also requires explicit
`--native-thin-cap 32`. The fat runner tests occupancies 32 and 33. Thin experts
use tensor-core accumulation and ordered FP16 partials. Fat experts use sequential K16 tensor-core accumulation with F32 outputs.
Preserve the test-only `--experimental-fat-mma-profile 32 128 1 0`
negative control, which changes the accumulation order. Preserve component failures and require the original full-model
gates before claiming parity.

## API behavior (Q3)

`collect_behavior.py` runs every bounded HTTP behavior case against a live
Atlas server and writes one receipt set (`*.request.json`,
`*.response.raw`, `*.events.json`, `manifest.json`); it exits nonzero unless
every receipt passes. Profile flags are enforced from the provenance file
(context 4096, prefill 128, batch 1, one sequence).

```bash
python3 tests/exl3_validation/collect_behavior.py --base-url "$ATLAS_URL" \
  --model "$MODEL" --provenance "$PROVENANCE" --out "$BEHAVIOR_DIR" \
  --context-limit 4096 --concurrency 4 --timeout 600
```

| Module | Surfaces |
|---|---|
| `collect_behavior.py` | thinking on/off/budget, blocking vs SSE identity, named/auto/streamed tool call + result follow-up, queued requests |
| `behavior_surfaces.py` | `/v1/chat/completions` and legacy `/v1/completions`, blocking and SSE: finish `stop` vs `length`, stop sequences, stream/blocking usage identity, `include_usage` chunk order, multi-turn recall, seeded sampling repeat, reasoning separation, special-token leaks |
| `behavior_tools.py` | `tool_choice` required / named / none / auto / invalid (400), parallel calls, streamed delta assembly, dependent follow-up from tool results, `response_format` `json_schema` and `json_object` |
| `behavior_messages.py` | `/v1/messages` blocking and SSE: event order, content blocks, thinking enabled/disabled, `stop_reason` / `stop_sequence`, usage, `tool_use` to `tool_result` round trip, parity with the OpenAI path |
| `behavior_robustness.py` | client cancel mid-stream and client timeout (chat and messages), then `/health`, an identical repeat of the baseline request, and `atlas_requests_active 0` |

Assertions are exact: expected answers, schema conformance
(`behavior_schema.py`, a stdlib subset that rejects unknown keywords), token
accounting identities and protocol order, never mere non-emptiness. The
legacy prompts embed the GLM-5.3 chat frame because `/v1/completions` applies
no template. Thinking-off without tools still yields brief native reasoning
(`glm5_next.jinja`), so only tool turns without a thinking directive must
report zero reasoning (`thinking_in_tools = false`).

Some gates encode the OpenAI/Anthropic contract where the current server code
differs, and are expected to fail until Atlas is fixed. Examples are
multi-token stop strings on blocking and legacy-streaming paths, the chat
`include_usage` order, streamed Anthropic `stop_sequence`, and seeded
sampling repeatability. Do not relax a gate to match the server.

### Endpoint-agnostic receipts: `q3_behavior_suite.py`

`q3_behavior_suite.py` runs the same families against any OpenAI/Anthropic
origin (Atlas, the vLLM reference, ...) without a provenance file or
profile-flag checks, and writes one receipt per case:

```bash
python3 tests/exl3_validation/q3_behavior_suite.py --endpoint "$ATLAS_URL" \
  --model "$MODEL" --out "$Q3_DIR" [--only CASE ...] [--timeout 600] \
  [--eos-token-id ID ...] [--client-timeout 1.0]
```

`--out` must not exist. It receives `<case>.json` (every assertion with
`pass` / `fail` / `skipped` and its evidence), `summary.json` (per-case
result, failed and skipped assertion names, suite source hash) and `raw/`
(each request payload, raw response bytes and decoded SSE events). Exit
status is 0 only when every case passes. A failure never stops a case or the
suite. Interruption cases run last. Run `--help` to list the case names.

| Case(s) | Gates |
|---|---|
| `chat-basic`, `chat-length` | exact answer; finish `stop` / `length`; `length` consumes exactly `max_tokens`; usage integers and `prompt + completion == total`; `include_usage` finish chunk then a final `choices: []` usage chunk; stream equals blocking |
| `completions-basic`, `completions-length` | the same on legacy `/v1/completions` (plain prompt, `.` stop) |
| `stop-string` | `Observation:` (multi-token) ends output at the match, blocking and SSE: marker and following text absent, text equals the unstopped greedy baseline's prefix, fewer tokens than the baseline |
| `stop-token-ids` | chooses a content token from a `return_token_ids` baseline; with `stop_token_ids` the content is truncated before it and `completion_tokens == position + 1` (vLLM `check_stop`) |
| `stop-include` | `include_stop_str_in_output`: blocking and SSE output equals the unstopped greedy baseline through the first `Observation:` (kept, once, nothing after); stream equals blocking |
| `stop-empty` | an empty stop string (`[""]`, `["Observation:", ""]`, bare `""`; chat and completions) is rejected with HTTP 400 (vLLM `sampling_params.py:763`) |
| `eos` | natural end: finish `stop` before `max_tokens`, no leaked specials; last token in `--eos-token-id` (otherwise `skipped`) |
| `messages-*` | raw Messages event order; `end_turn` / `max_tokens` / `stop_sequence` with the matched string echoed on blocking and SSE; usage; stream equals blocking |
| `seed`, `seed-json-schema` | same seed at temperature 1.0 twice: identical first token (`token_ids[0]`, or the first delta if the server exposes no ids) and identical output, without and with `json_schema`; a different seed must change the joke (rules out a server that ignores temperature) |
| `responses-seed` | `/v1/responses` with `seed` at temperature 1.0 twice: identical reasoning and `output_text`; a different seed changes it |
| `structured-*` | strict JSON parse, `behavior_schema` validation, exact values |
| `tools-*` | single and parallel calls: `tool_calls` finish, call count, distinct ids, strict-JSON schema-valid arguments; dependent follow-up must answer from counter-seasonal tool results |
| `reasoning-on`, `reasoning-off` | `enable_thinking`: reasoning present and separated (on), exact content, no `<think>` / `<|...|>` markers in either channel, `reasoning_tokens` consistent with the channel |
| `multi-turn` | the second turn recalls the first turn's code word (blocking and SSE) |
| `cancel-*`, `timeout-*` | `/health` before and after; cancel after 3 SSE frames or client timeout; the identical baseline request then matches the pre-interruption baseline; `atlas_requests_active 0` when exported (otherwise `skipped`) |

`reasoning-off` does not require an empty reasoning channel because the GLM
template keeps brief native reasoning (see above). Several gates are expected
to fail on servers that do not implement the contract. For example, vLLM's
Anthropic stream sends `stop_sequence: null`. These are findings, not reasons to relax a gate.

CPU self-tests use a local fake server only and are not qualification:

```bash
uv run --no-project --with pytest python -m pytest -q \
  tests/exl3_validation/test_behavior.py tests/exl3_validation/test_behavior_schema.py \
  tests/exl3_validation/test_behavior_anthropic_protocol.py \
  tests/exl3_validation/test_behavior_surfaces.py tests/exl3_validation/test_behavior_q3.py \
  tests/exl3_validation/test_collect_behavior.py tests/exl3_validation/test_q3_behavior_suite.py
```

`test_q3_behavior_suite.py` serves a deterministic fake engine. The fake must
pass every case, and each injected fault must fail the assertion it targets.
## Q4 task-quality corpus

`q4/corpus.jsonl` is the frozen held-out corpus (81 cases: arithmetic, executable
code, exact tool arguments, structured output, multilingual, long-context
retrieval near the 4096 limit, sustained generation); `q4/tuning.jsonl` holds 10
disjoint tuning cases. Both are generated by `q4/build_corpus.py`, which computes
every expected value and carries a reference solution per code case; never edit
the jsonl files by hand, and after the first measured run only add new ids.
Scoring is checker-only (`q4/checkers.py`): no case is compared with reference
engine output, and a failed request is a failed trial. Thresholds in
`q4/thresholds.json` are provisional until the qualified serial profile is
measured.

```bash
export TMPDIR=$PWD/.tmp   # the code checker runs generated code under $TMPDIR
python3 tests/exl3_validation/q4/run_q4.py --endpoint http://HOST:PORT --model MODEL \
  --profile serial-qualified --trials 5 --out RUN_DIR
python3 tests/exl3_validation/q4/run_q4.py --compare SERIAL_DIR/summary.json CANDIDATE_DIR/summary.json
python3 -m pytest -q tests/exl3_validation/test_q4.py   # CPU self-tests, fake server only
```

The runner sends `chat_template_kwargs.enable_thinking=false` with temperature 0
and a fixed seed. GLM-5.3 still reasons briefly in that mode (`low_think`), and
those tokens count against each case's `max_tokens`. Long-context prompt sizes
were measured with the checkpoint tokenizer; every trial records
`usage.prompt_tokens` to confirm them on the live server.
## Concurrency and context (F2/F3)

`f2_concurrency.py` and `f3_context.py` qualify F2 and F3 against a
live server. Both use `harness_client.py` (standard library only: strict SSE
parsing, an overall per-request deadline and cancellation enforced by closing
the socket, token-for-token comparison) and write one receipt that is never
overwritten. Requests are greedy (`temperature` 0, fixed `max_tokens`,
`return_token_ids`, thinking disabled unless `--thinking`). Comparisons use
`token_ids`, else logprob tokens, else text; the basis is recorded and a
text-only comparison fails as not token-level. Exit status is 0 only when
every check passes.

```bash
python3 tests/exl3_validation/f2_concurrency.py --endpoint "$ATLAS_URL" \
  --model "$MODEL" --fingerprint "$PROVENANCE" --out "$OUT/f2.json" \
  --concurrency 2 4 --max-tokens 96 --top-logprobs 5 --stagger 0.25 --timeout 600
python3 tests/exl3_validation/f3_context.py --endpoint "$ATLAS_URL" \
  --model "$MODEL" --fingerprint "$PROVENANCE" --out "$OUT/f3.json" \
  --ladder 1024 2047 2048 2049 4000 --max-tokens 96 --timeout 900
```

F2 runs every case isolated (`--isolated-repeats`, default 2; a repeat that
differs is reported as isolated nondeterminism), then interleaved at each C
with staggered starts, then at the largest C with one stream cancelled after
`--cancel-after` chunks, followed by an isolated repeat of the cancelled
case. Each interleaved output must equal its isolated output; failures name
the first divergent position, both tokens and the logprob gap. A phase must
show server-side overlap (some request's first token arrives while another
is still decoding); a server that runs requests one at a time fails with
"server appears to serialize requests", which is the expected result while
the launcher admits a single sequence. `/health` must be 200 after every
phase. TTFT/TPOT follow
`perf_baseline.py`; `aggregate_completion_tok_per_s_wall_unqualified` mixes
queueing, prefill and decode and is a load indicator only.

F3 (a) calibrates each rung to an exact templated prompt length with
`POST /tokenize` (an inexact rung fails unless `--allow-inexact`), checks the
needle and a byte-exact code/tool-call line, and requires
`usage.prompt_tokens` to equal the calibrated count. The top rung is
`max_model_len - max_tokens`; prompts of `max_model_len` and
`max_model_len + 256` must get a 4xx JSON error, a healthy server, and an
exact repeat of a sanity request. F3 (b) sends a nonce-headed shared prefix
with three suffixes (cold, warm, repeats), then refills the cache with
`--evict-prompts` unrelated prompts before repeating each suffix. All repeats
must equal the first output; `cached_tokens`, the warm/cold TTFT ratio and
`eviction_observed` are recorded as evidence, and `--require-cache-hit` makes
a missing warm hit a failure.

CPU self-tests use a local fake server only and are not qualification:

```bash
python3 -m pytest -q tests/exl3_validation/test_f2_f3_harness.py
```

## EXL3 statistical gate v2

`statistical_gate_v2.py` is a CPU evaluator for the pre-registered v2 rule in
`docs/design/exl3-statistical-gate-v2.md`. It imports the unchanged v1 artifact
loader, metric, discrete-rule and rank primitives. The v1 CLI and its historical
results remain unchanged.

Validate a frozen plan with:

```bash
PYTHONPATH=tests/exl3_validation python3 tests/exl3_validation/statistical_gate_v2.py \
  --validate-plan campaign-plan.json --out plan-validation.json
```

The plan JSON uses schema `atlas-exl3-gate-v2-plan-v1`. It records `campaign_id`,
integer `seed`, `attempt_order` (the 20 registered reference/held-out/candidate
names shuffled with Python `random.Random(seed).shuffle`), the fixed
`post_attempt_order`, the two selected routed layer indices, and exact engine
identities (each has `engine` and a `build` object containing `commit`,
`binary_sha256`, `source_sha256`, `image`). Validation does not start collection.

The campaign JSON uses schema `atlas-exl3-gate-v2-campaign-v1` and contains
`plan`, an `attempts` mapping from all 26 registered attempt names to comma-
joined short16/long artifact directories, and a `cache_receipts` mapping. Each
receipt has `pre_start` and `post_shutdown` entries with an attempt name,
UTC capture time, and the ordered cache categories `vllm`, `triton`, `tilelang`,
`cuda_compute`, `exl3_tune`. Every mount records host/container paths, sharing
across attempts, explicit `present`/`empty`/`missing` state, and sorted
relative-file size/SHA256 manifests. Receipts also prove attempts ran in the
frozen order without overlap. Missing artifacts or receipts produce
`INCOMPLETE`; failed reference LOO checks produce `UNVERIFIABLE`; complete
campaigns report `PASS` or `FAIL`.

Evaluate a recorded campaign with:

```bash
PYTHONPATH=tests/exl3_validation python3 tests/exl3_validation/statistical_gate_v2.py \
  --campaign campaign.json --out gate-report.json
```

The evaluator only reads artifacts and receipts. It does not collect, retry,
replace or top up attempts. Run the production-path fixtures with:

```bash
PYTHONPATH=tests/exl3_validation python3 -m unittest test_statistical_gate_v2
```
