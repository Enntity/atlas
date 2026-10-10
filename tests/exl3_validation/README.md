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

Use `docker/gb10/glm-5.3-flash-exl3/exl3/Dockerfile`. It requires explicit CUDA
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
  --native-source kernels/gb10/common/glm_indexer_quant.cu \
  --reference-source "$INSTALLED_KPOOL_SOURCE" \
  --reference-sha256 "$KPOOL_SOURCE_SHA256" \
  --reference-attention-sha256 "$ATTENTION_SOURCE_SHA256" \
  --out "$INDEX_COMPONENT_ARTIFACT_DIR"

python3 tests/exl3_validation/dense_activation_oracle.py \
  --native-source kernels/gb10/common/exl3.cu \
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
