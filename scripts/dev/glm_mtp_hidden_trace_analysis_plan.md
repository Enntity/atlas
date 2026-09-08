# Bounded GLM MTP hidden trace log analysis

Plan first: CPU-only scripts, no engine edits, nodes, Docker, GPU or Cargo.
The native producer is `layers/glm5_mtp/hidden_trace.rs::StepTrace::emit`.
This harness checks evidence; it does not explain a cache bug or measure speed.

The version3 first-attempt private-KV extension is specified separately in
`glm_mtp_first_kv_probe_plan.md`; legacy v1/v2 evidence stays explicitly
unavailable for its new KV fields rather than being filled with equality.

## Inputs and strict bounds

Add `analyze_glm_mtp_hidden_trace.py` plus unittest fixtures in a separate file.
Use only Python standard library. CLI takes one explicit JSON manifest and
prints JSON to stdout; never writes input files. The manifest names at most
eight independently collected requests, each with a unique human request/run
label, explicit expected attempt count1..8 and two rank log selectors
(`rank`, `path`, `slot`, `generation`). This is how the caller supplies actual
request ownership; local generation numbers need not agree across ranks/runs.
Manifest paths resolve relative to the manifest directory. Require nonempty
labels, distinct ranks0/1, nonzero generations and exact expected selectors.
Each input file is bounded to16MiB, each line16KiB, and each selected request to
32 trace records/rank. Multiple manifest requests may select different
slot/generation owners from the same complete log file; parse each file once.
Ordinary unrelated log lines are ignored, but every HIDDEN_TRACE line must parse
exactly and belong to one explicitly declared selector somewhere in the manifest.
Reject foreign/undeclared trace rows rather than filtering them away. Thus all
six root requests (warmup plus five) can reference the same two lossless rank
logs, and complete-file counts prove extraction did not hide unmatched records.
Explicit valid startup flag receipts (`ATLAS_GLM_MTP_HIDDEN_TRACE=0/1` or its
launcher alias) are not events; ignore them only when no actual event marker is
present. Malformed actual event markers and invalid flag receipts still reject.

Support the actual tracing text schema (optional timestamp/module/ANSI prefix),
including Rust `Some([byte,...])` or `None` argmax-pair formatting. Require every
producer field exactly once; reject duplicate/unknown trailing trace fields,
malformed 64-lowercase-hex digests, invalid booleans, out-of-range integers,
non16-byte pair payloads and nonfinite local maximum values. Do not parse full
vocabulary or arbitrary eval expressions. Trace lines remain small; no regex
over whole logs, shell commands, subprocesses or dependency installation.
The producer uses dotted shorthand field names `self.cache_before` and
`self.cache_after`: retain those exact wire names (confirmed in tracing0.1.44's
local macro implementation), normalize only after strict parsing.

## Per-request evidence checks

Validate uniqueness of `(rank,slot,generation,attempt,step)`, attempts exactly
1..expected and ordered steps0..3. Reject missing/duplicate/out-of-order records.
Within each attempt require constant position/seed/hidden_row/EH path, position
and token domains, exactly advancing private cursor (`after=before+1`), next
step input token equals preceding draft, next input SHA equals preceding final
SHA, and step0 token equals seed. Correlate both ranks by explicit manifest
request plus attempt/step and compare position/seed/hidden_row/cursor/path
metadata; mismatches are reported, never silently paired by line order.

The harness cannot know that a request which logged eight successful attempts
made no later attempts: exhausted trace is intentionally silent. It also cannot
distinguish an absent record caused by failed device I/O from a truncated log.
Missing rows are an integrity failure, not fabricated zero hashes.

## Comparison and limitations

Manifest may explicitly request pairs of request labels for comparison. Match
cross-request attempts by `(position,seed)` rather than ordinal, because earlier
acceptance can change later attempt numbering. Ambiguous repeated match keys
reject; unmatched attempts are counted and reported. Compare each matched step
within the same rank, retaining both full ownership/attempt keys in output.

Classify differences at the earliest observable boundary: changed conditioning
token/position or input SHA; equal conditioning tuple with different final SHA;
equal final SHA with changed optional argmax pair bytes; different final chosen
draft; or observed agreement. Report path/cursor metadata differences separately
and avoid attributing differences to hidden input when precision paths differ.
Both-rank comparisons use the same classification, but naturally rank-local
projection paths and state remain part of the evidence, not assumed equal.
If exactly one record lacks optional argmax-pair evidence, explicitly classify
that availability difference instead of calling the full observation agreement.

For available argmax pairs decode the already-read little-endian f32/u32 values
and retain them; selected global token cannot be independently reconstructed
without the model's local vocabulary width, so do not invent that width.
Equal input/final hashes do NOT establish equal private KV, target causal
prefixes, internal arithmetic, or root cause. Matching position/seed is a
comparison candidate, not proof of equal target history. State this in JSON.

## TDD and validation

Write valid/malformed fixtures against a callable analysis interface first;
the initial stub returns an empty report so the valid fixture fails behaviorally.
Cover complete both-rank four-step/eight-attempt traces; different local
generations; missing/duplicate/malformed/foreign/out-of-order rows; bounded file
and line rejection; broken cursor/token/hash chain; optional pair parsing;
input/final/pair/draft classification; cross-request shifted attempt matching
and ambiguous keys. Preserve RED and GREEN receipts in the campaign directory.
Tests generate tiny synthetic text using exact producer field names, not claims
of native numerical equivalence. Root will supply real clean logs later.

Send frozen script/test hashes and test receipts for independent source review
before commit or native-log conclusions. No throughput claim from traced runs.

## Invocation and manifest

`python3 scripts/dev/analyze_glm_mtp_hidden_trace.py --manifest /path/manifest.json`

The script prints evidence JSON to stdout, or an explicit error JSON to stderr
with exit2. The manifest is bounded64KiB and inputs must be regular files.
Example below shows two declared requests sharing both complete rank logs;
extend to all six root requests using their actual observed ownership stamps.
Request labels are caller provenance, not inferred from generations.

```json
{
  "requests": [
    {"id": "warmup", "expected_attempts": 8, "ranks": [
      {"rank": 0, "path": "rank0.log", "slot": 0, "generation": 1},
      {"rank": 1, "path": "rank1.log", "slot": 0, "generation": 1}
    ]},
    {"id": "repeat1", "expected_attempts": 8, "ranks": [
      {"rank": 0, "path": "rank0.log", "slot": 0, "generation": 2},
      {"rank": 1, "path": "rank1.log", "slot": 0, "generation": 2}
    ]}
  ],
  "comparisons": [["warmup", "repeat1"]]
}
```

Per-step comparisons retain both complete ownership/attempt keys, exact hashes,
chosen drafts, changed metadata values and decoded optional two-shard pairs.
Unmatched cross-request positions are reported rather than forced into matching
attempt ordinals. No assumptions about local vocabulary width or identical
rank-local maxima are required.

## CPU evidence

Receipts live in `/home/abc/storage/models/atlas-campaigns/20260908/`:

- `hidden-trace-analysis-red.log`: the empty-report stub fails the complete
  trace expectation and wrongly accepts malformed/missing/duplicate rows.
- `hidden-trace-analysis-focused-green.log`: initial strict parser, ownership,
  completeness and chain tests pass3/3 after implementation.
- `hidden-trace-analysis-review-red.log`: explicit startup flag receipt was
  wrongly rejected, and missing optional pair evidence was mislabeled agreement.
  `hidden-trace-analysis-review-green.log` passes after both review corrections.
- `hidden-trace-analysis-final-green.log`:13/13 tests pass, including all six
  requests declared against the same two complete files (192 trace records per
  rank), shifted attempt matching, malformed/foreign/duplicate/missing records,
  hash/token/cursor chains, exact dotted field names, pair decoding, byte/line
  bounds and actual CLI success/truncated-log failure. Final diff check passes.

These are synthetic CPU fixtures, not native trace evidence. The rejected v21
request produced no valid trace window; real both-rank logs must come from a
successful root-owned corrected native run before any divergence conclusion.
