# Strict offline C1 MTP receipt correlation

This tool reads existing benchmark JSON and a complete rank-0 log. It does not
contact a server, change flags, generate tokens, or replace benchmark metrics.

1. Accept exactly one benchmark result with concurrency1, explicit workload
   prompt length/output cap/warmup count/repetition count, and matching C1 runs.
   Preserve original median metrics and validate reported cap integrity against
   each measured request's completion count. Missing metadata is an error.
2. Strip ANSI escapes, parse every sequential `Chunked prefill start` through
   `Done` window, and reject nested starts, orphan completions, malformed known
   records, or an unfinished request. Other complete request shapes may precede
   or follow the benchmark. Match by exact prompt length and requested output
   cap, require exactly warmups+repetitions matches, and preserve their order.
   Never choose the last N matches. The log lacks prompt hashes/request IDs:
   shape/order correlation is not proof of token or hidden-state identity.
3. Retain per-request Done serial/mtp/p1/mean_na/tok_step/depth/reprobe fields.
   Check measured Done token counts against JSON, without claiming log tok/s is
   the client's full-wall rate. Missing MTP Done fields fail the matched window.
4. Preserve each 25-step fwd/propose timing pair with line references. Discard
   the first timing window in EVERY matched request because process-wide timing
   accumulators may include previous requests; no clean windows means an empty
   list, not an invented estimate. Warmup remains explicitly separate. These
   diagnostics are not GPU event measurements or benchmark replacements.

TDD: synthetic success with unrelated complete requests and ANSI; reject C4,
missing/wrong counts, extra matching requests, interleaving/incomplete windows,
invalid metadata/cap claims and mismatched Done counts. Verify exact metric
preservation, first-window discard, and empty timing results. Then validate
read-only against the existing v16 EH-BF16 receipt; future v18 uses the same CLI.

CLI: `python3 scripts/dev/summarize_glm53_c1_mtp.py BENCHMARK.json RANK0.log`
prints a JSON report to stdout; errors return nonzero without a partial report.

## Implementation receipt

Implemented offline only; 11 synthetic tests pass. Initial stub RED produced
18 failed assertions and3 errors across8 tests (including count, interleaving,
schema and cap rejection). Final tests add zero warmups, malformed diagnostics,
overflowing metadata counts and original-metric validation. Receipts under
`/tmp/atlas-glm53-phase6-20260907.J5PkkO/c1-mtp-summary-{red,green}.log`.

Read-only real-log validation succeeded for both complete rank0logs:

- v16 EH-BF16: exactly1 warmup+3 measured matches,4 unrelated quality requests;
  original fullwall median27.258tok/s, allcap. Measured mean_na2.395/2.583/2.534;
  six clean timing windows after discards, diagnostic medians110.935ms fwd and
  11.655ms propose.
- v18 M5-on: same match counts; original fullwall median27.947tok/s, allcap.
  Measured mean_na2.583/2.583/2.146; six clean windows, diagnostic medians109.05ms
  fwd and11.845ms propose. These are separate receipt summaries, not a controlled
  attribution claim between engine configurations.

The original benchmark summaries are authoritative and copied verbatim, not
recomputed from rounded log rates. Timing medians are separately named derived
diagnostics. A complete but unrelated serial request is allowed; nested starts
anywhere in the supplied log are rejected. Missing Done diagnostics in a matched
window fail closed rather than assuming zero serial work. No request IDs or
prompt-token identity can be recovered from this log format.
