# v22 native hidden-trace evidence

The corrected diagnostic completed six sequential148-input/64-output requests
on both ranks. Root owns native runs and quality gates. These synchronized
diagnostic runs are **not throughput evidence** and do not satisfy the30/60 goal.

## Complete-file integrity

The independently reviewed analyzer (`c054508c`,13 CPU tests passing) consumed
both complete logs, with every generation1..6/slot0 selector declared explicitly.
Manifest and full JSON result are in `atlas-campaigns/20260908/`:

- `v22-hidden-analysis-manifest.json`
- `v22-hidden-analysis.json` (exit0; empty `v22-hidden-analysis.stderr`)
- `v22-hidden-clean-rank0.log`:307644 bytes,192 trace records.
- `v22-hidden-clean-rank1.log`:282941 bytes,192 trace records.

All384 records are accounted for: six requests × two ranks × eight attempts ×
four steps. No foreign/missing/duplicate/out-of-order records; every within-
attempt token/hash/cursor chain validates. The five warmup-to-repeat and four
additional adjacent comparisons each match32 steps/rank, with no unmatched
position/seed keys and no compared cursor/path/seed/position metadata differences.

## Earliest observed boundary

At **generation1, attempt1, position149, seed1304, step0**, both ranks receive
the identical raw input SHA256:

`6c03a73705a818c124de98781562cbaf9fa239cdf3ba860ce4f1f1694ad75965`

Their final post-module-norm hashes differ:

- Rank0: `f83f7895aa582649ab311660380c1323cae818d2a8c470dde7ecde62cd94d10b`
- Rank1: `b2cdc71699edf14687cef52097bb437c42923e7b058d57a380e05a26f9231ec3`

Both select draft49449 from the same gathered two-shard pair payload. Across
all six requests, each of eight attempts has this same pattern: step0 input
hash/token/position agrees across ranks but final hash differs; steps1..3 then
consume their respective differing preceding final hidden rows. Gathered pair
bytes and selected draft tokens agree between ranks at all192 compared steps.
This is consistent with their shared pair collective; it is not a claim that
the two local maxima should have equal values or that local bodies/KV are equal.

Within rank0, every one of the nine selected request comparisons has:

- Eight exact step0 conditioning inputs, but eight different step0 final hashes.
- All32 final hashes different, and24 subsequent input hashes different.
- Identical logged projection-path flags and private cursor metadata.

For example, generation2 at the same first key has the same input hash above
but rank0 final hash
`5e085938dce45021aca75ad7921704c1c9c8bada0026578a85207b7a3841531e`.
Thus target conditioning is not the first differing observed boundary. The
unresolved region is the proposer computation/private state between input and
final snapshots, including EH/body/norm and private KV. This does **not** prove
an uninitialized/reset bug, equal private KV, or a particular kernel cause.

## Cross-request counts and the generation3 exception

`W` is generation1/warmup; `R1..R5` are generations2..6. Counts below are raw
field differences, not just the analyzer's earliest-difference classification.
All comparisons have rank0 final32/input24 hash differences as above.

| Comparison | Rank1 input hash differences | Rank1 final hash differences | Changed drafts, each rank | Changed gathered shard0 / shard1 pairs |
|---|---:|---:|---:|---:|
| W–R1 | 0 | 0 | 0 | 10 / 0 |
| W–R2 | 1 | 2 | 1 | 14 / 2 |
| W–R3 | 0 | 0 | 0 | 13 / 0 |
| W–R4 | 0 | 0 | 0 | 12 / 0 |
| W–R5 | 0 | 0 | 0 | 11 / 0 |
| R1–R2 | 1 | 2 | 1 | 14 / 2 |
| R2–R3 | 1 | 2 | 1 | 16 / 2 |
| R3–R4 | 0 | 0 | 0 | 14 / 0 |
| R4–R5 | 0 | 0 | 0 | 11 / 0 |

The exception is **generation3, attempt2, position152, seed284, step1**.
Both ranks select606 instead of957. Rank1's input and final hidden hashes at
that step are unchanged from warmup; the changed decision is present in the
gathered shard0 maximum: `(18.0,957)` becomes `(17.875,606)`, while shard1 remains
`(13.375,9684)`. Rank1 subsequently consumes token606: its step2 final hash changes,
then its step3 input/final hashes change. This explains why a summary based only
on the earliest classification would undercount rank1's downstream final-hash
differences. It does not identify why the rank0 maximum changed.

## Provenance and next step

SHA256 of rank0 log:
`e5c24887523ca448f9f9fe4b6360cd79bc91190d21c9e522c843bb31ba9ae75a`

SHA256 of rank1 log:
`1e609c5bb41597f173c5ce50f23b610bcf10217bc3372780b963d14eef647201`

SHA256 of manifest:
`86178034ba32db67f017e01a57bf20a944c5f1719b60abdbec1f64b0673acc0f`

SHA256 of analysis JSON:
`41e52d3c4f75acd3d615958d7e7fa2d2344cc20712da6a958270003b37f1c8e4`

Next: read-only audit of actual proposer body configuration, private primer/
repair state, stream/scratch ownership and final norm geometry. Select the
smallest bounded internal boundary probe only after that audit; no broad reset,
precision change or extra diagnostic GPU I/O is authorized by these observations.
