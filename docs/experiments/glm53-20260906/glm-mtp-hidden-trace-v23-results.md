# v23 post-EH hidden-boundary diagnostic

The exact post-EH row agrees at every measured matching step0, while final
hidden rows still differ. This narrows the observed divergence to the interval
after EH and through body/private-state/final-normalization processing. It does
not establish equal private KV, equal loaded body weights, or a root cause.
No throughput result or serving improvement is claimed from this traced run.

## Provenance and complete-log integrity

Root collected six sequential C1 requests:148 prompt tokens,64 requested output
tokens, one warmup plus five measured repeats, all reaching the output cap.
The five measured completion texts have the same SHA256
`1bdda4caa55ae2df2226c4c3c080d7ab23a6ca80ba56488a1de6b75e0ee2f669`.
Source is `1f675c8b`; native v23 SHA256 is
`a5b9e20c11a12bf24d9a2a3fb226b0722cd1507d58df4b248bed1a818e96bed9`,
with unchanged CUDA source base `189db87e` (root build provenance).

All files below are in `/home/abc/storage/models/atlas-campaigns/20260908/`:

- `v23-hidden-clean-rank0.log`:323613 bytes,192 records;
  SHA256 `3a658521b6be13dc65d53c2c9d7cd63b81db2ffd53a62960716fed7773fc71fb`.
- `v23-hidden-clean-rank1.log`:297904 bytes,192 records;
  SHA256 `1d9b9bdbb35b436858599a4eda56f291c121790fc69fa872e14b93119108ea69`.
- `v23-hidden-analysis-manifest.json`: all six slot0/generation1..6 owners
  explicitly declared against the complete two files;
  SHA256 `49d5b984b05802b3c09b4984980ba114599c46dcd9cb6fe77d7a038b75ccea9c`.
- `v23-hidden-analysis.json`: strict version2 analysis;
  SHA256 `3ecaab9fb9082cac421e820956c6088f31499e17e665a7b59fa5d08ce1fe3b11`.
  Independent CPU reanalysis of both complete files reproduces this JSON exactly.

Integrity passes for all384 records:32 steps/request/rank, ordered first8
attempts with steps0..3, cursor/token/hash-chain consistency, no duplicate,
missing or foreign records. Each rank has48 post-EH hashes, exclusively step0;
steps1..3 explicitly lack that boundary. There are no cross-rank metadata
differences, gathered-pair differences or chosen-draft differences within any
request. Hash equality here means equality of recorded SHA256 observations.

## New boundary result

Across ranks, all48 available step0 post-EH comparisons agree. Every one of
those48 also has equal conditioning input hash/token/position but different
final hidden hashes. The remaining144 step comparisons have different input
hashes, as they consume preceding rank-local final hidden rows; all192 final
hash comparisons differ. All192 gathered two-shard pair payloads and resulting
draft tokens nevertheless agree across the two ranks.

Across the nine explicitly selected cross-request comparisons, all66 available
matched post-EH comparisons per rank agree:132 pairwise equality observations,
not132 independently captured rows. All corresponding step0 input hashes also
agree. Later steps were not post-EH sampled and must not be labeled equal there.

Earliest representative key is generation1, attempt1, position149, seed1304,
step0, token1304, cursor148→149, hidden_row0, EH/head NVFP4 both false:

| Boundary | Rank0 | Rank1 |
| --- | --- | --- |
| Input SHA256 | `6c03a73705a818c124de98781562cbaf9fa239cdf3ba860ce4f1f1694ad75965` | same |
| Post-EH SHA256 | `1970b738b46101eed56d8b0989801578c713a42d2fd312d8399da94e03c6ea27` | same |
| Final SHA256 | `ea9724864b0d65f6981ded09e923ec8d5e5f841dbc893179c33ca6a036a58009` | `b2cdc71699edf14687cef52097bb437c42923e7b058d57a380e05a26f9231ec3` |

Both choose49449 from gathered shard0 `(26.625,49449)` and shard1
`(16.625,1307)`. Generation2 has the same input and post-EH hashes at this key;
rank0 final becomes
`c62998c1ef0cb737005a1af53cd7a28f6baaa2e84c75b3559aa5a8add3a263d7`,
while rank1 final remains unchanged. Thus neither the cross-rank nor this
cross-request final mismatch requires a differing observed EH output.

## Raw cross-request counts and the generation3 exception

The table counts raw fields, not just earliest-boundary classifications.
Matched counts are per rank. Input/final columns count unequal raw hashes;
draft counts apply identically to each rank. The shared pair evidence contains
both shards; local maxima from different shards are not expected to be equal.

| Comparison | Matched steps / post-EH pairs | Rank0 input/final differences | Rank1 input/final differences | Different drafts | Pair differences shard0/shard1 |
| --- | --- | --- | --- | --- | --- |
| warmup→repeat1 |32/8|24/32|0/0|0|14/0|
| warmup→repeat2 |24/6|18/24|7/11|9|24/8|
| warmup→repeat3 |32/8|24/32|0/0|0|10/0|
| warmup→repeat4 |32/8|24/32|0/0|0|12/0|
| warmup→repeat5 |32/8|24/32|0/0|0|12/0|
| repeat1→repeat2 |24/6|18/24|7/11|9|24/8|
| repeat2→repeat3 |24/6|18/24|7/11|9|24/8|
| repeat3→repeat4 |32/8|24/32|0/0|0|13/0|
| repeat4→repeat5 |32/8|24/32|0/0|0|15/0|

Repeat2 is generation3. Its three selected comparisons have two unmatched
attempts on each side: the other request's `(position,seed)` keys
`(161,497)` and `(172,1376)` versus generation3's `(160,9878)` and `(166,707)`.
These are changed draft/acceptance trajectories, not missing trace records;
do not extend24-step matched counts to32 or assume common causal history.
All other selected comparisons match32 steps with no metadata differences.

Among the24 matched steps involving generation3, eight have different
hidden_row metadata: position164 has2 versus3, and position169 has4 versus2,
each repeated over four steps. Rank1 also has different final step0 hashes at
these two later positions despite equal input/post-EH hashes. Earlier changed
drafts/private state remain possible contributors; matching input does not
erase that history.

The first changed draft relative to warmup is attempt2, position152, seed284,
step1:957 becomes284 on both ranks. Rank1's input and final hashes at this
decision remain exactly unchanged:

- Input `61914e41b4e7c933b83d938edaf55360b43dbe9f738c56ab78b75431b6b1abe6`.
- Final `2f64b799d87d93504590a1c80b0b7cb4e918ed408f057da25dcd54971a1036e8`.

The shared gathered shard0 maximum changes from `(18.0,957)` to
`(16.625,284)`; shard1 remains `(13.375,9684)`. Rank1 then consumes a different
token at step2, where its input hash still agrees but final hash differs;
its first input-hash difference occurs at step3. This demonstrates why raw
hash, token and pair differences must be reported separately. Post-EH was
not sampled at these later steps.

## Interpretation limit

The measured negative is specific: differing post-EH output bytes do not
explain the sampled step0 final mismatches. The remaining interval includes
the MLA/MoE body, existing private KV contents and final module normalization.
No internal phase or private-KV equality was measured, so no reset, precision,
kernel or synchronization fix follows yet. The source audit's full-head override
and same-stream findings remain source evidence, not native race exclusion.
Additional probes require a separate bounded plan. Root's subsequent quality
checks are separate from these frozen pre-quality logs and counts.

## Root quality, shutdown and recovery receipts

The native build exits0/OOMKilled=false after2m12; both packaged images verify
the executable SHA above. The actual container environments enable version2
hidden tracing, the K5 ledger and shared cache on both ranks. Loaded host
MemAvailable is11,900/12,085MiB, above the4096MiB guard. The initial controller
client invocation fails before HTTP because its selected Python lacks requests;
the preserved environment-error receipt is not a model failure. The successful
six-request run uses `/usr/bin/python3` and starts with generation1.

Four answer checks, the1984-token needle with16 output tokens, cancellation,
and four fresh recovery answers pass. Cancellation receives8831 stream bytes
before the deliberate2s client timeout (exit28). Both model containers stop
gracefully, exit0/OOMKilled=false, and are preserved as
`atlas-glm53-v23-hidden-clean-ep0/1`; both standard serving names are absent.
No standalone GPU test or native compiler overlaps this model run.

Controller and head persistent `phase7/v23-verified-receipts.tar` match SHA256
`fb454581eab2fc9b1d35d51962263e3fd1dc1f25719d1d599422875266f95c4f`.
This includes native/client/quality/stop evidence and the CPU post-EH and
shared/down extraction receipts. The later kernel promotion3332c36e and loader
extractiond6f4e554 are not part of the v23 executable.
