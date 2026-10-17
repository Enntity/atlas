# EXL3 reference verifiability diagnosis — 2026-10-14

The fixed qualification campaign remains **UNVERIFIABLE**. Offline replay
reproduces both reference failures exactly. The failing rows have singleton
byte classes outside their leave-one-out fits. Recorded configuration, capture
context and cache contents do not identify the runtime choice producing them.
A separately registered reference-only pinning experiment is justified;
changing qualification policy or thresholds is not yet supported by causal
evidence. No new GPU samples were collected.

This is a retrospective, single-campaign diagnosis, not numerical qualification
or an estimate of the whole-campaign error rate. The
[frozen protocol](exl3-statistical-gate-v2.md) and
[qualification verdict](exl3-statistical-gate-v2-qualification-20261014.md)
remain authoritative.

## Identity and reproduction

`excluded_reference` is a **zero-based index**, not an attempt suffix. H3 index
13 is `v2ref14`; H4 index 14 is `v2ref15`. Earlier shorthand “reference 13/14”
must be read this way. Both IDs are explicit in the new evidence.

The input campaign is
`/home/colum/Workspace/atlas-exl3-validation-20261010/gate-v2-qualification-20261013-luna1`.
The analysis reads all 15 reference short-suite artifacts through the current
parity loader and gate functions, plus reference long-suite manifests, attempt
records, cache receipts and failing-row head-position sidecars. It reads no
candidate, held-out or perturbation arrays. H3 uses reference 0 as its API's
placeholder candidate; the report retains only reference leave-one-out results.
H4 receives an empty candidate list. The short-suite replay checks 28 H3 items;
the complete campaign checked 56 across both suites. This analysis does not
re-evaluate long-suite numerical verifiability.

Reproduce from the repository root with a new output path:

```bash
python3 docs/design/evidence/gate-v2-reference-diagnosis-20261014/analyze.py \
  --root /home/colum/Workspace/atlas-exl3-validation-20261010/gate-v2-qualification-20261013-luna1 \
  --out /tmp/atlas-reference-diagnosis-new.json
```

[diagnosis.json](evidence/gate-v2-reference-diagnosis-20261014/diagnosis.json)
contains the 105 pairwise symmetric-KL distances for each failing row, full and
leave-one-out mode fits, all byte classes, chronology, provenance, cache
signatures, capture context and input/evaluator hashes.
[summary.json](evidence/gate-v2-reference-diagnosis-20261014/summary.json)
binds exact H3/H4 helper sources, the frozen protocol, configuration, verification
and the diagnosis output. Two offline executions produced byte-identical JSON;
the focused v1/v2 gate fixtures passed 24/24. No production or evaluator logic
was changed.

## H3: `short16/longer/r3`, index 13 (`v2ref14`)

There are six FP32 byte classes. In zero-based index order their memberships
are `[0,1,2,3,5,10,12]`, `[4,7,11]`, `[6,9]`, `[8]`, `[14]`, and `[13]`.
The full fit demonstrates two modes with sizes 14/1 and separation
28.409545; index 13 is the singleton. Its row SHA256 is
`b0a8b895d1643938270cd842282758515841b95ac2d881dcdf70802a8ecdf184`.

Excluding index 13 leaves a symmetric-KL spread of `0.001790604089123014`.
The unchanged H3 limit is `0.005371812267369042` (three times spread, exceeding
the `0.005` floor). The excluded row's nearest fit member is index 6
(`v2ref07`), at `0.05087024759113458`, or 9.469848 times the limit.
This exactly matches the corrected qualification report.

The remaining 14 draws do not demonstrate modes: the largest positive-distance
gap ratio is 2.034195, below `G_MIN=4`. H4 membership is consequently
inapplicable for that exclusion; H3 is what blocks it. All 15 argmax tokens
are 4695. The nearest-KL pair differs at 150831 of 154880 float entries,
with maximum absolute logit delta 1.546875. Identical argmax does not establish
logit parity.

## H4: `short16/code/r1`, index 14 (`v2ref15`)

The four FP32 byte classes have memberships `[0,2,3,4,5,8,9,10,12,13]`,
`[1,11]`, `[6,7]`, and `[14]`. The full ensemble does **not** demonstrate
the required two modes: the largest gap ratio is 11.468153 but produces three
components. The zero-based index 14 singleton's row SHA256 is
`356d53bae1f660edca75af056ce74f63e0343586310d7f6d250c50c9ba0d4a91`.

Excluding it produces demonstrated modes of sizes 10/4, with separation
29.303669. The nonzero within-mode radius is `0.000011911830450968014`, so
the unchanged membership limit is `0.000023823660901936028`. The excluded
row's distances to the two canonical centers are `0.00027333285218177154`
and `0.0001366066907714272`; the nearer center is 5.734076 times the limit.
This exactly matches the corrected qualification report. This is a positive
radius check, not the exact-byte or zero-radius ULP special case.

H3 passes that exclusion: the nearest distance is well below its `0.005`
floor. All 15 argmax tokens are 488. The nearest-KL pair differs at 148833
float entries, with maximum absolute logit delta 0.5234375. The evidence also
reports nearest-by-maximum-absolute-delta and ULP diagnostics separately;
those need not select the nearest-KL reference, and ULP values across sign
changes should not be interpreted as a small perturbation count.

## Recorded runtime and chronology

All 15 references have identical stable provenance in both suite manifests,
including engine/image/build, checkpoint, tokenizer, model configuration,
TP=2/EP=1, BF16 activations, eager execution, disabled speculation
and prefix caching, serve flags, environment, capture hooks, GPU UUIDs and
driver/CUDA/NCCL versions. Normalizing only the registered attempt name gives
identical Docker create commands. The short-suite gate provenance/identity
validators pass. No missing values are replaced with inferred defaults.

The legacy `execution.kv_dtype="bf16"` field is not the effective reference
cache format. The
[vLLM source contract](exl3-vllm-reference-contract.md) binds reference
`--kv-cache-dtype auto` to `fp8_ds_mla`. Preserve this effective format in
diagnostics; treating the legacy label as BF16 would change the experiment.

Both failing-row contexts also agree across all references, after removing
request ID and chunk counter: `computed_before=0`, terminal prefill, capture
enabled, 39 scheduled tokens for `longer/r3` and 20 for `code/r1`, with the
same runner hash and prefix hash. The stored head captures are `logits-000015`
and `logits-000009`, respectively. There is no recorded capture-shape mismatch.

In registered primary order, `v2ref14` was attempt position 3 (fourth), started
2026-10-13 19:22:46 UTC; `v2ref15` was position 18 (nineteenth), started
22:31:44 UTC. They are far apart in collection order. Repeated byte classes
occur both before and after the H3 singleton; the H4 singleton is the last
registered reference. These facts do not establish temporal causality.

The vLLM, Triton, TileLang and CUDA compute cache manifests are identical
before and after every reference. EXL3 tune adds two files during the first
reference (`v2ref06`); all subsequent pre/post EXL3 tune manifests are identical,
including both failing draws and many passing draws. A recorded on-disk cache
change therefore does not distinguish the failures. Cache equality does not
prove identical loaded plans, in-memory dispatch, communication order or
routing decisions. `v2ref06`'s shutdown record lacks `stopped_utc`; the analysis
preserves that record without manufacturing a timestamp.

## Decision and remaining attribution limits

The observed discrete row classes support investigating runtime choices, but
do not identify a kernel, routing tie, plan selection or NCCL algorithm as the
cause. These captures contain final logits and runner context, not per-layer
causal traces or selected-plan/collective telemetry. There is no controlled
one-variable comparison here. The failures cannot be dismissed as measurement
artifacts, model faults or candidate regressions.

Proceed with a reference-only diagnostic plan that freezes one runtime choice
per intervention, records both ranks' actual selected plans/collectives or
routing telemetry, and verifies the intervention engages. Preserve the
checkpoint, observer package, suite/request order and effective KV formats.
Use the vLLM source contract to choose a real, supported mechanism; the current
evidence cannot name a proven pinning switch. Include repeats and an independent
held-out check before claiming the reference distribution has stabilized.

Amendment 2 already permits separate runtime-policy diagnostics and requires a
new pre-collection amendment before such a policy enters qualification. That
route is justified; a concrete policy amendment should wait for diagnostic
evidence. Do not widen `C_RHO`, relax the two-component rule, exempt singleton
H3 failures, increase sample count or choose a favorable class from this data.
Any subsequent qualification campaign must be separately registered and retain
this terminal UNVERIFIABLE result. An unchanged retry seeking PASS would not
resolve the identified attribution gap.
