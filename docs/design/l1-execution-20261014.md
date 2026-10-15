# L1 collection and observer guard, 2026-10-14

The fresh L1 attempt collected all 32 prescribed requests and released both
hosts, but its **reference observer guard failed**. Candidate control and
observed logits match on all eight rows; reference control and observed
logits match on only one. No L1 attribution or numerical fix is established.

## Completed and preserved attempts

`l1-refexact-f099-01` completed both candidate windows and all eight
reference-control requests. Candidate observer neutrality passed. The
reference worker then received Docker stop SIGTERM followed by timeout
SIGKILL, exiting 137 without OOM. Its strict exit-zero guard rejected
teardown; no reference-observed window ran. The attempt remains failed.
Bounded Docker events and inspections diagnose the controlled shutdown;
they do not relabel the original receipt.

Correction `1204a210` adds explicit stop settings and an evidence-bound
reference-worker timeout policy. Candidate and reference leader still
require exit zero. A worker timeout exit requires a complete finite,
source/head/request-bound eight-row collection, both-rank attestations,
exact ownership, non-OOM states and bounded local container events proving
the registered TERM-to-KILL timeout sequence. It is recorded separately
from clean exit zero. CPU validation at this source freeze passed 612 tests
and 278 subtests; independent targeted review passed 104 tests.

Fresh attempt `l1-refexact-f099-02` uses the separately registered plan
`febca5ecce8b48cb8d8f6f8dd587408265a8a815921e63e005fa99ad5b52f48a`
in `q2-c2-worker-binding-20261014/l1-reference-exact-shutdown-v2/`.
The plan binds 110 inputs, the original failed receipt and its three
lifecycle fingerprints. Runtime images, numerical settings, reference v5
capture source and selected reference-exact C2 fixtures are unchanged.
The new explicit stop timeout is 30 seconds.

All four fresh windows completed eight requests each, with empty
teardown/retrieval error lists and **exit zero on both ranks**. No forced-stop
fallback was needed. The final host inspection shows no GPU owners or
running experiment containers. Failed and completed containers are preserved.

## Observer result

The unchanged production observer verifier checks schedule, token identity,
profile, text, thresholds and actual raw row/payload hashes before comparing
final-logit bytes. Text and thresholds match for all eight requests.

| Engine/case | Byte-identical rows | Different rows |
|---|---|---|
| candidate, arithmetic | 0–3 | none |
| candidate, longer | 0–3 | none |
| reference, arithmetic | 2 | 0, 1, 3 |
| reference, longer | none | 0–3 |

The receipt records `observer_passed=false`, `attribution_authorized=false`
and `L1 observer controls differ; attribution withheld`. The final paired
`binding.json` was not created. The strict L1 bisect was not invoked.

The candidate control also reproduces all eight selected C2 treatment rows.
Independent Luna review verified its actual source/provider, container,
shutdown and raw-data receipts, then verified the observed candidate's
544 sidecars, 896 payload extents and 64 mHC request bindings. Candidate
precision labels vary by row size and remain part of the recorded boundary
contract. Reference-control review verified all eight source-bound head
witnesses, both owned GPU worker provider attestations and clean shutdowns.

The [compact evidence](evidence/l1-execution-20261014/summary.json) binds
both attempts, every fresh manifest/lifecycle/shutdown receipt, the observer
rows, CPU record and final host inspection. Full evidence remains under
`/home/colum/Workspace/atlas-exl3-validation-20261010/q2-c2-worker-binding-20261014`.

## Next requirement

Establish reference observer verifiability before causal L1 analysis. This
single control/observed pair cannot distinguish intrinsic reference variation
from capture effects. A fresh study must pre-register its attempt order,
source/profile identity, engagement checks and interpretation rules before
collection. Retain every draw; do not retry unchanged for a byte match or
introduce a tolerance after inspecting these differences.

This guard failure is separate from the existing fresh-reference H3/H4
qualification failures. Both remain unresolved. The
[C2 result](c2-repaired-worker-execution-20261014.md) remains mixed, no
production numerical fix is qualified, and no image was published.
