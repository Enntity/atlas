# EXL3 reference runtime-mechanism probe — result (2026-10-15)

Diagnostic result for the [registered probe](exl3-runtime-mechanism-probe-registration.md)
under the [runtime-mechanism plan](../plans/exl3-reference-runtime-mechanism-plan.md).
This is reference-only evidence, not qualification. It does not change
gate v2, its H3/H4 constants or the UNVERIFIABLE campaign result. No probe
sample is a qualification draw.

Evidence root: `$EV/probe-runtime-mechanism-20261014/`
(`$EV=/home/colum/Workspace/atlas-exl3-validation-20261010`). Retained on the
Sparks: per-attempt cache copies and telemetry under `/home/colum/probe-rm-20261014/`.
The telemetry is also archived locally as `attempts/<a>/rank<r>-telemetry.tar`.

## Attempts (fixed order, all reported)

| Attempt | Arm | Package | Status | Notes |
|---|---|---|---|---|
| off01 | off | `9b6b2be8…` | complete | both ranks exit 0; NCCL live re-bind recorded |
| on01 | on | `9b6b2be8…` | **failed** | probe flush filename collision in engine warmup; retained, not retried |
| on02 | on | `c9745145…` | complete | both ranks exit 0 |
| off02 | off | `c9745145…` | complete | both ranks exit 0 |
| off03 | off | `c9745145…` | complete | both ranks exit 0 |
| on03 | on | `c9745145…` | complete | both ranks exit 0 |

Every attempt ran from its own copy of one frozen per-rank cache snapshot.
Each copy hashed equal to the snapshot before start and was unchanged
after shutdown, so no attempt wrote new vLLM, Triton, TileLang, nv or EXL3
tune-cache content. Pre-collection corrections 1–6 and the on01 record are
in the registration. Executed EXL3 launch plans were **not** observable:
nsys does not trace vLLM's forked workers (correction 6).

## Findings

**1. Neither failing singleton recurred, and no draw left the campaign
envelope.** Each completed probe draw was checked against the 15 campaign
references with gate v2's own H3 rule, descriptively, as if it were a
held-out draw (`h3-descriptive.json`). All five pass all 56 gated items.
The worst item is 0.155 of its limit (`on03`,
`index-boundary-2048/r1/ce_error`); every other draw's worst item is at
most 0.097. For comparison, v2ref14's campaign excursion on
`short16/longer/r3` was 9.47× the limit. At `short16/code/r1` all five
draws join the campaign majority byte class; v2ref15's singleton class did
not reappear. At `short16/longer/r3` no draw joins v2ref14's singleton
class (`classes-final.json`). With 5 draws, a mode at the campaign's
observed rates of 1/15 to 3/15 could easily be absent. This is "no
recurrence, mechanism of the excursions unresolved", not elimination.

**2. Byte-level variation is pervasive and not specific to an arm.** Across
the campaign's 15 draws, every long-suite row is byte-unique. Short16 rows
have 4–17 classes. The probe draws in both arms keep producing new byte
classes: each draw joins an existing campaign class on 7–14 of 32 rows.
As a baseline, each campaign draw shares a class with another campaign
draw on 6–11 of 32 rows. The probe draws fall in the same band, so
replacing the observer package at the same mount shows no byte-level
effect.
On/off neutrality is scoped accordingly: the descriptive H3 ratios
overlap between arms (on worst items 0.071 and 0.155; off 0.063–0.097),
and no telemetry-induced systematic class is visible. With two on draws,
this does not show distributional neutrality.

**3. Routing ties are a consequence, not the origin, of divergence between
two runs.** Engagement for on02 and on03 is complete. On both ranks, every
one of the 32 target requests has exactly 42 × (prefill chunks) captures
of gate logits, top-k weights and top-k IDs, with 42 router installs and
0 coverage violations. Comparing on02 with on03 per request and per layer
(both ranks agree exactly):
- the three rows whose final logits are byte-identical in both draws
  (`code` r0–r2) have zero router differences;
- every row with differing final logits also differs in routing;
- in every differing row the router **input** (gate logits) differs first,
  at a row-dependent layer between 8 and 33. Expert IDs first change at
  that layer or up to 4 layers later. Strict check over every
  (request, step, layer): there are zero cases where top-k IDs differ while
  that layer's gate logits are byte-identical, on either rank.

Gate logits are byte-identical through every earlier routed layer. The first
difference therefore enters between the previous layer's MoE gate and this
layer's gate. That span is the previous layer's expert computation and
its reduction, then this layer's attention, its all-reduce and the norms.
The entry depth varies by row and draw. That fits a nondeterministic or
run-varying numerical choice inside the layer stack. It does not fit a
fixed defective layer or a discrete routing tie-break.

**4. The vLLM collective dispatch sequence is identical between draws.**
on02 and on03 issue the same 26,497 collective calls on each rank, with the
same op, tensor metadata, backend (PYNCCL / PYNCCL_GATHER) and status. The
per-tensor backend selection is not the varying choice. NCCL's internal
algorithm/protocol choice is not observable with this telemetry.

## What this leaves

Remaining mechanism candidates, none observed directly: EXL3
cooperative-kernel plan choice in memory (disk tune state was identical
and unchanged throughout), nondeterministic accumulation in EXL3 or
attention kernels, and NCCL algorithm/protocol variation. Distinguishing
them needs telemetry this probe does not have:
- a separately registered observer that sees kernel launches in forked
  workers, e.g. a CACHEDEBUG `exllamav3_ext` build with neutrality checks,
  or tracing with spawn-started workers as its own declared profile;
- or a per-layer hidden-state capture around the first differing layer of a
  differing pair, now that this probe locates that layer per row.

The plan's one-variable interventions were conditioned on observations. The
grouped-top-k switch is not supported here, because routing flips follow
gate-input differences. A collective-backend change is not supported,
because dispatch is identical. An EXL3 plan-freeze intervention remains
untested.

Qualification is unchanged. The fixed v2 campaign stays UNVERIFIABLE. A
reference-pinning amendment still lacks a demonstrated mechanism.

## Analysis artifacts

- `classes-final.json`: byte classes of every short16/long row across
  v2ref01–15 and the five completed probe draws
  (`tests/exl3_validation/probe_classes.py`).
- `h3-descriptive.json`: per-draw descriptive H3 against the 15 campaign
  references (`statistical_gate_v2.evaluate_h3`).
- `route-diff-on02-on03-rank{0,1}.json`: the router layer-by-layer
  comparison of on02 and on03 (`tests/exl3_validation/probe_route_diff.py`,
  run on each Spark against the retained telemetry). The collective
  sequence comparison used the same records. They use the
  `ctx` (layer, invocation, request) bindings and file hashes in each flush
  window's `records.json`. Prefill chunking makes the invocation index
  cumulative per process, so comparisons key on request, layer and
  per-request order.
