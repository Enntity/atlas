# Repaired-worker C2 execution, 2026-10-14

Both registered C2 variants completed with passing measurement controls and
**mixed** final-logit results. Layer-0 norm substitution did not remove the
registered `longer` failures; L1 collection completed with its reference
observer guard failing.
This is an eight-prefix diagnostic, not a Q2 qualification verdict.

## Bound execution

Runtime repair `f099ac58` and its no-cache image are documented in the
[worker validation record](c2-worker-binding-validation-20261014.md).
The complete 32-request short16/long neutrality proof passes, including
actual binary/image identities, both-rank provider mappings, continuous
head indices, exact logits/text and successful teardown.

Fresh attempts `c2-refexact-f099-01` and `c2-v24-f099-01` each contain two
baseline and two treatment lifecycles. Both use the original eight forced
prefixes, prescribed request IDs, rank-local fixtures and capture layers.
All prior failed or administratively interrupted attempts remain preserved.

Per variant, the unchanged analyzer verifies:

- 32 archived baseline norm matches and 16 baseline observer logit matches;
- 32 native local o_proj matches against the sound reference `own-output.npy`;
- both arms byte-identical across repeats;
- all 16 substitutions consumed per treatment repeat, with exact payload
  hashes and aligned pre-substitution hashes;
- complete evidence retrieval and teardown.

The native-output check discharges C1's outstanding first-measurement flag for
these fixtures and prefixes. Archived weights and the retracted BF16 replay
are not its evidence. The v24 experiment substitutes its recorded fixture
values; it does not prove production v24 kernel engagement or qualification.

## Complete result

Each repeat scores every registered row against all nine frozen references
`ref04` through `ref12`, using the unchanged C2 scorer and catastrophic rules.
Both variants have exactly identical row scores in both repeats.

| Case | Rows | Treatment versus baseline |
|---|---|---|
| arithmetic | 0, 1 | Farther by the primary mean symmetric-KL metric |
| arithmetic | 2, 3 | Closer by that metric |
| longer | 0–3 | Final-logit bytes unchanged |

The `longer` rows 0–2 retain catastrophic KL in both arms; row 3 does not.
This is a mixed result, not evidence of a dominant normalization cause or a
general claim that normalization cannot matter. Residual failure satisfies
the registered L1 escalation condition.

Independent frozen-analyzer replays match all bindings, controls and scores;
only each scorer's two evaluation timestamps differ. Durable originals and
replays are preserved beneath
`/home/colum/Workspace/atlas-exl3-validation-20261010/q2-c2-worker-binding-20261014/c2`.
The [compact evidence](evidence/c2-repaired-worker-20261014/summary.json)
binds both full reports and driver receipts by SHA256.

## L1 follow-up

The source-bound helper is committed as `4beb57f9`; its contract is in the
[L1 tooling spec](../plans/l1-escalation-tooling-spec.md). The complete current
workspace CPU harness passed 590 tests and 278 subtests, including separately
reviewed reference probe tooling. No new candidate runtime image is required.

The reviewed preparation is `q2-c2-worker-binding-20261014/l1-reference-exact/`,
with plan SHA256
`fe00bc857d89906bc441a6b9d433b75b1a0f729d0ed65ca2a27bc4f2fa59f293`.
It explicitly carries the completed reference-exact treatment fixtures into
four sequential windows: candidate control/observed, reference control/observed.
The owner inspected both hosts after C2; no running containers or GPU owners
were present. Attempt `l1-refexact-f099-01` stopped after both candidate
windows and all eight reference-control requests completed. Candidate observer
neutrality passes for all eight logits/text, with 272 L0/L1 capture sidecars
per rank. Reference control passed actual source/provider attestations, but
its worker received Docker stop SIGTERM followed by timeout SIGKILL and
exited 137 without OOM. The strict exit-zero guard rejected teardown. The
failed attempt and bounded Docker event/inspection evidence are preserved
in `l1-attempt01-shutdown-diagnosis/`; no reference-observed window ran.
Source-reviewed correction `1204a210` adds explicit shutdown proof and passes
612 CPU tests plus 278 subtests. Candidate and leader still require exit zero;
reference worker exit 137 requires complete finite source/head/request data,
both-rank attestations, exact ownership, non-OOM states and bounded Docker
events proving the registered TERM-to-KILL timeout stop. It is labelled a
forced stop, separately from exit-zero teardown.

Fresh `l1-reference-exact-shutdown-v2/l1-refexact-f099-02` completed under
plan SHA256
`febca5ecce8b48cb8d8f6f8dd587408265a8a815921e63e005fa99ad5b52f48a`.
The original failed receipt and three lifecycle fingerprints remain bound
in the new registration. All four windows collected eight requests, and
every container exited zero without needing the timeout fallback. Candidate
observer matches 8/8; reference observer matches only 1/8, with unchanged
text/thresholds. The guard withholds attribution; no final paired binding
or L1 bisect was produced. Both hosts are released. See the
[L1 record](l1-execution-20261014.md).

Both ranks must attest actual sources and mapped providers before observation.
Reference worker selection is bound to the owned container's actual process.
Rank-local retrieval and exact head-position/response witnesses bind opaque
reference GPU request IDs. Observer logits, text and thresholds must match
before the strict L1 driver may inspect complete residual streams and attention
input. No L1 attribution or numerical fix is claimed yet.

The original reference H3/H4 failures still block Q2 qualification. No image
was published and no performance result is claimed.
