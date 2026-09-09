# Live three/four-owner MTP verification

2026-09-09, production source `a069efc3`.

## Implementation and safeguards

The scheduler now selects all ready three/four owners in canonical physical
order and sends one E7 verification transaction. Actual worker reception
reconstructs and compares every slot/generation/attempt/token receipt before
target writers. Every checked selection precedes the fixed verdict, and every
owner commit precedes any emission or scratch-reusing proposal. Post-header
failures terminally fail the retained session. E6 pair and scalar cold/drain
paths remain unchanged.

Activation is separate and explicit: `ATLAS_GLM_OWNER_VERIFY=joint` selects
wider verification; absent or literal0 keeps the established pair path.
Capacity alone never selects the new path. Shared immutable preflight checks
actual backend, topology, layers, host capacity, target cache budget and
workspace geometry. Separate three/four-owner commit/acceptance counters and
success-only rank-local E7 telemetry make native execution observable.

## Source verification

Evidence directory:
`/home/abc/storage/models/atlas-campaigns/20260909/glm-native-controller/`.

Actual runtime RED preceded cold selection, live transport and scheduler
implementation (`owner-policy-red.log`, `owner-transport-red.log`,
`owner-scheduler-red.log`). GREEN covers actual Model/worker E7 and E1
continuation, noncontiguous C3/C4, two scheduler rounds and cold/pair/scalar
drain. A test-oracle collision between equal80-byte read lengths was corrected
to check actual logits addresses; production behavior was not changed for it.

Clean committed-source reruns:

- `owner-live-a069efc3-model.log`:221 filtered model tests pass.
- `owner-live-a069efc3-scheduler.log`:16 selected scheduler tests pass.
- Workspace formatting, whitespace, new-file size and SPDX checks pass.
- `owner-live-clippy.log` retains the same ten pre-existing non-test deny
  errors as `owner-composed-clippy.log`; this is not a Clippy-clean claim.

These host fixtures do not prove native arithmetic or generated quality.

## Native candidate provenance

Source archive SHA256:
`e37d04d862f1b286c34e645b7b2a8b498d60eec0bed9288968e241d8f11e861c`.
Every changed deployed source was hash-checked. The bounded8GiB/CPU0..1 native
build exited0 after2m19s; no stub kernel build was selected.

- Server ELF: `4247393f38723cffd9e6133f855ed8667de322099ef22533e0a9eaedcf5916c6`.
- Head image: `e8afb24e836db7213ecad092e487d86b50dbf6ab90a562a8041d392847c18167`.
- Worker image: `8c2fa15cc174e48877cd59244077af494de566eb777848d37674b90f11d7be39`.

Guard, relay and supervisor hashes remain unchanged from source6e1e37f4.
Images are retained as `atlas-glm53-flash:paired-a069efc3`; previous images
remain available. Resource caps, TP2, context2044, prefill1024, capacity4,
MTP4, original numerical policy and4GiB available-memory guard are unchanged.

`materialize-owner.py` produces same-image OFF/ON recipes. Here OFF means
wider verification disabled **with JointSharedM10 pairs still enabled**, not
nonspeculative decoding. Two literal0 diagnostic settings are omitted to keep
the128-environment-entry bound; both corresponding diagnostics enable only
literal1, so their effective behavior remains OFF.

The pinned `native-workload-c4.py` first checks four concurrent coherence
answers, four actual structured tool calls, and four distinct needles at
768/800/832/864 tokens. It then runs the unchanged148-input/256-output coding
workload at C1..4, one warmup and three measured waves per width. Complete
retained outputs are compared with the prior reference; capped code is not
executed or certified as a complete program.

## First native A/B

Same source/images and workload, fresh process per mode. Warmup excluded;
medians of three measured waves. Values below are aggregate full-wall tok/s.

| Concurrency | M10-pair control | Wider E7 | Change |
| --- | ---: | ---: | ---: |
| C1 | 27.352 | 27.231 | -0.4% |
| C2 | 37.010 | 36.866 | -0.4% |
| C3 | 33.070 | 42.248 | +27.8% |
| C4 | 36.883 | 45.526 | +23.4% |

Wider decode-window medians:28.650/38.173/43.338/46.500tok/s.
Client TTFT medians:466.611/682.771/893.669/1103.768ms;
server TTFT:427.463/425.018/425.503/426.377ms. The client metric includes
queueing and transport; it is not interchangeable with the server metric.

Both modes pass every bounded quality check and all40 coding streams equal
the retained256-token reference. Both ranks log actual E7 commits for3/4
owners only in the wider mode. Both runs finish normal quiescence/release
and independently inspected rank exit0/OOMKilled=false. Zero swap observed.
Minimum available memory (head/worker KiB): control10,177,824/9,854,712;
wider10,187,520/10,464,592. The4GiB floor was not approached.

Evidence: `native-summary-a069efc3-owner-{off,joint}-first.json`, matching
head/worker logs and inspect files, and
`native-a069efc3-owner-{off,joint}-first-output-comparison.log` in the evidence
directory. Full controller receipts are the corresponding
`native-prepared-a069efc3-c4-owner-{off,owners-joint}-first` directories.

## Fresh-process wider repeat

The identical candidate and workload pass a second fresh process:

| Concurrency | Full-wall tok/s | Decode-window tok/s | Client TTFT ms | Server TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| C1 | 27.258 | 28.668 | 466.959 | 429.206 |
| C2 | 36.874 | 38.189 | 687.999 | 426.051 |
| C3 | 42.347 | 43.452 | 885.373 | 424.615 |
| C4 | 45.745 | 46.710 | 1101.547 | 426.308 |

All four full-wall medians are within0.5% of the first wider process. Every
coherence/tool/needle check passes again; all40 retained coding streams match
the first candidate and control. Both ranks independently log successful C3/C4
E7 commits. Normal quiescence/release and both-rank exit0/OOMKilled=false are
confirmed. Minimum head/worker available memory:9,787,076/9,692,868KiB;
zero swap observed. See the matching `owner-joint-repeat` summaries, logs,
inspect and output-comparison files alongside first-run evidence.

The three campaigns ran sequentially, never simultaneously:

- Control:19:15:32–19:22:32UTC.
- First wider:19:23:20–19:29:42UTC.
- Wider repeat:19:30:33–19:36:49UTC.

Both nodes were independently inspected idle after the repeat, with about
118,700MiB available as reported by `free -m` and zero swap used.
The narrowly scoped temporary helper sudoers rule was withdrawn on both nodes
to recoverable backups under `phase7/paired-a069efc3/`; immutable helpers,
images, source, binaries and receipts remain available. No reset/reboot or
driver/clock change was performed.

C4 exceeds the reference's
numeric43 target on this internal short-context workload, not a proven matched
reference workload. It remains below the earlier nonspeculative C4~47.3.
C1/C2 strict repeat-qualified targets and concurrent C6/C8 MTP remain open.
Wider C6/C8 requirements are in `mtp-c6-c8-extension-plan.md`; reference-workload
parity, comparable serving capabilities and fresh large-context qualification
also remain open.

## Follow-up profiling hypothesis

Source review found three whole-layer static validation traversals per E7
head transaction and two on the worker; E6 does not repeat these cold loops.
KDA also repeats identical FFN resource/environment checks per owner, and
capture validation reaches the driver. This is a hypothesis, not measured
cost. A future model-bound geometry/handle/policy certificate could remove
repetition only if actual dispatch consumes the same frozen policy. Keep all
live identity, lease, state, budget, metadata, alias, capture, health and
verdict checks. Do not remove guards merely because the first native run passed.
