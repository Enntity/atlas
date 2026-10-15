# C2 worker binding repair and validation, 2026-10-14

The distributed worker request-ID repair is committed as `f099ac58`.
The newly built image is **diagnostic and unqualified**. Full short16/long
neutrality passed; both repaired-image C2 variants completed with passing
controls and mixed results. See the [execution report](c2-repaired-worker-execution-20261014.md).

## Runtime change

The shared Model interface now transports bounded F8 request metadata before
prefill. Workers bind it to the addressed sequence and preserve it through
`ForwardContext`. Absent, empty and UTF-8 IDs remain distinct; an explicitly
absent ID clears a stale binding. Invalid extents, padding and UTF-8 fail before changing
the sequence. The numerical F0 prefill payload is unchanged.

This changes the shared distributed control protocol. Both peers must run the
matching image. CPU tests cover the actual worker F8-to-F0 dispatch, context
binding and scheduler send/replay paths. The model suite at the committed
runtime source passed 1660 tests, with 17 ignored. The final Python harness
suite passed 528 tests and 244 subtests at `3f48312a`.

Repository-wide clippy remains blocked. The lint findings were not exhaustively
baselined, so this record does not claim a clean workspace lint result.

## Image binding

| Field | Bound value |
|---|---|
| Runtime commit | `f099ac588494677257d70cd404182e73d6a72d62` |
| Build snapshot | `4ee96c12b44c23b3a713d2283d29b2e1cac5cec937b512011c93868ce244a21a` |
| Image tag | `atlas-exl3-validation:src-4ee96c12b44c-rust1.97.1` |
| Both-rank image ID | `sha256:82173ea631a8c78e4fb565f0e23bc6d55f94a43090dbfb27c850f8cc1e6b41d9` |
| Packaged server SHA256 | `140e16915a6fc19a655ebe63dccb8599cf3fdff5178a7a2afaf10ae10378f29a` |
| Build manifest SHA256 | `dc80f4155499254c55610c31a0c46923980adfd55f9d2932991dbbd4fa9c6ab3` |

The no-cache build passed. Both packaged executables and their 232 embedded
PTX modules were checked against the manifest. Toolchain, compiler closure,
PTX hashes/lengths, all 136 recorded runtime packages, all four recorded
runtime libraries and cuBLAS pins match the prior collected image exactly.
Both hosts loaded the same repaired image ID. Runtime provider mappings are
attested separately after each lifecycle reaches readiness.

The local and build-host archives differ only in `assets/atlas-demo.mp4`:
expanded Git LFS contents locally, the pointer on the build host. All other
6124 entries match. The video is outside the Docker COPY closure. A first
build using the expanded archive's label was cancelled; the completed build
uses the actual build-host snapshot above. Both logs remain preserved.

## Neutrality collection

`worker-binding-neutral01` completed all 16 short rows, with byte-identical
logits and matching generated text against frozen `v2cand01`. It also completed
seven long rows; separate raw-row checks find all seven byte-identical.
The overall remote command deadline had been inherited from the shorter C2
configuration and could not cover the full long schedule. The operator
interrupted the run, preserved its partial data and released both hosts.
Its lifecycle reports zero teardown/retrieval errors. It has no full proof.

A second preparation was rejected before GPU launch by the explicit budget
check. The frozen long suite's request timeout is 1800 seconds, whereas short16
uses 1200. The initial administrative audit had incorrectly attributed 1200
to both; the correction retains the original records and identifies that error.

The fresh `worker-binding-neutral03` uses a budget derived from the actual
frozen suite values and validated before launch. All 16 short and 16 long rows
match the frozen baseline exactly, including generated text. Both container
wait receipts report exit 0 and teardown/retrieval errors are empty. Teacher-forced token prefixes,
numerical profile, metrics and thresholds are unchanged. This is an administrative restart of a
diagnostic neutrality collection, not a replacement qualification draw.

The full proof requires all 32 raw captures, byte-identical logits and text,
continuous head indices 0–31 in one lifecycle, both actual binary hashes,
both image identities, pinned cuBLAS mappings and successful stop/wait receipts.
The archived proof passes all these checks and independent frozen-verifier
review. Its SHA256 is
`52b8f2de1fc300d7b2e194c273afaf5817d505adbbe127c5a20f5f3d5e64284f`.
The fresh C2 preparer revalidated it before allowing intervention.

## Evidence and next step

Durable root:
`/home/colum/Workspace/atlas-exl3-validation-20261010/q2-c2-worker-binding-20261014`.
Relevant records are `build-binding.json`, `image-fingerprint/verification.json`,
`image-fingerprint/scratch-verification.json`,
`image-fingerprint/runtime-comparison.json`, `image-transfer.json`,
`full-cpu-harness-validation-final.json`, `neutrality-restart.json`,
`neutrality-budget-audit-correction.json`,
`neutrality01-partial-long-row-checks.json` and
`neutrality03-controller-budget-validation.json`.

The completed proof is
`neutrality-bounded03/worker-binding-neutral03/neutrality-proof.json`. Fresh
C2 preparation is `c2/`, from `c2-preparation-config03.json`; all 52 registered
artifacts independently rehash. Both `c2-refexact-f099-01` and
`c2-v24-f099-01` completed; each passes all 32 native local o_proj comparisons,
baseline/observer controls and repeated byte-determinism. All nine reference
members are scored. Both variants produce mixed arithmetic results and leave
all four `longer` rows unchanged, three still catastrophic in KL. The C1
first-measurement flag is discharged for this bound slice. Inherited diagnostic annotations referring to the historical image
and “no runtime rebuild” describe the prior campaign; authoritative engine,
preparation and execution bindings identify the repaired runtime above.
The reference H3/H4 verifiability failures still block Q2 qualification.

L1 CPU readiness was separately repaired in `b49e2e71` and `3f48312a`: fused
phase pairing, materialized residual mapping, actual collection envelopes and
hash-pinned opaque reference GPU IDs. Its 61 focused tests pass. If C2 leaves
residual divergence, L1 still requires fresh, source-bound two-engine collection
and observer evidence; the readiness audit is `l1-collection-readiness.json`.

No production numerical fix or published image is claimed.
