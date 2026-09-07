# Benchmark output fingerprints

The fixed seed/cap receipts currently retain token counts and timing, but not
an output identity. Different accepted-draft means across repeated identical
requests therefore cannot be separated from different generated continuations.
This does not yet establish a correctness defect.

Add SHA256 of concatenated completion text encoded as UTF8, plus byte count,
to each request and its batch receipt. Accumulate received text chunks and
compute the digest after the existing end timestamp. Do not change the
request, sampler, repetition policy, token counts, timing arithmetic, or
persist complete generated code. This is a text fingerprint, not token-ID
identity or a semantic quality verdict.

First add mocked-stream tests that fail on the missing fields. Verify Unicode,
chunk-boundary independence, changed output identity, receipt propagation and
unchanged timing metrics; then implement and rerun the complete CPU harness
suite. Existing fixtures without fingerprints remain explicitly unknown.
An independent reviewer checks the change before commitment. Native model
requests are root-owned; no native build is needed for this client-only change.
