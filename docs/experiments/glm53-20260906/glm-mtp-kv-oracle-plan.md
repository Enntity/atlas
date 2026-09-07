# Resident-weight GLM MTP KV writer oracle

This diagnostic is default-off (`ATLAS_GLM_MTP_KV_REPAIR_VERIFY=1`) and is
not a benchmark. It uses the existing BF16 embedding/normalization/EH/MLA
KV-only chain, not the NVFP4 full proposer. No new GPU allocations, weights,
collectives, cursor updates, or graph execution are introduced.

Before any overwrite, validate both the real arbitrary destination and a
logical-row-zero reference destination, owned source spans, exclusive blocks,
eager BF16 NoPE512 geometry, and KV-only body capability. Limit each check to
one through four rows, block size 16, and at most three distinct touched
physical blocks. Larger prompt-primer writes retain their existing path.

Take host snapshots of the complete K and V sides of those blocks (at most
96 KiB of original bytes). Run a frozen copy of the pre-extraction BF16 chain
at logical row zero, reading the same already-shifted token IDs and owned
normalized hidden rows. Read its output rows and check every non-reference
byte against the snapshot. Restore all original blocks before the real writer
runs. Compare all real output rows bitwise with the reference, and every
untouched byte in the snapshot set with its original. No scalar NVFP4 proposal
comparison or independent mathematical accuracy claim is made.

Any reference/candidate error or mismatch triggers best-effort restoration of
every captured block, preserving the original error and reporting restoration
errors too. Do not publish proposer state after a failed check. A CUDA failure
can prevent restoration: the caller must fail the request coherently rather
than assume that the GPU state remains usable. Scratch is intentionally not
restored; hidden sources and the saved bonus must already be outside it.

Test first with MockGpuBackend: full K/V byte equality and guards for
cross-block/shuffled/overlapping reference destinations; reference and candidate
errors; mismatches; malformed ranges, aliases and unsupported capability fail
before writes; unchanged allocation count. Mock tests prove transactional
addressing/restoration, not CUDA numerics. Root owns all resident-weight GPU
validation and will review before enabling the diagnostic on either rank.

The only shared API addition is a read-only `TransformerLayer` KV-only
capability method (default false, NoPE MLA attention override). The writer
checks it before scratch work or primer block allocation. The diagnostic
currently checks every eligible 1..4-row call rather than caching shape-level
results; memory remains bounded, but diagnostic timings are not comparable
with serving timings.

CPU evidence: initial missing-implementation RED, followed by a real boundary
failure (12 passed, one rejected-span test failed). Explicitly checking both
pool ends fixed an overflow hidden by overlap short-circuiting; all 13 focused
KV tests then passed, including the independent repair adapter's two tests.
Final combined CPU gate passed 771/771 model tests, including all 14 KV tests
and the new restoration-copy-failure case. The independent reviewer also reran
all four transaction-oracle tests (4/4). Receipts are
`/tmp/atlas-glm53-phase6-20260907.J5PkkO/gu-m16-kv-repair-combined-model-cpu.log`
and `kv-oracle-final-cpu.log` in that directory. An actual-stream capture check
also rejects diagnostic use before any synchronization. No GPU result is
claimed here.
