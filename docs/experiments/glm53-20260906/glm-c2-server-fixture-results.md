# Actual paired Model fixture in server tests

2026-09-09. Code `d08daf41fe7d48c44f79f9636e49c0983f8809c0`.
Root's exact post-commit model suite passed **1100 / 1100** (86.11s, four test
threads); server passed **2366 / 2366**, 12 ignored (24.32s, one test thread).
Final root formatting/whitespace checks passed and all18 frozen hashes matched.
Author precommit measurements are identified separately below.

The non-default `glm-c2-test-utils` feature exposes the existing bounded CPU
fixture to server tests. It reuses one actual TransformerModel constructor,
genuine allocated SequenceStates, sealed paired capability, numerical-boundary
fixture and local Wire implementation. No Model adapter, Ready setter, factory
selection or serving caller was added. The ordinary server dependency remains
feature-off; its test-only dev-dependency explicitly enables the seam.

The server smoke calls actual prefill/bootstrap, then records selected E1/F5
and replays those packets through the actual rank-1 worker. Both owners and
execution orders are exercised. Genuine states move into the existing ActiveSeq
initializer. Actual K5 logits feed the checked five-row selection helper; output
tokens and canonical token/length fields remain unchanged during selection.
Public acceptance record, trim, commit and retirement complete before cleanup.
This E1/F5-only local replay does **not** qualify worker F0/cold-prefix transport,
concurrent NCCL, numerical GLM arithmetic or a selected scheduler driver.

The intended RED was the server's missing-facade import (E0432), a compile-time
exposure test, not a runtime inference bug. Two subsequent test-control errors
are retained: a private broadcast method was replaced with the existing public
Model method; an incorrect primed-prefix expectation was corrected to actual
private cursor P-1. The corrected test preserves that prefix and explicitly
checks the missing-tail and seed K/V bytes against detached H[P-1] and H[P].
No production producer or numerical handler changed to satisfy either control.

Author precommit frozen-source qualification:

- Server focused **3 PASS** (0.20s), including foreign/revoked/released snapshot
  refusal and actual legacy capability absence; existing model handoff **120
  PASS** (3.68s), retaining original subprocess test names.
- Separate model feature-OFF/ON library checks and ordinary feature-OFF CUDA
  server binary check passed. An external dependency probe rejects the facade
  without its feature and imports it successfully with the feature, without
  dependency `cfg(test)`. Engine Cargo.lock is unchanged.
- Full serial model **1100 PASS** (198.90s); full serial server **2366 PASS**,
  **12 ignored** (24.37s). These CPU times are not inference measurements.
- Formatting, whitespace and scoped SPDX checks passed. New files are <=500
  lines; existing allowlisted scheduler/mod.rs is 1164 lines (+2 registration).
- Clippy is **not green**: no-CUDA model checking stops on four inherited
  runtime Metal-stub argument-count errors; CUDA server checking reaches nine
  existing model lints across eight unchanged files. No suppressions were added.

Root and independent reviewer approved the 18-path manifest SHA-256
`340da10b4206aacbca6270b6c32bc50a944c1a468bc2f75d8e318bd38a76ee6d`.
Raw receipts and commands are under
`/home/abc/storage/models/atlas-campaigns/20260908/glm-c2-server-fixture/`.
See `hygiene.txt`, `qualify.sh` and `postcommit-closure.md` for exact distinctions.
Receipt archive: `glm-c2-server-fixture-d08daf41-receipts.tar`, with detached
SHA-256. It contains source, external compile-probe inputs and receipts only.
No native image, GPU execution, throughput, admission or terminal-supervision
qualification follows from this testing seam.
