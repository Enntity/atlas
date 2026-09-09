# Selected paired sequence ownership checkpoint

2026-09-09. Code `070848519a2fb824261e13bbf588d52d269ea8e6`.
This is CPU-tested Model ownership infrastructure, not a deployed throughput
improvement. Paired factory/admission and native execution remain disabled.

## Change and actual behavioral evidence

Selected allocation retains the genuine target SlotGuard outside its fallible
initialization body. Actual private/target/addressed identities must agree before
GPU initialization. On postclaim initialization or identity failure the guard
is neutralized without recycling and the paired session latch becomes terminal.
Preclaim private-capacity refusal is non-mutating. Unpaired initialization
shares the same extracted body and retains its legacy RAII behavior.

Selected F1 keeps the old sequence in its caller-owned slot until a complete,
correctly indexed replacement exists. Selected retirement validates genuine
private authority and target guard identity, then completes wait/zero/sync,
slot graph destruction, metadata frees and private completion before returning
target resources. The first error stops cleanup. A genuinely retired old object
returns before looking up slot-keyed graphs, so it cannot destroy a replacement's
graph. This does not change CUDA backend Drop or implement full T2 teardown.

The fixture now constructs paired sequences through actual Model allocation.
An initial fixture-only teardown adjustment yielded 96 passing characterization
tests; it is not counted as production RED. Then actual ownership RED produced
97 PASS / 9 FAIL, cleanup-profile RED 0 PASS / 2 FAIL, and control-derived
retirement RED 2 PASS / 6 FAIL. Those tests reproduced early guard recycling,
bad index publication, lost F1 caller ownership, accepted foreign/missing guards,
continued cleanup and destruction of a replacement's captured graph.

Final focused suite: **118 PASS / 0 FAIL**, 3.78 seconds. The retirement matrix
executes eight actual control-derived fault boundaries on both ranks and both
owners. Allocation covers the six real zero/reset/completion boundaries on both
ranks and either available owner. Later callback and legacy best-effort checks
are additional characterization, not independently claimed behavioral REDs.
Recorded graph handles prove lifetime/order, not CUDA replay arithmetic.

Source was reviewed independently; root separately reviewed both reviewer-authored
retirement children. Non-test model check passed in 5.34 seconds. Formatting,
diff whitespace and scoped SPDX checks passed. New Rust files are <=500 lines;
the existing allowlisted `impl_a2.rs` grows from 787 to 790 lines. Clippy is NOT
green: four unchanged argument-count errors in runtime `cublaslt_metal_stub.rs`
and `cutlass_metal_stub.rs` stop that check; no suppression was added.

## Deliberately remaining functional and deployment work

This checkpoint refuses arbitrary two-idle allocation churn. Target free slots
are LIFO but private allocation chooses the lowest available slot. Actual
head frees 0->1 followed by reallocation expose that mismatch, including across
ticks. The original executed failing lifecycle attempt is retained. The next
bounded fix is a genuine guarded claim for the actual private candidate, not
an assumption about retirement order. Aligned one-owner reuse and old-object
graph inertness are covered here; general concurrent serving is not qualified.

The serial paired scheduler, actual supervised registration/admission, complete
strict teardown and native healthy two-rank control remain separate gates.
No native fault injection, image rebuild, GPU/model run, reset or TPS claim was
made for this change. v26 remains the last qualified deployment.

## Receipts

Raw directory: `/home/abc/storage/models/atlas-campaigns/20260908/glm-c2-sequence-ownership/`.
Its 29-path source/plan manifest `frozen.sha256` has SHA-256
`3e02327f891a716646e64ce122538594fcb1958fd43ebfdc3aa9d9bc2d438953`.
Raw intermediate failures are retained distinctly from final passing receipts.
At the exact code commit above, root ran the full model library suite: **1094
PASS / 0 FAIL**, 197.42 seconds with `RUST_TEST_THREADS=1`; and the full server
binary suite: **2363 PASS / 0 FAIL / 12 ignored**, 23.78 seconds. Both commands
exited zero. Postcommit formatting/diff checks passed and all 29 frozen hashes
were reverified. Only unreferenced next-step tests/plans and the separate,
unregistered standalone guard draft were present; no engine-compiled WIP.

Both nodes remained stopped: no running Docker containers, available memory
121593904 / 121723240 kB (head/worker), swap unused. The receipt archive is
`glm-c2-sequence-ownership-07084851-receipts.tar`; its detached SHA-256 sidecar
records identity without embedding an archive's hash inside itself. Exact
commands, linking limits and scope are in raw `postcommit-closure.md`.
