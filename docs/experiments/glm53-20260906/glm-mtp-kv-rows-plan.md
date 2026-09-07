# GLM MTP bounded KV-row writer foundation

This extraction does not enable accepted-prefix repair. Scheduler, proposer
trim semantics, EP commands, and current primer token alignment remain unchanged.

## Contract and implementation

Extract the current BF16 primer's embedding copies, enorm/hnorm, concatenation,
BF16 EH projection, and NoPE MLA KV-only write into a child module. Retain the
existing cuBLAS-if-enabled-and-multiple-rows selection and dense fallback, chunk
size, same-stream ordering, and synchronization protecting pageable slot uploads.

The internal bounded entrypoint accepts already-shifted token IDs, an explicitly
sized owned post-final-norm BF16 hidden span, an arbitrary logical destination
row, and preallocated exclusive cache blocks. It neither advances proposer state
nor allocates cache blocks. Validate the entire request before its first GPU copy,
upload, or kernel: token IDs, checked source/embedding/destination arithmetic,
source capacity and disjointness from all touched scratch, chunk capacities,
metadata capacity, physical block range/uniqueness/ownership, and BF16 NoPE cache
geometry. Weight extents and appended-layer identity remain the loader's contract;
this foundation must not claim that an arbitrary DevicePtr proves ownership.

The existing primer remains the only caller. It preserves its early no-op,
prompt[1..] pairing, block allocation, and final seq_len update. Preparation must
reject malformed inputs before allocating blocks or performing GPU work; the
writer revalidates concrete destinations after those host allocations.

## Test-first gates

CPU tests exercise the production plan and actual embedding-copy/slot-upload
helpers with MockGpuBackend: shuffled blocks crossing boundaries, nonzero source
and destination offsets, multiple chunks, untouched rows, invalid final token,
undersized source, scratch aliasing, duplicate/out-of-range blocks,
metadata shortage, and arithmetic overflow. Invalid requests must perform no GPU
operations. A pure projection decision test covers cuBLAS versus dense selection;
CPU mocks do not validate CUDA arithmetic or execute cuBLAS.

Native enablement is separate: compare resident-weight execution against the
original primer/serial oracle using small KV snapshots, including boundary rows.
Do not duplicate the full EH matrix or allocate another model. Root owns all
GPU runs and deployment. No performance or numerical correctness claim is made
by the CPU foundation alone.

## CPU result (2026-09-07)

Initial missing-planner tests failed at compilation, then all three planning/
copy tests passed. Three additional tests drive the real primer/writer with a
metadata-checking stand-in for the MLA body; the scratch guards first rejected
the undersized synthetic fixture, which was corrected without weakening guards.
All six writer tests now pass. The combined model library suite (including the
separate pure pair planner) passed **751/751**, with raw output at
`/tmp/atlas-glm53-phase6-20260907.J5PkkO/slice-b-foundation-model-tests.log`.
The private MTP allocator allows exclusively owned physical block zero; unlike
the main cache, it does not reserve that block. Tests pin this behavior.
No accepted-prefix runtime route has been added, and no GPU test was run here.
