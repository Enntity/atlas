# Unactivated C6/C8 MLA projection prerequisite

2026-09-09. Approved bounded CUDA prerequisite; root owns all native commands,
source transfer and commits. This author owns only this plan and the existing
`bench_glm_mla_batch.cu` fixture initially. No Rust dispatch, factory, admission,
model build, runtime flags, watchdog or rollback-ring changes.

## RED boundary and later exports

Extend the existing fixture to reference `mla_batched_gemv_batch6`, `batch7`
and `batch8`, without defining them. Freeze the fixture and unchanged canonical
`kernels/gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu`. Root captures the actual
CPU-only nvcc missing-symbol RED after the current model matrix stops cleanly.
Only a subsequent explicit authorization permits three instantiations of the
existing `mla_batched_gemv_batch_impl<ROWS>` through its existing argument/call
macros. Do not change arithmetic, old exports, scalar control or launch ABI.

The model's `MODEL.toml` already selects `kernel_source="deepseek-v4-flash"`;
these eventual exports belong in the existing `mla_absorbed` artifact, not a
new kernel copy/module or build-script registration. A later engine slice must
resolve each matching export and validate exact independent row count before
dispatch. No runtime caller is added here.

## Existing fixture, expanded coverage

- Keep rows2/3/4/5/10 and M10's timed two-M5 plus separate scalar controls.
  Add rows6/7/8, each against that many actual scalar calls with exact BF16
  equality over all output bytes, including padding. Row1 remains the scalar
  reference, not an invented batched export.
- For every new row width use actual local GLM Q absorption `(heads32,N512,K256)`
  and V extraction `(heads32,N256,K512)`, both padded and unpadded strides.
  Keep the small padded `(heads3,N8,K8)` control; add the existing odd-output
  tail shape `(heads3,N9,K12)` in both padded and unpadded layouts.
- Exhaustive independent host-double dots for every new live output; retain
  the existing absolute/relative error gate, finite checks and allocation guards.
- At the same device input/output addresses, reverse all rows and rotate by one;
  compare both scalar and batched full rows with the corresponding original
  outputs. Preserve the existing M4/M10 permutation lists unchanged.
- Preserve the checked16MiB live-device-allocation cap. New widths are below
  existing M10, so maximum allocation does not increase. Expected full run:
  47 shape cases (old26 plus new21), with timing only on unpadded GLM shapes.
  Existing maximum100 repetitions, ten warmup pairs and five alternating-order
  timing rounds stay unchanged; repetition0 is correctness/memcheck only.

## Root-only qualification and limits

Use the existing isolated compile/probe recipe: one compiler, no GPU exposed
during compilation, production `-O3 --fmad=false -arch=sm_121a -std=c++17`,
4GiB compile ceiling; one GPU and2GiB process ceiling for the fixture, timeout300s,
compute-sanitizer memcheck error-exitcode99. No compile/model/GPU overlap. Root
checks both nodes' process state, swap0 and available memory before each phase.
On fault/timeout preserve receipts and stop subsequent GPU work for health checks.

Require exact numerical/canary gates and memcheck before interpreting latency.
Report scalar-row versus batch6/7/8 timings separately for both GLM projections;
do not replace the retained M10/two-M5 control or claim full-model throughput.

Rows are independent projection operands, not temporal K5 verification. A later
C6/C8 model implementation must cover every draining width1..8 with live owner
slot IDs, unique state bindings, causal lengths/block maps, context<=2048,
scratch limits and identical head/worker ordering. This fixture establishes none
of that ownership or recurrent/attention correctness. Keep existing rollback and
watchdog policy; the estimated C4-to-C8 capacity delta requires actual post-load
memory checks before any future serving admission.
