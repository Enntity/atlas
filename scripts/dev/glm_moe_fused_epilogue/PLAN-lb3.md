# One function-local launch-bounds screen

Keep the original candidate, baseline, previous results and 1.2x chain gate
unchanged. Derive `generated-lb3.cuh` by adding only
`__launch_bounds__(128,3)` to `atlas_dev_moe_fused_gate_up`. The generator
checks frozen source hashes and proves removing that one attribute recreates
the original candidate byte-for-byte. No global register-limit compiler flag.

The derivative harness changes only its candidate include and resource-query
report/gate. `--resources` requests function attributes and occupancy only,
without device buffers, fixture setup or kernel launches. Require candidate
occupancy >=3 blocks/SM and `localSizeBytes==0`. `--run` retains every original
fixture, oracle, poison/guard check, timing scope, and 1.2x fail-fast threshold.

CPU compile only is currently authorized: CUDA13, explicit sm121a, O3,
fmad=false, Xptxas=-v, cached builder digest
`sha256:b1ff6a353c269287e3e77a8f812fbb07a2d33112a23bc2d87b1d00048b13f701`.
Use an isolated container, network none, runc/no GPU, four CPUs,4GiB RAM/swap,
180-second compiler timeout. Preserve source overlay, hashes and compiler log
under private `parity-20260911/moe-fused-epilogue/lb3`.

Original candidate:255 registers/thread,23296 shared bytes, zero stack/spills,
two resident CTAs; original baseline127 registers and four CTAs. If the new
compiler reports any spills or local-stack growth, reject immediately with
no GPU probe and no additional variant. Otherwise parent owns the sole
`--resources` GPU-context probe and any subsequent unchanged chain screen.
Three CTAs/zero local storage are prerequisites, not a speed result.

## Disposition: rejected at compiler gate

The one CPU compilation succeeded, but ptxas reported168 registers plus232
bytes each of stack frame, spill stores and spill loads for the candidate.
Shared memory stayed23296 bytes. The baseline stayed127 registers, zero
stack/spills and23296 shared bytes. The original candidate was255 registers
with zero stack/spills. The launch bound therefore traded registers for local
memory, failing the declared gate. No `--resources`, `--run`, GPU context,
additional variant, model change or integration was attempted.

Local host tests and the attribute-only/unchanged-baseline/run_case checks
passed. Raw compiler evidence and exact commands are retained privately under
`parity-20260911/moe-fused-epilogue/lb3`. Previous rejection records and gates
remain unchanged.
