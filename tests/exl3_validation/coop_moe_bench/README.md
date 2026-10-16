# Routed-MoE coop microbenchmarks (GB10)

Standalone nvcc harnesses for the packed routed-MoE kernels. Build in the
spark1 builder container (nvcc, no GPU), run on the Spark host (GPU), always
under the shared bench lock. Compile flags must match the production EXL3
modules (`kernels/gb10/common/KERNEL.toml`: `--fmad=false -DTQ_PLUS_SIGNS
--expt-relaxed-constexpr`) and the target arch (`kernels/gb10/HARDWARE.toml`:
`sm_121f`):

```bash
flock /tmp/atlas-bench.lock ~/.local/bin/spark1 '
  docker exec atlas-dev-builder bash -c "cd /build/tests/exl3_validation/coop_moe_bench &&
    nvcc -O3 -arch=sm_121f -std=c++17 --fmad=false -DTQ_PLUS_SIGNS --expt-relaxed-constexpr \
         -I/build/kernels/gb10/glm-5.3-flash-exl3/exl3 -o /out/mb_fat fat_multirow_bench.cu" &&
  ~/atlas-dev/bin/mb_fat'
```

| File | Kernels | Compares |
|---|---|---|
| `bench.cu` | thin coop v2/v3/v6, v7 | per-slot variants, bit-exact vs v2, timing, roofline, side-stream overlap |
| `multirow_bench.cu` | thin coop v6/v8 | expert-grouped multi-row vs per-slot v6, bit-exact, bytes moved |
| `fat_multirow_bench.cu` | `exl3_moe_group_units` + thin v8 + `exl3_coop_moe_fat_multirow_b4_k16_n128` | the grouped path (`ATLAS_EXL3_COOP_MOE_FAT_MULTIROW`) vs the per-slot reference `exl3_moe_mul1_matmul_precision_mma`, byte-for-byte per slot over routing distributions with counts below/above `thin_cap` and groups above the 16-row M tile, plus timing at a realistic prefill shape (rows 128, topk 8, 288 experts, k/n from `ROWS`/`TOPK`/`EXPERTS` env) |
| `probe_identity.cu` | all of the above | localizes any identity difference: `native` vs `native_split` vs `coop_v1_per_slot` vs the grouped thin and fat kernels on one expert (thin and fat counts) |

`fat_multirow_bench.cu` env: `ITERS` (timed iterations, default 50), `ROWS`
(128), `TOPK` (8), `EXPERTS` (288), `THIN_CAP` (32). It fails (non-zero exit)
if any per-slot value differs or a lock row is not back to zero.

A standalone "bit-exact" claim is only a precondition: confirm it in-model with
`ab_full.sh` and the unchanged `output_sha256`.
