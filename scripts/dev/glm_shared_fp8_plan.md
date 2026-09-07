# Standalone predecoded-FP8 shared-expert comparison

Scope: new standalone files only. No production dispatch, loader, allocations,
flags, frozen M16 sources, native compilation or GPU/node operations by author.
Root owns native execution and any later promotion decision.

## Same arithmetic, explicit layouts

The actual existing symbol is `predequant_nvfp4_to_fp8` (no `_t` suffix), taking
original `[N,K/2]` packed weights and `[N,K/16]` E4M3 scales. The fixture keeps
that original and a CPU byte-transposed `[K/2,N]` / `[K/16,N]` copy. This is one
logical weight, not different quantization policies. Include the unmodified
production `w4a16_gemm.cu` for converter, transposed W4A16 reference and existing
`fp8_gemm_t`. Predecode writes `[N,K]` FP8 with scale2 incorporated.

The converter retains `(float)scale * scale2`, then FP4 lookup multiplication,
then `cvt.rn.satfinite.e4m3x2`. Existing FP8 GEMM keeps the same BF16-to-E4M3 A
conversion, K32 MMA accumulation sequence and BF16 store. A new small M16 FP8
variant changes four warps along M to four along N, using the same conversion,
K32 MMA and double-buffered pipeline. It does not edit or include the frozen
shared-M16 experiment and is independently tested against the original M64.

## CPU-first and native gates

1. TDD: make the actual host transpose index initially incorrect; a boundary
   assertion must fail before full implementation. Prove full byte transpose
   bijection/roundtrip for packed weights and scales at both production shapes,
   E4M3 ties/saturation/signed-zero reference, M16 output ownership, shared copy
   coverage, strict geometry and overflow-safe explicit allocation accounting.
2. Three sequential fixtures: distinct gate and up N2048/K4096 weights; down
   N4096/K2048. Each has original+transposed packed/scales, one predecoded FP8
   matrix, BF16 A and three output buffers. Hard cap32MiB including individual
   allocation canaries; there are no concurrent fixtures or duplicated model
   weights. No full-model allocation or resident-weight claims.
3. M0/1/4/5/16, exact-zero source rows and NaN inactive A padding. Two epochs
   refresh A, original/T weights and scales; predecode runs and synchronizes
   outside graph capture/replay. Require every predecoded byte equal to an
   independent CPU E4M3-RNE oracle, including both nibbles and scale2.
4. Full original-vs-existing-FP8-vs-M16-FP8 output bit equality, all outputs
   freshly written/finite, all inactive rows untouched. Independent CPU double
   columns after explicit E4M3 input/weight rounding allow at most one BF16 ULP;
   this is secondary to strict original-kernel equality.
5. Capture only the three GEMMs using fixed pointers. Poison outputs and replay
   after refreshing A and cached FP8 bytes; compare with fresh eager outputs.
   No conversion or weight mutation occurs inside a graph or timed region.
6. Check every original/T/FP8 weight byte and activation before and after graph
   replay and timing, plus all allocation guards. Timings require explicit
   `--timing` after all correctness cases pass, alternating CUDA-event samples,
   and renewed output/input checks afterward. Conversion cost is excluded from
   GEMM timings and reported separately as one-time setup, not an inference gain.

## Interpretation and next boundary

Shared geometry is replicated per TP2/EP2 rank: each of42 target MoE layers has
three8MiB FP8 matrices, so retaining current original/T weights and adding FP8
costs1008MiB/rank (1,056,964,608 bytes). Do not automatically include draft MoE.
Packed+scales traffic is9/16 bytes per coefficient versus1 byte for FP8:16/9x
logical bandwidth, or567→1008MiB across42 shared layers per five-row pass.
The experiment removes B conversion and one stage barrier, but a hot8MiB matrix
does not model cold full-model weights or predict TPS. Actual both-rank memory
headroom and cold-model A/B remain mandatory after standalone validation.

## Commands (native runs belong to root)

```bash
g++ -std=c++17 -O2 -DATLAS_SHARED_FP8_HOST_ONLY -x c++ scripts/dev/bench_glm_shared_fp8.cu -o /tmp/bench-glm-shared-fp8-host
/tmp/bench-glm-shared-fp8-host --host-test
nvcc -std=c++17 -O3 -arch=sm_121 -Xptxas=-v scripts/dev/bench_glm_shared_fp8.cu -o /tmp/bench-glm-shared-fp8
/tmp/bench-glm-shared-fp8
compute-sanitizer --tool memcheck --error-exitcode=9 /tmp/bench-glm-shared-fp8
/tmp/bench-glm-shared-fp8 --timing
```

Root should run both default FMA and `--fmad=false` native gates. CPU checks do
not establish device correctness. Source must receive independent review first.

## Local CPU receipts

The initial transpose helper retained original row-major addressing: the harness
compiled and failed `original-to-T column stride` (exit2). Correct transpose
addressing passes the full packed/scales bijection and roundtrip at both shapes,
output/copy ownership, allocation bounds and E4M3/BF16 reference assertions.
An added hand-computed64-byte predecode fixture checks both nibbles, scalar scale2,
negative zero, the16-element scale-group boundary, saturation and subnormal ties.

Raw phase6 receipts: `shared-fp8-host-red.log`, `shared-fp8-host-green.log`, and
`shared-fp8-host-sanitize.log` (local ASan/UBSan host tests also pass).
The exact9-allocation guarded totals are18,155,776 bytes for gate/up and
18,286,848 bytes for down. Shapes run sequentially and release all allocations.
CUDA context/event/graph bookkeeping is separate from these explicit allocations.

Source comparison confirms the M16 candidate's FP8 B loader, MMA instruction
text and complete K-stage pipeline match the original. Only A16 ownership and
four warp-N partitions change. Native compilation, GPU tests and timing remain
unrun by the author; these CPU receipts are not numerical CUDA evidence.
