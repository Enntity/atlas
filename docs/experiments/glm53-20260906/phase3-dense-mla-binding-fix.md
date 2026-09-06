# Phase 3: dense MLA dimension binding correction

Status: binding correction implemented; CPU and bounded GPU regression checks
passed. Full-model validation pending.

## Evidence and scope

GLM's `MODEL.toml` inherits `deepseek-v4-flash` CUDA sources. That bundle maps
`paged_decode_mla` to `paged_decode_attn_mla.cu`, whose compile-time `HDIM` is
576. `qwen3_attention/init.rs` binds that module without checking the model.
GLM's absorbed latent cache is 512 elements, and dense decode supplies runtime
`head_dim=512`. The CUDA kernel's compile-time vector loops still read/write
576 values per head while its runtime head stride is 512: adjacent heads
overlap, and the final head accesses an extra 64 elements. Large scratch
allocations can conceal this from allocation-level bounds checks.

The mismatch is a concrete correctness defect. It is not yet proven to be the
sole cause of the observed C4 failure and subsequent same-process failures.
Fresh C1 and C3 passing does not establish that the overlapping accesses are
safe or deterministic.

## Minimal change and tests-first sequence

1. Extract the MLA BF16 module choice into the existing pure initialization
   dispatch helper. Add a failing GLM-512 regression test before correcting
   the legacy module choice.
2. Bind exact `glm5_next`, latent rank 512, zero RoPE to the existing
   `paged_decode_attn_512` module, symbol `paged_decode_attn`. Reject unexpected
   GLM latent geometry and require the chosen kernel to load successfully.
   Preserve all non-GLM module choices, including DeepSeek-576 and Mistral-320.
3. Test GLM valid/invalid dimensions, preserved non-GLM choices, and the source
   kernel dimension/module alias contract. Run CPU model unit tests only.
4. The shared `paged_decode_mla_k` binding fixes ordinary scalar/batched dense
   attention and the dynamic sparse path's dense fallback together. No CUDA
   arithmetic or speculative width policy changes are needed.

CPU receipt: the new GLM module-choice test first failed with legacy
`paged_decode_mla` versus expected `paged_decode_attn_512`. After the binding
fix, all eight `init_kernel_dispatch` tests passed, including four new GLM
regressions and all four existing dtype dispatch tests. `git diff --check`
passed. The initializer was already 830 lines before this change; no unrelated
comments or code were removed to reduce its pre-existing size.

## GPU handoff, owned by root

Use `kernels/gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_512.cu`, symbol
`paged_decode_attn`, as the production comparator. Its ABI is Q, K cache,
V cache, output, block tables, sequence lengths, max blocks per sequence,
query heads, KV heads, head dimension, block size, scale, query stride, sliding
window. Launch grid `(32, rows, 1)`, block `(256,1,1)`, dynamic shared memory 0;
head dimension 512, KV heads 1, block size 16, query stride 16384, window 0.
The exact scalar comparator is the same kernel at grid `(32,1,1)` with each
row's query/output/table/length pointers explicitly offset. Do not launch the
known-wrong 576 kernel on 512-sized data.

Check rows 1–4, distinct/permuted page tables and row lengths, per-head/row
canaries, a CPU attention oracle, and scalar-versus-batch agreement. Dynamic
zero-length dense guards intentionally leave output untouched. Restart both
model processes before full-model short C4, C3/C2/C1 drain, and independent
needle tests; then recheck mixed 2048 threshold sparse graphs. Prior dense
numerical outputs are not a valid exact oracle for the corrected binding.

## Bounded GPU receipt

Root ran `scripts/dev/bench_glm_paged_decode_512.cu` on the head Spark with both
model containers stopped, within a 1 GiB no-network container. All 16 cases
passed for production contiguous query strides, and all 16 passed again with
64 poisoned padding elements between query rows. Scalar, batched, and graph
outputs are bit-exact; the independent double-precision CPU reference passes
the declared BF16 rounding tolerance. All guards and read-only data checks
pass. A third run under `compute-sanitizer --tool memcheck` passes all 16 cases
with zero reported errors. Device allocations are below 18 MiB. Raw receipt:
`/tmp/atlas-glm53-phase3-20260906/v7-paged512-gpu.log` on the controller.

The full CPU model unit suite also passes: 682 tests, zero failures. These are
numerical/structural regression tests, not a serving performance claim or a
substitute for the multi-model serve matrix.
