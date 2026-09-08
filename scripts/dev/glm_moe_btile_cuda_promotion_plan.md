# GLM B-tile CUDA-only production promotion

Scope: root-approved source promotion, not layout activation. Root owns commits,
native compilation, nodes and GPU gates. This author uses CPU Python/source
checks only; no Cargo, native compiler, Docker or model operations.

## Exact partition and source ownership

GLM's `kernels/gb10/glm-5.3-flash-nvfp4/MODEL.toml` redirects `kernel_source`
to `deepseek-v4-flash`. Its existing NVFP4 KERNEL.toml provides `--fmad=false`
and maps `moe_w4a16_grouped_gemm.cu` to module `moe_w4a16`; do not change it.

Move the following six helper implementations from `scripts/dev/` to
`kernels/gb10/deepseek-v4-flash/nvfp4/`, preserving every mathematical body:

- `glm_moe_btile.cuh` (M16, two scale policies).
- `glm_moe_btile_m64.cuh` and `glm_moe_btile_m64_bounds.h` (six M64 exports,
  gathered or route-major input, row bound1088).
- `glm_moe_btile_decode_register.cuh` (word and vector rows1/2/3; no winner
  selection).
- `glm_moe_btile_native_repack.cuh` and `glm_moe_btile_native_layout.h`.

Replace the old six paths with SPDX plus one include of the promoted path.
Existing standalone fixtures then compile the production helper bodies,
including host byte/bounds oracles, without copied implementations.
Only comment banners may change in promoted files.

Append the M16/M64 includes to the existing grouped translation unit, after
its current primitives. Add target-local `glm_moe_btile_decode.cu`, including
unchanged `../../common/moe_shared_expert_fused_t.cu` followed by the promoted
register header. The common file supplies the existing LUT/decoder; its extra
legacy exports are contained in this new module. Existing common sources,
legacy module lookup and all other model source trees stay unchanged.
Add `glm_moe_btile_native_repack.cu` including the promoted repack header.
The two new translation units use their file stems as module names.

No Rust handle/ops family, serving selector, env flag, loader caller, Ready
capability, resident conversion or numerical change belongs to this partition.

## CPU-first verification

Before production edits, add actual source-closure tests requiring the new
translation units, inherited target/module configuration, exactly one shim per
old helper, frozen helper body digests and unchanged shared dependencies.
Execute them to preserve behavioral RED on the actual missing registration/
SSOT closure; a syntax/import/compiler failure is not the required RED.

Add `check_glm_moe_btile_ptx.py` with three explicit PTX inputs (`--grouped`,
`--decode`, `--repack`). It must refuse missing/duplicate entries, absent or
empty/non-PTX files, mismatched parameter count/type/width, or non-sm121a
artifacts. It must never skip missing artifacts. Expected signatures:

- Repack:2 pointers +2u32 (4 parameters).
- BF16 word/vector rows1/2/3:21 parameters, including two FP32 shared scalars.
- M16 and M64 fused compact:18 parameters.
- M64 separate dense:11; separate compact:14.

Synthetic PTX tests prove parser/refusal behavior, not compiled CUDA validity.
Root will run the same checker on actual compiled output. Log paths are under
`atlas-campaigns/20260908/btile-cuda-promotion/`.

## Native handoff and remaining activation prerequisites

Root reruns unchanged fixtures against promoted SSOT: native repack full-byte
oracle; decode-register rows1/2/3 both variants; M16 C4/K5; M64 all three ABIs,
both scale policies, gathered/route-major eager/graph cases through1088.
Run strict PTX signature checks and native memory checks; no CPU test substitutes
for those gates. Preserve `--fmad=false` and frozen source hashes.

Future checked host dispatch must enforce rows1..1088, M16 only gathered rows
<=5, complete dense M-tile coverage (17 at1088), and exact geometry. Before
conversion, activation must reject unsupported BF16-input grouped prefill,
incompatible native-table readers and unsupported profiles. It must complete
loader ownership and all scalar/bootstrap/K2/K3/C4/K5/prefill/drain readers.
Missing B-tile handles must not fall back to old transposed readers. None of
these later capabilities is claimed by CUDA registration alone.

Freeze the exact changed file list and SHA256 closure for root review/overlay.
No native numerical or throughput claim accompanies this source promotion.

## Completed CPU evidence and exact handoff

Implementation has18 owned changed/new files: the six production headers and
six script shims listed above; the two new production translation units; the
existing grouped translation unit; this plan; the PTX checker; and
`test_glm_moe_btile_cuda_promotion.py`. No standalone benchmark body changed.
The campaign `changed-source.sha256` lists all18 exact paths/hashes for overlay.
`unchanged-dependencies.sha256` additionally pins the common scalar source,
common `mx_block_scale.cuh`, existing `glm_moe_gate_up_m16.cuh`, target
KERNEL.toml and GLM MODEL.toml. CUDA SDK headers are toolchain dependencies,
not vendored/copied source. The grouped TU includes both promoted headers;
the native fixture's later script shim includes are harmless via `#pragma once`.

Recorded CPU results in `btile-cuda-promotion/`:

- `source-ptx-red.log`:7 test methods executed,20 assertion failures from
  missing production closure and a PTX empty-report stub (not compiler RED).
- `source-ptx-green.log`: initial7/7 pass after promotion/implementation.
- `definition-red.log`: actual checker wrongly accepted nonvisible/extern
  declarations, declaration-only entries and duplicate parameter names.
- `target-whitespace-red.log`: actual checker rejected valid target-line
  trailing whitespace; review correction strips surrounding token whitespace
  without allowing another target or extra target modifiers.
- `final-green.log`:10/10 pass after strict visible-definition/parameter checks;
  includes frozen body digests, all15 exports, bounded input, missing/type/
  width/arity/target/duplicate refusal, and actual CLI success/failure.
- `kernel-shadows.log`: existing kernel-shadow structure check passes.
- Scoped SPDX and `git diff --check` pass. No Cargo/native compiler was run.

Root invocation after real native compilation (explicit paths, no defaults):

```bash
python3 scripts/dev/check_glm_moe_btile_ptx.py \
  --grouped /path/moe_w4a16_grouped_gemm.ptx \
  --decode /path/glm_moe_btile_decode.ptx \
  --repack /path/glm_moe_btile_native_repack.ptx
```

The checker requires three distinct regular ASCII PTX artifacts, each bounded
128MiB; exact sm_121a/64-bit metadata; every required visible definition once;
and ordered parameter kinds/widths/counts. It emits artifact SHA256 and complete
signature evidence, or JSON error/exit2. This is a signature gate, not a PTX
assembler, provenance attestation or numeric GPU oracle. Synthetic fixtures
cannot establish that the compiler produced an artifact.

Prior register-decode timing evidence exists in phase6-results.md:239; it does
not select a serving variant here. Promotion reruns are exact-source correctness
gates, not another exploratory timing sweep.
