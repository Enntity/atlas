# R-e — shared-HEAD build break triage (`*_opt` EXL3 kernels)

- Date: 2026-10-16
- Branch triaged: `sparkglm/installable-20260924` HEAD `d4b753e6`
  (contains `a04d27ff`, `4692fb8d`, `72f9a788`)
- Offending commits: `a04d27ff` (adds the modules), `4692fb8d` (P0 "fix" pass),
  `72f9a788` (Rust dispatch + `cooperative_mma()`)
- Perf-branch status: reverted by `6a5110c2`; `/home/colum/Workspace/RustroverProjects/atlas-re-opt-build`
  (branch `followup/re-opt-kernel-triage`) does **not** contain the files.
- Report-only item: **no fix was implemented anywhere.** The scratch copies used
  to prove the minimal fixes live under `/target/mb_re_opt_triage-*` on spark1,
  never in `~/atlas-dev*/bin`, never in a repo.

## Verdict

The three named modules (`exl3_moe_mma_opt`, `exl3_moe_precision_opt`,
`exl3_persistent_opt`) plus their two new dependencies
(`exl3_mma_coop.cuh`, `exl3_moe_fused.cu`) **fail nvcc with 8–38 hard errors
each**, reproduced with the exact flags `atlas-kernels/build_target.rs` uses.
A fourth new module (`exl3_linear_mma_opt`, also from `a04d27ff`) fails the same
way; it was not named in the ticket but it is in the same commit and breaks the
same build.

Because `collect_cu_files()` compiles **every** `.cu` under
`kernels/gb10/common/`, the failure is not gated on `KERNEL.toml` registration:
`exl3_persistent_opt.cu` is not registered in `[modules]` anywhere, yet it is
still compiled and still breaks the build. Every `gb10` target that includes
`common/` (i.e. all of them) fails.

**Recommendation: fix-and-keep is possible but not cheap; the honest
recommendation for the current owner decision is `revert` (keep the perf-branch
`6a5110c2` state) until someone is funded to re-do the kernels as a reviewed
patch.** Rationale in §6.

Nothing was implemented. The owner decision is still pending.

## 1. Reproduction (builder container, no GPU)

Host access: `spark1` does not resolve from this workstation
(`ssh spark1 …` → `Could not resolve hostname`). Per `continue.md` §5
(`spark1 = colum@10.50.0.11`, hostname `gx10-235f`, reached through jump host
`colum@192.168.1.50`), the working route is the wrapper
`/home/colum/.local/bin/spark1`.

```
$ timeout 40 /home/colum/.local/bin/spark1 'hostname; docker ps --format "{{.Names}}"'
gx10-235f
atlas-p0-nsys-p13trace-rank1
atlas-dev-builder2
atlas-dev-builder
```

Compiler: `nvidia/cuda:13.0.0-devel-ubuntu24.04` container `atlas-dev-builder`
(already running; `docker exec` used, **no** `--gpus`, **no** new container, no
devbuild script, nothing written to `/out` = `~/atlas-dev/bin`).

nvcc in it:

```
$ docker exec atlas-dev-builder nvcc --version | tail -2
Cuda compilation tools, release 13.0, V13.0.48
```

Flags derived from the build, not guessed:

- `crates/atlas-kernels/build_target.rs:96-121` — `NvidiaTarget::compile`
  emits `--ptx -arch=<arch> -O3 [windows -std=c++17] <extra_flags>
  [ATLAS_EXTRA_NVCC_FLAGS] --Werror all-warnings <src> -o <out>`.
- `kernels/gb10/HARDWARE.toml` — `arch = "sm_121f"`.
- `kernels/gb10/common/KERNEL.toml` `[build] extra_nvcc_flags` —
  `["--fmad=false", "-DTQ_PLUS_SIGNS", "--expt-relaxed-constexpr"]`.
  (`kernels/gb10/glm-5.3-flash-exl3/exl3/KERNEL.toml` adds no extra flags for
  these modules, and `[build.module_nvcc_flags]` only overrides `exl3_mhc`.)
- `--Werror all-warnings` is on by default: only the literal
  `ATLAS_KERNEL_NO_STRICT=1` disables it, so the strict-warning leg of the
  breakage is real for every normal build.

Files were copied read-only out of the shared repo
(`git -C /home/colum/Workspace/RustroverProjects/atlas show <rev>:kernels/gb10/common/<file>`,
`<rev> = d4b753e6`) into
`/home/colum/atlas-dev/target/mb_re_opt_triage-89d1b8df/`
(= `/target/mb_re_opt_triage-89d1b8df` in the container).

Exact command per module:

```
docker exec -w /target/mb_re_opt_triage-89d1b8df atlas-dev-builder \
  nvcc --ptx -arch=sm_121f -O3 --fmad=false -DTQ_PLUS_SIGNS \
       --expt-relaxed-constexpr --Werror all-warnings \
       exl3_moe_mma_opt.cu -o exl3_moe_mma_opt.ptx
```

**Control (toolchain/flags are sound):** the untouched sibling
`kernels/gb10/common/exl3_moe_precision.cu` at the same HEAD compiles clean
(`rc=0`) with the identical command. The breakage is specific to the new code.

Result: all four modules exit `rc=1`.

| module (main TU) | nvcc result |
| --- | --- |
| `exl3_moe_mma_opt.cu` | `rc=1` — `8 errors detected` |
| `exl3_moe_precision_opt.cu` | `rc=1` — `14 errors detected` |
| `exl3_persistent_opt.cu` | `rc=1` — `38 errors detected` |
| `exl3_linear_mma_opt.cu` | `rc=1` — `40 errors detected` |

Full logs: `/target/mb_re_opt_triage-89d1b8df/*.log` on spark1
(mirrored at `/tmp/re-opt-triage/logs/` on this workstation for the report).

## 2. Error catalogue with minimal fix per error

All line numbers are shared-HEAD (`d4b753e6`) line numbers.

### E1 — `exl3_common::` qualification does not exist (hits `exl3_persistent_opt`, `exl3_linear_mma_opt`)

`kernels/gb10/common/exl3_common.cuh` declares `mul1` (`:29`) and `tile_state`
(`:39`) at **global** scope; there is no `namespace exl3_common`. The new
headers qualify the calls anyway.

```
exl3_mma_coop.cuh(71): error: name followed by "::" must be a class or namespace name
          float weight = exl3_common::mul1(
exl3_mma_coop.cuh(72): error: name followed by "::" must be a class or namespace name
          exl3_common::tile_state(words, input & 15u, my_col & 15u, bits)
```

Sites: `exl3_mma_coop.cuh:71,72,207 (x2),208 (x2),211 (x2),212 (x2)`,
`exl3_fused_linear.cuh:167,168` — 12 references to `exl3_common::mul1` /
`exl3_common::tile_state`.

Minimal fix: drop the bogus qualification at those 12 call sites (or, if a
namespace is genuinely wanted, wrap `exl3_common.cuh` in
`namespace exl3_common` — a wider change that touches every consumer).

### E2 — missing `#include` (hits `exl3_moe_mma_opt`, `exl3_moe_precision_opt`)

`exl3_moe_fused.cu` includes `cuda_bf16.h`, `cuda_fp16.h`,
`exl3_common.cuh`, `exl3_expert.cuh`, `exl3_hadamard_warp.cuh` — but calls
`exl3_mma_coop::accumulate_coop` / `accumulate_logical` without including
`exl3_mma_coop.cuh`.

```
exl3_moe_fused.cu(265): error: name followed by "::" must be a class or namespace name
              exl3_mma_coop::accumulate_coop<32, 256, 4>(
exl3_moe_fused.cu(274): error #549-D: variable "total" is used before its value is set
                  yh[output_row + column] = total[0];
exl3_moe_fused.cu(285): error: name followed by "::" must be a class or namespace name
              exl3_mma_coop::accumulate_logical(
```

Minimal fix: add `#include "exl3_mma_coop.cuh"` to `exl3_moe_fused.cu:13`.

### E3 — `#pragma once` in a `.cu` that is itself a main TU (hits every target)

`exl3_moe_fused.cu:7` carries `#pragma once`. It is the **only** `.cu` under
`kernels/gb10/common/` with that pragma (checked: all 29 other common `.cuh`
files have it, as normal). Because the build compiles each `.cu` as its own
main translation unit, this fires under `--Werror all-warnings`:

```
exl3_moe_fused.cu:7:9: error: #pragma once in main file [-Werror]
    7 | #pragma once
cc1plus: all warnings being treated as errors
```

Minimal fix: delete line 7 (`#pragma once` is meaningless in a TU that is never
included).

### E4 — `accumulate_coop` is not a template, and one call omits `weight_cache` (hits `exl3_persistent_opt`, `exl3_linear_mma_opt`)

`exl3_mma_coop.cuh:81` declares a plain `__device__ __forceinline__` function
(11 params, ending in `float* weight_cache`), but the call sites use explicit
template arguments and/or drop the last argument:

```
exl3_persistent.cuh(66): error: operand types are incompatible
  ("void (*)(const float *, const unsigned int *, float *, unsigned long long, unsigned int,
   unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, float *)" and "int")
exl3_persistent.cuh(212): error: operand types are incompatible  (same signature)
```

- `exl3_persistent.cuh:66` — `accumulate_coop<32, 256, 4>(xh, trellis, total,
  row_offset, col_base, n, k, bits, slices, half_partial)`: 10 args for 11
  params, and the enclosing kernel `exl3_persistent_mul1_matmul_mma`
  (`exl3_persistent.cuh:21`) declares **no** `extern __shared__` at all.
- `exl3_moe_fused.cu:265` — passes `weight_cache` but still uses the
  non-existent `<32, 256, 4>` template arguments.
- `exl3_persistent.cuh:212` — inside `exl3_persistent_fused_linear`: drops
  `weight_cache` even though the kernel does have it (`:139`).
- The `#174-D: expression has no effect` ×9 at `exl3_persistent.cuh:67` and
  `:213` are pure fallout of the failed call expression (promoted to errors by
  `--Werror all-warnings`), not independent bugs.

Minimal fix: remove `<32, 256, 4>` from the three call sites; add the missing
`weight_cache` argument at `exl3_persistent.cuh:66` and `:212`; declare
`extern __shared__ float weight_cache[];` in
`exl3_persistent_mul1_matmul_mma` and size it 32×32 floats = **4096 B** at
launch (§4).

### E5 — the `*_opt.cu` wrappers re-define and then illegally call `__global__` functions (all four modules)

Each `*_opt.cu` `#include`s a TU that already defines the kernels as
`extern "C" __global__` inside a namespace, then defines an identically-named
`extern "C" __global__` wrapper that *calls* the first one from device code.

```
exl3_moe_mma_opt.cu(11): error: function "exl3_moe_fused::exl3_moe_mul1_matmul_coop_mma"
    has already been defined (previous definition at line 210 of exl3_moe_fused.cu)
exl3_moe_mma_opt.cu(32): error: a __global__ function call must be configured
      exl3_moe_fused::exl3_moe_mul1_matmult_coop_mma(
```
Same pattern at `exl3_moe_precision_opt.cu:14/25, 31/43, 49/64, 71/84`,
`exl3_persistent_opt.cu:14/27, 46/53, 57/68` and
`exl3_linear_mma_opt.cu:19/31, 35/48, 54/65`.

Minimal fix: the wrappers are redundant — the included TUs already emit
`extern "C"` (unmangled) symbols, so reducing each `*_opt.cu` to its
`#include` line(s) yields a module PTX whose `.globl` names match the
`gpu.kernel("exl3_*_opt", "<entry>")` lookups exactly (verified: §5). Note
`exl3_linear_mma_opt.cu` includes **both** `exl3_fused_linear.cuh` and
`exl3_persistent.cuh`, so today that module's PTX also exports
`exl3_persistent_*` — module-boundary leakage worth cleaning while it is open.

### E6 — unused local promoted by `--Werror all-warnings`

`exl3_persistent.cuh:140` (`float* weight_cache = shared_mem + k + 128;`,
`error #177-D: variable "weight_cache" was declared but never referenced`) and
`exl3_fused_linear.cuh:180` (`const unsigned long long row = row_offset;`,
same class). The first is only *actually* dead because E4 drops the argument
that uses it; the second is dead on its own. Minimal fix: remove the truly dead
one; restore `weight_cache` where E4 needs it.

## 3. Errors that nvcc cannot catch (same commit, same subsystem)

### M1 — two dispatch lookups resolve to symbols that do not exist

`crates/spark-model/src/layers/glm5_exl3_ffn/moe/setup.rs:100` turns
`policy.cooperative_mma()` on, and the resulting kernel list asks for:

```
setup.rs:127  gpu.kernel("exl3_moe_precision_opt", "exl3_moe_fused_combine_f32")?
setup.rs:140  gpu.kernel("exl3_moe_precision_opt", "exl3_moe_fused_route_counts")?
```

Neither symbol exists in `exl3_moe_fused.cu` (or anywhere in the new tree —
`grep -c` returns 0). With the E1–E6 fixes applied, `exl3_moe_precision_opt.ptx`
exports exactly:

```
exl3_moe_fused_input_hadamard_f16
exl3_moe_fused_output_hadamard_precision
exl3_moe_fused_activate_down_hadamard
exl3_moe_fused_down_weighted_output
exl3_moe_mul1_matmul_coop_mma
```

so model load would fail closed at lookup/audit time even after the kernels
compile. Minimal fix: add the two kernels to `exl3_moe_fused.cu` (they exist in
the non-opt module as `exl3_moe_precision.cu:32` `exl3_moe_route_counts` and
`:159` `exl3_moe_combine_f32`) **or** stop selecting the opt module for those
two stages. This is a design decision, not a typo — it needs the owner.

### M2 — no dynamic shared memory is ever set

Every kernel in the new tree uses `extern __shared__`:

- `exl3_moe_fused.cu:264` `extern __shared__ float weight_cache[]` (32×32 floats = **4096 B**)
- `exl3_mma_coop.cuh:180` `extern __shared__ float shared_mem[]` (weight cache, **4096 B**)
- `exl3_persistent.cuh:105` `extern __shared__ float scratch[]` (128 floats = **512 B**)
- `exl3_persistent.cuh:137` `xh[k] + 128 + 1024` floats (`k*4 + 512 + 4096 B`)
- `exl3_fused_linear.cuh:49` same layout (`k*4 + 512 + 4096 B`)

and no launch site sets it. `KernelLaunch::shared_mem(bytes)` exists
(`crates/spark-runtime/src/kernel_args.rs:97`, plumbed to cudarc's
`LaunchConfig.shared_mem_bytes` at
`crates/spark-runtime/src/cuda_backend/gpu_impl.rs:226-234`), but:

- `spark-runtime/src/weights/exl3/native.rs:279-325` `launch_fused()` (the
  `exl3_linear_mma_opt::exl3_fused_linear` path used by `new_fused()`,
  `native.rs:143-160`) has **no** `.shared_mem(...)` call.
- `spark-model/src/layers/glm5_exl3_ffn/**` contains **zero** `.shared_mem(`
  call sites — including `moe/project.rs:29,47,74,99`, which launches the
  MoE matmul kernels that E4 is about.

So even once it compiles, the first launch runs with 0 bytes of dynamic shared
memory while the kernel writes `weight_cache[0..1023]` — undefined behaviour /
`CUDA_ERROR_INVALID_VALUE`, not a slow path. For the reference geometry
(hidden 4096, intermediate 2048/1024-local per `continue.md` §6) the sizes stay
well under the 48 KB cap: 4096 B (MoE matmul), 512 B (output hadamard),
12800 B (fused linear at k=2048) and 20992 B (k=4096) — matching the
`k <= 11136` guard already in `native.rs:195` and `exl3_fused_linear.cuh:60`.

Minimal fix (Rust, `native.rs` + `glm5_exl3_ffn/moe/project.rs`): call
`.shared_mem(<bytes>)` on each launch, with the byte count derived from the same
explicit `k`/geometry the kernels validate — no hardcoded magic numbers
(PCND). Note this touches `crates/spark-model` files I do not own for this item,
so it is reported, not attempted.

### M3 — the "off" switch is hardcoded, and it is ON

`crates/spark-model/src/layers/glm5_exl3_ffn/moe/policy.rs:66-71`:

```rust
pub(super) fn cooperative_mma(self) -> bool {
    // Enable for the reference profile (thin_capacity=32, tile_k=32, tile_n=256)
    self.thin_capacity == 32 && self.tile_k == 32 && self.tile_n == 256 && self.slices == 8
}
```

`RoutedPrecisionPolicy::GLM53_TP2_REFERENCE` (`policy.rs:19-29`) is the only
value ever constructed — assigned as a `const` in
`crates/spark-model/src/layers/glm5_exl3_ffn/load.rs:208`. There is no env var,
no `MODEL.toml` key, and no CLI flag: the predicate is unconditionally **true**
for GLM-5.3-Flash-EXL3.

**Which flag keeps them off today: none.** What keeps them off in practice is
only that the perf/F4 branches do not contain the files (`6a5110c2` reverted
`a04d27ff`). Anyone building the shared HEAD `--no-cache` gets the breakage
above with no runtime knob to dodge it.

## 4. What the minimal fixes look like (validated, not applied)

Applied only to scratch copies under
`/target/mb_re_opt_triage-89d1b8df-fix/` on spark1 (script
`/target/mb_re_opt_triage-89d1b8df-fix/patch_fix.sh`): E1 (12 call sites), E2
(one include), E3 (delete `#pragma once`), E4 (template args + the two missing
`weight_cache` args + the missing `extern __shared__` declaration), E5 (reduce
the four `*_opt.cu` to their includes), E6 (delete the dead `row`).

Result with the **unchanged production flags**:

```
exl3_moe_fused      rc=0
exl3_moe_mma_opt    rc=0
exl3_moe_precision_opt rc=0
exl3_persistent_opt rc=0
exl3_linear_mma_opt rc=0
```

So the compile breakage is real, mechanical, and repairable in ~15 lines of
CUDA. **M1 (two missing kernels) and M2 (shared-memory sizing) are not
mechanical** — they are design work on the dispatch contract and would have to
be written, not un-typo'd. That split drives the recommendation.

## 5. Symbol check after the fix (why the wrappers are redundant)

`grep '.globl'` on the fixed module PTX:

```
exl3_moe_mma_opt.ptx      → exl3_moe_mul1_matmul_coop_mma (+ the 4 fused hadamards)
exl3_moe_precision_opt.ptx → exl3_moe_fused_input_hadamard_f16,
                             exl3_moe_fused_output_hadamard_precision,
                             exl3_moe_fused_activate_down_hadamard,
                             exl3_moe_fused_down_weighted_output,
                             exl3_moe_mul1_matmul_coop_mma
exl3_persistent_opt.ptx   → exl3_persistent_mul1_matmul_mma,
                             exl3_persistent_output_hadamard,
                             exl3_persistent_fused_linear
exl3_linear_mma_opt.ptx   → exl3_fused_linear (+ exl3_persistent_* leakage)
```

i.e. the entry-point names `setup.rs` and `native.rs` look up are already the
ones emitted by the included TUs; the wrapper TUs add nothing but the E5
errors (and, in `exl3_linear_mma_opt`, cross-module symbol leakage).

## 6. Recommendation: `revert` now, `keep` only behind a paid-for fix

- **Revert** (keep perf-branch state / re-apply `6a5110c2` on the shared
  branch): unblocks `--no-cache` builds immediately, costs nothing, and loses
  no shipped feature — `cooperative_mma()` is a hardcoded `true`, but the only
  branch that carries these kernels does not compile, and the perf/F4 branches
  reverted them, so no user can be exercising them on a working build. There
  is no served behaviour to preserve. The kernels are also unreviewed by anyone but
  the author: `4692fb8d`'s own message records an adversarial review that found
  8 P0 correctness bugs, and this triage found a *ninth class* (missing
  shared-memory argument, E4) that review did not catch.
- **Fix-and-keep** is viable but must be scoped as a real change, not a typo
  fix: E1–E6 (~15 CUDA lines) **plus** M1 (write `exl3_moe_fused_combine_f32`
  and `exl3_moe_fused_route_counts` or narrow the dispatch) **plus** M2 (size
  and pass dynamic shared memory in `native.rs` and the GLM MoE project path,
  files I do not own) **plus** a GPU numerics run before anyone claims
  behaviour. That is a new patch with its own review, not a drive-by.
- **Do not** ship `ATLAS_KERNEL_NO_STRICT=1` as the workaround: it only silences
  E3/E6, leaves E1/E2/E4/E5 hard failures, and weakens the kernel-quality gate
  repo-wide.

## 7. Not verified (no GPU on this host)

- No speed, correctness, or numerics claim about the `*_opt` kernels — none is
  runnable here. The kernels do not compile, so no A/B exists.
- Whether the fixed kernels are *correct* (E4's `weight_cache` contract,
  M2's sizes) is unverified. It would need, on a GB10 host after the fixes:
  `ATLAS_TARGET_MODEL=glm-5.3-flash-exl3 cargo build -p atlas-kernels` (clean
  `--no-cache` nvcc) and then the EXL3 validation slice from
  `tests/exl3_validation/` against the non-opt module as reference.
- Whether `exl3_persistent_opt` should exist at all: nothing dispatches it and
  it is absent from every `[modules]` table — deleting the file is an
  alternative to fixing it.

## 8. Evidence index

- Shared-repo reads (read-only, nothing checked out):
  `git -C /home/colum/Workspace/RustroverProjects/atlas show <d4b753e6|a04d27ff|4692fb8d|72f9a788>:kernels/gb10/common/<file>`.
- spark1 scratch: `/home/colum/atlas-dev/target/mb_re_opt_triage-89d1b8df/`
  (pristine copies + `*.log` nvcc output) and
  `/home/colum/atlas-dev/target/mb_re_opt_triage-89d1b8df-fix/`
  (patched copies + `patch_fix.sh`, all four modules `rc=0`).
- Local mirror of the pristine logs: `/tmp/re-opt-triage/logs/`.
- Host route that works: `/home/colum/.local/bin/spark1` → `colum@10.50.0.11`
  (`gx10-235f`) via jump `colum@192.168.1.50`. `ssh spark1` by name fails
  (no DNS/`~/.ssh/config` entry on this workstation) — worth recording, since
  the ticket's suggested command is not literally runnable here.
