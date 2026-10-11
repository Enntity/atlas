# EXL3 reference-policy KDA core (FLA bit-parity)

Status: design / groundwork only. No production code is changed by this
document. The CPU acceptance oracle for this specification is
`tests/exl3_validation/test_exl3_kda_reference.py`.

## 1. Problem and goal

Atlas's KDA recurrent core (`kernels/gb10/common/kda.cu`, dispatched from
`crates/spark-model/src/layers/glm5_kda/`) computes the correctly rounded
result. The frozen parity gates compare against the installed vLLM FLA
chunked implementation (`chunk_kda_with_fused_gate`), which is lossy relative
to exact math. Closing the frozen gates requires an EXL3-only
**reference-policy** KDA core that reproduces the FLA arithmetic bit-for-bit
on GB10, without changing the default path for ordinary models.

This document specifies that policy exactly as derived from the validated
NumPy emulation, states the PTX-level operations needed to reproduce Triton
behaviour, defines the EXL3-only opt-in dispatch seam, records the known
residual risk, and lists the validation gates.

## 2. Sources of truth

External validation artefacts (root:
`/home/colum/Workspace/atlas-exl3-validation-20261010/`):

- `arithmetic-audit/kda-core-divergence/findings.md` — divergence findings,
  ablation evidence, residual hypotheses.
- `arithmetic-audit/kda-core-divergence/emulate_fla_chunk.py` — single-chunk
  (M5) emulation; BF16 rounding-point flags and ablation harness.
- `arithmetic-audit/kda-core-divergence/emulate_fla_ext.py` — general
  multi-chunk prefill (M128) and one-row decode emulation; `DEFAULT` policy
  dict, `prefill()`, `decode()`.
- `arithmetic-audit/kda-core-divergence/fla-emulation-m5.json` — M5 ablation
  report (six required rounding points; min ablation cost 533 flips).
- `arithmetic-audit/kda-core-divergence/fla-emulation-ext.json` — recorded
  results: M5 0/0, decode 0/0 (f32 state), M128 1728/585 of 524288.
- `arithmetic-audit/kda-core-divergence/raw/` — captured reference input and
  output tensors per (rank, invocation); CPU fixtures for the harness.
- `arithmetic-audit/reference-kda-kernel-source/` — pinned reference kernels:
  `fla/ops/kda.py`, `fla/ops/solve_tril.py`, `fla/ops/chunk_delta_h.py`,
  `fla/ops/fused_recurrent.py`, `fla/ops/l2norm.py`,
  `reference-kda-kernel-source/causal_conv1d.py`, and the vLLM wrapper
  `arithmetic-audit/glm5next/nvidia/kda.py`.

In-repo seams:

- `crates/spark-model/src/layers/glm5_kda/recurrent.rs` — `KdaRecurrentPath`
  enum and `run_recurrent` dispatch (the pattern this policy follows).
- `crates/spark-model/src/layers/glm5_kda/forward_recurrent.rs` —
  `sequence_recurrent_rows` (conv → FlashKDA-or-`run_recurrent` prefill seam).
- `crates/spark-model/src/layers/glm5_kda/assembly.rs` — kernel resolution
  and env-flag parsing (`parse_register_resident_prefill` precedent).
- `crates/spark-model/src/layers/glm5_kda/exl3.rs` — `validate_exl3_forward`
  (the EXL3 gating precedent).
- `docs/design/exl3-support.md` — EXL3 architecture contract.

Geometry of the captured case: TP=2, rank-local `H=32` heads, `D=K=V=128`,
`lower_bound=-5`, layer 0. Shapes in the emulation are `HL=32`, `D=128`.

## 3. Reference-policy specification

### 3.1 Chunking and stage overview (prefill)

Chunk `BT = 64` (`FLA_CHUNK_SIZE`), sub-chunk `BC = 16`, `NC = BT/BC = 4`.
Pipeline per invocation of `T` rows (emulation `prefill()`, reference
`_chunk_kda_fwd_with_cumulative_g`):

1. causal conv (width 4, f32 acc, SiLU, BF16 out) → q/k/v planes
2. q/k L2 norm (prefill only, BF16 out), v unnormalised
3. beta = f32 sigmoid of BF16 raw logits (outside the kernel:
   `x.float().sigmoid()`)
4. gate = safe-gate in f32, **chunk-local** cumsum × `RCP_LN2` per 64 rows
   (the reference resets the cumsum every chunk — `chunk_local_cumsum`);
   stored f32
5. KKT: `A`, `Aqk` computed and stored f32 (diagonal 16×16 blocks by f32
   `tl.sum`; off-diagonal blocks by tf32 dots, `A *= beta_i` after the K loop,
   `Aqk` operands already carry `scale = D**-0.5`)
6. `solve_tril` → `Ai` in f32 arithmetic, stored **BF16**
   (`output_dtype=k.dtype`)
7. `recompute_w_u`: `u = dot(Ai, (v*beta).bf16)`, `w = dot(Ai,
   (k*beta*exp2(g)).bf16)`, `kg = (k*exp2(g_last - g)).bf16` — all three
   outputs stored BF16
8. `chunk_delta_h`: carried state `h` is f32 registers across chunks; every
   *consumer* sees `bf16(h)`; `v_new = bf16(u - dot(w, bf16(h)^T))`;
   `h *= exp2(gk_last)` (f32); `h += dot(kg^T, v_new)` (BF16 operands, f32
   acc); `final_state` stored f32 (no rounding)
9. output: `o = dot(bf16(bf16(q*scale)*exp2(g)), bf16(h)^T) + dot(bf16(tril
   Aqk), v_new)` (f32 acc), stored BF16 (into the v buffer in the reference)

### 3.2 (a) Exact BF16 rounding points

All BF16 casts are round-to-nearest-even (`cvt.rn.bf16.f32`). Thirteen
stages below hold fourteen distinct casts (stage 12 casts twice); every one
is required (M5 ablation costs from
`fla-emulation-m5.json`, M128 ablation costs from `fla-emulation-ext.json`,
mismatch counts vs the captured reference when the cast is removed):

| # | Cast point | Reference site | Emulation flag | M5 r0/r1 cost | M128 r0/r1 cost |
|---|---|---|---|---|---|
| 1 | conv output after SiLU | `causal_conv1d.py` store to `empty_like(x)` | `conv_out_bf16` | 11942 / 10811 | (not re-ablated at M128; required at M5) |
| 2 | q, k after L2 norm (prefill only) | `l2norm.py` kernel2 store | `qk_l2_bf16` | 7847 / 7723 | 232063 / 226435 |
| 3 | `Ai` from `solve_tril` | `solve_tril(..., output_dtype=k.dtype)` | `Ai_bf16` | 533 / 621 | 61522 / 65473 |
| 4 | `v*beta` before the u dot | `recompute_w_u`: `(b_v*b_b).to(b_v.dtype)` | `vb_bf16` | 4324 / 4369 | 192336 / 193511 |
| 5 | `u` | `recompute_w_u` store to `empty_like(v)` | `u_bf16` | 4787 / 4983 | 58035 / 59964 |
| 6 | `tril(Aqk)` at output-dot load | `kda.py` O kernel: `where(m_s,A,0).to(b_v.dtype)` | `Aqk_bf16` | 6596 / 6329 | 168630 / 179000 |
| 7 | `h` read per chunk | `chunk_delta_h`: `h = k.new_empty` (BF16) | `h_bf16` | inert (h=0, T=5) | 24016 / 21781 |
| 8 | `w` | `recompute_w_u` store to `empty_like(k)` | `w_bf16` | inert | 10919 / 8341 |
| 9 | `k*beta*exp2(g)` before the w dot | `b_w = tl.dot(b_A, b_kb.to(b_k.dtype))` | `kbg_bf16` | inert | 15290 / 14597 |
| 10 | `kg` | `recompute_w_u` store | `kg_bf16` | inert | 25230 / 22860 |
| 11 | `v_new = u - w·h` | `chunk_delta_h` SAVE_NEW_VALUE store | `vnew_bf16` | inert (v_new = u) | 51854 / 54810 |
| 12 | `q*scale`, then `*exp2(g)` (two casts) | O kernel: `(b_q*scale).to(b_q.dtype)` then `(b_q*exp2(b_g)).to(b_q.dtype)` | `qg_bf16` | inert | 27077 / 23510 |
| 13 | final output store | O kernel `tl.store(p_o, b_o.to(...))` | always on | (always) | (always) |

Notes:

- Casts 7–12 are inert for a single chunk with zero initial state (h = 0);
  they become live at M128 (two chunks) and are all required there.
- The M5 case needs exactly casts 1–6; dropping any leaves ≥ 533 mismatches
  (`findings.md`, confirmed in `fla-emulation-m5.json`).
- Conv products (`x*w`) are **not** BF16-rounded before accumulation; the
  products are exact in f32 anyway.
- Beta and gate values are stored f32, never BF16.

### 3.3 (b) Precision per stage

| Stage | Precision / order (reference → CPU oracle model) |
|---|---|
| conv acc | f32, taps j = 0..3 in order; BF16×BF16 products exact in f32 |
| conv SiLU | GPU `acc/(1+tl.exp(-acc))`; CPU oracle models f64 SiLU with one f32 rounding (numpy f32 SiLU misses one exact tie at M5 rank1) |
| q/k L2 norm (prefill) | f32 sum-of-squares + `eps=1e-6`; GPU `tl.rsqrt` (approx — open question Q2); CPU oracle: correctly rounded `1/sqrt`; store BF16 |
| beta | f32 sigmoid of BF16 input via the model's torch.compile'd `_cast_sigmoid`: `div.full(1, 1 + ex2.approx(-x·log2e))` (§3.6a; eager ATen sigmoid is **wrong**); decode: in-kernel `tl.sigmoid` on the f32-loaded value |
| gate | f32: `y = -5/(1+exp(-(exp(A_log)*(g1+bias))))`; `tl.exp` on GPU |
| gate cumsum | **chunk-local** (restarts every 64 rows), sequential f32, then × `RCP_LN2 = 1.4426950408889634`; reference computes it as `tl.dot(tril_mask, gate, allow_tf32=False) * RCP_LN2` (f32 ieee dot); stored f32. `cumsum='f64'` is measurably worse (2016 vs 1728); pairwise equals sequential |
| A / Aqk diagonal (16×16) | f32 elementwise + f32 `tl.sum` per column; `Aqk *= scale` in-operand; stored f32 |
| A / Aqk off-diagonal | tf32 mma with **both operands truncated** to tf32 (measured: truncation 1728/585; `rna` ≈ 58k; IEEE ≈ 56k), f32 accumulator, K sliced `BK ∈ {32,64}` (reference autotune; emulation default 64, 32 gives 1727/609); `A *= beta_i` after the K loop |
| solve_tril | f32 forward substitution on diagonal blocks; off-diagonal merges via `input_precision="ieee"` (exact f32) dots; stored BF16 (cast #3) |
| u dot | BF16 × BF16 operands (Ai, vb), f32 acc, reference slices `BV=64`; `input_precision="ieee"` (see Q4) |
| w dot | BF16 × BF16 operands (Ai, kbg), f32 acc, reference `BK=64` slices |
| delta h | state carried in f32 registers across chunks; consumers see BF16 casts of it; `bv` = sum of two K=64 BF16 dots; `h *= exp2(gk_last)` f32; `h += kg^T v_new` BF16 operands, f32 acc; `final_state` f32 |
| output | BF16 dot (qg·h^T) over `BK ∈ {32,64}` slices (measured insensitive: identical counts at 32 vs 64) + BF16 dot (tril-Aqk · v_new, `allow_tf32=False`), f32 accumulator sum; store BF16 |
| all `exp2` decay factors | f32 `exp2` in the reference (GPU `ex2.approx` — open question Q1); CPU oracle uses `np.exp2` on f32 inputs |

Chunk geometry: `BT=64`, `BC=16`, `D=K=V=128`; two chunks for the M128
fixture (64 + 64 rows).

### 3.4 (c) Decode fused_recurrent policy

Reference: `fused_recurrent_kda` (`fla/ops/fused_recurrent.py` +
`fla/ops/kda.py` `fused_recurrent_kda_fwd`), wrapper lines in
`glm5next/nvidia/kda.py` (`sigmoid_beta=True`, `COMPUTE_GATE`,
`use_qk_l2norm_in_kernel` irrelevant because in-kernel). CPU model:
`emulate_fla_ext.decode()`.

- Recurrent state `h` is **f32** in and out (initial state loaded `.to(f32)`,
  stored to f32 cache). A BF16 state is wrong: 725/752 mismatches
  (`fla-emulation-ext.json`, `state_bf16_*`).
- Initial state for the captured decode (invocation 000325) = the emulated
  M5 prefill final state (f32, unrounded); conv history = the last 3 **raw**
  qkv rows of the M5 prefill.
- Conv: same policy as prefill (f32 acc j=0..3, SiLU, BF16 out).
- q/k L2 norm computed **in-kernel in f32** (`x/sqrt(sum(x^2)+1e-6)`), never
  stored BF16 — this is the one policy difference vs prefill.
- `q *= scale` in f32 (no store).
- Gate computed in-kernel in f32 from raw `g1 + bias`, `exp(A_log)`, safe-gate
  `-5/(1+exp(-(a*(g1+bias))))`; `h *= exp(gate)` in f32 (natural `exp`, no
  `RCP_LN2` — the decode path never enters log2 space).
- Per row: `v -= sum_k(h*k)` f32 acc → `beta = sigmoid(raw)` f32 → `v *=
  beta` → `h += v ⊗ k` f32 → `o = sum_k(h*q)` f32 → store BF16 (cast #13).
  v and o are processed in slices (`BV = min(next_pow2(V), 8) = 8`); K is a
  single `BK = 128` slice.
- Ordering in `decode()`: conv → l2norm → scale → gate → `h*exp` → `v-=h·k`
  → sigmoid(beta) → `v*=` → `h+=` → `o` — matches the reference token loop.

### 3.5 (d) PTX operations required on GB10

| Op | Where | PTX / instruction |
|---|---|---|
| BF16 rounding (all 13 casts) | every cast above | `cvt.rn.bf16.f32` |
| `exp` / `exp2` | conv SiLU, gate, `exp(A_log)`, all decay factors, decode `exp` | Triton lowers to `ex2.approx.f32` after `mul.rn.f32 x, log2e` for `exp`; `exp2` → `ex2.approx.f32` — **not** bit-modelled by NumPy (Q1) |
| division | conv SiLU `acc/(1+exp(-acc))`, decode L2 `x/sqrt(...)`, beta/gate denominators | candidate ops: `div.rn.f32` vs `div.full.f32`; `findings.md` residual hypothesis names `div.full` (Q2) |
| rsqrt | prefill L2 norm `tl.rsqrt(ss + 1e-6)` | `rsqrt.approx.f32` (or `ex2.approx(-0.5*lg2)`) vs the CPU oracle's `sqrt.rn` + reciprocal (Q2) |
| tf32 mma | off-diagonal KKT dots | `mma.m16n8k16.row.col.f32.tf32.tf32.f32` fed **directly from f32 registers**; hardware truncates the low mantissa bits. Do **not** emit `cvt.rna.tf32.f32` (RNA ≈ 58k mismatches vs measured 1728/585 with truncation, `probe-m128-tf32.json`) (Q3) |
| bf16 mma | u, w, qg·h, tril-Aqk·v, kg·v_new dots | `mma.*.f32.bf16.bf16.f32` with f32 accumulator; k-step order is hardware-defined (Q4) |
| ieee f32 dots | gate cumsum (`allow_tf32=False`), solve_tril merges, O-kernel `tril(Aqk)·v` annotation | exact f32 accumulation — Triton lowers `input_precision="ieee"`/`allow_tf32=False` f32 dots to SIMT `fma.rn.f32` chains or non-tf32 mma (Q4) |
| f32 accumulation generally | conv acc, sums, state updates | `fma.rn.f32` where contracted |
| reductions | diagonal KKT `tl.sum`, decode sums | tree order unspecified by the spec; probes show order alone does not explain the residual (Q6) |

### 3.6 (e) EXL3-only opt-in dispatch in `Glm5KdaLayer`

Design constraints: ordinary models must be untouched; no default-path
behaviour change; fail fast rather than fall back (PCND).

1. **Policy enum.** Add `enum KdaArithmeticPolicy { CorrectlyRounded,
   FlaReference }` beside `KdaRecurrentPath` in
   `crates/spark-model/src/layers/glm5_kda/` (new small module, e.g.
   `reference_policy.rs`, ≤500 LoC), parsed from
   `ATLAS_KDA_FLA_REFERENCE` with the strict 0/1 semantics of
   `parse_register_resident_prefill` (`recurrent.rs`). Default when unset:
   `CorrectlyRounded` — identical kernel selection to today.
2. **EXL3 gating at assembly.** In `assembly.rs::Glm5KdaLayer::new`, if the
   policy is `FlaReference`, `ensure!` that all four projections are
   `Glm5Projection::Exl3` (the `exl3.rs` `validate_exl3_forward` precedent),
   `heads == 32 && dim == 128 && lower_bound == -5` (the FlashKDA
   qualification precedent), and that `ATLAS_KDA_FLASH_PREFILL=1` and
   `ATLAS_KDA_REGRESIDENT_PREFILL=1` are not also set (mutual exclusion).
   Setting the env on a non-EXL3 checkpoint is a startup error, never a
   silent toggle.
3. **Kernel handles.** Resolve new handles only when the policy is on:
   `fla_prefill_k = gpu.kernel("kda", "kda_fla_chunk_prefill_bf16")` and
   `fla_decode_k = gpu.kernel("kda", "kda_fla_recurrent_decode_bf16")`
   (`KernelHandle(0)` otherwise), in a new
   `kernels/gb10/common/kda_fla_reference.cu` (flagged as a shared-file
   change in the PR).
4. **Dispatch seams.**
   - Prefill: `sequence_recurrent_rows` (`forward_recurrent.rs`) gains a
     first arm for `FlaReference` (policy kernel over packed conv output),
     before the FlashKDA filter and `run_recurrent` — the same shape as the
     `flash_prefill` seam.
   - Decode: `run_recurrent` (`recurrent.rs`) gains a `FlaReference` arm
     for `decode == true` (fused_recurrent policy kernel).
   - `decode_multi_seq_inner` and `prepare_indexed_core` (`multi_seq.rs`,
     `indexed_core.rs`) `ensure!` the policy is off for the first slice —
     multi-sequence/indexed decode is out of scope until single-row decode
     passes its gate.
5. **Invariant.** With the env unset, no enum variant, kernel handle, or
   launch differs from today's behaviour; existing dispatch tests keep
   passing unchanged. The policy only ever makes the *reference* surface
   bit-exact; it does not relax any tolerance anywhere.

### 3.6b As implemented (2026-10-11)

The shipped implementation differs from the proposal above in three ways:
one CUDA kernel per reference Triton kernel in module `kda_fla`
(`kernels/gb10/common/kda_fla*.cu(h)`; launch order in `kda_fla.cu`) instead of
the two proposed kernels; prefill intermediates borrow the model-owned routed
MoE scratch buffer 7 (idle during attention on the same stream, size checked at
assembly) instead of a new allocation; and a one-row invocation always takes the
decode kernels, matching vLLM's `query_len == 1` classification. Rust:
`crates/spark-model/src/layers/ops/kda_fla.rs` (launches, scratch layout) and
`crates/spark-model/src/layers/glm5_kda/reference_policy.rs` (policy, gating,
dispatch). Recurrent state keeps Atlas's `[K, V]` layout.

### 3.6a GPU oracle and residual closure (2026-10-10)

`tests/exl3_validation/kda_reference_stage_dump.py` replays the pinned
reference (installed sources hash-identical to
`reference-kda-kernel-source/`) on GB10 inside the reference image, stage
by stage, from the captured `raw/` inputs. With the two corrections below it
reproduces **every captured output byte**: M5 0/0, decode 0/0, M128 0/0, on
both ranks, for both the staged replay and the model's varlen entry point.
The per-stage bytes and digests are archived at
`arithmetic-audit/kda-reference-stage-oracle-20261010/` (`SHA256SUMS`); they
replace the NumPy emulation as the acceptance oracle for T2–T6.

1. **Conv weights are f32.** The reference builds `q/k/v_conv1d` with
   `params_dtype=torch.float32`, so `x * w` is exact in f32 (Triton would
   otherwise round BF16×BF16 products to BF16).
2. **Beta comes from Inductor, not ATen.** `_cast_sigmoid` runs inside the
   model's `torch.compile` graph. Its Triton lowering is
   `div.full.f32(1, add.f32(ex2.approx.f32(mul.f32(-x, log2e)), 1))`. Using
   eager `sigmoid` instead leaves exactly the recorded residual pattern
   (rank0: 231 elements, one head cascading from one row; rank1: 10). The
   NumPy M128 residual (1728/585) is therefore explained by beta and the
   approximate transcendental ops, not by reference nondeterminism.

The reference is deterministic across repeats and, on the M128 fixture,
insensitive to every autotune configuration of the gate, KKT, solve, w/u
and output kernels (`kda_reference_config_sweep.py`). Lowered PTX
(`sm_121a`, Triton 3.7.1, archived Triton caches in
`reference-triton-cache-20261010/`) pins the open operations:

| Stage | Instructions |
|---|---|
| conv SiLU | `fma.rn.f32` taps, `ex2.approx.f32`, `div.full.f32` |
| prefill L2 norm | `fma.rn.f32.bf16` sum of squares, `rsqrt.approx.ftz.f32` |
| gate + chunk cumsum | `ex2.approx.f32`, `div.full.f32`, `fma.rn.f32` cumsum chain |
| KKT off-diagonal | `mma.sync.m16n8k8.f32.tf32.tf32.f32` (no `cvt.*.tf32`), `ex2.approx.ftz.f32` |
| KKT diagonal | `fma.rn.f32`, `ex2.approx.ftz.f32` |
| solve_tril merge | `fma.rn.f32` (ieee dots), `cvt.rn.bf16x2.f32` |
| recompute w/u, delta h, output | `mma.sync.m16n8k16.f32.bf16.bf16.f32`, `ex2.approx.ftz.f32`, `fma.rn.f32x2` |
| decode | `div.full.f32`, `ex2.approx.f32`, `sqrt.approx.ftz.f32`, `fma.rn.f32.bf16` |

### 3.7 (f) Known residual risk (M128) and GPU closure plan

Superseded by §3.6a for the captured fixtures; retained for history.

The emulation reproduces the captured M128 reference with 1728 (rank0) and
585 (rank1) mismatches of 524288 elements (chunk split 745/983 and 190/395),
mostly 1-ulp flips in the BF16 output. Ruled out by probes: tf32 operand
rounding mode (dominant, now pinned to truncation), tensor-core k-order
(≤ 18 flips, `tc_ks*` variants), `tl.sum` order, gate/cumsum/SiLU f32-vs-f64
variants (shift counts ±300 but do not eliminate). Remaining hypothesis
(`findings.md`): bit-level `ex2.approx` / `div.full` (and `rsqrt.approx`)
behaviour that NumPy cannot model.

Closure plan on GPU (ordered):

1. Standalone PTX probes in `tests/exl3_validation/` style (cf.
   `native_ptx.py`, `mma_probe.py`): for the exact operand bit patterns of
   the conv SiLU, gate, and L2-norm stages, compare `ex2.approx.f32` vs
   IEEE `exp2`, `div.full.f32` vs `div.rn.f32`, `rsqrt.approx.f32` vs
   `sqrt.rn` + reciprocal, and tf32 mma input truncation, on GB10.
2. Stage-tensor diff: run the pinned Triton reference on GB10 with the
   captured `raw/` inputs and dump each stage (conv out, l2, g, A, Aqk, Ai,
   u, w, kg, kbg, qg, h, v_new, o) as raw bytes; diff bitwise against the
   new CUDA kernel stage by stage. The first divergent stage attributes the
   residual to one operation (Q1/Q2).
3. Freeze per-stage sha256 digests beside `raw/` as the acceptance oracle
   for the kernel work; re-baseline the emulation only with evidence from
   steps 1–2, never by loosening the comparison.

Residual expectation until then: the CPU harness asserts the *exact*
recorded counts (1728/585) — this is an evidence record, not a tolerance;
the target for the CUDA kernel remains `vs_ref == 0` once Q1/Q2 close.

### 3.8 (g) Validation gates and fixtures

Fixtures: `arithmetic-audit/kda-core-divergence/raw/` (both ranks, ~10.6 MB,
present; sidecars carry shape/dtype). Harness:
`tests/exl3_validation/test_exl3_kda_reference.py` (CPU-only; skips with a
clear message when the external directory or a fixture is absent; override
the location with `ATLAS_KDA_DIVERGENCE_DIR` only if the emulation scripts
themselves are relocated, since they hard-code their own root).

| Gate | Fixtures | Expected bit-exactness | Harness test |
|---|---|---|---|
| Prefill single-chunk (M5, inv 000324) | `ffn-L0-rank{0,1}-000324-kda-core-{input-qkv_proj_states,input-g1,input-beta,output}.bin` | emulation vs captured reference: **0 / 0** both ranks | `test_m5_prefill_matches_reference_bit_exact_both_ranks` |
| Decode after M5 (inv 000325) | `ffn-L0-rank{0,1}-000325-kda-core-{...}.bin` (+ M5 inputs for state/history) | **0 / 0** both ranks with f32 state and f32 l2norm | `test_decode_after_m5_prefill_matches_reference_bit_exact_both_ranks` |
| Multi-chunk (M128, inv 000016) | `ffn-L0-rank{0,1}-000016-kda-core-{...}.bin` | documented residual, exact counts **1728 / 585** (chunks 745/983, 190/395); cross-checked against `fla-emulation-ext.json` | `test_m128_matches_recorded_residual_both_ranks`, `test_recorded_report_states_frozen_expectations` |
| Evidence consistency | `fla-emulation-ext.json` | records M5 0/0, decode f32-state 0/0 (BF16 state 725/752), M128 1728/585 | same as above |

Context (not a gate for the new kernel, recorded for attribution): Atlas's
current core vs the captured reference differs in 11163/11238 (M5),
1287/1011 (decode), ≈292891/298964 (M128) elements — the reference policy
targets the *reference* bytes, not Atlas's current bytes.

Note on reproducibility: the M128 exact counts assume the same NumPy/BLAS
accumulation order as the machine that produced `fla-emulation-ext.json`
(observed: `inter_bk` 32/64/128 move the count by ≤ 15). If a different CPU
host flips the recorded counts, re-run `emulate_fla_ext.py` there and
re-freeze the evidence — do not widen the assertion.

## 4. Implementation plan (CUDA kernel work)

Ordered tasks; each ends with its own acceptance check.

1. **T1 — Freeze the CPU oracle (done as part of this groundwork).**
   `tests/exl3_validation/test_exl3_kda_reference.py` green on CPU; it is
   the acceptance oracle for every later task.
2. **T2 — Prefill stage kernels.** One CUDA kernel per reference Triton
   kernel (conv, l2norm, beta, gate cumsum, KKT inter/intra, solve_tril,
   recompute w/u, delta h, output), each byte-compared against its GPU
   oracle stage (§3.6a) for every prefill fixture, both ranks.
3. **T3 — PTX decisions.** Closed by the archived lowered PTX (§3.6a table);
   kernels use the exact instructions via `kernels/gb10/common/kda_fla_ops.cuh`.
4. **T4 — Multi-chunk and carried state.** Covered by the oracle `seq*`
   fixtures (tails, 3 chunks + tail, non-zero initial conv/recurrent state);
   acceptance is 0 mismatches on every stage.
5. **T5 — Close the M128 residual.** Closed: beta lowering and f32 conv
   weights (§3.6a). The replay reproduces the captured bytes exactly.
6. **T6 — Decode fused_recurrent kernel.** `kda_fla_recurrent_decode_bf16`
   per §3.4 (f32 state, in-kernel f32 l2norm, no BF16 l2norm).
   Acceptance: 0/0 both ranks at inv 000325.
7. **T7 — Dispatch wiring.** Policy enum, assembly gating, seams (§3.6);
   default-path invariance test (env unset ⇒ identical kernel selection);
   `cargo fmt`, `ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo clippy
   --workspace --tests`, license-header and typo checks per `AGENTS.md`.
8. **T8 — GPU validation.** Rebuild the validation image from scratch and
   run the KDA slice of `tests/single_gpu_suite.py` plus the frozen parity
   gates; no tolerance changes. Flag `kernels/gb10/common/` and
   `glm5_kda/` seam edits as shared changes in the PR description.

### Open questions (emulation could not model GPU behaviour)

- **Q1 — `ex2.approx.f32`:** does the installed reference lower every
  `tl.exp`/`tl.exp2` to `ex2.approx.f32` (with `mul.rn.f32 x, log2e` for
  `exp`), and what is the correctly-rounded-vs-approx delta on the actual
  operand bit patterns (conv SiLU, gate, decay factors)? Primary residual
  suspect.
- **Q2 — division/rsqrt:** `div.rn.f32` vs `div.full.f32` for the conv
  SiLU and decode L2 division; `rsqrt.approx.f32` vs `sqrt.rn` + reciprocal
  for the prefill L2 norm. `findings.md` names `div.full` as hypothesis.
- **Q3 — tf32 operand conversion:** truncation of both operands is
  measured best (1728/585 vs ≈58k for RNA) — confirm on GB10 that the
  hardware/PXJ path truncates f32→tf32 inputs for `mma.*.tf32` and that no
  `cvt.rna.tf32.f32` is inserted.
- **Q4 — dot lowering:** which exact instruction implements (a) the
  `input_precision="ieee"` u dot on BF16 operands, (b) the `allow_tf32=False`
  f32 dots (gate cumsum, solve_tril merge, O-kernel Aqk·v), and (c) the bf16
  mma k-step order for u/w/qg·h/v_new·kg; probes vary ≤ 18 flips but the
  tie cases at M128 may still depend on it. Also: the off-diagonal KKT
  autotune config actually used by the frozen capture (`BK=32` vs `64` —
  1727/609 vs 1728/585, not distinguishable from outputs alone; recover
  from the reference image's Triton cache/PTX fingerprint).
- **Q5 — O-kernel autotune config:** `BK ∈ {32,64}`, `BV ∈ {64,128}` —
  measured insensitive on the fixtures (identical counts at 32 vs 64);
  confirm from the reference image which config ran, for faithfulness.
- **Q6 — reduction order:** `tl.sum` tree order in the diagonal KKT kernel
  and the decode reductions; probes indicate it is not the residual driver,
  but per-stage digests will confirm.

## 5. Discrepancies noted between `findings.md` and the emulation scripts

- `findings.md` headline ("six BF16 rounding points … any single omission
  leaves ≥ 533") is confirmed exactly by `fla-emulation-m5.json`; the M128
  extension adds seven more rounding points that the headline does not
  mention — §3.2 above merges both.
- `emulate_fla_ext.py` models the off-diagonal KKT dot with
  `inter_bk=64` slices by default; the reference source autotunes
  `BK ∈ {32,64}` (the launcher does **not** pass the `BK = max(next_pow2(K),16)`
  value — that one belongs to the *diagonal* intra kernel). Slicing is
  near-immaterial (≤ 15 flips across `inter_bk` 32/64/128) but is not pinned by the recorded evidence
  (see Q4).
- The emulation carries the inter-chunk `h` in unrounded f32 and rounds only
  at each consumer read (`hs = B(h)`); the reference keeps `h` in f32
  registers across chunks within one `chunk_delta_h` launch and stores BF16
  per chunk for consumers — the two agree for the captured ≤ 2-chunk cases.
  For ≥ 3 chunks the final-state bytes would need an explicit check.
- The gate cumsum resets per 64-row chunk in both the emulation and the
  reference (`chunk_local_cumsum`); it is *not* a running prefix over the
  whole invocation. Any re-derivation that "fixes" this would diverge from
  the captured reference.
