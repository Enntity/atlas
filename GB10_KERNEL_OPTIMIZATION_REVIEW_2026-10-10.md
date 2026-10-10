# GB10 Kernel Optimization Review — 2026-10-10

Scope: (1) the CUDA kernels added in the current uncommitted change on
`sparkglm/installable-20260924` (EXL3 support for GLM-5.3 Flash:
`kernels/gb10/common/exl3*`, `glm_indexer_quant.cu`, `glm_indexer_score.cuh`,
the `glm_indexer.cu` diff, and the new `kernels/gb10/glm-5.3-flash-exl3/exl3/`
target); (2) the hot-path kernels of the project as a whole; (3) whether 17
external repositories can add value on GB10.

> **Update (same day):** a cross-reference against local prior-art projects is in
> `GB10_PRIOR_ART_CROSSREF_2026-10-10.md`. It **corrects** CB1 (about half the
> estimated win; the reference keeps shared experts BF16 because of a barrier-buffer
> race, not accuracy), the NCCL small all-reduce cost (~50 µs measured, not ~100),
> the §3.6 indexer gate GEMV (~73 µs/layer, not ≥7), and the bandwidth denominator
> (~240 GB/s achievable, measured). It also adds new items (spill audit,
> compaction sysctl, managed-memory fallback guard, DFlash cost table, among others).
> Corrections are flagged inline below with **→ X#**.

**No existing project file was modified.** This report is the only file written
to the tree. It complements, and does not repeat, `CODE_REVIEW_2026-10-10.md`
(which covers EXL3 MoE scratch sizing, F32 vocab-head staging and missing FFN
stage timing — all three are consistent with findings here).

## How to read the numbers (measurement discipline)

Nothing was compiled, profiled or run on a GPU for this review (the review host
has no nvcc/GPU). Every number carries one of these labels:

| Label | Meaning |
|---|---|
| **[E]** | Estimate from reading code and doing byte/instruction arithmetic. A hypothesis, not a measurement. Arithmetic shown where it matters. |
| **[R]** | A number already recorded in-tree (commit message, source table, bench note). Cited by commit/file. Fingerprint is whatever that record carries. |
| **[R?]** | A number the sub-review attributed to an in-tree record, but whose source this review could not re-locate. Treat as unverified until the source is found. |
| **[C]** | A third-party claim from an external repo's README/docs. Not reproduced. Usually different hardware, engine, model or bit-rate — not comparable to Atlas as an A/B. |

None of the words "real / regression / inert / impossible" are used as
conclusions below; every item ends in an experiment. The code-review-graph MCP
tools named in `~/CLAUDE.md` were not available in this session, so the review
used direct file reads; ~16 of the sub-reviewers' `file:line` references were
spot-checked by hand (marked ✔).

## GB10 facts used

| Fact | Value | Source |
|---|---|---|
| Arch | sm_121f (CC 12.1), consumer Blackwell ISA: `mma.sync` incl. `kind::mxf4nvf4` block-scaled FP4 and FP8; **no** tcgen05/TMEM/wgmma; TMA tensor loads work, multicast/clusters>1 not used | `kernels/gb10/HARDWARE.toml`; TMA use in `crates/spark-runtime/src/cuda_backend/tensormap.rs` |
| DRAM | 119.7 GB unified LPDDR5X, **273 GB/s spec** | `HARDWARE.toml`, `docs/GB10_DEPLOYMENT_GUIDE.md:30` |
| Achieved DRAM | **→ X4: ~240 GB/s achievable plain-read ceiling, measured on 2 nodes (VML, MR)**; Marlin NVFP4 184–240 GB/s at M≤8 [R 3a6de55f]; routed MoE 222–235 GB/s [R 54507b8c]; kernel-ladder probe 231 GB/s [C] | in-tree commits; BEEB0SS/kernel-ladder |
| SMs | 48 (used by `NUM_SMS` split heuristics in-tree) | source; assumed, not re-probed |
| L2 | 24 MB [C kernel-ladder probe]; note `gated_delta_rule.cu:1007,1292` comments say "64 MB" — inconsistent, fix the comment | ✔ |
| Smem | ~99 KB per block opt-in [C kernel-ladder, sm120_nvfp4_ops] | assumed |
| Tensor peaks | BF16 ≈ FP8 ≈ 213 TFLOPS, dense FP4 ≈ 427 TFLOPS [C kernel-ladder] → **FP8 only wins by moving fewer bytes; FP4 wins on both bytes and math** | third-party probe |
| Interconnect | 2× GB10 via ConnectX-7, ~14 GB/s per PCIe half; no GPUDirect RDMA (Atlas `rdma_pair.rs` header; SWITCHLESS README) | in-tree + [C] |

Corollary that drives the whole report: at decode, Atlas's big streaming kernels
already sit at 80–88% of spec [R]. **The remaining decode wins are (a) moving
fewer bytes, (b) filling the 48 SMs during the small serial kernels between the
big ones, and (c) launch/latency overhead.** Faster math only matters where a
kernel has become ALU-bound by decoding (EXL3) or by `--fmad=false`.

---

## 1. Executive summary — top follow-ups, ranked by expected value

| # | Item | Area | Kind | Expected impact | Effort |
|---|---|---|---|---|---|
| 1 | EXL3 packed MMA GEMM fills only 4–64 CTAs at decode; split the serial "slices" loop across CTAs with an identical-order finalize fold | EXL3 (this change) | bit-identical | [E] the single largest EXL3 decode item; KDA/MLA projections plausibly 3–10× faster | M |
| 2 | EXL3 `exl3_mhc_pre_norm` runs on one CTA with 192 barrier phases + single-thread Sinkhorn | EXL3 (this change) | bit-identical | [E] 2.7–20 ms/token (90 calls/token) | S–M |
| 3 | EXL3 per-coefficient decode cost ~10× upstream (runtime `%`, two scalar loads, half→float→half) | EXL3 (this change) | bit-identical | [E] makes the head/projections bandwidth-bound instead of ALU-bound | M |
| 4 | GLM router + shared expert at ≤8 rows run on 5–32 CTAs, serially | GLM NVFP4 (project) | bit-identical | [E] 5–8% of a C1 verify step | L–M |
| 5 | **→ X1 (≈1.5 ms/step measured upstream, ~0.7 GB/token)** **Shared experts in the EXL3 path are BF16-resident** (~1.06 GB/token/rank ≈ 18% of the byte floor) | EXL3 | compute-for-bytes, needs re-qualification | [E] ~0.8 GB/token, ~2.9 ms/token, ~0.8 GB memory/rank | M |
| 6 | Use both ConnectX-7 PCIe halves (`NCCL_IB_HCA` lists both + `ATLAS_RDMA_RAILS`; A/B `NCCL_CROSS_NIC=1`) | comms | config only | [C] 110→188 Gb/s on stock NCCL 2-node; [C] +42% single-stream on a vLLM TP=2 MiMo build | S |
| 7 | PDL: prefetch static weights before `griddepcontrol.wait`; extend coverage (Qwen W4A16 hot path, all EXL3 kernels, tc16/32 tiers) | project + EXL3 | bit-identical | [E] 2–4% Qwen; EXL3 currently has zero PDL kernels ✔ | S |
| 8 | Indexer keys: store FP8 bytes (+exponent) instead of BF16 — lossless on the EXL3 path; opt-in MXFP4 keys (lossy) | indexer | compute-for-bytes | [E] 2× (FP8) / 3.8× (MXFP4) fewer indexer bytes at long context | S–M |
| 9 | GQA-grouped split-KV decode (one CTA per KV head serves the whole query group; dequant once) | attention | compute-for-bytes (L2→SM, DRAM on L2 miss) | [E] 5–10% of Qwen verify step if 12.7% attention share holds | M–H |
| 10 | Remove a network round-trip in `rdma_pair` (WRITE_WITH_IMM completion, pplx-garden style) | comms | latency | [E] one RTT per decode all-reduce × all-reduces/step | M |

---

## 2. Compute-for-bytes opportunities (the GB10-specific category)

These items trade extra ALU work, or a different format, for fewer DRAM (or
network) bytes. They are ordered by estimated bytes saved per token. Each one
names the experiment that would confirm it.

| # | Opportunity | Bytes today → proposed | Accuracy | Experiment (shape · metric · pass criterion) |
|---|---|---|---|---|
| CB1 | **→ X1** **EXL3 shared experts via the packed trellis kernel** instead of BF16 reconstruction (`glm5_exl3_ffn/dense.rs:15,83` `SharedBf16Weights` ✔) | [E] 25 MB/layer/rank BF16 → ~6 MB at ~4–5 bpw; ≈1.06 → ~0.25 GB/token | reference engine also uses BF16; needs opt-in + re-qualification | Opt-in flag; ncu `dram__bytes_read` per layer; nsys step time C1/C4; parity vs vLLM reference within frozen tolerance |
| CB2 | **NVFP4 sparse-MLA KV** (sm120_nvfp4_ops: 448 FP4 ch + 64 BF16 RoPE + E4M3 scales ≈ 384 B/token; Q quantized on the fly; QK and PV both on FP4 `m16n8k64`) | [E] 1024 B/token BF16 latent → 384 (2.7×); per layer at B=64 ≈ 42 → 16 MB | **lossy** — f0d9f9b3 [R] found NVFP4 MLA *projections* lossy (arith 40/40→37/40, needle 12/12→10/12); KV not yet tested | First: round-trip the latent through that format inside the existing kernel, rerun f0d9f9b3's arithmetic + needle probes. Only if it passes: port and A/B vs `glm_sparse_decode_split` at C1/C4/C16/C64, report KV GB/s |
| CB3 | **MLA latent stored in both K and V pools** (`attention_forward_mla.rs:342-366`, per project review) — alias V to K | [E] ~39% of MLA KV memory; readers that read both pools pay 2× | lossless | Allocator flag; KV-token capacity at fixed GB; needle + decode TPOT unchanged |
| CB4 | **Fused down-GEMM scatter-add for MoE prefill** (idea from ufg-ggemm): epilogue adds `prob × out` into `[T, H]` instead of writing `[T·topk, H]` and re-reading it in `moe_unpermute_reduce[_indexed]` (`common/moe_permute.cu`, `layers/moe/ep_prefill.rs`) | [E] T=2048, H 4–7K, topk 6–8 → 100–200 MB written + read per MoE layer (~0.9–1.7 ms at 231 GB/s if it spills L2) | atomics change summation order; use an ordered variant (one CTA reduces all topk partials of a token) for determinism | nsys: share of `moe_unpermute_reduce` in GLM/DSV4 prefill at 1K/4K/8K chunks; proceed if >3–5% of MoE time |
| CB5 | **Indexer key cache as FP8 bytes** — EXL3 path stores exact fp8×2^e values as BF16 (`glm_indexer_quant.cu`); store FP8 + 1-byte exponent; also preload `q` (8 KB) to smem/registers in `glm_indexer_score.cuh:49-59` and use 64-bit key loads | [E] 256 → 128 B/row (lossless); ds4x MXFP4 keys 68 B/row (lossy). At 256K context ≈ 184 → 92 → 49 MB/token | FP8: bit-identical scores; MXFP4: measure top-2048 overlap | Byte-identical top-k vs current on recorded traces; then MXFP4 opt-in with overlap ≥ agreed threshold |
| CB6 | **GQA-grouped decode with dequant-once** (rvLLM `v3/kernels/fp8_decode_v2.cu`, Apache-2.0: grid = kv_head × chunk × batch). Atlas `paged_decode_attn*` uses grid.x = q_head, so each query head re-reads and re-dequantizes the shared KV head | [E] L2→SM traffic ÷ GQA ratio (4–8×); DRAM ÷ GQA ratio whenever the sibling CTAs are not co-resident; dequant ALU ÷ GQA ratio for turbo/nvfp4 KV | lossless (same math, different grouping); keep split count independent of batch for determinism | First, measure Atlas decode bandwidth with L2 defeated (rotate block tables over ≥100 MB KV; `lts__t_bytes`, `lts__t_sectors_op_read`), GQA 4–8, B 1/4/16/64, ctx 2k/8k/32k, bf16/fp8/nvfp4/turbo4 |
| CB7 | **FP8 block-scaled compression of the prefill all-reduce / SP exchange** (network bytes; pplx/DeepEP-style) | [E] 67 → 34 MB per partial per layer per 8K chunk, ≈0.9 ms wire saved before overlap | lossy | Accuracy gate (KL vs BF16 AR) then 16K TTFT A/B (baseline 7.64 s [R]) |
| CB8 | **mHC tensors**: `fn` in BF16 (B4) and the persistent highway in BF16 (every reader rounds to BF16 already) | [E] ~70 MB/token (`fn`) + highway F32→BF16 | `fn` lossy-ish, highway likely exact | Parity harness on the mHC kernels |
| CB9 | **KDA/GDN state and snapshots**: production uses snapshot rollback; records mode (d0540c2e) [R] KV 940K→1.26M tokens, [R?] −2.5% step C1, −5% C4; K5 pair path hardcodes snapshots (`paired_verify.rs:130-147`). FP32 conv snapshots → BF16 (claimed lossless; verify) | [E] state bytes and capacity | records: proven; BF16 conv: verify exactness | Port pair path to records; byte-compare conv snapshot round trip |
| CB10 | **Vocab head**: rank-local argmax or BF16 logit gather instead of 310 KB F32/row through NCCL (also in `CODE_REVIEW_2026-10-10.md` P2) | [E] ~4× fewer head-gather bytes | greedy: exact; sampling: needs BF16 path | nsys of head gather at C1/C4 |
| CB11 | **FP4-MMA consumption of NVFP4 KV** in `paged_decode_attn_nvfp4.cu` (currently CUDA-core dequant, no `mma`): quantize Q on the fly, use `mxf4nvf4` | same bytes, much less ALU | Q quantization is lossy | Matters only where the kernel is ALU-bound (verify rows, long ctx). Note [C] mimo-spark & helios both found low-bit KV on GB10-class parts buys **capacity, not speed** — confirm before investing |

Background compute-for-bytes items that are already strong in Atlas, for
context: NVFP4 everywhere with smem E2M1 LUT, MXFP8/FP8 twins, TurboQuant 2–8-bit
KV, fused SiLU·mul + NVFP4 requant epilogue, and EXL3 itself (≈4 bpw weights
decoded in-register). The EXL3 kernels are the clearest current case where *the
decode cost itself* has to fall for the format's byte savings to show (§3.1–3.3).

Evaluated as **not** a bytes win on GB10: a fully fused persistent
gate-up-SiLU-down MoE at decode/verify. The intermediate is ~64–460 KB and
stays in L2, so fusing saves launches and tails, not DRAM bytes. CUDA graphs,
PDL and the fused routed+shared C4 combine already target those
(FlashMoE/UltraEP §6).

---

## 3. Findings — kernels in this change (EXL3 / indexer)

**Context.** `docs/design/exl3-support.md` already states that the current
matmul decodes each coefficient on demand, that optimized kernels follow
correctness validation, and that full-model parity vs vLLM is still failing. So
the findings below are not discoveries that EXL3 is slow. They are a concrete
plan for making it fast, split into:
- **[A] bit-identical:** same per-element arithmetic and fold order; only the
  parallelization changes. These can proceed during parity work.
- **[B] arithmetic-changing:** these need re-qualification after parity.

**Scope note.** Every file in
`kernels/gb10/glm-5.3-flash-exl3/exl3/` is byte-identical to a file in
`deepseek-v4-flash/nvfp4` or `longcat-flash-lite/nvfp4`. Its `KERNEL.toml` is
identical to the DSV4 one (checked with `cmp` ✔). The genuinely new kernel code is in `common/exl3*`
and the indexer files.

**Per-token byte floor, EXL3 TP=2, per rank** [E]. This assumes 42 MoE layers
(`first_k_dense_replace=3`); verify against config.json.

| Component | GB/token |
|---|---|
| KDA Q/K/V/O, 6-bit, 34 layers | 1.71 |
| MLA, 6-bit, 11 layers | ~0.50 |
| Routed experts, ~4-bit, top-8 | 2.11 |
| **Shared experts, BF16** | **1.06** |
| Dense FFN 5-bit / router / mHC `fn` / head 6-bit | 0.14 / 0.10 / 0.14 / 0.24 |
| **Total** | **≈6.0 GB → ≈22 ms/token at 273 GB/s (≈45 tok/s ceiling, before speculation)** |

### 3.1 [A] Packed MMA GEMM under-fills the GPU at decode — highest EV
`exl3_mma.cuh:44-73` ✔, `exl3_linear_mma.cu:31-44`, `exl3_moe_mma.cu:44-56`;
the grid is set in `native.rs:262-279`.

Each CTA has 4 warps, and each warp walks 4 N8 fragments in sequence. Inside a
fragment, the reference's `slices` (48 for linears, 8 for MoE) run as a **serial
loop in one warp** (`for (s = slices; s > 0; --s)` ✔). This reproduces the
reference's split-K fold order, but in the reference those slices were parallel
CTAs.

At M=1 the grids are [E]:
- KDA Q/K/V and O: 32 CTAs.
- MLA q_a: 12 CTAs.
- **kv_a (N=512): 4 CTAs.**
- MoE gate/up: 64 CTAs.

Each warp is a chain of ~1024 dependent k16 steps, so memory latency cannot be
hidden. [E] That is ~12–25 KB in flight device-wide, roughly 10–25 GB/s. A
12.6 MB projection (ideal ~46 µs) may take 0.15–0.6 ms, and there are 136 such
KDA projections per token.

**Fix (bit-identical):**
- One CTA (or warp) per (fragment, slice segment). This gives ~1300 short chains
  instead of ~128 long ones.
- Write F32 partials per segment.
- A finalize kernel performs the **same serial fold**
  `total = __fadd_rn(sum_s, total)` from s=slices down to 1, not a tree. Fuse
  output H128 and `svh` into that finalize.
- Validate with the existing packed byte-gate controls.

**Confirm:** ncu `dram__throughput.pct_of_peak_sustained_elapsed`,
`sm__warps_active`, `smsp__pcsamp_warps_issue_stalled_long_scoreboard`.

### 3.2 [A] mHC `exl3_mhc_pre_norm` is a single-CTA serial chain
`exl3_mhc.cu:109-189`.

At ≤16 rows it runs on one CTA:
- **Reductions:** 24 mixes × 8 splits = 192 passes over a fresh 1.57 MB `fn`.
  Each pass ends in `projection_reduce`, with 3 barriers and a serial 8-warp
  sum.
- **Sinkhorn:** 20 iterations × 32 dependent `__fdiv_rn`, all on lane 0.
- **Tail:** only 64 lanes are active.

[E] ~100–200 µs/call × 90 calls/token ≈ 10–20 ms/token; even 30 µs/call is
2.7 ms/token.

**Fix (bit-identical):**
- One pass with 24 accumulators per thread, keeping each mix's own FMA order,
  followed by butterfly reductions.
- Spread the splits across CTAs and fold them in order in the last CTA (atomic
  ticket).
- Sinkhorn on 16 lanes, reproducing `sum4` as `(a+c)+(b+d)` via shuffles.
- Copy the multi-CTA structure from the NVFP4 path's
  `glm_hc_decode_partial/finalize`.
- The TF32 path for >16 rows (`exl3_mhc_tf32.cuh:24-25`) carries one real row
  per M16 fragment, and every row's CTA re-reads the whole of `fn`.

**Confirm:** nsys time per token in `pre_norm`; ncu `stalled_barrier`. **This is
the best measured-win-per-effort candidate. Profile it first.**

### 3.3 [A] Coefficient decode ≈10× upstream's instruction cost
`exl3_common.cuh:39-53` ✔, `exl3_mma.cuh:7-17,50-62`.

For every coefficient, `tile_state` does:
- two scalar 32-bit loads, `words[first % count]` and `words[last % count]`, with
  a runtime modulo (a real integer division);
- a 64-bit shift;
- the codebook, then a half→float→half round trip (`pair_bits`).

That is [E] ~50–70 instructions per coefficient. The L1→register traffic is
~8 B per coefficient against a 0.75 B ideal at 6 bits. A lane's B fragment needs
stream states 8l..8l+7, which form one contiguous 2–3-word window.

**Fix**, following upstream exllamav3 (MIT; keep the notice if code is ported):
- One warp per 16×16 tile.
- `cp.async` multi-stage staging.
- Funnel-shift extraction.
- `mul1` byte-sum as `__dp4a(p, 0x01010101, 0)`.
- Two values per HFMA2, returning `__half` directly.

Decoded codebook values are exact FP16, so the MMA fragments stay bit-identical.
Also check whether `float acc[4][4]`, indexed by runtime `sub`, lives in local
memory: compare the `ptxas -v` stack frame and LDL count. Templating on
`tile_k ∈ {16,32}` removes the dead 48/64 branches.

[E] The head is ALU-bound today (≈1.3 ms against a 0.87 ms DRAM floor).

### 3.4 [A] Multi-row: weights refetched/decoded per row; MoE not grouped by expert
- **Rows are re-read:** rows sit in `grid.y`, and the M16 A fragment carries
  only one real row. Packing up to 16 rows into M is bit-identical, because each
  output depends only on its own row.
- **Head at M=8:** [E] the 238 MB head exceeds L2, so DRAM traffic is ~8× ideal.
- **MoE at a 128-row prefill chunk:** [E] 1024 slots against ~277 distinct
  experts, ~3.7× redundant.
- **Fix:** build per-(expert, 16-row) work items from the existing `counts`,
  exactly like the NVFP4 `moe_sort_by_expert` / `moe_build_tile_worklist`.
- **Why it matters for speculation:** this is a prerequisite for EXL3
  DFlash/MTP verify. helios [C] measured a 4-row verify at 6.8× one row when
  experts are not grouped.
- **Launch design:** the routed path is already right for GB10. It has no
  per-expert launch loop and no host readback (indirect `ids`/`table`
  descriptors).

### 3.5 [A] Launch count, Hadamard fusion, PDL
- **Q/K/V are components of one fused tensor with a replicated `suh`.** Today
  the input H128 is computed 3× and three 32-CTA matmuls launch
  (`forward_qkv.rs`). One launch with logical N=12288 is bit-identical and
  triples the grid.
- **MoE gate and up** each launch their own input H128 (`moe/forward.rs:105-106`).
- **Fusion:** input H128 fits the GEMM prologue for its own k-range, and output
  H128 lines up with the CTA's N128 tile in the epilogue. At M=1 this is a
  launch/latency win, not a bytes win. [E] ~1600 launches/token.
- **PDL:** no EXL3 kernel is in `PDL_KERNELS`
  (`crates/spark-runtime/src/cuda_backend/gpu_impl/pdl.rs:12` ✔).
- **Smaller items:**
  - `launch_f32` looks up its kernel on every call (`native.rs:212`).
  - `split_merged_bf16` issues two extra 2D memcpys per forward.
  - `hadamard128` uses 14 `__syncthreads`; four elements per thread plus
    `shfl_xor` needs none.
- [E] 1–3 ms/token combined.

### 3.6 [A] Smaller bit-identical items
- **MoE top-k on one thread.** `exl3_topk_sigmoid` runs top-k serially on
  thread 0 (`exl3_moe.cu:154-170` ✔). [E] ~10 µs/layer, ~0.4 ms/token. A warp
  argmax on `(value, -id)` reproduces the lowest-id tie-break.
- **mHC post/contract** (`exl3_mhc.cu:236-273`): one CTA per row, reloading
  `comb`/`post` per feature. [E] 0.5–0.9 ms/token. Grid over `hidden` and hoist
  those values into registers.
- **Indexer gate GEMV** `glm_index_weight_gemv_f32`: 32 active threads and a
  serial chain of 4096 adds, [E] ≥7 µs per MLA layer (**→ X3: the reference's
  equivalent measures ~73 µs/layer**). Overlap it with a side
  stream or PDL.
- **MoE combine:** fuse `down_weighted_output` with `combine_f32` in slot order.
  This removes the `[rows, topk, hidden]` F32 round trip and one launch.

### 3.7 [B] Post-parity (arithmetic-changing)
- **B1:** shared experts through the packed kernel. This is CB1, the largest
  single bytes item.
- **B2:** FP8-MMA indexer scoring, closer to the reference's DeepGEMM path.
- **B3:** FP8 router.
- **B4:** BF16 mHC `fn` (CB8).
- **B5:** head gather as BF16, or a rank-local argmax (CB10).
- **B6:** upstream's int8-activation `dp4a` GEMV. This is exact int32
  accumulation, a different path from FP16 MMA.

---

## 4. Findings — project as a whole

Time budget used to rank these [R]:
- **GLM-5.3 EP2:** a C1 step is a DFlash/MTP verify of ~5–7 rows, ≈94 ms with
  PDL (ee3ecb82). Routed experts take ~65–70% of the step at 222–235 GB/s
  (54507b8c). KDA recurrent state is ~1%.
- **Qwen3.6-27B with MTP K=4:** projection GEMVs 68%, attention 12.7%, drafter
  9%, lm_head 2.7%, GDN 1.6%. The bit-exact batched GEMV reaches 76.6% of peak at
  M=4 and 50.7% at M=8 (`w4a16_gemv.cu:666-672` ✔). That table is labeled
  "BEFORE this change" in source, so re-measure the current state.

**P1. GLM router + shared expert at ≤8 rows** — high value, L–M effort.
- **Router:** `dense_gemm_bf16_router` launches grid `[ceil(288/64)=5, ceil(m/16)]`
  (`ops/gemm_dense.rs:137` ✔) with a single-buffered K loop. [R?] 90 µs warm at
  M5 (77 µs with BN4) against a ~10 µs roofline. Fix: the same per-output FP32
  k-order, parallelized over N×M (≥144 CTAs) with a 4–6-stage cp.async ring.
- **Router consistency:** at 9–32 rows the router switches to a reassociating
  tensor-core tier (`helpers_c.rs:306-321`), so the same token can route
  differently at C1 and C4. The fix above removes that.
- **Shared expert:** the tensor-core tiers apply only when n>8
  (`forward_prefill_phase.rs:128`). Otherwise it runs on `fp8_gemm_n128` (2×
  bytes) or on `w4a16_gemm_t` with 16–32 CTAs (`shared_m16.rs:98`), [R?]
  ~148/76 µs against a ~62 µs roofline. PairVerify and OwnerVerify always use
  `w4a16_gemm_t`.
- **Overlap:** the shared expert and router→topk→sort→quant are independent. The
  fork/join exists but is gated `num_tokens > 64 && !graph_capture`
  (`forward_prefill.rs:203-208` ✔). Use event fork/join inside graph capture.
- [E] 5–8% of a C1 step combined; the estimate rests on eager-mode numbers, so
  run nsys with graphs and PDL on first.

**P2. PDL weight prefetch + coverage** — medium-high value, low effort.
- `atlas_pdl_enter()` calls `launch_dependents` and then `wait` immediately, so
  no work happens before the wait. Static weights never depend on the
  predecessor: compute tile coordinates, issue `prefetch.global.L2` or the first
  cp.async stages over the weight span, then wait. Never pre-read activations.
- **Coverage gaps:**
  - The Qwen hot path (`w4a16_gemv`, `_sw`, `batch2..8`, `dual`, `qg`, `qkvz`,
    norm/rope/KV) has no PDL.
  - The `*_tc16/32` tiers never call `atlas_pdl_enter`.
  - No EXL3 kernel has PDL.
- **GB10 twist:** DRAM is idle during the CX-7 all-reduce. A side-stream L2
  prefetch of the next router + shared gate/up (~12 MB, under the 24 MB L2)
  makes those reads nearly free.
- **Hygiene:** `atlas_pdl_enter()` is called twice and the header included twice
  in `deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu` (lines 13-14,
  2098-2099, 2827-2828 ✔).

**P3. Bit-exact batched W4A16 GEMV is instruction-bound at M=3..8** —
medium-high value, M effort.
- **Cause:** `w4a16_gemv_batchm_impl` (`w4a16_gemv.cu:462-604`) gives each
  thread one output. Per row and chunk it does two 16 B activation loads, 16
  unpacks and 17 FMAs against one 8 B weight load. [R] batch8 at M=4 is 6.1%
  slower than batch4, from code size alone.
- **Fix:** 2–4 outputs per thread sharing the activation loads and unpacks. The
  per-output FMA chain is unchanged, so results stay bit-identical. [E] ~76% →
  85% of peak at M=4, roughly 5–7% off the Qwen K=4 step.
- **Cross-cutting:** every `common/` kernel builds with `--fmad=false`
  (`kernels/gb10/common/KERNEL.toml:2` ✔). That roughly doubles FP instruction
  count in instruction-bound kernels.
  - FMA contraction stays run-to-run deterministic; what it breaks is
    bit-parity *between* kernels (M=1 vs batched).
  - Experiment: per-kernel opt-in, or explicit `fmaf` in both sibling kernels,
    gated by the bit-parity microtests and the KL gate.
  - This is the only place in Atlas where "more compute is free" is not true
    today.

**P4. GLM sparse MLA decode beyond 2K history uses the slowest kernel by
default.**
- **Default kernel:** `glm_sparse_mla_prefill_bf16` (`glm_indexer.cu:519-617`)
  runs 64 CTAs, one per head, and each re-streams ~2051 selected rows of K then
  V. [E] ~270 MB of L2 traffic per layer for 2.1 MB of unique data, with 2-byte
  loads and a serial softmax.
- **Tensor-core kernel:** `glm_sparse_decode_split.cu` reads K once and reuses
  it for PV, but it is off unless its env flag is `1`
  (`ops/glm_sparse_prefill_tc.rs:35-44` ✔). It also hard-codes 8 splits, and its
  two 32-head groups launch serially (16 of 48 SMs busy).
- **Fix:** one launch with grid `[heads/32, 1, S≤16]` and a double-buffered K
  tile. [E] 1.3–3 ms/token at C1 past 2K history.
- **Caveat:** a comment at `glm_sparse_prefill_tc.rs:306` says ordinary decode
  does not call this entry. Confirm the wiring before an A/B.

**P5. GLM dense MLA (≤2048 history) re-reads the single latent KV 64×.**
`paged_decode_attn_512.cu:43-240` runs one CTA per (head, row) with 32-bit loads
and no split-K. Reuse the causal-fill tensor-core route that `fp8_g128` takes
(`attention_forward_mla.rs:397-410`). Related: the V-pool alias (CB3).

**P6. Qwen GQA verify attention reads each KV head ~24×.**
- Today: one CTA per (q head, row), with split-K pinned to `max_decode_seqs`
  (usually 1 at C1). [R?] Attention is 14.6 ms of a 115 ms K=3 step.
- Fix: a grouped `mma.sync` kernel, one CTA per (KV head, split), holding all
  query heads × verify rows (CB6). `inferspark_prefill_paged_fp8` is a close
  template.
- Also re-test fp8 KV as a *capacity* lever. The 07-25 TPOT rejection predates
  a later fp8 dequant rewrite. The sub-review cited dd2ac46d, but that commit's
  title is about cold-load weight transposition, so the rewrite commit still
  needs identifying.

**P7. KDA recurrent** (CB9): records rollback for the pair path is the main
item. Smaller ~0.1–0.3 ms items:
- hoisting in `kda.cu:287-339`;
- a C1 decode kernel that fills 32 of 48 SMs and reads the state twice;
- 34 commit launches that could be 1;
- fusing pack, conv, recurrent and norm.

Do **not** move live state to BF16/FP16 at C≤4; that drift is already measured
[R].

**P8. Tensor-core W4A16 tiers trail vendored Marlin.**
- [R 3a6de55f] 167–200 vs 184–240 GB/s at M≤8, and 114–140 vs 174–225 GB/s at
  M=16–32.
- Causes (`w4a16_gemv.cu:1695-1909`): grid `ceil(N/16)`, a single register
  prefetch, a LUT rebuild in every CTA, and byte-wide scale loads.
- Fix: adopt Marlin's structure (persistent striped grid, cp.async ring,
  pre-permuted weights; `marlin_pack_nvfp4` exists). [E] 1.06–1.29× on GLM
  dense/shared projections. The tiers are still opt-in (`ATLAS_W4A16_TC=1` ✔).

**P9. EP decode all-reduce is latency-bound** (see §6, pplx-garden):
- Today the chain is: copy-engine copy into the pinned slot → stream write →
  host proxy polling → RDMA data write → wait → flag write → stream wait → a
  separate add kernel. [E] ~8 µs wire for 64 KB.
- Options: WRITE_WITH_IMM completion, or a chained flag write on the same queue
  pair.
- A producer kernel storing straight into the pinned slot would need a
  system-scope release. An earlier SM-copy attempt saw stale NIC reads, possibly
  because that fence was missing.

**P10. Small items.**
- `dense_gemv_bf16_batchm` loops over a runtime `m`, so its accumulators likely
  sit in local memory. KDA b_proj's N=64 gives only 16 CTAs.
- The MTP drafter keeps the first 100K token ids. A frequency-ranked ~32K set
  (like DFlash `d2t`) could save ~2% if acceptance holds.
- `if (n>=N) return;` precedes `__syncthreads` in both batched GEMVs. Harmless
  for today's shapes, but a latent hazard.
- Fix the 64 MB L2 comments in `gated_delta_rule.cu:1007,1292`.

### Evaluated and not recommended (project)
- **Persistent fused MoE at decode:** the intermediate stays in L2 (see §2).
- **Empty-CTA pruning in the dense routed grids:** `down` already runs at
  222–235 GB/s [R].
- **Deeper btile cp.async staging:** occupancy already keeps enough bytes in
  flight.
- **`cudaAccessPolicyWindow` / L2 persistence:** hundreds of MB stream through
  the 24 MB L2 each step. The exception is the targeted prefetch in P2.
- **Split-K at M=1 for 4096-wide outputs:** already 512+ CTAs.
- **Norm fused into the GEMV prologue:** ≤0.5% once PDL is on.
- **lm_head + argmax fusion:** small.
- **Already measured dead [R]:** BF16/FP16 live state, W4A8 dp4a, W4A4 decode
  (ds4x [C] agrees: generic W4A4 at 1–8 rows is ~10× slower than W4A8), K≥4
  drafts with the current drafter, and fp8 KV for TPOT at short context.

---

## 5. What Atlas already implements (baseline for the external comparison)

- **Weights / GEMV:**
  - NVFP4 everywhere, with the smem E2M1 LUT;
  - FP8/MXFP8 GEMV twins;
  - 16 B coalesced loads and smem FP32 activation preload;
  - one-warp-per-output M=1 kernel;
  - bit-exact batched GEMV, M=2..32;
  - tensor-core W4A16/MXFP8/BF16 tiers (opt-in);
  - vendored Marlin NVFP4 (bench);
  - fused dual/triple/qkv/qkvz/q-gate projections;
  - NVFP4/MXFP8/vocab-split LM heads;
  - native block-scaled FP4 MMA (`kind::mxf4nvf4`, W4A4) for verify/prefill.
- **MoE:**
  - batched expert GEMV with device pointer tables;
  - gate+up fusion, and fused SiLU·mul + NVFP4 requant;
  - shared expert as an extra slot;
  - MMA-ready btile layout;
  - compact expert/tile worklists (weights read once per expert across verify
    rows);
  - A-gather via `sorted_token_ids`;
  - shared-expert TP split and expert TP (dc805f71);
  - exact five-row router;
  - fused routed+shared atomic C4 combine.
- **Attention:**
  - paged split-K decode with LSE merge;
  - FP8, NVFP4 and TurboQuant 2/3/4/8-bit KV;
  - tensor-core sparse MLA with cp.async and K=V latent reuse (opt-in);
  - MXFP8/NVFP4 MLA projection twins;
  - WMMA index scorer (`glm_indexer_wmma`, referenced in `init.rs:576`);
  - 8-bit radix top-K;
  - graph-safe dynamic lengths;
  - sinks, h128/h512 prefill.
- **Recurrent:**
  - register-resident GDN/KDA verify;
  - WY2–4;
  - FLA/persistent/register-resident prefill and FlashKDA;
  - opt-in FP16 GDN state;
  - fused conv+L2norm verify;
  - snapshots;
  - records rollback (opt-in).
- **System:**
  - CUDA graphs, including the K=5 TP verify with NCCL in capture;
  - PDL on 27 kernels;
  - device-side routing with no host sync;
  - RDMA pair all-reduce over CX-7 (ATS pinned memory, segment pipelining,
    multi-rail capable);
  - sequence-parallel prefill;
  - DFlash/dflash2 + MTP with adaptive width;
  - Marconi SSM snapshot cache;
  - weights in `cuMemAlloc` device memory.

---

## 6. External repositories — verdicts

Licenses: Atlas is AGPL-3.0-only. MIT, Apache-2.0 and BSD-3 code can be taken in
one way if the notices are kept. A repo with no LICENSE file is ideas only.

| Repo | Exists / license | Verdict |
|---|---|---|
| [perplexityai/pplx-garden — fabric-lib](https://github.com/perplexityai/pplx-garden/blob/main/docs/fabric-lib.md) | ✔ MIT | **PARTIALLY USEFUL BECAUSE** its WRITE_WITH_IMM completion counting and N-rank CPU-proxy design map directly onto `rdma_pair` (drops a round trip per decode all-reduce), but the code needs GPUDirect RDMA/GDRCopy/NVLink, which GB10 lacks, and its all-to-all pattern has no byte advantage at EP=2. |
| [osayamenja/FlashMoE](https://github.com/osayamenja/FlashMoE) | ✔ BSD-3 | **NOT USEFUL BECAUSE** it is BF16-only (3.5–4× the expert bytes of NVFP4 on a 273 GB/s part), has no shared experts, requires NVSHMEM with P2P/GPUDirect, and its win (launch/tail removal) is already targeted by Atlas graphs + PDL + fused C4 combine. |
| [Dots-Infra/UltraEP](https://github.com/Dots-Infra/UltraEP) | ✔ MIT | **NOT USEFUL BECAUSE** it needs SM90/SM100, NVSHMEM and NVLink, and the GB10 version of its problem (per-rank expert-byte skew at EP=2) is already solved by expert TP (dc805f71); runtime replication would cost scarce unified memory. |
| [ondigo-winder/SWITCHLESS-NCCL-RING-4-GB10](https://github.com/ondigo-winder/SWITCHLESS-NCCL-RING-4-GB10) | ✔ Apache-2.0 (+NCCL license for the patch) | **USEFUL BECAUSE** its GB10 measurements show the second CX-7 PCIe half nearly doubles NCCL bandwidth even on stock NCCL with two nodes ([C] 110→188 Gb/s), which Atlas's EP2 scripts leave unused (`NCCL_IB_HCA=rocep1s0f0` in `start-ep2.sh:107`, `start-deepseek-ep2.sh:59`, bench scripts ✔), and it confirms Atlas's all-reduce MoE (not all-to-all) is the design that works on a switchless 4-node ring. |
| [VetleWammer2/warpcourier](https://github.com/VetleWammer2/warpcourier) | ✔ MIT (2 commits) | **NOT USEFUL BECAUSE** it is an early H200 prototype that by its own account loses to DeepEP on every axis, and its only working mode needs NVLink load/store; only its NCCL GIN / device-API research notes are worth reading. |
| [idrees2516/deepgemm-rs](https://github.com/idrees2516/deepgemm-rs) | ✔ no LICENSE file (README says MIT) | **NOT USEFUL BECAUSE** every kernel is tcgen05/TMEM/wgmma with zero `mma.sync`, it misclassifies sm_120a as SM100, it was never run on hardware (GPU-less sandbox, compile checks only), and the license grant is not clean. |
| [xinyang-zhou/sm120_nvfp4_ops](https://github.com/xinyang-zhou/sm120_nvfp4_ops) | ✔ MIT (CUTLASS dep, BSD-3) | **PARTIALLY USEFUL BECAUSE** its NVFP4 sparse-MLA KV format and kernel (QK and PV both on FP4 MMA, ~384 vs 1024 B/token) are a real 2.7× KV-bytes saving for high-concurrency GLM/DSV4 (CB2, quality-gated), while its GEMM/grouped/MoE parts do not beat Atlas at decode (TileM=128 pads 7/8 at M≤16). |
| [shuochen0113/ufg-ggemm](https://github.com/shuochen0113/ufg-ggemm) | ✔ Apache-2.0 | **PARTIALLY USEFUL BECAUSE** the fused down-GEMM probability-weighted scatter-add and the device-built per-expert tile worklist map onto real Atlas prefill costs (CB4, `[T·topk,H]` round trip), but its kernels are Hopper-only (wgmma/TMA multicast, forced 9.0a) and training-oriented. |
| [Logos-Flux/optimized-CUDA-GB10](https://github.com/Logos-Flux/optimized-CUDA-GB10) | ✔ Apache-2.0 | **NOT USEFUL BECAUSE** it contains only an unfused RMSNorm and GELU, which Atlas's `rms_norm.cu` (16 B loads, fused residual variants, PDL) already exceeds; only its NGC packaging notes are mildly informative. |
| [BEEB0SS/kernel-ladder](https://github.com/BEEB0SS/kernel-ladder) | ✔ MIT | **PARTIALLY USEFUL BECAUSE** its measured GB10 calibration (24 MB L2, 99 KB smem/block, FP8 = BF16 tensor rate, FP4 2×, no `dram__` ncu counters → use `lts__*`) is directly useful for Atlas's roofline and profiling notes; its kernels do not beat Atlas. |
| [r0b0tlab/glm53-flash-exl3-dflash2-sm121](https://github.com/r0b0tlab/glm53-flash-exl3-dflash2-sm121) | ✔ MIT (vendors exllamav3, MIT) | **PARTIALLY USEFUL BECAUSE** it is the same model/format/arch and lays out exactly which upstream exllamav3 kernel techniques Atlas's correctness-first EXL3 kernels still lack (§3.1–3.4), plus a GB10 finding of ~2 ms per device-to-host copy completion with GPU work in flight [C] and a DFlash2 K sweep by workload class; its kernels are upstream's, not new. |
| [neron82/helios-glm53flash](https://github.com/neron82/helios-glm53flash) | ✔ MIT | **PARTIALLY USEFUL BECAUSE** its diagnostic results transfer (ungrouped MoE makes a 4-row verify cost 6.8× one row — a prerequisite for EXL3 speculation; tensor-core indexer 10.5× scalar — reason to validate `glm_indexer_wmma`; MLA-latent precision study), but it targets 2× RTX 3090 with host expert streaming, which does not apply to GB10 unified memory, and its kernels are weaker than Atlas's. |
| [wangshang23/ds4x](https://github.com/wangshang23/ds4x) | ✔ MIT | **PARTIALLY USEFUL BECAUSE** it is the most bandwidth-relevant GB10-native repo: a packed 583 B MLA KV row (E4M3 + FP16 RoPE + UE8M0 scales) and MXFP4 indexer keys at 68 B/row scored directly with `mxf4nvf4` MMA (CB5), plus a working DSV4 CSA/HCA reference that Atlas DSV4 still lacks; it does not apply to GLM EXL3 weights. |
| [HeNryous/mimo-spark-optimized](https://github.com/HeNryous/mimo-spark-optimized) | ✔ MIT | **PARTIALLY USEFUL BECAUSE** it supplies one cheap untested lever (`NCCL_CROSS_NIC=1` across both RoCE NICs, [C] +42% single-stream TP=2 on vLLM; not set anywhere in the Atlas tree ✔) and a useful negative (NVFP4 KV on GB10 buys capacity, not speed); most of the rest Atlas already has. |
| [mneha05/gb10-attn](https://github.com/mneha05/gb10-attn) | ✔ no license | **PARTIALLY USEFUL BECAUSE** its measurement method transfers: rotate block tables over ≥4× L2 or L2 hits inflate "bandwidth" to 168% of peak, and split count by chain length (~ctx/32, ≤64) at B=1; its fp16 kernels are a subset of Atlas's and cannot be copied (no license). |
| [xiez-njust-xmu/FlashAttentionV2-PLUS](https://github.com/xiez-njust-xmu/FlashAttentionV2-PLUS) | ✔ no license | **NOT USEFUL BECAUSE** it is a non-paged head_dim-64 BF16 FA2 that is 1.5× slower than cuDNN by its own numbers; Atlas prefill is already paged, multi-head-dim, quantized-KV and cp.async/ldmatrix/mma.sync. Only takeaway: TMA + mbarrier producer/consumer warps do work on sm_121. |
| [solidsf-inc](https://github.com/solidsf-inc) (17 repos) | ✔ mixed MIT/Apache-2.0 | **PARTIALLY USEFUL BECAUSE** one sub-repo, **rvLLM** (Apache-2.0), contains a GQA-grouped split-KV FP8 decode kernel (`v3/kernels/fp8_decode_v2.cu`) that Atlas lacks (CB6/P6), and **dgx-spark-dual-qsfp** documents NIC/PCIe-path selection for dual-Spark RoCE ([C] +7.5%); the remaining sub-repos are **NOT USEFUL** — sfActivation/sfVector/sfMAC/sfGEMM/sfCORDIC are FP32 CUDA-core kernels that lose to or barely match trivial baselines, sfFFT targets Hyena/S4 long convolution, and sfConv/sfEigen/sfFFTGeneral/sfKrylov/sfLU/sfMod/sfSpMV/sfSVD/sts2-rs are scientific/unrelated (judged by description). |

### 6.1 Details and follow-up experiments per repo

**pplx-garden / fabric-lib.**
- **What it is:** a Rust RDMA TransferEngine (ibverbs/libfabric) plus
  p2p-all-to-all MoE kernels driven by a CPU worker (MLSys'26).
- **Techniques:** WRITE_WITH_IMM per-transfer immediate counting
  (`imm_count.rs`); GDRCopy flags; dmabuf GPU memory registration; FP8 dispatch
  with external scales.
- **Claims:** [C] EP8 decode dispatch ~42–50 µs on CX-7.
- **Byte math for Atlas:** at EP=2, top-8 of 288 experts, 99.6% of tokens need
  the peer. FP8 dispatch plus BF16 combine is ~3H bytes per token against 2H for
  Atlas's all-reduce, so do **not** adopt all-to-all at EP=2.
- **Experiments:**
  1. Read `rdma_pair`'s stats line (ready-wait / data / flag µs) to size the
     flag round trip.
  2. Add `post_write_imm` to atlas-rdma and turn the completion into the local
     `arrived` store.
  3. Cheaper variant: post the flag WRITE directly behind the data WRITE on the
     same QP. Usually ordered on mlx5, but not guaranteed by spec, so soak-test
     for divergence.
  4. Use its `topo.rs`/`domain_group.rs` as the reference for an N-rank
     `rdma_pair`.

**FlashMoE.**
- **What it is:** one persistent kernel fusing gate, dispatch, FFN and combine,
  with an in-kernel scheduler and NVSHMEM (NeurIPS'25).
- **Claims:** [C] up to 5.1× on A100/H100 at ~1024 tokens/rank.
- **Ideas only:**
  1. Extend 4923c9ef's segment pipeline into the MoE: post RDMA segment k as
     soon as down row-block k finishes.
  2. nsys idle gaps in the decode MoE tail with PDL on. If they are under 5% of
     the step, close the persistent-kernel idea.

**UltraEP.**
- **What it is:** per-microbatch expert placement and replication on NVLink
  domains.
- **Claims:** [C] imbalance 1.5–3.0 → 1.01 at EP64.
- **Experiment:** for EP=2 models without expert TP (MiniMax, DSV4, Step-3.7),
  log unique experts per rank. Try a static, calibration-derived expert→rank
  permutation, with no runtime replication.

**SWITCHLESS-NCCL-RING-4-GB10.**
- **What it is:** an NCCL 2.30.7 `net_ib/connect.cc` patch for 4 HCAs on a
  switchless 4×GB10 ring. Settings: `NCCL_SWITCHLESS_RING_ONLY=1`,
  `NCCL_IB_SUBNET_AWARE_ROUTING=1`, `NCCL_IB_MERGE_NICS=0`, GID 3, 8 channels,
  Ring, a /24 per cable half, MTU 9000.
- **Claims:**
  - [C] 1 MiB messages get *worse* with 4 HCAs (56→37 Gb/s).
  - [C] vLLM TP4: prefill +10–34%, decode within noise.
  - A ring cannot do all-to-all.
- **Experiments:**
  1. EP2 with both halves in `NCCL_IB_HCA` and `ATLAS_RDMA_RAILS`. Measure 16K
     TTFT (baseline 7.64 s [R]) and the C1/C4 step under measurement
     discipline. Expect a prefill gain and flat decode. Confirm the second half
     first with `ibv_devinfo` and `ib_write_bw`.
  2. Retune `SPLIT_MIN` and the segment count for 2 rails.
  3. Open question for the team: does the cluster behind `launch-atlas-ep4.sh`
     have a switch? If it is a ring, this patch is required. Also, `rdma_pair`
     is 2-rank only, so 4 ranks fall back to NCCL at ~50 µs per small
     all-reduce (**→ X2**: measured 46–51 µs median at 48 KB TP2; the "~100 µs"
     in `rdma_pair.rs:4-7` is stale).

**warpcourier.**
- **What it is:** a GPU-resident MoE dispatch layer on the NCCL 2.28 device API
  (LSA / GIN / host modes), tested on H200 only.
- **Follow-up:** check whether NCCL GIN has a CPU-proxy backend that works on
  GB10. If it does, it could replace the hand-rolled proxy for more than two
  ranks.

**deepgemm-rs.**
- **What it is:** a Rust/NVRTC port of DeepGEMM. All 17 commits landed on
  2026-10-10.
- **Follow-up:** none, beyond optionally writing Atlas's own bit-exact CPU
  E2M1/E4M3/UE8M0 reference using its `golden.rs` as a pattern. Do not copy it.

**sm120_nvfp4_ops.**
- **Techniques:**
  - 128×128×128 persistent TMA, warp-specialized GEMM (128 producer + 256 MMA
    threads);
  - TMA-store epilogue;
  - fixed split-K for M=128/256;
  - sparse-MLA kernel with 8 compute + 4 IO warps, 2-stage cp.async, 98 KiB
    smem, and V requantized in-CTA along the candidate axis so that PV also runs
    on FP4.
- **Claims:** [C] on RTX 5090, 1.53–1.69× vs FlashInfer for sparse decode at 32K
  context, B=64–1024.
- **Why its GEMM cannot win at decode:** its M=16 GEMM moves ~18.8 MB in
  25.2 µs on a 1.8 TB/s part, while the GB10 floor for the same bytes is ~81 µs
  and Atlas Marlin is already near that floor.
- **Experiments:**
  1. CB2 quality gate.
  2. Port the decode kernel for TP2 (32 heads).
  3. CB11.
  4. A/B `w4a4_gemm_mfast` against a persistent TMA warp-specialized split-K
     version for M=64–256. This is a prefill compute gain.

**ufg-ggemm.**
- **Experiments:**
  1. CB4 measurement.
  2. Count launched vs useful CTAs in `common/moe_w4a16_grouped_gemm.cu`
     (N_tiles × max_m_tiles × experts). If the ratio is over 10×, move to a 1-D
     worklist like `marlin_align`.
  3. Try an L2 raster swizzle across experts.

**optimized-CUDA-GB10.**
- **Note:** its "92 TFLOPS BF16 measured" contradicts kernel-ladder's 212.9.
  Don't cite it.

**kernel-ladder.**
- **Experiments:**
  1. Re-probe DRAM bandwidth with more loads in flight (≥8 outstanding 16 B
     `ld.global.nc.L1::no_allocate` per thread). Its 231 GB/s is likely an
     under-measurement, since Atlas Marlin hits 240. Do this before putting any
     measured figure into `HARDWARE.toml` or GB/s-efficiency reports.
  2. A/B register-chained P (QK accumulators fed straight into the PV operand)
     against smem P for prefill paths with head_dim ≤128.
  3. Add the `lts__*` / `sm__inst_executed_pipe_tensor` counter substitutes to
     the profiling docs.

**r0b0tlab glm53-flash-exl3-dflash2-sm121.**
- **What it is:** vLLM v0.30.0rc1, an EXL3 plugin and vendored exllamav3 v1.5.0,
  running on one GB10 at TP=1 with a 2.25 bpw pack (98.5 GB).
- **Claims:**
  - [C] AR 7.67 → 17.95 tok/s through fused MoE wiring.
  - [C] DFlash2 K=5: prose ~24, structured ~50 tok/s. On prose, K=7's extra
    positions are accepted only 10%/6%/1% of the time.
- **Byte ceiling** [E]: 3-bit experts ≈3.1 GB/token, so ≤88 tok/s from the MoE
  alone. Atlas's 4-bit experts ≈4.2 GB/token, so ≤65 tok/s per GB10 before TP.
- **Experiments:**
  1. Microbenchmark device-to-host copy completion with a decode graph in
     flight, then audit the GLM decode/verify loop for per-step host copies.
  2. After parity: dp4a decode → 8-weight fragment extraction → half-precision
     intermediates. Gate each step on byte identity.
  3. Expert-grouped EXL3 MoE.
  4. TP=1 on a single Spark with a 2.25 bpw pack.
  5. K sweep by workload class once EXL3 speculation lands.
  6. Off-by-one check: vLLM captures hidden states at `idx+1` ("output of layer
     idx"); Atlas `impl_b3_dflash.rs` uses `target_layer_ids` directly. Confirm
     the two agree.
- A sibling repo, `r0b0tlab/glm53-flash-exl3-exllamav3-gb10` (MIT), claims [C] a
  token-exact needle test at ~260K context on one Spark.

**helios-glm53flash.**
- **What it is:** a C++23/CUDA engine on 2× RTX 3090 with experts streamed from
  host memory.
- **Claims:** [C] 16–18 tok/s; block-int8 MLA latent relative error 1.96e-4, fp8
  ~10× worse, 4-bit ~60× worse.
- **Experiments:**
  1. Expert-grouped EXL3 MoE; verify multiplier at K=1..5 with
     `ATLAS_MOE_UNION_STATS=1`.
  2. Turn on `glm_indexer_wmma` behind a flag and check top-K set equality
     against the scalar scorer.
  3. Repeat the KV-precision study on GB10 (bf16 / int8-block / fp8 / nvfp4):
     error plus µs/layer at 32K and 128K.
- Its "sparse MLA decode is instruction-bound" finding is from a ~3.4× higher
  bandwidth part. Re-measure on GB10 rather than assume it holds.

**ds4x.**
- **What it is:** a fork of antirez/ds4 with llama.cpp MMQ, built only for
  sm_121a, running DeepSeek-V4-Flash Q2 on one GB10.
- **Claims:** [C] 15.8 tok/s at 4K, 10.85 at 1M (synthetic prefilled caches).
- **Experiments:**
  1. FP8 indexer keys (CB5).
  2. MXFP4 indexer keys, measuring top-2048 overlap.
  3. A DSV4 FP8 MLA row with an FP16 RoPE lane and UE8M0 scales. First verify
     Atlas's current `mla_cache_assemble_fp8.cu` layout; it was only partially
     read.
  4. Use ds4x as the reference for the pending DSV4 CSA/HCA attention.

**mimo-spark-optimized.**
- **What it is:** a vLLM build for MiMo-V2.5-NVFP4 on 2× GB10 at TP=2. Its README
  says it was written by Claude.
- **Claims:**
  - [C] NVFP4 KV at 180 vs 320 B/token, about 2× capacity.
  - [C] Its tensor-core NVFP4 decode is still ~1.9× slower than a bf16 cache.
  - [C] `NCCL_CROSS_NIC=1`: +42% single-stream, +83% at 8 streams.
- **Experiments:**
  1. List which collectives still go through NCCL rather than `rdma_pair`, then
     A/B `NCCL_CROSS_NIC=1` with both HCAs listed.
  2. Split a single-stream step into collective time vs weight-streaming time.

**gb10-attn.**
- **Claims:** [C] 82–85% of 273 GB/s at B=64; ~27% at B=1.
- **Follow-up:** audit any `unsigned int` pool offsets for the >2^31-element
  overflow it hit. Atlas's 64-bit page strides are likely fine.

**FlashAttentionV2-PLUS.** No follow-up.

**solidsf-inc / rvLLM.**
- **`fp8_decode_v2.cu`:** grid = (kv_head, chunk, batch), with chunks =
  clamp(128/(Hkv·B), 1, 32). With one chunk it writes output directly and skips
  the combine. Lane groups run independent online softmax with no
  `__syncthreads` in the token loop. It optionally uses per-token, per-KV-head
  FP8 scales, against Atlas's per-tensor scale; treat that as an accuracy
  experiment.
- **Portability:** it says sm_90a, but the code looks portable to sm_121. Not
  verified.
- **Side trick:** `lmhead_prune_argmax.cu` skips a monotonic softcap before a
  greedy argmax. Free, but low priority.

---

## 7. Correctness red flags noticed in passing

- **Silent return:** `exl3_mhc_pre_norm` returns on invalid `rounded_splits`
  without writing a sentinel (`exl3_mhc.cu:91` ✔), so stale outputs can flow
  downstream. Write NaN or trap.
- **Scale derivation:** `glm_index_quant_row` computes its scale as
  `exp2f(ceilf(log2f(max/448)))`. This is sensitive to `log2f` ulp error at
  exact powers of two; use `frexpf`, or confirm the oracle covers such inputs.
- **Dead branches:** `accumulate_logical` keeps dead tile_k 48/64 branches that
  inflate `acc`.
- **Barrier hazard:** in the batched GEMVs, `if (n>=N) return;` comes before
  `__syncthreads` (latent).
- **Routing consistency:** the router switches from exact to a reassociating
  tier at 9–32 rows, so the same token can route differently at C1 and C4 (P1).
- **Layer indexing:** DFlash target-layer indexing vs vLLM's `idx+1` convention.
  **→ X9: likely consistent** (both capture layer outputs); confirm against the
  drafter config.

## 8. Suggested test order

1. **One nsys trace each:**
   - GLM NVFP4 C1 verify step (graphs + PDL on);
   - GLM EXL3 decode step;
   - Qwen3.6-27B K=4 step.

   This confirms or kills §3.1, §3.2, §3.5, P1 and P6. Everything ranked above
   rests on [E] or eager-mode [R] numbers.
2. **Cheap levers** (config / S effort):
   - both CX-7 halves + `NCCL_CROSS_NIC` A/B;
   - device-to-host copy-completion microbenchmark;
   - EXL3 top-k warp argmax;
   - mHC `pre_norm` parallelization.
3. **Bit-identical EXL3 rework:** §3.1 + §3.3, then §3.4, then the §3.5 fusions
   + PDL. Gate on the existing packed byte-gate controls.
4. **GLM NVFP4 P1:** router/shared ≤8 rows + overlap in graph capture; P2 PDL
   prefetch.
5. **Bytes items with an accuracy gate:** CB5 FP8 indexer keys (lossless first),
   CB3 V-pool alias, CB9 records rollback for the pair path.
6. **Attention:** L2-defeated bandwidth harness (CB6 step 1), then the
   GQA-grouped decode prototype.
7. **Post-parity / lossy:** CB1 shared experts packed, CB2 NVFP4 sparse-MLA KV,
   CB7 FP8 prefill all-reduce, CB4 fused scatter-add.

Every result should carry a full run fingerprint (commit, serve flags,
checkpoint revision, harness/prompt class, thinking mode, box) per
`.claude/skills/measurement-discipline`, with n≥3 for anything that headlines.
