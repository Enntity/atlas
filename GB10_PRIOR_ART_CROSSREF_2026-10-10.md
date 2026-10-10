# GB10 Prior-Art Cross-Reference — 2026-10-10

This is a companion to `GB10_KERNEL_OPTIMIZATION_REVIEW_2026-10-10.md` (called
"the review" below). It checks the review's findings against seven local
prior-art projects. For each one it records where they agree with the review,
where they contradict it, and which new ideas are worth validating in the Atlas
engine.

Everything was read-only. Nothing was built, run or edited in any of the seven
projects. Detailed notes, with per-number paths and fingerprints, are in
`scratchpad/gb10-kernel-review-2026-10-10/prior_art_*.md` (gitignored).

## Evidence labels

The review's labels still apply. This document adds two:

| Label | Meaning |
|---|---|
| **[M-x]** | Measured in prior-art project *x*. The path and fingerprint are in the notes file. These are **not Atlas A/Bs**: the engine (usually vLLM + exllamav3), box, image and date differ. Many results come from a **single boot**. |
| **[H]** | A hypothesis offered by the prior art. The data does not establish it. |

Projects, with the short code used in labels:
- **MR** — `~/Workspace/mindreader`
- **SG** — `~/Workspace/spark-glm53`
- **SGo** — `~/PycharmProjects/spark_glm53`
- **K2** — `~/Workspace/k2-horizon-stuff`
- **VML** — `~/Workspace/vllm-memory-layout`
- **MC** — `~/dgx-spark-memcheck`
- **LD** — `~/PycharmProjects/local_drafter`

Some projects share sources, so they are not independent evidence:
- The `kcompactd` finding in SG and MR comes from the same MR reports (73, 75).
- The bad-LPDDR5X boxes in SGo (`zgx-0ff6`) and MC (203.0.113.191) are **retired** and no longer used (confirmed by the team, 2026-10-10).

---

## 1. Verdicts per project

| Project | Verdict |
|---|---|
| **mindreader** (MR) | **PARTIALLY USEFUL BECAUSE** it has independently reviewed GB10 measurements of **the exact EXL3 checkpoint and revision Atlas validates against** (MikeRoz 4.05bpw-h6, rev `dda99c7e`, TP=2 on 2×GB10, DFlash2). These give measured values for several of the review's [E] items, and two contradict it. It also found two host-level effects Atlas has no trace of: proxy-thread CPU placement and `kcompactd` compaction. It contributes no new kernels; its kernel work is a screen of stock exllamav3/CUTLASS-sm80/Marlin inside vLLM. |
| **spark-glm53** (SG) | **USEFUL BECAUSE** it is the other project with in-format GB10 kernel evidence on the same checkpoint (vLLM 487ecf187, TP=2, DFlash2 k=7). This includes a measured spill fix (+8–17% per layer), a cooperative split-K MoE (+7.3% C1), and a grouped "E3" prefill MoE (77–91 → 31 ms per layer at T=7168). Each of these directly supports a review item (§3.1, §3.3, §3.4, CB4). The relevant code is MIT or AGPL-3.0, so it is portable with notices. |
| **spark_glm53** (SGo, PycharmProjects) | **PARTIALLY USEFUL BECAUSE** it is an older (~2026-09-20), un-versioned snapshot of SG's Ansible tree. The only things it adds are the first-boot KV-pool number, a bad-LPDDR5X hardware log (`logs/spark2-hardware`: 239 memtester failures, and shard sha256 values that changed on re-read), and rendered per-node configs. Its `results/armab/COMPARISON.md` is an **empty template**: it holds no A/B data, and the HCA recorder failed (`nccl_hca_count=0`). |
| **k2-horizon-stuff** (K2) | **NOT USEFUL BECAUSE** it is a planning-only LoRA/RL up-training plan for K2-Horizon models, and by its own account (A-001) nothing in it was built or run. Its GB10 facts come from NVIDIA docs and from SG. |
| **vllm-memory-layout** (VML) | **PARTIALLY USEFUL BECAUSE** it has clean, independently re-checked measurements of GB10 plain `cudaMalloc` read bandwidth and device properties, which upgrade the review's hardware table. It also measured a **~35% read penalty for host-registered memory**, which bears on Atlas's managed-memory OOM fallback and on the `rdma_pair` receive slot. Its page-placement/colouring premise found no effect on GB10 and is closed. |
| **dgx-spark-memcheck** (MC) | **PARTIALLY USEFUL BECAUSE** it has no performance data, but it measured a silent-corruption failure mode that Atlas has no defence against. On a box with a faulty LPDDR5X device, about 1000 bits/GiB flipped while RAS/EDAC reported "No Memory errors". It also shows that boxes with reduced capacity exist (103.6 and 84.8 GiB visible to the GPU). Its GRUB/crashkernel/KASLR material is box administration. |
| **local_drafter** (LD) | **PARTIALLY USEFUL BECAUSE** its cost-model rules for speculative width are sound, and auditing Atlas against them surfaced a stale verify-cost table in Atlas (N1 below). Atlas already implements every rule with measured inputs. LD itself targets a different system (a laptop drafter feeding remote vLLM), and none of its numbers are measured: the bench artifacts say "DECLARED, not measured". |

---

## 2. Corrections to the review (contradictions)

These change numbers or reasoning in the review. The review has been annotated
in place to point here.

| # | Review item | What the review said | What the prior art shows | Effect |
|---|---|---|---|---|
| X1 | **CB1 / B1** (EXL3 shared experts packed instead of BF16) | [E] ~0.8 GB/token, ~2.9 ms/token; "the reference also uses BF16" read as an accuracy choice | [M-MR] Upstream packed vs BF16 at M=8: shared gate_up 77.7→54.1 µs and down 38.3→26.1 µs, about **1.5 ms/step over 42 layers**. Small 3–6 MB packed rows reach only 113–120 GB/s. [M-SG] MikeRoz shared experts are **K5–6**, not 4–5, so the byte saving is ~0.7 GB/token. The reference keeps them BF16 because of an **implementation hazard**: exllamav3's GEMM and `exl3_moe` share one lock/barrier buffer, so packed shared experts on vLLM's aux stream would race (MR plan 61 D10). It is not an accuracy decision. | Expected win is about half the estimate. It is still worth doing. A grep of Atlas's three EXL3 MMA files found no cross-CTA barrier or atomic, so Atlas probably doesn't inherit the hazard; confirm by reading them. vLLM hides much of the shared-expert time behind routing (only ≈0.3–3 of 5.6 ms is exposed), while Atlas's overlap is off under graph capture (P1). So CB1 + P1's overlap together may be worth more in Atlas than in vLLM. Measure at C1 with graphs on. |
| X2 | **P9 / §6 SWITCHLESS** (NCCL small all-reduce cost) | "~100 µs" for a decode-size NCCL all-reduce, quoted from `crates/spark-comm/src/nccl_backend/rdma_pair.rs:4-7` ✔ | [M-MR] 48 KB TP2 all-reduce on these Sparks: **46–51 µs median**, 34–56 µs minimum (MR reports 61, 62, 67) | Raw RDMA's decode advantage over NCCL is about 6×, not 12×. The 4-rank fallback cost in the review should be read as ~50 µs. That rdma_pair comment is stale and is worth correcting when that file is next touched. |
| X3 | **§3.6** (indexer gate GEMV `glm_index_weight_gemv_f32`) | [E] ≥7 µs per MLA layer | [M-MR] vLLM's equivalent fp32 `weights_proj` GEMV: **0.80 ms/step**, 7 GB/s on 2 CTAs, about **73 µs per MLA layer** | Likely about 10× the review's estimate, so this item moves up. Widen the grid (it currently has 32 active threads) before overlapping it. |
| X4 | **GB10 facts / "Achieved DRAM"** (the efficiency denominator) | 273 GB/s spec; kernel-ladder's 231 [C]; "80–88% of spec" | [M-VML] naive read kernel: **234–241 GB/s** on `cudaMalloc` at 64/128/256 MiB, flat from 2.7× to 10.7× L2, on two nodes. [M-MR] read-sum 240/241 GB/s; d2d copy 245/248 GB/s. L2 = **25,165,824 B** and **48 SMs**, read as device attributes on 2–3 boxes. | The **achievable ceiling is ~240 GB/s (~88% of spec)**. Atlas's 222–240 GB/s streaming kernels are at **92–100% of achievable**, not 80–88% of spec. This strengthens the review's corollary that the remaining decode wins are fewer bytes, fuller SMs and less launch overhead. Drop the kernel-ladder re-probe follow-up. Use ~240 GB/s as the denominator, and record it in `HARDWARE.toml` alongside the 273 spec. |
| X5 | **§3.4** (EXL3 MoE grouping by expert) | "~3.7× redundant" at a 128-row prefill chunk; helios [C] 6.8× verify cost | **Decode/verify:** [C-MR] ~34 distinct experts at 6 rows × top-8 (48 slots), so ≥**1.4×** extra bytes for a per-slot kernel. [C-SG] Expert reuse is ~1.5× across 8 verify rows. Helios's 6.8× does not transfer to GB10 decode. **Prefill:** [E-SG] at T=7168 Atlas's `Grid(n/128, rows*topk)` (`exl3_moe_mma.cu:9-10` ✔) re-decodes each expert about **200×**. [M-SG] Upstream E3 grouped prefill: 77–91 → **31 ms per layer**, cold prefill +37–45% (~1.5–1.6K tok/s up to 256K). | Grouping is a **moderate decode win (~1.4–1.5× MoE bytes)** and a **very large prefill win**. Rank the prefill side higher than the review did. |
| X6 | **§3.2** (mHC pre_norm scale) | [E] 2.7–20 ms/token for Atlas's single-CTA `exl3_mhc_pre_norm` | [M-MR] vLLM's tilelang mHC totals **≈1.9 ms/step** at 6 rows | This does not contradict the review's description of Atlas's kernel, but it sets the **target bar at ≤2 ms/token**. If nsys shows Atlas well above that, §3.2 is confirmed as a top item. |
| X7 | **§3.5** (launch count) | [E] ~1600 launches/token, 1–3 ms/token | [M-MR] Under full CUDA graphs the vLLM GPU is idle only ~1 ms per ~120 ms step | This shrinks the expected launch-gap win under graphs. Fusing the Hadamard transforms still pays through fewer F32 round trips and larger grids, not through idle-time recovery. |
| X8 | **CB7 / #6** (second CX-7 rail, FP8 all-reduce) | Headline-ranked (#6) | **Prefill:** [M-SG] all-reduce was 71 ms per 1024-token chunk on one port (bf16 RING_LL, not overlapped), 7.2% of vLLM's step. [E-SG] That is ~14% of a faster Atlas step, so a second rail recovers ≤3.6–7%, minus what `rdma_pair` already overlaps. **Decode:** [M-SG] fabric utilisation was 0.15%, so decode is latency-bound. | Still cheap and worth doing, but **don't headline it**. The correct same-cable pair is `rocep1s0f0,roceP2p1s0f0`. Atlas's 397B QUICKSTART already uses that pair, while the GLM/DSV4/MiniMax EP2 scripts use one HCA.  |
| X9 | **§7** (DFlash layer-index off-by-one) | Open question | Atlas captures the **output** of each layer listed in the drafter's `target_layer_ids` (capture is "called inside the prefill layer loop after each layer", `impl_b3_dflash.rs:96-99` ✔). vLLM's convention is the same: aux states are outputs of target layers [5,14,24,33,42], listed as hidden-state input indices (6,15,25,34,43). | **Likely consistent.** Confirm against the GLM drafter's `dflash_config.target_layer_ids` and the post-`hc_contract` capture point. Downgrade from red flag to check. |
| X10 | **Measurement-discipline gates** (byte-identity) | Bit-identical [A] track gated on byte identity | [R ee3ecb82] The GLM baseline "is not bit-stable even at C1" (PDL matched 51/52 items). [M-MR] The vLLM reference is nondeterministic within a boot: 0/4 greedy prompts identical, A–A agreement 0.975. | Byte-identity between *runs* is the wrong end-to-end gate on GLM. Keep the **kernel-level** byte gates (packed byte-gate controls), and use probe or distribution gates with an A–A floor end to end. The cause of C1 nondeterminism in Atlas is **not identified** (see N9). |

---

## 3. Corroborations that upgrade evidence

These are review items that now have measurements in the same format on GB10,
from vLLM/exllamav3 rather than Atlas.

| Review item | Prior-art evidence | Upgrade |
|---|---|---|
| **§3.3** (EXL3 decode cost; local-memory spill) | [M-SG] The upstream `exl3_moe` sibling **spills on sm_121** (88 B stack frame, 39 local loads). The spill-free FAST variant was +8–17% per layer; serving +7.7/+8.9/+14.6% (structured/code/json) against A/A drift of −3.5…+4.6% (`SG CHANGELOG`, `docs/sm121-perf-paths.md`). | The review's "likely local-memory spill" becomes a measured precedent in the same kernel family → **N1**. |
| **§3.1** (split the serial K chain across CTAs) | [M-SG] Cooperative two-stage MoE (exllamav3 58d4d732, MIT): K split over 16 warps with 128–256 blocks at M=1, structured C1 72.03 → 77.29 tok/s (+7.3%), matched A/B/A, 2026-09-16. It accumulates in FP16 (folded every 4 K-steps), so it is **not** bit-exact. Negative result: an 8192-block narrow-tile grid lost to 2048 blocks. [M-MR] Upstream packed EXL3 streams **113–221 GB/s** at M=8 on this checkpoint (KDA in_proj 189 µs, o_proj 66.5 µs, MLA o_proj 122.5 µs). | The format **can** reach close to the GB10 roofline, so Atlas's gap is kernel structure. Port the parallel structure but keep FP32 accumulation for the bit-identical track. Don't over-split. |
| **§3 byte-floor table** | [M-MR] Safetensors headers: routed experts are exactly 4.0 bpw (6.29 MB/expert/rank → 2.11 GB/token); attention, shared and head 6 bpw; dense 5 bpw; **42 served MoE layers**. | Table confirmed. The only change is shared experts at K5–6 (X1). |
| **Compute-for-bytes thesis** | [M-SG] Running BF16 KDA and dense projections as FP8 weight-only saved ~11 ms/step; byte arithmetic predicts ~10.6 ms (KL 0.002–0.013, provisional). [M-MR] Dense BF16 GEMMs on large rows reach 216–243 GB/s; kernel swaps are worth ≤1.9 ms. | Strong same-format evidence that **on GB10, step time tracks bytes**. |
| **§4 "W4A8 / activation quant measured dead"** | [M-SG] W8A8 on KDA collapsed a top-token probability 0.778 → 0.0051, with KL 7.99× stock-vs-stock. | Confirmed in a second engine. |
| **CB5** (indexer keys as FP8) | [M-SG] vLLM stores indexer keys as FP8 + scale (132 B/row) against Atlas EXL3's BF16 (256 B). [C-SG] Indexer decode cost is only ~1 ms/step at 100K. | The format is validated by the reference. Below ~100K context it is mainly a **capacity** item, not a speed one. |
| **§5 "weights in `cuMemAlloc`"** | [M-VML] Host-backed pages (4 KiB memfd, mlocked, `cudaHostRegisterMapped`, read through ATS) read at **153.7–157.9 GB/s, ~35% below `cudaMalloc`**. The cause is not isolated [H: TLB/translation; THP untested]. | Atlas's choice is supported by measurement, and its managed fallback gains a cost → **N4**. |
| **§4 "L2 persistence not recommended"** | [M-VML] Read bandwidth is flat from 64 to 256 MiB. | Weakly supported. |
| **CB3** (MLA cache layout) | [M-MR] vLLM wastes 19.5% of MLA bytes on a zero rope slot; Atlas uses `mla_cache_dim = kv_lora + mla_rope` (no padding). | Atlas is already better here. The K/V double-store (CB3) remains a separate question. |
| **Capacity accounting** | [M-VML] CUDA's free-memory report is unreliable on unified memory. | Atlas already uses max(`cuMemGetInfo`, MemAvailable) (`spark-runtime/src/cuda_backend.rs:280-305`) and sizes KV from its own footprint. |
| **r0b0tlab K-sweep-by-class (§6)** | [M-MR] DFlash2 acceptance varies 2.0× across prompt categories (S 1.70–3.38). [M-SG] k=7 per-position acceptance: structured .98→.83; prose .75/.58/.41/.28/.16/.09/.06. [R e74006b1] Atlas adaptive width vs fixed γ=8: prose 1-stream 22.7→26.6 tok/s, 4-stream 38.6→44.6; code flat. | Confirms that per-class width choice matters and that Atlas already does it (`dflash_width::best()`). The remaining issue is the stale cost table (N3). |
| **Indexer radix top-k determinism** | MR flagged nondeterminism on another platform. Atlas's top-k uses `atomicAdd` only for histogram counts, and output is scan-placed. | Not applicable to Atlas. |

---

## 4. New items worth validating (not in the review)

Each item was checked against the Atlas tree before being called new (✔ = I
re-checked it myself). The items are ordered by value per effort.

| # | Item | Source | Atlas status | Experiment (shape · metric · pass criterion) | Value / effort |
|---|---|---|---|---|---|
| **N1** | **EXL3 spill audit** | SG [M] | `float acc[4][4]` is indexed by runtime `sub` in `exl3_mma.cuh` (review §3.3) | `ptxas -v` / `cuobjdump -sass` on `exl3_linear_mma`, `exl3_moe_mma`, `exl3_mma.cuh`. Pass: 0 LDL/STL in the inner loop. If it spills, template on `tile_k ∈ {16,32}` and re-measure | Upstream precedent +8–17% per layer · **XS–S** |
| **N2** | **Proxy-thread / main-thread CPU placement** | MR [M] | **Mechanism does not apply** (team, 2026-10-10): Atlas Sparks run the stock CPU layout (no `isolcpus` / cgroup `AllowedCPUs`), the configuration that caused MR's 1.7–5.6 ms all-reduces. Atlas has no affinity code (0 hits ✔). | Optional only: if comms A/Bs show p99 all-reduce ≫ p50, log per-thread CPU and schedstat with `ATLAS_RDMA_PAIR_STATS=1` before attributing it to the fabric | Low · **XS** |
| **N3** | **Stale DFlash verify-cost table** (`dflash_width::step_ms(owners, rows)`) | LD audit, Atlas-internal | Measured once, 09-27/28. At least eight perf commits landed after it. [R ee3ecb82] width 7 at C1 is 94.0 ms (PDL) against 111.2 ms in the table, 18–21% high. The table has no history axis and no weight-format axis. | Sweep `ATLAS_DFLASH_FIXED_WIDTH` w=1..7 × owners {1,4,8} × history {0.5K, 4K, 16K, 64K} × prose/code, PDL on, n≥3. Regenerate the table if any cell is off by >10% or the per-row slope at ≥16K differs by >15%. Otherwise consider an online EWMA seeded from the table. Add a format axis before enabling EXL3 speculation (EXL3 per-row cost ≫ NVFP4 until X5 lands). | [E] 0–5% tok/s at long context. Also stops future kernel wins being lost to mis-pricing · **S–M** |
| **N4** | **Managed-memory OOM fallback should fail closed (or loud)** | VML [M] + MC [M] | Silent fallback to `cuMemAllocManaged` with one warning in `weights/loader/load_fns.rs:142-158`, `fast_weights/shard.rs:154-166`, `mistral_loader.rs`, `deepseek_v4/compute.rs` | Refuse when weights + reserve > `total_memory()`, or log the managed-memory byte total in the serve-start summary. Related: `hyper_connection_lowrank.rs:274-275` describes a model that loads at 113 of 119.6 GB, which would not fit on a 103.6 GiB box and would quietly fall back. | Prevents an invisible ~35% read penalty on any tensor that lands in managed memory · **S** |
| **N5** | **`vm.compaction_proactiveness=0`** on both Sparks | MR [M], cited by SG | Absent from the Atlas tree | ≥10 min C1 soak logging `/proc/vmstat` `compact_*` at 10 Hz, A/B the sysctl at 20 vs 0. Pass: no slow-step episodes correlated with compaction bursts, and mean−median step time shrinks | [M-MR] 63 slow-step episodes at ~1.72× every ~32–38 s, matching `kcompactd` bursts (~42K pages/s, 99% migrate-fail on GPU-pinned memory) at ~94% memory use. With the setting at 0: 0 bursts and 0 episodes, mean−median 5.6 → 0.8 ms. One boot per arm, so causation is [H] · **XS (config)**. Also protects every TPOT measurement. |
| **N6** | **Weight-integrity check + box qualification** (low priority: the known-bad boxes are retired) | MC [M], SGo [M] | No weight hash or scrub in Atlas `crates/` | Opt-in load-time hash of each uploaded tensor, read back once from the device ([E] ~0.25 s per ~60 GB per rank). Pass: 0 mismatches on a known-good box. Add a full-coverage CPU memtest result to the measurement-discipline fingerprint. **When a parity or byte-gate failure doesn't reproduce on another box, re-check hashes before bisecting kernels.** | Protects every bit-identical gate in the review (§3 [A], P3, CB5, the EXL3 byte gates) from hardware false failures. [M-MC] ~1000 bits/GiB flipped with EDAC silent; [M-SGo] shard sha256 changed on re-read · **S–M** |
| **N8** | **EXL3 KDA/MLA prefill crossover: packed vs transient BF16** | SG [M] | Atlas EXL3 linear re-decodes per row (rows sit in `grid.y`) | Sweep M = 1..8192 over three variants: packed, transient BF16 rebuild + MMA, and a smem-decode tile. Fix the crossover at load. | [M-SG] At M=7168, BF16 cuBLAS took 7.36 ms against 27.21 ms for FP8 Marlin (3.1–3.7× faster for M≥768); TTFT −11–12% · **M** |
| **N9** | **Find the source of C1 nondeterminism** | LD audit + [R ee3ecb82] | Cause unknown. `moe_decode_atomic_c4.cu` is a C4 path; other atomic users are `moe_permute.cu`, `glm_indexer.cu`, marlin and tq_plus. | Intersect the nsys kernel list of a GLM C1 verify step with the atomic and split-K kernels. Bisect by forcing ordered reductions; metric: C1 run-to-run byte identity | Diagnostic. Lets byte-identity gates work end to end again, and ties into P1's router-consistency point · **M** |
| **N10** | **`index_topk` 2048 vs tool-call fidelity** | SG [C, from NNNtrance] | Atlas hard-gates `index_topk == 2048` (`layers/ops/glm_sparse_native.rs:29` ✔) | Needle-in-JSON / tool-argument string probe at 32K, 64K and 128K context, n≥10, comparing topk 2048 vs 4096/8192 if the kernels allow it | Claim: 2048 corrupts tool-call string arguments past ~32K. Unverified, and affects quality rather than speed · **S** to test |
| **N11** | **RNR hazard if P9 adopts WRITE_WITH_IMM** | SG [M] | `crates/atlas-rdma/src/rdma_shim.c:173` `min_rnr_timer = 12` (≈0.64 ms), `:190` `rnr_retry = 7` ✔. Harmless today because Atlas posts plain RDMA WRITE only. | If IMM is adopted: pre-post enough receives, set `min_rnr_timer` to 1, and alarm on `rnr_nak_retry_err`. Pass: 0 per 10K all-reduces | [M-SG] One ring measured 403 RNR retries per token and ran at 5 tok/s. **Precondition for review item #10 (P9)** · **S** |
| **N12** | **Width settings in the measurement-discipline fingerprint** | LD audit | Rule 1 lists `--speculative + K`, but the real width comes from env vars | Add `ATLAS_DFLASH_ADAPTIVE_WIDTH`, `_FIXED_WIDTH`, `_WIDTH_PARAMS`, `ATLAS_DFLASH_ADAPTIVE*`, D-Cut, `ATLAS_MTP_RUNG_*`, `ATLAS_MTP_DRAFT_CONF` and `ATLAS_MTP_GATE_FORCE`, plus the realised width histogram | Stops comparisons between runs whose real widths differ · **XS** |
| **N13** | **Per-box read-bandwidth probe in the fingerprint** | VML [M] | — | Record a 10-second read probe per rank at serve start | [M-VML] The two nodes differ by 1.4–2.6% (234–237 vs 240–241 GB/s, one boot each). In EP2 the slower rank paces the step, so a cross-box difference under ~3% is not evidence · **XS** |
| **N14** | **Allocation-path read A/B** | VML [M] | — | Read kernel at 64 and 256 MiB, n≥3, on a known-good box over `cuMemAlloc`, `cuMemHostAlloc(PORTABLE\|DEVICEMAP)`, `cuMemAllocHost` and `cuMemAllocManaged`, plus optional `malloc + MADV_HUGEPAGE + register` (tests the THP hypothesis). Pass: each ≥0.95× `cuMemAlloc` | Decides N4. Also decides whether `rdma_pair`'s add kernel should stage its receive slot (`cuMemHostAlloc`, `rdma_pair.rs:11-16,155-166`) into device memory. [E] worst case ~0.15 ms per layer for a 67 MB prefill partial, mostly hidden; decode negligible · **S** |
| **N15** | **Drafter vocabulary subset: CPU-only pre-check** (gates review P10) | LD reasoning | P10 proposes a ~32K frequency-ranked drafter vocab | Per class (prose, code, tool JSON, multilingual, math), measure on held-out transcripts how often the emitted token falls outside the top-16K/32K/64K set. That rate is an upper bound on acceptance loss. Pass: ≤0.5% in every class, then run the GPU A/B | Free. Gates P10's [E] ~2% · **XS** |
| **N16** | **Recovery after idle** | MR [M] | — | Time the first 40 decode steps after 30, 120 and 300 s idle | [M-MR] The first ~33 steps after ≥30 s idle ran ~5× slower; this disappeared under pinning, with a tuner confound unresolved. Relevant to interactive latency · **XS** |
| **N17** | **Indexer buffers sized in pools, not tokens** | SG [M] | Not checked in Atlas | Audit Atlas's indexer key/score buffer sizing | vLLM sizes these in tokens and over-allocates 4×. Capacity only · **XS** |
| **N18** | **Load-path memory guard** | SG [M] | Related to N4 | Bound page-cache use during weight load | [M-SG] A page-cache-heavy load caused `NV_ERR_NO_MEMORY` and rebooted both nodes · **S** |
| **N19** | **Temperature in the fingerprint** for long-prefill and TTFT runs | MR [M], MC [C] | — | Record SoC/DRAM temperature with any 16K+ TTFT or long-soak number | [M-MR] Two worker hangs during a 64K warm-up prefill took the head RoCE links down, at 79–84 °C. [C-MC] Memory faults are temperature-dependent (no thermal log). Also challenges `atlas-plugin/src/hardware/policy.rs:255-261`'s rationale that "accuracy is not thermally sensitive". · **XS** |

**Additional review findings exposed by the cross-reference:**
- **EXL3 MoE tile width:** [M-MR] in upstream's kernel, N128 was 14–17% slower than N256 (6 matched pairs). Atlas's EXL3 MoE uses N128 per CTA, so include an N256 variant in the §3.1/§3.4 screen. This is a prior from a different kernel, not a result.
- **DFlash prefill capture:** `try_dflash_prefill_capture_layer` issues one strided d2d copy per token per capture layer. Its own comment estimates 640–20,480 launches per prefill and suggests a strided-scatter kernel if profiling shows a cost (`impl_b3_dflash.rs:107-111` ✔).

---

## 5. Closed doors (record so nobody re-opens them)

| Idea | Evidence | Source |
|---|---|---|
| Compressing EXL3 trellis weights further (lossless codec) | Best ratio 0.99998 across 3,710 codec × transform cells; no duplicate tensors | [M-MR] |
| Hardware decompression engine | GB10 is not on nvCOMP's Decompression Engine list | [C-MR] |
| CPU-side offload or decompression to feed the GPU | Max 29 GB/s on 10 A725 cores (3.2 GB/s with wrong affinity); Grace cores share the LPDDR5X bandwidth that verify is bound by | [M-MR], LD |
| KV content compression | Compresses only ~1.23×, and would sit on the hot path | [M-MR] |
| Physical page placement or colouring as a bandwidth lever | Every arm across physical bits 12–28 and three 16 GiB intervals read 153.7–157.9 GB/s, identical to the spread arm; 16 GiB ranges look like capacity regions, not bandwidth domains [H] | [M-VML] |
| Optimistic lookahead drafting (draft the next block before verify commits) | DFlash/MTP condition on target hidden states, so a pre-draft must assume full acceptance, which is rare on prose (positions 6–7 at ≤10%). On one GB10 it would also compete for DRAM. [E] <1% | LD reasoning + [M-SG] acceptance |
| Over-splitting K in the cooperative MoE | An 8192-block narrow-tile grid lost to 2048 blocks | [M-SG] |

---

## 6. Revised priorities (review §1 and §8, adjusted)

These changes come from the evidence above; the review's other rankings stand.

1. **N1 (spill audit)** and **N5 (compaction sysctl)** go first: near-free with measured precedent. Do them before any perf A/B. N5 also cleans up step-time tails for every later measurement.
2. **N11** (RNR settings) is a precondition for review item #10 (P9 round-trip removal). N2 is dropped to optional, because Atlas Sparks use the stock CPU layout.
3. Review §3.4 / X5: EXL3 expert grouping is promoted for **prefill**, with SG's E3 kernel as the reference (31 ms per layer at T=7168). On the decode side, expect ~1.4–1.5×, not 6.8×.
4. Review #5 / CB1 is demoted from ~2.9 to ~1.5 ms/step expected (X1), but it is still worth doing alongside P1's overlap-in-capture.
5. Review §3.6: the indexer gate GEMV is promoted (X3, ~73 µs per layer measured in the reference).
6. **N4 (fail closed or loud on managed fallback)** is a cheap guard. N6 (weight integrity) drops to low priority now that the known-bad boxes are retired.
7. **N3 (DFlash cost table)** should be regenerated after the next batch of kernel wins lands, and must have a format axis before EXL3 speculation is enabled.
