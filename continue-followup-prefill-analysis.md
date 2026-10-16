# Prefill / MoE grouping — capture-diagnosed, vLLM-compared, optimized (2026-10-16)
Branch context: `perf/decode-vllm-parity-20261015` (worktree `../atlas-perf`).
Evidence root (local / unpushed): `/home/colum/Workspace/RustroverProjects/atlas-perf/`.
No commits to shared branch; nothing pushed.

## 1. What was asked (disambiguated from continue.md + todo.md + user's clarifying answers)
- Component: **Prefill / MoE grouping** (`crates/spark-model/src/layers/moe/forward_prefill.rs`; `ep_prefill.rs`; grouping pipeline).
- Evidence: **Prefill/indexer snapshots** — `exl3-indexer-localization-result-20261015.md`; `research/flash-kda-prefill/READM E.md`; `decode-vllm-parity-ledger-20261015.md` (quoted for decode baseline only).
- Approach: **Diagnose via captures, then optimize** (both correctness + speed).
- Constraint (from continue.md §0, AGENTS.md): do not edit `sparkglm/installable-20260924`; work on new `followup/*` branch; use bench lock; label runs `<item>-<label>`.

## 2. Implementation dug (lazy read, not full audit — ponytail)
File: `crates/spark-model/src/layers/moe/forward_prefill.rs` (line 1–~300 shown above).
Pipeline (from code comments):
  gate → topK → sort → grouped gate/up GEMM → SiLU → grouped down GEMM → unpermute + weighted reduce → shared-expert blend.
Key feature (line 18–26): grouped GEMM cuts LPDDR5X reads ~6 GB → ~150 MB at N=1024 (weight-stationary per expert, tensor-core).
Falling back to `forward_batched` (per-token loop) when `num_tokens <= 8` (todo.md P2 note); also forced on `atlas_hip` for BF16 (grouped PTX stub missing — now ported for FP8 only, per code comment lines 55–65).
M2 padding / tails open; short-prefill (<8 tokens) loop remains.
Index/code path for sparse MLA (`atomics` / `mask` / indexer) lives in `glm5_kda/`, not in this file — divergence is downstream of grouping.

## 3. vLLM comparison (from `exl3-vllm-reference-contract.md`, rev 2026-10-12)
| Dimension | vLLM reference (SM120, pinned) | Atlas prefill grouping (current) | Divergence / implication |
|---|---|---|---|
| Prefill backend | `flashinfer_mla_sparse_sm120.py`: packed cache (`fp8_ds_mla`), dense-prefill disabled (`:36`, `:67`), MQA route at `:824` | Uses packed cache via `forward_prefill`; `dense_prefill_needs_full_recompute()` returns hits=misses=0 (continue.md 132) | No bypass; matches contract. |
| Head / gather | `mlu_attention.py`; head `pad=False` (`:3232`); logit export after TP gather + vocab truncation (`:143–153`) | Grouped GEMM output unpermutes then reduces; shared-expert blend after; TP gather separate | Numeric path differs only by reduction order (same pool set, permuted — see below). |
| Sampling / determinism | Gumbel greedy only if temp > 0 (`sample/gumbel.py:135`); seed 42 records policy not arithmetic | Not in grouping file; divergence localized in downstream indexer (see §4) | Correctness fix belongs in indexer, not grouping kernel. |
| Chunk size | Reference collects 128-token chunks (`reference_chunk_trace.py:44`, `max_tokens=1`) | `ab_full.sh` uses 128-token chunks (`continue.md` 560); grouping runs for all chunk sizes | Aligned; optimization target is larger/N=1024 where grouping wins. |

## 4. Capture-snapshot diagnosis (the evidence the user asked to use)
Source: `exl3-indexer-localization-result-20261015.md` (registered 2026-10-15; dumps on Sparks at `/home/colum/idx-20261015/`; not copied locally — read from worktree doc).
Method: 3 pairs × 4 rows × 2 ranks, loop routed (`EXL3_FUSED_MOE=0`), control row 2,051 tokens.
Finding (quoted, with file refs):
- Control (≤2,051 tokens): byte-identical in all 6 pair×rank comparisons, 3,927 tensors.
- Divergent rows (2,052–2,054): 14/18 first differ at indexer `attn-indexer_indexer_op`; 4/18 at reduced `o_proj` through TP gather.
- **Same set, different order.** 216 valid token rows checked; 38 differ in bytes; **0 select a different key set** (only order changes, up to 2,044/2,044 permuted). Pool regime: 513 complete pools of 4, select 512 + tail.
- Kernel confirmation (`arithmetic-audit/reference-topk-repeat32.json`): `top_k_per_row_prefill` with distinct scores returns fixed set in non-deterministic order (3–7 orders observed depending on score distribution).
- Downstream effect: sparse MLA accumulates over indices in buffer order → FP32 result changes → propagates through reduce → residual.

Implication for optimization (correctness-first leg):
- The grouped prefill GEMM is **not the divergence source** — divergence is the indexer's order-sensitive accumulation above 2,051 tokens.
- Correctness fix = sort selected pool IDs before expansion (unenumerated here; requires indexer path change, not grouping kernel).
- Once fixed, speed optimization can proceed safely.

Secondary captured divergence (continue.md 125–126, HSL result): routed MoE fused kernel (FP32 atomic accumulation of top-8 experts) differs from layer 3; removing `EXL3_FUSED_MOE=0` removes variation below 2,052 tokens. Not in grouping file, but relevant because grouped prefill feeds the MoE; keep `FUSED_MOE=0` for deterministic capture-verified runs until attribution complete.

## 5. Optimization recommendations (lazy; smallest diff that holds)
A. **Correctness — indexer order (captures required)**
  - Before expanding selected pools in `sparse_attn_indexer_kpool.py` (vLLM source line referenced in doc: `torch.ops._C.top_k_per_row_prefill`), sort the selected pool-ID list (ascending, stable).
  - Verify with same capture protocol: 3 pairs × 2 ranks, 2,052-token row, expect 0 differing index buffers and 0 differing `o_proj`.
  - Label: `prefill-idx-sort-<n>`; output to `/target/mb_prefill_idx_sort_*` per bench lock protocol.

B. **Performance — prefill grouping (close open items from todo.md P)**
  - M2 padding / tails (open in todo P2): add padded group tiles so groups that don't fill full tiles don't fall back; keep per-token fallback only for `num_tokens <= 8`.
  - Short-prefill loop (< 8 tokens, `forward_prefill` guard): verify it stays per-token but uses the same BF16/FP8 weight-stationary loader (no double load).
  - In-model TTFT measurement (`continue.md` 619: ~18–19 tok/s on 1,527-token prompt / 86 s): measure grouped vs batched at N=256 / 512 / 1024 / 2048 after (A) passes.

C. **Candidate accelerator — FlashKDA (optional, not default)**
  - Available at `atlas-perf/research/flash-kda-prefill/` (adapter.cuh / runtime_bridge.cu). Eligibility: H32/D128, 2048–4100 rows, lower-bound−5 only.
  - Independent gate: 3.38–3.62x at 2048/4096/4100; standalone, not full-model.
  - Condition: enable only with `ATLAS_KDA_FLASH_PREFILL=1` + absolute library path; abort on missing symbol (README.md). Do not change default binary or install dependency; document as opt-in only.
  - Recommendation: do NOT integrate into default `spark` until (A) verified and (B) TTFT contract established; keep adapter as separate `followup/flash-prefill` worktree.

D. **What NOT to change (from capture discipline + AGENTS.md)**
  - Do not modify `sparkglm/installable-20260924` (shared branch, other sessions' files).
  - Do not commit to `perf/decode-vllm-parity-20261015` (per continue.md §0); new work on `followup/prefill-moe-group`.
  - Do not claim speed improvement without ≥3 repeats in same session with same binary (measurement-discipline skill); until then label "preliminary, single run".

## 6. Evidence checklist (used)
- `continue.md` §§8–9 (performance table; decode ledger pointer; prefill TTFT; F4 phase; server state).
- `todo.md` "Now" B (§P prefill), "Now" §4 (bisect / capture defects), R-e (branch discipline).
- `crates/spark-model/src/layers/moe/forward_prefill.rs` (grouping pipeline; HIP/FP8 exception; M2 open).
- `docs/design/exl3-indexer-localization-result-20261015.md` (prefill/indexer capture divergence; pool-set same, order different).
- `docs/design/exl3-vllm-reference-contract.md` (SM120 prefill contract; head pad; sampling; chunking).
- `atlas-perf/research/flash-kda-prefill/README.md` (adapter eligibility; 3.4–3.6x claim; opt-in conditions).
- `decisions.md` / `assumptions.md` (new entries registered this turn).

No source file edited in this turn (analysis only). Next step: create `followup/prefill-moe-group` branch, apply (A) indexer sort, capture-verify, then (B) M2/padding.
