# Plan: Launch DeepSeek-V4.1-Flash + dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw

**Created:** 2026-10-15 | **Status:** Planning Phase | **HF_TOKEN:** `<HF_TOKEN redacted>`

---

## 1. Model Overview (Two Variants)

### 1.1 Official: deepseek-ai/DeepSeek-V4.1-Flash
- **Architecture:** Causal Encoder-Decoder (CED) — 20-layer causal encoder + 20-layer decoder
- **Backbone:** 552B total params, 8B active (prefill) / 16B active (decode)
- **Vision:** DeepSeek-ViT (32 layers, 1024 dim, 16 heads, patch_size=14, 2D-RoPE, downsample=3, max_tokens=1024)
- **Attention:** CSA2 (Compressed Sparse Attention 2) — Full / Reindex / Reuse modes with cross-layer KV reuse
- **MoE:** 384 routed experts, 6 active, 2304 intermediate, sqrtsoftplus routing
- **Memory:** Engram (196B params, sparsely accessed, at layers 1, 14), Single-Pass mHC
- **Speculative:** DSpark (separate training, semi-autoregressive draft)
- **KV Cache:** FP4 global KV (890 bytes/token), SWA Bounded Replay (persistent on SSD)
- **Quantization:** FP8 weights (block 32×32), FP4 expert weights, FP4 KV
- **Context:** Up to 1M tokens (1048576 max_position_embeddings)
- **Model Type:** `deepseek_v41` (text: `deepseek_v41_text`, vision: `deepseek_v41_vision`)
- **Config:** Nested `text_config` + `vision_config` (NOT flat like V4-Flash)
- **Image Token:** `<｜deepseek_image｜>` at id `129264`
- **License:** MIT

### 1.2 Uncensored Quantized: dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw
- **Source:** `https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`
- **Status:** Consolidated — repo points to FP8 version (`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8`)
- **Quant Method:** EXL3 v1.4.2 (not FP8)
- **Bitrate:** 2.9 avg bits/weight (head_bits=6, codebook=mul1)
- **Calibration:** 250 rows × 2048 cols
- **Original Quant:** FP8 with expert_dtype=fp4, weight_block_size=[32,32], scale_fmt=ue8m0
- **Weights:** Same checkpoint structure, quantized via EXL3 trellis (I16 trellis, F16 scales, mul1 marker)
- **Safety:** Weight-level abliteration — guardrails surgically removed, capability preserved (MMLU ~99.4%)
- **Note:** The EXL3 repo states: "This repository has been consolidated into a single UNCENSORED release. Please use → `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8`."
- **HL Token:** `HF_TOKEN=<HF_TOKEN redacted>` (for private access)

---

## 2. Critical Discovery: V4.1-Flash is NOT Compatible with V4-Flash

From the arXiv paper (`2609.19969`) and HF config:

| Dimension | V4-Flash (Atlas supports) | V4.1-Flash (New architecture) |
|-----------|-----------------------------|--------------------------------|
| **Structure** | Flat `config.json` | **Nested**: `text_config` + `vision_config` |
| **Layers** | 43 (all decode) | **40** (20 encoder + 20 decoder) |
| **Hidden Size** | 4096 | **5120** |
| **Experts** | 256 routed | **384 routed** |
| **Head Dim** | 512 | **512** (same) |
| **Q LoRA** | 1024 | **1280** |
| **O LoRA** | 1024 | **1024** |
| **O Groups** | 8 | **8** |
| **KV Lora** | 512 | Not explicitly declared |
| **Compression** | CSA/HCA hybrid (ratios: 0,0,4,128...) | **CSA2 pure** (ratios: 0,0,2,2... with 2:1 compression) |
| **KV Source Layers** | Not explicit | **[2,8,14,20]** |
| **Index Source Layers** | Not explicit | **[2,8,14,20,24,28,32,36]** |
| **Index Heads** | 64 | **32** |
| **Index Head Dim** | 128 | **128** |
| **Candidate Source** | Not present | **20**; topk_blocks=2048, block_size=8 |
| **RoPE Theta** | 160000 (compress) | 160000 (compress) |
| **Window** | 128 | **128** |
| **mHC** | Standard | **Single-Pass** (halves activation traffic) |
| **Engram** | None | **Layers 1, 14**; vocab 16000000; n_heads=8, dim=256 |
| **DSpark** | MTP (K=5) | **DSpark** (separate, bs=5, noise_tok=128799, targets=[37,38,39]) |
| **Vision** | 32 layers, 1024 dim, 16 heads, 14 patch | **Same** (deepseek_v41_vision) — but with pixel-unshuffle (3×3) |
| **Image Token** | `<｜deepseek_image｜>` id 129264 | **Same** |
| **Vision Max Tokens** | 384 | **1024** (increased) |
| **Vision Min Pixels** | 147456 | **295936** (increased) |
| **Weight Block** | [128,128] | **[32,32]** (smaller blocks) |
| **Quant** | FP8 / NVFP4 | **FP8 + FP4 experts** (original) / **EXL3 2.9bpw** (dealignai) |

---

## 3. What This Means for Atlas

The V4.1-Flash architecture requires:

### 3.1 New Model Type & Parser
```rust
// factory.rs
"deepseek_v41" => Ok(Box::new(DeepSeekV41WeightLoader)),

// config/dispatch.rs
"deepseek_v41" => parse_deepseek_v41(json),  // NEW PARSER
```

### 3.2 Nested Config Parsing (Critical Difference)
The `text_config` block must be parsed separately from `vision_config`. Unlike `qwen3_5_moe` which uses nested `text_config`, V4.1 uses `deepseek_v41_text` as the text model type and `deepseek_v41_vision` as vision.

```python
# Config structure
{
  "model_type": "deepseek_v41",
  "text_config": {"model_type": "deepseek_v41_text", ...},
  "vision_config": {"model_type": "deepseek_v41_vision", "num_hidden_layers": 32, ...},
  "quantization_config": {"quant_method": "fp8", "weight_block_size": [32, 32], ...},
  ...
}
```

### 3.3 CED Architecture (Major New Work)
Not a simple decoder — it has a **20-layer encoder** (layers 0-19) and **20-layer decoder** (layers 20-39):
- Encoder: First 2 layers use SWA only; rest use CSA2
- Decoder: All layers use CSA2 with cross-layer KV reuse
- Decoder global KV is **projected from final encoder hidden states** (not computed per layer)
- This requires new layer types and a new build path

### 3.4 CSA2 Attention (New Attention Mechanism)
From the inference `model.py`:
- `compress_ratios`: per-layer (0 = SWA only, r = pool r-to-1)
- `kv_source_layers`: which layers produce compressed KV (shared across layers with same ratio)
- `index_source_layers`: which layers produce index keys
- `candidate_source_layer`: layer that selects candidate blocks (20)
- Three modes per layer: Full → Reindex → Reuse
- Cross-layer KV reuse reduces storage ~1/4 vs V4-Flash

### 3.5 DeepSeek-ViT Vision (Different from Qwen3-VL)
From `inference/vision.py`:
- `PatchEmbed` uses `Linear(3 * patch_size², vision_dim)` not Conv2d
- Attention: `wqkv` single linear → chunk → 2D RoPE → SDPA
- MLP: SwiGLU (`w1` → chunk gate/up → `w2`)
- `ViT.forward`: applies 2D RoPE via `get_vision_cos_sin`
- `Aligner`: `downsample_ratio=3` → unfold (3×3) → Linear → GELU → Linear
- Pixel-unshuffle 3×3 reduces token count by 9× (up to ~1344×1344 support)

### 3.6 Engram Module (New — 196B Parameters)
From `model.py`:
```python
class Engram(nn.Module):
    def __init__(self, args, layer_id, layout):
        # ParallelEngramEmbedding: FP8 weights + F16 scales
        # n-gram hash lookup (hash_ids) → wkv projection
        # Gate: normalized dot product of x vs key → sigmoid
        # Writes into residual stream
```
- Layers: 1, 14
- Embeddings: 384006168 + 384016682 (two tables)
- Max n-gram: 4
- Compressed vocab: 99092 (from 16000000)
- 8 heads, 256 dim
- Weight format: FP8 (like main model)

### 3.7 DSpark Speculative Decoding (New — Separate Training)
From `model.py`:
```python
dspark_block_size = 5
dspark_noise_token_id = 128799
dspark_target_layer_ids = (37, 38, 39)
dspark_markov_rank = 256
dspark_n_routed_experts = 128
dspark_n_activated_experts = 3
```
- Not MTP — DSpark is a separate spec-decoding architecture
- Draft head uses 128 experts, 3 active, block size 5
- Target layers: 37, 38, 39 (last 3 layers of 40)
- Noise token: 128799 (special token for draft)
- Trained separately after backbone

### 3.8 Image Token Handling (Same Token, New Format)
From `image_processor.py`:
- `image_token_id = 129264` (from config.json)
- Placeholder: `<｜deepseek_image｜>` (from encoding.py)
- Token types: `TEXT=-1, IMAGE_START=0, IMAGE=1, IMAGE_NEW_LINE=2, IMAGE_END=3`
- Pattern: `[IMAGE_START] + ([IMAGE]*n_llm_w + [IMAGE_NEW_LINE])*n_llm_h + [IMAGE_END]`
- Every IMAGE position carries `image_token_id` in `input_ids`
- Only token types distinguish image from text positions
- `max_image_tokens = 1024` (vs 384 for V4-Flash)
- `min_pixels = 544 * 544 = 295936` (vs 147456)

---

## 4. Plan Document: V4.1-Flash + dealignai EXL3 Variant

### 4.1 Official V4.1-Flash Plan

Same structure as V4-Vision-Exp but for the new architecture:

```markdown
# Plan: DeepSeek-V4.1-Flash Launch

## Status: NEW MODEL FAMILY — Separate Epic from V4-Flash

### Phase 1: Foundation (Weeks 1-2)
- [ ] Register `deepseek_v41` in factory.rs
- [ ] Create parser `parse_deepseek_v41()` (nested text_config/vision_config)
- [ ] Create `DeepSeekV41WeightLoader`
- [ ] Create kernel target `kernels/gb10/deepseek-v4.1-flash/nvfp4/`
- [ ] Create MODEL.toml with correct params (5120 hidden, 384 experts, etc.)

### Phase 2: CED Architecture (Weeks 3-5)
- [ ] Implement encoder/decoder layer split
- [ ] Implement CSA2 attention (Full/Reindex/Reuse modes)
- [ ] Implement cross-layer KV reuse
- [ ] Implement Hierarchical Sparse Indexer
- [ ] Implement Single-Pass mHC

### Phase 3: Memory & Extensions (Weeks 6-7)
- [ ] Implement Engram module (196B params, FP8 weights)
- [ ] Implement FP4 KV cache support
- [ ] Implement SWA Bounded Replay
- [ ] Integrate DSpark (separate draft model)

### Phase 4: Vision (Weeks 8-10)
- [ ] Implement DeepSeek-ViT encoder (already similar to V4-Flash vision)
- [ ] Implement Aligner with 3×3 pixel-unshuffle
- [ ] Implement image preprocessing (image_processor.py)
- [ ] Add image token handling (129264)
- [ ] Add vision weight loading (model.visual.*)

### Phase 5: Quant Integration (Weeks 11-12)
- [ ] Support FP8 weights (block 32×32, ue8m0)
- [ ] Support FP4 expert weights
- [ ] Integrate EXL3 path for dealignai variant (2.9bpw)
- [ ] Verify EXL3 checkpoint compatibility (mul1 codebook, I16 trellis, F16 scales)

### Phase 6: Testing (Weeks 13-15)
- [ ] Unit tests for all new components
- [ ] Integration with official checkpoint
- [ ] Integration with dealignai EXL3 checkpoint
- [ ] Verifications: coherence, agentic, vision, tool calling
```

**Estimated: 15 weeks (3.5 months) for official; +2 weeks for EXL3 = ~17 weeks**

---

## 5. The dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw Variant

### 5.1 Source Info
- **Repo:** `https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`
- **Status:** Consolidated to FP8 version; EXL3 repo points to FP8
- **Model:** Same `deepseek_v41` architecture, quantized via EXL3
- **Access:** Private — requires `HF_TOKEN=<HF_TOKEN redacted>`
- **Weight Format:** EXL3 v1.4.2 (trellis I16, scales F16, mul1 codebook)
- **Bitrate:** 2.9 avg (head_bits=6, calibration 250×2048)
- **Quant Config:** `quant_method="exl3"`, `original_quantization_config` was FP8

### 5.2 EXL3 Compatibility with Atlas

Atlas has EXL3 support (from `spark-runtime/src/weights/exl3.rs`):
- Validates `mul1` marker (`0x83dc_d12d`)
- Requires I16 trellis, F16 input/output scales
- Requires dimensions divisible by 128 (Hadamard width)
- Currently used for GLM-5.3 with TP=2, EP=1

**The dealignai EXL3 weights should be compatible** because:
- They use `mul1` codebook (verified in config.json)
- Trellis format matches (I16)
- Bitrate 2.9 is within supported 1-8 range
- Scales are F16

**Potential issues:**
- V4.1-Flash uses 5120 hidden size (vs 4096 for V4-Flash) — must be divisible by 128 ✓ (5120/128=40)
- 384 experts — need to verify TP/EP sharding works
- Block size 32×32 in original (different from V4-Flash 128×128)
- Engram tables (196B) may need special handling

### 5.3 Plan for EXL3 Variant

```markdown
## dealignai EXL3 Variant — Additional Tasks

### Phase A: EXL3 Weight Validation (Week 1)
- [ ] Download weights with HF_TOKEN
- [ ] Verify checkpoint layout (model.safetensors.index.json, config.json)
- [ ] Validate EXL3 format (mul1 marker, trellis shapes, scale vectors)
- [ ] Check weight names match `model.*`, `model.visual.*` pattern
- [ ] Verify Enrram weights present (or missing, per `engram_table_dir=null`)

### Phase B: EXL3 Kernel Integration (Week 2)
- [ ] Use existing `exl3_linear_mma` kernels from GLM-5.3
- [ ] Verify block_size 32 compatibility (vs 128 for V4-Flash)
- [ ] Test with TP=1, TP=2 configurations
- [ ] Verify FP4 expert weights load correctly

### Phase C: Uncensored Model Integration (Week 3)
- [ ] Load EXL3 weights into WeightStore
- [ ] Build model with EXL3 layer loader
- [ ] Verify inference produces coherent output
- [ ] Safety: confirm guardrail removal (MMLU ~99.4% per dealignai claims)

### Phase D: Testing (Week 4)
- [ ] Benchmark comparison: FP8 vs EXL3 2.9bpw (quality vs speed)
- [ ] Verify no quality degradation at 2.9bpw
- [ ] Confirm vision encoder works with EXL3 quantized vision weights
```

---

## 6. Access Credentials

```
HF_TOKEN=<HF_TOKEN redacted>
```

Use for:
- `huggingface-cli download dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`
- Private model access (repo is private, requires token)
- Note: Repo is consolidated; use FP8 version if EXL3 unavailable

---

## 7. Key Files to Reference (From HF Repos)

### Official V4.1-Flash:
- `https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json`
- `https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/model.py`
- `https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/vision.py`
- `https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/image_processor.py`
- `https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/encoding/encoding.py`

### Uncensored EXL3:
- `https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw/raw/main/config.json`
- Note: Repo redirected to FP8 version; check `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8`

---

## 8. Implementation Checklist (Combined Plan)

### For Official V4.1-Flash:
- [ ] Register `deepseek_v41` model type
- [ ] Create nested config parser (`parse_deepseek_v41`)
- [ ] Implement CED architecture (encoder/decoder split)
- [ ] Implement CSA2 attention with cross-layer reuse
- [ ] Implement Hierarchical Sparse Indexer
- [ ] Implement Single-Pass mHC
- [ ] Implement Engram (FP8 weights, n-gram lookup)
- [ ] Implement FP4 KV cache
- [ ] Implement SWA Bounded Replay
- [ ] Add DSpark support (separate drafter)
- [ ] Implement DeepSeek-ViT vision encoder
- [ ] Add vision weight loading
- [ ] Create MODEL.toml and kernel targets
- [ ] Test with official checkpoint

### For dealignai EXL3 Variant (Additional):
- [ ] Download with HF_TOKEN (verify repo status)
- [ ] Validate EXL3 checkpoint (trellis, scales, mul1)
- [ ] Load EXL3 weights (use existing GLM-5.3 path if compatible)
- [ ] Test quality at 2.9bpw vs FP8
- [ ] Test with vision input
- [ ] Verify uncensored behavior (no guardrail refusal)
- [ ] Benchmark performance

---

## 9. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-10-15 | Plan V4.1-Flash as separate epic | Completely different architecture (CED vs decoder-only) |
| 2026-10-15 | Include dealignai EXL3 variant | User explicitly requested; repo is private (needs HF_TOKEN) |
| 2026-10-15 | Note EXL3 repo consolidation | Repo points to FP8 version; EXL3 may be unavailable |
| 2026-10-15 | Keep V4-Flash-Vision-Exp separate | V4.1 requires 5-7 months; V4-Flash vision is ~10-15 days |
| 2026-10-15 | Use `image_token_id=129264` | From V4.1 config.json; `<｜deepseek_image｜>` placeholder |
| 2026-10-15 | Note CED encoder/decoder split | 20+20 layers; decoder KV from encoder (not per-layer) |

---

*Plan complete. Two model variants covered: official `deepseek-ai/DeepSeek-V4.1-Flash` (552B CED) and `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw` (EXL3 2.9bpw quantized, now consolidated to FP8). Both require the new `deepseek_v41` model type and the full CED architecture implementation.*
