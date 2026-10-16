# Plan: Launch DeepSeek-V4-Flash-Vision-Exp on Atlas

## Executive Summary

DeepSeek-V4-Flash-Vision-Exp is a **multimodal variant** of the DeepSeek-V4-Flash architecture. It adds:
- A **vision encoder** (ViT with 2D RoPE)
- An **aligner** (spatial downsample + projection to LLM hidden dim)
- **Image token placeholder**: `<｜deepseek_image｜>` (not `<|image_pad|>`)

The text model is identical to DeepSeek-V4-Flash (already supported in Atlas). This plan covers the **vision-specific additions** needed.

---

## Current State Analysis

### ✅ Already Working (Text Path)
- `deepseek_v4` model type registered in `factory.rs`
- `DeepSeekV4WeightLoader` implemented in `weight_loader/deepseek_v4.rs`
- `deepseek_v4.jinja` template exists (checkpoint-native encoding)
- `MODEL.toml` at `kernels/gb10/deepseek-v4-flash/MODEL.toml` (text-only)
- Parser for `deepseek_v4` config in `atlas-core/config/parsers/deepseek_v4.rs`

### ❌ Missing (Vision Path)

| Component | Status | Location Needed |
|-----------|--------|-----------------|
| Vision config parsing for DeepSeek-V4 | ❌ | `atlas-core/config/parsers/deepseek_v4.rs` |
| Vision encoder (ViT) for DeepSeek-V4 | ❌ | `spark-model/src/layers/vision_encoder/` |
| Aligner (downsample + projection) | ❌ | `spark-model/src/layers/vision_encoder/` |
| Image token ID resolution (`<｜deepseek_image｜>`) | ❌ | `spark-server/src/tokenizer/` |
| Vision pad expansion (fan-out) | ❌ | `spark-server/src/tokenizer/chat_impl.rs` |
| Image preprocessing (CPU) | ❌ | `spark-model/src/vision_preprocess.rs` |
| Vision weight loading in `DeepSeekV4WeightLoader` | ❌ | `spark-model/src/weight_loader/deepseek_v4.rs` |
| MODEL.toml for vision variant | ❌ | `kernels/gb10/deepseek-v4-flash-vision/` |

---

## Detailed Implementation Plan

### 1. Vision Config Parsing (`atlas-core`)

**File**: `crates/atlas-core/src/config/parsers/deepseek_v4.rs`

Add vision config parsing to `parse_deepseek_v4()`:

```rust
// After parsing text config, add:
config.vision = parse_deepseek_v4_vision_config(&raw)?;
```

Create `parse_deepseek_v4_vision_config()` that reads from `config.json`:
- `vision_n_layers: 32`
- `vision_dim: 1024`
- `vision_n_heads: 16`
- `vision_inter_dim: 2816`
- `vision_patch_size: 14`
- `vision_rope_theta: 10000.0`
- `vision_downsample_ratio: 3`
- `vision_max_n_token: 384`
- `vision_min_pixels: 147456`
- `vision_max_wh_ratio: 8`

**Image token ID**: The tokenizer uses `<｜deepseek_image｜>` (from `encoding_dsv4.py`). This must be resolved at runtime from the tokenizer.

### 2. Vision Encoder + Aligner (`spark-model`)

**File**: `crates/spark-model/src/layers/vision_encoder/`

The DeepSeek-V4 vision encoder differs from Qwen3-VL:
- **Full bidirectional attention** (not causal)
- **2D RoPE** on patches (height + width positions)
- **Aligner**: downsample_ratio=3 spatial merge + linear projection

Create new module: `crates/spark-model/src/layers/vision_encoder/deepseek_v4.rs`

```rust
// Key differences from Qwen ViT:
// - Attention: F.scaled_dot_product_attention (bidirectional)
// - RoPE: 2D (separate hpos, wpos frequencies)
// - Aligner: unfold(downsample_ratio) + Linear(in_dim, dim) + GELU + Linear(dim, dim)
```

Register kernels in `vision_encoder.rs` init:
- `deepseek_vit_gemm_bias`
- `deepseek_vit_attention_rope`
- `deepseek_vit_norm`
- `deepseek_aligner_gemm`
- `deepseek_aligner_gelu`

### 3. Vision Weight Loading (`spark-model/weight_loader`)

**File**: `crates/spark-model/src/weight_loader/deepseek_v4.rs`

Add `load_vision_encoder()` method to `DeepSeekV4WeightLoader`:

```rust
fn load_vision_encoder(
    &self,
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<VisionEncoder>> {
    let vcfg = config.vision.as_ref()?;
    let vp = "model.visual";  // DeepSeek uses "model.visual" prefix
    
    // Load ViT blocks (32 layers)
    // Load Aligner (downsample + 2-layer MLP)
    // Return VisionEncoder
}
```

Weight naming (from HF config + DeepSeek inference code):
- `model.visual.patch_embed.proj.weight` / `.bias`
- `model.visual.blocks.{i}.norm1.weight` / `.bias`
- `model.visual.blocks.{i}.attn.qkv.weight` / `.bias`
- `model.visual.blocks.{i}.attn.proj.weight` / `.bias`
- `model.visual.blocks.{i}.norm2.weight` / `.bias`
- `model.visual.blocks.{i}.mlp.linear_fc1.weight` / `.bias`
- `model.visual.blocks.{i}.mlp.linear_fc2.weight` / `.bias`
- `model.visual.norm.weight`
- `model.visual.merger.w1.weight` / `.bias` (aligner first linear)
- `model.visual.merger.w2.weight` / `.bias` (aligner second linear)

### 4. Image Token Handling (`spark-server/tokenizer`)

**File**: `crates/spark-server/src/tokenizer/chat_impl.rs`

Add DeepSeek-V4 image token support:

```rust
pub fn deepseek_v4_image_token_id(&self) -> Option<u32> {
    self.encode("<｜deepseek_image｜>")
        .ok()
        .and_then(|ids| if ids.len() == 1 { Some(ids[0]) } else { None })
}
```

**File**: `crates/spark-server/src/tokenizer/deepseek_v4/preprocess.rs` or new `vision.rs`

Add vision content block rendering - the encoding already emits `<｜deepseek_image｜>` for images.

### 5. Vision Pad Expansion (`spark-server/tokenizer/chat_impl.rs`)

The DeepSeek-V4 template emits **one** `<｜deepseek_image｜>` per image, but the vision encoder produces `grid_h * grid_w` patches. Need fan-out similar to `expand_vision_pads` but for DeepSeek token.

```rust
pub fn expand_deepseek_v4_vision_pads(&self, tokens: Vec<u32>, pad_counts: &[usize]) -> Vec<u32> {
    let image_token = self.deepseek_v4_image_token_id()?;
    // Same logic as expand_vision_pads but matching deepseek_image token
}
```

Call this in `template.rs` render path when model is `deepseek_v4` and vision placeholders exist.

### 6. Image Preprocessing (`spark-model/vision_preprocess.rs`)

Add DeepSeek-V4 preprocessing function:

```rust
pub fn preprocess_deepseek_v4_image(
    data_uri: &str,
    vcfg: &VisionConfig,
    max_pixels: Option<usize>,
) -> Result<(Vec<f32>, usize, usize)> {
    // Similar to Qwen but:
    // - Normalization: ImageNet (mean=0.5, std=0.5) per DeepSeek code
    // - Patch size: 14
    // - Temporal patch size: 1 (images only, no video in Exp)
    // - Downsample ratio: 3 (handled by aligner on GPU, not CPU)
    // - Grid unit: patch_size (no spatial_merge_size on CPU side)
}
```

### 7. MODEL.toml for Vision Variant

**Directory**: `kernels/gb10/deepseek-v4-flash-vision/nvfp4/`

Create `MODEL.toml` (extend from text-only):

```toml
[model]
name = "deepseek-v4-flash-vision"
hf_id = "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
params = "284B"
active_params = "13B"
architecture = "MLA + MoE + CSA/HCA + Vision (ViT + Aligner) + mHC"

# ... text config from existing MODEL.toml ...

vision = true
vision_n_layers = 32
vision_dim = 1024
vision_n_heads = 16
vision_inter_dim = 2816
vision_patch_size = 14
vision_rope_theta = 10000.0
vision_downsample_ratio = 3
vision_max_n_token = 384
vision_min_pixels = 147456
vision_max_wh_ratio = 8

# Image token (not pad token like Qwen)
image_pad_token = "<｜deepseek_image｜>"

[[model_types]]
model_type = "deepseek_v4"

# ... sampling, behavior from existing ...
```

### 8. Kernel Targets

Need vision kernels for DeepSeek-V4. Check if existing kernels cover it or need new PTX.

**Directory**: `kernels/gb10/deepseek-v4-flash-vision/nvfp4/`

May reuse MLA kernels from text path. Vision encoder needs:
- ViT attention (bidirectional, 2D RoPE)
- ViT MLP (SwiGLU)
- Aligner (unfold + 2-layer GELU MLP)

---

## Testing Checklist

### Unit Tests
- [ ] Vision config parsing roundtrip
- [ ] Image preprocessing matches HF reference output
- [ ] Vision weight loading finds all tensors
- [ ] Token expansion produces correct pad counts

### Integration Tests
- [ ] Single image chat completion
- [ ] Multi-image chat completion
- [ ] Image + text interleaved
- [ ] Tool calling with images
- [ ] Thinking mode with images

### Performance Gates
- [ ] Cold TTFT < 5s (single image)
- [ ] Coherence: no CJK leakage, valid tool calls
- [ ] Agentic: multi-turn with images works

---

## Dependencies & Blockers

| Task | Depends On |
|------|------------|
| Vision config parsing | None |
| Vision encoder kernels | Vision config parsing |
| Vision weight loading | Vision encoder kernels |
| Image token handling | None |
| Pad expansion | Image token handling |
| Image preprocessing | Vision config |
| MODEL.toml | All above working |
| End-to-end test | MODEL.toml + image build |

---

## Effort Estimate

| Component | Effort |
|-----------|--------|
| Vision config parsing | 1 day |
| Vision encoder (Rust + kernels) | 3-5 days |
| Weight loading | 1-2 days |
| Tokenizer integration | 1 day |
| Preprocessing | 1 day |
| MODEL.toml + kernel build | 1 day |
| Testing & validation | 2-3 days |
| **Total** | **~10-15 days** |

---

## Quick-Start Command (When Ready)

```bash
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /models:/models \
  azeezish/atlas-gb10:latest \
  serve \
    --model-from-path /models/deepseek-v4-flash-vision-exp \
    --kv-cache-dtype fp8 \
    --max-seq-len 32768 \
    --vision-max-pixels 1572864
```

---

## Notes from DeepSeek Reference Implementation

Key files to reference from `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`:
- `inference/vision.py` — ViT + Aligner architecture
- `inference/image_processor.py` — Preprocessing (grid tokens, resize logic)
- `encoding/encoding_dsv4.py` — Prompt format, `<｜deepseek_image｜>` token
- `config.json` — All vision_config parameters

The reference uses:
- `torch.nn.MultiheadAttention` (bidirectional) for ViT
- `F.unfold` + `nn.Linear` for Aligner (downsample_ratio=3)
- SigLIP normalization (mean=0.5, std=0.5)
- Patch size 14, temporal_patch_size=1 (images only)

---

## Risk Mitigation

1. **Kernel availability**: Vision kernels may need new PTX compilation. Start kernel work early.
2. **Token alignment**: DeepSeek uses `<｜deepseek_image｜>` not `<|image_pad|>`. Ensure template + tokenizer + expansion all agree.
3. **Vision config location**: HF config has vision params at top level (not nested `vision_config`). Parser must handle this.
4. **Weight prefixes**: DeepSeek uses `model.visual.*` for vision weights. Verify against actual checkpoint.

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-10-15 | Create separate kernel target `deepseek-v4-flash-vision` | Vision adds encoder weights; keeps text kernels reusable |
| 2026-10-15 | Use `model.visual` weight prefix | Matches HF checkpoint structure |
| 2026-10-15 | New vision encoder module (not extend Qwen) | DeepSeek ViT is bidirectional + 2D RoPE; Qwen is causal + different merger |

---

## DeepSeek-V4.1-Flash Assessment (Separate Model Family)

**DeepSeek-V4.1-Flash is NOT a variant of V4-Flash — it is a completely new architecture** that would require a separate, substantial implementation effort.

### Key Architectural Differences from V4-Flash

| Aspect | V4-Flash | V4.1-Flash |
|--------|----------|------------|
| **Model Type** | `deepseek_v4` | `deepseek_v41` (text: `deepseek_v41_text`, vision: `deepseek_v41_vision`) |
| **Architecture** | Standard decoder-only | **Causal Encoder-Decoder (CED)**: 20-layer encoder + 20-layer decoder |
| **Active Params** | ~13B | 8B (prefill) / 16B (decode) |
| **Total Params** | 284B | 552B (+ 196B Engram) |
| **Attention** | CSA/HCA hybrid | **CSA2** (Full/Reindex/Reuse modes, cross-layer KV reuse) |
| **Indexer** | Single-level | **Hierarchical Sparse Indexer** (multi-level candidate pooling) |
| **mHC** | Standard | **Single-Pass mHC** (kernel fusion) |
| **Memory** | None | **Engram** (conditional memory, 196B params, sparsely accessed) |
| **Speculative** | MTP | **DSpark** (separate training, semi-autoregressive) |
| **KV Cache** | FP8/NVFP4 | **FP4 global KV** + SWA Bounded Replay |
| **Vision** | ViT + Aligner (downsample=3) | **DeepSeek-ViT** (2D-RoPE, pixel-unshuffle 3×3, linear patch embed) |
| **MoE Load Balancing** | Standard | **Modality-specific** (separate biases for text vs image) |
| **Config Structure** | Flat `config.json` | **Nested**: `text_config` + `vision_config` |
| **Quantization** | Block [128,128] | Block [32,32], expert_dtype=fp4 |

### New Components Required for V4.1-Flash

1. **CED Architecture**: Encoder/decoder split with decoder global KV projected from encoder outputs
2. **CSA2 Attention**: Three modes (Full/Reindex/Reuse) with cross-layer KV & index sharing
3. **Hierarchical Sparse Indexer**: Candidate pool → rescoring pipeline
4. **Single-Pass mHC**: Fused residual mixing kernel
5. **Engram Module**: 196B param conditional memory with n-gram embeddings
6. **FP4 KV Cache**: Training-time FP4 quantization for global KV
7. **SWA Bounded Replay**: Approximate SWA state reconstruction from last `n_win` tokens
8. **DeepSeek-ViT**: New vision encoder (2D-RoPE, linear patch embed, pixel-unshuffle)
9. **Multimodal MoE Load Balancing**: Separate expert correction biases per modality
10. **DSpark Speculative Decoding**: Separate drafter training, confidence-scheduled verification

### Effort Estimate for V4.1-Flash

| Component | Effort |
|-----------|--------|
| Config parsing (nested text/vision) | 2-3 days |
| CED encoder/decoder architecture | 2-3 weeks |
| CSA2 + Hierarchical Indexer | 3-4 weeks |
| Single-Pass mHC | 1-2 weeks |
| Engram module | 2-3 weeks |
| FP4 KV cache support | 1-2 weeks |
| SWA Bounded Replay | 1 week |
| DeepSeek-ViT vision encoder | 2-3 weeks |
| Multimodal MoE Load Balancing | 1 week |
| DSpark integration | 1-2 weeks |
| Kernel development (PTX) | 4-6 weeks |
| Testing & validation | 2-3 weeks |
| **Total** | **~20-30 weeks** (5-7 months) |

### Recommendation

**Do NOT attempt V4.1-Flash as part of the V4-Flash-Vision-Exp effort.** They share only the name "DeepSeek-V4" but are architecturally distinct. V4.1-Flash is a new model family requiring its own:
- Model type registration (`deepseek_v41`)
- Kernel target directory (`kernels/gb10/deepseek-v4.1-flash/`)
- Weight loader implementation
- Template/tokenizer support
- Full validation pipeline

The V4-Flash-Vision-Exp plan remains valid and should proceed independently. V4.1-Flash can be tracked as a separate epic.

### Note from DeepSeek
> "V4-Flash & V4-Flash-Vision-Exp are retired. For compatibility, deepseek-v4-flash and deepseek-v4-flash-vision-exp temporarily route to V4.1-Flash."

This means the **HF model IDs** `deepseek-ai/DeepSeek-V4-Flash` and `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` now serve V4.1-Flash weights. However, the **original V4-Flash checkpoints** (e.g., `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8`, `nvidia/DeepSeek-V4-Flash-NVFP4`) still exist and are what Atlas currently targets. The Atlas implementation should continue targeting the original V4-Flash architecture.

If Atlas needs to support the **new** V4.1-Flash API-compatible checkpoints, that is a separate model integration.