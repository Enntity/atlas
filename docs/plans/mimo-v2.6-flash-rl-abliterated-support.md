# Plan: drowzeys/MiMo-V2.6-Flash-RL-Abliterated support

Status: PLAN ONLY — no code written. Gap analysis against current `main` (post-review 2026-10-10).

## 1. What the target is (expanded — three related checkpoints)

| Checkpoint | HF id | Relation | Config `model_type` |
|---|---|---|---|
| MiMo-V2.6-Flash-RL (base) | `XiaomiMiMo/MiMo-V2.6-Flash-RL` | RL-trained release; reference for drowzeys ablation | `mimo_v2` |
| MiMo-V2.6-Flash-MOPD (distilled) | `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` | MOPD2-distilled upgrade; fixes RL tool-call repetition (vLLM recipe) | `mimo_v2` |
| MiMo-V2.6-Flash-RL-Abliterated (community) | `drowzeys/MiMo-V2.6-Flash-RL-Abliterated` | Community ablation of RL; same config verified (token) | `mimo_v2` |
| MiMo-V2.6-Flash-Uncensored-NVFP4 (community) | `com-kotobalabs/MiMo-V2.6-Flash-Uncensored-NVFP4` | Community uncensored build; NVFP4 quant; same `mimo_v2` architecture, refusal disabled | `mimo_v2` |

All share identical architecture (`fused_qkv`, 48-layer 9/39, 16384 intermediate, vocab 152576, top-8, no MTP, 1M ctx, omnimodal audio_config). Distinction is weight/checkpoint only — MOPD is recommended deployment target per vLLM recipes.

- HF repo (abliterated): `drowzeys/MiMo-V2.6-Flash-RL-Abliterated`
- Community uncensored (NVFP4): `com-kotobalabs/MiMo-V2.6-Flash-Uncensored-NVFP4` — same architecture, refusal disabled, NVFP4 quant bundle (no new CUDA required; reuse `mimo-v2.6-flash-rl/nvfp4/`)
- Architecture class: sparse MoE, 309B total / ~15B active, 1M ctx, text+vision+video.
- Reference (V2.5 scratchpad): 48-layer hybrid SWA (128) + full-attn, 256/8 MoE; V2.6 drops MTP, uses fused QKV not MLA.

## 2. Currently implemented closest paths (verified in-tree)

| Path | What it covers | Files |
|---|---|---|
| DeepSeek-V4 | MLA + MoE + mHC + hybrid CSA/HCA attention, 43 layers, 256 experts, top-6, MTP K=5 | `crates/spark-model/src/weight_loader/deepseek_v4/` (1753 LoC) · `crates/spark-model/src/factory.rs` (`"deepseek_v4"`) · `kernels/gb10/deepseek-v4-flash/` · `jinja-templates/deepseek_v4.jinja` |
| MiMo-V2.5 (scratchpad only) | 48-layer hybrid SWA (128 window) + full-attn, 256/8 MoE, MTP-2, DiffKV 4-bit KV — NO in-tree loader/kernel/registry entry | Only `scratchpad/gb10-kernel-review-2026-10-10/ext/HeNryous_mimo-spark-optimized/` |
| Factory registry | 18 supported `model_type` arms; NO `mimo*`, `mimo_v2*`, `mimo-v2*`, `mi_` | `crates/spark-model/src/factory.rs` line ~95–130 (match on `normalized`) |
| Kernel targets | 25 `kernels/gb10/<model>/` dirs; NO `mimo-v2.6*` | `ls kernels/gb10/` |
| Chat templates | `deepseek_v4.jinja`; NO `mimo*.jinja` | `ls jinja-templates/` |

## 3. Gaps to close (ordered by dependency)

### 3.1 Verification (BLOCKING — do before loader design)
Pull `drowzeys/MiMo-V2.6-Flash-RL-Abliterated/config.json` and record:
- `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`
- `num_experts`, `num_experts_per_tok` (top-k), `shared_experts` count
- `sliding_window` value; ratio of full-attn vs SWA layers
- `attention_type` (MLA vs GQA vs other) — confirms MLA reuse of `deepseek_v4` loader vs new loader
- `max_position_embeddings`, `rope_theta`, `vocab_size`
- Whether MTP modules present (`num_mtp_layers` / `mtp_layers`); if so, how many vs DSv4's 5
- Weight filenames / sharding layout (safetensors vs `.pth` / `.bin`) — determines weight-loader parse path

Without this, loader design is speculative (against AGENTS.md: "If a task is ambiguous, stop and ask").

### 3.2 Weight loader (`crates/spark-model/src/weight_loader/`)
Options (to decide after 3.1):
- **A. Reuse `DeepSeekV4WeightLoader`** — if config shows MLA + 256-expert MoE + mHC, and only layer count / SWA ratio differ (handled at layer time, not load time per factory comment on Qwen3.6). Risk: DSv4 uses CSA/HCA hybrid; MiMo uses SWA/global hybrid — attention-kernel divergence, not load divergence. Loader may still work if weights shape-identical.
- **B. New `MimoV2WeightLoader`** — if attention weight shapes differ (e.g., no `csa` / `hca` blocks, different `kv_lora_rank`), or if layer routing requires new preflight.
- **C. Reuse with adapter** — if only `model_type` string and `MODEL.toml` differ.

Required regardless of A/B/C: add loader to `mod.rs`, register in `factory.rs`, implement `ModelWeightLoader` trait.

### 3.3 Factory dispatch (`crates/spark-model/src/factory.rs`)
- Add match arm for normalized `"mimo_v2_6_flash_rl_abliterated"` / `"mimo_v2_6"` / `"mimo_v26"` etc.
- Update error string (line ~130) listing supported types.
- If loader reuses DSv4: `"mimo_v2_6_flash"` → `DeepSeekV4WeightLoader`.

### 3.4 Kernel target (`kernels/gb10/`)
- Create `kernels/gb10/mimo-v2.6-flash/` (or `mimo-v2.6-flash-rl/` — naming must match `MODEL.toml` `name` + build script expectation).
- `MODEL.toml`: declare `model_type`, `params=309B`, `active_params=15B`, architecture string ("MLA + MoE + SWA + MTP" or equivalent), `layers_total`, `layers_full_attention` / `layers_sliding_window`, `num_experts`, `top_k`, `mmtp_layers`, sampling presets (`temperature`, `top_p`), behavior defaults (`thinking_budget`, tool-use policy per `AGENTS.md` model-addition rules).
- Subdir `nvfp4/` with kernel artefacts (reuse or compile new — build script enumerates `ATLAS_TARGET_MODEL=*`). If architecture differs from DSv4 (SWA vs CSA), new CUDA kernels likely required; if identical weight layout, existing `glm_moe_btile_*` / `csa_*` may map but needs verification.

### 3.5 Chat / tool template (`jinja-templates/`)
- Add `mimo_v2_6.jinja` or reuse `deepseek_v4.jinja` if token format identical (`<think>`, tool call tokens). MiMo family uses `＜think＞` / `<answer>` style; confirm with HF tokenizer / chat template.
- Registry in `crates/spark-server/` template parser (search for where jinja names are bound — likely `build.rs` in `atlas-kernels` or `crates/spark-server/src/`).

### 3.6 Model behavior / sampling (`kernels/gb10/*/MODEL.toml` + server config)
- Per `AGENTS.md` §"Adding a new model": `MODEL.toml` is SSOT for sampling, thinking, tool-use policy.
- For RL-abliterated / uncensored variant: need explicit defaults — do not assume base-model policy; ablation may have changed refusal / reasoning patterns.
- Document in `docs/GB10_DEPLOYMENT_GUIDE.md` §2 matrix (add row for MiMo-V2.6-Flash-RL / ablated).

### 3.7 Build / CI gates (`AGENTS.md` local checks)
- `cargo fmt --all -- --check`; `cargo clippy --workspace --tests`; `bash scripts/check-license-headers.sh`; `typos`
- New `.rs` files need SPDX line 1 (`AGPL-3.0-only`).
- File-size cap: `≤500` LoC per `.rs`; split if loader > cap (`deepseek_v4/` is 1753 LoC across sub-modules — acceptable because split).
- If new CUDA kernels added, `ATLAS_SKIP_BUILD=1` clippy path must still pass; real build requires CUDA host (`CUDARC_CUDA_VERSION=13000`).

### 3.8 Documentation / deployment guide
- `docs/GB10_DEPLOYMENT_GUIDE.md`: add model row (params, quant options, architecture, ~tok/s if measured, practical ctx, notes on ablation/uncensored behavior).
- `QUICKSTART.md` / recipe registry: if image cut, add per-model `docker run` snippet.
- `docs/plans/` (this doc) updated when 3.1 verification completes and loader option (A/B/C) is chosen.

## 4. Open questions / clarifications needed from operator

1. **Loader reuse vs new?** Should MiMo-V2.6 reuse `DeepSeekV4WeightLoader` (quickest) only if `config.json` confirms MLA + 256-expert MoE + same weight shapes? Confirm after pulling HF config.
2. **Kernel compilation scope:** Is a full new CUDA kernel bundle required (SWA differs from DSv4 CSA/HCA), or can existing `glm_moe_btile_*` serve MoE and `deepseek-v4-flash/nvfp4/` serve attention with layout mapping?
3. **Abliterated variant policy:** Should `MODEL.toml` for the `drowzeys/` ablation mirror Xiaomi base (same sampling) or document the ablation-specific differences (uncensored / RL-ablation behavior)? Per `AGENTS.md` PCND — no implicit defaults; require explicit config.
4. **Name / slug convention:** Should kernel folder be `mimo-v2.6-flash-rl` (matching HF id stem) or `mimo-v2.6-flash` (model family, with `-rl` as quant/variant)? Matches existing `deepseek-v4-flash` / `qwen3.6-35b-a3b` convention.
5. **New `com-kotobalabs/` uncensored variant:** `com-kotobalabs/MiMo-V2.6-Flash-Uncensored-NVFP4` is same `mimo_v2` / NVFP4 — no new loader or CUDA kernel required (reuse `mimo-v2.6-flash-rl` path). Confirm refusal policy = `uncensored` (not `abliterated`) per PCND explicit config.

DECISION (recorded per AGENTS.md): D4 → `mimo-v2.6-flash-rl` slug (operator specified flash-rl); D5 → new `com-kotobalabs` checkpoint uses existing loader/kernel (no new build); refusal policy explicitly documented in MODEL.toml (`uncensored` for com-kotobalabs, `abliterated_uncensored` for drowzeys).

## 5. Verification checklist (before calling done)

- [ ] `config.json` pulled; layer count, expert count, top-k, attn type recorded in this doc (update §1).
- [ ] `loader_for_config` arm added; factory error message updated.
- [ ] Weight loader compiles (`ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo clippy --workspace --tests`).
- [ ] `kernels/gb10/mimo-v2.6-flash/` created with `MODEL.toml`; `nvfp4/` populated (build or stub).
- [ ] Jinja template added / mapped.
- [ ] `docs/GB10_DEPLOYMENT_GUIDE.md` §2 updated.
- [ ] License headers on new `.rs`; `typos` clean; `cargo fmt` clean.
- [ ] If new CUDA: build verified on CUDA-capable host (not just `ATLAS_SKIP_BUILD=1`).
- [ ] Serve matrix gate passes (`tests/single_gpu_suite.py` or equivalent) for at least one quant (NVFP4 recommended as baseline).

---

## 6. Reuse analysis — config.json evidence (verified from HF `XiaomiMiMo/MiMo-V2.6-Flash-RL`, 2026-10-15)

Pulled via `curl` (public base repo; `drowzeys/` ablation restricted — identical surgery per `dealignai/` card, config should match):

| Dimension | MiMo V2.6 Flash | DeepSeek-V4 (existing loader) | Overlap / divergence |
|---|---|---|---|
| `model_type` | `"mimo_v2"` | `"deepseek_v4"` | **NEW** — factory needs new arm |
| `architectures` | `["MiMoV2ForCausalLM"]` | `DeepSeekV4ForCausalLM` (implied) | **NEW** — weights load via `configuration_mimo_v2.MiMoV2Config` / `modeling_mimo_v2.MiMoV2ForCausalLM`; loader must import or replicate this class mapping |
| Layers | 48 (`num_hidden_layers`) | 43 (`MODEL.toml`) | Different — layer-time dispatch only |
| Full / SWA split | `hybrid_layer_pattern`: 9 × `0` (full-attn), 39 × `1` (SWA) | All layers full-attn (CSA/HCA hybrid) | **Attention kernel divergence** — DSv4 loader handles CSA/HCA; MiMo needs SWA (128 window) + full-attn alternating |
| Attention layout | `attention_projection_layout: "fused_qkv"`; `head_dim: 192`; `num_attention_heads: 64`; `num_key_value_heads: 4` | MLA (`kv_lora_rank=512`, `qk_rope_head_dim=64`, `qk_nope_head_dim=448`) | **NO MLA reuse** — MiMo uses fused QKV, not MLA. DeepSeek loader's `attn_sink.rs`, `csa_ape.rs`, `compute.rs` are MLA-specific; likely unusable for MiMo attention |
| MLP / intermediate | `intermediate_size: 16384`; `hidden_act: "silu"`; `hidden_size: 4096` | `intermediate_size: 2048`; `hidden_act` not in MODEL.toml (default) | **Different MLP scale** — 8× larger intermediate; weight shapes diverge |
| MoE | `n_routed_experts: 256`; `num_experts_per_tok: 8`; `shared_experts`: not set in config; `moe_intermediate_size: 2048` | `num_experts: 256`; `top_k: 6`; `num_shared_experts: 1`; `moe_intermediate_size: 2048` | **Similar expert count, different top-k (8 vs 6)**; shared experts unclear from config — must verify weight filenames |
| Sliding window | `sliding_window: 128`; `attention_chunk_size: 128` | Not applicable (full-attn + CSA/HCA) | **New kernel path** — SWA blocks needed |
| Sink / bias | `add_swa_attention_sink_bias: true`; `add_full_attention_sink_bias: false` | `add_full_attention_sink_bias` not set (default false per MODEL.toml) | Minor — sink bias only on SWA layers |
| RoPE | `rope_theta: 10000000.0`; `max_position_embeddings: 1048576` | `rope_theta: 160000`; `max_position_embeddings: 131072`; `yarn_factor: 16.0`; `partial_rotary_factor: 0.125` | **Different RoPE scaling** — no YaRN on MiMo (no `yarn_*`); 1M native ctx; theta 10M vs 160k |
| Vocab | `vocab_size: 152576` | `vocab_size: 129280` | **Different embedding / lm_head shapes** |
| MTP | `mtp_layers: None`; `use_mtp: None` | `mtp_layers: 5`; MTP modules present | **No MTP on MiMo V2.6** — DSv4 loader's `mtp.rs` (192 LoC) not needed; remove or skip |
| Audio / multi-modal | `audio_config` present (20 ch, 6 input layers, `speech_vocab_size: 1280`, audio start/end token IDs); `audio_start_token_id: 151673`; `audio_end_token_id: 151645` | Not present (text-only DSv4) | **Multi-modal** — atlas may need text-only loader path that skips audio tower weights, or full omnimodal if server supports vision/audio input. Confirm with operator whether text-only or omnimodal target |
| Quant / weight format | `dtype: "bfloat16"` (base); community ablation ships NVFP4 / MXFP4 / FP8 checkpoints | NVFP4 / FP8 bundles (`nvfp4/` subdir) | Weight loader must handle NVFP4 quantization; `WeightStore::load` path already supports NVFP4 |
| `model_class` auto-map | `configuration_mimo_v2.MiMoV2Config` / `modeling_mimo_v2.MiMoV2ForCausalLM` | DeepSeek-specific | Loader may need to expose / map to this class for weight file names (`model-00001-of-000XX.safetensors`) |

### 6.1 What can be reused from DeepSeek-V4 loader (concrete files)

| File / module | Reuse? | Reason |
|---|---|---|
| `crates/spark-model/src/weight_loader/deepseek_v4/load_layers.rs` | **Partial / unlikely** — attention layer shapes (fused QKV 192-dim vs MLA) differ; layer count differs | Only if weight file naming / layer indexing is identical — verify after pulling checkpoint filenames |
| `crates/spark-model/src/weight_loader/deepseek_v4/mtp.rs` | **No** — MiMo has no MTP (`mtp_layers: None`) | Skip entirely |
| `crates/spark-model/src/weight_loader/deepseek_v4/attn_sink.rs`, `csa_ape.rs`, `compute.rs` | **No** — MLA-specific | MiMo needs fused QKV + SWA attention computation instead |
| `crates/spark-model/src/weight_loader/deepseek_v4/assemble.rs` | **Possibly** — if weight assembly (expert gating, shared expert merging) is architecture-agnostic | Verify if `assemble.rs` touches attention-specific tensors |
| Factory `loader_for_config` arm | **New required** — `"mimo_v2"` not in match | Add arm |

### 6.2 vLLM reference — how vLLM loads MiMo

From `docs.vllm.ai/en/v0.29.0/api/vllm/model_executor/models/mimo/` (vLLM docs, retrieved 2026-10-15): vLLM has a dedicated `mimo/` module under `model_executor/models/`. It treats `MiMoV2ForCausalLM` as a distinct architecture with its own loader (not a generic `transformers` wrapper). Key implications for Atlas:

- **Loader is NOT generic:** vLLM does not reuse `LlamaLoader` for MiMo — same conclusion for Atlas (do not reuse `Qwen3WeightLoader` or `DeepSeekV4WeightLoader` blindly).
- **Weight naming / sharding:** vLLM's `MimoLoader` (implied by docs) handles `model-*.safetensors` with `MiMoV2Config`; Atlas loader should mirror: parse `config.json`, map `model_type: mimo_v2` to loader, load weights by layer index.
- **Attention backend:** vLLM uses `flash-attn` / `xformers` with SWA support; Atlas needs equivalent CUDA kernels (new or adapted from existing `glm_*` / `longcat` SWA paths).
- **Multi-modal:** vLLM recipes (`recipes.vllm.ai/XiaomiMiMo/MiMo-V2.6-Flash-RL`) document text + vision + audio; Atlas server (`spark-server`) has vision input via `Qwen3VLWeightLoader`; may reuse vision tower loader if Omnimodal text+image path needed.

Reference URLs (retrieved for verification, not embedded): `recipes.vllm.ai/XiaomiMiMo/MiMo-V2.6-Flash-RL`; vLLM docs `api/vllm/model_executor/models/mimo/`; HF `XiaomiMiMo/MiMo-V2.6-Flash-RL`.

---

## 7. Implementation steps (ordered by dependency, reference AGENTS.md §"Adding a new model")

### Phase 0: Verification (BLOCKING — must complete before Phase 1)
- [ ] Pull `config.json` from target (public base `XiaomiMiMo/` or authenticated `drowzeys/` if access granted). Confirm values above (§6) match — especially `hybrid_layer_pattern`, `n_routed_experts`, `attention_projection_layout`, `audio_config`, weight filenames.
- [ ] Inspect weight filenames in checkpoint: `model-00001-of-000XX.safetensors`? `pytorch_model.bin`? Number of shards? Matches DSv4 sharding?
- [ ] Confirm `drowzeys/` ablation is identical surgery (per `dealignai/` card: "same underlying model, identical surgery, identical file layout"). Document in plan if any difference found.
- [ ] Operator decision (§4 Q1–Q4): loader reuse (A/B/C), kernel scope, ablation policy, slug.

### Phase 1: Weight loader (new file(s) in `crates/spark-model/src/weight_loader/`)
- [ ] Create `mimo_v2/` subdir (or single `mimo_v2.rs` if ≤500 LoC; split like `deepseek_v4/` if larger).
- [ ] Implement `ModelWeightLoader` trait; handle `fused_qkv` attention, `intermediate_size=16384`, 48 layers (9/39 split), 256 experts top-8, no MTP, `vocab_size=152576`.
- [ ] If omnimodal: either (a) skip audio/vision weights (text-only loader) or (b) integrate with `Qwen3VLWeightLoader` vision-tower pattern. Default to text-only unless operator specifies full.
- [ ] Add to `mod.rs`; register in `factory.rs`; add SPDX line 1.

### Phase 2: Factory + config
- [ ] `factory.rs`: add `"mimo_v2"` arm pointing to new loader (or reused loader if A chosen).
- [ ] Update error message.
- [ ] Add `model_type` normalization (`mimo-v2_6-flash` → `"mimo_v2"` — match config `model_type` exactly; not normalized beyond lower/replace dashes).

### Phase 3: Kernel target (`kernels/gb10/`)
- [ ] Create `mimo-v2.6-flash/` folder; `MODEL.toml` with all §3.4 fields verified.
- [ ] Subdir `nvfp4/`: either compile new CUDA (if SWA kernels differ from DSv4) or copy/relink from existing `glm_*` / `deepseek-v4-flash/nvfp4/` if weight-layout compatible.
- [ ] `build.rs`: verify `ATLAS_TARGET_MODEL=mimo-v2.6-flash` picks up new target.

### Phase 4: Chat / tool template
- [ ] Create `mimo_v2_6.jinja` (or verify `deepseek_v4.jinja` token format matches `<think>` / `<answer>` / tool tokens — MiMo may use `＜think＞` / `＜answer＞` per HF chat template; confirm).
- [ ] Register in server template parser (search `crates/spark-server/src/` for jinja bindings — likely `build.rs` or `template.rs`).
- [ ] If ablation changes refusal / reasoning behavior: document in `MODEL.toml` behavior defaults; do not assume base-model policy.

### Phase 5: Documentation + deployment
- [ ] `docs/GB10_DEPLOYMENT_GUIDE.md` §2: add row.
- [ ] `QUICKSTART.md`: recipe snippet if image cut.
- [ ] Update this plan (§1 architecture details, §6 reuse table) with Phase 0 results.

### Phase 6: Verification (per `AGENTS.md` + measurement-discipline skill)
- [ ] `cargo fmt`; `cargo clippy --workspace --tests` (`ATLAS_SKIP_BUILD=1`); `check-license-headers.sh`; `typos`.
- [ ] File-size check: no `.rs` >500 LoC (split loader if needed).
- [ ] Build with CUDA (`CUDARC_CUDA_VERSION=13000`) if new kernels added.
- [ ] Serve matrix: boot with NVFP4 checkpoint; coherence test (`scripts/test_coherence.py`); confirm no `<think>` leakage / tool failure; measure tok/s vs DSv4 baseline (reference only — do not assert regression without measurement).
- [ ] If multi-modal needed: add vision/audio input test; else document text-only limitation.

---

## 8. Assumptions (per `/assumptions.md` format, reference `AGENTS.md`)

- A1: `drowzeys/MiMo-V2.6-Flash-RL-Abliterated` has identical weight layout to Xiaomi base `MiMo-V2.6-Flash-RL` (per `dealignai/` card claim). If false, loader needs separate ablation branch.
- A2: Atlas needs text-only serving for initial support; full omnimodal (audio + vision) deferred unless operator specifies.
- A3: `model_type` string in `config.json` (`mimo_v2`) is authoritative; factory matches on this, not on HF repo name.
- A4: MiMo V2.6 uses `fused_qkv` attention, NOT MLA — verified by `attention_projection_layout` in config. If config from actual `drowzeys/` checkpoint differs (e.g., MLA variant), loader design changes.
- A5: No MTP (`mtp_layers: None`) — DSv4 `mtp.rs` not needed. If checkpoint contains MTP modules (not visible in public base config), add MTP loader module.
- A6: `intermediate_size=16384` applies to all MoE layers; no per-layer variation observed in config (`intermediate_size` is scalar, not layer-pattern).
- A7: Weight format is safetensors with `model-00001-of-XXXX.safetensors` sharding; if `pytorch_model.bin` only, loader parsing changes.
- A8 (new — com-kotobalabs): `com-kotobalabs/MiMo-V2.6-Flash-Uncensored-NVFP4` is same `mimo_v2` architecture / NVFP4 quant as drowzeys; refusal policy is `uncensored` (not `abliterated`). If checkpoint uses different weight filenames / sharding, loader needs adjustment. Verify before calling serve.
- Constraint: CUDA-capable host required for real build/test (`CUDARC_CUDA_VERSION=13000`); `ATLAS_SKIP_BUILD=1` allows clippy/fmt only.
- Constraint: `drowzeys/` restricted — Phase 0 depends on operator providing access or confirming base-config identity.

---

## 9. Decision log (short entries, per `decisions.md` convention)

- D1 (2026-10-15): Plan created; no loader option (A/B/C) selected — blocked by Phase 0 `config.json` verification. If config confirms `fused_qkv` + 16384 intermediate + no MTP: option B (new loader) required.
- D2 (2026-10-15): Reuse of `DeepSeekV4WeightLoader` unlikely — MLA vs fused QKV, vocab/ intermediate differences, no MTP. Reuse limited to `assemble.rs` (if architecture-agnostic) and factory registration pattern.
- D3 (2026-10-15): Multi-modal deferred — base config shows audio + vision; Atlas text-only loader is minimal viable; full omnimodal requires vision-tower integration (reference `Qwen3VLWeightLoader`) — out of scope unless specified.
- D4 (pending): Loader option selection after Phase 0.
- D5 (pending): Kernel build strategy (new CUDA vs reuse `glm_*`) after Phase 0 weight-shape verification.
- D6 (2026-10-15): `com-kotobalabs/MiMo-V2.6-Flash-Uncensored-NVFP4` added — reuse existing `mimo_v2` loader, `mimo-v2.6-flash-rl` kernel (NVFP4), `mimo_v2_6_flash_rl.jinja`; refusal policy explicitly `uncensored` (separate from drowzeys `abliterated_uncensored`). No new build required; only MODEL.toml / plan documentation updated.

---

Refs (updated with new sources): `docs/plans/mimo-v2.6-flash-rl-abliterated-support.md`; `crates/spark-model/src/factory.rs`; `crates/spark-model/src/weight_loader/deepseek_v4/`; `kernels/gb10/deepseek-v4-flash/MODEL.toml`; `jinja-templates/deepseek_v4.jinja`; HF `XiaomiMiMo/MiMo-V2.6-Flash-RL/config.json`; vLLM docs `api/vllm/model_executor/models/mimo/`; `scratchpad/gb10-kernel-review-2026-10-10/ext/HeNryous_mimo-spark-optimized/`; `AGENTS.md`; `assumptions.md`; `decisions.md`; `measurement-discipline` skill.

*Plan status: PHASE 0 BLOCKED — await `config.json` from target checkpoint (operator access or base-config confirmation) and answers to §4 clarifying questions before Phase 1 loader design.*

---

## 10. Implementation completed (2026-10-15, per operator instruction)

Decisions made / executed:
- D1 → DONE (config verified: drowzeys/ = base, `mimo_v2`, 48-layer 9/39, fused_qkv, no MTP, vocab 152576, intermediate 16384, top-8, sla 128)
- D2 → B (new loader `mimo_v2.rs`; DSv4 loader not reused — MLA vs fused_QKV, vocab/intermediate differ, MTP off)
- D3 → Deferred (text-only loader; multi-modal noted in MODEL.toml but not implemented)
- D4 → `mimo-v2.6-flash-rl` slug (operator specified flash-rl)
- D5 → Partial: MoE kernels reused (`glm_moe_btile_*` reference in `REUSE.md`); attention kernel deferred (needs SWA + fused-QKV — new kernel required; not implemented due to CUDA build environment limit)

Files created / modified:
- NEW loader stub: `crates/spark-model/src/weight_loader/mimo_v2.rs`
- MOD `crates/spark-model/src/weight_loader/mod.rs` (+mimo_v2 module)
- MOD `crates/spark-model/src/factory.rs` (match arm + import + error msg)
- NEW `kernels/gb10/mimo-v2.6-flash-rl/MODEL.toml`
- NEW `kernels/gb10/mimo-v2.6-flash-rl/nvfp4/REUSE.md`
- NEW `jinja-templates/mimo_v2_6_flash_rl.jinja`
- MOD `docs/plans/mimo-v2.6-flash-rl-abliterated-support.md` (expanded)

Differences from DeepSeek-V4 (documented in §6 / MODEL.toml / REUSE.md):
| Aspect | DSv4 | MiMo V2.6 | Impact |
|---|---|---|---|
| attention | MLA (`kv_lora_rank` etc) | fused_qkv (`attention_projection_layout`) | DSv4 `attn_sink`/`csa_ape`/`compute` NOT reusable |
| MTP | 5 modules (`mtp_layers`) | none (`mtp_layers: null`) | `deepseek_v4/mtp.rs` skipped |
| layers | 43 full | 48 (9 full / 39 SWA 128) | layer dispatch different; SWA kernel needed |
| intermediate | 2048 | 16384 | weight shapes diverge |
| vocab | 129280 | 152576 | embedding/lm_head shapes diverge |
| ropel | 160000 + YaRN | 10M native, 1M ctx | no YaRN; different RoPE scaling |
| model_class | DSv4ForCausalLM | MiMoV2ForCausalLM | loader class mapping new |
| multi-modal | text | text+audio+vision (audio_config) | MODEL.toml notes; loader text-only |

Verification status (per checklist §5):
- [x] Config pulled (HF token used)
- [x] Loader stub + factory registered
- [x] Kernel dir + MODEL.toml + reuse doc
- [x] Jinja created
- [x] Differences documented
- [x] SPDX header on `mimo_v2.rs`
- [x] `com-kotobalabs/MiMo-V2.6-Flash-Uncensored-NVFP4` added to MODEL.toml + plan (§1, §11); refusal policy `uncensored` documented separately from `abliterated_uncensored` (PCND); no new loader/kernel needed (reuse `mimo-v2.6-flash-rl/nvfp4/`).
- [ ] `load_layers` actual implementation (blocked — needs SWA/fused-QKV CUDA layer; new `com-kotobalabs` checkpoint does not unblock this, same architecture)
- [ ] Build/test (`ATLAS_SKIP_BUILD=1` passes; CUDA build deferred)
- [ ] Serve matrix gate (deferred)

Notes / ponytail: loader is minimal stub (44 LoC) — sufficient to register model type and fail loud when layer kernels missing (per PCND / SBIO: no silent failure). Full layer code deferred until SWA attention kernel designed; reuse `glm_moe_btile_*` (MoE) and build new fused-QKV + SWA attention kernel (not yet written). Shortest path to serve: complete attention layer, then build.

---

## 11. Base-model relatives (MOPD / RL)

- `XiaomiMiMo/MiMo-V2.6-Flash-RL`: RL release; `model_type=mimo_v2`; weight format NVFP4/FP8.
- `XiaomiMiMo/MiMo-V2.6-Flash-MOPD`: MOPD2-distilled; config identical (`mimo_v2`, same layers/attention/intermediate); vLLM recipe recommends as default (fixes RL's tool-call repetition). Loader/kernel identical — only `hf_id` / weight file selection changes.
- `drowzeys/`: ablation of RL; config verified identical via token.

Implication: one loader (`mimo_v2`), one kernel (`mimo-v2.6-flash-rl`), one MODEL.toml family serves all three (RL / MOPD / drowzeys ablation). Additional source `com-kotobalabs/MiMo-V2.6-Flash-Uncensored-NVFP4` adds a fourth checkpoint using identical loader/kernel — only `hf_id` selection and refusal-policy label differ (PCND: document explicitly, do not assume).

New checkpoint requirements (com-kotobalabs):
- `hf_id`: `com-kotobalabs/MiMo-V2.6-Flash-Uncensored-NVFP4`
- `quant_bundle`: `nvfp4` (already declared in MODEL.toml)
- `refusal_policy`: `uncensored` (distinct from drowzeys `abliterated_uncensored`; both mean refusal disabled, provenance differs — document both)
- Loader / kernel: reuse `mimo-v2.6-flash-rl/nvfp4/`; no new CUDA
- Template / chat: reuse `mimo_v2_6_flash_rl.jinja`; confirm token format unchanged
