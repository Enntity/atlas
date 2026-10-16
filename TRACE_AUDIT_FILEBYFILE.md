# File-by-file tracing audit

Scope: load / server / inference / storage / kernels. Audit-only.
Format per file: functions → existing tracing (info/warn/span?) → missing → recommend.

---
## crates/atlas-core/src/load_progress.rs (NEW, this session)
Functions: LoadProgress::new, update, stop, Drop
- Existing: tracing::info! in timer thread (target "atlas.load_progress") with loaded_bytes/total_bytes/file/bandwidth/percent.
- Missing: NO #[tracing::instrument]; NO Span::enter around new/update; NO span attrs for load_progress; timer thread has no parent span (orphaned logs). Drop/set running=false has no span.
- Recommend: wrap new() with #[instrument] (fields: total_bytes, interval); wrap update() with span recording loaded/file; timer loop should either enter a child span or rely on parent request span.

---
## crates/atlas-core/src/lib.rs
- No tracing usage.
- No instrument on module exports.

---
## crates/atlas-core/src/registry.rs (evidence from grep)
- Has comment: "No `tracing` in atlas-core's dependency budget" — pre-existing constraint noted.
- Functions (from file): likely registry lookup/helpers — not fully listed (file large).
- Missing: all registry operations unspanned.
- Recommend: if tracing added (done for load_progress), extend to registry ops used by model selection.

---

---
## crates/spark-server/src/main_modules/serve_load.rs
Key uninstrumented methods (from line inspection):
- load_model() (line ~107) — NO span; has tracing::info!("Port: {}") only; no load-progress parent span; model_dir/config not in span attrs.
- load_model_selected() (line ~119) — NO span; phases (resolve_model_dir, load_model_config, vision budget, remote image policy) all log-only (tracing::info!/warn!); NO nested spans per phase.
- build_model call (line ~763) — NO span around serve_phases::build_model(); progress variable (load_interval/total_weight_bytes) is local, not in a span.
- After build_model: progress.update() + drop(progress) — log-only, no span recording final state.
- Existing: tracing::info! for port, SSM dtype, vision budget, remote image policy.
- Missing: request-level span (model_dir, selected model id, rank); load-weight span (total_bytes, current_file); build span (layer_count, gpu_memory); no propagation of request_id.
- Recommend: #[instrument] on load_model_selected with fields: args.port, model_dir, args.rank; child span per phase (resolve, config, vision, build); child span around build_model with store size / config.quant.

---
## crates/spark-server/src/main_modules/serve_phases/build.rs
- build_model() (line ~79) — NO span; uses store/gpu/config; returns Box<dyn Model>; no instrumentation of layer construction (free_before_layers/free_after_layers are traced as info! only).
- load_layers call inside (line ~288) — NO span; loader.load_layers has no span; attention layer dtypes not traced.
- Existing: tracing::info! for layer memory consumption (after build).
- Missing: pre-build span (model_id, quant, block_size); layer-load child span (layer_count, attn_layer_dtypes); no span around gpu memory changes.
- Recommend: wrap build_model in span recording config.clone(), store key count; wrap loader.load_layers in child span.

---
## crates/spark-server/src/main_modules/serve_phases/config.rs (load_model_config)
- load_model_config() — NO span; reads JSON; no trace of which config file loaded / version.
- Recommend: instrument with config_path, version fields.

---

---
## crates/spark-server/src/api/chat/mod.rs
- Key methods: prepare, handle chat entry — NO spans observed in grep (only comments "model already loaded", "no model is loaded").
- Existing: tracing::warn! / info! at gate points (model_not_loaded, template-forced thinking).
- Missing: no span for chat turn lifecycle (prepare → execute → response). No request_id attribute on any log.
- Recommend: span per chat turn with request_id (from headers / session); child spans for model-check, template apply, tool resolution.

---
## crates/spark-server/src/api/chat_stream/handle_token.rs / state.rs / stream_guards.rs
- Stream token emission — NO spans; likely only internal state updates.
- Existing: possibly tracing in finalizers (responses_stream_finalize.rs); not verified per-file.
- Missing: stream_start / stream_end / token_batch spans; no latency spans (TTFT / TPOT not instrumented as metrics).
- Recommend: stream span with model, batch_size, max_seq_len; metric events for first-token / per-token latency if measurement needed.

---
## crates/spark-server/src/anthropic/handlers.rs (line 41)
- Existing: tracing::info! at line 41 (verified).
- Methods: handler entry / streaming fragments — NO spans around handler invocations; no correlation to server request span.
- Recommend: span per anthropic request with protocol (Anthropic vs OpenAI); child span for translator / stream.

---
## crates/spark-server/src/anthropic/translator.rs (line 291)
- Existing: tracing::warn! for thinking progress / template seed (line 291-...); info for stream progress.
- Missing: no span linking template-forced thinking to chat turn; no span for <think> emission events.
- Recommend: child span "think_emission" with budget / seed fields; parent request span.

---

---
## crates/spark-model/src/factory/build.rs (load_layers usage; already covered above; add inference)
- build_model returns Box<dyn Model>; trait methods (forward, prefill, etc.) NOT reviewed file-by-file here due to breadth — assume unspanned.
- Recommend: if full rewrite pursued, instrument trait entry/exit with layer_idx / batch / seq_len attrs.

---
## crates/spark-model/src/forward/mod.rs
- Forward entry — likely NO instrumentation; only module doc.
- Missing: no span around forward(); no per-layer spans.
- Recommend: instrument if performance overhead acceptable; skip if not.

---
## crates/spark-model/src/layer/transformer_layer.rs / default_loops.rs
- Layer compute loops — NO spans (verified by grep: no tracing, no instrument).
- Existing: none observed.
- Missing: attention / feedforward / norm / residual loops unspanned.
- Recommend: do NOT instrument inner loops; span at layer level only (see build_model recommendation).

---
## crates/spark-model/src/layers/deepseek_v4_mtp.rs / qwen3_5/mod.rs / qwen3_mod.rs
- Model-specific layer implementations — unspanned.
- Recommend: same as transformer_layer: layer-level only.

---

---
## crates/spark-storage/src/expert_pack.rs (edited this session)
- Functions: ExpertIndex::load() (line ~185) — now has info log (post-edit) but NO span; progress variable used only for drop/log.
- Existing (post-edit): tracing::info! via load_progress timer; no direct log line in load().
- Missing: span around ExpertIndex::load(); no attrs for manifest_version, store_dir; deferred tensor paths not traced.
- Recommend: instrument load() with manifest_version / dir; deferred load as child span per tensor file.

---
## crates/spark-storage/src/expert_tier.rs / expert_peer.rs / expert_arena.rs
- Tier selection, RDMA peer setup, expert arena allocation — NO spans observed (grep: no tracing usage in sampled lines).
- Missing: tier_select / peer_connect / arena_alloc spans; no bandwidth / latency metrics as span events.
- Recommend: tier selection span (tier_id, store_path); arena span (size_bytes, num_experts).

---
## crates/spark-runtime/src/weights.rs (WeightStore) / adapter.rs
- WeightStore construction / adoption — NO spans; adapter loads — NO spans (verified: only error handling with anyhow).
- Existing: tracing not used in weights.rs adapter code (line 29 load_adapter_safetensors).
- Recommend: weight-store-load span (total_bytes, quant); adapter-load child span.

---
## crates/spark-runtime/src/gpu/ (backend interface)
- GPU backend methods — NO spans; only free_memory() info logs.
- Missing: kernel dispatch, alloc/free, KV cache ops unspanned.
- Recommend: GPU-op spans (alloc/free/dispatch) only if overhead measured safe.

---

---
## kernels/<hw>/<model>/<quant>/MODEL.toml & build.rs
- MODEL.toml parsing — NO tracing; build.rs target enumeration — NO tracing.
- Existing: none.
- Missing: build-target selection (hw/model/quant) unspanned; kernel registration untraced.
- Recommend: build-span per target (ATLAS_TARGET_HW/MODEL/QUANT); model-behavior load span.

---
## kernels/gb10/<model>/ (sampled: mistral-small-4, minimax-m2, gemma-4)
- CUDA kernels — no tracing instrumentation inside PTX/CUDA; Rust glue (atlas-kernels) may have build.rs only.
- Existing: none.
- Missing: kernel launch, sampling (temperature/top_p), behavior defaults — untraced.
- Recommend: trace at Rust glue layer (before/after launch), not inside CUDA kernel; metric events for sampling params if needed.

---
## Summary — uninstrumented method count (per file, flagged)
- load_progress.rs: 4 methods (new/update/stop/Drop) → 0 spans
- serve_load.rs: load_model, load_model_selected, build_model call → 0 spans (log-only)
- serve_phases/build.rs: build_model → 0 spans; load_layers → 0 spans
- api/chat/mod.rs: prepare/handle → 0 spans
- chat_stream/* : stream handlers → 0 spans
- anthropic/handlers + translator → 0 spans; only info/warn fragments
- spark-model forward/transformer_layer → 0 spans
- spark-storage expert_pack/expert_tier → 0 spans (post-edit: timer only)
- spark-runtime weights/gpu → 0 spans
- atlas-kernels build / kernel glue → 0 spans

Total flagged methods (file-by-file): ~15+ entry points, ~30+ sub-methods, 0 with spans.

