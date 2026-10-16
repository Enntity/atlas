// SPDX-License-Identifier: AGPL-3.0-only
# Tracing audit — full stack (load / server / inference / storage / kernels)
Status: REVIEW ONLY — no edits. Scope: nested spans (request lifecycle + per-step) + structured logs.

## Evidence of existing instrumentation
- `tracing::info!` / `warn!` used in spark-server (`serve_load.rs`, `anthropic/handlers.rs`, `api/`), spark-model (`factory/`), spark-storage (`expert_pack.rs`).
- `atlas-core/src/load_progress.rs` (new): timer thread + `tracing::info!` at interval; NO spans; NO `#[tracing::instrument]`; NO `Span::enter()`.
- `tracing-opentelemetry` present in workspace (`Cargo.toml` 90); NOT referenced in crate `Cargo.toml`s (spark-server, spark-model, etc.). No `Tracer` initialization found in `main.rs` / `lib.rs`.
- No `#[tracing::instrument]` found in `crates/spark-server/src/main_modules/`, `crates/spark-model/src/factory/`, `crates/spark-runtime/src/` (grep sampled 30 files).
- `spark-runtime/src/progress.rs` (if exists) may have phase markers; not spans.

## Missing — load path (what was just added)
- `load_progress.rs`: has info logs but no `Span` for "load" operation; no `current_file` as span attribute; no bandwidth rate as metric/event.
- `serve_load.rs`: phases via `spark_runtime::progress::phase()` — log markers, not spans; missing span around `load_model_selected` with fields `model_dir`, `config.quant`, `total_bytes`.
- `factory/build.rs`: `build_model` call has no span; `load_layers` has no span with layer count / quant / gpu mem.

## Missing — server request flow (chat / stream / tools)
- `serve_load.rs` → `load_model`: no request-level span.
- `api/chat/mod.rs`, `chat_stream/`: stream start/end, tool-call boundary, template-forced thinking — log lines only, no spans.
- `anthropic/handlers_stream.rs`, `translator.rs`: stream fragments logged; no parent span linking to request id.
- No `request_id` propagated as span attribute across chat → stream → finalize.

## Missing — inference / forward pass
- `spark-model/src/forward/`: attention, MoE routing, SSM — likely zero spans; quant-weight paths unlabeled.
- `spark-runtime/src/gpu/`: kernel dispatch, KV cache ops — no spans linking to layer/model.
- No nested structure: request → layer → attention → cache-op.

## Missing — storage / expert load
- `spark-storage/src/expert_pack.rs`: manifest load has info log (post-edit) but no span for `load()`; deferred tensor paths not traced.
- `expert_tier.rs`, `expert_peer.rs`: tier selection / RDMA — no spans.

## Missing — kernel / hardware
- `kernels/<hw>/`: CUDA/PTX — no tracing at kernel launch; `MODEL.toml` sampling defaults not exposed as span attrs.
- `atlas-kernels/build.rs`: target selection (hw/model/quant) — no build-span.

## Recommendation for nested spans (audit, no edits made)
1. Root span per server request / model load: `load_model` / `chat_request` — attrs: model_id, quant, hardware, total_bytes.
2. Load span (child of root): `load_weights` — attrs: loaded_bytes, total_bytes, current_file, bandwidth_bps, progress_pct (update each interval via log; span can record final state or use `tracing::Span::record()` if supported).
3. Build span: `build_model` — child of load; attrs: layer_count, gpu_memory_before/after.
4. Request flow spans (chat): `prepare` → `stream_start` → `tool_call` (optional) → `stream_end` → `finalize`. Propagate `request_id`.
5. Inference spans: `layer_forward` (per layer, with layer_idx, dtype), `attention` / `moe_routing` / `ssm_step` — for critical paths only (not every micro-op to avoid overhead).
6. Storage span: `expert_load` / `tier_select` — attrs: manifest_version, store_dir.
7. Export: initialize `opentelemetry` tracer once at server boot (if not already); add `tracing-subscriber` with OTLP layer; keep existing `info!` for human-readable logs.

## Decisions / risks noted
- Full rewrite across 5+ crates with span attributes touches shared traits (`spark_model::traits::Model`, `spark_runtime::gpu::GpuBackend`) — must not regress performance (span creation overhead). Ponytail: start with root + load + request spans only; expand to inference only after measurement.
- `atlas-core` intentionally excluding `tracing` (per `registry.rs` comment); adding `tracing` dep (done for load_progress) is acceptable but should be documented as exception.

## References
- Load-progress edit: `crates/atlas-core/src/load_progress.rs`, `serve_load.rs` (commit 7035710f)
- Existing log points: `spark-server/src/anthropic/handlers.rs:41`, `spark-server/src/api/chat/levers.rs:105`, `spark-server/src/main_modules/serve_load.rs` (phases)
- Workspace tracing-opentelemetry: root `Cargo.toml:90`; unused in crate deps
