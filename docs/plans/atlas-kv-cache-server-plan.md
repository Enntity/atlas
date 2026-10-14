# Plan: Remote KV Cache Server Crate (`atlas-kv-cache-server`)

Status: DESIGN — not implemented. Search performed (`lib.rs` + `crates.io`); reuse targets identified; no custom cache code written.

---

## 1. Clarification (scope — per AGENTS.md / ponytail)

| Dimension | Decision |
|---|---|
| What the server holds? | **KV payload bytes** (not metadata). In-memory pool + optional NVMe offload. All instances connect to it for remote read/write. |
| Reuse existing crates? | **Both pairs:** (a) `spark-runtime/src/kv_cache.rs` (in-memory paged KV blocks) + `spark-storage/src/kv_paging.rs` + `kv_spill.rs` (NVMe offload / spill); (b) `spark-storage/src/cache_peer/` (mod + registry + server_impl) + `rdma_kv_backend.rs` (one-sided RDMA serving). Custom code minimal — wrapper/server around these. |
| Configurable in-memory size? | Yes — `config.rs`: `max_memory_bytes`, `block_size`, `max_blocks`. |
| Optional local storage path? | Yes — `config.rs`: `nvme_path: Option<PathBuf>`; if `Some`, `kv_spill.rs` / `kv_paging.rs` handle offload; if `None`, pure in-mem (no disk). |
| HA? | Not current; design for future (same as previous plan: `ConsensusBackend` trait, not implemented here — server is single-instance data holder; HA via replication/consensus deferred). |
| External access? | gRPC (`tonic`) primary; RDMA (`cache_peer` / `rdma_kv_backend`) optional/feature-gated (`atlas_rdma_verbs`). |

---

## 2. Search Results — Existing Crates Used Instead of Custom

### 2.1 In-workspace (verified via `ls` + `head -30` on `lib.rs` / source)

| Need | Crate / file | Evidence | Reuse plan |
|---|---|---|---|
| In-memory paged KV blocks | `crates/spark-runtime/src/kv_cache.rs` (517 LoC, `KvCacheDtype` Bf16/Fp8/Nvfp4/Turbo4/Turbo3, `DevicePtr`, block allocator) | `pub(crate) const NVFP4_GROUP_SIZE`; manages fixed-size token-position blocks | **Direct reuse:** import `kv_cache::KvCache` / block types; server holds `Arc<KvCache>` pool |
| NVMe offload / spill / paging | `crates/spark-storage/src/kv_paging.rs`, `kv_spill.rs` | `public` modules in `lib.rs`; handle overflow to disk | **Direct reuse:** `kv_spill::spill_to_nvme()` / `kv_paging::restore()` called when in-mem full |
| Remote-RAM server (existing) | `crates/spark-storage/src/cache_peer/` (mod.rs, registry.rs, registry_tests.rs, server_impl.rs) | `pub mod cache_peer` in `lib.rs` line 67; docs say "passive remote-RAM RW tier served over one-sided RDMA" | **Direct reuse:** `cache_peer::ServerImpl` model; wrap with gRPC endpoint |
| RDMA KV transport backend | `crates/spark-storage/src/rdma_kv_backend.rs` | Gated behind `#[cfg(all(feature = "cuda", atlas_rdma_verbs))]`; one-sided READ | **Feature-gated reuse:** if `atlas_rdma_verbs` enabled, expose RDMA path; else gRPC only |

### 2.2 crates.io (verified via `curl api/v1/crates`)

| Need | Crate (version) | Why / not custom |
|---|---|---|
| Generic in-memory cache / memoization (reference patterns) | `cached` v4.0.1 | Reference only — our server needs typed KV blocks, not generic lambda cache; reuse workspace `kv_cache` instead |
| Remote cache protocol (reference wire contract) | `mbx-cache-protocol` v0.5.22 | Reference wire-format design; not adopted (our proto uses `tonic`/`prost` per workspace convention) |
| Columnar in-memory cache for AI/ML | `murr` v0.3.1 | Reference only — columnar layout different from Atlas block format |
| Remote cache client driver | `redis` v1.7.1 | Reference for external-client connection patterns; not embedded (no Redis server dependency) |

**No custom KV block allocator, page manager, or RDMA server written.** Custom work limited to: (a) server wrapper / config, (b) gRPC proto (`proto/kv_cache.proto`), (c) memory/size gate, (d) optional `nvme_path` integration with existing spill.

## 2.5 Compression & De-duplication — Search / Selection (added)

From `kv_cache.rs` (line 12–55): existing `KvCacheDtype` already provides quant-level compression — `Bf16`, `Fp8`, `Fp8G128`, `Nvfp4`, `Turbo4`, `Turbo3`, `Turbo4KTurbo3V`, `Turbo4KTurbo8V`. Reuse these; add byte-level `zstd` / `lz4` on top; dedup via `sha2` content hash.

| Need | Crate (version) | Why / lazy choice |
|---|---|---|
| Byte-level compression (before disk, optional before memory) | `zstd` (standard crate, `zstd-pure-rs` v0.2.0 / `oxiarc-zstd` v0.4.2 if pure-Rust required) | `zstd` is the standard; workspace likely already links it via CUDA/toolchains; use `zstd` directly (lazy — stdlib-equivalent for compression) |
| Fast compression alternative | `lz4` v1.28.1 | Use if speed > ratio (deferred — config selects `zstd` default) |
| Content-hash for dedup | `sha2` v0.11.0 (pure Rust SHA-256) | SHA-256 of compressed bytes = content key; identical values map to single storage entry |
| Dedup reference | `groupcache` / `redhac` (from earlier search) | Reference patterns only — our dedup is simpler: hash index + single storage |

**No custom compression algorithm; reuse `KvCacheDtype` quant + `zstd` + `sha2`.**

```
┌────────────────────────────────────────────────────────────────┐
│  atlas-kv-cache-server  (standalone / lib crate)               │
│  ┌─────────────┐  ┌─────────────────────────────────────────┐ │
│  │ Config      │  │ In-Memory Pool (kv_cache)              │ │
│  │ - mem_bytes │  │ - Block allocator (fixed-size)         │ │
│  │ - nvme_path │  │ - Quantized (Bf16/Fp8/Nvfp4)           │ │
│  └─────────────┘  └─────────────────────────────────────────┘ │
│  ┌─────────────┐  ┌─────────────────────────────────────────┐ │
│  │ gRPC Server │  │ NVMe Offload (kv_paging / kv_spill)    │ │
│  │ (tonic)     │  │ - Optional (configurable)                │ │
│  │ - Put/KV    │  │ - Spill when pool full / eviction        │ │
│  │ - Get/KV    │  └─────────────────────────────────────────┘ │
│  │ - Remove    │                                            │
│  └─────────────┘  ┌─────────────────────────────────────────┐ │
│                   │ RDMA (feature: atlas_rdma_verbs)        │ │
│                   │ - cache_peer server_impl reusable       │ │
│                   └─────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
         ▲              ▲
         │ gRPC / TCP   │ RDMA (optional)
    ┌────┴────┐    ┌────┴────┐
    │ Client  │    │ Client  │  (spark-runtime / spark-storage instances)
    │ (node A)│    │ (node B)│
    │ in-mem  │    │ in-mem  │
    └─────────┘    └─────────┘
```

### 3.1 Core Concepts

- **Server = data holder + allocator.** Holds KV blocks in memory (via reused `kv_cache`); spills to NVMe (via `kv_paging`/`kv_spill`) when `max_memory_bytes` reached or on explicit `Put` with `spill_hint`.
- **Configurable in-memory size:** `max_memory_bytes` caps pool; `kv_cache` block allocator manages fixed-size blocks (not variable-size objects — matches Atlas attention KV format).
- **Optional local disk:** `nvme_path: Option<PathBuf>`; if set, `kv_spill` handles page writes; if `None`, pure memory; server rejects `Put` that would exceed cap (instead of silent spill — explicit is safer per PCND).
- **Access protocol:** gRPC `Put(key, block) → ack`, `Get(key) → block`, `Remove(key) → ack`. Key is opaque string / bytes (not interpreted by server — matches `kv_cache` opaque block indexing).
- **Reuse of `cache_peer`:** the existing `cache_peer` server (RDMA) can serve as the data-plane when `atlas_rdma_verbs` enabled; gRPC as control + fallback.

---

## 4. Crate Structure (new `crates/atlas-kv-cache-server/`)

```
crates/atlas-kv-cache-server/
├── Cargo.toml              # feats: cuda, atlas_rdma_verbs, tonic; deps: spark-runtime, spark-storage, tonic, prost, serde, tokio, thiserror, tracing
├── src/
│   ├── lib.rs              # SPDX AGPL-3.0-only; deny; module exports
│   ├── config.rs           # ServerConfig: max_memory_bytes, nvme_path, block_size, bind_addr, enable_rdma
│   ├── server.rs           # tonic gRPC server (Put/Get/Remove); memory gate; spill trigger
│   ├── cache_pool.rs       # wrapper around spark-runtime kv_cache; alloc/free; size tracking
│   ├── nvme_spill.rs       # wrapper around spark-storage kv_spill / kv_paging; optional
│   ├── rdma_serve.rs       # feature-gated; wraps spark-storage cache_peer server_impl
│   ├── error.rs            # thiserror: PoolFull, KeyNotFound, SpillFailed, NvmePathInvalid
│   └── proto/
│       └── kv_cache.proto # PutRequest, GetRequest/Response (bytes block), RemoveRequest
└── tests/
    ├── pool_size_tests.rs  # max_memory_bytes enforced; spill triggered; pure-mem mode
    ├── nvme_offload_tests.rs # nvme_path set; put exceeds cap → spill; get restores
    ├── server_integration_tests.rs # gRPC roundtrip; multi-client; concurrent put/get
    └── rdma_feature_tests.rs # feature enabled; server starts; disabled = no compile error
```

---

## 5. Implementation Details

### 5.1 Config (`src/config.rs`)

```rust
#[derive(Clone, Debug, Deserialize)]
pub struct ServerConfig {
    pub bind_addr: String,           // "0.0.0.0:50052"
    pub max_memory_bytes: usize,     // e.g., 8 * 1024 * 1024 * 1024 = 8GB
    pub nvme_path: Option<PathBuf>,  // "/mnt/nvme/atlas-kv/" or None
    pub block_size: usize,           // per kv_cache block (e.g., 128 tokens)
    pub enable_rdma: bool,           // requires atlas_rdma_verbs + cuda features
    pub compression: CompressionMode, // zstd / lz4 / none; applies to disk and optional mem
    pub dedup_enabled: bool,         // SHA-256 content-hash dedup index
}

#[derive(Clone, Debug, Deserialize, Default, PartialEq)]
pub enum CompressionMode {
    #[default]
    None,
    Zstd,
    Lz4,
}

pub const DEDUP_ALGORITHM: &str = "sha2-256"; // SHA-256 of compressed value bytes
```

- `max_memory_bytes` enforced by `cache_pool.rs`: before allocating, check `used + block_size ≤ max`; if exceeds and `nvme_path` set → spill oldest / evicted block; if `None` → return `Error::PoolFull` (fail fast, not silent drop — PCND).
- `nvme_path` validated at startup: exists? writable? `fs::metadata` check; fail fast on invalid path.

### 5.2 Cache Pool (`src/cache_pool.rs`) — reuse `kv_cache`

```rust
use spark_runtime::kv_cache::{KvCache, Block}; // from workspace

pub struct CachePool {
    inner: Arc<KvCache>,             // reused
    max_bytes: usize,
    used_bytes: AtomicUsize,
    nvme_spill: Option<Arc<NvmeSpill>>,
}
```

- `Put(key, bytes)`: (1) serialize to `Block`; (2) `inner.allocate()`; (3) if `used_bytes + len > max_bytes` → `spill_oldest()` via `kv_spill`; (4) store mapping `key → block_ref`; (5) return `Ack`.
- `Get(key)`: lookup mapping; if in memory → return `Block`; if spilled → `nvme_spill.restore()` then return; if not found → `Error::KeyNotFound`.
- Mapping storage: `DashMap<String, BlockLocation>` (memory) or `HashMap` under `RwLock` (if future Raft needed). Phase 1: `DashMap` (concurrent, like previous plan's registry).

### 5.3 NVMe Spill (`src/nvme_spill.rs`) — reuse `kv_paging` / `kv_spill`

```rust
use spark_storage::{kv_paging, kv_spill};

pub struct NvmeSpill {
    path: PathBuf,
    // delegates to spark-storage's existing spill / restore APIs
}
```

- `spill(key, block_bytes)` → write to `{path}/{key_hash}.blk`; update index.
- `restore(key)` → read from disk; load into `CachePool`; optionally evict from disk after restore (configurable `keep_on_disk` flag — deferred decision).
- If `nvme_path` is `None`, `NvmeSpill` is `None`; `CachePool` never calls spill.

### 5.4 gRPC Server (`src/server.rs`)

```protobuf
service KvCache {
  rpc Put (PutRequest) returns (PutResponse);
  rpc Get (GetRequest) returns (GetResponse);
  rpc Remove (RemoveRequest) returns (RemoveResponse);
}
message PutRequest { string key = 1; bytes block = 2; bool allow_spill = 3; }
message GetResponse { bytes block = 1; bool from_nvme = 2; }
```

- `Put` validates `key.len() > 0`; validates `block.len() ≤ block_size * bytes_per_token` (optional guard — deferred strict enforcement to avoid rejecting valid blocks from different quant configs).
- `Get` returns `block` + `from_nvme` flag (client knows if latency is higher — useful for metrics / debugging).
- Concurrent: `tokio::sync::RwLock` or `DashMap` for mapping; `kv_cache` internal may use `Mutex` — document that high concurrency requires profiling (deferred benchmark F3-style).

### 5.6 Compression & De-duplication Pipeline (added)

**Most memory-efficient storage:**

1. **Quant (always):** `KvCacheDtype` (Fp8/Nvfp4/Turbo4 etc.) → reduces token-position elements at source; no extra CPU.
2. **Byte-level (disk + optional mem):** after quant, if `compression ≠ None`, `zstd` (default) or `lz4` (speed mode) compresses the `bytes` payload before writing to `nvme_path`; decompressed on `Get`.
3. **De-dup (optional):** if `dedup_enabled = true`, compute `sha2::Sha256::digest(&compressed_bytes)` → `hash_key`; store only if `hash_key` not in `dedup_index`; index maps `hash_key → storage_path` (for disk) or `hash_key → block_ref` (for mem). Identical values share one entry — saves both RAM and disk when multiple keys have same block content.

```rust
// Pipeline (lazy — built over existing modules)
fn store(&mut self, key: &str, raw_block: &[u8], config: &ServerConfig) -> Result<()> {
    // 1. Quantize (if block format requires) — delegated to kv_cache block types
    let quantized = self.quantize_if_needed(raw_block, config.dtypes);
    // 2. Compress
    let compressed = if config.compression == CompressionMode::Zstd {
        zstd::encode_all(&quantized[..], 3)? // level 3 default; configurable deferred
    } else if config.compression == CompressionMode::Lz4 { ... } else { quantized };
    // 3. Dedup
    if config.dedup_enabled {
        let hash = sha2::Sha256::digest(&compressed);
        if self.dedup_index.contains_key(&hash) { return Ok(AlreadyExists); }
        self.dedup_index.insert(hash.clone(), path_or_ref);
    }
    // 4. Write to mem or nvme
    self.pool.insert(key, compressed); // or spill
}
```

**Memory efficiency priority (lazy — simplest that works):** quant > zstd > dedup. All three can be enabled together; if `dedup_enabled` and `compression=None`, dedup operates on raw/block bytes (lower savings — document in config).

**Disk storage format (deferred detail, but specified):** file name = `{key_hash}.blk` OR (with dedup) `{sha256_digest}.blk` with index mapping key→digest. Index file: `.dedup.idx` (JSON / bincode) in `nvme_path`. Defer full index format to implementation; plan specifies key→hash mapping.

**Tests for compression/dedup:** see §7.4 (added).

- When feature enabled and `enable_rdma = true`: start `cache_peer::ServerImpl` (from `spark-storage`) on additional port / endpoint.
- Reuse `rdma_kv_backend` for pinned-host bounce + copy_h2d if CUDA needed.
- If feature disabled: module not compiled (no errors) — per workspace convention (`#[cfg(...)]`).

---

## 6. Core Tasks (ordered, acceptance criteria)

| # | Task | AC |
|---|---|---|
| T1 | Search + select crates (see §2) | Documented reuse of `kv_cache`, `kv_paging`, `cache_peer`; no custom allocator / server |
| T2 | Define `proto/kv_cache.proto` | Put/Get/Remove; `bytes block` field; compiles with `prost-build` |
| T3 | Implement `CachePool` wrapping `kv_cache` | Unit: allocate 10 blocks, total ≤ max_memory_bytes; exceed → spill or error |
| T4 | Implement `NvmeSpill` wrapping `kv_paging`/`kv_spill` | Unit: put with `nvme_path` set → file exists; restore → same bytes |
| T5 | Implement `config.rs` + validation | `max_memory_bytes` > 0; `nvme_path` exists & writable; fail-fast on bad config |
| T6 | Implement `server.rs` (tonic) | Integration: client Put → server stores; Get → same bytes; concurrent clients no panic |
| T7 | Implement `rdma_serve.rs` feature-gated | `atlas_rdma_verbs` enabled → module compiles; disabled → clean compile; no runtime error |
| T8 | `lib.rs` exports + `Cargo.toml` | `SPDX`; `deny(warnings)`; features declared; builds `cargo build --features cuda` |
| T9 | Size-guard test (`pool_size_tests.rs`) | `max_memory_bytes = 1MB`; put 2MB → `PoolFull` (if `nvme_path=None`) or spill (if set) |
| T10 | Offload test (`nvme_offload_tests.rs`) | `nvme_path=/tmp/test_nvme`; put 100MB (exceeds 10MB cap) → disk file; get → restored to memory |

---

## 7. Testing Requirements & Specific Tests

### 7.1 Unit (must pass in CI)

- `pool_size_tests.rs`
  - `test_pool_respects_max_memory`: set 1MB cap; put blocks totaling <1MB success; put exceeding → `PoolFull` (no-spill mode).
  - `test_spill_triggers_when_nvme_set`: cap 1MB; `nvme_path=/tmp`; put 2MB → spill succeeds; index updated; memory stays ≤ cap.
  - `test_pure_memory_no_disk`: `nvme_path=None`; put exceeds cap → error returned (not silent drop — PCND).

- `nvme_offload_tests.rs`
  - `test_nvme_path_invalid_fails_fast`: `nvme_path="/nonexistent/path"`; startup → `Error::NvmePathInvalid`.
  - `test_restore_from_nvme`: put with spill; delete from memory (simulated evict); `Get` → restore from disk; bytes match.
  - `test_multi_spill_sequence`: 10 puts, each 2MB, cap 1MB → oldest spilled each time; final memory ≤1MB; all 10 restorable.

- `rdma_feature_tests.rs`
  - `test_rdma_feature_compile`: `cargo build --features atlas_rdma_verbs` succeeds; module `rdma_serve` exists.
  - `test_rdma_disabled_clean`: `cargo build --no-default-features` succeeds; no `rdma_serve` module; no link errors.

### 7.2 Integration (require `tokio::test` / `tonic` server)

- `server_integration_tests.rs`
  - `test_grpc_roundtrip`: start server on ephemeral port; `Put(key, block)`; `Get(key)` → same `block`; `Remove(key)`; `Get(key)` → `KeyNotFound`.
  - `test_concurrent_put_get`: 10 clients, 100 ops each; no panic; all gets return correct data (race on mapping — `DashMap` handles).
  - `test_from_nvme_flag`: put with spill; get returns `from_nvme = true`; put without spill → `false`.
  - `test_pool_full_graceful`: cap 1MB; concurrent puts exceeding cap → some return `PoolFull`, not crash.

### 7.4 Compression & De-duplication Tests (added)

- `test_quant_reuse_fp8_turbo4`: put block with `KvCacheDtype::Fp8`; verify stored block size smaller than Bf16 (quant saves); `Get` returns correct dtype.
- `test_zstd_compress_decompress`: `compression = Zstd`; put 100KB block → disk file < 100KB; `Get` → original bytes restored (bit-exact).
- `test_lz4_faster_lower_ratio`: `compression = Lz4`; verify compressed size > zstd but decompress faster (soft check — log only, not hard CI).
- `test_dedup_same_content_different_keys`: put `key_a` with block X; put `key_b` with identical block X (`dedup_enabled = true`); second put returns `AlreadyExists`; storage has 1 file; `Get(key_a)` and `Get(key_b)` both return X.
- `test_dedup_different_content_same_key`: put `key_a` block X; put `key_a` block Y (overwrite); old hash removed; new hash stored; get returns Y.
- `test_compression_none_alone`: `compression = None`, `dedup_enabled = false`; disk file = raw quant bytes; no extra CPU overhead.
- `test_memory_efficient_combo`: `compression = Zstd`, `dedup_enabled = true`, quant active; put 3 identical blocks under 3 keys with 1MB cap; total storage < 1MB (dedup + zstd); all 3 gets succeed.

### 7.3 Behavior verification (design → test mapping)

| Intended | Test |
|---|---|
| Configurable in-memory size enforced | `test_pool_respects_max_memory` |
| Optional local storage (NVMe) offload | `test_nvme_path_invalid`; `test_spill_triggers` |
| Reuse existing `kv_cache` / `kv_paging` | Source review: imports from `spark_runtime` / `spark_storage`; no duplicate block allocator |
| Fail-fast on bad config (PCND) | `test_nvme_path_invalid_fails_fast`; startup validates `max_memory_bytes > 0` |
| Feature-gated RDMA reuse of `cache_peer` | `test_rdma_feature_compile`; `test_rdma_disabled_clean` |

---

## 8. General Task List with Requirements

- [ ] **Req-A:** Crate uses existing `kv_cache` / `kv_paging` / `kv_spill` / `cache_peer` — no custom block allocator, page manager, or RDMA server (verified by source review, §2).
- [ ] **Req-B:** `SPDX-License-Identifier: AGPL-3.0-only` line 1; `deny(warnings)`; file cap ≤500 LoC (split if needed).
- [ ] **Req-C:** Config validates at startup; `PoolFull` returned rather than silent drop (`PCND` — no implicit defaults for memory limit).
- [ ] **Req-D:** `nvme_path` optional — pure mem mode supported (`None`); disk mode requires path existence + writable.
- [ ] **Req-E:** `proto/kv_cache.proto` versioned; `GetResponse.from_nvme` is optional (backward-compat if removed, per proto rules).
- [ ] **Req-F]: `atlas_rdma_verbs` feature gates RDMA; clean compile without CUDA / RDMA (matches workspace `spark-storage` gating).
- [ ] **Req-H:** Compression + dedup: `KvCacheDtype` quant reused (no custom quant); `zstd`/`lz4` from crates.io; `sha2` SHA-256 for dedup index; memory-efficiency verified by `test_memory_efficient_combo`.

---

## 9. Decisions Log

| ID | Area | Decision | Why |
|---|---|---|---|
| D1 | Cache reuse | `kv_cache` + `kv_paging` / `kv_spill` + `cache_peer` | Existing workspace modules handle block format, offload, RDMA; avoids rewrite |
| D2 | Config | TOML (`ServerConfig` via `serde`); fail-fast validation | Matches `MODEL.toml`; PCND — no implicit defaults |
| D3 | Memory limit | `max_memory_bytes`; `PoolFull` error if exceeded and no NVMe | Explicit over implicit; no silent eviction unless configured |
| D4 | Protocol | `tonic` gRPC + `prost`; RDMA feature-gated | Workspace convention; `cache_peer` provides RDMA data-plane |
| D5 | Mapping concurrency | `DashMap` (Phase 1) | Concurrent put/get; future Raft lock deferred |
| D6 | Compression / dedup | Quant (existing) + `zstd`/`lz4` (crates.io) + `sha2` (content hash); 3-layer pipeline | Most memory-efficient; lazy — reuse existing quant, add standard crates only |

---

## 10. Assumptions

- `kv_cache` block format (quant, block_size) matches clients; server does not interpret quant — stores opaque bytes.
- `nvme_path` disk has sufficient IOPS for spill/restore (not benchmarked in Phase 1; deferred to F3-style perf test).
- Single instance (no HA); clients must retry / fall back if server down (same as previous plan's coordinator assumption, but now server holds data — fallback = local cache only).
- Key space opaque; server does not enforce key length strictly (defensive 4KB guard only; not a content policy).
- RDMA path requires `cuda` feature + `atlas_rdma_verbs` build flags; not available on Apple Silicon / no-CUDA hosts (matches `spark-storage` gating).

---

## 11. Deferred / Not Implemented (with triggers)

| ID | Item | Trigger | Notes |
|---|---|---|---|
| F1 | Benchmark `bench_pool_latency` | Pre-release perf validation | Deferred — `kv_cache` performance from runtime is reference |
| F2 | HA / replication of server data | HA goal added | Defer to consensus layer; server is stateful — replication is harder than metadata-only coordinator |
| F3 | Snapshot / persistence of server state | Crash recovery needed | `kv_paging` already persists spilled data; in-mem lost on crash (documented); snapshot of `DashMap` index deferred |
| F4 | Auth / TLS on gRPC | External network deploy | Same deferred pattern as previous plan (§16.2) |
| F5 | Eviction policy (LRU / LFU) | Cache size burns | Phase 1 spill = oldest (simple); LRU deferred to policy config |

---

---

## 12. Client Library (deferred — usage examples for `spark-runtime` / nodes)

*Status: not implemented; interface reserved in `proto/kv_cache.proto` so client can be written independently.*

```rust
// Example client (lazy — minimal, no custom wire code)
use atlas_kv_cache_server::client::KvCacheClient; // tonic-generated
let mut client = KvCacheClient::connect("http://kv-server:50052").await?;
client.put(PutRequest { key: "layer-7-block-0".into(), block: bytes, allow_spill: true }).await?;
let resp = client.get(GetRequest { key: "layer-7-block-0".into() }).await?;
// resp.block holds data; resp.from_nvme indicates disk retrieval
```

- Client knows nothing about quant / compression / dedup — server handles pipeline (§5.6). Client only passes opaque `bytes`.
- Retry policy: exponential backoff on `UNAVAILABLE`; fallback to local `kv_cache` if server unreachable (documented in `client.rs` doc, implemented by caller not library).

---

## 13. Metrics / Monitoring (deferred — reserve `metrics` integration)

| Metric | Source | Purpose |
|---|---|---|
| `kv_cache_lookups_total` | `server.rs` (counter, labeled by `from_nvme`) | Load / performance tracking |
| `kv_cache_memory_used_bytes` | `CachePool` (gauge) | Capacity planning; alert if near `max_memory_bytes` |
| `kv_cache_spill_total` | `NvmeSpill` (counter, labeled `reason=capacity`) | Disk wear / offload rate |
| `kv_cache_compressed_bytes` / `kv_cache_raw_bytes` | `CachePool` (after store) | Compression efficiency ratio |
| `kv_cache_dedup_hits_total` | `CachePool` (counter) | Dedup effectiveness |

- Integration: `metrics::register_counter!` / `register_gauge!` call sites reserved in `server.rs` comments; actual `metrics` crate dependency deferred until deploy needs Prometheus endpoint.
- Endpoint: `/metrics` (if enabled) served by `tonic` health/metrics interceptor — deferred to deploy config.

---

## 14. Deploy / Runbook Commands (deferred — reference workspace patterns)

Based on `GB10_DEPLOYMENT_GUIDE.md` / `ATLAS_SPARK_JOURNEY.md` patterns:

```bash
# Build (with CUDA if RDMA needed; without is fine for gRPC-only)
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo build --package atlas-kv-cache-server --features cuda,atlas_rdma_verbs

# Start (single instance, Phase 1)
./target/release/atlas-kv-cache-server --config /etc/atlas/kv-cache-server.toml
# Config: bind_addr=0.0.0.0:50052, max_memory_bytes=8589934592 (8GB), nvme_path=/mnt/nvme/atlas-kv/

# Verify (health / readiness)
curl -sf http://localhost:50052/health || echo "down"
# gRPC readiness: client Put empty key → expects INVALID_ARGUMENT (service up)

# Restart / upgrade (Phase 1 limitation — state lost; clients fall back to local)
systemctl restart atlas-kv-cache-server
# After restart: clients must re-register / reconnect; pool rebuilt from NVMe (spilled data survives; in-mem lost)

# Monitor
watch -n 5 'curl -s localhost:50052/metrics | grep kv_cache_memory_used'
```

---

## 15. Cost Update with Compression / Dedup (added)

| Resource | Phase 1 (no comp) | With zstd + dedup | Savings |
|---|---|---|---|
| In-memory | 8GB cap | 8GB cap (compressed mem only if `compression` applied to mem) | Quant saves ~2–6× (Fp8/Nvfp4); zstd saves additional 1.5–3× on disk |
| NVMe disk | 100MB/day logs + spill | Compress reduces spill footprint ~1.5–3×; dedup reduces if duplicate content common | Potential 2–4× reduction vs raw |
| Network (per Get) | ~200B meta + payload | Payload size reduced if client reads compressed (defer — server sends decompressed; client can opt compressed if proto extended) | Defer to proto v2 |

Lazy: cost update is order-of-magnitude; precise only after `test_memory_efficient_combo` and benchmark F3.

---

*Plan expanded 2026-05-29. Sections 12–15 added per clarification (option b: client/metrics/deploy). No custom implementation written; all design/spec only. Existing workspace patterns (`GB10_DEPLOYMENT_GUIDE.md`, `spark-server` client, `metrics` reservoir) referenced. Compression/dedup (§2.5 / §5.6 / §7.4 / §9 D6 / §11 F) remain integrated.*
