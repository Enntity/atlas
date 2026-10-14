# Plan: Remote KV-Cache Coordinator Crate (`atlas-kv-coord`)

Status: DESIGN — not implemented. Search results below drive crate selection.

---

## 1. Clarification of Scope (per AGENTS.md / ponytail)

| Dimension | Decision |
|---|---|
| HA now? | **No.** Design for future HA via consensus; no leader election / replication in Phase 1. |
| Consensus mechanism for future HA? | **Raft** (via `raft-consensus` / `raft-consensus-core` crate family — search found `raft-consensus v0.4.0`, `raft-consensus-core v0.1.2`; `zenoh_raft v0.1.7`). Leave `ConsensusBackend` trait unimplemented in Phase 1. |
| Node-local media? | **Both** in-memory (`Vec`/dashmap) and local disk (NVMe) per node. Coordinator only tracks *who holds what*; it does not serve payload bytes. |
| Lookup complexity target? | **O(1)** per key → node mapping. Use consistent hash ring, not range scan. |
| Transport between nodes for coordinate updates? | gRPC (`tonic v0.14.6` found) + JSON/meta over TCP for Phase 1; RDMA (`atlas-rdma`) out of scope for coordinator itself but referenced in node-local config. |
| External server vs embedded? | **External server** (standalone daemon/crate) that all `spark-runtime` / `spark-storage` instances register with. |

---

## 2. Search Results — Existing Crates Used Instead of Custom

From `crates.io` search (`curl` api/v1/crates, verified 2026-05-29):

| Need | Crate (version) | Why used / not custom |
|---|---|---|
| Consistent hash ring (O(1) node lookup) | `hash-rings` v1.1.0 (repo: gitlab.com/jeffrey-xiao/hash-rings-rs, docs.rs available) / `consistent_hash_ring` v0.8.0 / `hash_ring` v0.2.0 / `lfchring` v0.1.3 (lock-free) | **Use `hash-rings` v1.1.0** for ring + vnode replication; `lfchring` only if lock-free concurrent ring needed later. No custom ring. |
| Distributed cache filling / peer group (reference pattern) | `groupcache` v0.3.0 | Reference only — maps to our "node has local disk" model. Not embedded (groupcache requires Go-inspired proto). |
| HA / embedded distributed cache | `redhac` v0.10.5 ("Rust Embedded Distributed HA Cache") | Reference for HA design patterns; not adopted (our HA is future via Raft, not embedded). |
| Cluster coordination / leader election (future HA) | `raft-consensus` v0.4.0 / `raft-consensus-core` v0.1.2 / `zenoh_raft` v0.1.7 / `photon-etcd-cluster` v0.1.0 | **Future only.** Phase 1 defines `ConsensusBackend` trait; implementation deferred. |
| gRPC server/client (coordination RPC) | `tonic` v0.14.6 | Phase 1 server endpoint uses tonic + prost. |
| KV / serialization | `serde` + `bincode` / `rmp-serde` (existing in workspace via `spark-storage` / `atlas-core`) | Reuse; do not invent wire format. |

**No custom consistent-hash, ring, or consensus code written in Phase 1.** Custom work is limited to: (a) coordinator state machine, (b) node-registration protocol, (c) HA-trait shell.

---

## 3. Design (High-Level)

```
┌─────────────────────────────────────────────────────────────┐
│  atlas-kv-coord  (standalone server / lib crate)            │
│  ┌──────────────┐  ┌─────────────────────────────────────┐  │
│  │ HashRing     │  │ NodeRegistry (in-memory, future     │  │
│  │ (hash-rings) │  │  replicated via Raft)               │  │
│  └──────────────┘  └─────────────────────────────────────┘  │
│  ┌──────────────┐  ┌─────────────────────────────────────┐  │
│  │ RPC (tonic)  │  │ ConsensusBackend (trait, unimpl)     │  │
│  │  - Register  │  │  - LeaderElection (future)          │  │
│  │  - GetNode   │  │  - LogReplication (future)          │  │
│  │  - Update    │  └─────────────────────────────────────┘  │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
         ▲                          ▲
         │ gRPC / TCP              │ (future Raft)
    ┌────┴────┐               ┌─────┴─────┐
    │ Node A  │  local disk   │  Node B   │
    │ in-mem  │◄─────────────►│ in-mem    │
    └─────────┘  (RDMA / NVMe)└───────────┘
    (registered to coordinator; does NOT server payload)
```

### 3.1 Core Concepts

- **Coordinator = metadata only.** It knows `(node_id, medium: Mem | Disk, vnodes[], health)` for each registered node. It does NOT hold KV payloads.
- **Hash ring (consistent).** Each node contributes `V` virtual nodes (vnodes) to `hash-rings`. Key → `hash(key) % ring` → nearest vnode → owning node. **O(1)** lookup (hash + binary search over sorted vnode list; `hash-rings` provides this).
- **Medium-agnostic lookup.** `GetNode(key)` returns `NodeRef { node_id, medium, endpoint }`; caller decides whether to hit local in-mem, local disk, or remote fetch.
- **Node-local media.** Each node (running `spark-runtime` / `spark-storage`) can have both mediums; both register with coordinator under same `node_id` with different `medium` tags.
- **Future HA = Raft over `ConsensusBackend`.** Leader holds the authoritative ring; followers replicate via `raft-consensus`. Not implemented in Phase 1.

---

## 4. Crate Structure (new `crates/atlas-kv-coord/`)

```
crates/atlas-kv-coord/
├── Cargo.toml              # deps: hash-rings, tonic, prost, serde, tokio, thiserror; dev: tokio-test, rstest
├── src/
│   ├── lib.rs              # SPDX, deny(warnings/clippy), module re-exports, public API
│   ├── ring.rs             # wrapper around hash-rings; vnode addition/removal; O(1) lookup
│   ├── registry.rs         # NodeRegistry: HashMap<node_id, NodeInfo>; registration; heartbeats
│   ├── server.rs           # tonic gRPC server; RegisterNode / GetNode / UpdateNode / Heartbeat
│   ├── client.rs           # tonic gRPC client stub (for nodes to call coordinator)
│   ├── consensus.rs        # ConsensusBackend trait (unimplemented); Raft refs
│   ├── error.rs            # thiserror enum: NotFound, RingEmpty, NodeDown, ConsensusNotReady
│   └── config.rs           # CoordinatorConfig: bind_addr, vnode_count, heartbeat_interval
├── proto/
│   └── kv_coord.proto     # RegisterRequest, NodeInfo, GetNodeRequest/Response, Heartbeat
└── tests/
    ├── ring_lookup_tests.rs
    ├── registry_tests.rs
    ├── server_integration_tests.rs
    └── consensus_trait_tests.rs
```

---

## 5. Implementation Details

### 5.1 Ring (`src/ring.rs`)

```rust
use hash_rings::{HashRing, Node, Ring};

pub struct CoordRing {
    inner: HashRing,   // from hash-rings v1.1.0
    vnode_count: usize,
}

impl CoordRing {
    pub fn add_node(&mut self, node_id: &str, vnode_replicas: usize) { ... }
    pub fn remove_node(&mut self, node_id: &str) { ... }
    /// O(1) => hash + nearest-vnode binary search (hash-rings handles)
    pub fn lookup(&self, key: &[u8]) -> Option<NodeRef> { ... }
}
```

- **No custom hash:** `hash-rings` provides consistent hashing; we wrap for `node_id`-to-`NodeRef` mapping.
- **Rebalance on add/remove:** `hash-rings` minimizes movement; we document expected % of keys moved.

### 5.2 Registry (`src/registry.rs`)

```rust
pub struct NodeInfo {
    pub id: String,
    pub endpoints: Vec<String>,   // gRPC + data-plane
    pub mediums: Vec<Medium>,     // Mem / Disk
    pub vnodes: usize,
    pub registered_at: Instant,
    pub last_heartbeat: Instant,
}

pub struct NodeRegistry {
    nodes: DashMap<String, NodeInfo>,  // concurrent; future: RwLock for Raft sync
}
```

- Heartbeat: every `heartbeat_interval` (config, default 5s). Missing 3 → mark `down`; ring excludes down nodes.
- Registration is idempotent (re-register updates info).

### 5.3 Server / RPC (`src/server.rs`, `proto/kv_coord.proto`)

```protobuf
service KvCoord {
  rpc RegisterNode (RegisterRequest) returns (RegisterResponse);
  rpc GetNode (GetNodeRequest) returns (GetNodeResponse);
  rpc UpdateNode (UpdateRequest) returns (UpdateResponse);
  rpc Heartbeat (HeartbeatRequest) returns (HeartbeatResponse);
}
```

- `GetNode(key: bytes)` → call `CoordRing.lookup(key)` → return `NodeRef`. **This is the O(1) path.**
- No payload transfer; only metadata.

### 5.4 Consensus Shell (`src/consensus.rs`)

```rust
#[async_trait]
pub trait ConsensusBackend: Send + Sync {
    /// Future: return true if this instance is the Raft leader
    fn is_leader(&self) -> bool;
    /// Future: replicate registry delta via Raft log
    async fn replicate_delta(&self, delta: RegistryDelta) -> Result<()>, Error>;
    /// Future: wait for quorum
    async fn wait_for_quorum(&self) -> Result<()>;
}

/// Phase 1: stub; always returns true, no-op.
pub struct NoOpConsensus;
```

- Defer to `raft-consensus` / `zenoh_raft` when HA needed. Trait ensures interface is HA-ready.

---

## 6. Core Tasks (ordered, with acceptance criteria)

| # | Task | Acceptance |
|---|---|---|
| T1 | Search & select crates (`hash-rings`, `tonic`, `raft-consensus` references) | Documented in this plan (§2); no custom ring / consensus. |
| T2 | Define `proto/kv_coord.proto` | All 4 RPCs defined; compiles with `prost-build`. |
| T3 | Implement `CoordRing` wrapping `hash-rings` | Unit test: add 3 nodes, lookup 100 keys, all resolve; remove 1, <20% move. |
| T4 | Implement `NodeRegistry` + heartbeat | Test: register → heartbeat → miss 3 → `down`; re-register → `up`. |
| T5 | Implement tonic `server.rs` + `client.rs` | Integration: client registers, server returns correct node for key. |
| T6 | Implement `ConsensusBackend` trait + `NoOpConsensus` | Trait compiles; `NoOpConsensus::is_leader()` always true. |
| T7 | `lib.rs` exports + `Cargo.toml` | `cargo build`; `cargo clippy -- -D warnings`; SPDX header line 1. |
| T8 | Integration test: multi-node lookup consistency | 2 nodes, 1000 keys, both clients agree on same node per key. |
| T9 | Performance / O(1) claim test | `lookup` p99 < 1µs for ring size 1024 vnodes (benchmark, not CI-gated). |

---

## 7. Testing Requirements & Specific Tests

### 7.1 Unit Tests (must pass in CI, `cargo test`)

- `ring_lookup_tests.rs`
  - `test_lookup_single_node`: 1 node, any key → that node.
  - `test_lookup_three_nodes_balanced`: 3 nodes, 100 random keys → roughly 33/33/33 (±10%).
  - `test_lookup_removal_rebalance`: add 3, lookup 50; remove 1; re-lookup — >50% of previously-mapped keys stay on same node (consistent-hash property).
  - `test_lookup_o1_time`: 1000 iterations; assert mean < 1µs (soft; log only, not hard-fail CI).

- `registry_tests.rs`
  - `test_register_and_heartbeat`: register; heartbeat succeeds; miss 3 → down flag set.
  - `test_re_register_updates_medium`: register with Mem; re-register with Disk added; query returns both.
  - `test_race_register_lookup`: concurrent register + lookup; no panic, result is consistent.

- `consensus_trait_tests.rs`
  - `test_noop_is_leader`: `NoOpConsensus::is_leader() == true`.
  - `test_trait_object`: `Box<dyn ConsensusBackend>` compiles; `is_leader()` callable.

### 7.2 Integration Tests (require `tokio::test`, may run in CI with `cargo test --test`)

- `server_integration_tests.rs`
  - `test_client_server_roundtrip`: start server on ephemeral port; client registers node; `GetNode` returns registered node id.
  - `test_two_clients_agree`: 2 clients connect to 1 server; register different nodes; both query same key → same result.
  - `test_heartbeat_excludes_dead`: register with short interval; stop heartbeat; after timeout `GetNode` excludes node.

### 7.3 Behavior Verification (must match intended design)

| Intended behavior | Test that verifies |
|---|---|
| Lookup O(1) regardless of medium | `test_lookup_o1_time` + ring design doc (hash + nearest-vnode, not scan) |
| Node can hold both mem + disk | `test_re_register_updates_medium`; registry stores `Vec<Medium>` |
| Coordinator does NOT serve payload bytes | RPC proto review: `GetNodeResponse` only has `NodeRef`, no `bytes payload`; server never reads KV payload |
| HA possible via consensus in future | `ConsensusBackend` trait defined; `NoOpConsensus` stub; doc references `raft-consensus` / `zenoh_raft` |
| No custom ring code (reuse crate) | Source review: `ring.rs` imports `hash_rings`; no `fn hash_ring_custom` |

---

## 8. General Task List with Requirements

- [ ] **Req-A:** Crate compiles with `ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000` (no CUDA needed for coordinator — pure Rust / tokio).
- [ ] **Req-B:** SPDX `// SPDX-License-Identifier: AGPL-3.0-only` line 1 of every `.rs` / `.proto` / `.toml` comment.
- [ ] **Req-C:** File size cap ≤500 LoC per `.rs` (per `.github/workflows/file-size-cap.yml`). `server.rs` / `ring.rs` split if needed.
- [ ] **Req-D:** `deny(warnings); deny(clippy::all)` in `lib.rs` (per `AGENTS.md` ground rules).
- [ ] **Req-E:** `hash-rings` dependency pinned; document in `Cargo.toml` why not `consistent_hash_ring` (chosen for vnode replication support; verify at T1).
- [ ] **Req-F:** Consensus trait documented for future `raft-consensus` / `zenoh_raft` integration; no implementation until HA goal added.
- [ ] **Req-G:** Protocol (proto) versioned independently; `GetNodeResponse` must remain backward-compatible (only add fields, never rename/remove).

---

## 9. Decisions Log (per AAC / AGENTS.md)

| ID | Area | Decision | Why |
|---|---|---|---|
| D1 | Hash ring crate | `hash-rings` v1.1.0 | Consistent hashing + vnode replication; avoids custom ring code. |
| D2 | HA mechanism (future) | `raft-consensus` family / `zenoh_raft` | Existing crates provide election + replication; trait shell defers to them. |
| D3 | Transport | `tonic` gRPC (not raw TCP) | Existing in workspace patterns; protobuf serialization; easily extended to RDMA meta later. |
| D4 | Registry concurrency | `DashMap` (Phase 1); future `RwLock` under Raft | Phase 1 needs concurrent read/write; Raft will serialize writes via leader. |
| D5 | O(1) guarantee | Hash + nearest-vnode (not range partition) | Range partition is O(log n) worst-case; consistent hash is O(log V) where V=vnode count (constant for fixed ring size) — documented as O(1) for fixed config. |

---

## 10. Assumptions (per AGENTS.md assumption log pattern)

- Nodes register with a stable `node_id`; coordinator does not assign UUIDs.
- Network partition between node and coordinator = node considered `down` after 3 missed heartbeats; clients must handle fallback (not coordinator's job in Phase 1).
- Key space is opaque bytes (`Vec<u8>`); coordinator does not interpret key semantics.
- `hash-rings` v1.1.0 API is stable; if it breaks, upgrade path is to `lfchring` (lock-free) or `consistent_hash_ring`.
- No authentication on RPC in Phase 1 (same-network deployment); auth deferred to HA / cluster security layer.
- HA via Raft requires at least 3 coordinator instances; Phase 1 runs 1 instance.

---

---

## 11. Deferred: HA Consensus Design (Full Raft Integration, Not Phase 1)

*Status: deferred — trait shell (`ConsensusBackend`) exists (§5.4); implementation blocked until HA goal added to project scope. Design below so Phase-2 implementation does not redesign interface.*

### 11.1 Why deferred (scope clarification)
- HA is **not** a current goal (user instruction, §1).
- `raft-consensus` v0.4.0 / `raft-consensus-core` v0.1.2 / `zenoh_raft` v0.1.7 require multi-instance deployment (≥3), persistent log storage, and snapshot management — out of Phase-1 single-server scope.
- `ConsensusBackend` trait (§5.4) is the integration point; adding full implementation now would violate ponytail (write only what is needed; HA unneeded).

### 11.2 Design (future Phase 2) — based on `raft-consensus` / `zenoh_raft`

```
Phase-2 cluster (3 coordinator instances):
  Node C1 (leader)  ←─── Raft log replication ───→  Node C2 (follower)
        │                                             │
        └─────────── Raft log replication ────────────┘
        │                    ↑
  Registry writes only via leader; followers apply delta
```

**ConsensusBackend v2 (deferred interface expansion):**

```rust
#[async_trait::async_trait]
pub trait ConsensusBackend: Send + Sync {
    // Phase 1 (implemented as NoOp)
    fn is_leader(&self) -> bool;
    async fn replicate_delta(&self, delta: RegistryDelta) -> Result<(), Error>;
    async fn wait_for_quorum(&self) -> Result<()>;

    // Phase 2 (deferred — requires raft-consensus integration)
    async fn propose_registry_delta(&self, delta: RegistryDelta) -> Result<LogIndex>;  // leader proposes
    async fn apply_committed_delta(&self, index: LogIndex) -> Result<>;              // follower applies
    fn snapshot_registry(&self) -> RegistrySnapshot;                                 // periodic snapshot
    async fn restore_from_snapshot(&self, snap: RegistrySnapshot) -> Result<>;         // recovery
    fn leader_id(&self) -> Option<NodeId>;                                           // for client redirect
}
```

**Implementation approach (when deferred item activated):**
- Adopt `raft-consensus-core` v0.1.2 (lean core; election + replication + snapshot APIs) rather than `zenoh_raft` (larger, Zenoh transport dependency) unless mesh/network requires Zenoh.
- `NoOpConsensus` replaced by `RaftConsensus` wrapper around `raft_consensus::Raft` state machine.
- Registry (`DashMap` → `Arc<RwLock<RegistryState>>`) guarded by Raft log order: writes only on committed entries; reads allowed from committed state (not uncommitted leader proposal).
- Snapshot file: `snapshots/registry-<index>.bin` (bincode, same serde used for wire); truncated after 3 successful replicas.
- Client redirect: `GetNodeResponse` adds `redirect_to: Option<NodeId>` when queried on follower (future version of proto); client retries against leader.

### 11.3 Deferred tests (must be added when Phase 2 starts)

- `test_raft_election_3_nodes`: start 3 instances; kill leader; new leader elected within `election_timeout` (default 150ms); write succeeds.
- `test_raft_log_replicate_delta`: register on leader; follower applies same delta; query on follower after apply returns same node.
- `test_raft_snapshot_restore`: inject 500 deltas; trigger snapshot; restore follower from snapshot; registry consistent.
- `test_client_redirect_to_leader`: query follower; receive `redirect_to`; retry leader; success.
- `test_raft_quorum_two_down`: 3 nodes, 2 down; writes fail (not hang); reads from committed state still succeed.

### 11.4 Risks / constraints deferred to Phase 2
- Persistent log storage path (default: `/var/lib/atlas-kv-coord/log`) must be on NVMe (same as `spark-storage` disk tier) to avoid Raft bottleneck.
- Snapshot I/O must not block `GetNode` — run in `tokio::spawn` background.
- `hash-rings` ring state must be reconstructed from committed registry on each follower restart (from snapshot + replay).

---

## 12. Deferred: Testing (Benchmark + Failure + Auth)

*Status: partial (§7 has unit + integration); deferred below are the tests required for production but not needed for Phase-1 correctness proof.*

### 12.1 Performance / O(1) benchmark (deferred — log-only, not CI-gated)

**Test:** `bench_lookup_latency.rs` (separate `benches/` directory, not `tests/` so `cargo test` doesn't run it by default).

```rust
fn bench_lookup_1024_vnodes(c: &mut Criterion) {
    let ring = build_ring(10_nodes, 1024_vnodes);  // 10240 vnodes total
    c.bench_function("lookup_1024_vnodes", |b| {
        b.iter(|| ring.lookup(b"test-key-123"));
    });
}
```

**Requirements (deferred — must hold when benchmark run):**
- Mean < 1µs; p99 < 5µs (single-threaded, tokio current-thread).
- Throughput > 500K lookups/sec per core (defer until server under load).
- Memory: ring with 10 nodes × 1024 vnodes ≈ 10K vnode entries ≈ <2MB (measure, document).
- Regression guard: if mean > 2µs, flag for `hash-rings` upgrade to `lfchring`.

### 12.2 Failure-mode tests (deferred — Phase 1 assumes healthy network)

- `test_split_brain_no_ha`: 2 coordinator instances (not Raft-joined), both accept registrations with same `node_id`; document that Phase 1 does NOT handle split-brain (design limitation, not bug — HA required to resolve).
- `test_heartbeat_false_negative`: network latency > heartbeat interval; node temporarily `down`; after latency drops, heartbeat succeeds; ring reincludes node. Verify no duplicate vnode entries.
- `test_key_not_found_empty_ring`: `GetNode` on empty ring → `Error::RingEmpty`; client must fall back to local media (documented in `client.rs` doc comment, not enforced by server).
- `test_node_crash_during_register`: connection drops mid-Register; registry should not contain partial entry (atomic `DashMap::insert` is atomic, but document that `register` is all-or-none — no partial-state needed for 3-field struct).

### 12.3 Auth / security tests (deferred — Phase 1 same-network, no auth per §10)

When auth layer added (future, possibly via `tonic` interceptors + TLS + mTLS):

- `test_unauth_register_rejected`: client without valid token → `Register` returns `Status::unauthenticated`; registry unchanged.
- `test_token_expiry_heartbeat_rejected`: expired token → heartbeat fails; node marked down; re-register with new token succeeds.
- `test_tls_mtls_connection`: `client.connect("https://...", tls_config)` succeeds only with correct cert chain.

### 12.4 Data-medium integration tests (deferred — coordinator does not serve bytes, but must know medium)

- `test_medium_change_in_flight`: node registers Mem; during lookup, node updates to Disk; `GetNode` after update returns Disk — no stale medium.
- `test_both_mediums_same_key`: key exists on both Mem and Disk at same node; `GetNode` returns both (client decides which to use); registry stores `Vec<Medium>` not single value.

---

## 13. Deferred: Crate Integrations & Fallbacks

*Status: `hash-rings` + `tonic` selected (§2); below are deferred integrations if selected crates break or requirements change.*

### 13.1 Hash ring fallback (if `hash-rings` v1.1.0 fails / breaks)

| Trigger | Fallback | Migration effort |
|---|---|---|
| `hash-rings` API removed / unreleased | `lfchring` v0.1.3 (lock-free concurrent ring) | Low: `lrchring::Ring` has `add`/`remove`/`get`; wrapper `CoordRing` abstracts both. |
| `hash-rings` performance regresses >2µs mean | `consistent_hash_ring` v0.8.0 (simpler API) | Medium: different vnode model; may need to adjust `vnode_count` config. |
| Need lock-free concurrent ring (many concurrent clients) | `lfchring` v0.1.3 (explicitly concurrent, no `Mutex`) | Low: same wrapper. |

**Deferred action:** keep `hash-rings` in `Cargo.toml`; add `lfchring` as `optional = true`; only activate if benchmark (§12.1) flags regression or CI build breaks.

### 13.2 Distributed-cache reference pattern (if coordinator model needs cache-filling)

`groupcache` v0.3.0 provides groupcache-style cache filling (key not found → compute + fill). **Not adopted** — our coordinator is metadata-only, not a cache server. If future requirement is "coordinator can serve KV when node down," then evaluate `groupcache` as replacement (would change architecture from metadata-only to payload-serving; major design change, deferred to separate design doc).

`redhac` v0.10.5 (embedded HA cache) — deferred: if Phase-2 needs embedded (not external) HA cache with replication, evaluate `redhac` before writing custom replication layer.

### 13.3 Consensus crate decision deferred (when Phase 2 starts)

| Crate | Pros | Cons | Decision trigger |
|---|---|---|---|
| `raft-consensus` v0.4.0 | Complete, mature, well-documented | Slightly larger dependency tree | Default for Phase 2 |
| `raft-consensus-core` v0.1.2 | Lean core only; easy to wrap | Less docs; need to build election logic | If binary size critical |
| `zenoh_raft` v0.1.7 | Integrated with Zenoh mesh (useful if nodes already on Zenoh) | Adds Zenoh transport dependency; overkill if only TCP/gRPC | Only if workspace adopts Zenoh |
| `photon-etcd-cluster` v0.1.0 | Uses etcd (familiar ops) | Requires etcd cluster as separate service | If ops team requires etcd |

**Deferred design note:** `ConsensusBackend` trait (§5.4 / §11) abstracts all four — implementation choice doesn't affect `ring.rs` / `registry.rs` / `server.rs`.

### 13.4 Transport / wire format deferred choices

- Current (Phase 1): `tonic` gRPC + `prost` + JSON meta over TCP.
- Future (if RDMA meta needed): `atlas-rdma` one-sided READ for registry snapshots (not key lookup — lookup is too small for RDMA overhead); meta transport stays gRPC.
- Future (if high-throughput many clients): consider `tarpc` or raw TCP with `bincode`; deferred until benchmark (§12.1) shows gRPC bottleneck.

---

## 14. Deferred: Design / Architecture Decisions (Not Yet Committed)

*Status: listed in §9; below expanded with tradeoffs and when to decide.*

### 14.1 D3 — Transport (expanded, deferred)
**Committed:** `tonic` gRPC.
**Not committed:** TLS/mTLS, auth interceptor pipeline, RDMA meta, raw TCP fallback.
**Decision trigger:** when first external node (non-same-network) connects, or when `benchmark_lookup_latency` shows gRPC overhead > 20% of lookup time.

### 14.2 D4 — Registry concurrency (expanded, deferred)
**Committed Phase 1:** `DashMap` (concurrent read/write, no external locking needed for single instance).
**Not committed Phase 2:** `Arc<RwLock<RegistryState>>` under Raft (writes serialized by leader; followers apply sequentially).
**Migration:** `DashMap` → `RwLock` is localized to `registry.rs`; `server.rs` uses `Arc<Registry>` and doesn't care about inner lock type. Migration when Phase 2 activates is ~30 lines.

### 14.3 D5 — O(1) guarantee (expanded, deferred clarification)
**Committed:** consistent hash ring; lookup = hash(key) + nearest-vnode binary search.
**Clarified / deferred:** "O(1)" assumes fixed vnode count; actual complexity is O(log V) where V = total vnodes (constant per deployment). If vnode count scaled with node count, lookup would be O(log N). Design constraint: `vnode_count` per node is fixed in config (default 100, max 1024); not auto-scaled. Documented in `ring.rs` doc comment; test `test_lookup_o1_time` verifies fixed-config behavior.

### 14.4 D6 — Auth / security (new deferred decision not in §9)
**Status:** not decided.
**Options:** (a) none (Phase 1 — documented assumption §10); (b) TLS only; (c) mTLS + bearer token via `tonic` interceptor; (d) integration with existing `atlas-governance` (if workspace has governance/auth already — inspect `atlas-governance/src/lib.rs` for auth patterns).
**Trigger:** first external network deployment.
**Recommendation (lazy):** (b) TLS first (operator-configurable, no code change to `server.rs` — `tonic` accepts TLS config); (c) only if compliance requires it.

### 14.5 D7 — Snapshot / persistence format (new deferred decision)
**Status:** not decided.
**Options:** (a) `bincode` (same as wire — simplest); (b) `serde_json` (human-readable for ops debugging); (c) `rmp-serde` (MessagePack — compact, fast).
**Trigger:** Phase 2 Raft snapshot implementation.
**Recommendation:** `bincode` (lazy — already in workspace; no new dependency; human-readable not needed for automated recovery).

### 14.6 D8 — Node medium discovery (new deferred decision)
**Status:** not decided.
**Current:** node explicitly registers medium (`Vec<Medium>`).
**Alternative:** coordinator probes node via gRPC `Probe` RPC; auto-detects in-mem vs disk from node's response.
**Trigger:** if node's medium changes frequently (e.g., evicted from disk, reloaded to mem) and registration updates become noisy.
**Recommendation (lazy):** keep explicit registration (simpler, no probe protocol needed); add `UpdateNode` RPC for medium changes (§5.3).

---

## 15. Deferred Items — Master List (Cross-Reference)

| ID | Section | Item | Trigger / Condition | Effort estimate |
|---|---|---|---|---|
| F1 | 11 | Full Raft consensus implementation (`ConsensusBackend` v2) | HA goal added to scope | Medium-High (1–2 weeks) |
| F2 | 11 | Raft tests (`test_raft_election_*`, `test_raft_snapshot_*`) | F1 activates | Low (days) |
| F3 | 12.1 | Benchmark `bench_lookup_latency` + regression guard | Pre-release performance validation | Low (hours) |
| F4 | 12.2 | Failure-mode tests (split-brain, false-negative, empty ring) | Production readiness / chaos testing | Low–Medium |
| F5 | 12.3 | Auth / TLS / mTLS tests | External-network deployment | Low–Medium |
| F6 | 12.4 | Medium-change + dual-medium tests | Multi-tier cache (mem + disk) in production | Low |
| F7 | 13.1 | `lfchring` / `consistent_hash_ring` fallback integration | `hash-rings` breaks or regresses | Low (wrapper abstracted) |
| F8 | 13.2 | `groupcache` / `redhac` evaluation (if coordinator needs payload) | Requirement change to "serve KV on node failure" | High (architecture change) |
| F9 | 13.3 | Consensus crate selection (`raft-consensus` vs `zenoh_raft` vs `photon-etcd`) | F1 activates | Low (trait abstracts) |
| F10 | 13.4 | Transport alternative (raw TCP / RDMA meta) | Performance bottleneck shown in F3 | Low–Medium |
| F11 | 14.1 | Transport design (TLS / RDMA meta / auth) | External network | Low |
| F12 | 14.2 | Registry concurrency migration (`DashMap` → `RwLock`) | F1 activates | Low |
| F13 | 14.3 | O(1) clarification (fixed vnode = O(log V), documented) | Documentation review | Already done (§14.3) |
| F14 | 14.4 | Auth / security layer design | External network | Low (decision; implementation medium) |
| F15 | 14.5 | Snapshot format (`bincode` / JSON / MessagePack) | F1 activates | Low |
| F16 | 14.6 | Auto-discovery of medium vs explicit register | Frequent medium changes observed | Low |

**Lazy principle applied:** all deferred items are documented with trigger conditions so they activate only when needed; no speculative implementation in Phase 1. No custom code written for any deferred item — only design + test specs + integration points (`ConsensusBackend` trait, wrapper abstraction for ring, `UpdateNode` RPC for D8).

---

## 16. Deferred Plan Gaps — Ops / Security / Cost / ADR / Versioning

*Status: not implemented in Phase 1; covered here so deployment / security review can proceed without redesigning the crate.*

### 16.1 Operational Runbook (deferred — deploy / start / restart / monitor)

**Deployment (reference `ATLAS_SPARK_JOURNEY.md`, `GB10_DEPLOYMENT_GUIDE.md` patterns):**
- Binary: `atlas-kv-coord` (standalone daemon, not linked into `spark-server`). Deployed via Docker image `azeezish/atlas-gb10:latest` (same tag as inference stack) — add `crates/atlas-kv-coord/` to image build, expose port 50051 (tonic default / configurable via `config.rs`).
- Config file: `/etc/atlas/kv-coord.toml` (bind, vnode_count, heartbeat_interval, snapshot_path, log_path). Not JSON — TOML is existing workspace convention (see `MODEL.toml` in kernels).
- Start order: coordinator starts before worker nodes; workers retry `RegisterNode` with exponential backoff (250ms → 2s cap) rather than fail-on-start.
- Restart / upgrade: `GetNode` clients retry on `UNAVAILABLE`; coordinator restart loses registry (Phase 1 — no HA) — clients fall back to local media (§10 assumption). **Documented limitation, not a bug.**
- Monitor / metrics (deferred — add `metrics` crate or `prometheus` when deploy needs it): expose `kv_coord_lookups_total`, `kv_coord_nodes_registered`, `kv_coord_heartbeat_misses`, `kv_coord_ring_rebalance_duration` on `/metrics`. Not in Phase 1 code; interface reserved (`metrics::register_counter!` call site in `server.rs` comments).

### 16.2 Security Threat Model (deferred — Phase 1 same-network, no auth; expanded here for review)

| Threat | Phase 1 status | Mitigation (deferred / documented) |
|---|---|---|
| Unauthenticated `RegisterNode` from rogue node | **Present** (no auth) | Documented (§10); mitigation deferred to 14.4 (TLS + bearer token via `tonic` interceptor) |
| Man-in-the-middle on gRPC | **Present** (TCP, no TLS) | Deferred: `tonic` `tls` feature + mTLS (§14.1); operator-configurable |
| Registry permutation (attacker injects fake node id) | **Present** | Mitigation: node id derived from host identity (TLS cert CN or node fingerprint) — deferred to auth layer; not enforced by `registry.rs` |
| DoS via huge `key` payload to `GetNode` | Partial | `GetNodeRequest.key` has max length guard (default 4KB / configurable); server rejects with `INVALID_ARGUMENT`; no memory exhaustion |
| Snapshot file tampering (Phase 2) | N/A (Phase 1) | Phase 2: snapshot written to `snapshots/` with `0600` perms; verify hash before restore (§14.5) |
| Split-brain (2 coordinators, no Raft) | **Present** (§12.2) | Documented limitation: Phase 1 single instance only; HA via Raft (§11) resolves |

**Security review gate (deferred — activate before external-network deploy):** review `proto/kv_coord.proto` for information disclosure (`NodeRef` exposes endpoints — acceptable for same-network; may need redaction in public-facing deploy); add `tonic` TLS config to `Cargo.toml` (feature-gated); add `metrics` exposure; verify `snapshots/` directory permissions.

### 16.3 Cost / Infra Sizing (deferred — estimate for capacity planning)

| Resource | Phase 1 estimate | Scaling factor | Notes |
|---|---|---|---|
| Coordinator memory | ~10 nodes × 1024 vnodes × ~200B/vnode ≈ 2MB ring + 1MB registry ≈ **<10MB** | Linear with nodes (not keys) | Key-space size does NOT affect coordinator (metadata-only) |
| Coordinator CPU | 1 core per 100K lookups/sec (deferred benchmark F3 validates) | Linear with clients | gRPC overhead dominant; if >1M/sec, shard coordinators per key range (out of Phase 1 scope) |
| Coordinator disk (logs + snapshots Phase 2) | Log: ~100MB/day (heartbeat + registration events); Snapshot (Phase 2): ~10MB per 500-delta interval | Log grows with nodes (heartbeat rate); snapshot with delta rate | Phase 1: logs optional (`tracing`); Phase 2: snapshot directory required |
| Network (per lookup) | ~200 bytes (request + `NodeRef`) | Per client lookup | Negligible compared to KV payload fetch (which coordinator does NOT serve — payload fetch is node-to-node, not through coordinator) |

**Lazy sizing note:** coordinator is metadata-only → cost is negligible vs inference nodes; primary cost driver is not this crate but the `hash-rings` lookup overhead it eliminates (replacing range-scan with O(1) saves client CPU). No separate infra tier needed for Phase 1.

### 16.4 ADR — Architecture Decision Record (new, deferred — add to `docs/adr/` when crate merges)

File: `docs/adr/0017-kv-cache-coordinator.md` (next sequential; current max is 0016 per `ls docs/adr/` — verify before writing).

```markdown
# ADR-0017: Remote KV-Cache Coordinator (`atlas-kv-coord`)

Status: Proposed (not yet accepted — Phase 1 design only)
Date: 2026-05-29

## Context
Spark-runtime / spark-storage use local NVMe + in-memory KV caches per node.
Cross-node access requires knowing which node holds which KV block.
Range-scan lookup is O(log n) and scales poorly with node count.

## Decision
- External coordinator crate (`atlas-kv-coord`) using consistent hash ring (`hash-rings`).
- O(1) lookup via hash + nearest-vnode (fixed vnode count).
- Metadata-only (no payload serving); nodes register medium (Mem / Disk).
- HA deferred to Phase 2 via `ConsensusBackend` trait + `raft-consensus`.

## Consequences
- Positive: lookup becomes O(1); node-local media preserved; no custom ring code.
- Negative: Phase 1 single point of failure (documented; no HA); client must handle coordinator-down fallback (local media); new external service to deploy.
- Risk: if `hash-rings` breaks, migrate to `lfchring` (wrapper abstracts).

## Alternatives considered
- Embedded coordinator (no external server) — rejected: requires every node to hold full registry; harder to upgrade.
- Range partition (sorted node list) — rejected: O(log n) lookup; rebalance expensive.
- Custom consistent hash — rejected: `hash-rings` v1.1.0 exists; ponytail prevents rewrite.
```

**Lazy: ADR written when crate merges to main (not now — avoids doc drift from unimplemented design).**

### 16.5 Versioning / Backward-Compatibility (deferred — proto & ring)

| Element | Phase 1 | Phase 2 / future |
|---|---|---|
| Proto version | v1 (no version field needed — new service) | Add `version: int32` to `GetNodeResponse`; clients ignore unknown fields (protobuf default) |
| `GetNodeResponse` fields | `node_id`, `mediums[]`, `endpoints[]` | Add `redirect_to: string` (for Raft follower redirect); `ttl_seconds` (cache entry TTL — deferred) |
| `NodeRef` ring encoding | Not serialized externally (internal to `ring.rs`) | If snapshot needs ring state: serialize vnode list (not needed — registry rebuilds ring from registered nodes) |
| Ring rebalance | On register / remove / heartbeat `down` | Phase 2: rebalance only on committed log entries (Raft ensures consistency) |
| Rollback procedure | Restart coordinator from clean state; clients fall back to local (documented limitation) | Phase 2: restore from snapshot (`snapshots/`); replay log |

**Rule:** never rename/remove fields from `GetNodeResponse`; only add optional fields. Documented in `.proto` comment and `client.rs` doc.

---

## 17. What the Plan Now Covers (Post-Expansion Verification)

| Dimension | Covered in | Section |
|---|---|---|
| Design (high-level + detail) | Yes | 3, 5 |
| Crate structure | Yes | 4 |
| Search results / crate selection | Yes | 2 |
| Implementation details (ring, registry, server, proto) | Yes | 5 |
| Core tasks + AC | Yes | 6 |
| Testing (unit + integration + specific test names) | Yes | 7 |
| Task list + requirements (Req A–G) | Yes | 8 |
| Decisions (D1–D5 + D6–D8) | Yes | 9, 14 |
| Assumptions | Yes | 10 |
| HA / consensus design (deferred) | Yes | 11 |
| Deferred testing (benchmark + failure + auth) | Yes | 12 |
| Deferred crate fallbacks | Yes | 13 |
| Ops / runbook / deploy | **Yes (added)** | 16.1 |
| Security threat model | **Yes (added)** | 16.2 |
| Cost / infra sizing | **Yes (added)** | 16.3 |
| ADR entry (deferred) | **Yes (added)** | 16.4 |
| Versioning / backward-compat / rollback | **Yes (added)** | 16.5 |
| Master deferred list (F1–F16) | Yes | 15 |

No custom ring/consensus implementation written; all deferred items have triggers (F1–F16). Plan file: `docs/plans/atlas-kv-coord-plan.md` (477 + ~200 new lines = ~680 total — verify with `wc -l` after append; below confirms).
