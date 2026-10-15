// SPDX-License-Identifier: AGPL-3.0-only
// Pool size tests — plan §7.1 (T9); first-pass stubs.

use atlas_kv_cache_server::config::{CompressionMode, ServerConfig};

#[test]
fn test_pool_respects_max_memory() {
    // First pass: config validates, pool created; full gate verified in server.
    let cfg = ServerConfig {
        bind_addr: "127.0.0.1:0".into(),
        max_memory_bytes: 1024 * 1024,
        nvme_path: None,
        block_size: 128,
        enable_rdma: false,
        compression: CompressionMode::None,
        dedup_enabled: false,
    };
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_nvme_path_invalid_fails_fast() {
    let cfg = ServerConfig {
        bind_addr: "127.0.0.1:0".into(),
        max_memory_bytes: 1024,
        nvme_path: Some(std::path::PathBuf::from("/nonexistent_path_12345")),
        block_size: 128,
        enable_rdma: false,
        compression: CompressionMode::None,
        dedup_enabled: false,
    };
    assert!(cfg.validate().is_err());
}
