// SPDX-License-Identifier: AGPL-3.0-only
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum ServerError {
    #[error("pool full: max_memory_bytes exceeded and no nvme spill configured")]
    PoolFull,
    #[error("key not found: {0}")]
    KeyNotFound(String),
    #[error("spill failed: {0}")]
    SpillFailed(String),
    #[error("NVMe path invalid: {0}")]
    NvmePathInvalid(String),
    #[error("config invalid: {0}")]
    ConfigInvalid(String),
    #[error("compression error: {0}")]
    Compression(String),
    #[error("dedup index error: {0}")]
    Dedup(String),
}
