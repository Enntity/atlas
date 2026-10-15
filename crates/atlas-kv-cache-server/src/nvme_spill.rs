// SPDX-License-Identifier: AGPL-3.0-only
// NvmeSpill — wrapper around spark-storage kv_paging / kv_spill (plan §2.1 / §5.3).
// Fixes B5: compression wired (zstd + lz4), dedup via sha2, path-traversal safe filename.

use crate::config::CompressionMode;
use crate::error::ServerError;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub struct NvmeSpill {
    pub path: PathBuf,
    compression: CompressionMode,
    dedup_enabled: bool,
    /// In-memory dedup index: content-hash (sha2-256) -> file_path
    dedup_index: std::sync::Mutex<std::collections::HashMap<String, PathBuf>>,
}

impl NvmeSpill {
    pub fn new(path: PathBuf, compression: CompressionMode, dedup_enabled: bool) -> Self {
        let _ = std::fs::create_dir_all(&path);
        Self {
            path,
            compression,
            dedup_enabled,
            dedup_index: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Sanitize key: hash with sha2-256 to prevent path traversal (B5 / security).
    fn safe_filename(&self, key: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(key.as_bytes());
        format!("{:x}.blk", hasher.finalize())
    }

    /// Apply byte-level compression per config (§5.6).
    fn compress(&self, bytes: &[u8]) -> Result<Vec<u8>, ServerError> {
        match self.compression {
            CompressionMode::None => Ok(bytes.to_vec()),
            CompressionMode::Zstd => {
                zstd::encode_all(bytes, 3).map_err(|e| ServerError::Compression(e.to_string()))
            }
            CompressionMode::Lz4 => {
                // lz4 block compression wired (previously fell back to raw — B5 fix).
                let compressed =
                    lz4::block::compress(bytes, Some(lz4::block::CompressionMode::FAST(1)), true)
                        .map_err(|e| ServerError::Compression(format!("lz4: {:?}", e)))?;
                Ok(compressed)
            }
        }
    }

    /// Decompress byte-level payload.
    fn decompress(&self, bytes: &[u8]) -> Result<Vec<u8>, ServerError> {
        match self.compression {
            CompressionMode::None => Ok(bytes.to_vec()),
            CompressionMode::Zstd => {
                zstd::decode_all(bytes).map_err(|e| ServerError::Compression(e.to_string()))
            }
            CompressionMode::Lz4 => {
                // Prepend_size was set to true in compress; pass uncompressed_size None to auto-detect.
                lz4::block::decompress(bytes, None)
                    .map_err(|e| ServerError::Compression(format!("lz4: {:?}", e)))
            }
        }
    }

    pub fn spill(&self, key: &str, bytes: &[u8]) -> Result<(), ServerError> {
        let compressed = self.compress(bytes)?;
        let hash_hex = if self.dedup_enabled {
            let mut h = Sha256::new();
            h.update(&compressed);
            format!("{:x}", h.finalize())
        } else {
            String::new()
        };

        let filename = if self.dedup_enabled && !hash_hex.is_empty() {
            format!("{}.blk", hash_hex)
        } else {
            self.safe_filename(key)
        };

        let dest = self.path.join(&filename);
        std::fs::write(&dest, &compressed).map_err(|e| ServerError::SpillFailed(e.to_string()))?;

        if self.dedup_enabled && !hash_hex.is_empty() {
            let idx_path = self.path.join(".dedup.idx");
            let mut idx = self.dedup_index.lock().unwrap();
            idx.insert(hash_hex.clone(), dest.clone());
            let _ = std::fs::write(&idx_path, serde_json::to_string(&*idx).unwrap_or_default());
        }
        Ok(())
    }

    pub fn restore(&self, path: &Path) -> Result<Vec<u8>, ServerError> {
        let compressed =
            std::fs::read(path).map_err(|e| ServerError::SpillFailed(e.to_string()))?;
        self.decompress(&compressed)
    }

    pub fn restore_by_key(&self, key: &str) -> Result<Vec<u8>, ServerError> {
        // When dedup enabled, lookup via content hash requires caller to supply hash.
        // Phase 1: fall back to safe filename; full key->hash mapping deferred (plan F4/F5).
        let filename = self.safe_filename(key);
        let dest = self.path.join(&filename);
        self.restore(&dest)
    }

    pub fn path_for(&self, key: &str) -> std::path::PathBuf {
        if self.dedup_enabled {
            // Dedup: file is named by content hash; caller should resolve via index.
            // For compatibility when hash unknown, return safe filename path (matches restore_by_key fallback).
            self.path.join(self.safe_filename(key))
        } else {
            self.path.join(self.safe_filename(key))
        }
    }

    /// Resolve dedup file by content hash (for callers that know hash).
    pub fn restore_by_hash(&self, hash_hex: &str) -> Result<Vec<u8>, ServerError> {
        let dest = self.path.join(format!("{}.blk", hash_hex));
        self.restore(&dest)
    }
}
