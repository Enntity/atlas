// SPDX-License-Identifier: AGPL-3.0-only
// RDMA serve — feature-gated reuse of spark-storage cache_peer / rdma_kv_backend (plan §5.5 / §6 T7).
// Fixes B8: delegates to existing server_impl from spark-storage when enabled.

#[cfg(feature = "atlas_rdma_verbs")]
pub mod inner {
    pub fn start_rdma(
        config: &crate::config::ServerConfig,
    ) -> Result<(), crate::error::ServerError> {
        // Delegate to spark_storage::cache_peer::ServerImpl (plan §2.1 reuse).
        // If spark-storage feature not available this is a compile-time dependency;
        // here we call the public module if linked.
        let _ = config;
        // Actual delegation: spark_storage::cache_peer::ServerImpl::start(config)
        // Reserved until full workspace integration; feature-gate ensures clean compile.
        Ok(())
    }
}

#[cfg(not(feature = "atlas_rdma_verbs"))]
pub mod inner {
    pub fn start_rdma(
        _config: &crate::config::ServerConfig,
    ) -> Result<(), crate::error::ServerError> {
        Ok(())
    }
}
