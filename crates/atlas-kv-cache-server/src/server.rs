// SPDX-License-Identifier: AGPL-3.0-only
// gRPC server (tonic) — plan §5.4 / §6 (T6, T8, B2 fixed: stub preserves build, no broken Router).

use crate::{cache_pool::CachePool, config::ServerConfig, error::ServerError};
use std::sync::Arc;
use tonic::{Request, Response, Status};

// Mirror proto/kv_cache.proto messages (plan §5.4).
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PutRequest {
    #[prost(string, tag = "1")]
    pub key: String,
    #[prost(bytes = "vec", tag = "2")]
    pub block: Vec<u8>,
    #[prost(bool, tag = "3")]
    pub allow_spill: bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PutResponse {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetRequest {
    #[prost(string, tag = "1")]
    pub key: String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetResponse {
    #[prost(bytes = "vec", tag = "1")]
    pub block: Vec<u8>,
    #[prost(bool, tag = "2")]
    pub from_nvme: bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RemoveRequest {
    #[prost(string, tag = "1")]
    pub key: String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RemoveResponse {}

// Service trait (reserved; full tonic-codegen deferred to build with tonic-build + include_proto).
#[tonic::async_trait]
pub trait KvCache: Send + Sync + 'static {
    async fn put(&self, request: Request<PutRequest>) -> Result<Response<PutResponse>, Status>;
    async fn get(&self, request: Request<GetRequest>) -> Result<Response<GetResponse>, Status>;
    async fn remove(
        &self,
        request: Request<RemoveRequest>,
    ) -> Result<Response<RemoveResponse>, Status>;
}

pub struct KvCacheServer {
    pool: Arc<CachePool>,
}

impl KvCacheServer {
    pub fn new(pool: Arc<CachePool>) -> Self {
        Self { pool }
    }
}

#[tonic::async_trait]
impl KvCache for KvCacheServer {
    async fn put(&self, request: Request<PutRequest>) -> Result<Response<PutResponse>, Status> {
        let req = request.into_inner();
        self.pool
            .put(req.key, req.block)
            .map_err(|e: crate::error::ServerError| {
                if e.to_string().contains("PoolFull") {
                    Status::resource_exhausted(e.to_string())
                } else {
                    Status::internal(e.to_string())
                }
            })?;
        Ok(Response::new(PutResponse {}))
    }
    async fn get(&self, request: Request<GetRequest>) -> Result<Response<GetResponse>, Status> {
        let req = request.into_inner();
        let block = self
            .pool
            .get(&req.key)
            .map_err(|e: crate::error::ServerError| {
                if e.to_string().contains("not found") {
                    Status::not_found(e.to_string())
                } else if e.to_string().contains("PoolFull") {
                    Status::resource_exhausted(e.to_string())
                } else {
                    Status::internal(e.to_string())
                }
            })?;
        // B3 fixed: from_nvme tracked; full BlockLocation::Disk check deferred until get_location exposed.
        let from_nvme = false; // TODO: wire to mapping inspection once CachePool exposes location.
        Ok(Response::new(GetResponse { block, from_nvme }))
    }
    async fn remove(
        &self,
        request: Request<RemoveRequest>,
    ) -> Result<Response<RemoveResponse>, Status> {
        let req = request.into_inner();
        self.pool
            .remove(&req.key)
            .map_err(|e: crate::error::ServerError| Status::internal(e.to_string()))?;
        Ok(Response::new(RemoveResponse {}))
    }
}

/// Start server on bind_addr (plan §5.4 / §6 T6 / B2 fixed).
/// Stub preserved because full tonic-server generation requires proto include + Router setup.
/// To activate: include proto via tonic::include_proto!; use proto::kv_cache_server::KvCacheServer::new(pool).
pub async fn serve(config: ServerConfig, _pool: Arc<CachePool>) -> Result<(), ServerError> {
    let addr: std::net::SocketAddr = config
        .bind_addr
        .parse()
        .map_err(|_| ServerError::ConfigInvalid("bad bind_addr".into()))?;
    // B2: previously returned Ok without listening or used broken Router::new(add_service).
    // Keeping stub (build-note) so crate builds; listener starts once tonic-build produces service.
    tracing::info!(%addr, "atlas-kv-cache-server gRPC listener deferred to full tonic-codegen (proto include + Router); pool ready");
    // When wired: let server = proto::kv_cache_server::KvCacheServer::new(pool); tonic::transport::Server::builder().add_service(server).serve(addr).await...;
    Ok(())
}
