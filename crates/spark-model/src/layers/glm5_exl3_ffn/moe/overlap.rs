// SPDX-License-Identifier: AGPL-3.0-only
//! `ATLAS_EXL3_SHARED_OVERLAP=1`: run the MoE shared expert on a side stream
//! concurrently with the routed projections. The shared expert is built bound
//! to the side stream (same kernels, same scratch, same inputs: bit-exact);
//! the layer forks it after the router (so no cuBLAS work of the main stream
//! overlaps it) and joins before its output (scratch buffers[6]) is read.
//! Fork/join are event record/wait pairs, which stream capture records as
//! graph dependencies.
//!
//! Aliasing: the side stream reads only the layer input (read-only on the
//! main stream until the join) and writes its own FFN scratch and buffers[6];
//! the routed path never touches buffers[6] before the combine, which runs
//! after the join. The next layer's side work waits on that layer's fork,
//! recorded on the main stream after this layer's combine.
//!
//! Capture: the fork event is recorded on the (capturing) main stream, so the
//! side stream joins that capture and its work is captured as cross-stream
//! nodes — the same event fork/join pattern the TP communicator uses under
//! capture. That puts everything the side stream would otherwise create lazily
//! (its own classic-BLAS handle and workspace) INSIDE the capture, which is why
//! `overlap_side_stream` primes it at load time.
use super::*;
use std::sync::OnceLock;

pub(in crate::layers::glm5_exl3_ffn) fn shared_overlap() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_EXL3_SHARED_OVERLAP").as_deref() == Ok("1"))
}

/// One non-blocking side stream per main compute stream, shared by every
/// layer (layers are sequential on the main stream). Process lifetime.
///
/// Model-load time only (`load.rs` binds the shared expert with it), which is
/// why the stream-cache lock may span the prime below: nothing else asks for
/// the side stream while a model is being built.
pub(in crate::layers::glm5_exl3_ffn) fn overlap_side_stream(
    gpu: &dyn GpuBackend,
    main: u64,
) -> Result<u64> {
    static STREAMS: OnceLock<std::sync::Mutex<std::collections::HashMap<u64, u64>>> =
        OnceLock::new();
    let mut streams = STREAMS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(&side) = streams.get(&main) {
        return Ok(side);
    }
    let side = gpu.create_stream()?;
    // Prime BEFORE publishing the stream: a prime failure must not leave a
    // half-ready side stream for the layers to fork onto. Failed streams remain
    // backend-owned until teardown: GpuBackend has no stream-release API.
    prime_side_stream_blas(gpu, side)?;
    streams.insert(main, side);
    Ok(side)
}

/// Shape of the side-stream prime GEMM [rows, columns, inner]; trivial on
/// purpose, same order as the 64x64x64 GEMM `spark_runtime::cublaslt::prewarm`
/// uses to force the cuBLASLt path's own first-use cost.
const PRIME_SHAPE: [u32; 3] = [64, 64, 64];
/// Bytes of one BF16 `PRIME_SHAPE[0] x PRIME_SHAPE[1]` operand.
const PRIME_BYTES: usize = PRIME_SHAPE[0] as usize * PRIME_SHAPE[1] as usize * 2;

/// Initialize the per-stream classic-cuBLAS handle and workspace at model load.
/// Their lazy allocation during the first GEMM would invalidate an active
/// capture reached through the fork event. Scratch operands live until the
/// prime finishes; the pool retains its per-stream workspace for its lifetime.
/// Allocate transactionally: a later allocation failure releases prior operands.
fn allocate_prime_operands(
    mut alloc: impl FnMut() -> Result<DevicePtr>,
    mut free: impl FnMut(DevicePtr) -> Result<()>,
) -> Result<[DevicePtr; 3]> {
    let mut operands = Vec::with_capacity(3);
    for _ in 0..3 {
        match alloc() {
            Ok(ptr) => operands.push(ptr),
            Err(error) => {
                for ptr in operands {
                    let _ = free(ptr);
                }
                return Err(error);
            }
        }
    }
    Ok([operands[0], operands[1], operands[2]])
}

fn prime_side_stream_blas(gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
    let prime = (|| -> Result<()> {
        let [input, weight, output] =
            allocate_prime_operands(|| gpu.alloc(PRIME_BYTES), |ptr| gpu.free(ptr))?;
        let gemm = gpu.bf16_gemm_act_weight_t(spark_runtime::gpu::Bf16Gemm {
            numerics: spark_runtime::gpu::Bf16GemmNumerics::Glm53Classic,
            input,
            weight,
            output,
            rows: PRIME_SHAPE[0],
            columns: PRIME_SHAPE[1],
            inner: PRIME_SHAPE[2],
            stream,
        });
        // Even an error may follow partial enqueue. Never free operands until
        // the stream drains; on sync failure retain them for backend teardown.
        gpu.synchronize(stream)?;
        let mut cleanup = None;
        retain_error(&mut cleanup, gpu.free(input));
        retain_error(&mut cleanup, gpu.free(weight));
        retain_error(&mut cleanup, gpu.free(output));
        gemm.and(finish(cleanup))
    })();
    prime.context("EXL3 shared-overlap side-stream BLAS prime failed")
}

pub(in crate::layers::glm5_exl3_ffn) struct SharedOverlap {
    pub(super) side: u64,
    fork: u64,
    join: u64,
}

impl SharedOverlap {
    pub(super) fn new(gpu: &dyn GpuBackend, main: u64, side: u64) -> Result<Self> {
        ensure!(
            side != main,
            "EXL3 shared overlap needs a distinct side stream"
        );
        let fork = gpu.create_event()?;
        let join = match gpu.create_event() {
            Ok(e) => e,
            Err(error) => {
                let _ = gpu.destroy_event(fork);
                return Err(error);
            }
        };
        Ok(Self { side, fork, join })
    }

    /// Side stream waits for everything the main stream issued so far.
    pub(super) fn fork(&self, gpu: &dyn GpuBackend, main: u64) -> Result<()> {
        gpu.record_event(self.fork, main)?;
        gpu.stream_wait_event(self.side, self.fork)
    }

    /// Main stream waits for everything the side stream issued so far.
    pub(super) fn join(&self, gpu: &dyn GpuBackend, main: u64) -> Result<()> {
        gpu.record_event(self.join, self.side)?;
        gpu.stream_wait_event(main, self.join)
    }

    pub(super) fn release(self, gpu: &dyn GpuBackend) -> Result<()> {
        let a = gpu.destroy_event(self.fork);
        let b = gpu.destroy_event(self.join);
        a.and(b)
    }
}

#[cfg(test)]
#[path = "overlap_tests.rs"]
mod tests;
