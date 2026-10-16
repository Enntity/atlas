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
use super::*;
use std::sync::OnceLock;

pub(in crate::layers::glm5_exl3_ffn) fn shared_overlap() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_EXL3_SHARED_OVERLAP").as_deref() == Ok("1"))
}

/// One non-blocking side stream per main compute stream, shared by every
/// layer (layers are sequential on the main stream). Process lifetime.
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
    streams.insert(main, side);
    Ok(side)
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
