// SPDX-License-Identifier: AGPL-3.0-only

//! Kernel-only timing for GLM-5.3's exact mHC pre/post path on GB10.
//!
//! Usage:
//!   cargo run --release -p spark-model --example glm53_hc_bench \
//!     --features cuda,gpu-examples -- [rows] [iterations]

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

const HIDDEN: usize = 4096;
const HC: usize = 4;
const MIX: usize = (2 + HC) * HC;

fn event_time_ms(mut body: impl FnMut() -> Result<()>, stream: u64, iters: usize) -> Result<f64> {
    let (mut start, mut end) = (0u64, 0u64);
    if unsafe { cuEventCreate(&mut start, 0) } != 0 || unsafe { cuEventCreate(&mut end, 0) } != 0 {
        bail!("cuEventCreate failed");
    }
    if unsafe { cuEventRecord(start, stream) } != 0 {
        bail!("cuEventRecord(start) failed");
    }
    for _ in 0..iters {
        body()?;
    }
    if unsafe { cuEventRecord(end, stream) } != 0 || unsafe { cuEventSynchronize(end) } != 0 {
        bail!("CUDA event completion failed");
    }
    let mut elapsed_ms = 0f32;
    if unsafe { cuEventElapsedTime(&mut elapsed_ms, start, end) } != 0 {
        bail!("cuEventElapsedTime failed");
    }
    unsafe {
        cuEventDestroy_v2(start);
        cuEventDestroy_v2(end);
    }
    Ok(elapsed_ms as f64 / iters as f64)
}

struct Buffers {
    streams: DevicePtr,
    function: DevicePtr,
    scale: DevicePtr,
    base: DevicePtr,
    collapsed: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    block_output: DevicePtr,
    output: DevicePtr,
}

impl Buffers {
    fn new(gpu: &dyn GpuBackend, rows: usize) -> Result<Self> {
        let allocate_zero = |bytes| -> Result<DevicePtr> {
            let ptr = gpu.alloc(bytes)?;
            gpu.memset(ptr, 0, bytes)?;
            Ok(ptr)
        };
        Ok(Self {
            streams: allocate_zero(rows * HC * HIDDEN * 4)?,
            function: allocate_zero(MIX * HC * HIDDEN * 2)?,
            scale: allocate_zero(3 * 4)?,
            base: allocate_zero(MIX * 4)?,
            collapsed: allocate_zero(rows * HIDDEN * 2)?,
            post: allocate_zero(rows * HC * 4)?,
            comb: allocate_zero(rows * HC * HC * 4)?,
            block_output: allocate_zero(rows * HIDDEN * 2)?,
            output: allocate_zero(rows * HC * HIDDEN * 4)?,
        })
    }

    fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        for ptr in [
            self.streams,
            self.function,
            self.scale,
            self.base,
            self.collapsed,
            self.post,
            self.comb,
            self.block_output,
            self.output,
        ] {
            gpu.free(ptr)?;
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let rows = args.get(1).map_or(32, |value| value.parse().unwrap());
    let iters = args.get(2).map_or(100, |value| value.parse().unwrap());
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let buffers = Buffers::new(gpu, rows)?;
    let pre = gpu.kernel("hyper_connection", "hc_pre")?;
    let post = gpu.kernel("hyper_connection", "hc_post")?;

    let launch_pre = || {
        spark_model::layers::ops::hc_pre(
            gpu,
            pre,
            buffers.streams,
            buffers.function,
            buffers.scale,
            buffers.base,
            buffers.collapsed,
            buffers.post,
            buffers.comb,
            rows as u32,
            HIDDEN as u32,
            HC as u32,
            20,
            1e-6,
            1e-6,
            stream,
        )
    };
    let launch_post = || {
        spark_model::layers::ops::hc_post(
            gpu,
            post,
            buffers.block_output,
            buffers.streams,
            buffers.post,
            buffers.comb,
            buffers.output,
            rows as u32,
            HIDDEN as u32,
            HC as u32,
            stream,
        )
    };

    for _ in 0..8 {
        launch_pre()?;
        launch_post()?;
    }
    gpu.synchronize(stream)?;
    let pre_ms = event_time_ms(launch_pre, stream, iters)?;
    let post_ms = event_time_ms(launch_post, stream, iters)?;
    println!(
        "mHC rows={rows} pre={pre_ms:.4}ms post={post_ms:.4}ms pair={:.4}ms 90-pairs={:.2}ms",
        pre_ms + post_ms,
        (pre_ms + post_ms) * 90.0
    );
    buffers.free(gpu)
}
