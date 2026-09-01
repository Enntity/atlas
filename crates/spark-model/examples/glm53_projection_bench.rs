// SPDX-License-Identifier: AGPL-3.0-only

//! GB10 oracle for GLM-5.3's small-M BF16 projection groups.
//!
//! This benchmark uses the production cuBLASLt path and the exact checkpoint
//! widths to compare the former six KDA input projections with the appliance's
//! packed `[q,k,v,beta,forget_a,gate_a]` projection. It allocates several
//! weight sets and rotates
//! through them so the result is not an L2-resident toy.
//!
//! Usage:
//!   cargo run --release -p spark-model --example glm53_projection_bench \
//!     --features cuda,gpu-examples -- [M] [copies] [iterations]

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

const K: u32 = 4096;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn bf16(&mut self) -> u16 {
        let value = ((self.next() >> 40) as f32 / (1u64 << 23) as f32) - 1.0;
        let bits = value.to_bits();
        (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    }
}

fn bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn launch_separate(
    input: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
    m: u32,
    widths: &[u32],
    stream: u64,
) -> Result<()> {
    let mut weight_rows = 0usize;
    let mut output_elems = 0usize;
    for &n in widths {
        spark_runtime::cublaslt::bf16_gemm_act_weight_t(
            input.0,
            weight.offset(weight_rows * K as usize * 2).0,
            output.offset(output_elems * 2).0,
            m,
            n,
            K,
            stream,
        )?;
        weight_rows += n as usize;
        output_elems += m as usize * n as usize;
    }
    Ok(())
}

fn launch_packed(
    input: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
    m: u32,
    total_n: u32,
    stream: u64,
) -> Result<()> {
    spark_runtime::cublaslt::bf16_gemm_act_weight_t(
        input.0, weight.0, output.0, m, total_n, K, stream,
    )
}

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

fn read_bf16(gpu: &dyn GpuBackend, ptr: DevicePtr, elements: usize) -> Result<Vec<u16>> {
    let mut raw = vec![0u8; elements * 2];
    gpu.copy_d2h(ptr, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect())
}

fn as_f32(value: u16) -> f64 {
    f32::from_bits((value as u32) << 16) as f64
}

fn cosine_reordered(separate: &[u16], packed: &[u16], m: usize, widths: &[u32]) -> f64 {
    let total_n = widths.iter().map(|&n| n as usize).sum::<usize>();
    let mut projection_base = 0usize;
    let mut column_base = 0usize;
    let (mut dot, mut a2, mut b2) = (0f64, 0f64, 0f64);
    for &width in widths {
        let width = width as usize;
        for row in 0..m {
            for column in 0..width {
                let a = as_f32(separate[projection_base + row * width + column]);
                let b = as_f32(packed[row * total_n + column_base + column]);
                dot += a * b;
                a2 += a * a;
                b2 += b * b;
            }
        }
        projection_base += m * width;
        column_base += width;
    }
    dot / (a2.sqrt() * b2.sqrt())
}

fn bench_group(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    m: u32,
    copies: usize,
    iters: usize,
    label: &str,
    widths: &[u32],
    rng: &mut Rng,
) -> Result<()> {
    let stream = gpu.create_stream()?;
    let total_n = widths.iter().sum::<u32>();
    let weight_elements = total_n as usize * K as usize;
    let weight_host = (0..weight_elements).map(|_| rng.bf16()).collect::<Vec<_>>();
    let mut weights = Vec::with_capacity(copies);
    for _ in 0..copies {
        let weight = gpu.alloc(weight_elements * 2)?;
        gpu.copy_h2d(&bytes(&weight_host), weight)?;
        weights.push(weight);
    }
    let output_elements = m as usize * total_n as usize;
    let separate = gpu.alloc(output_elements * 2)?;
    let packed = gpu.alloc(output_elements * 2)?;

    launch_separate(input, weights[0], separate, m, widths, stream)?;
    launch_packed(input, weights[0], packed, m, total_n, stream)?;
    gpu.synchronize(stream)?;
    let separate_host = read_bf16(gpu, separate, output_elements)?;
    let packed_host = read_bf16(gpu, packed, output_elements)?;
    let cosine = cosine_reordered(&separate_host, &packed_host, m as usize, widths);
    if !cosine.is_finite() || cosine < 0.999 {
        bail!("{label} packed projection failed cosine gate: {cosine:.7}");
    }

    for i in 0..8 {
        let weight = weights[i % copies];
        launch_separate(input, weight, separate, m, widths, stream)?;
        launch_packed(input, weight, packed, m, total_n, stream)?;
    }
    gpu.synchronize(stream)?;

    let mut index = 0usize;
    let separate_ms = event_time_ms(
        || {
            let weight = weights[index % copies];
            index += 1;
            launch_separate(input, weight, separate, m, widths, stream)
        },
        stream,
        iters,
    )?;
    index = 0;
    let packed_ms = event_time_ms(
        || {
            let weight = weights[index % copies];
            index += 1;
            launch_packed(input, weight, packed, m, total_n, stream)
        },
        stream,
        iters,
    )?;

    println!(
        "{label}: M={m} widths={widths:?} copies={copies} separate={separate_ms:.4}ms packed={packed_ms:.4}ms speedup={:.3}x cosine={cosine:.7}",
        separate_ms / packed_ms
    );

    for weight in weights {
        gpu.free(weight)?;
    }
    gpu.free(separate)?;
    gpu.free(packed)?;
    Ok(())
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let m = args.get(1).map_or(32, |value| value.parse().unwrap());
    let copies = args.get(2).map_or(4, |value| value.parse().unwrap());
    let iters = args.get(3).map_or(40, |value| value.parse().unwrap());
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let mut rng = Rng(0x53_53_F1A5);
    let input_host = (0..m as usize * K as usize)
        .map(|_| rng.bf16())
        .collect::<Vec<_>>();
    let input = gpu.alloc(input_host.len() * 2)?;
    gpu.copy_h2d(&bytes(&input_host), input)?;

    bench_group(
        gpu,
        input,
        m,
        copies,
        iters,
        "kda-input-six-to-one",
        &[4096, 4096, 4096, 32, 128, 128],
        &mut rng,
    )?;
    gpu.free(input)?;
    Ok(())
}
