// SPDX-License-Identifier: AGPL-3.0-only

//! Production-shaped asynchronous MoE chain, including the activation/down
//! alias. The isolated projection tests intentionally read back between
//! kernels; this test must not do so inside the asynchronous chain.

use super::*;
use anyhow::ensure;

const EXPERTS: u32 = 288;
const INTER: u32 = 2048;

fn expand_tables(gpu: &dyn GpuBackend, tables: [DevicePtr; 3]) -> Result<[DevicePtr; 3]> {
    let mut expanded = [DevicePtr::NULL; 3];
    for (index, bytes) in [8usize, 8, 4].into_iter().enumerate() {
        let mut source = vec![0; E as usize * bytes];
        gpu.copy_d2h(tables[index], &mut source)?;
        let repeated: Vec<u8> = (0..EXPERTS as usize)
            .flat_map(|expert| {
                source[(expert % E as usize) * bytes..(expert % E as usize + 1) * bytes]
                    .iter()
                    .copied()
            })
            .collect();
        expanded[index] = upload(gpu, &repeated)?;
    }
    Ok(expanded)
}

fn routing(rows: usize, seed: usize) -> Route {
    let ids: Vec<[usize; TOPK]> = (0..rows)
        .map(|token| {
            std::array::from_fn(|slot| {
                if slot < 4 {
                    slot
                } else {
                    4 + (token * 17 + slot + seed) % 284
                }
            })
        })
        .collect();
    let mut offsets = vec![0i32; EXPERTS as usize + 1];
    for expert in ids.iter().flatten() {
        offsets[expert + 1] += 1;
    }
    for expert in 0..EXPERTS as usize {
        offsets[expert + 1] += offsets[expert];
    }
    let mut fill = offsets.clone();
    let mut sorted = vec![0; rows * TOPK];
    for (token, row) in ids.iter().enumerate() {
        for &expert in row {
            sorted[fill[expert] as usize] = token as i32;
            fill[expert] += 1;
        }
    }
    Route {
        offsets,
        sorted,
        pos: vec![],
    }
}

#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn persistent_moe_chain_needs_no_host_barriers() -> Result<()> {
    let backend = backend()?;
    let gpu: &dyn GpuBackend = &backend;
    let mut rng = Rng(0x5327_f00d);
    let gate = expand_tables(gpu, tables(gpu, &mut rng, INTER, H, 1.0 / 256.0)?)?;
    let up = expand_tables(gpu, tables(gpu, &mut rng, INTER, H, 1.0 / 128.0)?)?;
    let down = expand_tables(gpu, tables(gpu, &mut rng, H, INTER, 1.0 / 64.0)?)?;
    let input = upload(gpu, &rng.bytes(32 * H as usize / 2, false))?;
    let scales = upload(gpu, &rng.bytes(32 * H as usize / 16, true))?;
    let c = gpu.alloc(MAX_ROUTED * H as usize * 2)?;
    let q = [
        gpu.alloc(MAX_ROUTED * INTER as usize / 2)?,
        gpu.alloc(MAX_ROUTED * INTER as usize / 16)?,
    ];
    let offsets = gpu.alloc((EXPERTS as usize + 1) * 4)?;
    let sorted = gpu.alloc(MAX_ROUTED * 4)?;
    let scratch = gpu.alloc(16 + MAX_ROUTED * 8)?;
    let builder = gpu.kernel("moe", "moe_build_tile_worklist")?;
    let families = [family(gpu, "m16s")?, family(gpu, "m32s")?];
    for rows in [8u32, 16, 27, 32] {
        let route = routing(rows as usize, rows as usize);
        gpu.copy_h2d(words(&route.offsets), offsets)?;
        gpu.copy_h2d(words(&route.sorted), sorted)?;
        let kernels = families[usize::from(rows > 16)];
        let output_bytes = rows as usize * TOPK * H as usize * 2;
        let run = |pair: [KernelHandle; 2], ctas: u32, barriers: bool| -> Result<Vec<u8>> {
            let schedule = if ctas == 0 {
                ops::K128wSchedule::Grid {
                    bound: (rows * TOPK as u32).min(EXPERTS),
                }
            } else {
                ops::K128wSchedule::Stride { ctas }
            };
            let kernel = |grid| ops::K128wKernel {
                grid,
                persist: KernelHandle(0),
            };
            for _ in 0..16 {
                let packed_bytes = rows as usize * H as usize / 2;
                let a_scale = c.offset(packed_bytes);
                gpu.copy_d2d_async(input, c, packed_bytes, 0)?;
                gpu.copy_d2d_async(scales, a_scale, rows as usize * H as usize / 16, 0)?;
                ops::moe_build_tile_worklist(
                    gpu,
                    builder,
                    offsets,
                    gate[0],
                    scratch.offset(16),
                    scratch,
                    EXPERTS,
                    1,
                    64,
                    0,
                )?;
                if barriers {
                    gpu.synchronize(0)?;
                }
                ops::moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w(
                    gpu,
                    kernel(pair[0]),
                    c,
                    a_scale,
                    gate,
                    up,
                    [q[0], q[1], DevicePtr::NULL],
                    offsets,
                    sorted,
                    EXPERTS,
                    INTER,
                    H,
                    scratch,
                    schedule,
                    0,
                )?;
                if barriers {
                    gpu.synchronize(0)?;
                }
                ops::moe_w4a4_grouped_gemm_prequant_k128w(
                    gpu,
                    kernel(pair[1]),
                    q[0],
                    q[1],
                    down[0],
                    down[1],
                    down[2],
                    c,
                    offsets,
                    DevicePtr::NULL,
                    EXPERTS,
                    H,
                    INTER,
                    scratch,
                    schedule,
                    0,
                )?;
                if barriers {
                    gpu.synchronize(0)?;
                }
            }
            gpu.synchronize(0)?;
            let mut output = vec![0; output_bytes];
            gpu.copy_d2h(c, &mut output)?;
            crate::layers::glm_gpu_test_backend::finite_bf16("asynchronous MoE down", &output)?;
            Ok(output)
        };
        let reference = run(kernels[0], 0, true)?;
        for repetition in 0..4 {
            let actual = run(kernels[0], 0, false)?;
            ensure!(
                actual == reference,
                "rows={rows} repetition={repetition}: grid async mismatch"
            );
        }
        for ctas in [48, 7, 1] {
            let synced = run(kernels[1], ctas, true)?;
            ensure!(
                synced == reference,
                "rows={rows} ctas={ctas}: persistent/grid mismatch"
            );
            for repetition in 0..4 {
                let actual = run(kernels[1], ctas, false)?;
                ensure!(
                    actual == reference,
                    "rows={rows} ctas={ctas} repetition={repetition}: async mismatch"
                );
            }
        }
    }
    Ok(())
}
