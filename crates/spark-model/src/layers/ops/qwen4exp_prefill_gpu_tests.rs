// SPDX-License-Identifier: AGPL-3.0-only

//! GPU checks of the `qwen4exp_prefill` launchers against the default kernels
//! they replace, through the production Rust launch code (`#[ignore]`: run on
//! a GB10 with `--ignored`).

use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::weight_map::DenseWeight;

fn lcg_bytes(seed: &mut u64, n: usize, f: impl Fn(f32) -> Vec<u8>) -> Vec<u8> {
    (0..n)
        .flat_map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            f(((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0)
        })
        .collect()
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = g.alloc(bytes.len()).unwrap();
    g.copy_h2d(bytes, p).unwrap();
    p
}

/// `qwen4exp_ba_gates_prefill_rows` must equal `dense_gemm_ba_gates_prefill`
/// byte for byte at the TP2 rank shape (48 BA outputs, 24 v-heads, hidden
/// 2560): odd and even row counts, one row, a prefill chunk.
#[test]
#[ignore]
fn ba_gates_rows_matches_default_bitwise() {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL='*'");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let k_def = g
        .kernel("ssm_preprocess", "dense_gemm_ba_gates_prefill")
        .unwrap();
    let (nv, k) = (24usize, 2560usize);
    let (n, vpg, gs) = (2 * nv, 2usize, 2 * nv);
    let mut seed = 0x0ba5_u64;
    let bf = |x: f32| ((x.to_bits() >> 16) as u16).to_le_bytes().to_vec();
    let w = DenseWeight {
        weight: up(g, &lcg_bytes(&mut seed, n * k, |x| bf(0.05 * x))),
    };
    let a_log = up(g, &lcg_bytes(&mut seed, nv, |x| x.to_le_bytes().to_vec()));
    let dt_bias = up(g, &lcg_bytes(&mut seed, nv, |x| x.to_le_bytes().to_vec()));
    for m in [1usize, 7, 30, 4097, 16016] {
        let a = up(g, &lcg_bytes(&mut seed, m * k, bf));
        let (o1, o2) = (g.alloc(m * gs * 4).unwrap(), g.alloc(m * gs * 4).unwrap());
        super::super::dense_gemm_ba_gates_prefill(
            g, k_def, a, &w, a_log, dt_bias, o1, m as u32, n as u32, k as u32, k as u32, gs as u32,
            nv as u32, vpg as u32, stream,
        )
        .unwrap();
        let dims = [m, n, k, k, gs, nv, vpg].map(|v| v as u32);
        assert!(
            super::launch_ba_gates_rows(g, [a, w.weight, a_log, dt_bias, o2], dims, stream)
                .unwrap(),
            "the rows kernel must serve the TP2 shape"
        );
        g.synchronize(stream).unwrap();
        let (mut v1, mut v2) = (vec![0u8; m * gs * 4], vec![0u8; m * gs * 4]);
        g.copy_d2h(o1, &mut v1).unwrap();
        g.copy_d2h(o2, &mut v2).unwrap();
        let diff = v1.iter().zip(&v2).filter(|(x, y)| x != y).count();
        assert_eq!(diff, 0, "m={m}: {diff} gate bytes differ");
        println!("ba gates rows == default at m={m}");
        for p in [a, o1, o2] {
            g.free(p).unwrap();
        }
    }
}
