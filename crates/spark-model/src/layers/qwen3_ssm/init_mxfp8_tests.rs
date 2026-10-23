// SPDX-License-Identifier: AGPL-3.0-only

//! The MXFP8 GDN decode copy on the mock backend: what it accepts, and that
//! the batched verify reads it for both projections.

use super::*;
use crate::weight_map::Mxfp8Weight;

/// The BF16 GDN build `dense_bf16_layer` makes: sequential BF16 QKVZ plus a
/// dense out_proj, nothing requantized.
fn bf16_gdn_layer(gpu: &MockGpuBackend, config: &ModelConfig) -> Qwen3SsmLayer {
    let mut layer = native_fp8_gdn_layer(gpu, config, false, false);
    let (h, value_dim) = (
        config.hidden_size,
        config.linear_num_value_heads * config.linear_value_head_dim,
    );
    layer.ssm.in_proj_qkvz = DenseWeight {
        weight: gpu.alloc(config.ssm_qkvz_size() * h * 2).unwrap(),
    };
    layer.out_proj_dense = Some(DenseWeight {
        weight: gpu.alloc(h * value_dim * 2).unwrap(),
    });
    layer
}

fn twin(gpu: &MockGpuBackend, n: usize, k: usize) -> (Mxfp8Weight, [usize; 2]) {
    let w = Mxfp8Weight {
        data: gpu.alloc(n * k).unwrap(),
        scales: gpu.alloc(n * k / 32).unwrap(),
        tiled: false,
    };
    (w, [n, k])
}

/// `(qkvz, out)` twins of the layer's real shapes.
fn twins(
    gpu: &MockGpuBackend,
    config: &ModelConfig,
) -> ((Mxfp8Weight, [usize; 2]), (Mxfp8Weight, [usize; 2])) {
    let h = config.hidden_size;
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    (
        twin(gpu, config.ssm_qkvz_size(), h),
        twin(gpu, h, value_dim),
    )
}

#[test]
fn batched_verify_reads_the_mxfp8_copy_for_both_projections() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let (qkvz_n, h) = (config.ssm_qkvz_size() as u32, config.hidden_size as u32);
    for ks in [&[4usize, 3][..], &[2, 2], &[1]] {
        let gpu = MockGpuBackend::new();
        let mut layer = bf16_gdn_layer(&gpu, &config);
        let (qkvz, out) = twins(&gpu, &config);
        layer
            .install_mxfp8_decode(&gpu, &config, qkvz, out)
            .unwrap();
        run_batched_verify(&gpu, &config, &layer, ks).unwrap();
        // One `mxfp8_gemv_tc*` launch per projection (rows <= 32): a CTA per
        // 16 weight rows.
        let seen = gpu_launch_grids(&gpu);
        for (what, n) in [("QKVZ", qkvz_n), ("out_proj", h)] {
            let launches = seen
                .iter()
                .filter(|g| **g == ([n / 16, 1, 1], [256, 1, 1]))
                .count();
            assert_eq!(launches, 1, "{what} at ks={ks:?}; grids seen: {seen:?}");
        }
    }
}

#[test]
fn install_takes_only_a_bf16_build_and_the_layer_s_own_shapes() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    // A requantized (block FP8) build already has a decode copy.
    let mut fp8 = native_fp8_gdn_layer(&gpu, &config, true, true);
    let (qkvz, out) = twins(&gpu, &config);
    let e = fp8
        .install_mxfp8_decode(&gpu, &config, qkvz, out)
        .unwrap_err();
    assert!(
        e.to_string().contains("not a sequential BF16 GDN build"),
        "{e}"
    );
    // Shapes off by a rank's slice in either projection.
    let mut layer = bf16_gdn_layer(&gpu, &config);
    let (h, qkvz_n) = (config.hidden_size, config.ssm_qkvz_size());
    for (q, o) in [
        (twin(&gpu, qkvz_n * 2, h), twin(&gpu, h, 4096)),
        (twin(&gpu, qkvz_n, h), twin(&gpu, h, 8192)),
    ] {
        let e = layer.install_mxfp8_decode(&gpu, &config, q, o).unwrap_err();
        assert!(e.to_string().contains("layer wants"), "{e}");
    }
    let (q, o) = twins(&gpu, &config);
    layer.install_mxfp8_decode(&gpu, &config, q, o).unwrap();
}
