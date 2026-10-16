// SPDX-License-Identifier: AGPL-3.0-only

//! The MXFP8 attention twins (`dense_mx.rs`) on the mock: a BF16 build
//! takes them, the multi-sequence Q/K/V and O read them once for all rows,
//! and a build with a quantized decode copy refuses them.

use super::*;
use crate::weight_map::Mxfp8Weight;

fn twin(gpu: &MockGpuBackend, n: usize, k: usize) -> (Mxfp8Weight, [usize; 2]) {
    let w = Mxfp8Weight {
        data: gpu.alloc(n * k).unwrap(),
        scales: gpu.alloc(n * k / 32).unwrap(),
        tiled: false,
    };
    (w, [n, k])
}

/// The `--attn-proj-dtype bf16` build: dense Q/K/V, no quantized copy, a
/// BF16 O, plus `[q, k, v, o]` twins of the real shapes.
fn bf16_layer(
    gpu: &MockGpuBackend,
    config: &ModelConfig,
) -> (Qwen3AttentionLayer, [(Mxfp8Weight, [usize; 2]); 4]) {
    let mut layer = attn_layer(gpu, config);
    layer.q_weight = None;
    layer.k_weight = None;
    layer.v_weight = None;
    let h = config.hidden_size;
    let q_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_key_value_heads * config.head_dim;
    layer.set_o_dense_bf16(DenseWeight {
        weight: gpu.alloc(h * q_dim * 2).unwrap(),
    });
    let twins = [
        twin(gpu, 2 * q_dim, h),
        twin(gpu, kv_dim, h),
        twin(gpu, kv_dim, h),
        twin(gpu, h, q_dim),
    ];
    (layer, twins)
}

#[test]
fn multi_seq_qkv_reads_each_twin_once_for_every_row() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let (mut layer, twins) = bf16_layer(&gpu, &config);
    layer.install_dense_mxfp8(twins).unwrap();
    let q_n = (2 * config.num_attention_heads * config.head_dim) as u32;
    let kv_n = (config.num_key_value_heads * config.head_dim) as u32;
    for n in [1usize, 3, 8] {
        let (_, launches) = qkv_phase_d2d(&gpu, &config, &layer, n);
        let mx = |rows: u32| {
            launches
                .iter()
                .filter(|l| l.grid == [rows / 16, 1, 1] && l.block == [256, 1, 1])
                .count()
        };
        // One launch per projection whatever the row count: Q, K and V.
        assert_eq!(mx(q_n), 1, "Q at n={n}");
        assert_eq!(mx(kv_n), 2, "K/V at n={n}");
    }
}

#[test]
fn o_rows_take_the_twin_and_a_wrong_shape_is_an_error() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let (mut layer, twins) = bf16_layer(&gpu, &config);
    let o = layer.o_dense_bf16.unwrap();
    let (h, q_dim) = (
        config.hidden_size as u32,
        (config.num_attention_heads * config.head_dim) as u32,
    );
    let (a, c) = (DevicePtr(0x1000), DevicePtr(0x2000));
    // No twins installed: the caller keeps its own arm.
    assert!(
        !layer
            .mx_rows(&gpu, a, &o, c, 5, h, q_dim, h as usize, 0)
            .unwrap()
    );
    layer.install_dense_mxfp8(twins).unwrap();
    let before = gpu.launch_count();
    assert!(
        layer
            .mx_rows(&gpu, a, &o, c, 5, h, q_dim, h as usize, 0)
            .unwrap()
    );
    assert_eq!(gpu.launch_count(), before + 1);
    assert!(
        layer
            .mx_rows(&gpu, a, &o, c, 5, h, q_dim / 2, h as usize, 0)
            .is_err()
    );
}

#[test]
fn a_build_with_a_quantized_decode_copy_refuses_twins() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let (_, twins) = bf16_layer(&gpu, &config);
    // NVFP4 Q/K/V as built by the default arm.
    let mut nvfp4 = attn_layer(&gpu, &config);
    nvfp4.set_o_dense_bf16(DenseWeight {
        weight: gpu.alloc(64).unwrap(),
    });
    assert!(nvfp4.install_dense_mxfp8(twins).is_err());
    // No BF16 O.
    let (mut no_o, twins) = bf16_layer(&gpu, &config);
    no_o.o_dense_bf16 = None;
    assert!(no_o.install_dense_mxfp8(twins).is_err());
}
