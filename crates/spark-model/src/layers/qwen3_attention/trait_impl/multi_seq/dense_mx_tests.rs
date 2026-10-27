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
    // Keep the compact synthetic geometry, but select the model that ships
    // checkpoint-native MXFP8 attention twins.
    let config = ModelConfig {
        model_type: "qwen4_exp".into(),
        ..ModelConfig::qwen3_next_80b_nvfp4()
    };
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
    // Keep the compact synthetic geometry, but select the model that ships
    // checkpoint-native MXFP8 attention twins.
    let config = ModelConfig {
        model_type: "qwen4_exp".into(),
        ..ModelConfig::qwen3_next_80b_nvfp4()
    };
    let gpu = MockGpuBackend::new();
    let (mut layer, twins) = bf16_layer(&gpu, &config);
    let (h, q_dim) = (
        config.hidden_size as u32,
        (config.num_attention_heads * config.head_dim) as u32,
    );
    let (a, c) = (DevicePtr(0x1000), DevicePtr(0x2000));
    let rows = |l: &Qwen3AttentionLayer, w: &DenseWeight, k: u32| {
        l.mx_rows(&gpu, a, w, c, 5, h, k, h as usize, 0)
    };
    // No twins installed: the caller keeps its own arm.
    assert!(!rows(&layer, layer.o_dense_bf16.as_ref().unwrap(), q_dim).unwrap());
    layer.install_dense_mxfp8(twins).unwrap();
    let before = gpu.launch_count();
    assert!(rows(&layer, layer.o_dense_bf16.as_ref().unwrap(), q_dim).unwrap());
    assert_eq!(gpu.launch_count(), before + 1);
    assert!(rows(&layer, layer.o_dense_bf16.as_ref().unwrap(), q_dim / 2).is_err());
    // A copy of the O weight is not the layer's O slot.
    let copy = layer.o_dense_bf16.unwrap();
    assert!(!rows(&layer, &copy, q_dim).unwrap());
}

#[test]
fn twins_are_found_by_slot_once_the_bf16_copies_are_null() {
    // Keep the compact synthetic geometry, but select the model that ships
    // checkpoint-native MXFP8 attention twins.
    let config = ModelConfig {
        model_type: "qwen4_exp".into(),
        ..ModelConfig::qwen3_next_80b_nvfp4()
    };
    let gpu = MockGpuBackend::new();
    let (mut layer, twins) = bf16_layer(&gpu, &config);
    layer.install_dense_mxfp8(twins).unwrap();
    // What `install_w8a8_prefill` leaves behind: every BF16 slot NULL.
    for w in [
        &mut layer.attn.q_proj,
        &mut layer.attn.k_proj,
        &mut layer.attn.v_proj,
    ] {
        w.weight = DevicePtr::NULL;
    }
    layer.o_dense_bf16.as_mut().unwrap().weight = DevicePtr::NULL;
    let h = config.hidden_size as u32;
    let kv = (config.num_key_value_heads * config.head_dim) as u32;
    let q_dim = (config.num_attention_heads * config.head_dim) as u32;
    let (a, c) = (DevicePtr(0x1000), DevicePtr(0x2000));
    let o = layer.o_dense_bf16.as_ref().unwrap();
    assert!(
        layer
            .mx_rows(&gpu, a, o, c, 3, h, q_dim, h as usize, 0)
            .unwrap()
    );
    let k = &layer.attn.k_proj;
    assert!(
        layer
            .mx_rows(&gpu, a, k, c, 3, kv, h, kv as usize, 0)
            .unwrap()
    );
    // V is its own slot, not K's twin again: same shape, other bytes.
    let v = &layer.attn.v_proj;
    assert_eq!(layer.mx_slot(v), Some(2));
    assert_eq!(layer.mx_slot(k), Some(1));
}

#[test]
fn a_build_with_a_quantized_decode_copy_refuses_twins() {
    // Keep the compact synthetic geometry, but select the model that ships
    // checkpoint-native MXFP8 attention twins.
    let config = ModelConfig {
        model_type: "qwen4_exp".into(),
        ..ModelConfig::qwen3_next_80b_nvfp4()
    };
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
