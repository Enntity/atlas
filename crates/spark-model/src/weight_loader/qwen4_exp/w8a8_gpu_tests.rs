// SPDX-License-Identifier: AGPL-3.0-only

//! GPU: `ATLAS_QWEN4EXP_PREFILL_W8A8` on the REAL FP4-Local layers 0 (GDN)
//! and 3 (QSA), built by the production loader at TP1. `#[ignore]`; on a
//! GB10 with the checkpoint and a qwen3.8-flash-next kernel build:
//!
//! ```text
//! ATLAS_FP4L_CKPT=<model dir> ATLAS_QWEN4EXP_DEBUG_LAYERS=4 ATLAS_QWEN4EXP_NO_PLE=1 \
//!   ATLAS_QWEN4EXP_PREFILL_W8A8=0 ATLAS_QWEN4EXP_PREFILL_HC=1 \
//!   cargo test --release -p spark-model --lib w8a8_gpu -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `=0` makes the loader keep the BF16 copies; the test installs W8A8 itself
//! (`install_w8a8_prefill`, as the loader does) and keeps the BF16 copies it
//! hands back as the reference.
//!
//! 1. GDN layer 0's whole prefill (mHC collapses, the fused producer under
//!    `_PREFILL_HC`, the block, the MoE) under W8A8 against BF16: the
//!    layer's update of the highway, relative error.
//! 2. Every W8A8 projection of both layers against the BF16 GEMM of its BF16
//!    copy (relative error), and row invariance: a pass of `T` rows equals,
//!    byte for byte, the same rows run as two passes cut at a 64-aligned row.

use std::path::PathBuf;

use spark_runtime::buffers::BufferArena;
use spark_runtime::fast_weights::FastSafetensorsLoader;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use spark_runtime::weights::WeightLoader;

use crate::layer::{ForwardContext, MoeLoraRoute, TransformerLayer};
use crate::layers::ops::{self, DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::layers::qwen3_ssm::exact_rows::GdnProj;
use crate::layers::{Qwen3AttentionLayer, Qwen3SsmLayer};
use crate::weight_loader::{ModelWeightLoader, Qwen4ExpWeightLoader};

const LAYERS: usize = 4;
/// Arena rows; the passes are `T` rows, cut at `CUT` (64-aligned, not 128).
const T_MAX: usize = 4096;
const T: usize = 3000;
const CUT: usize = 1344;

struct Rng(u64);
impl Rng {
    /// BF16 bits of roughly N(0, scale^2) values.
    fn bf16(&mut self, n: usize, scale: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|_| {
                let mut s = 0.0f32;
                for _ in 0..4 {
                    self.0 = self
                        .0
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    s += (self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                }
                atlas_core::numeric::f32_to_bf16(s * 1.732 * scale).to_le_bytes()
            })
            .collect()
    }
}

fn d2h(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Vec<u8> {
    let mut v = vec![0u8; bytes];
    g.copy_d2h(p, &mut v).unwrap();
    v
}

fn bf16s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

/// `||a - b|| / ||b||`.
fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut n) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += ((x - y) as f64).powi(2);
        n += (*y as f64).powi(2);
    }
    (d / n).sqrt()
}

struct Env {
    /// Owns the uploaded tensors the layers point into.
    _store: spark_runtime::weights::WeightStore,
    gpu: spark_runtime::cuda_backend::AtlasCudaBackend,
    config: atlas_core::config::ModelConfig,
    layers: Vec<Box<dyn TransformerLayer>>,
}

fn load() -> Env {
    let dir = PathBuf::from(std::env::var("ATLAS_FP4L_CKPT").expect("ATLAS_FP4L_CKPT"));
    for (k, v) in [
        ("ATLAS_QWEN4EXP_DEBUG_LAYERS", "4"),
        ("ATLAS_QWEN4EXP_NO_PLE", "1"),
        ("ATLAS_QWEN4EXP_PREFILL_W8A8", "0"),
    ] {
        assert_eq!(std::env::var(k).as_deref(), Ok(v), "set {k}={v}");
    }
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL=qwen3.8-flash-next");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let mut config = atlas_core::config::parse_config(
        &std::fs::read_to_string(dir.join("config.json")).unwrap(),
    )
    .unwrap();
    let mut loader = FastSafetensorsLoader::new();
    loader.skip_mtp = true;
    loader.skip_vision = true;
    loader.skip_activation_scales = true;
    loader.demand_paged_patterns = vec![".ple.ple_embedding.ngram_embedding.shard_".into()];
    let store = loader.load(&dir, &gpu, 0).unwrap();
    spark_runtime::weights::auto_detect_weight_prefix(&store, &mut config);
    config.num_hidden_layers = LAYERS;
    config.layer_types.truncate(LAYERS);
    config.max_seq_len = T_MAX;
    config.max_batch_tokens = T_MAX;
    let attn = 1;
    let layers = Qwen4ExpWeightLoader
        .load_layers(&store, &config, &gpu, &vec![KvCacheDtype::Bf16; attn])
        .unwrap();
    Env {
        _store: store,
        gpu,
        config,
        layers,
    }
}

fn ctx<'a>(
    g: &'a dyn GpuBackend,
    config: &'a atlas_core::config::ModelConfig,
    buffers: &'a BufferArena,
    parts: &'a (GemmDispatch, DerivedWeights, ModelLevers, ModelStats),
) -> ForwardContext<'a> {
    ForwardContext {
        ssm_batch: None,
        buffers,
        gpu: g,
        config,
        dispatch: &parts.0,
        derived: &parts.1,
        levers: &parts.2,
        stats: &parts.3,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        gdn_exact_replay: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: MoeLoraRoute::Skip,
    }
}

fn parts() -> (GemmDispatch, DerivedWeights, ModelLevers, ModelStats) {
    (
        GemmDispatch::defaults(),
        DerivedWeights::new(),
        ModelLevers::defaults(),
        ModelStats::new(),
    )
}

/// One W8A8 projection checked against BF16 and for row invariance.
/// `run(x, out, rows)` is the layer's W8A8 entry point.
fn check_proj(
    g: &dyn GpuBackend,
    label: &str,
    [n, k]: [usize; 2],
    bf16_w: DevicePtr,
    run: &dyn Fn(DevicePtr, DevicePtr, usize) -> anyhow::Result<()>,
) -> f64 {
    let s = g.default_stream();
    let x = g.alloc(T * k * 2).unwrap();
    g.copy_h2d(&Rng(0x5eed ^ (n * k) as u64).bf16(T * k, 1.0), x)
        .unwrap();
    let [want, one, two] = [0; 3].map(|_| g.alloc(T * n * 2).unwrap());
    ops::cublas_bf16_proj_dense(x, bf16_w, want, T as u32, n as u32, k as u32, s).unwrap();
    run(x, one, T).unwrap();
    run(x, two, CUT).unwrap();
    run(x.offset(CUT * k * 2), two.offset(CUT * n * 2), T - CUT).unwrap();
    g.synchronize(s).unwrap();
    let (w, a, b) = (
        d2h(g, want, T * n * 2),
        d2h(g, one, T * n * 2),
        d2h(g, two, T * n * 2),
    );
    let diff = a.iter().zip(&b).filter(|(p, q)| p != q).count();
    assert_eq!(
        diff, 0,
        "{label}: {T} rows in one pass vs cut at {CUT}: {diff} bytes differ"
    );
    let e = rel(&bf16s(&a), &bf16s(&w));
    eprintln!("{label} [{n} x {k}]: W8A8 vs BF16 rel {e:.3e}; one pass == two passes (cut {CUT})");
    assert!(e > 1e-4 && e < 6e-2, "{label}: rel {e}");
    for p in [x, want, one, two] {
        g.free(p).unwrap();
    }
    e
}

#[test]
#[ignore]
fn w8a8_gpu_projections_match_bf16_and_are_row_invariant() {
    let mut env = load();
    let g: &dyn GpuBackend = &env.gpu;
    let s = g.default_stream();
    let c = &env.config;
    let buffers = BufferArena::new(c, T_MAX, T_MAX, 16, 8, g).unwrap();
    let parts = parts();
    let (h, vd) = (
        c.hidden_size,
        c.linear_num_value_heads * c.linear_value_head_dim,
    );

    let gdn = env.layers[0]
        .as_any_mut()
        .and_then(|a| a.downcast_mut::<Qwen3SsmLayer>())
        .expect("layer 0 is GDN");
    let swap = gdn.install_w8a8_prefill(g, T_MAX, s).unwrap();
    let ctx = ctx(g, c, &buffers, &parts);
    for (i, (proj, nk)) in [
        (GdnProj::Qkvz, [c.ssm_qkvz_size(), h]),
        (GdnProj::Out, [h, vd]),
    ]
    .into_iter()
    .enumerate()
    {
        let run = |x, out, rows| gdn.w8a8_proj(&ctx, proj, x, out, rows, s).map(|_| ());
        check_proj(g, &format!("GDN {proj:?}"), nk, swap.bf16[i].0, &run);
    }

    let attn = env.layers[3]
        .as_any_mut()
        .and_then(|a| a.downcast_mut::<Qwen3AttentionLayer>())
        .expect("layer 3 is QSA");
    let swap = attn.install_w8a8_prefill(g, T_MAX, s).unwrap();
    let q_n = c.num_attention_heads * c.head_dim * 2;
    let kv_n = c.num_key_value_heads * c.head_dim;
    for (slot, n) in [q_n, kv_n, kv_n].into_iter().enumerate() {
        let run = |x, out, rows: usize| {
            let act = attn.w8a8_qkv_act(&ctx, x, rows as u32, s)?.unwrap();
            attn.w8a8_qkv(slot, act, out, [rows as u32, n as u32], s)
        };
        let label = ["QSA q", "QSA k", "QSA v"][slot];
        check_proj(g, label, [n, h], swap.bf16[slot].0, &run);
    }
    let kd = c.num_attention_heads * c.head_dim;
    let run = |x, out, rows: usize| {
        let dims = [rows as u32, h as u32, kd as u32];
        attn.w8a8_oproj(x, out, dims, &ctx, s).map(|_| ())
    };
    check_proj(g, "QSA o", [h, kd], swap.bf16[3].0, &run);
}

#[test]
#[ignore]
fn w8a8_gpu_gdn_layer_prefill_tracks_bf16() {
    let mut env = load();
    let g: &dyn GpuBackend = &env.gpu;
    let s = g.default_stream();
    let c = env.config.clone();
    let buffers = BufferArena::new(&c, T_MAX, T_MAX, 16, 8, g).unwrap();
    let parts = parts();
    let h = c.hidden_size;
    let row = c.hc_mult * h;
    let kv_config = KvCacheConfig {
        block_size: 16,
        num_kv_heads: c.num_key_value_heads,
        head_dim: c.head_dim,
        num_layers: 1,
        dtype: KvCacheDtype::Bf16,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    };
    let mut kv = PagedKvCache::new(kv_config, 8, g).unwrap();
    let x = Rng(0x1a7e).bf16(T * h, 0.05);
    let gdn = env.layers[0]
        .as_any_mut()
        .and_then(|a| a.downcast_mut::<Qwen3SsmLayer>())
        .expect("layer 0 is GDN");

    // The layer's update of the highway: its streams after the prefill (the
    // deferred FFN post flushed) minus the input expanded to every stream.
    let mut run = |gdn: &Qwen3SsmLayer| -> Vec<f32> {
        g.copy_h2d(&x, buffers.hidden_states()).unwrap();
        let mut state = gdn.alloc_state(g).unwrap();
        let ctx = ctx(g, &c, &buffers, &parts);
        let (mut bt, mut d1, mut d2) = (Vec::new(), Vec::new(), Vec::new());
        gdn.prefill(
            buffers.hidden_states(),
            buffers.residual(),
            T,
            state.as_mut(),
            &mut kv,
            0,
            &mut bt,
            &mut d1,
            &mut d2,
            0,
            &ctx,
            s,
        )
        .unwrap();
        let hc = gdn.hc.as_ref().unwrap();
        let post_k = g.kernel("hyper_connection", "hc_post").unwrap();
        ops::qwen4exp_prefill_seam::flush_pending(
            g,
            post_k,
            hc,
            buffers.hc_streams(),
            buffers.hc_post(),
            h as u32,
            s,
        )
        .unwrap();
        g.synchronize(s).unwrap();
        let out = d2h(g, buffers.hc_streams(), T * row * 4);
        let xin = bf16s(&x);
        out.chunks_exact(4)
            .enumerate()
            .map(|(i, b)| {
                let (t, d) = (i / row, i % row % h);
                f32::from_le_bytes([b[0], b[1], b[2], b[3]]) - xin[t * h + d]
            })
            .collect()
    };
    let bf16 = run(gdn);
    let again = run(gdn);
    assert_eq!(rel(&again, &bf16), 0.0, "the BF16 prefill is deterministic");
    let _swap = gdn.install_w8a8_prefill(g, T_MAX, s).unwrap();
    let w8a8 = run(gdn);
    let e = rel(&w8a8, &bf16);
    // `_PREFILL_HC=1` (the collapse emits the qkvz input as MXFP8) and `=0`
    // (one quant pass) must print the same hash: the producer's bytes are
    // the quant's.
    let hash = w8a8.iter().fold(0xcbf29ce484222325u64, |h, v| {
        (h ^ v.to_bits() as u64).wrapping_mul(0x100000001b3)
    });
    eprintln!(
        "GDN layer 0 prefill, {T} rows: highway update W8A8 vs BF16 rel {e:.3e}; \
         W8A8 hash {hash:016x} (ATLAS_QWEN4EXP_PREFILL_HC={})",
        std::env::var("ATLAS_QWEN4EXP_PREFILL_HC").unwrap_or_default()
    );
    assert!(e > 1e-4 && e < 1e-1, "rel {e}");
}

/// What the W8A8 install adds to a TP2 rank's load: tuning the four
/// distinct per-rank shapes' row buckets to 16384 rows (the production
/// `--max-prefill-tokens`). No checkpoint needed.
#[test]
#[ignore]
fn w8a8_gpu_tp2_tuning_time() {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL=qwen3.8-flash-next");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let s = gpu.default_stream();
    let t = std::time::Instant::now();
    for (n, k) in [(8192, 2560), (2560, 3072), (6144, 2560), (256, 2560)] {
        ops::mx_prepare(n, k, 16384, s).unwrap();
    }
    eprintln!(
        "TP2 W8A8 tuning, 4 shapes to 16384 rows: {:.2} s",
        t.elapsed().as_secs_f64()
    );
}
