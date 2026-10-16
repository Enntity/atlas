// SPDX-License-Identifier: AGPL-3.0-only

//! GPU: the first four decoder layers of the REAL FP4-Local checkpoint load
//! through the production path — the fast loader (CSF decode at upload), the
//! MXFP8 funnel in `dense_auto`, and `Qwen4ExpWeightLoader::load_layers` with
//! its per-layer MXFP8 source release. `#[ignore]`; on a GB10 with the
//! checkpoint on disk and a qwen3.8-flash-next kernel build:
//!
//! ```text
//! ATLAS_FP4L_CKPT=<model dir> ATLAS_QWEN4EXP_DEBUG_LAYERS=4 ATLAS_QWEN4EXP_NO_PLE=1 \
//!   cargo test --release -p spark-model --lib fp4local_gpu -- --ignored --nocapture
//! ```
//!
//! `ATLAS_QWEN4EXP_DEBUG_LAYERS=4` makes the loader drop every tensor of
//! layers 4.. at upload (~9 GB resident instead of ~95), and the test refuses
//! to run without it. PLE is off because its NVFP4 n-gram table is another
//! agent's port; layer 1 is the PLE host.

use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use atlas_core::config::LayerType;
use spark_runtime::fast_weights::FastSafetensorsLoader;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{WeightDtype, WeightLoader, nvfp4_csf};

use super::*;
use crate::weight_loader::ModelWeightLoader;

const LAYERS: usize = 4;

/// One tensor's raw bytes and shape, straight from the shard on disk.
fn read(dir: &Path, name: &str) -> (Vec<u8>, Vec<usize>) {
    let index: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json")).unwrap())
            .unwrap();
    let shard = dir.join(index["weight_map"][name].as_str().unwrap());
    let f = std::fs::File::open(shard).unwrap();
    let mut len = [0u8; 8];
    f.read_exact_at(&mut len, 0).unwrap();
    let n = u64::from_le_bytes(len) as usize;
    let mut h = vec![0u8; n];
    f.read_exact_at(&mut h, 8).unwrap();
    let header: serde_json::Value = serde_json::from_slice(&h).unwrap();
    let info = &header[name];
    let (a, b) = (
        info["data_offsets"][0].as_u64().unwrap() as usize,
        info["data_offsets"][1].as_u64().unwrap() as usize,
    );
    let mut buf = vec![0u8; b - a];
    f.read_exact_at(&mut buf, (8 + n + a) as u64).unwrap();
    let shape = info["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d.as_u64().unwrap() as usize)
        .collect();
    (buf, shape)
}

fn d2h(g: &dyn GpuBackend, p: spark_runtime::gpu::DevicePtr, n: usize) -> Vec<u8> {
    let mut v = vec![0u8; n];
    g.copy_d2h(p, &mut v).unwrap();
    v
}

#[test]
#[ignore]
fn fp4local_gpu_first_layers_load_exactly() {
    let dir = PathBuf::from(std::env::var("ATLAS_FP4L_CKPT").expect("ATLAS_FP4L_CKPT"));
    assert_eq!(
        std::env::var("ATLAS_QWEN4EXP_DEBUG_LAYERS").as_deref(),
        Ok("4"),
        "refusing to upload the whole checkpoint: set ATLAS_QWEN4EXP_DEBUG_LAYERS=4"
    );
    assert_eq!(std::env::var("ATLAS_QWEN4EXP_NO_PLE").as_deref(), Ok("1"));
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL=qwen3.8-flash-next");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;

    let mut config = atlas_core::config::parse_config(
        &std::fs::read_to_string(dir.join("config.json")).unwrap(),
    )
    .unwrap();
    let mut loader = FastSafetensorsLoader::new();
    loader.skip_mtp = true;
    loader.skip_vision = true;
    loader.skip_activation_scales = true;
    loader.demand_paged_patterns = vec![".ple.ple_embedding.ngram_embedding.shard_".into()];
    let t = std::time::Instant::now();
    let store = loader.load(&dir, g, 0).unwrap();
    eprintln!(
        "uploaded {} tensors in {:.1} s",
        store.len(),
        t.elapsed().as_secs_f64()
    );
    spark_runtime::weights::auto_detect_weight_prefix(&store, &mut config);
    config.num_hidden_layers = LAYERS;
    config.layer_types.truncate(LAYERS);
    config.max_seq_len = 4096;
    config.max_batch_tokens = 2048;

    // 1. CSF scales on the device are the host decode of the disk bytes.
    for (l, e, p) in [(0, 0, "gate"), (3, 511, "down"), (2, 300, "up")] {
        let stem = format!("model.language_model.layers.{l}.mlp.experts.{e}.{p}_proj.weight_scale");
        let (f, fs) = read(&dir, &format!("{stem}.nvfp4_csf_fixed"));
        let (x, _) = read(&dir, &format!("{stem}.nvfp4_csf_exceptions"));
        let (want, shape) = nvfp4_csf::decode_nvfp4_csf(&f, &fs, &x).unwrap();
        let t = store.get(&stem).unwrap();
        assert_eq!(
            (t.dtype, t.shape.as_slice()),
            (WeightDtype::FP8E4M3, &shape[..]),
            "{stem}"
        );
        assert_eq!(d2h(g, t.ptr, want.len()), want, "{stem}");
    }

    // 2. Every MXFP8 projection of these layers expands, through dense_auto,
    //    to exactly the host expansion of its disk bytes.
    let mut checked = 0;
    for i in 0..LAYERS {
        let lp = config.layer_prefix(i);
        for c in format::layer_components(config.layer_types[i]) {
            for stem in c.stems(&lp) {
                let (w, ws) = read(&dir, &format!("{stem}.weight"));
                let (s, _) = read(&dir, &format!("{stem}.weight_scale"));
                let want: Vec<u8> = crate::weight_map::mxfp8_to_bf16_host(&w, &s, ws[0], ws[1])
                    .unwrap()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                let got =
                    crate::weight_map::dense_auto(&store, &format!("{stem}.weight"), g).unwrap();
                assert_eq!(d2h(g, got.weight, want.len()), want, "{stem}");
                g.free(got.weight).unwrap();
                checked += 1;
            }
        }
    }
    eprintln!("{checked} MXFP8 projections: device BF16 == host expansion");

    // 3. The layers build, and their MXFP8 sources are released afterwards.
    let format = Qwen4ExpFormat::detect(&store, &config).unwrap();
    assert!(format.is_fp4_local(), "{format:?}");
    let attn = config
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::FullAttention)
        .count();
    let t = std::time::Instant::now();
    let layers = Qwen4ExpWeightLoader
        .load_layers(&store, &config, g, &vec![KvCacheDtype::Bf16; attn])
        .unwrap();
    eprintln!(
        "built {} layers in {:.1} s",
        layers.len(),
        t.elapsed().as_secs_f64()
    );
    assert_eq!(layers.len(), LAYERS);
    for i in 0..LAYERS {
        let lp = config.layer_prefix(i);
        for c in format::layer_components(config.layer_types[i]) {
            for stem in c.stems(&lp) {
                for part in ["weight", "weight_scale"] {
                    let n = format!("{stem}.{part}");
                    assert!(store.was_reclaimed(&n), "{n} still resident");
                }
            }
        }
    }
}
