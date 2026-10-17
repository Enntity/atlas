// SPDX-License-Identifier: AGPL-3.0-only

//! GPU: the real Qwen3.8-Flash-Next ViT gives every image the rows it gets
//! when encoded alone, however a prefill chunk's images are split into
//! encoder passes (Atlas-Inf/atlas#180). `#[ignore]`; on a GB10 with a
//! checkpoint on disk and a qwen3.8-flash-next kernel build:
//!
//! ```text
//! ATLAS_Q38_CKPT=<model dir> \
//!   cargo test --release -p spark-model --lib qwen_passes_gpu -- --ignored --nocapture
//! ```
//!
//! Only the vision tower is uploaded (the language model is skipped by
//! prefix): ~1 GB of weights plus the encoder's ~2.3 GB of scratch.

use std::path::PathBuf;

use spark_runtime::fast_weights::FastSafetensorsLoader;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightLoader;

use crate::VisionItem;
use crate::layers::VisionEncoder;
use crate::weight_loader::{ModelWeightLoader, Qwen4ExpWeightLoader};

/// The production shape: `--max-prefill-tokens 16384 --max-num-seqs 4`.
const CHUNK: usize = 16_388;
const PATCH_DIM: usize = 1536;

/// Fields drop in order: the tower and the weights it points into go before
/// the context that owns their memory.
struct Tower {
    ve: VisionEncoder,
    _store: spark_runtime::weights::WeightStore,
    gpu: Box<spark_runtime::cuda_backend::AtlasCudaBackend>,
}

fn tower() -> Tower {
    let dir = PathBuf::from(std::env::var("ATLAS_Q38_CKPT").expect("ATLAS_Q38_CKPT"));
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL=qwen3.8-flash-next");
    let gpu = Box::new(
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend"),
    );
    let mut config = atlas_core::config::parse_config(
        &std::fs::read_to_string(dir.join("config.json")).unwrap(),
    )
    .unwrap();
    let mut loader = FastSafetensorsLoader::new();
    loader.skip_mtp = true;
    loader.skip_activation_scales = true;
    loader.skip_layer_prefix = Some("model.language_model.".into());
    loader.demand_paged_patterns = vec![".ple.ple_embedding.ngram_embedding.shard_".into()];
    let store = loader.load(&dir, gpu.as_ref(), 0).unwrap();
    // The checkpoint bound (4096^2) clamps to the 2048^2 ceiling, as served.
    config.vision.as_mut().expect("vision config").max_pixels = Some(16_777_216);
    config.max_batch_tokens = CHUNK;
    config.max_seq_len = 262_144;
    let ve = Qwen4ExpWeightLoader
        .load_vision_encoder(&store, &config, gpu.as_ref())
        .unwrap()
        .expect("vision tower");
    Tower {
        ve,
        _store: store,
        gpu,
    }
}

/// A deterministic still of `gh x gw` patches (even sides: merge 2).
fn image(gh: usize, gw: usize, seed: u32) -> VisionItem {
    let mut x = seed.wrapping_mul(2_654_435_761).max(1);
    let pixels = (0..gh * gw * PATCH_DIM)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    VisionItem::image(pixels, gh, gw)
}

impl Tower {
    fn g(&self) -> &dyn GpuBackend {
        self.gpu.as_ref()
    }

    /// Encode `items` as one call and return each item's rows, as bytes.
    fn encode(&self, items: &[&VisionItem]) -> anyhow::Result<Vec<Vec<u8>>> {
        let g = self.g();
        let per = self.ve.forward_items(items, g, g.default_stream())?;
        g.synchronize(g.default_stream())?;
        let row = self.ve.out_hidden_size * 2;
        let mut at = 0;
        let mut out = Vec::new();
        for &(_, _, rows) in &per {
            let mut v = vec![0u8; rows * row];
            g.copy_d2h(self.ve.buf_out.offset(at * row), &mut v)?;
            at += rows;
            out.push(v);
        }
        Ok(out)
    }

    fn alone(&self, item: &VisionItem) -> Vec<u8> {
        self.encode(&[item]).unwrap().remove(0)
    }
}

#[test]
#[ignore]
fn qwen_passes_gpu_every_image_gets_its_alone_rows() {
    let t = tower();
    assert_eq!(t.ve.p_max, 16_384);
    assert_eq!(
        t.ve.output_rows(),
        CHUNK + 2 * 4096,
        "window sized for a chunk"
    );
    let full_a = image(128, 128, 1); // 16384 patches, 4096 rows
    let full_b = image(128, 128, 2);
    let small = image(32, 32, 3); // 1024 patches, 256 rows
    let wide = image(48, 96, 4); // 4608 patches, 1152 rows

    // More patches than one batch: encoded in per-image passes. Exact by
    // construction: each pass is the single-image kernel sequence.
    let set = [&full_a, &small, &full_b];
    let got = t.encode(&set).unwrap();
    for (i, item) in set.iter().enumerate() {
        assert!(got[i] == t.alone(item), "pass item {i} differs from alone");
    }

    // More rows than the old 16384-row cap (the issue's case): five full-size
    // images and a thumbnail, 20736 rows, in one window.
    let fulls: Vec<VisionItem> = (10..15).map(|s| image(128, 128, s)).collect();
    let mut set: Vec<&VisionItem> = fulls.iter().collect();
    set.push(&wide);
    let rows: usize = set.iter().map(|it| it.grid_h * it.grid_w / 4).sum();
    assert!(rows > 16_384 && rows <= t.ve.output_rows(), "{rows}");
    let got = t.encode(&set).unwrap();
    for (i, item) in set.iter().enumerate() {
        assert!(
            got[i] == t.alone(item),
            "window item {i} differs from alone"
        );
    }

    // Past the window: refused before any launch, not an overrun.
    let many: Vec<VisionItem> = (20..27).map(|s| image(128, 128, s)).collect();
    let refs: Vec<&VisionItem> = many.iter().collect();
    assert!(t.encode(&refs).is_err(), "7 x 4096 rows exceed the window");
    eprintln!(
        "per-image passes == each image alone; window {}",
        t.ve.output_rows()
    );
}

/// Packing several images into ONE batch (Σp ≤ p_max) shares the
/// weight-bound GEMMs; attention, norms and mergers stay per image or per
/// row. Today's per-chunk encode already relies on this: a straddling image
/// is re-encoded with different companions at each chunk it reaches. Kept
/// separate so a packing difference is reported on its own.
#[test]
#[ignore]
fn qwen_passes_gpu_packing_is_per_image_neutral() {
    let t = tower();
    let a = image(32, 32, 5);
    let b = image(48, 96, 6);
    let c = image(64, 64, 7);
    let set = [&a, &b, &c];
    let got = t.encode(&set).unwrap();
    for (i, item) in set.iter().enumerate() {
        let alone = t.alone(item);
        let same = got[i] == alone;
        let diff = got[i].iter().zip(&alone).filter(|(x, y)| x != y).count();
        eprintln!(
            "packed item {i}: {} ({diff} differing bytes)",
            if same { "==" } else { "!=" }
        );
        assert!(same, "packed item {i} differs from alone in {diff} bytes");
    }
}
