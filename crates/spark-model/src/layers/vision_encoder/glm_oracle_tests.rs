// SPDX-License-Identifier: AGPL-3.0-only

//! Ignored, GPU-only GLM-5.3 vision dump oracle.
//!
//! The test deliberately consumes a vision-only safetensors index.  It keeps
//! the native Rust result small and reproducible so the companion
//! `tools/glm5_vision_oracle.py` can run the pinned vLLM math against the same
//! BF16 checkpoint tensors without loading the language model.

use std::fs;
use std::io::Cursor;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use atlas_core::config::parse_config;
use base64::Engine as _;
use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use serde_json::json;
use sha2::{Digest, Sha256};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{SafetensorsLoader, WeightLoader};

use super::VisionEncoder;
use crate::VisionItem;
use crate::weight_loader::{Glm5WeightLoader, ModelWeightLoader};

const REFERENCE_REVISION: &str = "487ecf187d3dfe74d2cf6119a92881dba403c219";

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn fixture_png(width: u32, height: u32) -> Result<String> {
    let mut image = RgbImage::new(width, height);
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        // A deterministic, non-separable pattern exercises all three
        // channels and makes an accidentally transposed patch visible.
        let r = ((x * 17 + y * 3 + (x * y) % 29) & 0xff) as u8;
        let g = ((x * 5 + y * 19 + (x + y) % 31) & 0xff) as u8;
        let b = ((x * 23 + y * 7 + (x * 3 + y * 11) % 37) & 0xff) as u8;
        *pixel = Rgb([r, g, b]);
    }
    let mut bytes = Vec::new();
    DynamicImage::ImageRgb8(image)
        .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
        .context("encode oracle fixture PNG")?;
    Ok(format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    ))
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn write_bytes(path: &Path, bytes: &[u8]) -> Result<String> {
    fs::write(path, bytes).with_context(|| format!("write {}", path.display()))?;
    Ok(sha256_hex(bytes))
}

fn dump_case(
    out_dir: &Path,
    name: &str,
    pixels: &[f32],
    grid_h: usize,
    grid_w: usize,
    output: &[u8],
    output_rows: usize,
    output_hidden: usize,
) -> Result<serde_json::Value> {
    let input_bytes = f32_bytes(pixels);
    let input_name = format!("{name}.pixels.f32le");
    let output_name = format!("{name}.output.bf16le");
    let input_sha256 = write_bytes(&out_dir.join(&input_name), &input_bytes)?;
    let output_sha256 = write_bytes(&out_dir.join(&output_name), output)?;
    Ok(json!({
        "name": name,
        "pixels_file": input_name,
        "output_file": output_name,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "patch_dim": pixels.len() / (grid_h * grid_w),
        "output_rows": output_rows,
        "output_hidden": output_hidden,
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
    }))
}

fn finite_bf16(bytes: &[u8]) -> bool {
    bytes.chunks_exact(2).all(|pair| {
        let bits = u16::from_le_bytes([pair[0], pair[1]]) as u32;
        f32::from_bits(bits << 16).is_finite()
    })
}

#[test]
#[ignore = "requires a prepared vision-only GLM checkpoint and a CUDA host"]
fn glm5_vision_native_encoder_oracle_dump() -> Result<()> {
    let model_dir = std::env::var("ATLAS_GLM_VISION_ORACLE_MODEL_DIR")
        .context("set ATLAS_GLM_VISION_ORACLE_MODEL_DIR to a vision-only checkpoint dir")?;
    let out_dir = std::env::var("ATLAS_GLM_VISION_ORACLE_OUT")
        .context("set ATLAS_GLM_VISION_ORACLE_OUT to a private oracle output dir")?;
    let model_dir = std::path::PathBuf::from(model_dir);
    let out_dir = std::path::PathBuf::from(out_dir);
    fs::create_dir_all(&out_dir).context("create oracle output directory")?;

    let config_json =
        fs::read_to_string(model_dir.join("config.json")).context("read checkpoint config.json")?;
    let config = parse_config(&config_json).context("parse checkpoint config.json")?;
    ensure!(
        config.model_type == "glm5_next",
        "oracle requires model_type=glm5_next"
    );
    let vision = config
        .vision
        .as_ref()
        .context("GLM checkpoint has no parsed vision_config")?;
    ensure!(
        vision.is_glm5_next,
        "parsed vision config is not native GLM"
    );

    let target = atlas_kernels::ptx_for_exact_target("glm-5.3-flash-nvfp4", "nvfp4")
        .context("resolve exact GLM CUDA kernel target")?;
    let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &target.modules)
        .context("create CUDA backend")?;
    let gpu: &dyn GpuBackend = &gpu;
    let store = SafetensorsLoader::new()
        .load(&model_dir, gpu, 0)
        .context("load vision-only safetensors index")?;
    ensure!(!store.is_empty(), "vision-only weight store is empty");
    ensure!(
        store.names().all(|name| {
            name.starts_with("model.visual.") || name.starts_with("model.language_model.visual.")
        }),
        "prepared oracle index contains non-visual tensors"
    );
    let encoder = Glm5WeightLoader
        .load_vision_encoder(&store, &config, gpu)
        .context("construct native GLM vision encoder")?
        .context("native GLM vision loader returned no encoder")?;
    let stream = gpu.default_stream();

    let fixtures = [("square_112", 112, 112), ("wide_112x224", 224, 112)];
    let mut cases = Vec::with_capacity(fixtures.len());
    for (name, width, height) in fixtures {
        let uri = fixture_png(width, height)?;
        let (pixels, grid_h, grid_w) = crate::vision_preprocess::preprocess_image(&uri, vision)
            .with_context(|| format!("preprocess {name}"))?;
        ensure!(
            grid_h % 2 == 0 && grid_w % 2 == 0,
            "oracle grid must merge by 2"
        );
        let item = VisionItem::image(pixels.clone(), grid_h, grid_w);
        let items = [&item];
        let geometry = encoder
            .forward_items(&items, gpu, stream)
            .with_context(|| format!("native GLM forward {name}"))?;
        gpu.synchronize(stream)
            .context("synchronize native encoder")?;
        ensure!(
            geometry.len() == 1,
            "one fixture must produce one output geometry"
        );
        let output_rows = geometry[0].2;
        let output_bytes_len = output_rows
            .checked_mul(encoder.out_hidden_size)
            .and_then(|elements| elements.checked_mul(2))
            .context("oracle output byte length overflow")?;
        let mut output = vec![0u8; output_bytes_len];
        gpu.copy_d2h(encoder.buf_out, &mut output)
            .with_context(|| format!("copy native output {name}"))?;
        ensure!(
            finite_bf16(&output),
            "native output {name} contains non-finite BF16"
        );
        cases.push(dump_case(
            &out_dir,
            name,
            &pixels,
            grid_h,
            grid_w,
            &output,
            output_rows,
            encoder.out_hidden_size,
        )?);
    }
    let manifest = json!({
        "format": "atlas-glm5-vision-oracle-v1",
        "reference_revision": REFERENCE_REVISION,
        "model_type": config.model_type,
        "vision": {
            "depth": vision.depth,
            "hidden_size": vision.hidden_size,
            "num_heads": vision.num_heads,
            "patch_size": vision.patch_size,
            "temporal_patch_size": vision.temporal_patch_size,
            "spatial_merge_size": vision.spatial_merge_size,
            "intermediate_size": vision.intermediate_size,
            "out_hidden_size": vision.out_hidden_size,
            "projection_intermediate_size": vision.projection_intermediate_size,
            "rms_norm_eps": vision.rms_norm_eps,
            "swiglu_limit": vision.swiglu_limit,
        },
        "cases": cases,
    });
    let manifest_bytes =
        serde_json::to_vec_pretty(&manifest).context("serialize oracle manifest")?;
    write_bytes(&out_dir.join("manifest.json"), &manifest_bytes)?;
    Ok(())
}
