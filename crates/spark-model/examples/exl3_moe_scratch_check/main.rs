// SPDX-License-Identifier: AGPL-3.0-only
//! Actual rank-local native FFN A→B→A replay with isolated versus shared scratch.
mod guarded;
mod protocol;
use anyhow::{Context, Result, ensure};
use atlas_core::{
    config::{ModelConfig, parse_config},
    scope::ModelResource,
};
use protocol::{ReplayIORouter, digest, validate_bf16};
use spark_model::layers::{
    FfnComponent,
    glm5_exl3_ffn::{
        Glm5Exl3FfnLayer, MoeScratch, MoeScratchGeometry, load_glm5_exl3_ffn_with_scratch,
    },
};
use spark_runtime::{
    cuda_backend::AtlasCudaBackend,
    gpu::{DevicePtr, GpuBackend},
    weights::exl3::Exl3CheckpointRouter,
};
use std::collections::BTreeMap;
const GUARD: usize = 256;

fn guarded_upload(
    gpu: &dyn GpuBackend,
    bytes: &[u8],
    owned: &mut Vec<DevicePtr>,
) -> Result<DevicePtr> {
    let mut data = vec![0xa5; bytes.len() + 2 * GUARD];
    data[GUARD..GUARD + bytes.len()].copy_from_slice(bytes);
    let ptr = gpu.alloc(data.len())?;
    owned.push(ptr);
    gpu.copy_h2d(&data, ptr)?;
    Ok(ptr.offset(GUARD))
}
fn guarded_read(gpu: &dyn GpuBackend, ptr: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut full = vec![0; bytes + 2 * GUARD];
    gpu.copy_d2h(DevicePtr(ptr.0 - GUARD as u64), &mut full)?;
    ensure!(
        full[..GUARD]
            .iter()
            .chain(&full[GUARD + bytes..])
            .all(|b| *b == 0xa5),
        "GPU allocation guard overwritten"
    );
    Ok(full[GUARD..GUARD + bytes].to_vec())
}
fn local(
    layer: &FfnComponent,
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    output: DevicePtr,
    rows: usize,
) -> Result<()> {
    let FfnComponent::Exl3(native) = layer else {
        anyhow::bail!("native EXL3 layer required")
    };
    let Glm5Exl3FfnLayer::Moe(moe) = native.as_ref() else {
        anyhow::bail!("two routed layers required")
    };
    moe.forward_local(gpu, input, output, rows)
}
fn run_mode(
    io: &ReplayIORouter,
    gpu: &dyn GpuBackend,
    c: &ModelConfig,
    checkpoint: &Exl3CheckpointRouter,
    stream: u64,
    shared: bool,
) -> Result<BTreeMap<String, Vec<u8>>> {
    let mut scratch = if shared {
        Some(MoeScratch::allocate(
            gpu,
            MoeScratchGeometry::new(c, io.plan.capacity, stream)?,
        )?)
    } else {
        None
    };
    let mut layers = Vec::new();
    let mut allocations = Vec::new();
    let result = (|| {
        for index in io.plan.layers {
            layers.push(load_glm5_exl3_ffn_with_scratch(
                checkpoint,
                c,
                index,
                gpu,
                io.plan.capacity,
                stream,
                spark_model::weight_loader::glm53_tp2_packed_numerics,
                scratch.as_ref().map(MoeScratch::borrow),
                None,
            )?);
        }
        let mut values = BTreeMap::new();
        for case in &io.plan.cases {
            let a = ReplayIORouter::read_bound(&case.input_a)?;
            let b = ReplayIORouter::read_bound(&case.input_b)?;
            validate_bf16(&a, case.rows, c.hidden_size)?;
            validate_bf16(&b, case.rows, c.hidden_size)?;
            let inputs = [
                guarded_upload(gpu, &a, &mut allocations)?,
                guarded_upload(gpu, &b, &mut allocations)?,
            ];
            let outputs = [
                guarded_upload(gpu, &vec![0; a.len()], &mut allocations)?,
                guarded_upload(gpu, &vec![0; a.len()], &mut allocations)?,
                guarded_upload(gpu, &vec![0; a.len()], &mut allocations)?,
            ];
            // No synchronization between submissions: exercise actual shared GPU lifetime.
            for (step, index) in [0usize, 1, 0].into_iter().enumerate() {
                local(&layers[index], gpu, inputs[index], outputs[step], case.rows)?;
            }
            gpu.synchronize(stream)?;
            ensure!(
                guarded_read(gpu, inputs[0], a.len())? == a
                    && guarded_read(gpu, inputs[1], b.len())? == b,
                "FFN modified input bytes"
            );
            let captures: Vec<_> = outputs
                .into_iter()
                .map(|p| guarded_read(gpu, p, a.len()))
                .collect::<Result<_>>()?;
            for bytes in &captures {
                validate_bf16(bytes, case.rows, c.hidden_size)?;
            }
            ensure!(
                captures[0] != captures[1],
                "uninformative fixture: two checkpoint layers produced identical output"
            );
            ensure!(captures[0] == captures[2], "A→B→A replay changed A output");
            for (step, value) in captures.into_iter().enumerate() {
                let name = format!("{}-step{step}", case.id);
                io.write(
                    &format!("{}-{name}.bf16", if shared { "shared" } else { "isolated" }),
                    &value,
                )?;
                values.insert(name, value);
            }
        }
        Ok(values)
    })();
    // Unknown completion never permits freeing operands touched by queued kernels.
    gpu.synchronize(stream)
        .context("scratch replay completion required before teardown")?;
    let mut cleanup = None;
    for layer in &mut layers {
        if let Err(e) = layer.release_exl3(gpu) {
            cleanup.get_or_insert(e);
        }
    }
    if let Some(owner) = scratch.as_mut() {
        if let Err(e) = owner.release(gpu) {
            cleanup.get_or_insert(e);
        }
    }
    for p in allocations {
        if let Err(e) = gpu.free(p) {
            cleanup.get_or_insert(e);
        }
    }
    match (result, cleanup) {
        (Ok(v), None) => Ok(v),
        (Err(e), None) => Err(e),
        (Ok(_), Some(e)) => Err(e),
        (Err(e), Some(cleanup)) => {
            Err(e.context(format!("scratch replay rollback failed: {cleanup:#}")))
        }
    }
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 5 && args[1] == "--plan" && args[3] == "--out",
        "usage: exl3_moe_scratch_check --plan PLAN.json --out NEW_DIR"
    );
    let io = ReplayIORouter::open(
        std::path::Path::new(&args[2]),
        std::path::Path::new(&args[4]),
    )?;
    ensure!(
        std::env::var_os("ATLAS_EXL3_FFN_DUMP_DIR").is_none(),
        "scratch oracle excludes diagnostic I/O synchronization"
    );
    let mut c = parse_config(&io.config()?)?;
    ensure!(
        c.model_type == "glm5_next" && c.hidden_size == 4096 && c.num_experts == 288,
        "actual GLM53 checkpoint required"
    );
    for layer in io.plan.layers {
        ensure!(
            layer < c.num_hidden_layers && !c.mlp_only_layers.contains(&layer),
            "explicit layers must both be routed checkpoint layers"
        );
    }
    c.tp_rank = io.plan.rank;
    c.tp_world_size = io.plan.world;
    c.ep_rank = 0;
    c.ep_world_size = 1;
    c.expert_tp = false;
    let target = atlas_kernels::ptx_for_exact_target("glm-5.3-flash-exl3", "exl3")
        .context("actual native EXL3 embedded target required")?;
    ensure!(
        !target.modules.is_empty() && atlas_kernels::TARGET_CLOSURES != "{}",
        "stub kernel build is unqualified"
    );
    let modules: BTreeMap<_, _> = target
        .modules
        .iter()
        .map(|(name, ptx)| (*name, digest(ptx)))
        .collect();
    let backend = AtlasCudaBackend::new(io.plan.device, &target.modules)?;
    let gpu = guarded::GuardedGpu::new(&backend);
    let stream = gpu.default_stream();
    let checkpoint = Exl3CheckpointRouter::open(&io.plan.checkpoint)?;
    let isolated = run_mode(&io, &gpu, &c, &checkpoint, stream, false)?;
    let shared = run_mode(&io, &gpu, &c, &checkpoint, stream, true)?;
    ensure!(
        isolated.keys().eq(shared.keys()),
        "replay case identity differs"
    );
    for (key, value) in &isolated {
        ensure!(
            shared[key] == *value,
            "shared scratch altered exact BF16 output for {key}"
        );
    }
    gpu.check_all()?;
    ensure!(
        gpu.outstanding() == 0,
        "replay left guarded model allocations live"
    );
    let outputs: BTreeMap<_, _> = isolated.iter().map(|(k, v)| (k, digest(v))).collect();
    let report = serde_json::json!({"schema":"atlas-native-two-layer-scratch-replay-v1","passed":true,
        "scope":"actual production loader/NativeMoe forward rank-local A→B→A; no TP collective or vLLM numerical equivalence claim",
        "rank":io.plan.rank,"world":io.plan.world,"layers":io.plan.layers,"capacity":io.plan.capacity,
        "stream":stream,"plan_sha256":io.plan_sha256,"executable_sha256":ReplayIORouter::executable_hash()?,
        "ptx_sha256":modules,"kernel_set_hash":atlas_kernels::KERNEL_SET_HASH,
        "target_closures":serde_json::from_str::<serde_json::Value>(atlas_kernels::TARGET_CLOSURES)?,
        "checkpoint_config_sha256":io.plan.config_sha256,"checkpoint_index_sha256":io.plan.index_sha256,
        "output_sha256":outputs,"guards_passed":true,"input_unchanged":true,
        "loaded_blas_sha256":ReplayIORouter::loaded_blas()?,
        "shared_scratch_bytes":MoeScratchGeometry::new(&c,io.plan.capacity,stream)?.total_bytes()?});
    io.write("report.json", &serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}
