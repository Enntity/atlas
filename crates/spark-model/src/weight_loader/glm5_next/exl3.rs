// SPDX-License-Identifier: AGPL-3.0-only

//! EXL3 routed-expert tensor validation and device pointer tables.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::tensor;
use crate::layers::ops::Glm53Exl3PointerTables;

fn upload_pointer_table(values: &[DevicePtr], gpu: &dyn GpuBackend) -> Result<DevicePtr> {
    let bytes = values
        .iter()
        .flat_map(|pointer| pointer.0.to_le_bytes())
        .collect::<Vec<_>>();
    let table = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, table)?;
    Ok(table)
}

pub(super) fn trellis_shape(input: usize, output: usize) -> [usize; 3] {
    // EXL3 stores a logical [K, N] matrix as 16x16 tiles. The final dimension
    // is 16 * bits; this target deliberately supports the Mia 4-bit experts.
    [input / 16, output / 16, 64]
}

fn projection(
    store: &WeightStore,
    root: &str,
    input: usize,
    output: usize,
) -> Result<[DevicePtr; 3]> {
    let trellis = tensor(store, &format!("{root}.trellis"), WeightDtype::Int16)?;
    let suh = tensor(store, &format!("{root}.suh"), WeightDtype::FP16)?;
    let svh = tensor(store, &format!("{root}.svh"), WeightDtype::FP16)?;
    let mcg = tensor(store, &format!("{root}.mcg"), WeightDtype::Int32)?;
    ensure!(
        trellis.shape == trellis_shape(input, output),
        "GLM EXL3 tensor `{root}.trellis` has shape {:?}",
        trellis.shape
    );
    ensure!(
        suh.shape == [input],
        "GLM EXL3 tensor `{root}.suh` has shape {:?}",
        suh.shape
    );
    ensure!(
        svh.shape == [output],
        "GLM EXL3 tensor `{root}.svh` has shape {:?}",
        svh.shape
    );
    ensure!(
        mcg.shape == [1],
        "GLM EXL3 tensor `{root}.mcg` has shape {:?}",
        mcg.shape
    );
    Ok([trellis.ptr, suh.ptr, svh.ptr])
}

pub(super) fn pointer_tables(
    store: &WeightStore,
    root: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Glm53Exl3PointerTables> {
    ensure!(
        config.num_experts == 288,
        "GLM EXL3 kernel is specialized for 288 experts"
    );
    let mut tables = vec![vec![DevicePtr::NULL; config.num_experts]; 9];
    let (local_start, local_end) = config.local_expert_range();
    let local_intermediate = config.moe_intermediate_size / config.tp_world_size;
    for expert in local_start..local_end {
        let expert_root = format!("{root}.experts.{expert}");
        let gate = projection(
            store,
            &format!("{expert_root}.gate_proj"),
            config.hidden_size,
            local_intermediate,
        )?;
        let up = projection(
            store,
            &format!("{expert_root}.up_proj"),
            config.hidden_size,
            local_intermediate,
        )?;
        let down = projection(
            store,
            &format!("{expert_root}.down_proj"),
            local_intermediate,
            config.hidden_size,
        )?;
        for (table, pointer) in tables
            .iter_mut()
            .zip(gate.into_iter().chain(up).chain(down))
        {
            table[expert] = pointer;
        }
    }
    let uploaded = tables
        .iter()
        .map(|table| upload_pointer_table(table, gpu))
        .collect::<Result<Vec<_>>>()?;
    Ok(Glm53Exl3PointerTables {
        gate_trellis: uploaded[0],
        gate_suh: uploaded[1],
        gate_svh: uploaded[2],
        up_trellis: uploaded[3],
        up_suh: uploaded[4],
        up_svh: uploaded[5],
        down_trellis: uploaded[6],
        down_suh: uploaded[7],
        down_svh: uploaded[8],
    })
}
