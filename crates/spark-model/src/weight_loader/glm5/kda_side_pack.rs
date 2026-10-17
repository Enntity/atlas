// SPDX-License-Identifier: AGPL-3.0-only
//! `ATLAS_GLM_KDA_SIDE_PROJ_LT`: the KDA b / f_a / g_a weights moved into one
//! allocation, rows back to back, so prefill projects them as one
//! `[n_b + 2 * n_a, hidden]` GEMM (`ops::dense_gemm_packed_triple_n`). The
//! three `DenseWeight`s point into it, so every other path reads the same
//! bytes, and the sources are released: the resident size is unchanged.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use super::retirement::{LoadedDense, RetirementLog};
use crate::tp_shard::TpGdnDims;
use crate::weight_map::DenseWeight;

/// The layer's `[b, f_a, g_a]` packed: `b` is this rank's `[local_nk, h]`
/// shard, f_a / g_a (`{prefix}.{f,g}_a_proj.weight`) the replicated `[kd, h]`.
pub(super) fn pack_side<'a>(
    store: &'a WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
    retirement: Option<&'a RetirementLog<'a>>,
    b: LoadedDense<'a>,
    dims: &TpGdnDims,
) -> Result<[DenseWeight; 3]> {
    let side = |name: &str| {
        LoadedDense::load(
            store,
            &format!("{prefix}.{name}.weight"),
            gpu,
            retirement,
            false,
        )
    };
    let a_bytes = dims.kd * dims.h * 2;
    pack(
        gpu,
        [b, side("f_a_proj")?, side("g_a_proj")?],
        [dims.local_nk * dims.h * 2, a_bytes, a_bytes],
    )
}

/// `parts[i]` (`bytes[i]` long) copied to consecutive offsets of one new
/// allocation, then released.
fn pack(
    gpu: &dyn GpuBackend,
    parts: [LoadedDense<'_>; 3],
    bytes: [usize; 3],
) -> Result<[DenseWeight; 3]> {
    let base = gpu.alloc(bytes.iter().sum())?;
    let mut offset = 0;
    let mut packed = [DenseWeight { weight: base }; 3];
    for ((part, len), slot) in parts.into_iter().zip(bytes).zip(&mut packed) {
        slot.weight = base.offset(offset);
        gpu.copy_d2d(part.dense.weight, slot.weight, len)?;
        part.release(gpu)?;
        offset += len;
    }
    Ok(packed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};
    use std::collections::HashMap;

    #[test]
    fn packed_side_weights_hold_the_sources_back_to_back() {
        let gpu = MockGpuBackend::new();
        let sizes = [64usize, 256, 256];
        let names = ["b.weight", "fa.weight", "ga.weight"];
        let mut map = HashMap::new();
        let mut want = Vec::new();
        for (i, (name, bytes)) in names.iter().zip(sizes).enumerate() {
            let data: Vec<u8> = (0..bytes).map(|j| (j * 3 + i * 101) as u8).collect();
            let ptr = gpu.alloc(bytes).unwrap();
            gpu.copy_h2d(&data, ptr).unwrap();
            want.extend_from_slice(&data);
            map.insert(
                (*name).into(),
                WeightTensor {
                    ptr,
                    shape: vec![bytes / 8, 4],
                    dtype: WeightDtype::BF16,
                },
            );
        }
        let store = WeightStore::from_map(map);
        let parts = names.map(|n| LoadedDense::load(&store, n, &gpu, None, false).unwrap());
        let [b, fa, ga] = pack(&gpu, parts, sizes).unwrap();
        assert_eq!(fa.weight, b.weight.offset(sizes[0]));
        assert_eq!(ga.weight, fa.weight.offset(sizes[1]));
        let mut got = vec![0u8; want.len()];
        gpu.copy_d2h(b.weight, &mut got).unwrap();
        assert_eq!(got, want);
    }
}
