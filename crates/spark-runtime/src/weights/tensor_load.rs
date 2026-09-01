// SPDX-License-Identifier: AGPL-3.0-only

//! Load-time tensor slicing for appliance-specialized parallel topologies.
//!
//! GLM-5.3-Flash cannot afford to upload replicated tensors and make a second
//! TP copy afterwards. This policy slices both the dense trunk and every EXL3
//! expert on the host before GPU allocation, so each Spark owns exactly one
//! TP2 half of the model and both ranks execute every routed expert.

use std::borrow::Cow;

use anyhow::{Result, ensure};

use super::WeightDtype;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TensorLoadPolicy {
    #[default]
    Replicated,
    /// Exact two-Spark GLM-5.3 layout: pure TP2 for the trunk, shared expert,
    /// and all 288 EXL3 routed experts. There is deliberately no EP variant.
    Glm53Tp2 { rank: usize },
}

pub struct PreparedTensor<'a> {
    pub data: Cow<'a, [u8]>,
    pub shape: Vec<usize>,
}

impl TensorLoadPolicy {
    pub fn validate(self) -> Result<()> {
        if let Self::Glm53Tp2 { rank } = self {
            ensure!(rank < 2, "GLM-5.3 TP2 rank {rank} is outside 0..2");
        }
        Ok(())
    }

    /// Exact resident byte count after applying this load policy.
    pub fn resident_bytes(self, name: &str, source_bytes: usize) -> usize {
        if self.shard_axis(name).is_some() {
            source_bytes / 2
        } else {
            source_bytes
        }
    }

    pub fn prepare<'a>(
        self,
        name: &str,
        shape: &[usize],
        dtype: WeightDtype,
        source: &'a [u8],
    ) -> Result<PreparedTensor<'a>> {
        self.validate()?;
        let Some(axis) = self.shard_axis(name) else {
            return Ok(PreparedTensor {
                data: Cow::Borrowed(source),
                shape: shape.to_vec(),
            });
        };
        ensure!(
            source.len().is_multiple_of(2),
            "GLM-5.3 TP2 tensor `{name}` has an odd byte length"
        );
        let rank = match self {
            Self::Glm53Tp2 { rank } => rank,
            Self::Replicated => unreachable!("replicated tensors have no shard axis"),
        };
        axis_half(name, shape, dtype, source, rank, axis)
    }

    fn shard_axis(self, name: &str) -> Option<usize> {
        if !matches!(self, Self::Glm53Tp2 { .. })
            || !name.starts_with("model.language_model.layers.")
        {
            return None;
        }

        if name.contains(".mlp.experts.") {
            // Mia's EXL3 layout is tiled [K/16, N/16, 16*bits]. TP2 narrows
            // gate/up on N and down on K, exactly matching its vLLM loader.
            if name.ends_with(".gate_proj.trellis") || name.ends_with(".up_proj.trellis") {
                return Some(1);
            }
            if name.ends_with(".down_proj.trellis")
                || name.ends_with(".gate_proj.svh")
                || name.ends_with(".up_proj.svh")
                || name.ends_with(".down_proj.suh")
            {
                return Some(0);
            }
            return None;
        }

        if name.ends_with(".self_attn.o_proj.weight")
            || name.ends_with(".mlp.down_proj.weight")
            || name.ends_with(".mlp.shared_experts.down_proj.weight")
        {
            return Some(1);
        }

        if [
            ".self_attn.q_proj.weight",
            ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight",
            ".self_attn.b_proj.weight",
            ".self_attn.f_b_proj.weight",
            ".self_attn.g_b_proj.weight",
            ".self_attn.q_conv1d.weight",
            ".self_attn.k_conv1d.weight",
            ".self_attn.v_conv1d.weight",
            ".self_attn.dt_bias",
            ".self_attn.A_log",
            ".self_attn.q_b_proj.weight",
            ".self_attn.kv_b_proj.weight",
            ".mlp.gate_proj.weight",
            ".mlp.up_proj.weight",
            ".mlp.shared_experts.gate_proj.weight",
            ".mlp.shared_experts.up_proj.weight",
        ]
        .iter()
        .any(|suffix| name.ends_with(suffix))
        {
            return Some(0);
        }
        None
    }
}

fn axis_half<'a>(
    name: &str,
    shape: &[usize],
    _dtype: WeightDtype,
    source: &'a [u8],
    rank: usize,
    axis: usize,
) -> Result<PreparedTensor<'a>> {
    ensure!(
        axis < shape.len(),
        "GLM-5.3 TP2 axis {axis} is outside tensor `{name}` shape {shape:?}"
    );
    let dimension = shape[axis];
    ensure!(
        dimension.is_multiple_of(2),
        "GLM-5.3 TP2 axis {axis} of `{name}` ({dimension}) is not divisible by 2"
    );
    let elements = shape
        .iter()
        .try_fold(1usize, |product, value| product.checked_mul(*value));
    let elements = elements.ok_or_else(|| anyhow::anyhow!("tensor `{name}` shape overflow"))?;
    ensure!(
        elements > 0 && source.len().is_multiple_of(elements),
        "GLM-5.3 TP2 tensor `{name}` byte count does not match shape {shape:?}"
    );
    let element_bytes = source.len() / elements;
    let inner = shape[axis + 1..].iter().product::<usize>();
    let outer = shape[..axis].iter().product::<usize>();
    let local_dimension = dimension / 2;
    let local_run_bytes = local_dimension * inner * element_bytes;
    let source_run_bytes = dimension * inner * element_bytes;
    let rank_offset = rank * local_run_bytes;
    let mut local_shape = shape.to_vec();
    local_shape[axis] = local_dimension;
    if axis == 0 {
        return Ok(PreparedTensor {
            data: Cow::Borrowed(&source[rank_offset..rank_offset + local_run_bytes]),
            shape: local_shape,
        });
    }
    let mut local = Vec::with_capacity(source.len() / 2);
    for item in 0..outer {
        let start = item * source_run_bytes + rank_offset;
        local.extend_from_slice(&source[start..start + local_run_bytes]);
    }
    Ok(PreparedTensor {
        data: Cow::Owned(local),
        shape: local_shape,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROOT: &str = "model.language_model.layers.3";

    #[test]
    fn glm_policy_slices_column_and_row_parallel_bytes() {
        let bytes = (0_u16..24).flat_map(u16::to_le_bytes).collect::<Vec<_>>();
        let col = TensorLoadPolicy::Glm53Tp2 { rank: 1 }
            .prepare(
                &format!("{ROOT}.self_attn.q_b_proj.weight"),
                &[6, 4],
                WeightDtype::BF16,
                &bytes,
            )
            .unwrap();
        assert_eq!(col.shape, [3, 4]);
        assert_eq!(&*col.data, &bytes[24..]);

        let row = TensorLoadPolicy::Glm53Tp2 { rank: 1 }
            .prepare(
                &format!("{ROOT}.self_attn.o_proj.weight"),
                &[6, 4],
                WeightDtype::BF16,
                &bytes,
            )
            .unwrap();
        assert_eq!(row.shape, [6, 2]);
        let values = row
            .data
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect::<Vec<_>>();
        assert_eq!(values, [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23]);
    }

    #[test]
    fn routed_experts_are_tp2_while_indexer_stays_replicated() {
        let policy = TensorLoadPolicy::Glm53Tp2 { rank: 0 };
        assert_eq!(
            policy.resident_bytes(&format!("{ROOT}.mlp.experts.7.gate_proj.trellis"), 1024),
            512
        );
        assert_eq!(
            policy.resident_bytes(&format!("{ROOT}.self_attn.indexer.wq_b.weight"), 1024),
            1024
        );
        assert_eq!(
            policy.resident_bytes(&format!("{ROOT}.mlp.shared_experts.up_proj.weight"), 1024),
            512
        );
    }

    #[test]
    fn kda_scalar_vectors_preserve_f32_and_slice_by_head() {
        let values = (0_u32..8).collect::<Vec<_>>();
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let tensor = TensorLoadPolicy::Glm53Tp2 { rank: 1 }
            .prepare(
                &format!("{ROOT}.self_attn.A_log"),
                &[8],
                WeightDtype::FP32,
                &bytes,
            )
            .unwrap();
        assert_eq!(tensor.shape, [4]);
        assert_eq!(&*tensor.data, &bytes[16..]);
    }

    #[test]
    fn exl3_trellis_uses_projection_specific_axis() {
        let values = (0_u16..32).flat_map(u16::to_le_bytes).collect::<Vec<_>>();
        let rank1 = TensorLoadPolicy::Glm53Tp2 { rank: 1 };
        let gate = rank1
            .prepare(
                &format!("{ROOT}.mlp.experts.7.gate_proj.trellis"),
                &[2, 4, 4],
                WeightDtype::Int16,
                &values,
            )
            .unwrap();
        assert_eq!(gate.shape, [2, 2, 4]);
        let gate_values = gate
            .data
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect::<Vec<_>>();
        assert_eq!(
            gate_values,
            [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]
        );

        let down = rank1
            .prepare(
                &format!("{ROOT}.mlp.experts.7.down_proj.trellis"),
                &[4, 2, 4],
                WeightDtype::Int16,
                &values,
            )
            .unwrap();
        assert_eq!(down.shape, [2, 2, 4]);
        assert_eq!(&*down.data, &values[32..]);
    }
}
