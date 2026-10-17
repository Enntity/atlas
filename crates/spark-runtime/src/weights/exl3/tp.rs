// SPDX-License-Identifier: AGPL-3.0-only

//! Hadamard-aligned tensor-parallel partitioning of checkpoint-native EXL3.

use std::ops::Range;

use anyhow::{Context, Result, ensure};
use safetensors::{Dtype, tensor::TensorView};

use super::{Exl3LinearView, HADAMARD_WIDTH, MUL1_MULTIPLIER, TILE_WIDTH};

mod device;
pub use device::Exl3DeviceShard;

/// Logical partition of a linear, independent of on-disk tensor axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardAxis {
    Replicated,
    /// Row parallel: split inputs; sum rank-local outputs.
    Input,
    /// Column parallel: split outputs; preserve the global output order.
    Output,
}

/// Validated slicing plan, shared by host materialization and device upload.
///
/// Trellis storage is input-major, unlike the usual `[output,input]` weight.
/// All boundaries must align to H128, not merely the 16×16 storage tile.
#[derive(Debug, Clone)]
pub struct TpShardPlan {
    axis: ShardAxis,
    rank: usize,
    world: usize,
    full_input: usize,
    full_output: usize,
    bits: u8,
    input: Range<usize>,
    output: Vec<Range<usize>>,
    local_output: usize,
}

impl TpShardPlan {
    pub fn new(
        view: &Exl3LinearView<'_>,
        axis: ShardAxis,
        rank: usize,
        world: usize,
    ) -> Result<Self> {
        validate_rank(rank, world)?;
        let k = view.in_features();
        let n = view.out_features();
        let input = if axis == ShardAxis::Input {
            aligned_partition(k, rank, world)?
        } else {
            0..k
        };
        let output = if axis == ShardAxis::Output {
            aligned_partition(n, rank, world)?
        } else {
            0..n
        };
        Ok(Self {
            axis,
            rank,
            world,
            full_input: k,
            full_output: n,
            bits: view.bits(),
            input,
            local_output: output.len(),
            output: vec![output],
        })
    }

    /// Split each fused projection separately, e.g. full `[Q,K,V]` becomes
    /// `[Q_rank,K_rank,V_rank]`. A flat split would assign entire components
    /// to different ranks and break attention head ownership.
    pub fn new_output_segments(
        view: &Exl3LinearView<'_>,
        segments: &[usize],
        rank: usize,
        world: usize,
    ) -> Result<Self> {
        validate_rank(rank, world)?;
        ensure!(
            !segments.is_empty(),
            "EXL3 output segments must be explicit"
        );
        let mut offset = 0usize;
        let mut output = Vec::with_capacity(segments.len());
        for &width in segments {
            let local = aligned_partition(width, rank, world)?;
            let end = offset
                .checked_add(width)
                .context("EXL3 output segments overflow")?;
            ensure!(
                end <= view.out_features(),
                "EXL3 output segments exceed full width"
            );
            output.push(offset + local.start..offset + local.end);
            offset = end;
        }
        ensure!(
            offset == view.out_features(),
            "EXL3 output segments must cover full width"
        );
        Ok(Self {
            axis: ShardAxis::Output,
            rank,
            world,
            full_input: view.in_features(),
            full_output: view.out_features(),
            bits: view.bits(),
            input: 0..view.in_features(),
            local_output: view.out_features() / world,
            output,
        })
    }

    /// Select one component from a fused projection without decoding or
    /// repacking. Uses the full segment validation so component ownership is
    /// identical to `new_output_segments`, including H128 TP alignment.
    pub fn new_output_component(
        view: &Exl3LinearView<'_>,
        segments: &[usize],
        component: usize,
        rank: usize,
        world: usize,
    ) -> Result<Self> {
        let mut plan = Self::new_output_segments(view, segments, rank, world)?;
        let range = plan
            .output
            .get(component)
            .context("EXL3 fused component index out of range")?
            .clone();
        plan.local_output = range.len();
        plan.output = vec![range];
        Ok(plan)
    }

    pub fn axis(&self) -> ShardAxis {
        self.axis
    }
    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn world(&self) -> usize {
        self.world
    }
    pub fn input_range(&self) -> Range<usize> {
        self.input.clone()
    }
    pub fn output_ranges(&self) -> &[Range<usize>] {
        &self.output
    }
    pub fn in_features(&self) -> usize {
        self.input.len()
    }
    pub fn out_features(&self) -> usize {
        self.local_output
    }
    pub fn bits(&self) -> u8 {
        self.bits
    }

    pub fn trellis_byte_size(&self) -> usize {
        self.in_features() / TILE_WIDTH
            * (self.out_features() / TILE_WIDTH)
            * 32
            * usize::from(self.bits)
    }

    pub(super) fn validate_source(&self, view: &Exl3LinearView<'_>) -> Result<()> {
        ensure!(
            view.in_features() == self.full_input
                && view.out_features() == self.full_output
                && view.bits() == self.bits,
            "EXL3 shard plan does not match source dimensions/bitrate"
        );
        Ok(())
    }

    /// Visit only the selected checkpoint ranges, in local storage order.
    /// Each callback receives a local destination offset and borrowed source
    /// bytes, allowing IORouters to upload directly without a full device copy.
    pub(super) fn visit_trellis(
        &self,
        view: &Exl3LinearView<'_>,
        mut copy: impl FnMut(usize, &[u8]) -> Result<()>,
    ) -> Result<()> {
        self.validate_source(view)?;
        let bytes_per_tile = 32 * usize::from(self.bits);
        let full_row_bytes = self.full_output / TILE_WIDTH * bytes_per_tile;
        let local_row_bytes = self.local_output / TILE_WIDTH * bytes_per_tile;
        // Complete output rows are contiguous: input-parallel and replication.
        if self.output.len() == 1 && self.output[0] == (0..self.full_output) {
            let start = self.input.start / TILE_WIDTH * full_row_bytes;
            return copy(
                0,
                &view.trellis().data()[start..start + self.trellis_byte_size()],
            );
        }
        for (local_it, global_it) in
            (self.input.start / TILE_WIDTH..self.input.end / TILE_WIDTH).enumerate()
        {
            let mut dst = local_it * local_row_bytes;
            for range in &self.output {
                let src = global_it * full_row_bytes + range.start / TILE_WIDTH * bytes_per_tile;
                let len = range.len() / TILE_WIDTH * bytes_per_tile;
                copy(dst, &view.trellis().data()[src..src + len])?;
                dst += len;
            }
        }
        Ok(())
    }

    pub(super) fn visit_output_scale(
        &self,
        view: &Exl3LinearView<'_>,
        mut copy: impl FnMut(usize, &[u8]) -> Result<()>,
    ) -> Result<()> {
        self.validate_source(view)?;
        let mut dst = 0;
        for range in &self.output {
            copy(
                dst,
                &view.output_scale().data()[range.start * 2..range.end * 2],
            )?;
            dst += range.len() * 2;
        }
        Ok(())
    }

    pub fn materialize(&self, view: &Exl3LinearView<'_>) -> Result<OwnedExl3Shard> {
        self.validate_source(view)?;
        let mut trellis = vec![0; self.trellis_byte_size()];
        self.visit_trellis(view, |offset, bytes| {
            trellis[offset..offset + bytes.len()].copy_from_slice(bytes);
            Ok(())
        })?;
        let mut output = vec![0; self.out_features() * 2];
        self.visit_output_scale(view, |offset, bytes| {
            output[offset..offset + bytes.len()].copy_from_slice(bytes);
            Ok(())
        })?;
        Ok(OwnedExl3Shard {
            trellis,
            input: view.input_scale().data()[self.input.start * 2..self.input.end * 2].to_vec(),
            output,
            k: self.in_features(),
            n: self.out_features(),
            bits: self.bits,
        })
    }
}

fn validate_rank(rank: usize, world: usize) -> Result<()> {
    ensure!(
        world > 0 && rank < world,
        "EXL3 invalid TP rank={rank}, world={world}"
    );
    Ok(())
}

fn aligned_partition(width: usize, rank: usize, world: usize) -> Result<Range<usize>> {
    ensure!(
        width > 0 && width.is_multiple_of(world),
        "EXL3 width {width} does not split {world} ways"
    );
    let local = width / world;
    ensure!(
        local.is_multiple_of(HADAMARD_WIDTH),
        "EXL3 local width {local} must align to H{HADAMARD_WIDTH}"
    );
    Ok(rank * local..(rank + 1) * local)
}

/// Owned host shard for an IORouter or the reconstruction oracle. The scalar
/// marker was validated before slicing and is constant for every mul1 shard.
pub struct OwnedExl3Shard {
    trellis: Vec<u8>,
    input: Vec<u8>,
    output: Vec<u8>,
    k: usize,
    n: usize,
    bits: u8,
}

impl OwnedExl3Shard {
    pub fn trellis_bytes(&self) -> &[u8] {
        &self.trellis
    }

    pub fn view(&self) -> Result<Exl3LinearView<'_>> {
        const MARKER: [u8; 4] = MUL1_MULTIPLIER.to_le_bytes();
        Exl3LinearView::new(
            TensorView::new(
                Dtype::I16,
                vec![
                    self.k / TILE_WIDTH,
                    self.n / TILE_WIDTH,
                    TILE_WIDTH * usize::from(self.bits),
                ],
                &self.trellis,
            )?,
            TensorView::new(Dtype::F16, vec![self.k], &self.input)?,
            TensorView::new(Dtype::F16, vec![self.n], &self.output)?,
            TensorView::new(Dtype::I32, vec![], &MARKER)?,
            self.k,
            self.n,
        )
    }
}

#[cfg(test)]
mod tests;
