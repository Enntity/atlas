// SPDX-License-Identifier: AGPL-3.0-only

use super::{Exl3DeviceShard, ShardAxis, TpShardPlan};
use crate::weights::exl3::Exl3LinearView;
use safetensors::{Dtype, tensor::TensorView};

struct Fixture {
    k: usize,
    n: usize,
    bits: u8,
    trellis: Vec<u8>,
    input: Vec<u8>,
    output: Vec<u8>,
    marker: [u8; 4],
}

impl Fixture {
    fn new(k: usize, n: usize, bits: u8) -> Self {
        Self {
            k,
            n,
            bits,
            trellis: (0..k * n * usize::from(bits) / 8)
                .map(|i| ((i * 73 + i / 137 + 41) % 256) as u8)
                .collect(),
            input: (0..k)
                .flat_map(|i| {
                    half::f16::from_f32(if i % 3 == 0 { -0.75 } else { 1.25 }).to_le_bytes()
                })
                .collect(),
            output: (0..n)
                .flat_map(|i| {
                    half::f16::from_f32(if i % 5 == 0 { -0.5 } else { 0.25 }).to_le_bytes()
                })
                .collect(),
            marker: 0x83dc_d12du32.to_le_bytes(),
        }
    }

    fn view(&self) -> Exl3LinearView<'_> {
        Exl3LinearView::new(
            TensorView::new(
                Dtype::I16,
                vec![self.k / 16, self.n / 16, 16 * usize::from(self.bits)],
                &self.trellis,
            )
            .unwrap(),
            TensorView::new(Dtype::F16, vec![self.k], &self.input).unwrap(),
            TensorView::new(Dtype::F16, vec![self.n], &self.output).unwrap(),
            TensorView::new(Dtype::I32, vec![], &self.marker).unwrap(),
            self.k,
            self.n,
        )
        .unwrap()
    }
}

#[test]
fn tp2_copies_the_correct_trellis_axes_and_companion_scales() {
    for bits in [4, 5, 6] {
        let fixture = Fixture::new(256, 512, bits);
        let full = fixture.view();
        for rank in 0..2 {
            for axis in [ShardAxis::Input, ShardAxis::Output] {
                let plan = TpShardPlan::new(&full, axis, rank, 2).unwrap();
                let owned = plan.materialize(&full).unwrap();
                let local = owned.view().unwrap();
                assert_eq!(local.bits(), bits);
                let input = plan.input_range();
                let output = plan.output_ranges()[0].clone();
                assert_eq!(
                    local.input_scale().data(),
                    &fixture.input[2 * input.start..2 * input.end]
                );
                assert_eq!(
                    local.output_scale().data(),
                    &fixture.output[2 * output.start..2 * output.end]
                );
                for it in 0..local.in_features() / 16 {
                    for ot in 0..local.out_features() / 16 {
                        assert_eq!(
                            local.tile_states(it, ot).unwrap(),
                            full.tile_states(it + input.start / 16, ot + output.start / 16)
                                .unwrap()
                        );
                    }
                }
                assert_eq!(owned.trellis_bytes().len(), fixture.trellis.len() / 2);
            }
        }
    }
}

#[test]
fn fused_qkv_shards_each_component_instead_of_the_whole_output() {
    let fixture = Fixture::new(256, 768, 6);
    let full = fixture.view();
    for rank in 0..2 {
        let plan = TpShardPlan::new_output_segments(&full, &[256, 256, 256], rank, 2).unwrap();
        assert_eq!(
            plan.output_ranges(),
            &[
                rank * 128..rank * 128 + 128,
                256 + rank * 128..384 + rank * 128,
                512 + rank * 128..640 + rank * 128,
            ]
        );
        let owned = plan.materialize(&full).unwrap();
        let local = owned.view().unwrap();
        assert_eq!(local.out_features(), 384);
        for (segment, range) in plan.output_ranges().iter().enumerate() {
            for tile in 0..8 {
                assert_eq!(
                    local.tile_states(5, segment * 8 + tile).unwrap(),
                    full.tile_states(5, range.start / 16 + tile).unwrap()
                );
            }
        }
    }
}

#[test]
fn invalid_topologies_and_splits_fail_before_allocating() {
    let fixture = Fixture::new(256, 384, 4);
    let full = fixture.view();
    for (axis, rank, world) in [
        (ShardAxis::Input, 0, 0),
        (ShardAxis::Input, 2, 2),
        (ShardAxis::Replicated, 1, 1),
        (ShardAxis::Output, 0, 2),
        (ShardAxis::Input, 0, 3),
    ] {
        assert!(TpShardPlan::new(&full, axis, rank, world).is_err());
    }
    for segments in [
        &[][..],
        &[128, 128][..],
        &[192, 192][..],
        &[usize::MAX, 1][..],
    ] {
        assert!(TpShardPlan::new_output_segments(&full, segments, 0, 2).is_err());
    }
}

#[test]
fn replicated_and_tp1_shards_preserve_all_bytes() {
    let fixture = Fixture::new(128, 256, 5);
    let full = fixture.view();
    for (axis, rank, world) in [
        (ShardAxis::Replicated, 1, 2),
        (ShardAxis::Input, 0, 1),
        (ShardAxis::Output, 0, 1),
    ] {
        let local = TpShardPlan::new(&full, axis, rank, world)
            .unwrap()
            .materialize(&full)
            .unwrap();
        assert_eq!(local.trellis_bytes(), fixture.trellis);
        assert_eq!(local.view().unwrap().input_scale().data(), fixture.input);
        assert_eq!(local.view().unwrap().output_scale().data(), fixture.output);
    }
}

#[test]
fn upload_only_local_bytes_and_teardown_is_retryable() {
    use crate::gpu::{GpuBackend, mock::MockGpuBackend};
    use atlas_core::scope::ModelResource;
    let fixture = Fixture::new(256, 768, 6);
    let view = fixture.view();
    for rank in 0..2 {
        for axis in [ShardAxis::Input, ShardAxis::Output] {
            let gpu = MockGpuBackend::new();
            let plan = TpShardPlan::new(&view, axis, rank, 2).unwrap();
            let expected = plan.materialize(&view).unwrap();
            let expected_view = expected.view().unwrap();
            let mut device = Exl3DeviceShard::upload(&gpu, &view, &plan).unwrap();
            assert_eq!(gpu.alloc_count(), 3);
            for (buffer, expected) in device.buffers().unwrap().into_iter().zip([
                expected.trellis_bytes(),
                expected_view.input_scale().data(),
                expected_view.output_scale().data(),
            ]) {
                let mut actual = vec![0; expected.len()];
                gpu.copy_d2h(buffer, &mut actual).unwrap();
                assert_eq!(actual, expected);
            }
            gpu.fail_next_free();
            assert!(device.release(&gpu).is_err());
            assert_eq!(gpu.alloc_count(), 1);
            assert!(device.buffers().is_err());
            device.release(&gpu).unwrap();
            device.release(&gpu).unwrap();
            assert_eq!(gpu.alloc_count(), 0);
        }
    }
}

#[test]
fn mismatched_source_is_rejected_before_device_allocation() {
    use crate::gpu::mock::MockGpuBackend;
    let original = Fixture::new(256, 512, 4);
    let plan = TpShardPlan::new(&original.view(), ShardAxis::Output, 1, 2).unwrap();
    for fixture in [Fixture::new(256, 512, 5), Fixture::new(512, 512, 4)] {
        let gpu = MockGpuBackend::new();
        assert!(plan.materialize(&fixture.view()).is_err());
        assert!(Exl3DeviceShard::upload(&gpu, &fixture.view(), &plan).is_err());
        assert_eq!(gpu.alloc_count(), 0);
    }
}

fn input_batch(k: usize, batch: usize) -> Vec<f32> {
    (0..k * batch)
        .map(|i| ((i * 31 % 97) as f32 - 48.0) / 512.0)
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && e.is_finite());
        assert!(
            (a - e).abs() <= 2e-4 + 2e-5 * e.abs(),
            "element {index}: {a} vs {e}"
        );
    }
}

#[test]
fn tp2_column_gather_and_row_sum_match_unsplit_linear() {
    use crate::weights::exl3::reference_linear;
    for bits in [4, 5, 6] {
        let fixture = Fixture::new(256, 512, bits);
        let full = fixture.view();
        let batch = 3;
        let input = input_batch(fixture.k, batch);
        let expected = reference_linear(&full, &input, batch).unwrap();
        for axis in [ShardAxis::Input, ShardAxis::Output] {
            let mut combined = vec![0.0; batch * fixture.n];
            for rank in 0..2 {
                let plan = TpShardPlan::new(&full, axis, rank, 2).unwrap();
                let shard = plan.materialize(&full).unwrap();
                let selected = plan.input_range();
                let local_input: Vec<f32> = input
                    .chunks_exact(fixture.k)
                    .flat_map(|row| row[selected.clone()].iter().copied())
                    .collect();
                let local = reference_linear(&shard.view().unwrap(), &local_input, batch).unwrap();
                for (row, values) in local.chunks_exact(plan.out_features()).enumerate() {
                    match axis {
                        ShardAxis::Input => {
                            for (index, value) in values.iter().enumerate() {
                                combined[row * fixture.n + index] += value;
                            }
                        }
                        ShardAxis::Output => {
                            let range = &plan.output_ranges()[0];
                            combined[row * fixture.n + range.start..row * fixture.n + range.end]
                                .copy_from_slice(values);
                        }
                        ShardAxis::Replicated => unreachable!(),
                    }
                }
            }
            assert_close(&combined, &expected);
        }
    }
}

#[test]
fn fused_qkv_tp2_preserves_global_component_order() {
    use crate::weights::exl3::reference_linear;
    let fixture = Fixture::new(256, 768, 6);
    let full = fixture.view();
    let input = input_batch(256, 2);
    let expected = reference_linear(&full, &input, 2).unwrap();
    let mut combined = vec![0.0; expected.len()];
    for rank in 0..2 {
        let plan = TpShardPlan::new_output_segments(&full, &[256, 256, 256], rank, 2).unwrap();
        let shard = plan.materialize(&full).unwrap();
        let local = reference_linear(&shard.view().unwrap(), &input, 2).unwrap();
        for (batch, values) in local.chunks_exact(plan.out_features()).enumerate() {
            let mut offset = 0;
            for range in plan.output_ranges() {
                combined[batch * 768 + range.start..batch * 768 + range.end]
                    .copy_from_slice(&values[offset..offset + range.len()]);
                offset += range.len();
            }
        }
    }
    assert_close(&combined, &expected);
}

#[test]
fn tp2_gate_up_activation_down_matches_unsplit_ffn() {
    use crate::weights::exl3::reference_linear;
    let gate = Fixture::new(256, 256, 4);
    let up = Fixture::new(256, 256, 5);
    let down = Fixture::new(256, 256, 6);
    let input = input_batch(256, 2);
    let activate = |gate: &[f32], up: &[f32]| -> Vec<f32> {
        gate.iter()
            .zip(up)
            .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
            .collect()
    };
    let full_gate = reference_linear(&gate.view(), &input, 2).unwrap();
    let full_up = reference_linear(&up.view(), &input, 2).unwrap();
    let expected = reference_linear(&down.view(), &activate(&full_gate, &full_up), 2).unwrap();
    let mut combined = vec![0.0; expected.len()];
    for rank in 0..2 {
        let g = TpShardPlan::new(&gate.view(), ShardAxis::Output, rank, 2)
            .unwrap()
            .materialize(&gate.view())
            .unwrap();
        let u = TpShardPlan::new(&up.view(), ShardAxis::Output, rank, 2)
            .unwrap()
            .materialize(&up.view())
            .unwrap();
        let d = TpShardPlan::new(&down.view(), ShardAxis::Input, rank, 2)
            .unwrap()
            .materialize(&down.view())
            .unwrap();
        let local_gate = reference_linear(&g.view().unwrap(), &input, 2).unwrap();
        let local_up = reference_linear(&u.view().unwrap(), &input, 2).unwrap();
        let output =
            reference_linear(&d.view().unwrap(), &activate(&local_gate, &local_up), 2).unwrap();
        for (value, partial) in combined.iter_mut().zip(output) {
            *value += partial;
        }
    }
    assert_close(&combined, &expected);
}

#[test]
fn selected_fused_component_reuses_segment_validation_and_preserves_source_tiles() {
    let fixture = Fixture::new(256, 1024, 4);
    let view = fixture.view();
    for rank in 0..2 {
        for component in 0..3 {
            let plan =
                TpShardPlan::new_output_component(&view, &[256, 256, 512], component, rank, 2)
                    .unwrap();
            let fused = TpShardPlan::new_output_segments(&view, &[256, 256, 512], rank, 2).unwrap();
            let expected = fused.output_ranges()[component].clone();
            assert_eq!(plan.output_ranges(), std::slice::from_ref(&expected));
            assert_eq!(plan.out_features(), expected.len());
            let owned = plan.materialize(&view).unwrap();
            let local = owned.view().unwrap();
            assert_eq!(
                local.output_scale().data(),
                &fixture.output[expected.start * 2..expected.end * 2]
            );
            for it in 0..local.in_features() / 16 {
                for ot in 0..local.out_features() / 16 {
                    assert_eq!(
                        local.tile_states(it, ot).unwrap(),
                        view.tile_states(it, ot + expected.start / 16).unwrap()
                    );
                }
            }
        }
    }
    assert!(TpShardPlan::new_output_component(&view, &[256, 256, 512], 3, 0, 2).is_err());
    assert!(TpShardPlan::new_output_component(&view, &[256, 256], 0, 0, 2).is_err());
    assert!(TpShardPlan::new_output_component(&view, &[128, 384, 512], 0, 0, 2).is_err());
}
