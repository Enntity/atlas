// SPDX-License-Identifier: AGPL-3.0-only
//! Packed dense constructor and failure rollback.
use super::*;
use spark_runtime::weights::exl3::native::Exl3LinearNumerics;

impl NativeDenseFfn {
    pub fn new(
        gpu: &dyn GpuBackend,
        weights: [Exl3DeviceShard; 3],
        capacity: usize,
        stream: u64,
        activation: NativeDenseActivation,
        role: NativeDenseFfnRole,
    ) -> Result<Self> {
        Self::setup(gpu, weights, capacity, stream, activation, role, None)
    }

    pub fn new_with_numerics(
        gpu: &dyn GpuBackend,
        weights: [Exl3DeviceShard; 3],
        capacity: usize,
        stream: u64,
        activation: NativeDenseActivation,
        numerics: [Exl3LinearNumerics; 2],
    ) -> Result<Self> {
        Self::setup(
            gpu,
            weights,
            capacity,
            stream,
            activation,
            NativeDenseFfnRole::DensePacked,
            Some(numerics),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn setup(
        gpu: &dyn GpuBackend,
        mut weights: [Exl3DeviceShard; 3],
        capacity: usize,
        stream: u64,
        activation: NativeDenseActivation,
        role: NativeDenseFfnRole,
        numerics: Option<[Exl3LinearNumerics; 2]>,
    ) -> Result<Self> {
        let result = (|| -> Result<_> {
            ensure!(
                role == NativeDenseFfnRole::DensePacked,
                "shared FFN requires global BF16 reconstruction before TP"
            );
            let [g, u, d] = weights.each_ref().map(|w| w.plan());
            ensure!(
                g.axis() == ShardAxis::Output
                    && u.axis() == ShardAxis::Output
                    && d.axis() == ShardAxis::Input,
                "EXL3 native FFN requires output gate/up and input down TP axes"
            );
            ensure!(
                g.in_features() == u.in_features()
                    && g.out_features() == u.out_features()
                    && g.in_features() == d.out_features()
                    && g.out_features() == d.in_features(),
                "EXL3 native FFN local shape mismatch"
            );
            ensure!(
                g.rank() == u.rank()
                    && g.rank() == d.rank()
                    && g.world() == u.world()
                    && g.world() == d.world()
                    && g.output_ranges() == u.output_ranges()
                    && g.output_ranges() == [d.input_range()],
                "EXL3 native FFN TP ownership mismatch"
            );
            activation.validate()?;
            let hidden = g.in_features();
            let intermediate = g.out_features();
            let rank = g.rank();
            let world = g.world();
            let silu = gpu.kernel("exl3", "exl3_silu_mul_dense")?;
            let bytes = capacity
                .checked_mul(intermediate)
                .and_then(|n| n.checked_mul(2))
                .context("EXL3 FFN activation bytes overflow")?;
            if let Some([gate, down]) = numerics {
                ensure!(
                    g.bits() == gate.bits && u.bits() == gate.bits && d.bits() == down.bits,
                    "EXL3 dense numerical bitrate mismatch"
                );
                Exl3LinearWorkspace::preflight_numerics(gpu, hidden, intermediate, capacity, gate)?;
                Exl3LinearWorkspace::preflight_numerics(gpu, intermediate, hidden, capacity, down)?;
            }
            let workspace = |k, n, index| match numerics {
                Some(policies) => Exl3LinearWorkspace::new_with_numerics(
                    gpu,
                    k,
                    n,
                    capacity,
                    stream,
                    policies[index],
                ),
                None => Exl3LinearWorkspace::new_with_input_precision(
                    gpu,
                    k,
                    n,
                    capacity,
                    stream,
                    role.packed_input_precision()?,
                ),
            };
            let gate = workspace(hidden, intermediate, 0)?;
            let down = match workspace(intermediate, hidden, 1) {
                Ok(w) => w,
                Err(error) => {
                    let mut gate = gate;
                    gate.release(gpu)?;
                    return Err(error);
                }
            };
            let mut activations = [DevicePtr::NULL; 2];
            for ptr in &mut activations {
                match gpu.alloc(bytes) {
                    Ok(p) => *ptr = p,
                    Err(error) => {
                        for &p in &activations {
                            if !p.is_null() {
                                gpu.free(p)?;
                            }
                        }
                        let mut gate = gate;
                        let mut down = down;
                        gate.release(gpu)?;
                        down.release(gpu)?;
                        return Err(error);
                    }
                }
            }
            Ok((
                gate,
                down,
                activations,
                silu,
                hidden,
                intermediate,
                rank,
                world,
            ))
        })();
        match result {
            Ok((gate, down, activations, silu, hidden, intermediate, rank, world)) => Ok(Self {
                layer: None,
                capture: None,
                projections: DenseProjections::Packed {
                    weights,
                    workspaces: [gate, down],
                },
                activations,
                merged_activation: DevicePtr::NULL,
                silu,
                activation,
                role,
                hidden,
                intermediate,
                rank,
                world,
                capacity,
                stream,
                active: true,
            }),
            Err(error) => {
                let mut cleanup = None;
                for w in &mut weights {
                    retain_error(&mut cleanup, w.release(gpu));
                }
                Err(match cleanup {
                    None => error,
                    Some(e) => error.context(format!("EXL3 FFN cleanup failed: {e:#}")),
                })
            }
        }
    }
}
