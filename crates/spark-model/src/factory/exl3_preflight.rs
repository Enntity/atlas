// SPDX-License-Identifier: AGPL-3.0-only
//! Packed arithmetic admission before checkpoint uploads or model assembly.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::{gpu::GpuBackend, weights::exl3::native::Exl3LinearWorkspace};

use crate::weight_loader::{GLM53_TP2_HEAD_BITS, glm53_tp2_packed_numerics};

/// Admit the pinned TP2 arithmetic and resolve its complete packed kernel
/// family without uploading checkpoint tensors or allocating device memory.
pub fn preflight_glm53_tp2_packed(
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    capacity: usize,
) -> Result<()> {
    ensure!(
        config.model_type == "glm5_next"
            && config.tp_world_size == 2
            && config.tp_rank < config.tp_world_size
            && config.ep_world_size == 1
            && config.ep_rank == 0
            && !config.expert_tp,
        "Pinned GLM-5.3 EXL3 arithmetic requires TP=2, EP=1 and a valid TP rank"
    );
    ensure!(
        config.vocab_size > 0 && config.vocab_size.is_multiple_of(config.tp_world_size),
        "EXL3 physical vocabulary must partition evenly across TP ranks"
    );
    let local_vocab = config.vocab_size / config.tp_world_size;
    let numerics = glm53_tp2_packed_numerics(
        config.tp_world_size,
        config.hidden_size,
        local_vocab,
        GLM53_TP2_HEAD_BITS,
        local_vocab,
        0,
    )?;
    Exl3LinearWorkspace::preflight_numerics(
        gpu,
        config.hidden_size,
        local_vocab,
        capacity,
        numerics,
    )
}

#[cfg(test)]
mod tests {
    use super::preflight_glm53_tp2_packed;
    use atlas_core::config::ModelConfig;
    use spark_runtime::gpu::mock::MockGpuBackend;

    fn config() -> ModelConfig {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "glm5_next".into();
        config.hidden_size = 4096;
        config.vocab_size = 154880;
        config.tp_world_size = 2;
        config.tp_rank = 0;
        config.ep_world_size = 1;
        config.ep_rank = 0;
        config.expert_tp = false;
        config
    }

    #[test]
    fn exl3_preflight_resolves_complete_packed_family_without_allocation() {
        let gpu = MockGpuBackend::new();
        preflight_glm53_tp2_packed(&config(), &gpu, 128).unwrap();
        assert_eq!(gpu.alloc_count(), 0);
        assert_eq!(gpu.launch_count(), 0);
        assert_eq!(
            gpu.kernel_requests_snapshot(),
            vec![
                ("exl3".into(), "exl3_input_hadamard_f16".into()),
                ("exl3_linear_mma".into(), "exl3_mul1_matmul_mma".into()),
                ("exl3".into(), "exl3_output_hadamard".into()),
            ]
        );
    }

    #[test]
    fn exl3_preflight_rejects_unqualified_topology_and_geometry_before_gpu_requests() {
        let base = config();
        for invalid in [
            ModelConfig {
                tp_world_size: 1,
                ..base.clone()
            },
            ModelConfig {
                tp_rank: 2,
                ..base.clone()
            },
            ModelConfig {
                ep_world_size: 2,
                ..base.clone()
            },
            ModelConfig {
                ep_rank: 1,
                ..base.clone()
            },
            ModelConfig {
                expert_tp: true,
                ..base.clone()
            },
            ModelConfig {
                hidden_size: 2048,
                ..base.clone()
            },
            ModelConfig {
                vocab_size: 154881,
                ..base.clone()
            },
            ModelConfig {
                model_type: "qwen3".into(),
                ..base.clone()
            },
        ] {
            let gpu = MockGpuBackend::new();
            assert!(preflight_glm53_tp2_packed(&invalid, &gpu, 128).is_err());
            assert!(gpu.kernel_requests_snapshot().is_empty());
            assert_eq!(gpu.alloc_count(), 0);
        }
    }

    #[test]
    fn exl3_preflight_missing_export_or_bad_capacity_never_allocates() {
        for capacity in [0, 65536] {
            let gpu = MockGpuBackend::new();
            assert!(preflight_glm53_tp2_packed(&config(), &gpu, capacity).is_err());
            assert!(gpu.kernel_requests_snapshot().is_empty());
        }
        let gpu = MockGpuBackend::new();
        gpu.fail_next_kernel();
        assert!(preflight_glm53_tp2_packed(&config(), &gpu, 128).is_err());
        assert_eq!(gpu.alloc_count(), 0);
        assert_eq!(gpu.launch_count(), 0);
    }
}
