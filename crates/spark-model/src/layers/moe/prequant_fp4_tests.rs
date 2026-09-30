// SPDX-License-Identifier: AGPL-3.0-only

use super::{c3_grouped_shape, c4_grouped_shape, compact_gate_up_worklist_bytes, k128w_schedule};
use crate::layers::ops::K128wSchedule;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::DevicePtr;

fn glm_config() -> ModelConfig {
    ModelConfig {
        model_type: "glm5_next".into(),
        hidden_size: 4096,
        moe_intermediate_size: 2048,
        shared_expert_intermediate_size: 2048,
        num_experts: 288,
        num_experts_per_tok: 8,
        tp_world_size: 2,
        ep_world_size: 2,
        scoring_func: "sigmoid".into(),
        ..ModelConfig::qwen3_next_80b_nvfp4()
    }
}

#[test]
fn c3_shape_excludes_single_session_verifiers_and_other_widths() {
    let config = glm_config();
    assert!(c3_grouped_shape(&config, 3, 3));
    assert!(!c3_grouped_shape(&config, 3, 1));
    for rows in [1, 2, 4, 5] {
        assert!(!c3_grouped_shape(&config, rows, 3));
    }
}

#[test]
fn c3_shape_requires_exact_model_expert_layout_and_topology() {
    let mutations: [fn(&mut ModelConfig); 8] = [
        |c| c.model_type = "deepseek_v4".into(),
        |c| c.num_experts = 256,
        |c| c.num_experts_per_tok = 4,
        |c| c.moe_intermediate_size = 1024,
        |c| c.shared_expert_intermediate_size = 4096,
        |c| c.tp_world_size = 1,
        |c| c.ep_world_size = 1,
        |c| c.scoring_func = "sqrtsoftplus".into(),
    ];
    for mutate in mutations {
        let mut config = glm_config();
        mutate(&mut config);
        assert!(!c3_grouped_shape(&config, 3, 3));
    }
}

#[test]
fn compact_worklist_covers_every_distinct_route_without_new_scratch() {
    // Every routed token may choose different experts. Each receives one
    // M64 tile and 16 N128 tiles; each item occupies two u32 words.
    assert_eq!(compact_gate_up_worklist_bytes(3, 8, 2048), 16 + 24 * 16 * 8);
    assert_eq!(compact_gate_up_worklist_bytes(5, 8, 2048), 16 + 40 * 16 * 8);
    assert!(compact_gate_up_worklist_bytes(3, 8, 2048) <= 4096 * 4);
}

#[test]
fn three_row_router_scratch_fits_sort_metadata_but_two_rows_do_not() {
    // sorted token/expert IDs, expert offsets, then token-to-permutation.
    // This is why the prototype must not simply accept every tiny batch.
    let sorted_bytes = |rows: usize| 3 * rows * 8 * 4 + (288 + 1) * 4;
    assert!(sorted_bytes(3) <= 3 * 288 * 2);
    assert!(sorted_bytes(2) > 2 * 288 * 2);
}

#[test]
fn c4_shape_does_not_broaden_c3_or_other_models() {
    let config = glm_config();
    assert!(c4_grouped_shape(&config, 4, 4));
    assert!(!c4_grouped_shape(&config, 4, 3));
    assert!(!c3_grouped_shape(&config, 4, 4));
    for rows in [1, 2, 3, 5] {
        assert!(!c4_grouped_shape(&config, rows, 4));
    }
    for mutate in [
        (|c: &mut ModelConfig| c.hidden_size = 2048) as fn(&mut ModelConfig),
        |c| c.model_type = "deepseek_v4".into(),
        |c| c.tp_world_size = 1,
        |c| c.ep_world_size = 1,
        |c| c.scoring_func = "softmax".into(),
    ] {
        let mut config = config.clone();
        mutate(&mut config);
        assert!(!c4_grouped_shape(&config, 4, 4));
    }
}

#[test]
fn k128w_schedule_is_persistent_only_when_enabled_for_serving_chunks() {
    let counter = DevicePtr(0x4000);
    // Off (no persistent kernels): the row-tile grid bound at every size.
    assert_eq!(
        k128w_schedule(0, 65536, 288, counter),
        K128wSchedule::Grid { bound: 1024 + 288 }
    );
    // On: 8K-token chunks (65536 rows) and up take the persistent CTAs.
    for rows in [65536, 131072] {
        assert_eq!(
            k128w_schedule(96, rows, 288, counter),
            K128wSchedule::Persistent {
                ctas: 96,
                next_work: counter
            }
        );
    }
    // On, but a 4K-token chunk stays on the grid, where it measured faster.
    assert_eq!(
        k128w_schedule(96, 32768, 288, counter),
        K128wSchedule::Grid { bound: 512 + 288 }
    );
}
