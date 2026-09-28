// SPDX-License-Identifier: AGPL-3.0-only

//! Explicit graph opt-in; the eager semantic-index path remains the oracle.

use anyhow::{Result, ensure};

fn graph_policy(model_type: &str, sparse: bool, graphs: bool) -> Result<bool> {
    if model_type != "glm5_next" {
        return Ok(false);
    }
    ensure!(
        !graphs || sparse,
        "ATLAS_GLM_MULTI_SEQ_SPARSE_GRAPHS=1 requires ATLAS_GLM_MULTI_SEQ_SPARSE=1"
    );
    Ok(graphs)
}

pub fn glm_multi_seq_sparse_graphs_enabled(model_type: &str) -> Result<bool> {
    graph_policy(
        model_type,
        super::glm_multi_seq_sparse_enabled(model_type),
        std::env::var("ATLAS_GLM_MULTI_SEQ_SPARSE_GRAPHS").as_deref() == Ok("1"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_opt_in_requires_eager_feature_and_exact_model() {
        assert!(!graph_policy("glm5_next", false, false).unwrap());
        assert!(!graph_policy("glm5_next", true, false).unwrap());
        assert!(graph_policy("glm5_next", true, true).unwrap());
        assert!(graph_policy("glm5_next", false, true).is_err());
        assert!(!graph_policy("qwen3_next", true, true).unwrap());
        assert!(!graph_policy("deepseek_v4", false, true).unwrap());
    }
}
