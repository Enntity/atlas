// SPDX-License-Identifier: AGPL-3.0-only

use super::configured_eos_tokens;
use atlas_core::config::ModelConfig;

#[test]
fn checkpoint_multi_eos_is_preserved_without_generation_config() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.eos_token_id = 154_820;
    config.eos_token_ids = vec![154_820, 154_827, 154_829];
    assert_eq!(configured_eos_tokens(&config), config.eos_token_ids);
}

#[test]
fn legacy_single_eos_remains_unchanged() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    assert_eq!(configured_eos_tokens(&config), vec![config.eos_token_id]);
}
