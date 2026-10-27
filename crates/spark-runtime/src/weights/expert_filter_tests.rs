// SPDX-License-Identifier: AGPL-3.0-only
use super::SafetensorsLoader;

#[test]
fn replicated_prefix_overrides_ep_expert_filter() {
    let mut loader = SafetensorsLoader::with_ep(1, 2, 288);
    let appended = "model.language_model.layers.45.mlp.experts.1.gate_proj.weight";
    let target = "model.language_model.layers.44.mlp.experts.1.gate_proj.weight";
    assert!(loader.should_skip_tensor(appended));
    loader.replicated_expert_prefix = Some(".layers.45.".into());
    assert!(!loader.should_skip_tensor(appended));
    assert!(loader.should_skip_tensor(target));
}

#[test]
fn skip_vision_drops_only_the_vision_tower() {
    let mut loader = SafetensorsLoader::with_ep(1, 2, 288);
    let tower = [
        "model.visual.blocks.0.attn.qkv.weight",
        "model.language_model.visual.merger.proj.weight",
    ];
    let kept = "model.language_model.layers.3.self_attn.q_proj.weight";
    assert!(
        tower.iter().all(|t| !loader.should_skip_tensor(t)),
        "off by default"
    );
    loader.skip_vision = true;
    assert!(tower.iter().all(|t| loader.should_skip_tensor(t)));
    assert!(!loader.should_skip_tensor(kept));
}
