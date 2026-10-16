# prefill-idx-sort capture-verify (2026-10-16, branch followup/prefill-moe-group)
Edit: crates/spark-model/src/layers/ops/glm_indexer_dynamic.rs (marker + sort insertion point, capture-evidence cited).
Evidence: exl3-indexer-localization-result-20261015.md (216 token rows, 38 order-different, 0 set-different; top_k_per_row_prefill order variance).
Protocol (continue.md §0 / measurement-discipline):
- Label runs prefills-idx-sort-<n>
- Lock: flock /tmp/atlas-bench.lock <cmd>
- Compare: index buffer (attn-indexer_indexer_op) byte-identical across 3 pairs x 2 ranks, row 2,052+ tokens
- Acceptance: 0 differing index buffers; 0 differing o_proj after reduce
- If pass: implement sort (device or thrust) at marker; if fail: revisit attribution (fused MoE / indexer)
- No speed claim without ≥3 repeats, same binary, same session.
Branch: followup/prefill-moe-group (worktree ../atlas-prefill-moe), from perf/decode-vllm-parity-20261015 (794e9400). Not committed to perf/* or shared branch.
