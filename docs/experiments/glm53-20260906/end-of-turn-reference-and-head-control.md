# End-of-turn reference and LM-head control

Status: 2026-09-10. This is a bounded diagnosis, not a model-quality or
maximum-context qualification. See [release progress](release-qualification-progress.md).

## Official parser and checkpoint contract

The reviewed vLLM revision is `285cbce6bd0982f5936d729d3fa2ac310dfeca59`.
Its `glm45` reasoning-parser alias selects the GLM47 adapter. The
[GLM parser](https://github.com/vllm-project/vllm/blob/285cbce6bd0982f5936d729d3fa2ac310dfeca59/vllm/parser/glm47_moe.py#L97)
starts in reasoning; `</think>` or native tool-start transitions out.
[Generation stop detection](https://github.com/vllm-project/vllm/blob/285cbce6bd0982f5936d729d3fa2ac310dfeca59/vllm/v1/core/sched/utils.py#L87)
honors sampled EOS/stop IDs without requiring a reasoning-end marker.

At stream completion, the
[parser finish operation](https://github.com/vllm-project/vllm/blob/285cbce6bd0982f5936d729d3fa2ac310dfeca59/vllm/parser/engine/streaming_parser_engine.py#L287)
flushes pending text and emits an internal reasoning-end event. It does not
generate a closing token or reclassify preceding reasoning as answer content.
[Extraction](https://github.com/vllm-project/vllm/blob/285cbce6bd0982f5936d729d3fa2ac310dfeca59/vllm/parser/engine/parser_engine.py#L509)
therefore permits reasoning with no content; its streaming fallback also returns
no content. This explains the parser result for an unterminated block, not why
the model generated that block. It does not justify hoisting reasoning.

Official Flash checkpoint revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`
specifies EOS IDs `[154820,154827,154829]`, temperature `1.0` and top-p `0.95` in
[generation_config.json](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/generation_config.json).
Its [template](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/chat_template.jinja)
opens `<think>` unconditionally, resolves effort to Low/High/Max, and defaults
to Max. `clear_thinking` controls historical assistant reasoning, not the fresh
generation suffix. The official [serving recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash)
uses reasoning parsing; our greedy, explicitly budgeted requests are not an
identical reference-inference configuration.

## Atlas stop correction

Native tracing on `14b4e485` showed the first answer token `153`, then sampled
EOS `154827` (`<|user|>`), suppressed solely because thinking remained open.
The continued generation invented a new prime-factor question and answered it.
Commit `da4ec65d316bf3c57796e975633a6e8e78bfc099` honors native GLM EOS in that
situation while retaining other stop guards. It neither fabricates `</think>`
nor moves reasoning into visible content. Actual ordinary/MTP and selected
transaction regressions passed. Removing the invented continuation is a valid
stop correction; reasoning-only `153` with empty content still fails the exact
visible-answer gate.

## BF16-head comparison: negative for the observed failure

LibertAIDAI excludes `lm_head` from quantization in the
[historical checkpoint config](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4/blob/11d73216cd636238e82e1d77fe1042ffab36e7fa/config.json).
A bounded header read at publisher revision
`caca4e6a4ebbd66f159d3d2fc256683fd6e27177` confirmed `lm_head.weight` is BF16,
shape `[154880,4096]`. Atlas's explicit `--lm-head-dtype=nvfp4` instead invokes
[runtime head quantization](../../../crates/spark-model/src/factory/lm_head_setup.rs).
The original BF16 tensor remains resident; this is an additive quantized copy,
not the checkpoint's original precision. That difference was a concrete
diagnostic candidate, not evidence of causation.

The ordinary native control finished at **02:00:29 UTC**, using the same
`da4ec65d` engine and effort driver. Both ranks logged BF16-head selection,
exited zero, had `OOMKilled=false`, and recorded zero swap. The only profile
change was target-head dtype; the external profile SHA256 is
`ee0958bdb847837a4841d874263fa5878702b4510d1db00faae6faa3fe98982d`.

Root compared all seven exact request objects, response `choices`, and
prompt/completion counts against the prior NVFP4 run: **all matched**.

| Request group | BF16-head result | Completion tokens |
|---|---|---:|
| Default effort, five probes including fresh/repeat/post-cancel and requested-false controls | **FAIL**: reasoning `153`, empty content | 2 each |
| Explicit Low, budget16 | Exact visible-answer **PASS** | 5 |
| Explicit High, budget16 | Exact visible-answer **PASS** | 4 |

Requested-false is not a proven no-thinking control: the GLM no-tools template
still opens reasoning. The first failure preceded cancellation. These two
effort successes are individual probes, not general Low/High qualification.

Retained campaign artifacts are named
`reuse-da4ec65d-16384-bf16head-effort-first-summary.json`,
`reuse-da4ec65d-16384-bf16head-effort-first-receipts/summary.json`, and matching
`-run/00170-rank0-collect.json` / `00171-rank1-collect.json`.
The overall workload result is failure despite clean operational shutdown;
this ordinary watchdog run does not establish paired quiescent release.

Thus the head-precision mismatch remains real in general, but this matched
native control **did not fix or change the observed failure**. No additional
MTP BF16-head test was executed for this finding.

## Post-release investigation

1. Compare exact rendered prompt bytes and token IDs with the pinned stock
   template; replay retained output IDs through the pinned official parser.
2. Compare matched reference inference at the first EOS/`</think>` decision,
   retaining raw logits and identical prefixes rather than comparing divergent
   continuations.
3. Evaluate official sampling settings separately from the greedy control.

Do not substitute more unmeasured flags, invented closing tokens, or reasoning
hoisting for a demonstrated explanation and a visible-answer quality gate.
