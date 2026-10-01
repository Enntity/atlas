# Opt-in FlashKDA prefill experiment

This adapter targets the pinned FlashKDA native bridge from the earlier Atlas
research archive: upstream MoonshotAI/FlashKDA commit
`1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b`, CUTLASS
`5c149f52a436782210263fb2f19b354443a61c6a`, with the retained SM121 state-slot patch.
The exact library SHA256 is
`7fe3fa22f7fcf2159848bf47189f7caa1adb7286d99e4a334376a410047dc092`.
FlashKDA is MIT licensed; the Atlas bridge and these adapter files are AGPL-3.0.
Preserve upstream notices when distributing the separately built library.

Enable explicitly with `ATLAS_KDA_FLASH_PREFILL=1` and an absolute
`ATLAS_KDA_FLASH_LIBRARY` path to `libatlas_mango_flash.so`. The original
`libatlas_glm53_flash_kda.so` must be alongside it; link the shim with `$ORIGIN`
rpath. Disabled by default. Missing library/symbols or wrong ABI fail startup.

Only ordinary H32/D128 prefill with 2048 through4100 rows and lower-bound−5 is
eligible. Decode, graph capture, MTP intermediate snapshots and short tails retain
the existing recurrence. FlashKDA already normalizes Q/K; the adapter copies raw
BF16 operands. Canonical FP32 key-major state is transposed in and out on the
existing stream. No new persistent GPU allocation: use dead post-convolution
ssm_qkvz and idle expert buffers with explicit capacity and overlap checks.
A launch/validation error aborts the request; never retry a mutated chunk in place.

The standalone gate passed independent short FP64 recurrence comparisons,
zero/tiny Q/K, saturated gate/beta, asymmetric nonzero state, exact packing,
split/resume and continued native recurrence. Preparation-inclusive speedups were
3.38–3.62x at2048/4096/4100 tokens on GB10. These are candidate operator results,
not full-model quality or C4 throughput claims. The combined shim matched the
qualified probe bitwise and rejected insufficient scratch before modifying state.
It passed a native smoke test in the frozen serving image. Focused Rust tests
cover strict flags, prefill-only eligibility, overlap and address overflow.

Development receipts and complete standalone harness remain in the private
`experiments/atlas-nvidia-20260910/parity-20260911/flash-kda` directory.
The initial adapter's double normalization and DeepSeek draft defects were caught
and corrected before integration; original failure receipts remain available.
Model-level qualification is pending for this commit.
