# Logical positions for committed speculative emission

2026-09-09. Code `96f83f7f8548bb69e47483845e425991d6aadda6`.
Exact post-commit CPU suites passed: model **1100**, zero failures (85.48s,
four test threads); server **2369**, zero failures, 12 ignored (24.61s,
one test thread). These are correctness tests, not inference measurements.

The existing emitter now delegates to one shared body with an explicit logical
position. Only the two sequence-ceiling comparisons use that position; ordinary
callers pass the canonical sequence length and retain existing behavior. The
future paired driver can commit its complete accepted prefix before emitting
individual rows without treating the first row as the last row. Canonical state
is never rewound. No driver, admission, factory or transport caller is activated.

New-entry TDD first used a transparent wrapper that ignored the logical
position: one control passed and two tests failed because earlier rows stopped
at the final canonical ceiling. This is evidence for the new commit-before-emit
ordering, not a claimed regression in existing ordinary callers. The final
emitter family passed all 15 tests. Three new tests cover successive positions
with fixed canonical state, suppressed EOS with thinking budget, and ordinary
wrapper parity. Suppressed thinking EOS still decrements its existing budget;
the one-token-budget control correctly finishes and was retained unchanged.

The non-test server check passed. Workspace formatting, whitespace, SPDX and
the three frozen hashes passed. The new test file is 82 lines; emit_step.rs is
924 lines under its existing size-cap exception. Server clippy remains blocked
by nine inherited model lints across eight unchanged files; no suppression or
CI-green claim. Independent review approved the final three-path manifest,
SHA-256 `2cb595bc24905f7d9b02f68989f7bdb3a59b6e4f0cb0543c85ae138bef5fa522`.

Raw commands/results are retained in campaign `20260908/glm-c2-emission-position`.
Archive `glm-c2-emission-position-96f83f7f-receipts.tar` has a detached checksum
and contains source, documentation and CPU receipts only. No native build, GPU
execution, concurrent-serving qualification or new throughput result follows.
