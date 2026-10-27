# SPDX-License-Identifier: AGPL-3.0-only
"""Exercise the actual checker before inference, including cached environments."""
from score import _score_ast

DESCRIPTION = [{
    "name": "add",
    "description": "Add two integers.",
    "parameters": {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    },
}]
TRUTH = [{"add": {"a": [1], "b": [2]}}]
correct = _score_ast(
    [{"name": "add", "arguments": {"a": 1, "b": 2}}],
    TRUTH, DESCRIPTION, "simple_python",
)
wrong = _score_ast(
    [{"name": "add", "arguments": {"a": 1, "b": 99}}],
    TRUTH, DESCRIPTION, "simple_python",
)
if (correct, wrong) != (1.0, 0.0):
    raise RuntimeError(f"BFCL scorer preflight failed: correct={correct}, wrong={wrong}")
print("BFCL scorer accepted correct arguments and rejected incorrect arguments")
