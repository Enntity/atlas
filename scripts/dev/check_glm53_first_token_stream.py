#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Validate a saved GLM first-</think> SSE regression; never execute generated code."""

import argparse
import ast
import json
from pathlib import Path
import re
import unittest


def validate(events: list[dict], end_think_id: int) -> dict:
    choices = [c for event in events for c in event.get("choices", [])]
    ids = [t for choice in choices for t in choice.get("token_ids", [])]
    content = "".join(c.get("delta", {}).get("content", "") for c in choices)
    reasoning = "".join(c.get("delta", {}).get(name, "")
                        for c in choices for name in ("reasoning", "reasoning_content"))
    usage = next((e["usage"] for e in reversed(events) if e.get("usage")), {})
    finishes = [c["finish_reason"] for c in choices if c.get("finish_reason")]
    fenced = re.fullmatch(r"\s*```(?:python)?\s*\n(.*?)\n```\s*", content, re.DOTALL)
    syntax_ok = False
    if fenced:
        try:
            tree = ast.parse(fenced.group(1))
            syntax_ok = len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
        except SyntaxError:
            pass
    checks = {
        "first_sample_closes_thinking": bool(ids) and ids[0] == end_think_id,
        "one_closing_marker": ids.count(end_think_id) == 1,
        "no_reasoning_text": not reasoning,
        "no_falsely_counted_reasoning": usage.get("completion_tokens_details", {}).get(
            "reasoning_tokens") == 0,
        "single_syntactically_valid_function": syntax_ok,
        "normal_stop": finishes == ["stop"],
        "bounded_positive_output": type(usage.get("completion_tokens")) is int
        and 0 < usage["completion_tokens"] <= 256,
    }
    return {"passed": all(checks.values()), "checks": checks, "usage": usage,
            "returned_token_ids": len(ids), "content": content}


class ValidatorTests(unittest.TestCase):
    def good(self):
        return [{"choices": [{"token_ids": [9, 1, 2], "delta": {
            "content": "```python\ndef valid_order(nodes, edges, order):\n    return True\n```"}}]},
            {"usage": {"completion_tokens": 4,
                       "completion_tokens_details": {"reasoning_tokens": 0}}},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]}]

    def test_accepts_structural_gate_without_executing_or_grading_function(self):
        self.assertTrue(validate(self.good(), 9)["passed"])

    def test_rejects_lifecycle_symptoms_independently(self):
        for mutation in ("reasoning_count", "second_marker", "length", "duplicated_code"):
            events = self.good()
            if mutation == "reasoning_count":
                events[1]["usage"]["completion_tokens_details"]["reasoning_tokens"] = 62
            elif mutation == "second_marker":
                events[0]["choices"][0]["token_ids"].append(9)
            elif mutation == "length":
                events[-1]["choices"][0]["finish_reason"] = "length"
            else:
                events[0]["choices"][0]["delta"]["content"] *= 2
            self.assertFalse(validate(events, 9)["passed"], mutation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--sse", type=Path)
    parser.add_argument("--think-end-id", type=int)
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__])
    if args.sse is None or args.think_end_id is None:
        parser.error("require --sse and --think-end-id, or --self-test")
    lines = args.sse.read_text().splitlines()
    if "data: [DONE]" not in lines:
        raise SystemExit("Incomplete SSE: no DONE marker")
    events = [json.loads(line[6:]) for line in lines
              if line.startswith("data: ") and line[6:] != "[DONE]"]
    if any("error" in event for event in events):
        raise SystemExit("SSE contains a server error")
    result = validate(events, args.think_end_id)
    print(json.dumps(result))
    raise SystemExit(0 if result["passed"] else 1)
