#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only tests for Q4 task-quality checkers.

These tests verify the checker logic with synthetic responses, without
requiring a live model endpoint. Each checker is tested against positive
and negative cases.
"""
import json
import os
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from checkers import CHECKERS, check_code  # noqa: E402


class TestArithmeticChecker(unittest.TestCase):
    """Tests for check_arithmetic."""

    def test_exact_match(self):
        response = {"content": "The calculation gives 42.\nAnswer: 42", "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["arithmetic"](response, {"expected": "42"})
        self.assertTrue(result["passed"], result["reason"])

    def test_with_thousands_separator(self):
        response = {"content": "Answer: 1,234", "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["arithmetic"](response, {"expected": "1234"})
        self.assertTrue(result["passed"], result["reason"])

    def test_with_dollar_sign(self):
        response = {"content": "The total is $1,000.00\nAnswer: $1,000.00", "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["arithmetic"](response, {"expected": "1000.00"})
        self.assertTrue(result["passed"], result["reason"])

    def test_wrong_answer(self):
        response = {"content": "Answer: 43", "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["arithmetic"](response, {"expected": "42"})
        self.assertFalse(result["passed"])

    def test_no_answer_line(self):
        response = {"content": "The answer is forty two.", "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["arithmetic"](response, {"expected": "42"})
        self.assertFalse(result["passed"])

    def test_last_answer_line_used(self):
        response = {"content": "First try: Answer: 40\nCorrection: Answer: 42", "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["arithmetic"](response, {"expected": "42"})
        self.assertTrue(result["passed"], result["reason"])

    def test_reasoning_stripped(self):
        response = {"content": "Let me think... workingAnswer: 42", "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["arithmetic"](response, {"expected": "42"})
        self.assertTrue(result["passed"], result["reason"])


class TestStructuredChecker(unittest.TestCase):
    """Tests for check_structured."""

    def test_valid_json_matches_expected(self):
        response = {"content": '{"name": "Alice", "age": 30}', "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["structured"](response, {
            "schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]},
            "expected": {"name": "Alice", "age": 30}
        })
        self.assertTrue(result["passed"], result["reason"])

    def test_valid_json_schema_only(self):
        response = {"content": '{"name": "Bob", "age": 25}', "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["structured"](response, {
            "schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}
        })
        self.assertTrue(result["passed"], result["reason"])

    def test_invalid_json(self):
        response = {"content": "{name: Alice}", "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["structured"](response, {
            "schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        })
        self.assertFalse(result["passed"])

    def test_schema_violation(self):
        response = {"content": '{"name": "Alice", "age": "thirty"}', "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["structured"](response, {
            "schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}
        })
        self.assertFalse(result["passed"])

    def test_mismatched_expected(self):
        response = {"content": '{"name": "Bob", "age": 25}', "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["structured"](response, {
            "schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]},
            "expected": {"name": "Alice", "age": 30}
        })
        self.assertFalse(result["passed"])

    def test_with_json_fence(self):
        response = {"content": "```json\n{\"name\": \"Alice\", \"age\": 30}\n```", "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["structured"](response, {
            "schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]},
            "expected": {"name": "Alice", "age": 30}
        })
        self.assertTrue(result["passed"], result["reason"])


class TestToolArgsChecker(unittest.TestCase):
    """Tests for check_tool_args."""

    def test_exact_match(self):
        response = {"content": "", "tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city": "Paris", "units": "celsius"}'}}], "finish_reason": "stop"}
        result = CHECKERS["tool_args"](response, {
            "name": "get_weather",
            "expected_args": {"city": "Paris", "units": "celsius"},
            "schema": {"type": "object", "properties": {"city": {"type": "string"}, "units": {"type": "string"}}, "required": ["city", "units"]}
        })
        self.assertTrue(result["passed"], result["reason"])

    def test_wrong_function(self):
        response = {"content": "", "tool_calls": [{"function": {"name": "get_time", "arguments": '{}'}}], "finish_reason": "stop"}
        result = CHECKERS["tool_args"](response, {
            "name": "get_weather",
            "expected_args": {},
            "schema": {"type": "object", "properties": {}}
        })
        self.assertFalse(result["passed"])

    def test_multiple_calls(self):
        response = {"content": "", "tool_calls": [
            {"function": {"name": "get_weather", "arguments": '{}'}},
            {"function": {"name": "get_time", "arguments": '{}'}}
        ], "finish_reason": "stop"}
        result = CHECKERS["tool_args"](response, {
            "name": "get_weather",
            "expected_args": {},
            "schema": {"type": "object", "properties": {}}
        })
        self.assertFalse(result["passed"])

    def test_schema_violation(self):
        response = {"content": "", "tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city": 123}'}}], "finish_reason": "stop"}
        result = CHECKERS["tool_args"](response, {
            "name": "get_weather",
            "expected_args": {"city": "Paris"},
            "schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        })
        self.assertFalse(result["passed"])

    def test_argument_mismatch(self):
        response = {"content": "", "tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city": "London"}'}}], "finish_reason": "stop"}
        result = CHECKERS["tool_args"](response, {
            "name": "get_weather",
            "expected_args": {"city": "Paris"},
            "schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        })
        self.assertFalse(result["passed"])


class TestMultilingualChecker(unittest.TestCase):
    """Tests for check_multilingual."""

    def test_french(self):
        response = {"content": "Le chat est sur la table.", "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["multilingual"](response, {
            "language": "fr",
            "all_of": [["chat", "chien"], ["table", "chaise"]]
        })
        self.assertTrue(result["passed"], result["reason"])

    def test_german(self):
        response = {"content": "Der Hund läuft im Park.", "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["multilingual"](response, {
            "language": "de",
            "all_of": [["Hund", "Katze"], ["Park", "Garten"]]
        })
        self.assertTrue(result["passed"], result["reason"])

    def test_spanish(self):
        response = {"content": "El gato está en la casa.", "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["multilingual"](response, {
            "language": "es",
            "all_of": [["gato", "perro"], ["casa", "coche"]]
        })
        self.assertTrue(result["passed"], result["reason"])

    def test_chinese(self):
        response = {"content": "猫在桌子上。", "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["multilingual"](response, {
            "language": "zh",
            "all_of": [["猫", "狗"], ["桌子", "椅子"]]
        })
        self.assertTrue(result["passed"], result["reason"])

    def test_wrong_language(self):
        response = {"content": "The cat is on the table.", "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["multilingual"](response, {
            "language": "fr",
            "all_of": [["chat"], ["table"]]
        })
        self.assertFalse(result["passed"])

    def test_missing_fact(self):
        response = {"content": "Le chien est dans le jardin.", "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["multilingual"](response, {
            "language": "fr",
            "all_of": [["chat", "chien"], ["table", "chaise"]]
        })
        self.assertFalse(result["passed"])


class TestLongContextChecker(unittest.TestCase):
    """Tests for check_long_context."""

    def test_needle_found(self):
        response = {"content": "The secret code is ABC123 and it was hidden in the text.", "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["long_context"](response, {
            "needle": "ABC123",
            "forbidden": ["XYZ789"]
        })
        self.assertTrue(result["passed"], result["reason"])

    def test_needle_not_found(self):
        response = {"content": "I couldn't find the code.", "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["long_context"](response, {
            "needle": "ABC123",
            "forbidden": []
        })
        self.assertFalse(result["passed"])

    def test_forbidden_found(self):
        response = {"content": "The code is ABC123 but also XYZ789 appears.", "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["long_context"](response, {
            "needle": "ABC123",
            "forbidden": ["XYZ789"]
        })
        self.assertFalse(result["passed"])


class TestSustainedChecker(unittest.TestCase):
    """Tests for check_sustained."""

    def test_prose_mode(self):
        response = {"content": "This is a long piece of text that should have enough words to pass the minimum word count requirement without any repetition loops.", "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["sustained"](response, {"mode": "prose", "min_words": 10})
        self.assertTrue(result["passed"], result["reason"])

    def test_prose_too_short(self):
        response = {"content": "Too short.", "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["sustained"](response, {"mode": "prose", "min_words": 10})
        self.assertFalse(result["passed"])

    def test_prose_repetition_loop(self):
        response = {"content": "hello hello hello hello", "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["sustained"](response, {"mode": "prose", "min_words": 5, "min_repeats": 4})
        self.assertFalse(result["passed"])

    def test_ints_mode(self):
        response = {"content": "2\n3\n5\n7\n11\n", "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["sustained"](response, {"mode": "ints", "expected": [2, 3, 5, 7, 11]})
        self.assertTrue(result["passed"], result["reason"])

    def test_ints_mismatch(self):
        response = {"content": "2\n3\n5\n7\n13\n", "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["sustained"](response, {"mode": "ints", "expected": [2, 3, 5, 7, 11]})
        self.assertFalse(result["passed"])

    def test_lines_mode(self):
        response = {"content": "line1\nline2\nline3\n", "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["sustained"](response, {"mode": "lines", "expected": ["line1", "line2", "line3"]})
        self.assertTrue(result["passed"], result["reason"])

    def test_json_mode(self):
        response = {"content": '[{"id": 1}, {"id": 2}]', "tool_calls": None, "finish_reason": "stop"}
        result = CHECKERS["sustained"](response, {"mode": "json", "expected": [{"id": 1}, {"id": 2}]})
        self.assertTrue(result["passed"], result["reason"])

    def test_wrong_finish_reason(self):
        response = {"content": "Truncated", "tool_calls": None, "finish_reason": "length"}
        result = CHECKERS["sustained"](response, {"mode": "prose", "min_words": 1})
        self.assertFalse(result["passed"])


class TestCodeChecker(unittest.TestCase):
    """Tests for check_code (requires TMPDIR)."""

    def setUp(self):
        self.tmp_root = os.environ.get("TMPDIR")
        if not self.tmp_root or not os.path.isdir(self.tmp_root):
            self.skipTest("TMPDIR not set or not a directory; code checker tests require it")

    def test_code_passes(self):
        response = {"content": "```python\ndef solve():\n    return [1, 2, 3]\n```", "tool_calls": None, "finish_reason": "stop"}
        result = check_code(response, {
            "entry_point": "solve",
            "hidden_tests": "assert mod.solve() == [1, 2, 3]"
        }, tmp_root=self.tmp_root)
        self.assertTrue(result["passed"], result["reason"])

    def test_code_fails(self):
        response = {"content": "```python\ndef solve():\n    return [1, 2, 4]\n```", "tool_calls": None, "finish_reason": "stop"}
        result = check_code(response, {
            "entry_point": "solve",
            "hidden_tests": "assert mod.solve() == [1, 2, 3]"
        }, tmp_root=self.tmp_root)
        self.assertFalse(result["passed"])

    def test_no_code_block(self):
        response = {"content": "def solve():\n    return [1, 2, 3]", "tool_calls": None, "finish_reason": "stop"}
        result = check_code(response, {
            "entry_point": "solve",
            "hidden_tests": "assert mod.solve() == [1, 2, 3]"
        }, tmp_root=self.tmp_root)
        self.assertFalse(result["passed"])

    def test_syntax_error(self):
        response = {"content": "```python\ndef solve()\n    return 1\n```", "tool_calls": None, "finish_reason": "stop"}
        result = check_code(response, {
            "entry_point": "solve",
            "hidden_tests": "assert mod.solve() == 1"
        }, tmp_root=self.tmp_root)
        self.assertFalse(result["passed"])

    def test_import_error(self):
        response = {"content": "```python\nimport nonexistent_module\ndef solve():\n    return 1\n```", "tool_calls": None, "finish_reason": "stop"}
        result = check_code(response, {
            "entry_point": "solve",
            "hidden_tests": "assert mod.solve() == 1"
        }, tmp_root=self.tmp_root)
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()