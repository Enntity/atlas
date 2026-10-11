#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic builder for the frozen Q4 held-out corpus and tuning set.

Every expected answer is computed here in Python (arithmetic via exact
Fractions, sequences via generators) rather than typed in, and every code
case carries a reference solution in REFERENCE_SOLUTIONS that test_q4.py runs
through the real code checker. test_q4.py also rebuilds both files and
requires byte equality with the committed corpus.jsonl / tuning.jsonl, so
the frozen files cannot drift from this source silently.

  python3 tests/exl3_validation/q4/build_corpus.py            # rewrite both files
  python3 tests/exl3_validation/q4/build_corpus.py --check    # verify only

The corpus is frozen: after the first measured run, change cases only by
adding new ids, never by editing an existing case.
"""
import copy
import json
import os
import re
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_long_context import generate  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261011
SAMPLING = {"temperature": 0.0, "seed": SEED}
# Budgets are generous on purpose: with thinking off and no tools the GLM-5.3
# template still opens <think> for brief native reasoning (low_think), and
# those tokens count against max_tokens. A truncated answer fails its case.
ANSWER_SUFFIX = " Show brief working if you like, but end with a final line of the form 'Answer: <number>'."


def decimal_text(value):
    """Exact decimal text of a Fraction with a terminating expansion."""
    value = Fraction(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    scale = 0
    while (value * 10 ** scale).denominator != 1:
        scale += 1
        if scale > 12:
            raise ValueError("non-terminating expected answer: " + str(value))
    digits = str((value * 10 ** scale).numerator).rjust(scale + 1, "0")
    text = digits if scale == 0 else digits[:-scale] + "." + digits[-scale:]
    return sign + text


def case(cid, category, prompt, checker, params, max_tokens, split="heldout", **extra):
    row = {"id": cid, "category": category, "split": split,
           "messages": [{"role": "user", "content": prompt}],
           "sampling": dict(SAMPLING), "max_tokens": max_tokens, "thinking": False,
           "checker": checker, "checker_params": params}
    row.update(extra)
    return row


# ---------------------------------------------------------------- arithmetic
F = Fraction
ARITHMETIC = [
    ("ar01", "A warehouse holds 1248 boxes. Three trucks each take away 175 boxes, then a delivery "
             "adds 412 boxes. How many boxes are in the warehouse now?", F(1248) - 3 * 175 + 412),
    ("ar02", "A shirt costs $48.00. It is discounted by 25%, and then 8% sales tax is added to the "
             "discounted price. What is the final price in dollars?", F(48) * F(3, 4) * F(108, 100)),
    ("ar03", "A store sells notebooks at $2.50 each and pens at $1.20 each. Sam buys 3 notebooks and "
             "2 pens. How many dollars does Sam spend in total?", 3 * F(5, 2) + 2 * F(6, 5)),
    ("ar04", "A train travels 240 km in 3 hours, then 150 km in 2 hours. What is its average speed "
             "in km/h over the whole trip?", F(240 + 150, 3 + 2)),
    ("ar05", "You have 84 coins. You give one third of them to Ana, then give one quarter of the "
             "coins you still have to Ben. How many coins do you have left?",
     (F(84) - F(84, 3)) * F(3, 4)),
    ("ar06", "What is 2 to the power 10 minus 3 to the power 5?", F(2 ** 10 - 3 ** 5)),
    ("ar07", "What is the sum of all positive integers up to and including 250 that are divisible "
             "by 7?", F(sum(n for n in range(1, 251) if n % 7 == 0))),
    ("ar08", "A rectangular sheet measures 12.5 cm by 8 cm. A 3 cm by 3 cm square is cut out of it. "
             "What is the remaining area in square centimetres?", F(25, 2) * 8 - 9),
    ("ar09", "What is the mean of 14, 22, 30, 38 and 51?", F(14 + 22 + 30 + 38 + 51, 5)),
    ("ar10", "You deposit $2000 at 5% annual interest, compounded once per year. How many dollars "
             "are in the account after exactly 3 years?", F(2000) * F(105, 100) ** 3),
    ("ar11", "A recipe needs 2.5 cups of flour to make 12 cookies. How many cups of flour are "
             "needed for 30 cookies?", F(5, 2) * F(30, 12)),
    ("ar12", "Compute 17 * 23 - 156 / 12.", F(17 * 23) - F(156, 12)),
    ("ar13", "An empty 360-litre tank is filled at 12 litres per minute while a leak drains 4.5 "
             "litres per minute. How many minutes does it take to fill the tank?",
     F(360) / (12 - F(9, 2))),
    ("ar14", "What is the product of the first six prime numbers?", F(2 * 3 * 5 * 7 * 11 * 13)),
    ("ar15", "Compute -17 + 4 * (-3) - (-25).", F(-17 + 4 * (-3) - (-25))),
]
TUNING_ARITHMETIC = [
    ("tu01", "A cinema has 18 rows of 24 seats. 137 seats are taken. How many seats are free?",
     F(18 * 24 - 137)),
    ("tu02", "A bike costs $320 and is discounted by 15%. What is the sale price in dollars?",
     F(320) * F(85, 100)),
]

# ---------------------------------------------------------------------- code
CODE_SUFFIX = (" Reply with the complete function in a single ```python code block. Do not include "
               "tests, example usage or input() calls.")
# id -> (prompt, hidden tests, reference solution). Hidden tests run with the
# generated module bound to `mod`.
CODE = {
    "co01": ("Write a Python function fibonacci(n) that returns a list of the first n Fibonacci "
             "numbers, starting 0, 1. For n = 0 return an empty list.",
             "assert mod.fibonacci(0) == []\nassert mod.fibonacci(1) == [0]\n"
             "assert mod.fibonacci(2) == [0, 1]\n"
             "assert mod.fibonacci(10) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]\n",
             "def fibonacci(n):\n    out = []\n    a, b = 0, 1\n    for _ in range(n):\n"
             "        out.append(a)\n        a, b = b, a + b\n    return out\n"),
    "co02": ("Write a Python function is_prime(n) that returns True if the integer n is a prime "
             "number and False otherwise. n may be zero or negative.",
             "assert [n for n in range(-5, 60) if mod.is_prime(n)] == "
             "[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]\n"
             "assert mod.is_prime(7919) is True\nassert mod.is_prime(7917) is False\n"
             "assert all(mod.is_prime(n) is False for n in (-7, 0, 1, 4))\n",
             "def is_prime(n):\n    if n < 2:\n        return False\n    d = 2\n"
             "    while d * d <= n:\n        if n % d == 0:\n            return False\n"
             "        d += 1\n    return True\n"),
    "co03": ("Write a Python function gcd(a, b) that returns the greatest common divisor of two "
             "non-negative integers. gcd(0, 0) must return 0. Do not use the math module.",
             "assert mod.gcd(48, 18) == 6\nassert mod.gcd(0, 5) == 5\nassert mod.gcd(7, 0) == 7\n"
             "assert mod.gcd(0, 0) == 0\nassert mod.gcd(17, 5) == 1\nassert mod.gcd(270, 192) == 6\n",
             "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n"),
    "co04": ("Write a Python function median(nums) that returns the median of a non-empty list of "
             "numbers. For an even count return the mean of the two middle values. The function "
             "must not modify its input list.",
             "data = [4, 1, 3, 2]\nassert mod.median(data) == 2.5\nassert data == [4, 1, 3, 2]\n"
             "assert mod.median([3, 1, 2]) == 2\nassert mod.median([5]) == 5\n"
             "assert mod.median([-1, -9, 10, 7, 0]) == 0\n",
             "def median(nums):\n    s = sorted(nums)\n    m = len(s) // 2\n"
             "    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2\n"),
    "co05": ("Write a Python function is_leap_year(year) that returns True if year is a leap year "
             "in the Gregorian calendar and False otherwise.",
             "for y, want in [(2000, True), (1900, False), (2024, True), (2023, False), "
             "(2100, False), (2400, True)]:\n    assert mod.is_leap_year(y) is want, y\n",
             "def is_leap_year(year):\n    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)\n"),
    "co06": ("Write a Python function roman_to_int(s) that converts a valid uppercase Roman numeral "
             "string (1 to 3999) to an integer.",
             "for s, want in [('III', 3), ('LVIII', 58), ('XLIX', 49), ('MCMXCIV', 1994), "
             "('MMXXVI', 2026), ('MMMCMXCIX', 3999)]:\n    assert mod.roman_to_int(s) == want, s\n",
             "def roman_to_int(s):\n    v = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n"
             "    total = 0\n    for i, c in enumerate(s):\n"
             "        if i + 1 < len(s) and v[c] < v[s[i + 1]]:\n            total -= v[c]\n"
             "        else:\n            total += v[c]\n    return total\n"),
    "co07": ("Write a Python function run_length_encode(s) that encodes a string by replacing each "
             "maximal run of a repeated character with the character followed by the run length in "
             "decimal, for example 'aaabbc' -> 'a3b2c1'. The empty string encodes to ''.",
             "assert mod.run_length_encode('aaabbc') == 'a3b2c1'\nassert mod.run_length_encode('') == ''\n"
             "assert mod.run_length_encode('abc') == 'a1b1c1'\n"
             "assert mod.run_length_encode('z' * 12) == 'z12'\n"
             "assert mod.run_length_encode('aabaa') == 'a2b1a2'\n",
             "def run_length_encode(s):\n    out = []\n    i = 0\n    while i < len(s):\n"
             "        j = i\n        while j < len(s) and s[j] == s[i]:\n            j += 1\n"
             "        out.append(s[i] + str(j - i))\n        i = j\n    return ''.join(out)\n"),
    "co08": ("Write a Python function flatten(nested) that takes a list whose elements are integers "
             "or (arbitrarily deeply) nested lists of the same kind, and returns a flat list of all "
             "the integers in left-to-right order.",
             "assert mod.flatten([1, [2, [3, [4]], 5]]) == [1, 2, 3, 4, 5]\n"
             "assert mod.flatten([]) == []\nassert mod.flatten([[[]]]) == []\n"
             "assert mod.flatten([[1, 2], 3, [[4], [5, [6]]]]) == [1, 2, 3, 4, 5, 6]\n",
             "def flatten(nested):\n    out = []\n    for x in nested:\n"
             "        if isinstance(x, list):\n            out.extend(flatten(x))\n"
             "        else:\n            out.append(x)\n    return out\n"),
    "co09": ("Write a Python function word_frequencies(text) that returns a dict mapping each word to "
             "how many times it occurs. Lowercase the text first; a word is a maximal run of the "
             "ASCII letters a-z, and every other character is a separator.",
             "assert mod.word_frequencies('The cat and the hat.') == "
             "{'the': 2, 'cat': 1, 'and': 1, 'hat': 1}\nassert mod.word_frequencies('') == {}\n"
             "assert mod.word_frequencies(\"It's it, IT!\") == {'it': 3, 's': 1}\n"
             "assert mod.word_frequencies('a1b2a') == {'a': 2, 'b': 1}\n",
             "import re\n\ndef word_frequencies(text):\n    out = {}\n"
             "    for w in re.findall('[a-z]+', text.lower()):\n        out[w] = out.get(w, 0) + 1\n"
             "    return out\n"),
    "co10": ("Write a Python function binary_search(items, target) that returns the index of target "
             "in the ascending list items of distinct integers, or -1 if it is absent. It must run "
             "in O(log n) time.",
             "items = list(range(0, 2000, 3))\nfor i, x in enumerate(items):\n"
             "    assert mod.binary_search(items, x) == i\n"
             "for x in (-3, 1, 2, 1999, 5000):\n    assert mod.binary_search(items, x) == -1\n"
             "assert mod.binary_search([], 4) == -1\n",
             "def binary_search(items, target):\n    lo, hi = 0, len(items) - 1\n    while lo <= hi:\n"
             "        mid = (lo + hi) // 2\n        if items[mid] == target:\n            return mid\n"
             "        if items[mid] < target:\n            lo = mid + 1\n        else:\n"
             "            hi = mid - 1\n    return -1\n"),
    "co11": ("Write a Python function merge_intervals(intervals) that takes a list of [start, end] "
             "integer pairs in any order and returns the merged intervals as a list of [start, end] "
             "lists sorted by start. Intervals that overlap or touch (end == next start) are merged.",
             "norm = lambda r: [list(x) for x in r]\n"
             "assert norm(mod.merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]])) == "
             "[[1, 6], [8, 10], [15, 18]]\n"
             "assert norm(mod.merge_intervals([[1, 4], [4, 5]])) == [[1, 5]]\n"
             "assert norm(mod.merge_intervals([])) == []\n"
             "assert norm(mod.merge_intervals([[5, 7], [1, 2]])) == [[1, 2], [5, 7]]\n"
             "assert norm(mod.merge_intervals([[1, 10], [2, 3]])) == [[1, 10]]\n",
             "def merge_intervals(intervals):\n    out = []\n"
             "    for s, e in sorted(intervals):\n        if out and s <= out[-1][1]:\n"
             "            out[-1][1] = max(out[-1][1], e)\n        else:\n            out.append([s, e])\n"
             "    return out\n"),
    "co12": ("Write a Python function caesar_cipher(text, shift) that shifts every ASCII letter by "
             "shift positions in the alphabet, wrapping around and preserving case. Other characters "
             "are unchanged. shift may be negative or larger than 26.",
             "assert mod.caesar_cipher('Hello, World!', 3) == 'Khoor, Zruog!'\n"
             "assert mod.caesar_cipher('abc', -1) == 'zab'\nassert mod.caesar_cipher('xyz', 29) == 'abc'\n"
             "assert mod.caesar_cipher('Zz 9', 1) == 'Aa 9'\n",
             "def caesar_cipher(text, shift):\n    out = []\n    for c in text:\n"
             "        if 'a' <= c <= 'z':\n            out.append(chr((ord(c) - 97 + shift) % 26 + 97))\n"
             "        elif 'A' <= c <= 'Z':\n            out.append(chr((ord(c) - 65 + shift) % 26 + 65))\n"
             "        else:\n            out.append(c)\n    return ''.join(out)\n"),
}
TUNING_CODE = {
    "tu03": ("Write a Python function reverse_words(s) that returns the words of s (separated by "
             "single spaces) in reverse order, joined by single spaces.",
             "assert mod.reverse_words('hello world') == 'world hello'\n"
             "assert mod.reverse_words('a b c') == 'c b a'\nassert mod.reverse_words('one') == 'one'\n",
             "def reverse_words(s):\n    return ' '.join(reversed(s.split(' ')))\n"),
    "tu04": ("Write a Python function digit_sum(n) that returns the sum of the decimal digits of the "
             "non-negative integer n.",
             "assert mod.digit_sum(0) == 0\nassert mod.digit_sum(9875) == 29\n"
             "assert mod.digit_sum(10 ** 20) == 1\n",
             "def digit_sum(n):\n    return sum(int(c) for c in str(n))\n"),
}
REFERENCE_SOLUTIONS = {cid: ref for table in (CODE, TUNING_CODE) for cid, (_, _, ref) in table.items()}

# ----------------------------------------------------------------- tool_args
def _tool(name, description, properties, required=None):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "additionalProperties": False,
                       "required": list(properties) if required is None else required,
                       "properties": properties}}}


CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF"]
# Every case offers the whole toolbox: picking the right tool is part of the
# task. All parameters are required and closed (additionalProperties false)
# so the exact expected arguments are the only schema-valid correct call.
TOOLBOX = [
    _tool("get_weather", "Current weather for one city.",
          {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}),
    _tool("web_search", "Search the web.",
          {"query": {"type": "string"}, "num_results": {"type": "integer", "minimum": 1, "maximum": 20}}),
    _tool("convert_currency", "Convert an amount between currencies.",
          {"amount": {"type": "number"}, "from_currency": {"type": "string", "enum": CURRENCIES},
           "to_currency": {"type": "string", "enum": CURRENCIES}}),
    _tool("create_event", "Create a calendar event.",
          {"title": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"},
           "duration_minutes": {"type": "integer", "minimum": 1}}),
    _tool("set_thermostat", "Set the target temperature in degrees Celsius for one room.",
          {"room": {"type": "string", "enum": ["kitchen", "bedroom", "office", "living_room"]},
           "temperature": {"type": "number"}}),
    _tool("lookup_inventory", "Look up stock for a SKU in one warehouse.",
          {"sku": {"type": "string"}, "warehouse": {"type": "string", "enum": ["north", "south"]}}),
    _tool("book_flight", "Book a flight between two airports given as IATA codes.",
          {"origin": {"type": "string"}, "destination": {"type": "string"},
           "passengers": {"type": "integer", "minimum": 1}}),
    _tool("get_stock_price", "Latest price for a stock ticker symbol.",
          {"ticker": {"type": "string"}}),
]
TOOL_CASES = [
    ("ta01", 'What is the weather in "Paris" right now? I want it in celsius.',
     "get_weather", {"city": "Paris", "unit": "celsius"}),
    ("ta02", 'Search the web for "Atlas GB10 inference" and give me 5 results.',
     "web_search", {"query": "Atlas GB10 inference", "num_results": 5}),
    ("ta03", "How much is 250.75 US dollars in euros?",
     "convert_currency", {"amount": 250.75, "from_currency": "USD", "to_currency": "EUR"}),
    ("ta04", 'Put an event titled "Q4 review" in my calendar on 2026-11-03 lasting one and a half hours.',
     "create_event", {"title": "Q4 review", "date": "2026-11-03", "duration_minutes": 90}),
    ("ta05", "Set the bedroom thermostat to 19.5 degrees Celsius.",
     "set_thermostat", {"room": "bedroom", "temperature": 19.5}),
    ("ta06", 'Check stock of SKU "KX-200" in the south warehouse.',
     "lookup_inventory", {"sku": "KX-200", "warehouse": "south"}),
    ("ta07", "Book a flight from SFO to NRT for 3 passengers.",
     "book_flight", {"origin": "SFO", "destination": "NRT", "passengers": 3}),
    ("ta08", 'Get the current stock price for the ticker "NVDA".',
     "get_stock_price", {"ticker": "NVDA"}),
    ("ta09", 'What is the weather like in "Tokyo"? Use fahrenheit.',
     "get_weather", {"city": "Tokyo", "unit": "fahrenheit"}),
    ("ta10", "Convert 1200 Japanese yen to British pounds.",
     "convert_currency", {"amount": 1200, "from_currency": "JPY", "to_currency": "GBP"}),
]
TUNING_TOOL_CASES = [
    ("tu05", 'What is the weather in "Madrid"? Use celsius.',
     "get_weather", {"city": "Madrid", "unit": "celsius"}),
]

# ---------------------------------------------------------------- structured
def _obj(properties, required=None):
    return {"type": "object", "additionalProperties": False,
            "required": list(properties) if required is None else required, "properties": properties}


S, I, N, B = ({"type": "string"}, {"type": "integer"}, {"type": "number"}, {"type": "boolean"})
ONLY_JSON = " Reply with only the JSON, no prose."
# (id, prompt, schema, expected value, send response_format json_schema?)
STRUCTURED = [
    ("st01", "Extract the order from this text as a JSON object with keys order_id (integer), "
             "customer (string), sku (string), quantity (integer) and unit_price (number): "
             "'Order #4471 was placed by Maria Lopez for 3 units of SKU KX-200 at $19.99 each.'",
     _obj({"order_id": I, "customer": S, "sku": S, "quantity": I, "unit_price": N}),
     {"order_id": 4471, "customer": "Maria Lopez", "sku": "KX-200", "quantity": 3, "unit_price": 19.99},
     True),
    ("st02", "Return a JSON object with one key, planets, whose value is the list of the four "
             "terrestrial planets of the Solar System in order of distance from the Sun.",
     _obj({"planets": {"type": "array", "items": S}}),
     {"planets": ["Mercury", "Venus", "Earth", "Mars"]}, False),
    ("st03", "Parse 'March 7, 2025 at 14:30' into a JSON object with integer keys year, month, day, "
             "hour and minute (24-hour clock).",
     _obj({"year": I, "month": I, "day": I, "hour": I, "minute": I}),
     {"year": 2025, "month": 3, "day": 7, "hour": 14, "minute": 30}, True),
    ("st04", "Return a JSON object describing the chemical element iron with keys name (string, "
             "capitalised English name), symbol (string) and atomic_number (integer).",
     _obj({"name": S, "symbol": S, "atomic_number": I}),
     {"name": "Iron", "symbol": "Fe", "atomic_number": 26}, False),
    ("st05", "Return a JSON array of 5 objects, one for each integer n from 1 to 5 in order, each "
             "with integer keys n and square (n squared).",
     {"type": "array", "minItems": 5, "maxItems": 5, "items": _obj({"n": I, "square": I})},
     [{"n": n, "square": n * n} for n in range(1, 6)], False),
    ("st06", "Classify the sentiment of this review as positive, negative or neutral and return a "
             "JSON object with the single key sentiment: 'The battery died after two hours and "
             "support never replied to my emails.'",
     _obj({"sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]}}),
     {"sentiment": "negative"}, True),
    ("st07", "Convert this sentence into a JSON object with keys person (an object with name and "
             "age) and tags (a list of strings, in the order given): 'Bob Smith, aged 42, has the "
             "tags admin and ops.'",
     _obj({"person": _obj({"name": S, "age": I}), "tags": {"type": "array", "items": S}}),
     {"person": {"name": "Bob Smith", "age": 42}, "tags": ["admin", "ops"]}, False),
    ("st08", "Return a JSON object for a product named Widget that is still on sale and therefore "
             "has no replacement product. Keys: name (string), discontinued (boolean) and "
             "replacement (string, or null when there is none).",
     _obj({"name": S, "discontinued": B, "replacement": {"type": ["string", "null"]}}),
     {"name": "Widget", "discontinued": False, "replacement": None}, True),
    ("st09", "For the word 'banana', return a JSON object with keys word (string), length (integer, "
             "number of letters) and unique_letters (integer, number of distinct letters).",
     _obj({"word": S, "length": I, "unique_letters": I}),
     {"word": "banana", "length": len("banana"), "unique_letters": len(set("banana"))}, False),
    ("st10", "Express a temperature of 25 degrees Celsius as a JSON object with numeric keys "
             "celsius, fahrenheit and kelvin.",
     _obj({"celsius": N, "fahrenheit": N, "kelvin": N}),
     {"celsius": 25, "fahrenheit": float(F(25) * F(9, 5) + 32), "kelvin": float(F(25) + F(27315, 100))},
     True),
]
TUNING_STRUCTURED = [
    ("tu06", "Return a JSON object with keys city (string) and country (string) for the capital of "
             "Canada.",
     _obj({"city": S, "country": S}), {"city": "Ottawa", "country": "Canada"}, False),
]

# -------------------------------------------------------------- multilingual
# all_of: every group needs one variant present (after NFKC + casefold).
MULTILINGUAL = [
    ("ml01", "fr", "Quelle est la capitale de l'Australie ? Réponds en français par une phrase complète.",
     [["Canberra"]]),
    ("ml02", "de", "Wie viele Beine hat eine Spinne? Antworte auf Deutsch in einem ganzen Satz.",
     [["8", "acht"]]),
    ("ml03", "zh", "世界上面积最大的海洋是哪一个？请用中文回答。", [["太平洋"]]),
    ("ml04", "ja", "日本で一番高い山は何ですか？日本語で答えてください。", [["富士山"]]),
    ("ml05", "fr", "Combien de côtés a un hexagone ? Réponds en français par une phrase complète.",
     [["6", "six"]]),
    ("ml06", "de", "Welches chemische Element hat das Symbol O? Antworte auf Deutsch in einem ganzen Satz.",
     [["Sauerstoff"]]),
    ("ml07", "zh", "一个星期有几天？请用中文回答。", [["7", "七"]]),
    ("ml08", "ja", "一年は何か月ありますか？日本語で答えてください。", [["12", "十二"]]),
    ("ml09", "es", "¿Cuál es el planeta más grande del sistema solar? Responde en español con una frase completa.",
     [["Júpiter", "Jupiter"]]),
    ("ml10", "de", "Translate into German: 'The cat sleeps on the sofa.' Reply with only the German sentence.",
     [["Katze"], ["schläft"], ["Sofa", "Couch"]]),
    ("ml11", "fr", "Translate into French: 'I would like a glass of water, please.' Reply with only "
                   "the French sentence.", [["verre"], ["eau"]]),
    ("ml12", "zh", "Translate into Simplified Chinese: 'Today is Monday.' Reply with only the Chinese "
                   "sentence.", [["今天"], ["星期一", "周一", "礼拜一"]]),
]
TUNING_MULTILINGUAL = [
    ("tu07", "de", "Wie viele Stunden hat ein Tag? Antworte auf Deutsch in einem ganzen Satz.",
     [["24", "vierundzwanzig"]]),
    ("tu08", "fr", "De quelle couleur est le ciel par une journée claire ? Réponds en français.",
     [["bleu"]]),
]

# -------------------------------------------------------------- long_context
# (id, seed, filler chars, needle depth). 14200 chars ~ 3000 tokens and
# 18000 chars ~ 3700 tokens (near the 4096 serving limit) on the GLM-5.3
# tokenizer; MEASURED_TOKENS holds the exact content counts.
LONG = [
    ("lc01", 101, 14200, 0.05), ("lc02", 102, 14200, 0.25), ("lc03", 103, 14200, 0.5),
    ("lc04", 104, 14200, 0.75), ("lc05", 105, 14200, 0.95), ("lc06", 106, 14200, 0.4),
    ("lc07", 107, 14200, 0.6), ("lc08", 108, 14200, 0.15),
    ("lc09", 109, 18000, 0.1), ("lc10", 110, 18000, 0.5), ("lc11", 111, 18000, 0.9),
    ("lc12", 112, 18000, 0.3),
]
TUNING_LONG = [("tu09", 901, 6000, 0.5)]
# Content tokens (user message only, no chat template) measured with the
# checkpoint tokenizer.json, sha256 19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d.
# Hard-coded: re-measure them if the generator or any LONG row changes.
SERVING_LIMIT = 4096  # --max-seq-len of the qualified profile (collect_behavior.py)
TEMPLATE_RESERVE = 64  # chat-template tokens around the user message, with margin
MEASURED_TOKENS = {"lc01": 2978, "lc02": 3005, "lc03": 2981, "lc04": 2944, "lc05": 2957, "lc06": 2989,
                   "lc07": 2974, "lc08": 2919, "lc09": 3645, "lc10": 3664, "lc11": 3703, "lc12": 3694,
                   "tu09": 1372}

# ----------------------------------------------------------------- sustained
def _primes(count):
    out, n = [], 2
    while len(out) < count:
        if all(n % p for p in out if p * p <= n):
            out.append(n)
        n += 1
    return out


def _fib(count):
    out, a, b = [], 0, 1
    for _ in range(count):
        out.append(a)
        a, b = b, a + b
    return out


US_STATES = sorted([
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
    "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
    "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico",
    "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
    "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
    "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"])
NUM_ONLY = " Output only the numbers, with no other text."
# (id, prompt, params, max_tokens). mode: ints (every integer in the visible
# answer, in order, must equal `expected`), lines (every non-empty line,
# stripped, must equal `expected`), json (the parsed document must equal
# `expected`) or prose (word floor + loop detection only).
SUSTAINED = [
    ("su01", "List the integers from 1 to 200 in increasing order, one per line." + NUM_ONLY,
     {"mode": "ints", "expected": list(range(1, 201))}, 2048),
    ("su02", "List the first 100 prime numbers in increasing order, separated by commas." + NUM_ONLY,
     {"mode": "ints", "expected": _primes(100)}, 2048),
    ("su03", "List the multiples of 7 from 7 up to and including 700, one per line." + NUM_ONLY,
     {"mode": "ints", "expected": list(range(7, 701, 7))}, 2048),
    ("su04", "Count down from 150 to 1, one number per line." + NUM_ONLY,
     {"mode": "ints", "expected": list(range(150, 0, -1))}, 2048),
    ("su05", "List the squares of the integers 1 through 40 (1, 4, 9, ...), one per line." + NUM_ONLY,
     {"mode": "ints", "expected": [n * n for n in range(1, 41)]}, 2048),
    ("su06", "List the first 50 Fibonacci numbers starting 0, 1, one per line." + NUM_ONLY,
     {"mode": "ints", "expected": _fib(50)}, 2048),
    ("su07", "Return a JSON array of 60 objects. Object i (for i from 1 to 60, in order) has the "
             "integer key id equal to i and the string key name equal to \"item_\" followed by i, "
             "e.g. {\"id\": 1, \"name\": \"item_1\"}. Reply with only the JSON.",
     {"mode": "json", "expected": [{"id": i, "name": f"item_{i}"} for i in range(1, 61)]}, 3072),
    ("su08", "List all 50 U.S. states in alphabetical order, one per line, with no numbering and no "
             "other text.", {"mode": "lines", "expected": US_STATES}, 2048),
    ("su09", "Write a story of at least 400 words about a lighthouse keeper who finds a message in a "
             "bottle.", {"mode": "prose", "min_words": 300}, 2048),
    ("su10", "Explain in at least 350 words how a household refrigerator keeps food cold.",
     {"mode": "prose", "min_words": 250}, 2048),
]
TUNING_SUSTAINED = [
    ("tu10", "List the integers from 1 to 60 in increasing order, one per line." + NUM_ONLY,
     {"mode": "ints", "expected": list(range(1, 61))}, 2048),
]


# --------------------------------------------------------------------- build
def _arith(rows, split):
    return [case(cid, "arithmetic", prompt + ANSWER_SUFFIX, "arithmetic",
                 {"expected": decimal_text(value)}, 1024, split) for cid, prompt, value in rows]


def _code(table, split):
    return [case(cid, "code", prompt + CODE_SUFFIX, "code",
                 {"entry_point": re.search(r"function (\w+)\(", prompt).group(1),
                  "hidden_tests": tests, "timeout_sec": 10}, 2048, split)
            for cid, (prompt, tests, _) in table.items()]


def _tools(rows, split):
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in TOOLBOX}
    return [case(cid, "tool_args", prompt, "tool_args",
                 {"name": name, "expected_args": args, "schema": copy.deepcopy(schemas[name])},
                 256, split, tools=copy.deepcopy(TOOLBOX), tool_choice="auto")
            for cid, prompt, name, args in rows]


def _structured(rows, split):
    out = []
    for cid, prompt, schema, expected, enforce in rows:
        extra = {}
        if enforce:
            extra["response_format"] = {"type": "json_schema", "json_schema": {
                "name": cid, "strict": True, "schema": schema}}
        out.append(case(cid, "structured", prompt + ONLY_JSON, "structured",
                        {"schema": schema, "expected": expected}, 1024, split, **extra))
    return out


def _multilingual(rows, split):
    return [case(cid, "multilingual", prompt, "multilingual", {"language": lang, "all_of": groups},
                 1024, split) for cid, lang, prompt, groups in rows]


def _long(rows, split):
    out = []
    for cid, seed, chars, depth in rows:
        prompt, needle, distractors, _ = generate(seed, chars, depth)
        # Thinking-off turns without tools still reason briefly (low_think in
        # glm5_next.jinja) and that counts against max_tokens, so give the
        # answer every token the 4096 limit leaves after TEMPLATE_RESERVE.
        budget = min(512, SERVING_LIMIT - TEMPLATE_RESERVE - MEASURED_TOKENS[cid])
        out.append(case(cid, "long_context", prompt, "long_context",
                        {"needle": needle, "forbidden": distractors}, budget, split,
                        measured_content_tokens=MEASURED_TOKENS[cid]))
    return out


def _sustained(rows, split):
    return [case(cid, "sustained", prompt, "sustained", params, max_tokens, split)
            for cid, prompt, params, max_tokens in rows]


def build():
    heldout = (_arith(ARITHMETIC, "heldout") + _code(CODE, "heldout") + _tools(TOOL_CASES, "heldout")
               + _structured(STRUCTURED, "heldout") + _multilingual(MULTILINGUAL, "heldout")
               + _long(LONG, "heldout") + _sustained(SUSTAINED, "heldout"))
    tuning = (_arith(TUNING_ARITHMETIC, "tuning") + _code(TUNING_CODE, "tuning")
              + _tools(TUNING_TOOL_CASES, "tuning") + _structured(TUNING_STRUCTURED, "tuning")
              + _multilingual(TUNING_MULTILINGUAL, "tuning") + _long(TUNING_LONG, "tuning")
              + _sustained(TUNING_SUSTAINED, "tuning"))
    ids = [row["id"] for row in heldout + tuning]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case id")
    return heldout, tuning


def render(rows):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def main(argv):
    heldout, tuning = build()
    files = {"corpus.jsonl": render(heldout), "tuning.jsonl": render(tuning)}
    if argv == ["--check"]:
        stale = [name for name, text in files.items()
                 if open(os.path.join(HERE, name), encoding="utf-8").read() != text]
        if stale:
            sys.exit("stale: " + ", ".join(stale))
        print("corpus files match build_corpus.py")
        return
    if argv:
        sys.exit("usage: build_corpus.py [--check]")
    for name, text in files.items():
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as handle:
            handle.write(text)
    print(f"wrote {len(heldout)} held-out and {len(tuning)} tuning cases")


if __name__ == "__main__":
    main(sys.argv[1:])
