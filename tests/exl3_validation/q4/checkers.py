# SPDX-License-Identifier: AGPL-3.0-only
"""Q4 task-quality checkers.

Each checker is `check_<type>(response, params) -> {"passed": bool, "reason": str}`
where `response` is the runner's normalised view of one chat completion:
{"content": str, "tool_calls": list | None, "finish_reason": str | None}.
A bare string is accepted as content-only for convenience. Checkers score the
visible answer against the case's own objective expectation only; nothing is
ever compared with a reference engine's output, so a reference engine making
the same mistake cannot turn a wrong answer into a pass.

Reasoning is stripped before every check: complete <think>...</think> blocks,
everything before an orphan </think> (template-prefilled <think>), and an
unterminated trailing <think> (max_tokens hit mid-reasoning). The same for
[THINK]...[/THINK]. Only check_code has side effects: it runs generated code
in a subprocess with an empty environment, isolated interpreter (-I), no
stdin, rlimits and a timeout, inside a fresh directory under the given
temporary root (default $TMPDIR; it refuses to fall back to /tmp).
"""
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unicodedata
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from behavior_schema import parse_strict_json, schema_errors  # noqa: E402


def _view(response):
    if isinstance(response, str):
        return {"content": response, "tool_calls": None, "finish_reason": None}
    if not isinstance(response, dict):
        raise TypeError("response must be a dict or str")
    return {"content": response.get("content") or "", "tool_calls": response.get("tool_calls"),
            "finish_reason": response.get("finish_reason")}


def strip_reasoning(text):
    text = text or ""
    for open_tag, close_tag in (("<think>", "</think>"), ("[THINK]", "[/THINK]")):
        o, c = re.escape(open_tag), re.escape(close_tag)
        text = re.sub(o + ".*?" + c, "", text, flags=re.DOTALL | re.IGNORECASE)
        # An orphan close tag means the open tag was in the prompt.
        parts = re.split(c, text, flags=re.IGNORECASE)
        text = parts[-1]
        text = re.sub(o + ".*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _ok(reason):
    return {"passed": True, "reason": reason}


def _fail(reason):
    return {"passed": False, "reason": reason}


def same_json(a, b):
    """Strict JSON equality: bool never equals a number, 2 == 2.0 is allowed."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_json(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_json(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


# ---------------------------------------------------------------- arithmetic
_ANSWER = re.compile(r"answer\s*\**\s*[:：]\s*\**\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)


def check_arithmetic(response, params):
    """The LAST 'Answer: <number>' must equal params['expected'] exactly.
    Thousands separators and a leading $ are tolerated; any other number
    elsewhere in the text is ignored."""
    expected = Fraction(params["expected"])
    visible = strip_reasoning(_view(response)["content"]).replace("−", "-")
    matches = _ANSWER.findall(visible)
    if not matches:
        return _fail("no 'Answer: <number>' line")
    raw = matches[-1]
    if "," in raw and not re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", raw):
        return _fail("malformed number " + repr(raw))
    got = Fraction(raw.replace(",", ""))
    if got != expected:
        return _fail(f"answer {raw} != expected {params['expected']}")
    return _ok("exact answer " + raw)


# ---------------------------------------------------------------------- code
_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", re.DOTALL)


def extract_code(text, entry_point):
    """Return the first fenced python (or untagged) block defining
    `entry_point`; with no fences, the whole visible text if it defines it."""
    visible = strip_reasoning(text)
    define = re.compile(r"^\s*def\s+" + re.escape(entry_point) + r"\s*\(", re.MULTILINE)
    blocks = _FENCE.findall(visible)
    if blocks:
        for lang, body in blocks:
            if lang.lower() in ("", "python", "py", "python3") and define.search(body):
                return body
        return None
    return visible if define.search(visible) else None


_HARNESS = textwrap.dedent('''
    import importlib.util, sys
    def main():
        spec = importlib.util.spec_from_file_location("solution", "solution.py")
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except BaseException as error:
            sys.stderr.write("IMPORT_ERROR: %r\\n" % (error,))
            return 3
        source = open("hidden_tests.py", encoding="utf-8").read()
        try:
            exec(compile(source, "hidden_tests.py", "exec"), {"mod": mod})
        except AssertionError as error:
            sys.stderr.write("ASSERTION_FAILED: %r\\n" % (error,))
            return 4
        except BaseException as error:
            sys.stderr.write("TEST_ERROR: %r\\n" % (error,))
            return 5
        sys.stdout.write("@NONCE@\\n")
        return 0
    sys.exit(main())
''')


def _limits(cpu_seconds):
    cpu_seconds = int(math.ceil(cpu_seconds))

    def apply():
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
    return apply


def check_code(response, params, tmp_root=None):
    root = tmp_root if tmp_root is not None else os.environ.get("TMPDIR")
    if not root or not os.path.isdir(root):
        raise RuntimeError("code checker needs an existing TMPDIR (or tmp_root); refusing /tmp")
    code = extract_code(_view(response)["content"], params["entry_point"])
    if code is None:
        return _fail("no python code block defining " + params["entry_point"])
    timeout = params.get("timeout_sec", 10)
    workdir = tempfile.mkdtemp(prefix="q4_code_", dir=root)
    # A per-run nonce, so generated code cannot fake the pass sentinel.
    nonce = "Q4_PASS_" + secrets.token_hex(16)
    try:
        for name, text in (("solution.py", code), ("hidden_tests.py", params["hidden_tests"]),
                           ("harness.py", _HARNESS.replace("@NONCE@", nonce))):
            with open(os.path.join(workdir, name), "w", encoding="utf-8") as handle:
                handle.write(text)
        try:
            proc = subprocess.run([sys.executable, "-I", "harness.py"], cwd=workdir, env={},
                                  stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                  timeout=timeout, preexec_fn=_limits(timeout + 1))
        except subprocess.TimeoutExpired:
            return _fail(f"timed out after {timeout}s")
        if proc.returncode == 0 and proc.stdout.strip().splitlines()[-1:] == [nonce]:
            return _ok("hidden tests passed")
        return _fail(f"exit={proc.returncode} stderr={proc.stderr[-400:]!r}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ----------------------------------------------------------------- tool_args
def check_tool_args(response, params):
    """Exactly one tool call, the expected function, arguments that are a
    strict JSON object valid against the tool schema and equal (strict JSON
    equality) to the expected arguments."""
    calls = _view(response)["tool_calls"]
    if not calls:
        return _fail("no tool_calls")
    if len(calls) != 1:
        return _fail(f"expected one tool call, got {len(calls)}")
    call = calls[0] if isinstance(calls[0], dict) else {}
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    if function.get("name") != params["name"]:
        return _fail(f"called {function.get('name')!r}, expected {params['name']!r}")
    raw = function.get("arguments")
    try:
        args = parse_strict_json(raw) if isinstance(raw, str) else raw
    except ValueError as error:
        return _fail("arguments are not strict JSON: " + str(error))
    if not isinstance(args, dict):
        return _fail("arguments are not an object")
    errors = schema_errors(args, params["schema"])
    if errors:
        return _fail("schema: " + "; ".join(errors))
    if not same_json(args, params["expected_args"]):
        return _fail(f"arguments {args!r} != expected {params['expected_args']!r}")
    return _ok("exact tool call")


# ---------------------------------------------------------------- structured
def _unfence(text):
    match = re.fullmatch(r"```[ \t]*(?:json)?[ \t]*\n(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def check_structured(response, params):
    """The whole visible answer (optionally one ```json fence) must be strict
    JSON, valid against the schema and equal to the expected document."""
    text = _unfence(strip_reasoning(_view(response)["content"]))
    try:
        value = parse_strict_json(text)
    except ValueError as error:
        return _fail("not strict JSON: " + str(error))
    errors = schema_errors(value, params["schema"])
    if errors:
        return _fail("schema: " + "; ".join(errors))
    if "expected" in params and not same_json(value, params["expected"]):
        return _fail(f"value {value!r} != expected {params['expected']!r}")
    return _ok("schema-valid expected document")


# -------------------------------------------------------------- multilingual
MARKERS = {
    "fr": {"le", "la", "les", "l", "est", "sont", "de", "des", "du", "d", "et", "un", "une", "en",
           "que", "qui", "il", "elle", "je", "vous", "au", "aux", "a", "ce", "c", "s"},
    "de": {"der", "die", "das", "den", "dem", "des", "ist", "sind", "und", "ein", "eine", "einen",
           "hat", "haben", "mit", "auf", "im", "nicht", "es", "zu"},
    "es": {"el", "la", "los", "las", "es", "son", "de", "del", "y", "un", "una", "en", "que", "tiene"},
}
ENGLISH = {"the", "is", "are", "of", "and", "has", "have", "it", "this", "with", "was", "which"}
_HAN = re.compile(r"[一-鿿㐀-䶿]")
_KANA = re.compile(r"[぀-ヿ]")
_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")


def language_ok(text, language):
    if language in ("zh", "ja"):
        han, kana, latin = len(_HAN.findall(text)), len(_KANA.findall(text)), len(_LATIN.findall(text))
        if language == "zh":
            return han >= 2 and kana == 0 and han > latin
        return kana >= 1 and han + kana > latin
    words = re.findall(r"[^\W\d_]+", text)
    target = sum(word in MARKERS[language] for word in words)
    english = sum(word in ENGLISH for word in words)
    return target >= 1 and target > english


def _contains(text, variant):
    if re.search(r"[぀-ヿ㐀-鿿]", variant):
        return variant in text
    # Whole-word match against Latin letters and digits only: CJK text has
    # no spaces, so "12" must match in "一年は12か月" but not in "112".
    edge = r"0-9A-Za-z\u00c0-\u024f"
    return re.search(f"(?<![{edge}])" + re.escape(variant) + f"(?![{edge}])", text) is not None


def check_multilingual(response, params):
    """Answer must be in the requested language and contain, for each group
    in all_of, at least one accepted variant (NFKC + casefold; Latin variants
    match whole words)."""
    text = unicodedata.normalize("NFKC", strip_reasoning(_view(response)["content"])).casefold()
    if not text:
        return _fail("empty answer")
    if not language_ok(text, params["language"]):
        return _fail("answer is not in " + params["language"])
    for group in params["all_of"]:
        if not any(_contains(text, unicodedata.normalize("NFKC", v).casefold()) for v in group):
            return _fail(f"missing any of {group!r}")
    return _ok("language and facts present")


# -------------------------------------------------------------- long_context
def check_long_context(response, params):
    visible = strip_reasoning(_view(response)["content"])
    if params["needle"] not in visible:
        return _fail("needle code not retrieved")
    wrong = [code for code in params.get("forbidden", []) if code in visible]
    if wrong:
        return _fail(f"distractor codes in answer: {wrong}")
    return _ok("needle retrieved")


# ----------------------------------------------------------------- sustained
def has_repetition_loop(text, min_repeats=4, max_ngram=12):
    """True if some 1..max_ngram word n-gram repeats min_repeats times back
    to back (same rule as tests/single_gpu_suite.py _has_repetition_loop)."""
    toks = text.split()
    for n in range(1, max_ngram + 1):
        for i in range(len(toks) - n * min_repeats + 1):
            window = toks[i:i + n]
            if all(toks[i + r * n:i + (r + 1) * n] == window for r in range(1, min_repeats)):
                return True
    return False


def _first_difference(got, expected, visible=""):
    if has_repetition_loop(visible):
        return "repetition loop; " + _first_difference(got, expected)
    for index, (a, b) in enumerate(zip(got, expected)):
        if a != b:
            return f"item {index}: got {a!r}, expected {b!r}"
    return f"length {len(got)} != expected {len(expected)}"


def check_sustained(response, params):
    """Long output that must finish on its own (finish_reason 'stop') and
    match exactly: catches loops, skipped items and premature stops."""
    view = _view(response)
    if view["finish_reason"] not in (None, "stop"):
        return _fail(f"finish_reason={view['finish_reason']} (truncated or aborted)")
    visible = strip_reasoning(view["content"])
    mode = params["mode"]
    if mode == "prose":
        if has_repetition_loop(visible, params.get("min_repeats", 4)):
            return _fail("repetition loop")
        words = len(visible.split())
        if words < params["min_words"]:
            return _fail(f"{words} words < {params['min_words']}")
        return _ok(f"{words} words, no loop")
    if mode == "ints":
        # Lines containing letters (a preamble such as "Here are the first
        # 100 primes:") are not part of the sequence; numbering such as
        # "1. 2" still corrupts it and fails, as the prompts forbid it.
        got = [int(x) for line in visible.splitlines() if not re.search(r"[^\W\d_]", line)
               for x in re.findall(r"-?\d+", line)]
    elif mode == "lines":
        got = [line.strip() for line in visible.splitlines() if line.strip()]
    elif mode == "json":
        try:
            got = parse_strict_json(_unfence(visible))
        except ValueError as error:
            return _fail("not strict JSON: " + str(error))
        if not same_json(got, params["expected"]):
            return _fail("JSON differs: " + (_first_difference(got, params["expected"], visible)
                                             if isinstance(got, list) else "not an array"))
        return _ok("exact JSON document")
    else:
        raise ValueError("unknown sustained mode " + repr(mode))
    if got != params["expected"]:
        return _fail("sequence differs: " + _first_difference(got, params["expected"], visible))
    return _ok(f"exact {len(got)}-item sequence")


CHECKERS = {
    "arithmetic": check_arithmetic,
    "code": check_code,
    "tool_args": check_tool_args,
    "structured": check_structured,
    "multilingual": check_multilingual,
    "long_context": check_long_context,
    "sustained": check_sustained,
}
# Parameters each checker requires; corpus validation enforces these.
REQUIRED_PARAMS = {
    "arithmetic": {"expected"},
    "code": {"entry_point", "hidden_tests"},
    "tool_args": {"name", "expected_args", "schema"},
    "structured": {"schema", "expected"},
    "multilingual": {"language", "all_of"},
    "long_context": {"needle", "forbidden"},
    "sustained": {"mode"},
}
