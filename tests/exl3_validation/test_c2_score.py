# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only tests for C2 diagnostic scoring (c2_score)."""
import copy
import math

import numpy as np
import pytest

from c2_score import FINGERPRINT_TOKENS, score, validate_arm_identity
from statistical_gate import FLOORS, merge_run
from test_parity import provenance

VOCAB = 12
CASES = [{"id": "c", "prompt_ids": [1], "continuation_ids": [2, 3]}]
ROWS = 3


def _manifest(environment=None, engine="vllm", serve_flags=("serve", "test")):
    prov = provenance(engine)
    prov["environment"] = dict(environment or {})
    prov["serve_flags"] = list(serve_flags)
    return {"provenance": prov, "cases": copy.deepcopy(CASES),
            "tokens_sha256": "0" * 64, "vocab": VOCAB,
            "request_policy": {"temperature": 0, "seed": 42, "max_tokens": 1},
            "generated_text": {"c": ["t"] * ROWS}}


def _run(rows, environment=None):
    return merge_run([(_manifest(environment), {"c": np.asarray(rows, dtype=np.float32)})])


BASE = np.random.default_rng(0).normal(0, 1, (ROWS, VOCAB))
BASE[:, 0] += 20  # clear argmax margin so noise never flips it
# Softmax is shift-invariant: distributions change only with a non-constant perturbation.
RAMP = np.linspace(0.0, 1.0, VOCAB)


def _refs(rng, n=3, sigma=0.05):
    return [_run(BASE + rng.normal(0, sigma, BASE.shape)) for _ in range(n)]


def _log_softmax(values):
    total = sum(math.exp(value - max(values)) for value in values)
    return [value - max(values) - math.log(total) for value in values]


def _sym_kl_scalar(x, y):
    lx, ly = _log_softmax(list(x)), _log_softmax(list(y))
    return sum(math.exp(a) * (a - b) for a, b in zip(lx, ly)) + \
        sum(math.exp(b) * (b - a) for a, b in zip(lx, ly))


def _ce_scalar(x, y, token):
    lx, ly = _log_softmax(list(x)), _log_softmax(list(y))
    return abs(lx[token] - ly[token])


def _topk_miss_scalar(x, y):
    top_x = set(np.argsort(-np.asarray(x), kind="stable")[:10].tolist())
    top_y = set(np.argsort(-np.asarray(y), kind="stable")[:10].tolist())
    return 1 - len(top_x & top_y) / 10


def test_hand_computed_sym_kl_ce_error_topk_miss_and_delta_sign():
    rng = np.random.default_rng(3)
    refs = _refs(rng)
    x = BASE + rng.normal(0, 0.2, BASE.shape)
    arm_a, arm_b = _run(x), _run(x + rng.normal(0, 0.2, BASE.shape))
    report = score([f"ref{i}" for i in range(3)], refs, arm_a, arm_b)
    assert len(report["rows"]) == ROWS
    for entry in report["rows"]:
        row = entry["row"]
        case = CASES[0]
        # expected values computed from the same float32-rounded run rows
        a_row = arm_a["rows"][(0, "c")][row]
        b_row = arm_b["rows"][(0, "c")][row]
        ref_rows = [refs[i]["rows"][(0, "c")][row] for i in range(3)]
        expected_sym = np.mean([_sym_kl_scalar(a_row, ref_row) for ref_row in ref_rows])
        assert entry["arm_a"]["sym_kl_mean"] == pytest.approx(expected_sym, rel=1e-9, abs=1e-12)
        if row < len(case["continuation_ids"]):
            token = case["continuation_ids"][row]
            expected_ce = np.mean([_ce_scalar(a_row, ref_row, token) for ref_row in ref_rows])
            assert entry["arm_a"]["ce_error_mean"] == pytest.approx(expected_ce, rel=1e-9, abs=1e-12)
        else:
            assert entry["arm_a"]["ce_error_mean"] is None  # last row has no continuation
        expected_topk = np.mean([_topk_miss_scalar(a_row, ref_row) for ref_row in ref_rows])
        assert entry["arm_a"]["topk_miss_mean"] == pytest.approx(expected_topk, rel=1e-9, abs=1e-12)
        expected_delta = (np.mean([_sym_kl_scalar(b_row, ref_row) for ref_row in ref_rows])
                          - expected_sym)
        assert entry["primary_delta_sym_kl"] == pytest.approx(expected_delta, rel=1e-9, abs=1e-12)
        assert entry["delta_sign"] == ("closer" if expected_delta < 0 else
                                       "farther" if expected_delta > 0 else "no-metric-change")


def test_delta_sign_direction_closer_is_negative():
    rng = np.random.default_rng(4)
    refs = _refs(rng)
    near = BASE + 0.05 * RAMP
    far = BASE + 0.6 * RAMP
    report = score([f"ref{i}" for i in range(3)], refs, _run(far), _run(near))
    for entry in report["rows"]:
        assert entry["primary_delta_sym_kl"] < 0
        assert entry["delta_sign"] == "closer"
    assert report["row_sign_result"] == "closer"
    assert report["row_delta_summary"] == {"closer": ROWS, "farther": 0,
                                           "no-metric-change": 0}


def test_mixed_row_signs_reported_as_mixed():
    rng = np.random.default_rng(5)
    refs = _refs(rng, sigma=0.0)
    rows_a = BASE + 0.1 * RAMP
    rows_b = BASE.copy()
    rows_b[0] = BASE[0] + 0.05 * RAMP  # row 0 closer than A
    rows_b[1] = BASE[1] + 0.6 * RAMP   # row 1 farther than A
    rows_b[2] = BASE[2] + 0.1 * RAMP   # row 2 identical to A
    report = score([f"ref{i}" for i in range(3)], refs, _run(rows_a), _run(rows_b))
    signs = [entry["delta_sign"] for entry in report["rows"]]
    assert signs == ["closer", "farther", "no-metric-change"]
    assert report["row_sign_result"] == "mixed"
    assert report["row_delta_summary"] == {"closer": 1, "farther": 1, "no-metric-change": 1}


def test_raw_logit_byte_equality_reported_separately():
    rng = np.random.default_rng(6)
    refs = _refs(rng)
    rows_b = BASE + 0.05 * RAMP
    report = score([f"ref{i}" for i in range(3)], refs, _run(rows_b.copy()), _run(rows_b.copy()))
    assert report["raw_logit_byte_equality"] == {"rows_total": ROWS, "rows_byte_equal": ROWS,
                                                 "all_rows_equal": True}
    for entry in report["rows"]:
        assert entry["raw_logit_bytes"]["a_b_byte_equal"] is True
        assert entry["raw_logit_bytes"]["arm_a_sha256"] == entry["raw_logit_bytes"]["arm_b_sha256"]
        assert entry["primary_delta_sym_kl"] == 0.0
        assert entry["delta_sign"] == "no-metric-change"
    assert report["row_sign_result"] == "no-metric-change"


def test_per_row_catastrophic_checks_against_frozen_limits():
    rng = np.random.default_rng(7)
    refs = _refs(rng, sigma=0.0)
    rows_b = BASE.copy()
    rows_b[2, 3:] = 0.0  # one catastrophic row for B only
    rows_b[2, 2] = 40.0
    report = score([f"ref{i}" for i in range(3)], refs, _run(BASE.copy()), _run(rows_b))
    catastrophic_rows = [entry for entry in report["rows"]
                         if entry["catastrophic"]["arm_b"]["sym_kl"]["catastrophic"]]
    assert [entry["row"] for entry in catastrophic_rows] == [2]
    check = catastrophic_rows[0]["catastrophic"]["arm_b"]["sym_kl"]
    assert check["nearest"] > check["limit"]
    assert check["limit"] >= FLOORS["sym_kl"]
    assert all(entry["catastrophic"]["arm_a"]["sym_kl"]["catastrophic"] is False
               for entry in report["rows"])
    assert report["scoring_rules"]["catastrophic_limits"] == {
        "sym_kl": FLOORS["sym_kl"], "ce_error": FLOORS["ce_error"], "catastrophe_factor": 3.0}


def test_five_token_fingerprint_is_explanatory_only(tmp_path):
    rng = np.random.default_rng(8)
    refs = _refs(rng)
    report = score([f"ref{i}" for i in range(3)], refs, _run(BASE + 0.01), _run(BASE + 0.02))
    fingerprint = report["five_token_fingerprint"]
    assert fingerprint["role"] == "explanatory-only"
    assert fingerprint["verdict_influence"] == "none"
    assert fingerprint["tokens"] == list(FINGERPRINT_TOKENS)
    assert fingerprint["missing_tokens"] == list(FINGERPRINT_TOKENS)  # VOCAB too small
    assert [entry["row"] for entry in fingerprint["rows"]] == [0, 1, 2]
    assert "passed" not in report and report["q2_gate"]["pass_claimed"] is False


def test_fingerprint_values_reported_without_touching_verdicts():
    vocab = 30000
    rows = np.zeros((2, vocab), dtype=np.float32)
    rows[:, 0] = 5.0
    for token in FINGERPRINT_TOKENS:
        rows[1, token] = 3.0 - token / 10000.0

    def wide(offset):
        return merge_run([({"provenance": provenance("vllm"),
                            "cases": [{"id": "w", "prompt_ids": [1], "continuation_ids": [2]}],
                            "tokens_sha256": "0" * 64, "vocab": vocab,
                            "request_policy": {"temperature": 0, "seed": 42, "max_tokens": 1},
                            "generated_text": {"w": ["t", "t"]}},
                           {"w": (rows + offset).astype(np.float32)})])

    refs = [wide(0.01 * i) for i in range(3)]
    report = score([f"ref{i}" for i in range(3)], refs, wide(0.0), wide(0.0))
    fingerprint = report["five_token_fingerprint"]
    assert fingerprint["missing_tokens"] == []
    runs = fingerprint["rows"][1]["runs"]
    assert runs["arm_a"]["order"] == sorted(
        (str(token) for token in FINGERPRINT_TOKENS),
        key=lambda token: -(3.0 - int(token) / 10000.0))
    assert report["row_sign_result"] == "no-metric-change"  # verdicts ignore the fingerprint


def test_every_row_scored_without_subset_knobs():
    rng = np.random.default_rng(9)
    cases = [{"id": "a", "prompt_ids": [1], "continuation_ids": [2]},
             {"id": "b", "prompt_ids": [1], "continuation_ids": [2, 3]}]

    def multi(offset):
        arrays = {"a": (BASE[:2] + offset).astype(np.float32),
                  "b": (BASE + offset).astype(np.float32)}
        manifest = _manifest()
        manifest["cases"] = copy.deepcopy(cases)
        manifest["generated_text"] = {"a": ["t"] * 2, "b": ["t"] * 3}
        return merge_run([(manifest, arrays)])

    refs = [multi(0.01 * i) for i in range(3)]
    report = score([f"ref{i}" for i in range(3)], refs, multi(0.05), multi(0.06))
    assert [(entry["case"], entry["row"]) for entry in report["rows"]] == [
        ("a", 0), ("a", 1), ("b", 0), ("b", 1), ("b", 2)]
    assert len(report["five_token_fingerprint"]["rows"]) == 5


def test_mixed_configurations_between_arms_refused():
    rng = np.random.default_rng(10)
    refs = _refs(rng)
    knob = {"ATLAS_EXL3_NORM_SUBSTITUTE_DIR": "/fixtures",
            "ATLAS_EXL3_NORM_SUBSTITUTE_RECORD_DIR": "/records"}
    arm_a = _run(BASE, environment={})
    arm_b = _run(BASE, environment=knob)
    report = score([f"ref{i}" for i in range(3)], refs, arm_a, arm_b)
    assert report["arm_configuration"]["knob"]["arm_a"]["engaged"] is False
    assert report["arm_configuration"]["knob"]["arm_b"]["engaged"] is True

    mixed = _run(BASE, environment={"TRACE": "on", **knob})
    with pytest.raises(ValueError, match="beyond the knob"):
        score([f"ref{i}" for i in range(3)], refs, arm_a, mixed)

    other_flags = _run(BASE, environment=knob)
    other_flags["manifests"][0]["provenance"]["serve_flags"] = ["serve", "other"]
    with pytest.raises(ValueError, match="serve_flags"):
        score([f"ref{i}" for i in range(3)], refs, arm_a, other_flags)

    mixed_case = _run(BASE, environment=knob)
    mixed_case["manifests"][0]["cases"][0]["continuation_ids"] = [2, 7]
    with pytest.raises(ValueError, match="cases"):
        score([f"ref{i}" for i in range(3)], refs, arm_a, mixed_case)


def test_knob_engagement_mismatch_marks_attribution_unresolved():
    rng = np.random.default_rng(11)
    refs = _refs(rng)
    report = score([f"ref{i}" for i in range(3)], refs, _run(BASE), _run(BASE + 0.05 * RAMP))
    assert report["arm_configuration"]["attribution"]["status"] == "unresolved"


def test_validate_arm_identity_accepts_knob_only_difference():
    a = _run(BASE, environment={})
    b = _run(BASE, environment={"ATLAS_EXL3_NORM_SUBSTITUTE_DIR": "/fixtures"})
    configuration = validate_arm_identity(a, b)
    assert configuration["arm_b_environment"]["ATLAS_EXL3_NORM_SUBSTITUTE_DIR"] == "/fixtures"
    assert "ATLAS_EXL3_NORM_SUBSTITUTE_DIR" not in configuration["arm_a_environment"]


OUTPUT_KEYS = ("ATLAS_EXL3_LOGITS_DIR", "ATLAS_EXL3_UPSTREAM_DUMP_DIR", "ATLAS_EXL3_FFN_DUMP_DIR")


def _diagnostic_arm(arm):
    run_name = f"arm-{arm}-repeat-00"
    root = f"/evidence/c2/runs/{run_name}"
    environment = dict(zip(OUTPUT_KEYS, [root + "/logits", "/evidence/c2/gated_norm/" + run_name,
                                        root + "/ffn"]))
    if arm == "b":
        environment.update({"ATLAS_EXL3_NORM_SUBSTITUTE_DIR": "/fixtures",
                            "ATLAS_EXL3_NORM_SUBSTITUTE_RECORD_DIR": "/records"})
    run = _run(BASE, environment)
    run["manifests"][0]["captures"] = [{"capture": root + f"/logits/logits-{i:06d}.json"}
                                         for i in range(ROWS)]
    return run


def test_fresh_diagnostic_outputs_are_bound_and_reported_without_changing_metrics():
    refs = _refs(np.random.default_rng(12))
    a, b = _diagnostic_arm("a"), _diagnostic_arm("b")
    report = score([f"ref{i}" for i in range(3)], refs, a, b)
    outputs = report["arm_configuration"]["diagnostic_outputs"]
    assert outputs["arm_a"] == {key: a["manifests"][0]["provenance"]["environment"][key]
                                for key in OUTPUT_KEYS}
    assert outputs["arm_b"] == {key: b["manifests"][0]["provenance"]["environment"][key]
                                for key in OUTPUT_KEYS}
    assert report["raw_logit_byte_equality"]["all_rows_equal"]


@pytest.mark.parametrize("damage", ["relative", "missing", "wrong-role", "foreign-capture",
                                    "no-captures", "non-string", "dotdot", "wrong-arm", "foreign-attempt"])
def test_unbound_diagnostic_outputs_refused(damage):
    a, b = _diagnostic_arm("a"), _diagnostic_arm("b")
    manifest = b["manifests"][0]
    env = manifest["provenance"]["environment"]
    if damage == "relative":
        env[OUTPUT_KEYS[0]] = "relative/logits"
    elif damage == "missing":
        del env[OUTPUT_KEYS[1]]
    elif damage == "wrong-role":
        env[OUTPUT_KEYS[2]] = env[OUTPUT_KEYS[0]]
    elif damage == "foreign-capture":
        manifest["captures"][0]["capture"] = "/foreign/logits-000000.json"
    elif damage == "no-captures":
        manifest["captures"] = []
    elif damage == "non-string":
        env[OUTPUT_KEYS[0]] = 3
    elif damage == "dotdot":
        env[OUTPUT_KEYS[2]] += "/../ffn"
    elif damage == "foreign-attempt":
        for key in OUTPUT_KEYS:
            env[key] = env[key].replace("/evidence/c2/", "/different/attempt/")
        for capture in manifest["captures"]:
            capture["capture"] = capture["capture"].replace("/evidence/c2/", "/different/attempt/")
    elif damage == "wrong-arm":
        env.update(a["manifests"][0]["provenance"]["environment"])
        manifest["captures"] = a["manifests"][0]["captures"]
    with pytest.raises(ValueError, match="diagnostic output"):
        validate_arm_identity(a, b)


@pytest.mark.parametrize("key", ["ATLAS_KDA_FLA_REFERENCE", "ATLAS_EXL3_UPSTREAM_DUMP_LAYERS",
                                 "ATLAS_EXL3_FFN_DUMP_LAYERS", "ATLAS_EXL3_VERIFY_IO_DIR"])
def test_diagnostic_path_normalization_never_exempts_other_environment(key):
    a, b = _diagnostic_arm("a"), _diagnostic_arm("b")
    b["manifests"][0]["provenance"]["environment"][key] = "different"
    with pytest.raises(ValueError, match="beyond the knob"):
        validate_arm_identity(a, b)
