# SPDX-License-Identifier: AGPL-3.0-only
"""Sweep the reference FLA KDA autotune configurations on captured inputs.

The reference picks Triton configs by timing, so its output bytes can depend
on which config won. This probe pins every autotuned KDA kernel to one config
at a time (others stay at the supplied baseline), replays the captured M128
prefill through the model's varlen entry point, and reports byte mismatches
against the captured output. It also repeats the baseline to check
run-to-run determinism.

Usage (inside the reference container):
  python3 kda_reference_config_sweep.py --raw RAW --params PARAMS --rank R \
      [--baseline-cache TRITON_CACHE_COPY] --out OUT.json
"""

import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kda_reference_stage_dump import HL, D, LOWER_BOUND, Ops, load_case, load_params, mismatches  # noqa: E402

KERNELS = {
    "kda_gate_cumsum_fwd_kernel": "kda",
    "chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter": "kda",
    "chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra": "kda",
    "recompute_w_u_fwd_kernel": "kda",
    "chunk_gla_fwd_kernel_o": "kda",
    "merge_16x16_to_64x64_inverse_kernel": "solve_tril",
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": "chunk_delta_h",
}


def cfg_desc(c):
    return {"kwargs": dict(c.kwargs), "num_warps": c.num_warps, "num_stages": c.num_stages}


def same(c, d):
    return dict(c.kwargs) == d["kwargs"] and c.num_warps == d["num_warps"] and c.num_stages == d["num_stages"]


def baseline_from_cache(cache: str, name: str):
    """Configs chosen (min timing) in a reference autotune cache, one per cache key."""
    out = []
    for f in sorted(glob.glob(f"{cache}/*/{name}.autotune.json")):
        d = json.load(open(f))
        best = min(d["configs_timings"], key=lambda c: c[1][0])[0]
        out.append({"key": d["key"], "kwargs": best["kwargs"], "num_warps": best["num_warps"],
                    "num_stages": best["num_stages"]})
    return out


def run_e2e(ops, x, g1, beta_raw, conv_w, a_log, dt_bias, dev):
    t = x.shape[0]
    conv_states = torch.zeros(2, 3 * HL * D, 3, dtype=torch.bfloat16, device=dev)
    qsl = torch.tensor([0, t], dtype=torch.int32, device=dev)
    conv = ops.causal_conv1d_fn(
        x.transpose(0, 1), conv_w, None, activation="silu", conv_states=conv_states,
        has_initial_state=torch.zeros(1, dtype=torch.bool, device=dev),
        cache_indices=torch.ones(1, dtype=torch.int32, device=dev), query_start_loc=qsl,
    ).transpose(0, 1)
    beta = beta_raw.squeeze(0).float().sigmoid().unsqueeze(0)
    o, _ = ops.chunk_kda_with_fused_gate(
        q=conv[:, : HL * D].reshape(1, t, HL, D), k=conv[:, HL * D : 2 * HL * D].reshape(1, t, HL, D),
        v=conv[:, 2 * HL * D :].reshape(1, t, HL, D), raw_g=g1, beta=beta, A_log=a_log, g_bias=dt_bias,
        initial_state=torch.zeros(1, HL, D, D, dtype=torch.float32, device=dev), output_final_state=True,
        use_qk_l2norm_in_kernel=True, cu_seqlens=qsl, safe_gate=True, lower_bound=LOWER_BOUND,
    )
    torch.cuda.synchronize()
    return o.reshape(t, HL, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--baseline-cache", required=True)
    ap.add_argument("--inv", type=int, default=16)
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from pathlib import Path

    dev = torch.device("cuda")
    ops = Ops()
    conv_w, a_log, dt_bias = load_params(Path(args.params), args.rank, dev)
    x, g1, beta_raw, ref = load_case(Path(args.raw), args.rank, args.inv, args.rows, dev)

    tuners, all_cfgs, base = {}, {}, {}
    for name, mod in KERNELS.items():
        tuner = getattr(ops.modules[mod], name)
        while not hasattr(tuner, "configs"):  # unwrap @triton.heuristics
            tuner = tuner.fn
        tuners[name] = tuner
        all_cfgs[name] = list(tuner.configs)
        choices = baseline_from_cache(args.baseline_cache, name)
        # Prefer the choice for the varlen key (last key element differs by variant); keep all for the record.
        pick = None
        for ch in choices:
            for c in all_cfgs[name]:
                if same(c, ch):
                    pick = c
        base[name] = {"choices": choices, "picked": cfg_desc(pick) if pick else None}
        tuner.configs = [pick] if pick else all_cfgs[name][:1]

    def evaluate():
        for t in tuners.values():
            t.cache.clear() if hasattr(t, "cache") else None
        o = run_e2e(ops, x, g1, beta_raw, conv_w, a_log, dt_bias, dev)
        return mismatches(o, ref), o

    report = {"rank": args.rank, "inv": args.inv, "baseline": base, "repeats": [], "sweep": {}}
    first = None
    for _ in range(args.repeats):
        n, o = evaluate()
        if first is None:
            first = o.clone()
        report["repeats"].append({"vs_captured": n, "vs_first_repeat": mismatches(o, first)})
    print("baseline repeats", report["repeats"], flush=True)

    for name, cfgs in all_cfgs.items():
        keep = tuners[name].configs
        rows = []
        for c in cfgs:
            tuners[name].configs = [c]
            n, _ = evaluate()
            rows.append({"config": cfg_desc(c), "vs_captured": n})
        tuners[name].configs = keep
        report["sweep"][name] = rows
        print(name, sorted({(json.dumps(r["config"]["kwargs"]), r["vs_captured"]) for r in rows}), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
