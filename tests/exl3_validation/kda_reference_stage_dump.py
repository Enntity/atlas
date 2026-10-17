# SPDX-License-Identifier: AGPL-3.0-only
"""Run the pinned FLA/vLLM KDA reference stage by stage on captured inputs.

Executes inside the reference image (vLLM + Triton) on GB10. For every
captured layer-0 case it replays conv -> l2norm -> gate cumsum -> KKT ->
solve_tril -> recompute_w_u -> delta_h -> output (prefill) or
conv update -> fused_recurrent (decode), stores each intermediate as raw
little-endian bytes with a JSON manifest (shape, dtype, sha256), and compares
the final output against the captured reference output byte for byte.

The per-stage bytes are the acceptance oracle for the Atlas
reference-policy KDA kernels (docs/design/exl3-kda-reference-core.md §3.7).
Set TRITON_CACHE_DIR to an empty directory to capture the lowered PTX.

Usage (inside the container):
  python3 kda_reference_stage_dump.py --raw RAW_DIR --params PARAMS_JSON \
      --rank 0 --out OUT_DIR [--source-root PINNED_FLA_SOURCE]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

HL, D, H_TOTAL = 32, 128, 64
LOWER_BOUND = -5.0


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().tobytes()
    return t.numpy().tobytes()


class Dumper:
    def __init__(self, out: Path):
        self.out = out
        self.manifest = {}
        out.mkdir(parents=True, exist_ok=True)

    def put(self, name: str, t: torch.Tensor):
        b = tensor_bytes(t)
        (self.out / f"{name}.bin").write_bytes(b)
        self.manifest[name] = {
            "dtype": str(t.dtype).replace("torch.", ""),
            "shape": list(t.shape),
            "sha256": sha256(b),
        }


def _cast_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """The reference model's `_cast_sigmoid` (fp32 cast + sigmoid)."""
    return x.float().sigmoid()


# The served model runs `_cast_sigmoid` inside its torch.compile graph, so the
# captured beta comes from an Inductor kernel; `--beta eager` keeps ATen.
BETA_FN = _cast_sigmoid


def load_bf16(path: Path, shape, dev):
    data = bytearray(path.read_bytes())
    t = torch.frombuffer(data, dtype=torch.int16).view(torch.bfloat16)
    return t.reshape(shape).to(dev)


def load_params(params_json: Path, rank: int, dev):
    d = json.loads(params_json.read_text())

    def get(suffix, dtype):
        (key,) = [k for k in d if k.endswith(suffix)]
        raw = bytearray(bytes.fromhex(d[key]["hex"]))
        if dtype == torch.bfloat16:
            t = torch.frombuffer(raw, dtype=torch.int16).view(torch.bfloat16)
        else:
            t = torch.frombuffer(raw, dtype=torch.float32)
        return t.reshape(d[key]["shape"])

    conv = get("conv1d.weight", torch.bfloat16)[:, 0, :]
    rows = torch.cat(
        [
            torch.arange(c * H_TOTAL * D + rank * HL * D, c * H_TOTAL * D + (rank + 1) * HL * D)
            for c in range(3)
        ]
    )
    # The reference conv modules use params_dtype=float32 (BF16 checkpoint
    # values widened), so the conv products are exact in f32.
    conv_w = conv[rows].float().contiguous().to(dev)
    a_log = get("A_log", torch.float32)[rank * HL : (rank + 1) * HL].contiguous().to(dev)
    dt_bias = get("dt_bias", torch.float32)[rank * HL * D : (rank + 1) * HL * D].contiguous().to(dev)
    return conv_w, a_log, dt_bias


def load_case(raw: Path, rank: int, inv: int, t: int, dev):
    def rd(stage, shape):
        return load_bf16(raw / f"ffn-L0-rank{rank}-{inv:06d}-kda-core-{stage}.bin", shape, dev)

    return (
        rd("input-qkv_proj_states", (t, 3 * HL * D)),
        rd("input-g1", (1, t, HL, D)),
        rd("input-beta", (1, t, HL)),
        rd("output", (t, HL, D)),
    )


def mismatches(a: torch.Tensor, ref: torch.Tensor) -> int:
    a16 = a.detach().contiguous().view(torch.int16).reshape(-1)
    r16 = ref.detach().contiguous().view(torch.int16).reshape(-1)
    return int((a16 != r16).sum().item())


def prefill(ops, x, g1, beta_raw, conv_w, a_log, dt_bias, dump: Dumper, dev, conv_states=None, h_init=None):
    """Chunked prefill. `conv_states` is a 2-slot cache (slot 1 live) and `h_init`
    a [1, HL, V, K] f32 state; both None means a fresh sequence."""
    t = x.shape[0]
    has_init = conv_states is not None
    if conv_states is None:
        # Slot 0 is vLLM's NULL_BLOCK_ID (skipped by the kernels); use slot 1.
        conv_states = torch.zeros(2, 3 * HL * D, 3, dtype=torch.bfloat16, device=dev)
    if h_init is None:
        h_init = torch.zeros(1, HL, D, D, dtype=torch.float32, device=dev)
    dump.put("in_x", x)
    dump.put("in_g1", g1)
    dump.put("in_beta_raw", beta_raw)
    dump.put("conv_state_before", conv_states[1])
    dump.put("initial_state", h_init)
    conv_before = conv_states.clone()
    qsl = torch.tensor([0, t], dtype=torch.int32, device=dev)
    conv = ops.causal_conv1d_fn(
        x.transpose(0, 1),
        conv_w,
        None,
        activation="silu",
        conv_states=conv_states,
        has_initial_state=torch.full((1,), has_init, dtype=torch.bool, device=dev),
        cache_indices=torch.ones(1, dtype=torch.int32, device=dev),
        query_start_loc=qsl,
    ).transpose(0, 1)
    dump.put("conv_out", conv)
    dump.put("conv_state_after", conv_states[1])
    q, k, v = (c.reshape(1, t, HL, D) for c in conv.split(HL * D, dim=-1))
    q = ops.l2norm_fwd(q.contiguous())
    k = ops.l2norm_fwd(k.contiguous())
    v = v.contiguous()
    dump.put("q_l2", q)
    dump.put("k_l2", k)
    beta = BETA_FN(beta_raw.squeeze(0)).unsqueeze(0)
    dump.put("beta", beta)
    scale = D**-0.5
    # The model always passes cu_seqlens, selecting the varlen kernel variants.
    ci = ops.prepare_chunk_indices(qsl, 64)
    vl = {"cu_seqlens": qsl, "chunk_indices": ci}
    g = ops.fused_kda_gate_chunk_cumsum(
        g1.contiguous(), A_log=a_log, g_bias=dt_bias, safe_gate=True, lower_bound=LOWER_BOUND, **vl
    )
    dump.put("g_cumsum", g)
    a, aqk = ops.chunk_kda_scaled_dot_kkt_fwd(
        q=q, k=k, gk=g, beta=beta, scale=scale, output_dtype=torch.float32, **vl
    )
    dump.put("A", a)
    dump.put("Aqk", aqk)
    ai = ops.solve_tril(A=a, cu_seqlens=qsl, output_dtype=k.dtype)
    dump.put("Ai", ai)
    w, u, _, kg = ops.recompute_w_u_fwd(k=k, v=v, beta=beta, A=ai, gk=g, **vl)
    dump.put("w", w)
    dump.put("u", u)
    dump.put("kg", kg)
    h0 = h_init.clone()
    h, v_new, final_state = ops.chunk_gated_delta_rule_fwd_h(
        k=kg, w=w, u=u, gk=g, initial_state=h0, output_final_state=True, use_exp2=True, **vl
    )
    dump.put("h", h)
    dump.put("v_new", v_new)
    dump.put("final_state", final_state)
    o = ops.chunk_gla_fwd_o_gk(q=q, v=v_new, g=g, A=aqk, h=h, o=v.clone(), scale=scale, **vl)
    dump.put("output", o)
    # End-to-end call of the same wrapper the model uses, for a self-consistency check.
    o_e2e, fs_e2e = ops.chunk_kda_with_fused_gate(
        q=conv[:, : HL * D].reshape(1, t, HL, D),
        k=conv[:, HL * D : 2 * HL * D].reshape(1, t, HL, D),
        v=conv[:, 2 * HL * D :].reshape(1, t, HL, D),
        raw_g=g1,
        beta=beta,
        A_log=a_log,
        g_bias=dt_bias,
        initial_state=h_init.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=qsl,
        safe_gate=True,
        lower_bound=LOWER_BOUND,
    )
    dump.put("output_e2e", o_e2e)
    dump.put("final_state_e2e", fs_e2e)
    del conv_before
    return o, o_e2e, final_state, conv_states


def decode(ops, x, g1, beta_raw, conv_w, a_log, dt_bias, conv_state, state, dump: Dumper, dev):
    """One decode row; `conv_state` (2-slot) is updated in place, `state` is [1, HL, V, K]."""
    dump.put("in_x", x)
    dump.put("in_g1", g1)
    dump.put("in_beta_raw", beta_raw)
    dump.put("conv_state_before", conv_state[1])
    conv = ops.causal_conv1d_update(
        x,
        conv_state,
        conv_w,
        None,
        activation="silu",
        conv_state_indices=torch.ones(1, dtype=torch.int32, device=dev),
    )
    dump.put("conv_out", conv)
    q, k, v = (c.reshape(1, 1, HL, D) for c in conv.split(HL * D, dim=-1))
    ssm = torch.zeros((2, *state.shape[1:]), dtype=torch.float32, device=dev)
    ssm[1] = state[0]
    dump.put("initial_state", ssm[1])
    o, _ = ops.fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g1,
        beta=beta_raw,
        initial_state=ssm,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=torch.tensor([0, 1], dtype=torch.int32, device=dev),
        ssm_state_indices=torch.ones(1, dtype=torch.int32, device=dev),
        sigmoid_beta=True,
        a_log=a_log,
        g_bias=dt_bias,
        compute_gate=True,
        lower_bound=LOWER_BOUND,
    )
    dump.put("output", o)
    dump.put("final_state", ssm[1])
    dump.put("conv_state_after", conv_state[1])
    return o, ssm[1:2].clone()


# Carried-state fixtures: prefill splits of the captured M128 rows (tails,
# chunk boundaries inside an invocation, >=3 chunks) then repeated decodes.
SEQUENCE_PREFILL = ((0, 5), (5, 64), (69, 59), (None, 200))
SEQUENCE_DECODE_ROWS = (0, 1, 2, 3)


def run_sequence(ops, args, conv_w, a_log, dt_bias, dev):
    x, g1, beta_raw, _ = load_case(args.raw, args.rank, 16, 128, dev)
    idx200 = torch.cat([torch.arange(128), torch.arange(72)]).to(dev)
    conv_states, h, rows = None, None, {}
    for step, (start, n) in enumerate(SEQUENCE_PREFILL):
        sel = idx200 if start is None else torch.arange(start, start + n, device=dev)
        dump = Dumper(args.out / f"rank{args.rank}-seq{step}-prefill{n}")
        o, _, h, conv_states = prefill(ops, x[sel].contiguous(), g1[:, sel].contiguous(),
                                       beta_raw[:, sel].contiguous(), conv_w, a_log, dt_bias, dump, dev,
                                       conv_states=conv_states, h_init=h)
        torch.cuda.synchronize()
        (dump.out / "manifest.json").write_text(json.dumps(dump.manifest, indent=1) + "\n")
        rows[dump.out.name] = {"rows": n, "has_initial_state": step > 0}
    for j, r in enumerate(SEQUENCE_DECODE_ROWS, start=len(SEQUENCE_PREFILL)):
        dump = Dumper(args.out / f"rank{args.rank}-seq{j}-decode")
        _, h = decode(ops, x[r : r + 1].contiguous(), g1[:, r : r + 1].contiguous(),
                      beta_raw[:, r : r + 1].contiguous(), conv_w, a_log, dt_bias, conv_states, h, dump, dev)
        torch.cuda.synchronize()
        (dump.out / "manifest.json").write_text(json.dumps(dump.manifest, indent=1) + "\n")
        rows[dump.out.name] = {"rows": 1}
    return rows


class Ops:
    def __init__(self):
        from vllm.model_executor.layers.mamba.ops import causal_conv1d as cc
        from vllm.third_party.flash_linear_attention.ops import chunk_delta_h, kda, l2norm, solve_tril
        from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices

        self.prepare_chunk_indices = prepare_chunk_indices

        self.modules = {"causal_conv1d": cc, "kda": kda, "l2norm": l2norm,
                        "solve_tril": solve_tril, "chunk_delta_h": chunk_delta_h}
        self.causal_conv1d_fn = cc.causal_conv1d_fn
        self.causal_conv1d_update = cc.causal_conv1d_update
        self.l2norm_fwd = l2norm.l2norm_fwd
        self.fused_kda_gate_chunk_cumsum = kda.fused_kda_gate_chunk_cumsum
        self.chunk_kda_scaled_dot_kkt_fwd = kda.chunk_kda_scaled_dot_kkt_fwd
        self.solve_tril = solve_tril.solve_tril
        self.recompute_w_u_fwd = kda.recompute_w_u_fwd
        self.chunk_gated_delta_rule_fwd_h = chunk_delta_h.chunk_gated_delta_rule_fwd_h
        self.chunk_gla_fwd_o_gk = kda.chunk_gla_fwd_o_gk
        self.chunk_kda_with_fused_gate = kda.chunk_kda_with_fused_gate
        self.fused_recurrent_kda = kda.fused_recurrent_kda

    def source_hashes(self, pinned: Path | None):
        rows = {}
        for name, mod in self.modules.items():
            path = Path(mod.__file__)
            entry = {"installed": str(path), "sha256": sha256(path.read_bytes())}
            if pinned is not None:
                rel = "causal_conv1d.py" if name == "causal_conv1d" else f"fla/ops/{path.name}"
                p = pinned / rel
                entry["pinned_sha256"] = sha256(p.read_bytes()) if p.exists() else None
                entry["matches_pinned"] = entry["pinned_sha256"] == entry["sha256"]
            rows[name] = entry
        return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--rank", type=int, required=True, choices=(0, 1))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source-root", type=Path)
    ap.add_argument("--beta", choices=("eager", "inductor"), required=True)
    ap.add_argument("--sequence", action="store_true",
                    help="also replay carried-state invocations built from the M128 rows")
    args = ap.parse_args()
    global BETA_FN
    if args.beta == "inductor":
        BETA_FN = torch.compile(_cast_sigmoid, dynamic=False)
    dev = torch.device("cuda")
    ops = Ops()
    conv_w, a_log, dt_bias = load_params(args.params, args.rank, dev)
    report = {"rank": args.rank, "beta": args.beta, "sources": ops.source_hashes(args.source_root),
              "torch": torch.__version__, "device": torch.cuda.get_device_name(),
              "cases": {}}
    import triton

    report["triton"] = triton.__version__

    for inv, t in ((324, 5), (16, 128)):
        x, g1, beta_raw, ref = load_case(args.raw, args.rank, inv, t, dev)
        dump = Dumper(args.out / f"rank{args.rank}-{inv:06d}")
        o, o_e2e, final_state, conv_state = prefill(ops, x, g1, beta_raw, conv_w, a_log, dt_bias, dump, dev)
        torch.cuda.synchronize()
        report["cases"][f"{inv:06d}"] = {
            "rows": t,
            "staged_vs_captured": mismatches(o.reshape(t, HL, D), ref),
            "e2e_vs_captured": mismatches(o_e2e.reshape(t, HL, D), ref),
            "staged_vs_e2e": mismatches(o, o_e2e),
            "elements": ref.numel(),
            "stages": dump.manifest,
        }
        (dump.out / "manifest.json").write_text(json.dumps(dump.manifest, indent=1) + "\n")
        if inv == 324:
            xd, gd, bd, refd = load_case(args.raw, args.rank, 325, 1, dev)
            dd = Dumper(args.out / f"rank{args.rank}-000325")
            od, _ = decode(ops, xd, gd, bd, conv_w, a_log, dt_bias, conv_state.clone(), final_state, dd, dev)
            torch.cuda.synchronize()
            report["cases"]["000325"] = {
                "rows": 1,
                "decode_vs_captured": mismatches(od.reshape(1, HL, D), refd),
                "elements": refd.numel(),
                "stages": dd.manifest,
            }
            (dd.out / "manifest.json").write_text(json.dumps(dd.manifest, indent=1) + "\n")
    if args.sequence:
        report["sequence"] = run_sequence(ops, args, conv_w, a_log, dt_bias, dev)
    (args.out / f"report-rank{args.rank}.json").write_text(json.dumps(report, indent=1) + "\n")
    summary = {k: {kk: vv for kk, vv in v.items() if kk != "stages"} for k, v in report["cases"].items()}
    print(json.dumps({"sources": report["sources"], "cases": summary}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
