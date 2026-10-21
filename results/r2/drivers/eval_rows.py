"""Run-2 row tables for the lead's decision evaluations (D1/D2/D3) and the REPORT.

Usage: pixi run python results/r2/drivers/eval_rows.py <rows.jsonl>... [--r2] [--mem]
  default : R1 view: per (model, arm, W): per-seed g_r1 verdicts, b2/floor, pooled b2/floor,
            median grads/chain, frozen eps and L, re-seed T0.
  --r2    : R2 view: per (model, arm): per-seed max_rhat, min bulk/tail ESS, E-BFMI, divs/1k,
            grads/ESS, b2, plus medians and the count of seeds with max_rhat <= 1.01.
  --mem   : peak memory and wall per row.
The verdict is recomputed with tools/r1_bias.g_r1 (the gate), never re-implemented here.
"""
import json
import os
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
from r1_bias import g_r1  # noqa: E402


def load(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return [r for r in rows if not r.get("error")]


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def fmt(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}g}" if isinstance(x, float) else str(x))


def verdict(r):
    return g_r1(r["b2_max"], r["b2_floor"], r["chains"] * r["draws_per_chain"])


def r1_view(rows):
    cells = defaultdict(list)
    for r in rows:
        cells[(r["model"], r["arm"], r["warmup"])].append(r)
    print("| model | arm | W | per-seed g_r1 | per-seed b2/floor | pooled b2/floor | med grads/chain | eps | L | T0 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (m, a, w), rs in sorted(cells.items()):
        seeds = sorted([r for r in rs if not r.get("pooled")], key=lambda r: str(r["seed"]))
        pooled = [r for r in rs if r.get("pooled")]
        v = "/".join(verdict(r).split(" ")[0] for r in seeds)
        bf = " ".join(f"{r['b2_max']:.4g}/{r['b2_floor']:.3g}" for r in seeds)
        pb = " ".join(f"{r['b2_max']:.4g}/{r['b2_floor']:.3g}" for r in pooled) or "—"
        print(f"| {m} | {a} | {w} | {v} | {bf} | {pb} | {fmt(med([r.get('grads_per_chain_total') for r in seeds]))} "
              f"| {fmt(med([r.get('frozen_eps') for r in seeds]), 3)} | {fmt(med([r.get('frozen_L_nominal') for r in seeds]), 3)} "
              f"| {fmt(med([r.get('metric_refresh_T0') for r in seeds]), 3)} |")


def r2_view(rows):
    cells = defaultdict(list)
    for r in rows:
        if not r.get("pooled"):
            cells[(r["model"], r["arm"])].append(r)
    print("| model | arm | seeds | rhat<=1.01 | med grads/ESS | per-seed max_rhat | min bulk ESS | min tail ESS | ebfmi min | divs/1k | warmup divs | b2 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (m, a), rs in sorted(cells.items()):
        rs = sorted(rs, key=lambda r: str(r["seed"]))
        ok = sum(1 for r in rs if r.get("max_rhat") is not None and r["max_rhat"] <= 1.01)
        j = lambda k, nd=4: " ".join(fmt(r.get(k), nd) for r in rs)  # noqa: E731
        print(f"| {m} | {a} | {len(rs)} | {ok} | {fmt(med([r.get('grads_per_ess_worst') for r in rs]))} | {j('max_rhat')} "
              f"| {j('min_bulk_ess', 4)} | {j('min_tail_ess', 4)} | {j('ebfmi_min', 3)} | {j('divs_per_1k', 3)} "
              f"| {j('warmup_divs', 3)} | {j('b2_max', 3)} |")


def mem_view(rows):
    print("| model | arm | C | W | seed | rep | cold | peak MiB | wall_s | pool rows |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        eff = r.get("effective") or {}
        pk = r.get("peak_mem_bytes")
        print(f"| {r['model']} | {r['arm']} | {r['chains']} | {r['warmup']} | {r['seed']} | {r.get('rep')} | {r.get('cold')} "
              f"| {fmt(pk / 2**20 if pk else None, 5)} | {fmt(r.get('wall_s'), 4)} | {eff.get('pf_pool_rows')} |")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rows = load(args)
    if "--r2" in sys.argv:
        r2_view(rows)
    elif "--mem" in sys.argv:
        mem_view(rows)
    else:
        r1_view(rows)
